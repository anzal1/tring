"""Tests for the v0.4 speech-to-speech work: Gemini Live, transcript
reconciliation, and hybrid mid-call downgrade.

Three features, one file, because they are one story: a provider that can
fail, a transcript that has to survive the failure, and a runtime that has to
keep the caller talking to *something* afterwards. The downgrade test at the
bottom is the one that proves it hangs together -- it asserts the cascade
answers a question the caller asks *after* the handover using context the S2S
leg accumulated *before* it.

Nothing here touches a network, an API key, a GPU or a ``websockets`` install.
The Gemini adapter takes the same scripted-socket ``connect`` seam the other
S2S providers do; the cascade half runs on fake STT/LLM/TTS providers
registered under ``v2_*`` names in the real registry, so provider resolution
goes through the same ``routing -> registry.create`` path a production spec
does.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any

import pytest

from tring.agent import AgentSpec, ProviderSelection, RuntimeConfig, RuntimeMode, ToolDef
from tring.cost.rates import RateCard
from tring.events import (
    BotUtterance,
    CostComponent,
    SessionEnded,
    SessionError,
    SessionStarted,
    TranscriptAvailability,
    UserTranscript,
)
from tring.providers import registry
from tring.providers.base import (
    LLMChunk,
    LLMProvider,
    S2SProvider,
    STTProvider,
    STTResult,
    TTSChunk,
    TTSProvider,
)
from tring.providers.cloud.s2s_gemini import (
    GEMINI_LIVE_INPUT_MIME,
    GEMINI_LIVE_OUTPUT_RATE,
    GEMINI_LIVE_RATES,
    GEMINI_LIVE_URL,
    GeminiLiveS2S,
)
from tring.providers.registry import register
from tring.runtimes.base import AudioFrame
from tring.runtimes.hybrid import DowngradePolicy, HybridRuntime
from tring.runtimes.s2s import (
    S2SRuntime,
    reconcile_transcript,
    transcript_offset,
)
from tring.session import CallSession

WIRE_SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Scripted websocket (same seam the other S2S provider tests use)
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """Replays ``inbound``, records everything sent.

    The ``gate`` keeps these tests deterministic rather than timing-dependent:
    the provider's pump task drains caller frames concurrently with the read
    loop, so the inbound replay is held until the frame source signals it is
    exhausted. By the time the first server message is delivered, every client
    message has already been sent and ``sent`` can be asserted without a sleep.
    """

    def __init__(
        self, inbound: list[str | bytes] | None = None, gate: asyncio.Event | None = None
    ) -> None:
        self.inbound: list[str | bytes] = list(inbound or [])
        self.sent: list[str | bytes] = []
        self.closed = False
        self._gate = gate

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        return self._replay()

    async def _replay(self) -> AsyncIterator[str | bytes]:
        if self._gate is not None:
            await self._gate.wait()
        for message in self.inbound:
            yield message

    @property
    def sent_json(self) -> list[dict[str, Any]]:
        return [json.loads(m) for m in self.sent if isinstance(m, str)]


class RecordingConnect:
    """A ``WebSocketConnect`` handing back a fixed socket, remembering the
    url and headers it was called with."""

    def __init__(self, socket: FakeWebSocket) -> None:
        self.socket = socket
        self.url: str | None = None
        self.headers: dict[str, str] | None = None

    async def __call__(self, url: str, headers: dict[str, str]) -> FakeWebSocket:
        self.url = url
        self.headers = dict(headers)
        return self.socket


async def _frames(items: list[AudioFrame], gate: asyncio.Event) -> AsyncIterator[AudioFrame]:
    for frame in items:
        yield frame
    gate.set()


def _silence(sample_rate: int = WIRE_SAMPLE_RATE, ms: int = 20) -> AudioFrame:
    return AudioFrame(
        pcm=b"\x00\x01" * (sample_rate * ms // 1000), sample_rate=sample_rate, channels=1
    )


async def _drain(provider: S2SProvider, frames: AsyncIterator[AudioFrame]) -> list[TTSChunk]:
    return [chunk async for chunk in provider.converse(frames)]


def _open_gemini(
    inbound: list[str | bytes], *, gate_open: bool = True
) -> tuple[GeminiLiveS2S, FakeWebSocket, asyncio.Event]:
    gate = asyncio.Event()
    if gate_open:
        gate.set()
    socket = FakeWebSocket(inbound, gate=gate)
    return GeminiLiveS2S(connect=RecordingConnect(socket)), socket, gate


def _server_content(**fields: Any) -> str:
    return json.dumps({"serverContent": fields})


# ---------------------------------------------------------------------------
# Gemini Live: request construction
# ---------------------------------------------------------------------------


def test_gemini_live_url_carries_the_key_as_a_query_parameter() -> None:
    provider = GeminiLiveS2S()
    assert provider.build_url("abc123") == f"{GEMINI_LIVE_URL}?key=abc123"
    assert GEMINI_LIVE_URL.endswith(
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )


def test_gemini_live_url_escapes_a_key_with_url_significant_characters() -> None:
    # An unescaped '&' would silently truncate the key into a second parameter.
    assert GeminiLiveS2S().build_url("a&b=c").endswith("?key=a%26b%3Dc")


def test_gemini_live_setup_message_shape() -> None:
    provider = GeminiLiveS2S(
        model="gemini-3.8-live",
        system_instruction="Be brief.",
        temperature=0.4,
    )
    setup = provider.build_setup()["setup"]

    # The model is addressed as a resource name, not a bare id.
    assert setup["model"] == "models/gemini-3.8-live"
    # responseModalities lives inside generationConfig, not beside it.
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["generationConfig"]["temperature"] == 0.4
    assert setup["systemInstruction"] == {"parts": [{"text": "Be brief."}]}
    # AudioTranscriptionConfig is an empty message: presence is the opt-in.
    assert setup["inputAudioTranscription"] == {}
    assert setup["outputAudioTranscription"] == {}
    assert "speechConfig" not in setup["generationConfig"]


def test_gemini_live_setup_does_not_double_prefix_a_resource_name() -> None:
    setup = GeminiLiveS2S(model="models/gemini-3.8-live").build_setup()["setup"]
    assert setup["model"] == "models/gemini-3.8-live"


def test_gemini_live_transcription_is_omitted_when_disabled() -> None:
    setup = GeminiLiveS2S(
        input_transcription=False, output_transcription=False
    ).build_setup()["setup"]
    assert "inputAudioTranscription" not in setup
    assert "outputAudioTranscription" not in setup


def test_gemini_live_speech_config_is_a_verbatim_passthrough() -> None:
    # The docs do not pin speechConfig's shape, so the adapter must not
    # construct one -- whatever the operator supplies goes through unchanged.
    block = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}
    setup = GeminiLiveS2S(speech_config=block).build_setup()["setup"]
    assert setup["generationConfig"]["speechConfig"] == block


def test_gemini_live_tools_are_passed_through_when_declared() -> None:
    tools = [{"functionDeclarations": [{"name": "book_slot", "description": "book"}]}]
    assert GeminiLiveS2S(tools=tools).build_setup()["setup"]["tools"] == tools
    assert "tools" not in GeminiLiveS2S().build_setup()["setup"]


def test_gemini_live_realtime_input_is_base64_pcm_at_16k() -> None:
    event = GeminiLiveS2S().build_realtime_input(_silence(WIRE_SAMPLE_RATE, ms=20))
    audio = event["realtimeInput"]["audio"]
    assert audio["mimeType"] == GEMINI_LIVE_INPUT_MIME == "audio/pcm;rate=16000"
    # 16 kHz is already the canonical wire rate: the input edge is a no-op.
    assert len(base64.b64decode(audio["data"])) // 2 == 320


def test_gemini_live_resamples_a_non_canonical_input_frame_to_16k() -> None:
    event = GeminiLiveS2S().build_realtime_input(_silence(48000, ms=10))
    decoded = base64.b64decode(event["realtimeInput"]["audio"]["data"])
    assert len(decoded) // 2 == 160  # 10 ms at 16 kHz


# ---------------------------------------------------------------------------
# Gemini Live: the session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_live_handshake_and_audio_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-unit-test")
    # 480 samples of 24 kHz output audio -> 320 samples (640 bytes) at 16 kHz.
    delta = base64.b64encode(b"\x00\x20" * 480).decode("ascii")
    provider, socket, gate = _open_gemini(
        [
            json.dumps({"setupComplete": {}}),
            _server_content(modelTurn={"parts": [{"inlineData": {"data": delta}}]}),
        ],
        gate_open=False,
    )
    connect = provider._connect
    assert isinstance(connect, RecordingConnect)

    chunks = await _drain(provider, _frames([_silence(), _silence()], gate))

    assert connect.url == f"{GEMINI_LIVE_URL}?key=gk-unit-test"
    assert connect.headers == {}  # the key is in the URL; no auth header exists
    sent = socket.sent_json
    assert "setup" in sent[0]  # configuration first, always
    assert [next(iter(m)) for m in sent[1:]] == ["realtimeInput"] * 3
    # One append per frame, then the documented end-of-audio marker.
    assert sent[-1] == {"realtimeInput": {"audioStreamEnd": True}}
    assert socket.closed

    assert provider.setup_complete is True
    assert len(chunks) == 1
    assert chunks[0].frame.sample_rate == WIRE_SAMPLE_RATE
    assert len(chunks[0].frame.pcm) // 2 == 320
    assert chunks[0].usage == []


@pytest.mark.asyncio
async def test_gemini_live_accepts_binary_framed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The Live API frames its JSON as binary as often as text.
    monkeypatch.setenv("GEMINI_API_KEY", "gk-unit-test")
    provider, _socket, gate = _open_gemini(
        [_server_content(outputTranscription={"text": "hello"}).encode("utf-8")]
    )

    await _drain(provider, _frames([], gate))

    assert await provider.post_call_transcript() == [
        {"role": "bot", "text": "hello", "language": None}
    ]


@pytest.mark.asyncio
async def test_gemini_live_missing_api_key_names_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider, _socket, gate = _open_gemini([])
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        await _drain(provider, _frames([], gate))


@pytest.mark.asyncio
async def test_gemini_live_transcript_fragments_concatenate_and_split_on_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-unit-test")
    provider, _socket, gate = _open_gemini(
        [
            _server_content(inputTranscription={"text": "book me ", "languageCode": "en"}),
            _server_content(inputTranscription={"text": "for tuesday"}),
            _server_content(outputTranscription={"text": "Tuesday works."}),
            _server_content(turnComplete=True),
            _server_content(inputTranscription={"text": "thanks"}),
        ]
    )

    await _drain(provider, _frames([], gate))

    # Fragments within a turn concatenate; turnComplete starts a new turn, so
    # "thanks" is its own utterance and not glued onto the first one.
    assert await provider.post_call_transcript() == [
        {"role": "user", "text": "book me for tuesday", "language": "en"},
        {"role": "bot", "text": "Tuesday works.", "language": None},
        {"role": "user", "text": "thanks", "language": None},
    ]


@pytest.mark.asyncio
async def test_gemini_live_interruption_also_closes_the_open_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-unit-test")
    provider, _socket, gate = _open_gemini(
        [
            _server_content(outputTranscription={"text": "Let me check tha"}),
            _server_content(interrupted=True),
            _server_content(outputTranscription={"text": "Sorry, go ahead."}),
        ]
    )

    await _drain(provider, _frames([], gate))

    transcript = await provider.post_call_transcript()
    assert [t["text"] for t in transcript] == ["Let me check tha", "Sorry, go ahead."]


@pytest.mark.asyncio
async def test_gemini_live_usage_is_exact_and_reported_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-unit-test")
    provider, _socket, gate = _open_gemini(
        [
            json.dumps(
                {
                    "usageMetadata": {
                        "promptTokenCount": 400,
                        "responseTokenCount": 100,
                        "totalTokenCount": 500,
                    }
                }
            ),
            json.dumps(
                {
                    "usageMetadata": {
                        "promptTokenCount": 1200,
                        "cachedContentTokenCount": 900,
                        "responseTokenCount": 300,
                        "totalTokenCount": 1500,
                    }
                }
            ),
        ]
    )

    chunks = await _drain(provider, _frames([], gate))

    # Two periodic snapshots of one running total: the last one is the answer,
    # and summing them would have billed this call for 1600 prompt tokens.
    assert len(chunks) == 1
    assert chunks[0].frame.pcm == b""  # zero-length usage carrier
    tokens_in, tokens_out = chunks[0].usage
    assert (tokens_in.units, tokens_in.unit_name) == (1200.0, "tokens_in")
    assert tokens_in.estimated is False
    assert tokens_in.cached_units == 900.0  # the prompt-cache honesty bit
    assert tokens_in.model == "gemini-3.8-live"
    assert (tokens_out.units, tokens_out.unit_name) == (300.0, "tokens_out")
    assert tokens_out.cached_units is None


@pytest.mark.asyncio
async def test_gemini_live_without_usage_metadata_reports_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-unit-test")
    provider, _socket, gate = _open_gemini([_server_content(turnComplete=True)])
    assert await _drain(provider, _frames([], gate)) == []


@pytest.mark.asyncio
async def test_gemini_live_go_away_duration_is_parsed_or_left_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-unit-test")
    provider, _socket, gate = _open_gemini([json.dumps({"goAway": {"timeLeft": "12.5s"}})])
    await _drain(provider, _frames([], gate))
    assert provider.go_away_seconds == 12.5

    other, _socket2, gate2 = _open_gemini([json.dumps({"goAway": {"timeLeft": "soon"}})])
    await _drain(other, _frames([], gate2))
    assert other.go_away_seconds is None  # unparseable stays unknown, not 0.0


# ---------------------------------------------------------------------------
# Gemini Live: tool calls and turn boundaries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gemini_live_bridges_tool_calls_and_answers_with_tool_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-unit-test")
    provider, socket, gate = _open_gemini(
        [
            json.dumps(
                {
                    "toolCall": {
                        "functionCalls": [
                            {"id": "fc-1", "name": "book_slot", "args": {"date": "tue"}}
                        ]
                    }
                }
            )
        ]
    )
    seen: list[tuple[str, dict[str, Any]]] = []

    async def bridge(name: str, args: dict[str, Any]) -> dict[str, Any]:
        seen.append((name, args))
        return {"ok": True, "confirmation_id": "abc123"}

    provider.on_tool_call = bridge

    await _drain(provider, _frames([], gate))

    assert seen == [("book_slot", {"date": "tue"})]
    reply = socket.sent_json[-1]
    assert reply == {
        "toolResponse": {
            "functionResponses": [
                {
                    "name": "book_slot",
                    "response": {"ok": True, "confirmation_id": "abc123"},
                    "id": "fc-1",
                }
            ]
        }
    }


@pytest.mark.asyncio
async def test_gemini_live_unbridged_tool_call_still_gets_an_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An unanswered function call leaves the model waiting forever, which the
    # caller hears as a dead line. An explicit error beats silence.
    monkeypatch.setenv("GEMINI_API_KEY", "gk-unit-test")
    provider, socket, gate = _open_gemini(
        [json.dumps({"toolCall": {"functionCalls": [{"name": "book_slot"}]}})]
    )

    await _drain(provider, _frames([], gate))

    responses = socket.sent_json[-1]["toolResponse"]["functionResponses"]
    assert "no tool handler is bound" in responses[0]["response"]["error"]
    assert "id" not in responses[0]  # nothing invented for a call without one


@pytest.mark.asyncio
async def test_gemini_live_reports_turn_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-unit-test")
    provider, _socket, gate = _open_gemini(
        [
            _server_content(turnComplete=True),
            _server_content(interrupted=True),
            _server_content(turnComplete=True),
        ]
    )
    turns = 0

    async def on_turn() -> None:
        nonlocal turns
        turns += 1

    provider.on_turn_complete = on_turn

    await _drain(provider, _frames([], gate))

    # Only completed turns count; an interruption is not a finished turn.
    assert turns == 2


# ---------------------------------------------------------------------------
# Gemini Live: registry and rates
# ---------------------------------------------------------------------------


def test_gemini_live_is_registered_and_constructible() -> None:
    registry._load_builtin()
    assert ("s2s", "gemini_live") in registry.available("s2s")
    provider = registry.create("s2s", "gemini_live", model="gemini-3.8-live")
    assert isinstance(provider, GeminiLiveS2S)


def test_gemini_live_rates_are_s2s_priced_and_dated() -> None:
    assert GEMINI_LIVE_RATES
    for rate in GEMINI_LIVE_RATES:
        assert rate.component is CostComponent.S2S
        assert rate.as_of == "2026-09"
        assert rate.currency == "USD"
        assert rate.price_per_unit > 0.0
        # A mismatch here is the classic silent-zero-cost bug: the meter looks
        # rates up by provider.name, so the two spellings have to agree.
        assert rate.provider == GeminiLiveS2S.name


def test_gemini_live_rates_are_usable_by_the_cost_engine() -> None:
    card = RateCard(version="test", rates=GEMINI_LIVE_RATES)
    tokens_in = card.lookup(
        CostComponent.S2S, "gemini_live", "tokens_in", model="gemini-3.8-live"
    )
    assert tokens_in is not None
    assert tokens_in.price_per_unit == pytest.approx(3.0 / 1_000_000)

    tokens_out = card.lookup(
        CostComponent.S2S, "gemini_live", "tokens_out", model="gemini-3.8-live"
    )
    assert tokens_out is not None
    assert tokens_out.price_per_unit == pytest.approx(12.0 / 1_000_000)


def test_gemini_live_output_rate_is_the_documented_24k() -> None:
    assert GEMINI_LIVE_OUTPUT_RATE == 24000


# ---------------------------------------------------------------------------
# Transcript reconciliation
# ---------------------------------------------------------------------------


def test_transcript_offset_prefers_the_provider_timing() -> None:
    assert transcript_offset({"at": 4.25}, 99.0) == 4.25
    assert transcript_offset({"start": 2.0}, 99.0) == 2.0
    assert transcript_offset({"at": 1.0, "start": 2.0}, 99.0) == 1.0


def test_transcript_offset_falls_back_when_the_provider_is_silent() -> None:
    assert transcript_offset({"text": "hi"}, 99.0) == 99.0
    # bool is an int subclass; a stray flag is not a timestamp.
    assert transcript_offset({"at": True}, 99.0) == 99.0
    assert transcript_offset({"at": "12:03"}, 99.0) == 99.0


def test_reconcile_transcript_is_a_no_op_without_timings() -> None:
    turns: list[Mapping[str, Any]] = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    assert [t for _at, t in reconcile_transcript(turns, 7.0)] == turns
    assert [at for at, _t in reconcile_transcript(turns, 7.0)] == [7.0, 7.0, 7.0]


def test_reconcile_transcript_orders_by_reported_time() -> None:
    turns: list[Mapping[str, Any]] = [
        {"role": "bot", "text": "second", "at": 2.0},
        {"role": "user", "text": "first", "at": 1.0},
    ]
    assert [t["text"] for _at, t in reconcile_transcript(turns, 0.0)] == ["first", "second"]


class TimedTranscriptS2S(S2SProvider):
    """A provider that knows when each turn happened, and says so."""

    name = "v2-timed"

    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self._turns = turns

    async def converse(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[TTSChunk]:
        async for _frame in frames:
            continue
        yield TTSChunk(frame=AudioFrame(pcm=b""))

    async def post_call_transcript(self) -> list[dict[str, Any]]:
        return list(self._turns)


def _fixed_clock() -> Callable[[], float]:
    """A clock frozen at zero, so ``session.elapsed`` is exactly 0.0 and an
    untimed transcript's stamps can be asserted rather than bounded."""
    return lambda: 0.0


def _s2s_agent(s2s: str = "v2-fake") -> AgentSpec:
    return AgentSpec(
        name="test-agent",
        persona="You are a test agent.",
        runtime=RuntimeConfig(
            mode=RuntimeMode.S2S, routing={"default": ProviderSelection(s2s=s2s)}
        ),
    )


@pytest.mark.asyncio
async def test_s2s_runtime_stamps_timed_transcripts_with_provider_timing() -> None:
    session = CallSession(agent=_s2s_agent(), clock=_fixed_clock())
    provider = TimedTranscriptS2S(
        [
            {"role": "bot", "text": "hi, how can I help?", "at": 2.5},
            {"role": "user", "text": "hello there", "at": 1.5},
        ]
    )
    runtime = S2SRuntime(session, provider=provider)

    await runtime.start()
    await runtime.stop()

    user = [e for e in session.history if isinstance(e, UserTranscript)]
    bot = [e for e in session.history if isinstance(e, BotUtterance)]
    assert (user[0].text, user[0].at) == ("hello there", 1.5)
    assert user[0].availability is TranscriptAvailability.POST_CALL
    assert (bot[0].text, bot[0].at) == ("hi, how can I help?", 2.5)
    # Reordered onto the timeline: the caller spoke before the bot answered,
    # whatever order the provider chose to hand the two halves over in.
    assert session.history.index(user[0]) < session.history.index(bot[0])


@pytest.mark.asyncio
async def test_s2s_runtime_untimed_transcripts_behave_exactly_as_before() -> None:
    session = CallSession(agent=_s2s_agent(), clock=_fixed_clock())
    provider = TimedTranscriptS2S(
        [
            {"role": "bot", "text": "hi, how can I help?"},
            {"role": "user", "text": "hello there"},
        ]
    )
    runtime = S2SRuntime(session, provider=provider)

    await runtime.start()
    await runtime.stop()

    transcripts = [e for e in session.history if isinstance(e, UserTranscript | BotUtterance)]
    # Arrival order preserved, every turn stamped with the end-of-call clock.
    assert [e.text for e in transcripts] == ["hi, how can I help?", "hello there"]
    assert {e.at for e in transcripts} == {0.0}


# ---------------------------------------------------------------------------
# Hybrid v2: fake providers for the cascade that takes over
# ---------------------------------------------------------------------------

#: Every message list the downgraded cascade's LLM was called with. Module
#: level because registry resolution builds the provider, so a test cannot
#: hold a reference to the instance before it exists.
LLM_CALLS: list[list[dict[str, Any]]] = []


@register("stt", "v2_text")
class TextFrameSTT(STTProvider):
    """Treats each frame's bytes as its transcript (the text_input trick)."""

    name = "v2_text"

    def __init__(self, **_options: Any) -> None:
        pass

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]:
        async for frame in frames:
            text = frame.pcm.decode("utf-8").strip()
            if text:
                yield STTResult(text=text, final=True, language=language)


@register("llm", "v2_echo")
class EchoLLM(LLMProvider):
    """Answers in the cascade envelope, echoing the latest user message.

    Echoing is what makes the downgrade test meaningful: the reply proves the
    cascade saw the *new* question, and ``LLM_CALLS`` proves it also saw the
    old conversation.
    """

    name = "v2_echo"

    def __init__(self, **_options: Any) -> None:
        pass

    async def generate(
        self,
        messages: list[dict[Any, Any]],
        tools: list[dict[Any, Any]] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        LLM_CALLS.append([dict(m) for m in messages])
        latest = next(
            (str(m["content"]) for m in reversed(messages) if m["role"] == "user"), ""
        )
        yield LLMChunk(
            text=json.dumps({"speak": f"cascade heard {latest}", "tool_call": None})
        )
        yield LLMChunk(text="", finish=True)


@register("tts", "v2_pcm")
class TextEchoTTS(TTSProvider):
    """Synthesizes each text fragment as its own utf-8 'audio' frame."""

    name = "v2_pcm"

    def __init__(self, **_options: Any) -> None:
        pass

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        async for fragment in text:
            yield TTSChunk(frame=AudioFrame(pcm=fragment.encode("utf-8")))


class ScriptedS2S(S2SProvider):
    """A scripted S2S leg: some audio, a transcript, optionally an explosion."""

    name = "v2-scripted"

    def __init__(self, *, fail_with: Exception | None = None) -> None:
        self._fail_with = fail_with
        self.frames_received: list[AudioFrame] = []

    async def converse(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[TTSChunk]:
        if self._fail_with is not None:
            raise self._fail_with
        async for frame in frames:
            self.frames_received.append(frame)
        yield TTSChunk(frame=AudioFrame(pcm=b"s2s-audio"))

    async def post_call_transcript(self) -> list[dict[str, Any]]:
        return [
            {"role": "user", "text": "I want to move my tuesday booking", "at": 1.0},
            {"role": "bot", "text": "Sure, what day suits you?", "at": 2.0},
        ]


_BOOK_SLOT = ToolDef(
    name="book_slot",
    description="Book a slot on a given date.",
    parameters={
        "type": "object",
        "properties": {"date": {"type": "string"}},
        "required": ["date"],
    },
    handler="booking.book",
)

_CHOREOGRAPHED_ARGS = {
    "date": "2026-09-22",
    "waiting_message": "One moment...",
    "spoken_mode": "answer_pending",
    "post_tool_response": "respond",
}


def _hybrid_agent() -> AgentSpec:
    """A spec that can run either way: an s2s slot *and* a full cascade trio."""
    return AgentSpec(
        name="test-agent",
        persona="You are a test agent.",
        greeting="Thanks for calling Acme.",
        runtime=RuntimeConfig(
            mode=RuntimeMode.HYBRID,
            routing={
                "default": ProviderSelection(
                    s2s="v2-scripted", stt="v2_text", llm="v2_echo", tts="v2_pcm"
                )
            },
        ),
        tools=[_BOOK_SLOT],
    )


def _hybrid(
    session: CallSession,
    provider: ScriptedS2S,
    policy: DowngradePolicy | None,
    handlers: dict[str, Any] | None = None,
) -> HybridRuntime:
    runtime = HybridRuntime(session, handlers=handlers or {}, downgrade=policy)
    # The same test seam the v0.1 hybrid tests use: inject the provider the
    # way a real deployment would supply it through routing.
    runtime._s2s._provider = provider
    return runtime


async def _until(predicate: Callable[[], bool], limit: int = 200) -> None:
    """Yield to the loop until ``predicate`` holds. No wall-clock sleeping:
    everything under test is cooperative, so this terminates in a handful of
    iterations or it is a real failure."""
    for _ in range(limit):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition never became true")


@pytest.fixture(autouse=True)
def _clear_llm_calls() -> Iterator[None]:
    LLM_CALLS.clear()
    yield
    LLM_CALLS.clear()


# ---------------------------------------------------------------------------
# Hybrid v2: no policy means nothing changed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_without_a_policy_never_downgrades() -> None:
    session = CallSession(agent=_hybrid_agent())
    runtime = _hybrid(session, ScriptedS2S(), policy=None, handlers={"book_slot": _ok})

    await runtime.start()
    for _ in range(5):
        await runtime.on_provider_tool_call("book_slot", dict(_CHOREOGRAPHED_ARGS))
    await runtime.stop()

    assert runtime.downgraded is False
    assert runtime.capabilities.live_transcripts is False
    assert not [e for e in session.history if isinstance(e, SessionError)]


@pytest.mark.asyncio
async def test_hybrid_without_a_policy_still_surfaces_a_provider_failure_at_stop() -> None:
    # The v0.1 behaviour, kept exactly: with no supervisor attached the
    # exception propagates out of the consume task when stop() awaits it.
    session = CallSession(agent=_hybrid_agent())
    runtime = _hybrid(session, ScriptedS2S(fail_with=RuntimeError("socket closed")), None)

    await runtime.start()
    with pytest.raises(RuntimeError, match="socket closed"):
        await runtime.stop()


# ---------------------------------------------------------------------------
# Hybrid v2: the downgrade
# ---------------------------------------------------------------------------


async def _ok(args: dict[str, Any]) -> dict[str, Any]:
    return {"confirmation_id": "abc123"}


@pytest.mark.asyncio
async def test_hybrid_downgrades_on_a_tool_heavy_stretch() -> None:
    session = CallSession(agent=_hybrid_agent())
    policy = DowngradePolicy(tool_calls=2, within_turns=1)
    runtime = _hybrid(session, ScriptedS2S(), policy, handlers={"book_slot": _ok})

    await runtime.start()
    first = await runtime.on_provider_tool_call("book_slot", dict(_CHOREOGRAPHED_ARGS))
    assert runtime.downgraded is False  # one call is not a stretch
    assert first["ok"] is True
    await runtime.on_provider_tool_call("book_slot", dict(_CHOREOGRAPHED_ARGS))

    assert runtime.downgraded is True
    error = next(e for e in session.history if isinstance(e, SessionError))
    assert error.recoverable is True  # the call continues; only the runtime changed
    assert "downgrading hybrid session to cascade" in error.message
    assert "2 tool calls" in error.message


@pytest.mark.asyncio
async def test_hybrid_tool_load_window_slides_with_turns() -> None:
    # Two calls, but a turn boundary between them: with a one-turn window the
    # first call has aged out, so this stays a hybrid session.
    session = CallSession(agent=_hybrid_agent())
    policy = DowngradePolicy(tool_calls=2, within_turns=1)
    runtime = _hybrid(session, ScriptedS2S(), policy, handlers={"book_slot": _ok})

    await runtime.start()
    await runtime.on_provider_tool_call("book_slot", dict(_CHOREOGRAPHED_ARGS))
    await runtime.on_provider_turn_complete()
    await runtime.on_provider_tool_call("book_slot", dict(_CHOREOGRAPHED_ARGS))

    assert runtime.downgraded is False


@pytest.mark.asyncio
async def test_hybrid_counts_failed_tool_calls_towards_the_threshold() -> None:
    # A model retrying a tool that is not even declared is exactly the case
    # worth escaping, so configuration failures count like successes.
    session = CallSession(agent=_hybrid_agent())
    runtime = _hybrid(session, ScriptedS2S(), DowngradePolicy(tool_calls=2, within_turns=2))

    await runtime.start()
    await runtime.on_provider_tool_call("no_such_tool", {})
    await runtime.on_provider_tool_call("no_such_tool", {})

    assert runtime.downgraded is True


@pytest.mark.asyncio
async def test_hybrid_downgrades_when_the_s2s_stream_fails() -> None:
    session = CallSession(agent=_hybrid_agent())
    runtime = _hybrid(session, ScriptedS2S(fail_with=RuntimeError("socket closed")),
                      DowngradePolicy())

    await runtime.start()
    await _until(lambda: runtime.downgraded)

    error = next(e for e in session.history if isinstance(e, SessionError))
    assert error.recoverable is True
    assert "socket closed" in error.message
    # The failure never reaches the caller of stop(): it was handled mid-call.
    await runtime.stop()
    assert sum(1 for e in session.history if isinstance(e, SessionEnded)) == 1


@pytest.mark.asyncio
async def test_hybrid_capabilities_follow_the_leg_that_is_actually_running() -> None:
    session = CallSession(agent=_hybrid_agent())
    runtime = _hybrid(session, ScriptedS2S(), DowngradePolicy(tool_calls=1, within_turns=1))

    await runtime.start()
    assert runtime.capabilities.live_transcripts is False
    assert runtime.capabilities.exact_usage_reporting is False

    await runtime.on_provider_tool_call("no_such_tool", {})

    # Declared, never assumed: a cascade really does have live transcripts.
    assert runtime.capabilities.live_transcripts is True
    assert runtime.capabilities.local_capable is True


@pytest.mark.asyncio
async def test_downgraded_cascade_continues_the_conversation_with_carried_context() -> None:
    """The whole point of the feature, asserted end to end."""
    session = CallSession(agent=_hybrid_agent())
    runtime = _hybrid(
        session,
        ScriptedS2S(fail_with=RuntimeError("provider closed the socket")),
        DowngradePolicy(),
    )
    spoken: list[bytes] = []
    runtime.on_bot_audio = lambda frame: spoken.append(frame.pcm)

    await runtime.start()
    await _until(lambda: runtime.downgraded)

    # The caller keeps talking; the cascade is what answers now.
    await runtime.push_audio(AudioFrame(pcm=b"can we do wednesday instead"))
    cascade = runtime._cascade
    assert cascade is not None
    await cascade.drain()
    await runtime.stop()

    heard = b"".join(spoken).decode("utf-8")
    assert "cascade heard can we do wednesday instead" in heard
    # The greeting is NOT replayed: the caller is mid-call, not arriving.
    assert "Thanks for calling Acme" not in heard

    # The cascade's prompt carries the S2S leg's conversation, in order, with
    # the bot's half re-encoded into the envelope the model is held to.
    messages = LLM_CALLS[0]
    roles = [(m["role"], m["content"]) for m in messages]
    user_texts = [content for role, content in roles if role == "user"]
    assert user_texts == [
        "I want to move my tuesday booking",
        "can we do wednesday instead",
    ]
    assistant = [content for role, content in roles if role == "assistant"]
    assert json.loads(assistant[0]) == {
        "speak": "Sure, what day suits you?",
        "tool_call": None,
    }

    # The transcript also reached the event bus, stamped with the provider's
    # own offsets, rather than being lost with the leg that produced it.
    user_events = [e for e in session.history if isinstance(e, UserTranscript)]
    carried = next(e for e in user_events if e.at == 1.0)
    assert carried.text == "I want to move my tuesday booking"
    assert carried.availability is TranscriptAvailability.POST_CALL

    # Exactly one ending, and a second SessionStarted marking the new leg.
    assert sum(1 for e in session.history if isinstance(e, SessionEnded)) == 1
    started = [e for e in session.history if isinstance(e, SessionStarted)]
    assert [e.runtime_mode for e in started] == ["hybrid", "hybrid"]


@pytest.mark.asyncio
async def test_downgrade_rebinds_handlers_onto_the_cascade_lookup() -> None:
    """Hybrid binds handlers by tool name, cascade by ``ToolDef.handler``."""
    session = CallSession(agent=_hybrid_agent())
    runtime = _hybrid(
        session,
        ScriptedS2S(),
        DowngradePolicy(tool_calls=1, within_turns=1),
        handlers={"book_slot": _ok},
    )

    await runtime.start()
    await runtime.on_provider_tool_call("book_slot", dict(_CHOREOGRAPHED_ARGS))

    cascade = runtime._cascade
    assert cascade is not None
    # _BOOK_SLOT.handler is "booking.book", which is the key cascade resolves.
    assert cascade.handlers["booking.book"] is _ok
    assert cascade.handlers["book_slot"] is _ok


@pytest.mark.asyncio
async def test_downgrade_is_idempotent_and_restores_the_session_spec() -> None:
    session = CallSession(agent=_hybrid_agent())
    runtime = _hybrid(session, ScriptedS2S(), DowngradePolicy(tool_calls=1, within_turns=1))

    await runtime.start()
    await runtime.on_provider_tool_call("no_such_tool", {})
    first = runtime._cascade
    await runtime.on_provider_tool_call("no_such_tool", {})

    assert runtime._cascade is first  # one downgrade per call, not one per trigger
    assert sum(1 for e in session.history if isinstance(e, SessionError)) == 1
    # The greeting suppression is scoped to cascade start-up only.
    assert session.agent.greeting == "Thanks for calling Acme."
