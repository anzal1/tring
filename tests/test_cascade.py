"""End-to-end tests for :class:`~alaap.runtimes.cascade.CascadeRuntime`.

Everything here runs on fakes. No model weights, no network, no audio device,
no API key -- which is the point: the cascade runtime's job is *orchestration*,
and orchestration bugs (speaking after the tool instead of before it, losing a
usage line, emitting events out of order) are exactly the bugs that real
providers hide behind latency and nondeterminism.

The fakes are registered in the real provider registry under ``test_*`` names,
so the tests exercise the same resolution path a production spec does:
``routing -> registry.create -> provider instance``.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from typing import Any

import pytest

from alaap.agent import (
    AgentSpec,
    LanguagePolicy,
    Limits,
    ProviderSelection,
    RuntimeConfig,
    ToolDef,
)
from alaap.cost.meter import CostMeter
from alaap.cost.rates import Rate, RateCard
from alaap.events import CostComponent
from alaap.providers.base import (
    LLMChunk,
    LLMProvider,
    STTProvider,
    STTResult,
    TTSChunk,
    TTSProvider,
    Usage,
)
from alaap.providers.registry import register
from alaap.runtimes.base import AudioFrame
from alaap.runtimes.cascade import SPOKEN_CHARS_PER_SECOND, CascadeRuntime
from alaap.session import CallSession

# ---------------------------------------------------------------------------
# Fake providers
# ---------------------------------------------------------------------------

#: Named LLM scripts, looked up by the ``script_id`` provider option. Going
#: through options rather than a constructor argument keeps the spec plain
#: data, which is what registry resolution requires.
LLM_SCRIPTS: dict[str, list[str]] = {}

#: How many characters of a completion arrive per streamed chunk. Small and
#: deliberately not aligned to JSON tokens, so the envelope parser is fed
#: fragments like ``'k": "We ar'`` -- the condition real providers create and
#: the one a buffer-then-parse implementation would pass anyway.
_CHUNK = 9


@register("stt", "test_scripted")
class ScriptedSTT(STTProvider):
    """Turns each frame's bytes into one final transcript (the text_input trick).

    Reports a fixed ``audio_seconds`` line so cost routing for the STT stage is
    observable without synthesizing audio of a known duration.
    """

    name = "test_scripted"

    def __init__(self, audio_seconds: float = 1.5, **_options: Any) -> None:
        self.audio_seconds = audio_seconds

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]:
        async for frame in frames:
            text = frame.pcm.decode("utf-8").strip()
            if not text:
                continue
            yield STTResult(
                text=text,
                final=True,
                language=language,
                usage=[
                    Usage(
                        units=self.audio_seconds,
                        unit_name="audio_seconds",
                        estimated=False,
                        model="fake-whisper",
                    )
                ],
            )


@register("llm", "test_scripted")
class ScriptedLLM(LLMProvider):
    """Replays a scripted list of completions, one per ``generate`` call."""

    name = "test_scripted"

    def __init__(self, script_id: str = "", **_options: Any) -> None:
        self.script = list(LLM_SCRIPTS[script_id])
        self.calls: list[list[dict]] = []

    async def generate(
        self, messages: list[dict], tools: list[dict] | None = None
    ) -> AsyncIterator[LLMChunk]:
        self.calls.append(list(messages))
        if not self.script:
            raise AssertionError("scripted LLM ran out of responses")
        completion = self.script.pop(0)
        for i in range(0, len(completion), _CHUNK):
            yield LLMChunk(text=completion[i : i + _CHUNK])
        yield LLMChunk(
            text="",
            finish=True,
            usage=[
                Usage(units=42, unit_name="tokens_in", estimated=False, model="fake-llm"),
                Usage(units=17, unit_name="tokens_out", estimated=False, model="fake-llm"),
            ],
        )


@register("tts", "test_scripted")
class ScriptedTTS(TTSProvider):
    """Buffers text to sentence boundaries, then emits one frame per sentence.

    Mirrors the flush policy of the real Kokoro provider so the event stream
    has the same shape it would in production, and sizes each frame to the
    runtime's assumed speech rate so the playback ledger's char accounting is
    exercised with realistic numbers rather than a token frame.
    """

    name = "test_scripted"

    def __init__(self, **_options: Any) -> None:
        self.spoken: list[str] = []

    def _frame_for(self, text: str) -> AudioFrame:
        seconds = len(text) / SPOKEN_CHARS_PER_SECOND
        samples = math.ceil(seconds * 16000)
        return AudioFrame(pcm=b"\x00\x00" * samples, sample_rate=16000, channels=1)

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        buffer = ""
        async for delta in text:
            buffer += delta
            if buffer.rstrip().endswith((".", "!", "?")):
                segment, buffer = buffer.strip(), ""
                self.spoken.append(segment)
                yield TTSChunk(
                    frame=self._frame_for(segment),
                    usage=[
                        Usage(
                            units=float(len(segment)),
                            unit_name="tts_chars",
                            estimated=False,
                            model="fake-voice",
                        )
                    ],
                )
        tail = buffer.strip()
        if tail:
            self.spoken.append(tail)
            yield TTSChunk(
                frame=self._frame_for(tail),
                usage=[
                    Usage(
                        units=float(len(tail)),
                        unit_name="tts_chars",
                        estimated=False,
                        model="fake-voice",
                    )
                ],
            )


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

GREETING = "Hello, front desk here."
PLAIN_ANSWER = "We are open nine to five."
PRE_TOOL_SPEECH = "Sure, let me book that."
WAITING_MESSAGE = "One moment while I check the diary."
POST_TOOL_SPEECH = "You are booked for two at seven."

TWO_TURN_SCRIPT = [
    '{"speak": "' + PLAIN_ANSWER + '", "tool_call": null}',
    (
        '{"speak": "' + PRE_TOOL_SPEECH + '", "tool_call": {"name": "book_table", '
        '"arguments": {"party_size": 2, "waiting_message": "' + WAITING_MESSAGE + '", '
        '"spoken_mode": "answer_pending", "post_tool_response": "respond"}}}'
    ),
    '{"speak": "' + POST_TOOL_SPEECH + '", "tool_call": null}',
]

TEST_RATES = RateCard(
    version="test-0",
    rates=[
        Rate(
            component=CostComponent.STT,
            provider="test_scripted",
            unit_name="audio_seconds",
            price_per_unit=0.001,
            as_of="2026-01-01",
        ),
        Rate(
            component=CostComponent.LLM,
            provider="test_scripted",
            unit_name="tokens_in",
            price_per_unit=0.002,
            as_of="2026-01-01",
        ),
        Rate(
            component=CostComponent.LLM,
            provider="test_scripted",
            unit_name="tokens_out",
            price_per_unit=0.003,
            as_of="2026-01-01",
        ),
        Rate(
            component=CostComponent.TTS,
            provider="test_scripted",
            unit_name="tts_chars",
            price_per_unit=0.0001,
            as_of="2026-01-01",
        ),
    ],
)


def build_agent(script_id: str, greeting: str | None = GREETING) -> AgentSpec:
    LLM_SCRIPTS.setdefault(script_id, list(TWO_TURN_SCRIPT))
    return AgentSpec(
        name="front-desk",
        persona="You are a friendly front-desk assistant for a small restaurant.",
        greeting=greeting,
        language=LanguagePolicy(primary="en"),
        runtime=RuntimeConfig(
            routing={
                "default": ProviderSelection(
                    stt="test_scripted",
                    llm="test_scripted",
                    tts="test_scripted",
                    options={"llm": {"script_id": script_id}},
                )
            }
        ),
        tools=[
            ToolDef(
                name="book_table",
                description="Reserve a table.",
                parameters={
                    "type": "object",
                    "properties": {"party_size": {"type": "integer"}},
                    "required": ["party_size"],
                },
                handler="book_table",
            )
        ],
        limits=Limits(max_tool_calls=4),
    )


def types_of(session: CallSession, exclude: tuple[str, ...] = ()) -> list[str]:
    return [e.type for e in session.history if e.type not in exclude]


def texts_of(session: CallSession, event_type: str) -> list[str]:
    return [e.text for e in session.history if e.type == event_type]


async def run_two_turn_call(
    script_id: str,
    meter: CostMeter | None = None,
    handlers: dict[str, Any] | None = None,
    session: CallSession | None = None,
) -> tuple[CallSession, CascadeRuntime, list[AudioFrame]]:
    """Drive the scripted two-turn conversation end to end."""
    if session is None:
        session = CallSession(build_agent(script_id))
    audio: list[AudioFrame] = []
    runtime = CascadeRuntime(session, handlers=handlers, meter=meter)
    runtime.on_bot_audio = audio.append

    await runtime.start()
    await runtime.push_audio(AudioFrame(pcm=b"what are your hours"))
    await runtime.drain()
    await runtime.push_audio(AudioFrame(pcm=b"book me a table for two"))
    await runtime.drain()
    await runtime.stop()
    return session, runtime, audio


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_capabilities_declare_the_cascade_tradeoffs() -> None:
    session = CallSession(build_agent("caps"))
    caps = CascadeRuntime(session).capabilities
    assert caps.live_transcripts is True  # transcripts arrive mid-call, not post-call
    assert caps.mid_call_tool_calls is True
    assert caps.barge_in is True
    assert caps.exact_usage_reporting is True
    assert caps.local_capable is True


async def test_full_ordered_event_sequence() -> None:
    """The whole conversation, asserted as one ordered event sequence.

    Ordering is the contract here, not just membership: a consumer replaying
    this stream must be able to reconstruct what the caller experienced.
    """
    calls: list[dict[str, Any]] = []

    async def book_table(arguments: dict[str, Any]) -> dict[str, Any]:
        calls.append(arguments)
        return {"confirmation": "R-417", "time": "19:00"}

    session, _runtime, _audio = await run_two_turn_call(
        "ordered", handlers={"book_table": book_table}
    )

    assert types_of(session) == [
        "session_started",
        "bot_utterance",  # greeting, spoken before the caller says anything
        "bot_speech_played",
        "user_transcript",  # turn 1: a plain question
        "bot_utterance",
        "bot_speech_played",
        "user_transcript",  # turn 2: the choreographed tool call
        "bot_utterance",  # ... spoken BEFORE the tool runs
        "bot_speech_played",
        "tool_call_started",
        "bot_utterance",  # the waiting message, because spoken_mode is answer_pending
        "bot_speech_played",
        "tool_call_completed",
        "bot_utterance",  # the follow-up turn, because post_tool_response is respond
        "bot_speech_played",
        "session_ended",
    ]

    assert texts_of(session, "bot_utterance") == [
        GREETING,
        PLAIN_ANSWER,
        PRE_TOOL_SPEECH,
        WAITING_MESSAGE,
        POST_TOOL_SPEECH,
    ]
    # Only the tool's own declared arguments reach the handler; the three
    # choreography fields are consumed by the primitive, not the tool.
    assert calls == [{"party_size": 2}]


async def test_speech_is_emitted_before_the_tool_call_starts() -> None:
    """The latency guarantee, stated as an ordering assertion.

    If ``BotUtterance`` for the pre-tool speech landed *after*
    ``ToolCallStarted``, the caller would have sat in silence for the whole
    tool round trip -- which is the exact failure the envelope, the streaming
    parser and the choreography primitive exist to prevent.
    """

    async def book_table(_arguments: dict[str, Any]) -> dict[str, Any]:
        return {"confirmation": "R-417"}

    session, _runtime, _audio = await run_two_turn_call(
        "ordering", handlers={"book_table": book_table}
    )

    kinds = types_of(session)
    tool_started = kinds.index("tool_call_started")
    speech_indexes = [
        i
        for i, event in enumerate(session.history)
        if event.type == "bot_utterance" and event.text == PRE_TOOL_SPEECH
    ]
    assert speech_indexes and speech_indexes[0] < tool_started

    # And the waiting message lands inside the tool call, not after it.
    waiting = next(
        i
        for i, event in enumerate(session.history)
        if event.type == "bot_utterance" and event.text == WAITING_MESSAGE
    )
    assert tool_started < waiting < kinds.index("tool_call_completed")


async def test_bot_audio_frames_reach_the_transport() -> None:
    async def book_table(_arguments: dict[str, Any]) -> dict[str, Any]:
        return {"confirmation": "R-417"}

    _session, _runtime, audio = await run_two_turn_call(
        "audio", handlers={"book_table": book_table}
    )

    # One frame per spoken utterance: greeting, plain answer, pre-tool speech,
    # waiting message, follow-up.
    assert len(audio) == 5
    assert all(frame.sample_rate == 16000 and frame.channels == 1 for frame in audio)
    assert all(len(frame.pcm) > 0 for frame in audio)


async def test_cost_is_recorded_per_stage_when_a_meter_is_attached() -> None:
    async def book_table(_arguments: dict[str, Any]) -> dict[str, Any]:
        return {"confirmation": "R-417"}

    session = CallSession(build_agent("cost"))
    meter = CostMeter(session, TEST_RATES)
    session, _runtime, _audio = await run_two_turn_call(
        "cost", meter=meter, handlers={"book_table": book_table}, session=session
    )

    costs = [e for e in session.history if e.type == "cost_recorded"]
    assert costs, "a meter was attached but nothing was metered"

    by_component: dict[CostComponent, list[Any]] = {}
    for event in costs:
        by_component.setdefault(event.component, []).append(event)

    # Two caller utterances transcribed.
    assert len(by_component[CostComponent.STT]) == 2
    # Three LLM turns (two caller turns plus the post-tool follow-up), each
    # reporting tokens in and out.
    assert len(by_component[CostComponent.LLM]) == 6
    # Five synthesized utterances.
    assert len(by_component[CostComponent.TTS]) == 5

    # Honest accounting: the fakes report measured units, so nothing here may
    # be flagged as estimated, and the rate card must have priced all of it.
    assert all(event.estimated is False for event in costs)
    assert meter.missing_rate_notes == []
    assert all(event.amount > 0 for event in costs)


async def test_cost_events_are_absent_without_a_meter() -> None:
    """A runtime with no meter stays silent rather than emitting zero-cost noise."""

    async def book_table(_arguments: dict[str, Any]) -> dict[str, Any]:
        return {"confirmation": "R-417"}

    session, _runtime, _audio = await run_two_turn_call(
        "nometer", handlers={"book_table": book_table}
    )
    assert not [e for e in session.history if e.type == "cost_recorded"]


async def test_prompt_prefix_is_stable_and_carries_the_contracts() -> None:
    """The prefix the model sees: persona, envelope contract, augmented schemas.

    Byte-stability of this prefix is what keeps provider prompt caches warm, so
    it is asserted directly: every turn must send the same opening messages,
    with the language directive appended at the end rather than folded in.
    """

    async def book_table(_arguments: dict[str, Any]) -> dict[str, Any]:
        return {"confirmation": "R-417"}

    session, runtime, _audio = await run_two_turn_call(
        "prompt", handlers={"book_table": book_table}
    )
    llm = runtime._llm
    assert isinstance(llm, ScriptedLLM)
    assert len(llm.calls) == 3

    first = llm.calls[0]
    assert first[0]["content"] == session.agent.persona
    assert '"speak"' in first[1]["content"] and '"tool_call"' in first[1]["content"]
    # Choreography turned the tool's own schema into one the model cannot
    # satisfy without planning the silence around the call.
    assert "waiting_message" in first[2]["content"]
    assert "spoken_mode" in first[2]["content"]
    assert "post_tool_response" in first[2]["content"]

    # The language lock rides at the very end, and the prefix never moves.
    assert first[-1]["content"].startswith("LANGUAGE DIRECTIVE")
    for call in llm.calls[1:]:
        assert call[:3] == first[:3]
        assert call[-1] == first[-1]

    # The conversation only ever grew between the prefix and the directive.
    assert len(llm.calls[1]) > len(llm.calls[0])
    assert any(m["role"] == "tool" for m in llm.calls[2])


async def test_unbound_tool_handler_degrades_into_a_spoken_apology() -> None:
    """A missing handler must not kill the call: it becomes a normal tool failure.

    ``post_tool_response`` was ``respond`` here anyway, but choreography would
    force a response regardless -- a model that planned to stay silent did so
    believing the tool would work.
    """
    session, _runtime, _audio = await run_two_turn_call("unbound", handlers={})

    kinds = types_of(session)
    assert "tool_call_completed" in kinds
    completed = next(e for e in session.history if e.type == "tool_call_completed")
    assert completed.ok is False
    assert completed.result_summary is not None
    assert "book_table" in completed.result_summary
    # The caller still hears a closing line, and the call ends normally.
    assert texts_of(session, "bot_utterance")[-1] == POST_TOOL_SPEECH
    assert kinds[-1] == "session_ended"


async def test_session_ended_reports_reason_and_duration() -> None:
    LLM_SCRIPTS["ended"] = list(TWO_TURN_SCRIPT)
    session = CallSession(build_agent("ended", greeting=None))
    runtime = CascadeRuntime(session)
    await runtime.start()
    await runtime.stop(reason="caller_hung_up")
    await runtime.stop()  # idempotent: no second SessionEnded

    ended = [e for e in session.history if e.type == "session_ended"]
    assert len(ended) == 1
    assert ended[0].reason == "caller_hung_up"
    assert ended[0].duration_seconds >= 0.0


async def test_missing_provider_in_routing_fails_with_a_useful_message() -> None:
    agent = build_agent("missing")
    agent.runtime.routing["default"] = ProviderSelection(
        stt="test_scripted", llm=None, tts="test_scripted"
    )
    runtime = CascadeRuntime(CallSession(agent))
    with pytest.raises(ValueError, match="cascade runtime needs a llm provider"):
        await runtime.start()


async def test_builtin_text_input_stt_drives_the_pipeline_without_ml_deps() -> None:
    """The console dev loop, proven: a real built-in provider, no ML stack.

    ``text_input`` is a shipped provider, not a test double, so this also
    covers the registry path that ``_load_builtin`` sets up -- the same one a
    ``pip install alaap`` user hits when they point a spec at it.
    """
    LLM_SCRIPTS["console"] = [TWO_TURN_SCRIPT[0]]
    agent = build_agent("console", greeting=None)
    agent.runtime.routing["default"] = ProviderSelection(
        stt="text_input",
        llm="test_scripted",
        tts="test_scripted",
        options={"llm": {"script_id": "console"}},
    )

    session = CallSession(agent)
    runtime = CascadeRuntime(session)
    await runtime.start()
    await runtime.push_audio(AudioFrame(pcm=b"what are your hours"))
    await runtime.drain()
    await runtime.stop()

    assert texts_of(session, "user_transcript") == ["what are your hours"]
    assert texts_of(session, "bot_utterance") == [PLAIN_ANSWER]
