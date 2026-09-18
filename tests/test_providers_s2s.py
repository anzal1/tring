"""Tests for the speech-to-speech provider adapters (OpenAI Realtime, Ultravox).

No network, no API keys, no websockets install. Both providers take a
``connect`` seam (see ``_WebSocketLike`` in ``providers/cloud/s2s_extra.py``)
that a scripted fake socket satisfies, and Ultravox's REST leg runs through an
``httpx.MockTransport``. What is actually asserted is the part that breaks
silently in production: the URL, the auth header, the handshake payload, and
the event-to-``TTSChunk`` mapping.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from tring.cost.rates import RateCard
from tring.events import CostComponent
from tring.providers import registry
from tring.providers.base import S2SProvider, TTSChunk
from tring.providers.cloud.s2s_extra import (
    OPENAI_REALTIME_RATE,
    S2S_RATES,
    WIRE_SAMPLE_RATE,
    OpenAIRealtimeS2S,
    UltravoxS2S,
    _resample_pcm16,
)
from tring.providers.registry import UnknownProviderError
from tring.runtimes.base import AudioFrame

# ---------------------------------------------------------------------------
# Scripted fakes
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """A scripted websocket: replays ``inbound``, records everything sent.

    The ``gate`` is what makes these tests deterministic rather than
    timing-dependent. Each provider runs a *pump task* that drains the caller
    frames into the socket concurrently with the read loop; the gate holds the
    inbound replay until the frame source signals it is exhausted, so by the
    time the first server message is delivered every client message has
    already been sent and ``sent`` can be asserted on without a sleep.
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

    # -- convenience for assertions ------------------------------------------

    @property
    def sent_json(self) -> list[dict[str, Any]]:
        return [json.loads(m) for m in self.sent if isinstance(m, str)]

    @property
    def sent_binary(self) -> list[bytes]:
        return [m for m in self.sent if isinstance(m, bytes)]


class RecordingConnect:
    """A ``WebSocketConnect`` that hands back a fixed socket and remembers the
    url/headers it was called with."""

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
    samples = sample_rate * ms // 1000
    return AudioFrame(pcm=b"\x00\x01" * samples, sample_rate=sample_rate, channels=1)


async def _drain(provider: S2SProvider, frames: AsyncIterator[AudioFrame]) -> list[TTSChunk]:
    return [chunk async for chunk in provider.converse(frames)]


# ---------------------------------------------------------------------------
# _resample_pcm16
# ---------------------------------------------------------------------------


def test_resample_is_identity_at_the_same_rate() -> None:
    pcm = b"\x01\x02\x03\x04"
    assert _resample_pcm16(pcm, 16000, 16000) is pcm


def test_resample_16k_to_24k_scales_sample_count_by_three_halves() -> None:
    pcm = b"\x00\x10" * 160  # 10 ms at 16 kHz
    out = _resample_pcm16(pcm, WIRE_SAMPLE_RATE, OPENAI_REALTIME_RATE)
    assert len(out) // 2 == 240  # 10 ms at 24 kHz


def test_resample_24k_to_16k_scales_sample_count_by_two_thirds() -> None:
    pcm = b"\x00\x10" * 240
    out = _resample_pcm16(pcm, OPENAI_REALTIME_RATE, WIRE_SAMPLE_RATE)
    assert len(out) // 2 == 160


def test_resample_tolerates_a_half_sample_tail() -> None:
    # A websocket chunk boundary is not obliged to land on a sample boundary.
    out = _resample_pcm16(b"\x00\x10" * 160 + b"\x07", WIRE_SAMPLE_RATE, OPENAI_REALTIME_RATE)
    assert len(out) % 2 == 0


def test_resample_of_empty_audio_is_empty() -> None:
    assert _resample_pcm16(b"", 24000, 16000) == b""


# ---------------------------------------------------------------------------
# OpenAI Realtime: request construction
# ---------------------------------------------------------------------------


def test_openai_realtime_url_and_auth_header() -> None:
    provider = OpenAIRealtimeS2S()
    assert provider.build_url() == "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
    assert provider.build_headers("sk-test") == {"Authorization": "Bearer sk-test"}
    # The pre-GA OpenAI-Beta header is deprecated and must not be sent.
    assert "OpenAI-Beta" not in provider.build_headers("sk-test")


def test_openai_realtime_session_update_payload() -> None:
    provider = OpenAIRealtimeS2S(
        voice="cedar", instructions="Be brief.", transcription_model="whisper-1"
    )
    event = provider.build_session_update()

    assert event["type"] == "session.update"
    session = event["session"]
    assert session["type"] == "realtime"
    assert session["model"] == "gpt-realtime-2.1"
    assert session["output_modalities"] == ["audio"]
    assert session["instructions"] == "Be brief."
    # 24 kHz PCM is the Realtime API's only supported PCM rate.
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["voice"] == "cedar"
    assert session["audio"]["input"]["turn_detection"] == {"type": "semantic_vad"}
    assert session["audio"]["input"]["transcription"] == {"model": "whisper-1"}


def test_openai_realtime_transcription_is_omitted_when_disabled() -> None:
    session = OpenAIRealtimeS2S(transcription_model=None).build_session_update()["session"]
    assert "transcription" not in session["audio"]["input"]


def test_openai_realtime_audio_append_is_base64_pcm_at_24k() -> None:
    provider = OpenAIRealtimeS2S()
    event = provider.build_audio_append(_silence(WIRE_SAMPLE_RATE, ms=20))
    assert event["type"] == "input_audio_buffer.append"
    decoded = base64.b64decode(event["audio"])
    assert len(decoded) // 2 == 480  # 20 ms at 24 kHz, resampled from 16 kHz


# ---------------------------------------------------------------------------
# OpenAI Realtime: the session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_realtime_handshake_and_audio_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    gate = asyncio.Event()
    # 480 samples of 24 kHz audio -> 320 samples (640 bytes) at 16 kHz.
    delta = base64.b64encode(b"\x00\x20" * 480).decode("ascii")
    socket = FakeWebSocket(
        [
            json.dumps({"type": "session.created"}),
            json.dumps({"type": "response.output_audio.delta", "delta": delta}),
        ],
        gate=gate,
    )
    connect = RecordingConnect(socket)
    provider = OpenAIRealtimeS2S(connect=connect)

    chunks = await _drain(provider, _frames([_silence(), _silence()], gate))

    assert connect.url == "wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1"
    assert connect.headers == {"Authorization": "Bearer sk-unit-test"}
    # First thing on the wire is the configuration event, then one append per frame.
    sent = socket.sent_json
    assert sent[0]["type"] == "session.update"
    assert [m["type"] for m in sent[1:]] == ["input_audio_buffer.append"] * 2
    # No manual commit / response.create: semantic VAD drives turns server-side.
    assert not any(m["type"] == "input_audio_buffer.commit" for m in sent)

    assert len(chunks) == 1
    assert chunks[0].frame.sample_rate == WIRE_SAMPLE_RATE
    assert len(chunks[0].frame.pcm) // 2 == 320
    assert chunks[0].usage == []


@pytest.mark.asyncio
async def test_openai_realtime_missing_api_key_names_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    provider = OpenAIRealtimeS2S(connect=RecordingConnect(FakeWebSocket()))
    gate = asyncio.Event()
    gate.set()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await _drain(provider, _frames([], gate))


@pytest.mark.asyncio
async def test_openai_realtime_response_done_usage_is_exact_with_cached_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    gate = asyncio.Event()
    gate.set()
    socket = FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "response.done",
                    "response": {
                        "status": "completed",
                        "usage": {
                            "total_tokens": 1500,
                            "input_tokens": 1200,
                            "output_tokens": 300,
                            "input_token_details": {
                                "cached_tokens": 900,
                                "audio_tokens": 1100,
                                "text_tokens": 100,
                            },
                        },
                    },
                }
            )
        ],
        gate=gate,
    )
    provider = OpenAIRealtimeS2S(connect=RecordingConnect(socket))

    chunks = await _drain(provider, _frames([], gate))

    assert len(chunks) == 1
    # The usage carrier is a zero-length frame: a no-op for any transport.
    assert chunks[0].frame.pcm == b""
    tokens_in, tokens_out = chunks[0].usage
    assert (tokens_in.units, tokens_in.unit_name) == (1200.0, "tokens_in")
    assert tokens_in.estimated is False
    assert tokens_in.cached_units == 900.0  # the prompt-cache honesty bit
    assert tokens_in.model == "gpt-realtime-2.1"
    assert (tokens_out.units, tokens_out.unit_name) == (300.0, "tokens_out")
    assert tokens_out.estimated is False
    assert tokens_out.cached_units is None


@pytest.mark.asyncio
async def test_openai_realtime_response_done_without_usage_reports_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    gate = asyncio.Event()
    gate.set()
    socket = FakeWebSocket(
        [json.dumps({"type": "response.done", "response": {"status": "cancelled"}})],
        gate=gate,
    )
    provider = OpenAIRealtimeS2S(connect=RecordingConnect(socket))
    assert await _drain(provider, _frames([], gate)) == []


@pytest.mark.asyncio
async def test_openai_realtime_collects_both_halves_of_the_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    gate = asyncio.Event()
    gate.set()
    socket = FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "item_id": "item_1",
                    "transcript": "I need to change my booking.",
                }
            ),
            json.dumps(
                {
                    "type": "response.output_audio_transcript.done",
                    "transcript": "Sure, what date works?",
                }
            ),
        ],
        gate=gate,
    )
    provider = OpenAIRealtimeS2S(connect=RecordingConnect(socket))

    await _drain(provider, _frames([], gate))

    assert await provider.post_call_transcript() == [
        {"role": "user", "text": "I need to change my booking."},
        {"role": "bot", "text": "Sure, what date works?"},
    ]


@pytest.mark.asyncio
async def test_openai_realtime_surfaces_error_events(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-unit-test")
    gate = asyncio.Event()
    gate.set()
    socket = FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "bad voice"},
                }
            )
        ],
        gate=gate,
    )
    provider = OpenAIRealtimeS2S(connect=RecordingConnect(socket))
    with pytest.raises(RuntimeError, match="bad voice"):
        await _drain(provider, _frames([], gate))
    assert socket.closed  # the socket is closed even on the error path


# ---------------------------------------------------------------------------
# Ultravox: the REST leg
# ---------------------------------------------------------------------------


def _ultravox_transport(
    captured: list[httpx.Request], payload: dict[str, Any] | None = None
) -> httpx.MockTransport:
    body = payload or {"callId": "call-abc", "joinUrl": "wss://ws.ultravox.ai/calls/abc"}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(201, json=body)

    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_ultravox_create_call_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRAVOX_API_KEY", "uv-unit-test")
    captured: list[httpx.Request] = []
    provider = UltravoxS2S(
        system_prompt="You are a scheduling agent.",
        voice="Mark",
        language_hint="en",
        transport=_ultravox_transport(captured),
    )

    join_url = await provider.create_call()

    assert join_url == "wss://ws.ultravox.ai/calls/abc"
    assert provider.call_id == "call-abc"
    request = captured[0]
    assert request.method == "POST"
    assert str(request.url) == "https://api.ultravox.ai/api/calls"
    assert request.headers["X-API-Key"] == "uv-unit-test"
    body = json.loads(request.content)
    assert body["model"] == "ultravox-v0.7"
    assert body["voice"] == "Mark"
    assert body["systemPrompt"] == "You are a scheduling agent."
    assert body["languageHint"] == "en"
    assert body["medium"] == {
        "serverWebSocket": {
            "inputSampleRate": 16000,
            "outputSampleRate": 16000,
            "clientBufferSizeMs": 30000,
        }
    }


@pytest.mark.asyncio
async def test_ultravox_missing_api_key_names_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ULTRAVOX_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ULTRAVOX_API_KEY"):
        await UltravoxS2S(transport=_ultravox_transport([])).create_call()


@pytest.mark.asyncio
async def test_ultravox_missing_join_url_is_an_explicit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ULTRAVOX_API_KEY", "uv-unit-test")
    provider = UltravoxS2S(transport=_ultravox_transport([], payload={"callId": "x"}))
    with pytest.raises(RuntimeError, match="joinUrl"):
        await provider.create_call()


# ---------------------------------------------------------------------------
# Ultravox: the websocket leg
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ultravox_bridges_audio_and_answers_ping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ULTRAVOX_API_KEY", "uv-unit-test")
    gate = asyncio.Event()
    socket = FakeWebSocket(
        [
            json.dumps({"type": "call_started", "callId": "call-abc"}),
            b"\x00\x10" * 160,  # raw s16le at the negotiated 16 kHz
            json.dumps({"type": "ping", "timestamp": 12.5}),
            json.dumps({"type": "state", "state": "speaking"}),
        ],
        gate=gate,
    )
    connect = RecordingConnect(socket)
    provider = UltravoxS2S(transport=_ultravox_transport([]), connect=connect)

    chunks = await _drain(provider, _frames([_silence(), _silence()], gate))

    # joinUrl is pre-authenticated: no auth header goes on the socket.
    assert connect.url == "wss://ws.ultravox.ai/calls/abc"
    assert connect.headers == {}
    # Caller audio goes out as raw binary, no base64 and no JSON envelope.
    assert len(socket.sent_binary) == 2
    assert socket.sent_binary[0] == _silence().pcm
    # ping is answered with a timestamp-echoing pong.
    assert socket.sent_json == [{"type": "pong", "timestamp": 12.5}]

    audio_chunks = [c for c in chunks if c.frame.pcm]
    assert len(audio_chunks) == 1
    assert audio_chunks[0].frame.sample_rate == WIRE_SAMPLE_RATE
    assert audio_chunks[0].frame.pcm == b"\x00\x10" * 160


@pytest.mark.asyncio
async def test_ultravox_resamples_a_non_canonical_output_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ULTRAVOX_API_KEY", "uv-unit-test")
    gate = asyncio.Event()
    gate.set()
    socket = FakeWebSocket([b"\x00\x10" * 480], gate=gate)  # 10 ms at 48 kHz
    provider = UltravoxS2S(
        input_sample_rate=48000,
        output_sample_rate=48000,
        transport=_ultravox_transport([]),
        connect=RecordingConnect(socket),
    )

    chunks = await _drain(provider, _frames([], gate))
    audio = [c for c in chunks if c.frame.pcm]

    assert len(audio) == 1
    assert audio[0].frame.sample_rate == WIRE_SAMPLE_RATE
    assert len(audio[0].frame.pcm) // 2 == 160  # 10 ms at 16 kHz


@pytest.mark.asyncio
async def test_ultravox_usage_is_duration_and_labelled_estimated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ULTRAVOX_API_KEY", "uv-unit-test")
    gate = asyncio.Event()
    gate.set()
    provider = UltravoxS2S(
        transport=_ultravox_transport([]), connect=RecordingConnect(FakeWebSocket(gate=gate))
    )

    chunks = await _drain(provider, _frames([], gate))

    assert len(chunks) == 1
    (usage,) = chunks[-1].usage
    assert usage.unit_name == "audio_seconds"
    # Socket lifetime is a proxy for billed minutes, never a measurement.
    assert usage.estimated is True
    assert usage.units >= 0.0
    assert chunks[-1].frame.pcm == b""


@pytest.mark.asyncio
async def test_ultravox_transcript_deltas_fold_by_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ULTRAVOX_API_KEY", "uv-unit-test")
    gate = asyncio.Event()
    gate.set()
    socket = FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "transcript",
                    "role": "user",
                    "medium": "voice",
                    "text": "hello",
                    "final": True,
                    "ordinal": 0,
                }
            ),
            json.dumps(
                {
                    "type": "transcript",
                    "role": "agent",
                    "medium": "voice",
                    "delta": "hi ",
                    "final": False,
                    "ordinal": 1,
                }
            ),
            json.dumps(
                {
                    "type": "transcript",
                    "role": "agent",
                    "medium": "voice",
                    "delta": "there",
                    "final": True,
                    "ordinal": 1,
                }
            ),
        ],
        gate=gate,
    )
    provider = UltravoxS2S(transport=_ultravox_transport([]), connect=RecordingConnect(socket))

    await _drain(provider, _frames([], gate))

    # "agent" is translated to the event contract's "bot"; deltas concatenate.
    assert await provider.post_call_transcript() == [
        {"role": "user", "text": "hello"},
        {"role": "bot", "text": "hi there"},
    ]


# ---------------------------------------------------------------------------
# Registry + rates
# ---------------------------------------------------------------------------


def test_providers_are_registered_under_their_names() -> None:
    registry._load_builtin()
    assert ("s2s", "openai_realtime") in registry.available("s2s")
    assert ("s2s", "ultravox") in registry.available("s2s")


def test_registry_create_builds_the_right_classes() -> None:
    registry._load_builtin()
    assert isinstance(registry.create("s2s", "openai_realtime"), OpenAIRealtimeS2S)
    assert isinstance(registry.create("s2s", "ultravox"), UltravoxS2S)


def test_registry_create_passes_options_through() -> None:
    registry._load_builtin()
    provider = registry.create(
        "s2s", "openai_realtime", model="gpt-realtime", api_key_env="OPENAI_KEY_PROD"
    )
    assert provider.build_url().endswith("?model=gpt-realtime")
    assert provider._api_key_env == "OPENAI_KEY_PROD"


def test_unknown_s2s_provider_still_raises() -> None:
    registry._load_builtin()
    with pytest.raises(UnknownProviderError):
        registry.create("s2s", "not-a-real-vendor")


def test_s2s_rates_are_s2s_priced_and_dated() -> None:
    assert S2S_RATES
    for rate in S2S_RATES:
        assert rate.component is CostComponent.S2S
        assert rate.as_of == "2026-09"
        assert rate.currency == "USD"
        assert rate.price_per_unit > 0.0


def test_s2s_rates_are_usable_by_the_cost_engine() -> None:
    card = RateCard(version="test", rates=S2S_RATES)
    tokens_in = card.lookup(
        CostComponent.S2S, "openai_realtime", "tokens_in", model="gpt-realtime-2.1"
    )
    assert tokens_in is not None
    assert tokens_in.price_per_unit == pytest.approx(32.0 / 1_000_000)

    minutes = card.lookup(CostComponent.S2S, "ultravox", "audio_seconds")
    assert minutes is not None
    assert minutes.price_per_unit * 60 == pytest.approx(0.05)


def test_rate_provider_names_match_the_registered_provider_names() -> None:
    # A mismatch here is the classic silent-zero-cost bug: the meter looks a
    # rate up by provider.name, so the two spellings have to agree.
    priced = {rate.provider for rate in S2S_RATES}
    assert priced == {OpenAIRealtimeS2S.name, UltravoxS2S.name}
