"""Tests for the Twilio Media Streams transport (mu-law codec + bridge).

No network, no real WebSocket, no ``websockets`` package touched: the codec
functions are pure, and ``_TwilioBridge`` -- the message-handling core of
``transports/twilio.py`` -- is deliberately independent of the ``websockets``
library (see its ``_WebSocketLike`` Protocol), so it is driven here with a
scripted fake exactly like ``FakeWebSocket`` in ``tests/test_providers_s2s.py``.

mu-law reference values below (byte codes and decode-of-encode round trips)
were cross-checked, at development time only, against CPython's
``audioop.lin2ulaw``/``audioop.ulaw2lin`` -- the stdlib's own G.711
implementation -- across the *entire* 16-bit domain (all 65536 values in
both directions) with zero mismatches. ``audioop`` is not imported here or
anywhere in the shipped module (it is removed in Python 3.13); the values
are hardcoded so this test suite never depends on it either.
"""

from __future__ import annotations

import array
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from tring.agent import AgentSpec
from tring.events import BotSpeechPlayed, BotUtterance
from tring.primitives.interruption import PlaybackLedger
from tring.runtimes.base import AudioFrame, RuntimeAdapter, RuntimeCapabilities
from tring.session import CallSession
from tring.transports.mulaw import (
    linear16_to_mulaw,
    linear_to_ulaw_sample,
    mulaw_to_linear16,
    resample_linear16,
    ulaw_to_linear_sample,
)
from tring.transports.twilio import _TwilioBridge

# ---------------------------------------------------------------------------
# mu-law codec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sample", "expected_byte"),
    [
        (0, 0xFF),  # silence
        (1, 0xFF),
        (-1, 0x7E),
        (124, 0xEF),
        (-124, 0x6F),
        (1000, 0xCE),
        (-1000, 0x4E),
        (30000, 0x82),
        (-30000, 0x02),
        (32767, 0x80),  # positive full scale
        (-32768, 0x00),  # negative full scale
    ],
)
def test_linear_to_ulaw_sample_matches_g711_reference(sample: int, expected_byte: int) -> None:
    assert linear_to_ulaw_sample(sample) == expected_byte


@pytest.mark.parametrize(
    ("byte", "expected_sample"),
    [
        (0xFF, 0),
        (0x7E, -8),
        (0xEF, 132),
        (0x6F, -132),
        (0xCE, 988),
        (0x4E, -988),
        (0x82, 30076),
        (0x02, -30076),
        (0x80, 32124),
        (0x00, -32124),
        (0xFE, 8),
    ],
)
def test_ulaw_to_linear_sample_matches_g711_reference(byte: int, expected_sample: int) -> None:
    assert ulaw_to_linear_sample(byte) == expected_sample


@pytest.mark.parametrize(
    ("sample", "expected_after_round_trip"),
    [
        (0, 0),
        (5000, 5116),
        (-5000, -5116),
        (20000, 19836),
        (-20000, -19836),
        (32000, 32124),
        (-32000, -32124),
    ],
)
def test_ulaw_round_trip_is_lossy_but_deterministic(
    sample: int, expected_after_round_trip: int
) -> None:
    """encode(decode(x)) does not return x -- mu-law is lossy by design --
    but it must return the *same* quantized neighbor every time."""
    encoded = linear_to_ulaw_sample(sample)
    assert ulaw_to_linear_sample(encoded) == expected_after_round_trip


def test_linear16_to_mulaw_drops_a_trailing_odd_byte_instead_of_raising() -> None:
    """A transport slices audio on arbitrary chunk boundaries; half a
    16-bit sample at the edge of a chunk must not crash a live call."""
    assert linear16_to_mulaw(b"\x00\x00\x01") == linear16_to_mulaw(b"\x00\x00")


def test_mulaw_buffer_helpers_round_trip_a_whole_buffer() -> None:
    original = array.array("h", [0, 1000, -1000, 32767, -32768])
    encoded = linear16_to_mulaw(original.tobytes())
    assert len(encoded) == len(original)  # one mu-law byte per PCM16 sample
    decoded = array.array("h")
    decoded.frombytes(mulaw_to_linear16(encoded))
    assert list(decoded) == [
        ulaw_to_linear_sample(linear_to_ulaw_sample(s)) for s in original
    ]


# ---------------------------------------------------------------------------
# resampling
# ---------------------------------------------------------------------------


def test_resample_upsamples_8k_to_16k_by_doubling_sample_count() -> None:
    pcm_8k = array.array("h", range(0, 1600)).tobytes()  # 1600 samples
    up = resample_linear16(pcm_8k, 8000, 16000)
    assert len(up) // 2 == 3200


def test_resample_downsamples_16k_to_8k_by_halving_sample_count() -> None:
    pcm_16k = array.array("h", range(0, 3200)).tobytes()  # 3200 samples
    down = resample_linear16(pcm_16k, 16000, 8000)
    assert len(down) // 2 == 1600


def test_resample_same_rate_is_a_passthrough() -> None:
    pcm = b"\x01\x02\x03\x04"
    assert resample_linear16(pcm, 16000, 16000) is pcm


def test_resample_rejects_too_short_input_gracefully() -> None:
    assert resample_linear16(b"", 8000, 16000) == b""
    # One sample: no second sample to interpolate against.
    assert resample_linear16(b"\x00\x00", 8000, 16000) == b""


# ---------------------------------------------------------------------------
# Twilio Media Streams bridge -- scripted fakes
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """A scripted, queue-backed double for a Twilio Media Streams connection.

    Mirrors ``tests/test_providers_s2s.py``'s ``FakeWebSocket`` in spirit
    (records everything sent, satisfies the module's ``_WebSocketLike``
    Protocol) but uses a queue rather than a fixed replay list: the
    outbound-mark tests need to feed a scripted "ack" back mid-connection,
    something a fixed inbound list can't express deterministically.
    """

    def __init__(self) -> None:
        self._inbound: list[str] = []
        self.sent: list[str] = []
        self.closed = False

    def push(self, message: dict[str, Any]) -> None:
        self._inbound.append(json.dumps(message))

    async def send(self, message: str | bytes) -> None:
        assert isinstance(message, str)
        self.sent.append(message)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        return self._replay()

    async def _replay(self) -> AsyncIterator[str | bytes]:
        for message in self._inbound:
            yield message

    @property
    def sent_events(self) -> list[dict[str, Any]]:
        return [json.loads(m) for m in self.sent]


class FakeRuntime(RuntimeAdapter):
    """A minimal ``RuntimeAdapter`` recording what the transport does to it.

    Carries a real :class:`PlaybackLedger` (not a mock) so the mark-driven
    exact-playout tests exercise the actual production reconciliation
    logic in ``primitives/interruption.py``, not a stand-in for it.
    """

    def __init__(self, session: CallSession) -> None:
        super().__init__(session)
        self.started = False
        self.stop_reason: str | None = None
        self.pushed: list[AudioFrame] = []
        self.ledger = PlaybackLedger(session)

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            live_transcripts=True,
            mid_call_tool_calls=False,
            barge_in=True,
            exact_usage_reporting=True,
        )

    async def start(self) -> None:
        self.started = True

    async def push_audio(self, frame: AudioFrame) -> None:
        self.pushed.append(frame)

    async def stop(self, reason: str = "completed") -> None:
        self.stop_reason = reason


def _session() -> CallSession:
    return CallSession(AgentSpec(name="telephony-test", persona="You are a test agent."))


_STREAM_SID = "MZ00000000000000000000000000000000"


# ---------------------------------------------------------------------------
# inbound: start -> media -> stop
# ---------------------------------------------------------------------------


async def test_start_media_stop_decodes_and_upsamples_to_the_wire_format() -> None:
    session = _session()
    runtime = FakeRuntime(session)
    ws = FakeWebSocket()
    bridge = _TwilioBridge(ws, runtime)

    silence_8k_mulaw = bytes([0xFF]) * 160  # 20ms of mu-law silence at 8kHz

    ws.push({"event": "connected", "protocol": "Call", "version": "1.0.0"})
    ws.push(
        {
            "event": "start",
            "sequenceNumber": "1",
            "streamSid": _STREAM_SID,
            "start": {
                "streamSid": _STREAM_SID,
                "accountSid": "AC_test",
                "callSid": "CA_test",
                "tracks": ["inbound"],
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
            },
        }
    )
    ws.push(
        {
            "event": "media",
            "sequenceNumber": "2",
            "streamSid": _STREAM_SID,
            "media": {
                "track": "inbound",
                "chunk": "1",
                "timestamp": "20",
                "payload": base64.b64encode(silence_8k_mulaw).decode("ascii"),
            },
        }
    )
    ws.push(
        {
            "event": "stop",
            "sequenceNumber": "3",
            "streamSid": _STREAM_SID,
            "stop": {"accountSid": "AC_test", "callSid": "CA_test"},
        }
    )

    await bridge.run()

    assert runtime.started
    assert runtime.stop_reason == "twilio_stream_ended"
    assert bridge._stream_sid == _STREAM_SID
    assert len(runtime.pushed) == 1
    frame = runtime.pushed[0]
    assert frame.sample_rate == 16000
    assert frame.channels == 1
    # 160 mu-law bytes (8kHz) -> 160 PCM16 samples -> upsampled to 320 samples
    # at 16kHz -> 640 bytes.
    assert len(frame.pcm) == 640


async def test_unexpected_media_format_is_logged_not_fatal(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _session()
    runtime = FakeRuntime(session)
    ws = FakeWebSocket()
    bridge = _TwilioBridge(ws, runtime)

    ws.push(
        {
            "event": "start",
            "streamSid": _STREAM_SID,
            "start": {
                "streamSid": _STREAM_SID,
                "mediaFormat": {"encoding": "audio/x-alaw", "sampleRate": 8000, "channels": 1},
            },
        }
    )
    ws.push({"event": "stop", "streamSid": _STREAM_SID, "stop": {}})

    with caplog.at_level("WARNING"):
        await bridge.run()

    assert bridge._stream_sid == _STREAM_SID  # still binds the stream
    assert any("mediaFormat" in message for message in caplog.messages)


# ---------------------------------------------------------------------------
# outbound: bot audio -> encoded media frames + marks -> exact playout
# ---------------------------------------------------------------------------


async def test_outbound_audio_is_chunked_to_160_bytes_with_a_trailing_mark() -> None:
    session = _session()
    runtime = FakeRuntime(session)
    ws = FakeWebSocket()
    bridge = _TwilioBridge(ws, runtime)
    bridge._stream_sid = _STREAM_SID

    text = "Hello there caller, how can I help you today"
    runtime.ledger.utterance_started("utt-1", text)
    bridge._on_bot_utterance(BotUtterance(session_id=session.session_id, at=0.0, text=text))

    # 0.2s of 16kHz PCM16 (content doesn't matter -- this checks framing).
    frame = AudioFrame(pcm=b"\x00\x00" * 3200, sample_rate=16000, channels=1)
    await bridge._send_frame(frame)

    media_msgs = [m for m in ws.sent_events if m["event"] == "media"]
    mark_msgs = [m for m in ws.sent_events if m["event"] == "mark"]

    # 3200 samples @16kHz -> 1600 @8kHz -> 1600 mu-law bytes -> 10 chunks of 160.
    assert len(media_msgs) == 10
    assert all(m["streamSid"] == _STREAM_SID for m in media_msgs)
    first_payload = base64.b64decode(media_msgs[0]["media"]["payload"])
    assert len(first_payload) == 160

    assert len(mark_msgs) == 1
    assert mark_msgs[0]["streamSid"] == _STREAM_SID
    mark_name = mark_msgs[0]["mark"]["name"]
    assert mark_name in bridge._pending_marks
    assert bridge._pending_marks[mark_name].utterance_id == "utt-1"
    assert bridge._pending_marks[mark_name].bytes_forwarded == len(frame.pcm)


async def test_mark_ack_drives_the_real_ledger_to_an_exact_played_span() -> None:
    """The selling point end to end: a Twilio mark ack, not a generation-speed
    guess, is what tells the *real* PlaybackLedger how much was heard."""
    session = _session()
    runtime = FakeRuntime(session)
    ws = FakeWebSocket()
    bridge = _TwilioBridge(ws, runtime)
    bridge._stream_sid = _STREAM_SID

    text = "Hello there caller, how can I help you today"
    runtime.ledger.utterance_started("utt-1", text)
    bridge._on_bot_utterance(BotUtterance(session_id=session.session_id, at=0.0, text=text))

    frame = AudioFrame(pcm=b"\x00\x00" * 3200, sample_rate=16000, channels=1)  # 0.2s
    await bridge._send_frame(frame)
    (mark_name,) = bridge._pending_marks.keys()

    bridge._handle_mark_ack(mark_name)

    assert bridge._pending_marks == {}
    played = [e for e in session.history if isinstance(e, BotSpeechPlayed)]
    assert len(played) == 1
    # 6400 bytes / (2 bytes/sample * 16000 Hz) = 0.2s genuinely confirmed
    # played; at the documented 15 chars/sec fallback that is round(3) chars.
    assert played[0].text == text[:3]


async def test_mark_ack_for_an_unknown_name_is_a_silent_no_op() -> None:
    session = _session()
    runtime = FakeRuntime(session)
    bridge = _TwilioBridge(FakeWebSocket(), runtime)

    bridge._handle_mark_ack("nonexistent-mark")  # must not raise

    assert session.history == []


async def test_no_ledger_on_the_runtime_still_delivers_audio_without_marks() -> None:
    """S2S-style runtimes may have no interruption ledger at all (their vendor
    session owns interruption handling itself). Audio must still reach
    Twilio; there is simply no utterance id to hang a mark on, so none is
    sent -- and an ack for a name that was never sent is a silent no-op."""
    session = _session()
    runtime = FakeRuntime(session)
    del runtime.ledger  # simulate a RuntimeAdapter with no `ledger` attribute
    ws = FakeWebSocket()
    bridge = _TwilioBridge(ws, runtime)
    bridge._stream_sid = _STREAM_SID
    bridge._on_bot_utterance(
        BotUtterance(session_id=session.session_id, at=0.0, text="hi there")
    )

    frame = AudioFrame(pcm=b"\x00\x00" * 320, sample_rate=16000, channels=1)
    await bridge._send_frame(frame)

    assert [m for m in ws.sent_events if m["event"] == "media"]
    assert bridge._pending_marks == {}  # no utterance id -> nothing to track

    bridge._handle_mark_ack("whatever")  # must not raise


# ---------------------------------------------------------------------------
# barge-in: Interruption -> clear + stale-generation frames dropped
# ---------------------------------------------------------------------------


async def test_interruption_bumps_generation_and_clears_the_twilio_buffer() -> None:
    session = _session()
    runtime = FakeRuntime(session)
    ws = FakeWebSocket()
    bridge = _TwilioBridge(ws, runtime)
    bridge._stream_sid = _STREAM_SID

    await bridge._on_interruption()

    assert bridge._generation == 1
    clear_msgs = [m for m in ws.sent_events if m["event"] == "clear"]
    assert clear_msgs == [{"event": "clear", "streamSid": _STREAM_SID}]


async def test_pump_drops_audio_queued_before_a_barge_in() -> None:
    """Audio for an interrupted utterance that was already queued locally
    (but not yet sent) must never reach Twilio after the `clear` -- doing so
    would put the caller right back into hearing the bot talk over them."""
    session = _session()
    runtime = FakeRuntime(session)
    ws = FakeWebSocket()
    bridge = _TwilioBridge(ws, runtime)
    bridge._stream_sid = _STREAM_SID

    stale = AudioFrame(pcm=b"\x01\x00" * 160, sample_rate=16000, channels=1)
    bridge._on_bot_audio(stale)  # queued under generation 0

    await bridge._on_interruption()  # bumps to generation 1, sends `clear`

    fresh = AudioFrame(pcm=b"\x02\x00" * 160, sample_rate=16000, channels=1)
    bridge._on_bot_audio(fresh)  # queued under generation 1
    bridge._outbound.put_nowait(None)  # tell the pump to stop after draining

    await bridge._pump_outbound()

    media_msgs = [m for m in ws.sent_events if m["event"] == "media"]
    # Exactly one frame's worth of media should have gone out: the one
    # queued *after* the interruption. The stale one was silently dropped.
    assert len(media_msgs) == 1
