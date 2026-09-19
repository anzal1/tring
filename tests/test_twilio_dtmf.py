"""Tests for the Twilio Media Streams DTMF path (``_TwilioBridge._handle_dtmf``).

Same house pattern as ``tests/test_twilio_transport.py``: no network, no
``websockets`` package, ``_TwilioBridge`` driven end to end with a scripted
``FakeWebSocket`` double satisfying its ``_WebSocketLike`` Protocol. Kept in
its own file (rather than added to ``test_twilio_transport.py``) because
DTMF is an additive, independent feature -- see ``transports/twilio.py``'s
module docstring and ``events.py``'s ``DtmfReceived`` -- not a change to the
audio-framing behavior that file already covers exhaustively.

Message shape verified live against
https://www.twilio.com/docs/voice/media-streams/websocket-messages,
2026-09-19: ``{"event": "dtmf", "streamSid": ..., "sequenceNumber": ...,
"dtmf": {"track": "inbound_track", "digit": "1"}}``.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from tring.agent import AgentSpec
from tring.events import DtmfReceived
from tring.runtimes.base import AudioFrame, RuntimeAdapter, RuntimeCapabilities
from tring.session import CallSession
from tring.transports.twilio import _TwilioBridge

_STREAM_SID = "MZ00000000000000000000000000000000"


class FakeWebSocket:
    """Minimal scripted double for a Twilio Media Streams connection.

    A smaller, purpose-built copy of ``test_twilio_transport.py``'s
    ``FakeWebSocket`` (push a fixed script, replay it, record what got
    sent) -- this file only drives the inbound message-dispatch path, never
    outbound audio, so it does not need that file's queue-based version.
    """

    def __init__(self) -> None:
        self._inbound: list[str] = []
        self.sent: list[str] = []

    def push(self, message: dict[str, Any]) -> None:
        self._inbound.append(json.dumps(message))

    async def send(self, message: str | bytes) -> None:
        assert isinstance(message, str)
        self.sent.append(message)

    async def close(self) -> None:
        pass

    def __aiter__(self) -> AsyncIterator[str | bytes]:
        return self._replay()

    async def _replay(self) -> AsyncIterator[str | bytes]:
        for message in self._inbound:
            yield message


class FakeRuntime(RuntimeAdapter):
    """A bare ``RuntimeAdapter`` recording nothing but lifecycle calls --
    these tests only care what lands on ``session.history``."""

    def __init__(self, session: CallSession) -> None:
        super().__init__(session)
        self.stop_reason: str | None = None

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            live_transcripts=True,
            mid_call_tool_calls=False,
            barge_in=False,
            exact_usage_reporting=True,
        )

    async def start(self) -> None:
        pass

    async def push_audio(self, frame: AudioFrame) -> None:
        pass

    async def stop(self, reason: str = "completed") -> None:
        self.stop_reason = reason


def _session() -> CallSession:
    return CallSession(AgentSpec(name="dtmf-test", persona="You are a test agent."))


def _start_message() -> dict[str, Any]:
    return {
        "event": "start",
        "sequenceNumber": "1",
        "streamSid": _STREAM_SID,
        "start": {
            "streamSid": _STREAM_SID,
            "accountSid": "AC_test",
            "callSid": "CA_test",
            "tracks": ["inbound", "outbound"],
            "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
        },
    }


def _dtmf_message(digit: str, sequence: str = "2") -> dict[str, Any]:
    return {
        "event": "dtmf",
        "streamSid": _STREAM_SID,
        "sequenceNumber": sequence,
        "dtmf": {"track": "inbound_track", "digit": digit},
    }


async def test_a_single_dtmf_digit_emits_dtmf_received() -> None:
    session = _session()
    runtime = FakeRuntime(session)
    ws = FakeWebSocket()
    bridge = _TwilioBridge(ws, runtime)

    ws.push(_start_message())
    ws.push(_dtmf_message("5"))
    ws.push({"event": "stop", "streamSid": _STREAM_SID, "stop": {}})

    await bridge.run()

    digits = [e for e in session.history if isinstance(e, DtmfReceived)]
    assert [d.digit for d in digits] == ["5"]
    assert runtime.stop_reason == "twilio_stream_ended"


async def test_multiple_dtmf_digits_emit_in_the_order_received() -> None:
    session = _session()
    runtime = FakeRuntime(session)
    ws = FakeWebSocket()
    bridge = _TwilioBridge(ws, runtime)

    ws.push(_start_message())
    for digit in ["1", "4", "#", "*"]:
        ws.push(_dtmf_message(digit))
    ws.push({"event": "stop", "streamSid": _STREAM_SID, "stop": {}})

    await bridge.run()

    digits = [e.digit for e in session.history if isinstance(e, DtmfReceived)]
    assert digits == ["1", "4", "#", "*"]


def test_dtmf_message_with_no_digit_field_is_a_silent_no_op(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _session()
    runtime = FakeRuntime(session)
    bridge = _TwilioBridge(FakeWebSocket(), runtime)

    with caplog.at_level("DEBUG"):
        bridge._handle_dtmf({"track": "inbound_track"})  # missing "digit"

    assert [e for e in session.history if isinstance(e, DtmfReceived)] == []


async def test_dtmf_with_an_empty_digit_string_is_a_silent_no_op() -> None:
    session = _session()
    runtime = FakeRuntime(session)
    bridge = _TwilioBridge(FakeWebSocket(), runtime)

    bridge._handle_dtmf({"track": "inbound_track", "digit": ""})

    assert session.history == []


async def test_connected_event_alongside_dtmf_is_still_a_pure_no_op() -> None:
    """Regression guard: splitting the old combined ``("connected", "dtmf")``
    branch (see transports/twilio.py) must not change "connected" handling."""
    session = _session()
    runtime = FakeRuntime(session)
    ws = FakeWebSocket()
    bridge = _TwilioBridge(ws, runtime)

    ws.push({"event": "connected", "protocol": "Call", "version": "1.0.0"})
    ws.push(_start_message())
    ws.push(_dtmf_message("9"))
    ws.push({"event": "stop", "streamSid": _STREAM_SID, "stop": {}})

    await bridge.run()

    digits = [e.digit for e in session.history if isinstance(e, DtmfReceived)]
    assert digits == ["9"]
