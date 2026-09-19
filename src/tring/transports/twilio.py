"""Twilio Media Streams transport: bidirectional telephone audio over a
WebSocket, driven by TwiML's ``<Connect><Stream>`` verb.

Message schema (event names, nesting, field names) verified live against
https://www.twilio.com/docs/voice/media-streams/websocket-messages on
2026-09-19: ``connected`` -> ``start`` -> many ``media`` -> ``stop``, with
``mark`` and ``clear`` as the two bidirectional control messages and
``dtmf`` (bidirectional Streams only) forwarded as the ``DtmfReceived``
session event -- see :meth:`_TwilioBridge._handle_dtmf`. Audio is
always G.711 mu-law at 8 kHz mono, base64 inside ``media.payload`` --
:mod:`tring.transports.mulaw` is the codec, kept in its own module because
it has nothing Twilio-specific in it (a SIP/RTP bridge needs the exact same
codec; see ``docs/TELEPHONY.md``).

**The mark trick, and why it is worth the plumbing below.** Every other
transport in this stack learns "has the caller heard this yet" by asking
:class:`~tring.runtimes.cascade.CascadeRuntime` to *guess*: convert audio
duration to characters at an assumed speech rate, on the assumption that a
frame handed to ``on_bot_audio`` is a frame already at the caller's ear
(see ``SPOKEN_CHARS_PER_SECOND`` in ``runtimes/cascade.py``). That
assumption is false on a phone call -- there is a real jitter buffer
between "we sent this media frame" and "the caller's ear moved" -- and it
is false in a way that matters most exactly when it is checked most: at
the instant of a barge-in.

Twilio's ``mark`` message closes that gap for real. A ``mark`` sent after
a ``media`` message is acknowledged by Twilio, mark for mark, only once
the audio *before* it has actually finished playing
(https://www.twilio.com/docs/voice/media-streams/websocket-messages#mark-message).
That is a genuine playout clock, not an estimate -- so this transport uses
it to drive :class:`~tring.primitives.interruption.PlaybackLedger` itself
(``runtime.ledger.mark_played``, the exact integration point
``docs/EXTENDING.md`` describes), upgrading the heard/unheard split at a
barge-in from "assumed" to "confirmed by the carrier." See
:meth:`_TwilioBridge._handle_mark_ack` for the one honestly-documented wart
in this pipeline: the ledger does not yet expose *which* utterance is
current through public API, so that one lookup reaches into
``PlaybackLedger``'s internal state, defensively.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from tring.events import BotUtterance, DtmfReceived, Interruption
from tring.runtimes.base import AudioFrame
from tring.session import CallSession
from tring.transports.mulaw import linear16_to_mulaw, mulaw_to_linear16, resample_linear16

if TYPE_CHECKING:
    from tring.runtimes.base import RuntimeAdapter

logger = logging.getLogger(__name__)

# Twilio Media Streams audio format -- verified live against
# https://www.twilio.com/docs/voice/media-streams/websocket-messages,
# 2026-09-19: inbound and outbound audio is always mono G.711 mu-law at
# 8 kHz, base64-encoded inside `media.payload`. There is no negotiation and
# no format field to opt into anything else on the outbound side.
_TWILIO_SAMPLE_RATE = 8000

# Canonical stack wire format -- see runtimes/base.py's AudioFrame docstring.
_WIRE_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # 16-bit linear PCM

# 20ms of 8kHz mu-law audio (8000 * 0.02 * 1 byte/sample). Twilio's own
# quickstart streamers chunk outbound media to this size; sending audio in
# real-time-sized pieces rather than one giant payload per TTS chunk keeps
# the `mark` messages below granular enough to be useful for barge-in
# timing instead of only confirming whole sentences at a time.
_OUTBOUND_CHUNK_BYTES = 160

# The same speech-rate approximation CascadeRuntime documents on
# `SPOKEN_CHARS_PER_SECOND` (runtimes/cascade.py) -- not imported from
# there: ARCHITECTURE.md's layering rule is that a transport never imports
# a concrete runtime, so this is a second, independent copy of the same
# documented number, not a shared one. What this transport upgrades is not
# the rate model, which stays an approximation either way; it is the
# *input* to it. Cascade's own copy converts audio the runtime has
# *generated* into an estimated played-position. This one converts audio
# Twilio has *confirmed played* (see _handle_mark_ack) into the same
# played-position -- trading "assumed instant playback" for "playback
# confirmed by the carrier," which is the entire value of this transport.
_FALLBACK_CHARS_PER_SECOND = 15.0


class _WebSocketLike(Protocol):
    """The three things this module needs from a websocket connection.

    Mirrors ``providers/cloud/s2s_extra.py``'s ``_WebSocketLike`` exactly
    (small local copy, not imported: that Protocol describes a *client*
    seam for outbound vendor connections, this one a *server* connection
    Twilio opens to us, and the two call sites have no reason to share
    code just because the shape happens to match today). Narrowing to a
    Protocol is what makes :class:`_TwilioBridge` testable without the
    ``websockets`` package installed: ``tests/test_twilio_transport.py``
    drives it with a scripted fake implementing exactly these three
    members.
    """

    async def send(self, message: str | bytes) -> None: ...

    async def close(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[str | bytes]: ...


@runtime_checkable
class _PlaybackLedgerLike(Protocol):
    """The one method this module calls on a runtime's ``ledger``.

    A ``Protocol`` rather than importing
    ``primitives.interruption.PlaybackLedger`` directly so that a runtime
    without an interruption ledger at all (a bare ``RuntimeAdapter``, most
    S2S runtimes -- their vendor session owns interruption handling itself)
    is simply not this shape, and :func:`_current_utterance_id` /
    :meth:`_TwilioBridge._handle_mark_ack` degrade to "no exact-playout
    upgrade" instead of raising. ``@runtime_checkable`` makes that an
    ``isinstance`` check mypy can narrow on, rather than a bare ``hasattr``
    it cannot.
    """

    def mark_played(self, utterance_id: str, upto_chars: int) -> None: ...


@dataclass(frozen=True)
class _PendingMark:
    """What a ``mark`` name we sent to Twilio corresponds to, so that when
    Twilio acks it we know what to report to the ledger.

    ``bytes_forwarded`` is the cumulative count of wire-format (16 kHz, 2
    bytes/sample) PCM bytes forwarded for the *current utterance* as of the
    moment this mark was sent -- reset to 0 whenever a new ``BotUtterance``
    event is seen (see :meth:`_TwilioBridge._watch_session_events`). Tracking
    per-utterance rather than call-wide is what lets
    :meth:`_TwilioBridge._handle_mark_ack` convert "bytes played" into "a
    position within this utterance's text" at all.
    """

    utterance_id: str
    utterance_text: str
    bytes_forwarded: int


def _current_utterance_id(ledger: object | None) -> str | None:
    """Best-effort read of which utterance id a ``PlaybackLedger`` currently
    considers "in flight."

    This is the one honest wart in this module. ``PlaybackLedger``
    (``primitives/interruption.py``) does not expose the current utterance
    id through public API -- it is generated inside ``CascadeRuntime`` with
    ``uuid.uuid4().hex`` and consumed only by the ledger itself, because
    nothing before this transport ever needed a transport to reproduce it.
    Exact telephony reconciliation needs it anyway: ``ledger.mark_played``
    requires the *exact* id the ledger is already tracking, or it silently
    no-ops (by design -- see that method's docstring on stale reports).

    So this reaches into ``PlaybackLedger``'s private ``_current`` field,
    defensively: every step is a ``getattr`` with a ``None`` fallback, so a
    future refactor of ``interruption.py`` that changes this shape disables
    the exact-playout upgrade (this transport falls back to whatever
    estimate the runtime's own ledger calls already produce) rather than
    raising mid-call. The real fix is a two-line addition upstream -- a
    public ``PlaybackLedger.current_utterance_id`` property -- tracked as a
    roadmap item in ``docs/TELEPHONY.md`` rather than done here, because
    this module's brief is the transport, not ``primitives/interruption.py``.
    """
    current = getattr(ledger, "_current", None)
    if current is None:
        return None
    utterance_id = getattr(current, "utterance_id", None)
    return utterance_id if isinstance(utterance_id, str) else None


class _TwilioBridge:
    """Bridges one Twilio Media Streams connection to one ``RuntimeAdapter``.

    Kept independent of the ``websockets`` package (see ``_WebSocketLike``)
    so the entire message-handling contract -- decode inbound audio, encode
    and chunk outbound audio, mark bookkeeping, barge-in ``clear`` -- is
    testable with a scripted fake and no socket, matching this codebase's
    house pattern for anything that talks to a wire protocol.

    **Why outbound audio goes through a queue instead of being sent
    directly from ``on_bot_audio``.** ``RuntimeAdapter._emit_audio`` (see
    ``runtimes/cascade.py``) calls ``on_bot_audio(frame)`` as a plain
    synchronous call -- it is never awaited, so an ``async def`` callback
    assigned there would hand back an un-awaited coroutine and silently
    drop every frame (a real bug present in this repo's own
    ``transports/websocket.py`` reference transport today). ``_on_bot_audio``
    is therefore synchronous and only does ``Queue.put_nowait``; a
    dedicated ``_pump_outbound`` task does the actual (necessarily async)
    encode-and-send.

    **Why queue items carry a generation number.** A genuine barge-in
    (``Interruption`` event) means the runtime has already stopped
    generating audio for the interrupted utterance -- but this bridge may
    still have some of that utterance's audio sitting in ``_outbound``,
    not yet sent. Sending it anyway after telling Twilio to ``clear`` its
    buffer would put the caller right back into hearing the bot talk over
    them. Tagging every queued item with the generation counter in effect
    when it was queued, and bumping the counter on every ``Interruption``,
    makes the pump loop drop stale items on its own without needing to
    reach into or drain the queue from the event handler -- which matters
    because draining it from another task while the pump loop might be
    mid-``await``-on-`.get()` on the very same queue is exactly the kind of
    thing that is easy to get subtly wrong under concurrency.
    """

    def __init__(self, websocket: _WebSocketLike, runtime: RuntimeAdapter) -> None:
        self._ws = websocket
        self._runtime = runtime
        self._stream_sid: str | None = None

        self._outbound: asyncio.Queue[tuple[int, AudioFrame] | None] = asyncio.Queue()
        self._generation = 0

        self._mark_seq = count(1)
        self._pending_marks: dict[str, _PendingMark] = {}

        # Bookkeeping for the utterance currently being spoken, refreshed on
        # every BotUtterance event (see _on_bot_utterance).
        self._utterance_id: str | None = None
        self._utterance_text = ""
        self._utterance_bytes_forwarded = 0

    # ------------------------------------------------------------- lifecycle

    async def run(self) -> None:
        """Drive one call end to end: bind audio, read Twilio's event stream."""
        self._runtime.on_bot_audio = self._on_bot_audio

        watcher = asyncio.create_task(self._watch_session_events())
        pump = asyncio.create_task(self._pump_outbound())
        # Give the watcher a turn to actually subscribe (an async generator
        # only registers its queue once first iterated) before start() can
        # emit anything -- otherwise a greeting's SessionStarted/BotUtterance
        # could fire before anyone is listening for it. This narrows the
        # race described in interruption.py's own docstring about frames
        # emitted before an utterance is armed; it does not close it
        # entirely, which is why _handle_mark_ack degrades gracefully
        # rather than assuming it always has an utterance id in hand.
        await asyncio.sleep(0)

        await self._runtime.start()

        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    logger.debug("twilio: dropping unexpected binary frame")
                    continue
                if await self._handle_message(raw):
                    break  # Twilio sent `stop`: the call is over
        finally:
            await self._runtime.stop(reason="twilio_stream_ended")
            self._outbound.put_nowait(None)
            watcher.cancel()
            for task in (watcher, pump):
                (result,) = await asyncio.gather(task, return_exceptions=True)
                # CancelledError is a BaseException, not an Exception, so the
                # expected shutdown path (watcher.cancel() above) never hits
                # this branch. What lands here is genuinely unexpected --
                # e.g. a send racing a closed connection -- and the call is
                # ending either way, so it is worth a debug breadcrumb, not
                # an alarm.
                if isinstance(result, Exception):
                    logger.debug("twilio: %s ended with %r", task.get_name(), result)

    # -------------------------------------------------------- inbound: Twilio -> runtime

    async def _handle_message(self, raw: str) -> bool:
        """Handle one JSON frame from Twilio. Returns True iff it was `stop`."""
        try:
            msg: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("twilio: dropping non-JSON frame")
            return False

        event = msg.get("event")
        if event == "start":
            self._handle_start(msg.get("start", {}))
        elif event == "media":
            await self._handle_inbound_media(msg.get("media", {}))
        elif event == "mark":
            name = msg.get("mark", {}).get("name")
            if isinstance(name, str):
                self._handle_mark_ack(name)
        elif event == "stop":
            return True
        elif event == "dtmf":
            self._handle_dtmf(msg.get("dtmf", {}))
        elif event == "connected":
            pass  # nothing this transport needs to act on
        else:
            logger.debug("twilio: unhandled event %r", event)
        return False

    def _handle_start(self, start: dict[str, Any]) -> None:
        stream_sid = start.get("streamSid")
        if isinstance(stream_sid, str):
            self._stream_sid = stream_sid
        media_format = start.get("mediaFormat") or {}
        if media_format and (
            media_format.get("encoding") != "audio/x-mulaw"
            or media_format.get("sampleRate") != _TWILIO_SAMPLE_RATE
        ):
            # Not fatal: Twilio has never shipped anything else here, but a
            # transport that hard-fails on a vendor's future format change
            # is worse than one that logs and tries anyway.
            logger.warning("twilio: unexpected mediaFormat %r", media_format)

    async def _handle_inbound_media(self, media: dict[str, Any]) -> None:
        payload = media.get("payload")
        if not isinstance(payload, str) or not payload:
            return
        mulaw_bytes = base64.b64decode(payload)
        pcm_8k = mulaw_to_linear16(mulaw_bytes)
        pcm_16k = resample_linear16(pcm_8k, _TWILIO_SAMPLE_RATE, _WIRE_SAMPLE_RATE)
        if not pcm_16k:
            return
        await self._runtime.push_audio(
            AudioFrame(pcm=pcm_16k, sample_rate=_WIRE_SAMPLE_RATE, channels=1)
        )

    def _handle_dtmf(self, dtmf: dict[str, Any]) -> None:
        """A caller pressed a keypad digit mid-call.

        Message shape verified live against
        https://www.twilio.com/docs/voice/media-streams/websocket-messages
        on 2026-09-19: ``{"event": "dtmf", "streamSid": ..., "sequenceNumber":
        ..., "dtmf": {"track": "inbound_track", "digit": "1"}}``. DTMF
        messages are only sent on bidirectional Streams -- i.e. exactly the
        ``<Connect><Stream>`` verb this transport already requires (see the
        module docstring), so no capability check is needed here. ``track``
        is documented as always ``"inbound_track"`` (a caller's keypad tone,
        never the bot's own audio) and carries nothing this transport needs
        to branch on -- only ``digit`` is forwarded, as the new
        ``DtmfReceived`` session event (``events.py``) tool handlers and
        analytics consumers subscribe to instead of parsing Twilio's wire
        format themselves.
        """
        digit = dtmf.get("digit")
        if not isinstance(digit, str) or not digit:
            logger.debug("twilio: dropping dtmf message with no digit: %r", dtmf)
            return
        self._runtime.session.emit(
            DtmfReceived(
                session_id=self._runtime.session.session_id,
                at=self._runtime.session.elapsed,
                digit=digit,
            )
        )

    def _handle_mark_ack(self, name: str) -> None:
        """Twilio confirms ``name``'s audio has finished playing: the exact
        moment ``docs/EXTENDING.md``'s "transport drives the ledger" pattern
        exists for. See the module docstring and :func:`_current_utterance_id`
        for what this does and does not guarantee.
        """
        pending = self._pending_marks.pop(name, None)
        if pending is None:
            return
        ledger = getattr(self._runtime, "ledger", None)
        if not isinstance(ledger, _PlaybackLedgerLike):
            return
        played_seconds = pending.bytes_forwarded / (_BYTES_PER_SAMPLE * _WIRE_SAMPLE_RATE)
        upto_chars = min(
            len(pending.utterance_text),
            round(played_seconds * _FALLBACK_CHARS_PER_SECOND),
        )
        ledger.mark_played(pending.utterance_id, upto_chars)

    # -------------------------------------------------------- outbound: runtime -> Twilio

    def _on_bot_audio(self, frame: AudioFrame) -> None:
        """``runtime.on_bot_audio`` callback. Synchronous -- see the class
        docstring for why an ``async def`` here would be a real bug."""
        self._outbound.put_nowait((self._generation, frame))

    async def _pump_outbound(self) -> None:
        while True:
            item = await self._outbound.get()
            if item is None:
                return
            generation, frame = item
            if generation != self._generation:
                continue  # superseded by a barge-in; see the class docstring
            await self._send_frame(frame)

    async def _send_frame(self, frame: AudioFrame) -> None:
        if self._stream_sid is None:
            return  # Twilio hasn't sent `start` yet; nowhere to address audio
        pcm_8k = resample_linear16(frame.pcm, frame.sample_rate, _TWILIO_SAMPLE_RATE)
        mulaw = linear16_to_mulaw(pcm_8k)
        self._utterance_bytes_forwarded += len(frame.pcm)
        for offset in range(0, len(mulaw), _OUTBOUND_CHUNK_BYTES):
            chunk = mulaw[offset : offset + _OUTBOUND_CHUNK_BYTES]
            await self._ws.send(
                json.dumps(
                    {
                        "event": "media",
                        "streamSid": self._stream_sid,
                        "media": {"payload": base64.b64encode(chunk).decode("ascii")},
                    }
                )
            )
        await self._send_mark()

    async def _send_mark(self) -> None:
        if self._stream_sid is None or self._utterance_id is None:
            return
        name = str(next(self._mark_seq))
        self._pending_marks[name] = _PendingMark(
            utterance_id=self._utterance_id,
            utterance_text=self._utterance_text,
            bytes_forwarded=self._utterance_bytes_forwarded,
        )
        await self._ws.send(
            json.dumps({"event": "mark", "streamSid": self._stream_sid, "mark": {"name": name}})
        )

    async def _send_clear(self) -> None:
        if self._stream_sid is None:
            return
        # Empties whatever Twilio has already buffered and flushes any
        # outstanding marks for it back to us immediately (see
        # https://www.twilio.com/docs/voice/media-streams/websocket-messages#clear-message).
        # Those flushed marks resolve against `pending.utterance_id`, which
        # by now is stale -- but PlaybackLedger.mark_played silently ignores
        # a report for an utterance it no longer considers current, so this
        # is a correct no-op, not a bug that needs guarding against here.
        await self._ws.send(json.dumps({"event": "clear", "streamSid": self._stream_sid}))

    # -------------------------------------------------------------- session events

    async def _watch_session_events(self) -> None:
        async for event in self._runtime.session.subscribe():
            if isinstance(event, BotUtterance):
                self._on_bot_utterance(event)
            elif isinstance(event, Interruption):
                await self._on_interruption()

    def _on_bot_utterance(self, event: BotUtterance) -> None:
        """A new bot utterance has started: reset per-utterance bookkeeping.

        Split out of :meth:`_watch_session_events` (rather than inlined)
        purely so tests can drive it directly without needing to run the
        event-subscription loop as a background task.
        """
        self._utterance_id = _current_utterance_id(getattr(self._runtime, "ledger", None))
        self._utterance_text = event.text
        self._utterance_bytes_forwarded = 0

    async def _on_interruption(self) -> None:
        """A genuine barge-in: supersede whatever's still queued, tell Twilio."""
        self._generation += 1
        await self._send_clear()


async def serve_twilio(
    runtime_factory: Callable[[CallSession], RuntimeAdapter],
    host: str = "0.0.0.0",
    port: int = 8080,
    path: str = "/twilio",
) -> None:
    """Serve Twilio's ``<Connect><Stream>`` WebSocket protocol.

    Point a TwiML ``<Stream url="wss://your-host/twilio">`` (see
    ``docs/TELEPHONY.md`` for the full webhook + TwiML setup) at this
    server. For each connection, a fresh ``CallSession``-shaped session is
    handed to ``runtime_factory`` -- exactly the ``transports/websocket.py``
    reference transport's convention: the factory is free to build and
    return its own properly-constructed ``CallSession`` (wrapping an
    ``AgentSpec``) instead, in which case everything here follows
    ``runtime.session``, never the placeholder passed in.

    Args:
        runtime_factory: builds one ``RuntimeAdapter`` per call.
        host: interface to bind to.
        port: port to listen on.
        path: the URL path Twilio's ``<Stream>`` should be pointed at;
            mismatches are logged, not rejected, since Twilio's own request
            path can carry a query string this transport has no need to
            parse strictly.

    Raises:
        ImportError: if the ``websockets`` package is not installed.
    """
    try:
        import websockets
    except ImportError as exc:
        raise ImportError(
            "twilio transport needs the 'websockets' package. "
            "Install tring[transports] to enable it."
        ) from exc

    async def handle_connection(websocket: Any) -> None:
        request = getattr(websocket, "request", None)
        request_path = getattr(request, "path", None)
        if request_path is not None and not request_path.startswith(path):
            logger.warning("twilio: connection on %r, expected prefix %r", request_path, path)

        session = CallSession.__new__(CallSession)  # placeholder; see docstring
        runtime = runtime_factory(session)
        bridge = _TwilioBridge(websocket, runtime)

        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            await bridge.run()
        with contextlib.suppress(websockets.exceptions.ConnectionClosed):
            await websocket.close()

    async with websockets.serve(handle_connection, host, port):
        logger.info("Twilio Media Streams server listening on ws://%s:%d%s", host, port, path)
        await asyncio.Future()


__all__ = ["serve_twilio"]
