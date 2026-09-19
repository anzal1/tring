"""Tring Studio's back end: static files, a JSON config API, and a live session.

The wire contract is ``docs/STUDIO_PROTOCOL.md``; the browser app in
``static/`` depends on every field name in it. This module is the server half.

Why the HTTP layer is hand-rolled
---------------------------------

The obvious design is the one the protocol doc sketches: run ``websockets``
and serve the HTTP endpoints from its ``process_request`` hook, so one port
carries everything and there is no new dependency. That works for ``GET`` and
breaks for ``POST``. ``process_request`` is invoked as soon as the request
*headers* have been parsed, before any body has been read, and the sans-I/O
handshake machinery underneath it offers no supported way to pull the body out
of the connection at that point. Driving the studio's "save agent" button
against such a server returns ``ERR_EMPTY_RESPONSE`` in the browser, because
the response goes out while the request body is still unread in the socket.

So the port is owned by :func:`asyncio.start_server` instead. It speaks just
enough HTTP/1.1 for a single-page app (request line, headers, optional
``Content-Length`` body, one response, ``Connection: close``) and hands any
connection carrying a WebSocket upgrade to ``websockets``' **Sans-I/O** layer,
:class:`websockets.server.ServerProtocol`, which exists for exactly this: it
owns the handshake, framing, masking, ping/pong and the closing handshake,
while this module owns the reads and writes. No private API, no second
listening socket, no new dependency.

References:

* Sans-I/O server API —
  https://websockets.readthedocs.io/en/stable/reference/sansio/server.html
* Sans-I/O integration pattern (receive, send, then handle events) —
  https://websockets.readthedocs.io/en/stable/howto/sansio.html

Why keep-alive is not implemented: every response closes its connection. A
studio session is one long-lived WebSocket plus a handful of asset fetches on a
loopback address, so connection reuse buys nothing and a half-correct
persistent-connection implementation is a genuine source of hangs.

The session bridge
------------------

One live :class:`~tring.session.CallSession` per WebSocket connection. The
browser has no microphone path, so the STT slot is forced to ``text_input`` —
the provider that reads UTF-8 text out of an :class:`AudioFrame`'s ``pcm``
field — and typed turns are pushed in as ordinary audio frames. Everything
above that (turn loop, envelope parsing, tool choreography, playback ledger,
cost metering) is the unmodified production runtime; the studio subscribes to
its event stream like any other consumer and forwards events verbatim.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from html import escape
from http import HTTPStatus
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote

import yaml
from pydantic import ValidationError

from tring import __version__
from tring.agent import AgentSpec, ProviderSelection, RuntimeConfig
from tring.cost.meter import CostMeter
from tring.cost.rates import DEFAULT_RATES
from tring.events import SessionError
from tring.providers import registry
from tring.runtimes.base import AudioFrame

# Private, and imported on purpose. `_options_for` encodes the rule that splits
# one ProviderSelection.options dict across the stt/llm/tts slots. The studio
# probes the configured TTS provider by constructing it exactly as the runtime
# would, so it must split options exactly as the runtime does; a local copy of
# the rule would drift and start probing a provider with the wrong arguments.
from tring.runtimes.cascade import CascadeRuntime, _options_for
from tring.session import CallSession
from tring.studio.silent_tts import SILENT_TTS_NAME

if TYPE_CHECKING:  # pragma: no cover - typing only
    from asyncio import StreamReader, StreamWriter

    from websockets.server import ServerProtocol

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8900
DEFAULT_AGENT_PATH = Path("agent.yaml")

#: Where the built single-page app lives. The frontend is a React bundle:
#: ``index.html`` plus a hashed ``assets/`` directory, all served from here.
DEFAULT_STATIC_DIR = Path(__file__).resolve().parent / "static"

#: The STT slot the studio always forces. See the module docstring.
TEXT_INPUT_STT_NAME = "text_input"

# Request limits. Generous for a local tool, finite so a stray client cannot
# make the server allocate without bound.
_MAX_HEADER_BYTES = 64 * 1024
_MAX_BODY_BYTES = 4 * 1024 * 1024
_WS_READ_BYTES = 64 * 1024

#: Per the Sans-I/O how-to: once the closing handshake is done, do not wait
#: forever for the peer to close the TCP connection.
_WS_CLOSE_TIMEOUT = 10.0

#: Minimum gap between ``audio_progress`` messages. A real TTS provider emits
#: 20 ms frames (50 per second per utterance); the UI only needs to know that
#: speech is happening, so coalescing keeps the socket quiet without losing the
#: signal. The byte count itself is always cumulative and exact.
_AUDIO_PROGRESS_INTERVAL = 0.1

#: How long :meth:`StudioServer.close` lets live connections finish their own
#: teardown before cancelling them.
_SHUTDOWN_GRACE = 5.0

# Explicit rather than `mimetypes.guess_type`, whose mapping for `.js`, `.mjs`
# and `.map` depends on the host's mime database (and on Windows, the registry).
# A dev server that serves JavaScript as `text/plain` on one machine and not
# another is a bug that costs an afternoon to find.
_CONTENT_TYPES: dict[str, str] = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".ico": "image/x-icon",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".webmanifest": "application/manifest+json",
    ".woff2": "font/woff2",
}

#: The spec served (and offered for editing) when ``--agent`` names a file that
#: does not exist yet. Local-first, matching ``examples/agent.yaml``: the studio
#: forces the STT slot to ``text_input`` anyway, and a TTS provider that cannot
#: be constructed falls back to ``studio_silent`` with the reason shown in the
#: UI, so this spec opens the studio on any machine.
DEMO_AGENT = AgentSpec(
    name="studio-demo",
    persona=(
        "You are a warm, concise voice assistant demonstrating the Tring "
        "studio. Answer in one or two spoken sentences. If you do not know "
        "something, say so plainly rather than guessing."
    ),
    greeting="Hi, you are talking to the Tring studio demo agent. What can I do for you?",
    runtime=RuntimeConfig(
        routing={
            "default": ProviderSelection(
                stt=TEXT_INPUT_STT_NAME,
                llm="ollama",
                tts="kokoro",
            )
        }
    ),
)


@dataclass(frozen=True)
class _Request:
    """One parsed HTTP/1.1 request head."""

    method: str
    path: str  # percent-decoded, query and fragment stripped
    headers: dict[str, str]  # lowercased names
    raw_head: bytes  # verbatim bytes, replayed into the WebSocket handshake


@dataclass(frozen=True)
class _Response:
    status: HTTPStatus
    body: bytes
    content_type: str


def _text_response(status: HTTPStatus, message: str) -> _Response:
    return _Response(status, message.encode("utf-8"), "text/plain; charset=utf-8")


def _json_response(payload: dict[str, Any]) -> _Response:
    """Every JSON endpoint answers 200, including validation failures.

    ``POST /api/agent`` reports a rejected spec as ``{"ok": false, "error": ...}``
    rather than as an HTTP error, because the error text is UI copy: the studio
    renders it next to the editor, and a 4xx would make ``fetch`` callers guess
    at whether the body is machine-readable.
    """
    return _Response(
        HTTPStatus.OK,
        json.dumps(payload).encode("utf-8"),
        "application/json; charset=utf-8",
    )


def _collapse(text: str) -> str:
    """Flatten a multi-line exception message into one UI-sized line."""
    return " ".join(str(text).split())


def _describe_validation_error(exc: ValidationError) -> str:
    """Render pydantic's error list as something an editor pane can show."""
    parts = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error["loc"]) or "(root)"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts) or str(exc)


def _require_websockets() -> None:
    """Fail fast, with the pip extra named, if the optional dep is missing.

    Checked when the server binds rather than when the first browser upgrades,
    so ``python -m tring.studio`` cannot come up looking healthy and then break
    the moment someone presses Start.
    """
    try:
        import websockets.server  # noqa: F401
    except ImportError as exc:  # pragma: no cover - needs the package absent
        raise ImportError(
            "the websockets package is required to run Tring Studio. "
            'Install it with:\n    pip install "tring[transports]"'
        ) from exc


# ---------------------------------------------------------------------------
# HTTP/1.1, the little of it a single-page app needs
# ---------------------------------------------------------------------------


async def _read_head(reader: StreamReader) -> _Request | None:
    """Read and parse the request head. ``None`` means "answer 400 and stop"."""
    try:
        raw = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ValueError):
        return None
    if len(raw) > _MAX_HEADER_BYTES:
        return None
    return _parse_head(raw)


def _parse_head(raw: bytes) -> _Request | None:
    try:
        text = raw.decode("latin-1")  # the encoding HTTP/1.1 defines for heads
    except UnicodeDecodeError:  # pragma: no cover - latin-1 decodes any byte
        return None

    lines = text.split("\r\n")
    request_line = lines[0].split(" ")
    if len(request_line) != 3:
        return None
    method, target, _version = request_line

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, separator, value = line.partition(":")
        if not separator:
            return None
        key = name.strip().lower()
        stripped = value.strip()
        # Repeated headers fold into a comma list, which is what `Connection:
        # keep-alive, Upgrade` arriving as two lines has to mean.
        headers[key] = f"{headers[key]}, {stripped}" if key in headers else stripped

    path = unquote(target.split("?", 1)[0].split("#", 1)[0])
    return _Request(method=method.upper(), path=path, headers=headers, raw_head=raw)


def _is_websocket_upgrade(request: _Request) -> bool:
    upgrade = request.headers.get("upgrade", "").lower()
    connection = request.headers.get("connection", "").lower()
    return "websocket" in upgrade and "upgrade" in connection


async def _read_body(reader: StreamReader, request: _Request) -> bytes | None:
    """Read a ``Content-Length`` body. ``None`` means the request is unusable.

    Chunked bodies are refused rather than mis-parsed. The only client of this
    API is the studio's own frontend, which sends a length; anything else
    arriving chunked is better off seeing a clear 400 than a truncated spec.
    """
    if "transfer-encoding" in request.headers:
        return None
    raw_length = request.headers.get("content-length")
    if raw_length is None:
        return b""
    try:
        length = int(raw_length)
    except ValueError:
        return None
    if length < 0 or length > _MAX_BODY_BYTES:
        return None
    try:
        return await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        return None


def _write_response(
    writer: StreamWriter, response: _Response, with_body: bool = True
) -> None:
    """Write one response. ``with_body=False`` serves a ``HEAD``.

    A HEAD response carries the headers the GET would have carried, including
    ``Content-Length``, and no body at all.
    """
    head = (
        f"HTTP/1.1 {response.status.value} {response.status.phrase}\r\n"
        f"Content-Type: {response.content_type}\r\n"
        f"Content-Length: {len(response.body)}\r\n"
        # A studio is edited while it runs: a cached bundle or a cached
        # /api/agent would show the developer yesterday's state.
        "Cache-Control: no-store\r\n"
        "Connection: close\r\n"
        "\r\n"
    ).encode("latin-1")
    writer.write(head + response.body if with_body else head)


# ---------------------------------------------------------------------------
# WebSocket: Sans-I/O framing over one asyncio stream pair
# ---------------------------------------------------------------------------


class _WebSocketLink:
    """Reads and writes WebSocket messages for one connection.

    All the protocol work belongs to ``websockets``' sans-I/O
    :class:`~websockets.server.ServerProtocol`; this class is only the I/O
    half. The one invariant worth stating: ``send_text``/``receive_data`` and
    the ``data_to_send`` that flushes them are called together with no ``await``
    in between, so two producers (the event forwarder and the read pump's
    automatic pong/close replies) can never interleave half a frame.
    """

    def __init__(
        self, protocol: ServerProtocol, reader: StreamReader, writer: StreamWriter
    ) -> None:
        self._protocol = protocol
        self._reader = reader
        self._writer = writer

    async def handshake(self) -> bool:
        """Answer the upgrade request already fed into the protocol."""
        from websockets.http11 import Request as WSRequest

        events = self._protocol.events_received()
        request = events[0] if events else None
        if not isinstance(request, WSRequest):  # pragma: no cover - head was complete
            self._protocol.send_response(
                self._protocol.reject(
                    HTTPStatus.BAD_REQUEST.value, "expected a WebSocket handshake\n"
                )
            )
        else:
            # `accept` returns a rejection response (and sets handshake_exc)
            # when the request is not a valid handshake, so the response is
            # always sent and the verdict always read off handshake_exc.
            self._protocol.send_response(self._protocol.accept(request))
        self._flush()
        await self._drain()
        return self._protocol.handshake_exc is None

    def send_json(self, payload: dict[str, Any]) -> None:
        """Send one JSON text frame. A no-op once the peer has started closing."""
        if not self._is_open():
            return
        self._protocol.send_text(json.dumps(payload).encode("utf-8"))
        self._flush()

    def close(self, code: int = 1000, reason: str = "") -> None:
        """Start the closing handshake, if it has not started already."""
        if not self._is_open():
            return
        self._protocol.send_close(code, reason)
        self._flush()

    def _is_open(self) -> bool:
        from websockets.protocol import State

        return self._protocol.state is State.OPEN

    async def messages(self) -> AsyncIterator[str]:
        """Yield complete text messages until the connection ends."""
        from websockets.frames import Frame, Opcode

        buffer = bytearray()
        collecting = False

        while True:
            # The how-to's rule for a finished closing handshake: do not wait
            # indefinitely for a peer that will never close the TCP connection.
            timeout = _WS_CLOSE_TIMEOUT if self._protocol.close_expected() else None
            try:
                data = await asyncio.wait_for(self._reader.read(_WS_READ_BYTES), timeout)
            except (TimeoutError, ConnectionResetError):
                return
            if not data:
                self._protocol.receive_eof()
                self._flush()
                return

            self._protocol.receive_data(data)
            # Flush before handling events: the protocol has already queued
            # pong frames, close echoes and protocol-error responses, and the
            # documented pattern is to put them on the wire immediately.
            self._flush()

            for event in self._protocol.events_received():
                if not isinstance(event, Frame):  # pragma: no cover - post-handshake
                    continue
                if event.opcode is Opcode.TEXT:
                    buffer.clear()
                    buffer += event.data
                    collecting = True
                elif event.opcode is Opcode.CONT and collecting:
                    buffer += event.data
                else:
                    # Binary payloads and control frames: the studio speaks
                    # JSON text only, and control frames are already answered.
                    collecting = False
                    continue
                if event.fin:
                    collecting = False
                    message = bytes(buffer).decode("utf-8", errors="replace")
                    buffer.clear()
                    yield message

            if not await self._drain():
                return

    def _flush(self) -> None:
        for data in self._protocol.data_to_send():
            if data:
                self._writer.write(data)
            elif self._writer.can_write_eof():
                # The sans-I/O layer's sentinel for "half-close the TCP
                # connection now"; the read side stays open for the peer's
                # own close frame.
                self._writer.write_eof()

    async def _drain(self) -> bool:
        try:
            await self._writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            return False
        return True


# ---------------------------------------------------------------------------
# The live session bridge
# ---------------------------------------------------------------------------



def _studio_mock_handlers(spec: AgentSpec) -> dict[str, Any]:
    """Bind a mock handler to every tool the spec declares.

    Studio test sessions have no Python backend to execute real tools, and a
    LookupError mid-conversation tests nothing. A mock that echoes the parsed
    arguments lets a builder verify the half that Studio CAN test, which is
    that the model chose the right tool, filled the right arguments, and
    choreographed the silence around the call. The result is clearly labeled
    so nobody mistakes it for a real integration.
    """

    def make(tool_name: str) -> Any:
        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            return {
                "status": "ok",
                "note": f"studio mock: '{tool_name}' is not connected to a real backend",
                "echoed_arguments": args,
            }

        return handler

    return {tool.name: make(tool.name) for tool in spec.tools}

def _force_studio_providers(spec: AgentSpec) -> list[str]:
    """Rewrite a loaded spec for text-only studio use; return what changed.

    Mutates the in-memory copy only — nothing is written back to ``agent.yaml``,
    so the spec the user edits stays the spec they deploy.

    Every routing entry is rewritten, not just the one the agent's primary
    language selects, so switching ``language.primary`` in the editor cannot
    leave a microphone-only provider wired into the next session.
    """
    notes: list[str] = []
    for language, selection in spec.runtime.routing.items():
        selection.stt = TEXT_INPUT_STT_NAME
        reason = _tts_unavailable(selection)
        if reason is None:
            continue
        notes.append(
            f"routing[{language}]: the configured TTS provider "
            f"{selection.tts!r} is unavailable ({reason}). Falling back to "
            f"{SILENT_TTS_NAME!r}: no audio, exact character counts."
        )
        selection.tts = SILENT_TTS_NAME
    return notes


def _tts_unavailable(selection: ProviderSelection) -> str | None:
    """Why the selected TTS provider cannot be used, or ``None`` if it can.

    The probe is construction: ask the registry for the provider with the same
    options the runtime would pass. That catches a name the registry has never
    heard of, a vendor SDK that is not installed, and a provider that refuses
    to construct without credentials.

    It does **not** catch everything, and pretending otherwise would be the
    dishonest part. Providers whose heavy import or key lookup happens inside
    ``synthesize`` (``kokoro`` and the cloud engines both do this, deliberately,
    to keep ``registry._load_builtin`` cheap) construct fine here and fail on
    the first utterance instead. That failure is not swallowed: it reaches the
    UI as a ``SessionError`` on the normal event stream, naming the missing
    package — at which point the fix is to pick ``studio_silent`` in the editor.
    """
    name = selection.tts
    if name is None:
        return "no TTS provider is configured for this routing entry"
    if name == SILENT_TTS_NAME:
        return None
    registry._load_builtin()
    try:
        registry.create("tts", name, **_options_for(selection, "tts"))
    except (ImportError, RuntimeError, LookupError, OSError) as exc:
        return _collapse(str(exc)) or type(exc).__name__
    return None


class _StudioBridge:
    """One browser connection and the live session it drives.

    Owns exactly three pieces of state — the session, its runtime, and the task
    forwarding events to the socket — and rebuilds all three together. ``start``
    and ``reset`` are the same operation for that reason: on a fresh connection
    there is simply nothing to tear down first.
    """

    def __init__(self, link: _WebSocketLink, server: StudioServer) -> None:
        self._link = link
        self._server = server
        self._session: CallSession | None = None
        self._runtime: CascadeRuntime | None = None
        self._forwarder: asyncio.Task[None] | None = None
        self._audio_bytes = 0
        self._last_progress = float("-inf")

    async def run(self) -> None:
        try:
            async for raw in self._link.messages():
                await self._dispatch(raw)
        finally:
            await self._teardown()

    # ----------------------------------------------------------- dispatching

    async def _dispatch(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            self._error("message was not valid JSON")
            return
        if not isinstance(message, dict):
            self._error("message must be a JSON object")
            return

        kind = message.get("type")
        if kind in ("start", "reset"):
            await self._start_session()
        elif kind == "user_text":
            await self._user_text(message)
        else:
            self._error(f"unknown message type {kind!r}")

    async def _user_text(self, message: dict[str, Any]) -> None:
        text = message.get("text")
        if not isinstance(text, str) or not text.strip():
            self._error('"user_text" needs a non-empty "text" field')
            return
        runtime = self._runtime
        if runtime is None:
            self._error('no live session: send {"type": "start"} first')
            return
        # The text_input STT provider reads UTF-8 out of the pcm field, so a
        # typed turn enters the runtime through the same door as real audio.
        await runtime.push_audio(AudioFrame(pcm=text.encode("utf-8")))

    def _error(self, message: str) -> None:
        self._link.send_json({"type": "error", "message": message})

    # ------------------------------------------------------------- lifecycle

    async def _start_session(self) -> None:
        await self._teardown()

        try:
            spec = self._server.load_agent_spec()
        except (OSError, yaml.YAMLError, ValidationError) as exc:
            self._error(f"could not load the agent spec: {_collapse(str(exc))}")
            return

        fallback_notes = _force_studio_providers(spec)
        session = CallSession(spec)
        runtime = CascadeRuntime(
            session,
            handlers=_studio_mock_handlers(spec),
            meter=CostMeter(session, DEFAULT_RATES),
        )
        runtime.on_bot_audio = self._count_bot_audio
        self._session = session
        self._runtime = runtime
        self._audio_bytes = 0
        self._last_progress = float("-inf")

        self._forwarder = asyncio.create_task(self._forward_events(session))
        # `create_task` only schedules; the subscription inside `_forward_events`
        # registers its queue on the task's first step. Yielding once here lets
        # that step run, because nothing below awaits before `runtime.start()`
        # emits SessionStarted — and an unregistered subscriber would miss it.
        await asyncio.sleep(0)

        self._link.send_json(
            {
                "type": "ready",
                "session_id": session.session_id,
                "capabilities": runtime.capabilities.model_dump(mode="json"),
            }
        )
        for note in fallback_notes:
            # Reported as a session event, not as a transport-level error: it
            # is a fact about this call, and it belongs in the event log and
            # the replayable history alongside everything else.
            session.emit(
                SessionError(
                    session_id=session.session_id,
                    at=session.elapsed,
                    message=note,
                    recoverable=True,
                )
            )

        try:
            await runtime.start()
        except Exception as exc:
            # Binding providers or speaking the greeting can fail for as many
            # reasons as there are providers. None of them should take the
            # socket down: report it and leave the studio usable.
            self._error(f"session failed to start: {_collapse(str(exc))}")
            await self._teardown()

    async def _teardown(self) -> None:
        runtime, session, forwarder = self._runtime, self._session, self._forwarder
        self._runtime, self._session, self._forwarder = None, None, None

        if runtime is not None:
            runtime.on_bot_audio = None
        if forwarder is not None:
            # Cancelled synchronously and first: whatever happens to the awaits
            # below (including this coroutine itself being cancelled), the
            # forwarder task cannot outlive the session it was reading.
            forwarder.cancel()
        if runtime is not None:
            with suppress(Exception):
                await runtime.stop(reason="studio_disconnected")
        if session is not None:
            session.close()
        if forwarder is not None:
            with suppress(asyncio.CancelledError):
                await forwarder

    async def _forward_events(self, session: CallSession) -> None:
        """Every SessionEvent, verbatim, exactly as any other consumer sees it."""
        async for event in session.subscribe():
            self._link.send_json({"type": "event", "event": event.model_dump(mode="json")})

    def _count_bot_audio(self, frame: AudioFrame) -> None:
        """Count bot audio instead of shipping it (protocol doc, "WebSocket /ws").

        Synchronous because ``RuntimeAdapter.on_bot_audio`` is called, not
        awaited; the send is a buffered write, so there is nothing to await.
        """
        self._audio_bytes += len(frame.pcm)
        now = time.monotonic()
        if now - self._last_progress < _AUDIO_PROGRESS_INTERVAL:
            return
        self._last_progress = now
        self._link.send_json({"type": "audio_progress", "bytes": self._audio_bytes})


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


class StudioServer:
    """Serves the studio UI, the agent config API and the live session socket.

    Args:
        agent_path: the spec the studio reads and writes. It does not have to
            exist: until it does, the bundled demo spec is served and the first
            save creates the file.
        host: interface to bind. Defaults to loopback, because the studio edits
            and runs agent code and has no authentication of its own.
        port: TCP port; ``0`` binds an ephemeral one, readable from
            :attr:`port` once :meth:`start` has returned.
        static_dir: where the built frontend lives. Overridable so a Vite build
            directory can be served directly during UI work.
    """

    def __init__(
        self,
        agent_path: str | Path = DEFAULT_AGENT_PATH,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        static_dir: str | Path | None = None,
    ) -> None:
        self.agent_path = Path(agent_path).expanduser().resolve()
        self.host = host
        self.port = port
        self.static_dir = (
            Path(static_dir).expanduser().resolve() if static_dir else DEFAULT_STATIC_DIR
        )
        self._server: asyncio.Server | None = None
        self._connections: set[asyncio.Task[Any]] = set()
        self._writers: set[StreamWriter] = set()

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Bind and begin accepting. Sets :attr:`port` when ``port=0`` was used."""
        _require_websockets()
        self._server = await asyncio.start_server(
            self._handle_connection, self.host, self.port, limit=_MAX_HEADER_BYTES
        )
        sockets = self._server.sockets
        if sockets:
            self.port = int(sockets[0].getsockname()[1])

    async def serve_forever(self) -> None:
        if self._server is None:
            raise RuntimeError("await start() before serve_forever()")
        await self._server.serve_forever()

    async def close(self) -> None:
        """Stop accepting, let live connections tear their sessions down, exit.

        Connections are closed at the socket rather than cancelled at the task,
        so each one's own ``finally`` runs to completion: a cancelled handler
        would be interrupted at the first ``await`` inside teardown and could
        leave a runtime's STT task alive. Cancellation is the fallback for
        anything still running after the grace period.
        """
        server, self._server = self._server, None
        if server is not None:
            server.close()
        for writer in list(self._writers):
            with suppress(Exception):
                writer.close()
        tasks = list(self._connections)
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=_SHUTDOWN_GRACE)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        if server is not None:
            with suppress(Exception):
                await server.wait_closed()

    @property
    def url(self) -> str:
        """The address to open in a browser (a wildcard bind is shown as loopback)."""
        host = "127.0.0.1" if self.host in ("", "0.0.0.0", "::") else self.host
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    # ----------------------------------------------------------- agent specs

    def agent_yaml(self) -> str:
        """The text behind ``GET /api/agent``.

        An existing file is returned byte for byte, comments and formatting
        intact — the studio is an editor, not a reformatter.

        The bundled demo is serialized as JSON instead. JSON is a subset of
        YAML, so ``yaml.safe_load`` on the way back in is unaffected, and it is
        what lets the frontend open its structured editor on first run: it tries
        ``JSON.parse`` and falls back to a raw-text pane when that fails, rather
        than round-tripping hand-written YAML through a lossy parser.
        """
        if self.agent_path.is_file():
            return self.agent_path.read_text(encoding="utf-8")
        return json.dumps(DEMO_AGENT.model_dump(mode="json"), indent=2)

    def load_agent_spec(self) -> AgentSpec:
        """The spec a new session runs, re-read from disk every time.

        Re-reading rather than caching is what makes "save, then start" work
        without a cache-invalidation rule: the file is the only state.
        """
        if self.agent_path.is_file():
            return AgentSpec.from_yaml(self.agent_path)
        return DEMO_AGENT.model_copy(deep=True)

    def save_agent_yaml(self, body: bytes) -> dict[str, Any]:
        """Validate and persist a ``POST /api/agent`` body.

        The submitted text is written verbatim once it validates, so what the
        editor shows and what the file holds are the same bytes. Validation is
        a real ``AgentSpec`` parse: the studio must never persist a spec that
        the runtime would then refuse to load.
        """
        try:
            envelope = json.loads(body)
        except json.JSONDecodeError:
            return {"ok": False, "error": "request body was not valid JSON"}
        if not isinstance(envelope, dict) or not isinstance(envelope.get("yaml"), str):
            return {
                "ok": False,
                "error": 'expected a JSON object with a "yaml" string field',
            }

        text: str = envelope["yaml"]
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            return {"ok": False, "error": f"invalid YAML: {_collapse(str(exc))}"}
        if not isinstance(data, dict):
            return {"ok": False, "error": "the agent spec must be a YAML mapping"}
        try:
            AgentSpec.model_validate(data)
        except ValidationError as exc:
            return {"ok": False, "error": _describe_validation_error(exc)}

        try:
            self.agent_path.parent.mkdir(parents=True, exist_ok=True)
            self.agent_path.write_text(text, encoding="utf-8")
        except OSError as exc:
            return {"ok": False, "error": f"could not write {self.agent_path}: {exc}"}
        return {"ok": True}

    def meta(self) -> dict[str, Any]:
        """Registry contents for the editor's provider dropdowns."""
        registry._load_builtin()
        return {
            "version": __version__,
            "providers": {
                "stt": [name for _, name in registry.available("stt")],
                "llm": [name for _, name in registry.available("llm")],
                "tts": [name for _, name in registry.available("tts")],
                "s2s": [name for _, name in registry.available("s2s")],
            },
        }

    # -------------------------------------------------------------- serving

    async def _handle_connection(self, reader: StreamReader, writer: StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)
        self._writers.add(writer)
        upgraded = False
        try:
            try:
                request = await _read_head(reader)
                if request is None:
                    _write_response(
                        writer, _text_response(HTTPStatus.BAD_REQUEST, "bad request\n")
                    )
                elif _is_websocket_upgrade(request):
                    upgraded = True
                    await self._serve_websocket(request, reader, writer)
                else:
                    _write_response(
                        writer,
                        await self._route(request, reader),
                        with_body=request.method != "HEAD",
                    )
            except (ConnectionResetError, BrokenPipeError):
                return
            except Exception:
                # A bug in a handler must not leave the browser holding a
                # socket that will never answer. Log it where a developer
                # working on the studio will see it, and answer. Nothing has
                # been written yet on this path: `_route` is evaluated before
                # `_write_response` is called.
                logger.exception("studio request handler failed")
                if not upgraded:
                    _write_response(
                        writer,
                        _text_response(
                            HTTPStatus.INTERNAL_SERVER_ERROR, "internal studio error\n"
                        ),
                    )
            with suppress(ConnectionResetError, BrokenPipeError):
                await writer.drain()
        finally:
            self._writers.discard(writer)
            if task is not None:
                self._connections.discard(task)
            with suppress(Exception):
                writer.close()

    async def _route(self, request: _Request, reader: StreamReader) -> _Response:
        if request.method in ("GET", "HEAD"):
            if request.path == "/api/meta":
                return _json_response(self.meta())
            if request.path == "/api/agent":
                return _json_response({"yaml": self.agent_yaml()})
            return self._static(request.path)
        if request.method == "POST":
            if request.path != "/api/agent":
                return _text_response(HTTPStatus.NOT_FOUND, "not found\n")
            body = await _read_body(reader, request)
            if body is None:
                return _text_response(HTTPStatus.BAD_REQUEST, "unreadable request body\n")
            return _json_response(self.save_agent_yaml(body))
        return _text_response(HTTPStatus.METHOD_NOT_ALLOWED, "method not allowed\n")

    def _static(self, path: str) -> _Response:
        """Serve any file under ``static_dir``, by path, traversal-safe.

        The frontend is a hashed bundle (``index.html`` plus ``assets/``), so
        this cannot be a two-file special case. Safety is one rule applied after
        the fact: resolve the joined path — which collapses ``..`` and follows
        symlinks — and refuse anything that did not land inside the root.
        """
        relative = path.lstrip("/") or "index.html"
        root = self.static_dir
        try:
            target = (root / relative).resolve()
        except OSError:  # pragma: no cover - platform-specific path limits
            return _text_response(HTTPStatus.NOT_FOUND, "not found\n")
        if not target.is_relative_to(root):
            return _text_response(HTTPStatus.NOT_FOUND, "not found\n")
        if not target.is_file():
            if relative == "index.html":
                return self._frontend_not_built()
            return _text_response(HTTPStatus.NOT_FOUND, "not found\n")

        content_type = _CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        try:
            return _Response(HTTPStatus.OK, target.read_bytes(), content_type)
        except OSError as exc:
            return _text_response(
                HTTPStatus.NOT_FOUND, f"could not read {relative}: {exc}\n"
            )

    def _frontend_not_built(self) -> _Response:
        """A page that says what is missing, instead of a bare 404.

        Answered 200 on purpose: the server is healthy and the API below is
        live. What is absent is a build artifact, and the useful response to
        that is an explanation a developer can act on.
        """
        page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tring Studio &mdash; frontend not built</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.6 ui-sans-serif, system-ui, sans-serif; margin: 0;
         display: grid; place-items: center; min-height: 100vh; padding: 2rem; }}
  main {{ max-width: 34rem; }}
  h1 {{ font-size: 1.3rem; margin: 0 0 .75rem; }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: .9em; }}
  p code {{ overflow-wrap: anywhere; }}
  ul {{ padding-left: 1.2rem; }}
</style>
</head>
<body>
<main>
  <h1>Tring Studio: the frontend is not built</h1>
  <p>The server is running, but there is no <code>index.html</code> in the
     static directory it serves from:</p>
  <p><code>{escape(str(self.static_dir))}</code></p>
  <p>Build the studio UI and place the result &mdash; <code>index.html</code>
     plus its <code>assets/</code> directory &mdash; in that path, then reload.</p>
  <p>The back end is already live and can be used without the UI:</p>
  <ul>
    <li><code>GET /api/meta</code> &mdash; version and registered providers</li>
    <li><code>GET /api/agent</code> &mdash; the current agent spec as YAML</li>
    <li><code>POST /api/agent</code> &mdash; validate and save a spec</li>
    <li><code>ws://&hellip;/ws</code> &mdash; the live session socket</li>
  </ul>
</main>
</body>
</html>
"""
        return _Response(HTTPStatus.OK, page.encode("utf-8"), "text/html; charset=utf-8")

    async def _serve_websocket(
        self, request: _Request, reader: StreamReader, writer: StreamWriter
    ) -> None:
        if request.path != "/ws":
            _write_response(writer, _text_response(HTTPStatus.NOT_FOUND, "not found\n"))
            return

        from websockets.server import ServerProtocol

        protocol = ServerProtocol()
        # The handshake request has already been read off the socket by the
        # HTTP layer; replaying the exact bytes lets the protocol parse it
        # itself, keys and all, rather than being handed a reconstruction.
        protocol.receive_data(request.raw_head)

        link = _WebSocketLink(protocol, reader, writer)
        if not await link.handshake():
            return
        bridge = _StudioBridge(link, self)
        try:
            await bridge.run()
        finally:
            link.close(1001, "studio shutting down")
            with suppress(ConnectionResetError, BrokenPipeError):
                await writer.drain()


__all__ = [
    "DEFAULT_AGENT_PATH",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "DEFAULT_STATIC_DIR",
    "DEMO_AGENT",
    "StudioServer",
]
