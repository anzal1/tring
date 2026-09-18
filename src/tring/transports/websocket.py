"""Real-time audio-over-WebSocket transport for cloud and LAN deployment.

Accepts WebSocket connections and runs a full voice conversation session over
them. Binary messages carry 16kHz mono 16-bit PCM audio frames; the runtime
processes them through STT -> LLM -> TTS. Bot audio is sent back as binary
frames. Text messages with {"type":"end"} cleanly stop the session.

Typical deployment: run a local Ollama for LLM and Kokoro for TTS, expose
this server on a private network, and connect from a browser or mobile client.

Example:
    >>> from tring import AgentSpec, CallSession
    >>> from tring.runtimes.cascade import CascadeRuntime
    >>> from tring.transports.websocket import serve
    >>>
    >>> agent = AgentSpec.from_yaml("agent.yaml")
    >>> def factory(session):
    ...     return CascadeRuntime(session)
    >>>
    >>> await serve(factory, host="0.0.0.0", port=8765)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from tring.runtimes.base import AudioFrame
from tring.session import CallSession

if TYPE_CHECKING:
    from tring.runtimes.base import RuntimeAdapter

logger = logging.getLogger(__name__)


async def serve(
    runtime_factory: Callable[[CallSession], RuntimeAdapter],
    host: str = "0.0.0.0",
    port: int = 8765,
) -> None:
    """Serve a WebSocket endpoint for real-time audio conversations.

    For each incoming WebSocket connection, creates a new CallSession and
    Runtime via the factory. Binary messages are treated as 16kHz mono 16-bit
    PCM audio frames and pushed to the runtime. Bot audio is streamed back as
    binary frames. Text messages (JSON) with {"type":"end"} stop the session.

    Args:
        runtime_factory: A callable that takes a CallSession and returns a
            configured RuntimeAdapter. Called once per connection.
        host: The network interface to bind to (default: all interfaces).
        port: The port number to listen on (default: 8765).

    Raises:
        ImportError: if the websockets library is not installed.
    """
    try:
        import websockets
    except ImportError as e:
        raise ImportError(
            "websockets library not found. "
            "Install tring[transports] to enable WebSocket support."
        ) from e

    async def handle_connection(websocket: websockets.WebSocketServerProtocol) -> None:
        """Handle one WebSocket connection: run a full voice session.

        Args:
            websocket: The WebSocket connection from a client.
        """
        # Create a new session and runtime for this connection.
        # The factory is expected to initialize the session as needed.
        session = CallSession.__new__(CallSession)  # Will be initialized by factory
        runtime = runtime_factory(session)

        # Set up bot audio callback to send frames back as binary WebSocket messages.
        async def send_bot_audio(frame: AudioFrame) -> None:
            with contextlib.suppress(websockets.exceptions.ConnectionClosed):
                await websocket.send(frame.pcm)

        runtime.on_bot_audio = send_bot_audio

        # Start the runtime and listen for incoming messages in a background task.
        await runtime.start()
        stop_event = asyncio.Event()

        async def receive_messages() -> None:
            """Receive binary audio or JSON control messages from the client."""
            try:
                async for message in websocket:
                    if isinstance(message, bytes):
                        # Binary audio frame: push to the runtime's STT.
                        frame = AudioFrame(
                            pcm=message,
                            sample_rate=16000,
                            channels=1,
                        )
                        await runtime.push_audio(frame)
                    else:
                        # Text message: expect JSON control commands.
                        try:
                            cmd = json.loads(message)
                            if cmd.get("type") == "end":
                                # Client requested session end.
                                stop_event.set()
                                break
                        except json.JSONDecodeError:
                            logger.warning("Received invalid JSON: %s", message)
            except websockets.exceptions.ConnectionClosed:
                stop_event.set()

        receive_task = asyncio.create_task(receive_messages())

        # Wait for either the receive loop to finish or a stop signal.
        await stop_event.wait()
        receive_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receive_task

        # Clean up the runtime.
        await runtime.stop(reason="client_closed")
        await websocket.close()

    # Start the WebSocket server.
    async with websockets.serve(handle_connection, host, port):
        logger.info("WebSocket server listening on ws://%s:%d", host, port)
        # Keep the server running indefinitely.
        await asyncio.Future()
