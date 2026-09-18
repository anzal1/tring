"""Text-based console transport for local development iteration without audio.

Reads lines from stdin (UTF-8), wraps each line as an AudioFrame of text bytes,
and feeds it into the runtime as if it were speech audio. Subscribes to the
session event stream and pretty-prints events with ANSI colors.

Designed for rapid local iteration where you want to test agent logic without
audio hardware or cloud dependencies. The runtime's STT provider should be set
to "text_input", a pass-through provider that treats UTF-8 input bytes as
transcribed text.

Example:
    >>> transport = ConsoleTransport(runtime)
    >>> await transport.run()
    User> hello
    Bot> Hi there!
    User> what's the time
    Bot> It's 3 PM
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from typing import TYPE_CHECKING

from tring.events import (
    BotSpeechPlayed,
    BotUtterance,
    CostRecorded,
    SessionEnded,
    SessionStarted,
    ToolCallCompleted,
    ToolCallStarted,
    UserTranscript,
)
from tring.runtimes.base import AudioFrame

if TYPE_CHECKING:
    from tring.runtimes.base import RuntimeAdapter


# ANSI color codes for terminal output (work in light and dark themes)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_USER = "\033[36m"  # Cyan
_BOT = "\033[35m"  # Magenta
_TOOL = "\033[33m"  # Yellow
_COST = "\033[32m"  # Green
_ERROR = "\033[31m"  # Red


class ConsoleTransport:
    """A text-based transport for console-driven development and testing.

    Reads UTF-8 lines from stdin, wraps them as audio frames, and sends them to
    the runtime's STT provider. Subscribes to the session event stream and
    pretty-prints all events with simple ANSI colors. The on_bot_audio callback
    just counts bytes generated (no actual playback).

    Attributes:
        runtime: The RuntimeAdapter controlling this conversation.
        _bot_audio_bytes: Total bytes of bot audio generated.
    """

    def __init__(self, runtime: RuntimeAdapter) -> None:
        """Initialize the console transport.

        Args:
            runtime: A RuntimeAdapter managing the conversation session.
        """
        self.runtime = runtime
        self._bot_audio_bytes = 0

    async def run(self) -> None:
        """Run the console input loop until EOF or SessionEnded event.

        Reads lines from stdin using asyncio's run_in_executor (non-blocking),
        wraps each UTF-8 line as an AudioFrame, and pushes it to the runtime.
        Simultaneously subscribes to the session event stream and pretty-prints
        all events.

        The loop terminates on:
        - EOF (no more input on stdin)
        - SessionEnded event from the runtime

        Raises:
            RuntimeError: if runtime.start() fails.
        """
        # Attach the bot audio callback: just count bytes.
        self.runtime.on_bot_audio = self._on_bot_audio

        # Start the runtime and subscribe to events.
        await self.runtime.start()
        event_task = asyncio.create_task(self._watch_events())
        input_task = asyncio.create_task(self._read_input())

        # Wait for either the input loop to finish (EOF) or events to signal end.
        _, pending = await asyncio.wait(
            [event_task, input_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel remaining tasks and clean up.
        for task in pending:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        await self.runtime.stop(reason="completed")

    async def _read_input(self) -> None:
        """Read lines from stdin in a thread pool and push them as audio frames.

        Each line is encoded as UTF-8 and wrapped in an AudioFrame. The
        frame's pcm field contains the raw UTF-8 bytes; the STT provider
        (text_input) will decode and use it as a transcript.

        This method runs in a background task and terminates on EOF.
        """
        loop = asyncio.get_event_loop()
        while True:
            try:
                # Read a line from stdin without blocking the event loop.
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    # EOF on stdin.
                    break

                # Strip trailing newline and encode as UTF-8 bytes.
                text = line.rstrip("\n\r")
                pcm_bytes = text.encode("utf-8")

                # Wrap as an AudioFrame. Frame's sample_rate and channels
                # don't matter for text input; the STT provider ignores them.
                frame = AudioFrame(pcm=pcm_bytes, sample_rate=16000, channels=1)

                # Push the frame to the runtime's STT pipeline.
                await self.runtime.push_audio(frame)

            except Exception as e:
                print(f"{_ERROR}{_BOLD}Error reading input: {e}{_RESET}", file=sys.stderr)
                break

    async def _watch_events(self) -> None:
        """Subscribe to session events and pretty-print them.

        Runs until a SessionEnded event is received. Each event type gets
        a distinct color and format for quick visual parsing in the console.
        """
        async for event in self.runtime.session.subscribe():
            if isinstance(event, SessionStarted):
                print(f"{_DIM}[Session {event.session_id[:8]}...] "
                      f"{event.agent_name} on {event.runtime_mode}{_RESET}")

            elif isinstance(event, UserTranscript):
                print(f"{_USER}{_BOLD}User>{_RESET} {event.text}")

            elif isinstance(event, BotUtterance):
                print(f"{_BOT}{_BOLD}Bot (thinking)>{_RESET} {event.text}")

            elif isinstance(event, BotSpeechPlayed):
                print(f"{_BOT}{_BOLD}Bot>{_RESET} {event.text}")

            elif isinstance(event, ToolCallStarted):
                args_str = ", ".join(
                    f"{k}={v!r}" for k, v in event.arguments.items()
                ) if event.arguments else ""
                print(f"{_TOOL}{_BOLD}[Tool]{_RESET} "
                      f"{event.tool_name}({args_str})")
                if event.waiting_message:
                    print(f"{_DIM}  {event.waiting_message}{_RESET}")

            elif isinstance(event, ToolCallCompleted):
                status = "OK" if event.ok else "FAILED"
                print(f"{_TOOL}{_BOLD}[Tool Done]{_RESET} "
                      f"{event.tool_name}: {status}")
                if event.result_summary:
                    print(f"{_DIM}  {event.result_summary}{_RESET}")

            elif isinstance(event, CostRecorded):
                unit_str = event.unit_name
                symbol = "*" if event.estimated else ""
                print(f"{_COST}[Cost]{_RESET} "
                      f"{event.component.value}/"
                      f"{event.provider}: "
                      f"{event.units:.1f} {unit_str}{symbol} = "
                      f"${event.amount:.6f}")

            elif isinstance(event, SessionEnded):
                print(f"{_DIM}[Session ended: {event.reason}] "
                      f"Duration: {event.duration_seconds:.1f}s{_RESET}")
                # Signal to stop the input loop and exit.
                break

    def _on_bot_audio(self, frame: AudioFrame) -> None:
        """Callback invoked when the runtime generates bot audio.

        In a real transport, this would queue audio for playback. Here we
        just count total bytes for simple instrumentation.

        Args:
            frame: An AudioFrame containing bot audio to "play".
        """
        self._bot_audio_bytes += len(frame.pcm)
