"""RuntimeAdapter — the contract every runtime implements.

The adapter consumes caller audio frames and produces bot audio frames, while
emitting the unified SessionEvent stream on the session. Capabilities are
declared, never assumed: consumers must query ``capabilities`` instead of
guessing how a given architecture behaves.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from pydantic import BaseModel

from awaaz.session import CallSession


@dataclass(frozen=True)
class AudioFrame:
    """PCM audio. Canonical wire format: 16 kHz, mono, 16-bit linear."""

    pcm: bytes
    sample_rate: int = 16000
    channels: int = 1


class RuntimeCapabilities(BaseModel):
    live_transcripts: bool  # transcripts mid-turn (cascade) vs post-call (S2S)
    mid_call_tool_calls: bool
    barge_in: bool  # runtime supports caller interruption
    exact_usage_reporting: bool  # vendor reports exact tokens/units
    voice_cloning: bool = False
    local_capable: bool = False  # can run with zero cloud dependencies


class RuntimeAdapter(abc.ABC):
    """One conversation runtime bound to one session.

    Lifecycle: ``start`` → many ``push_audio`` → ``stop``. Bot audio is
    delivered through the ``on_bot_audio`` callback set by the transport.
    All conversational state changes are emitted as SessionEvents.
    """

    def __init__(self, session: CallSession) -> None:
        self.session = session
        self.on_bot_audio: object | None = None  # Callable[[AudioFrame], None]

    @property
    @abc.abstractmethod
    def capabilities(self) -> RuntimeCapabilities: ...

    @abc.abstractmethod
    async def start(self) -> None:
        """Bind providers, warm caches, emit SessionStarted, speak greeting."""

    @abc.abstractmethod
    async def push_audio(self, frame: AudioFrame) -> None:
        """Feed caller audio into the runtime."""

    @abc.abstractmethod
    async def stop(self, reason: str = "completed") -> None:
        """Tear down, flush post-call transcripts if any, emit SessionEnded."""
