"""Provider interfaces: STT, LLM, TTS, S2S.

All interfaces are streaming-first — voice latency budgets do not allow
request/response round trips. Implementations live in ``providers/local`` and
``providers/cloud`` behind lazy imports; the core never imports vendor SDKs.

Usage reporting contract: providers return exact vendor-reported usage when
available and set ``estimated=True`` on anything inferred. Silently estimated
usage is a bug.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from trunkline.runtimes.base import AudioFrame


@dataclass
class Usage:
    units: float
    unit_name: str  # "tokens_in", "tokens_out", "tts_chars", "audio_seconds"
    estimated: bool = False
    model: str | None = None
    cached_units: float | None = None


@dataclass
class STTResult:
    text: str
    final: bool
    language: str | None = None
    usage: list[Usage] = field(default_factory=list)


@dataclass
class LLMChunk:
    """One streamed token/segment of raw model output."""

    text: str
    usage: list[Usage] = field(default_factory=list)  # populated on final chunk
    finish: bool = False


@dataclass
class TTSChunk:
    frame: AudioFrame
    usage: list[Usage] = field(default_factory=list)


class STTProvider(abc.ABC):
    name: str = "stt"

    @abc.abstractmethod
    def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]: ...


class LLMProvider(abc.ABC):
    name: str = "llm"

    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        """Stream raw text. Tool-call framing is the speak_parser's job —
        the provider must not buffer output waiting for complete JSON."""


class TTSProvider(abc.ABC):
    name: str = "tts"

    @abc.abstractmethod
    def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        """Consume a *stream* of text (speak-while-thinking) and stream audio."""


class S2SProvider(abc.ABC):
    """A speech-to-speech model session (audio in, audio out, tools inline)."""

    name: str = "s2s"

    @abc.abstractmethod
    def converse(
        self, frames: AsyncIterator[AudioFrame]
    ) -> AsyncIterator[TTSChunk]: ...

    @abc.abstractmethod
    async def post_call_transcript(self) -> list[dict]:
        """Best-effort transcript after the call; may be empty."""
