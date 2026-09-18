"""Local, open-model providers: faster-whisper STT, Ollama LLM, Kokoro TTS.

These are the providers that make Alaap's "local-first" claim real: a complete
cascade pipeline that runs on your own hardware at $0 of vendor spend.

Two rules shape every class in this file.

**Lazy imports.** Importing this module must stay cheap and must never fail,
because :func:`alaap.providers.registry._load_builtin` imports it eagerly just
to run the ``@register`` decorators. A core install has no ML stack, so every
heavy dependency (``faster_whisper``, ``httpx``, ``kokoro_onnx``, ``numpy``) is
imported *inside* the method that needs it, and a missing one is re-raised as
an ``ImportError`` that names the extra to install. Import-time failure would
take down provider discovery for everybody, including users who only wanted a
cloud provider.

**Honest usage.** Every :class:`~alaap.providers.base.Usage` here is measured,
not inferred, so all of them carry ``estimated=False``. We know the exact audio
duration we fed to Whisper, Ollama reports exact token counts, and we count the
exact characters we hand to Kokoro. When a provider cannot measure, it must say
so rather than quietly producing a plausible number.
"""

from __future__ import annotations

import array
import asyncio
import io
import json
import math
import re
import wave
from collections.abc import AsyncIterator
from typing import Any

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

_LOCAL_EXTRA = 'pip install "alaap[local]"'

# Canonical wire format for the whole stack (see AudioFrame).
WIRE_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # 16-bit linear PCM

# 20 ms of 16 kHz mono 16-bit audio. Small frames keep the playback ledger's
# "what did the caller actually hear" accounting fine-grained, and keep
# barge-in latency below the threshold where callers notice they were ignored.
_FRAME_SAMPLES = WIRE_SAMPLE_RATE // 50
_FRAME_BYTES = _FRAME_SAMPLES * _BYTES_PER_SAMPLE


def _missing(package: str, why: str) -> ImportError:
    """Build the ImportError we raise when an optional dependency is absent.

    The message names the concrete package *and* the extra, because the most
    common failure mode for a local stack is a user who installed plain
    ``alaap`` and then pointed an agent spec at ``faster_whisper``.
    """
    return ImportError(
        f"{package} is required to {why}. Install the local provider stack with:"
        f"\n    {_LOCAL_EXTRA}"
    )


def _pcm_rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of 16-bit little-endian PCM, 0.0 - 1.0.

    Written against the stdlib ``array`` module rather than ``audioop`` because
    ``audioop`` was removed in Python 3.13, and rather than numpy because the
    endpointer must work in a core install that has no numpy.
    """
    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    total = sum(float(s) * float(s) for s in samples)
    return math.sqrt(total / len(samples)) / 32768.0


def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int) -> io.BytesIO:
    """Wrap raw PCM in an in-memory WAV container.

    faster-whisper accepts a file-like object and decodes it itself, so this
    lets us feed it audio without a numpy dependency of our own. A 30-second
    utterance is under 1 MB, so the copy is not worth optimizing away.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(_BYTES_PER_SAMPLE)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    buf.seek(0)
    return buf


def _frames_from_pcm(pcm: bytes, sample_rate: int = WIRE_SAMPLE_RATE) -> list[AudioFrame]:
    """Slice a PCM buffer into fixed-size AudioFrames (last one may be short)."""
    return [
        AudioFrame(pcm=pcm[i : i + _FRAME_BYTES], sample_rate=sample_rate, channels=1)
        for i in range(0, len(pcm), _FRAME_BYTES)
    ]


# --------------------------------------------------------------------------
# STT
# --------------------------------------------------------------------------


@register("stt", "text_input")
class TextInputSTT(STTProvider):
    """An STT provider that "transcribes" UTF-8 text carried in the PCM field.

    This is a deliberate trick, and it is the single highest-leverage one in
    the repo. :class:`~alaap.runtimes.base.AudioFrame` is just bytes, so a
    transport can put ``"book me a table".encode("utf-8")`` in ``pcm`` and this
    provider hands it back as a final transcript. The whole runtime above it --
    turn loop, envelope parsing, tool choreography, interruption ledger, cost
    metering -- runs completely unchanged.

    That buys two things:

    * a console dev loop where you iterate on a persona by typing, with no
      microphone, no model downloads and no GPU;
    * tests of the *pipeline* that are pure and fast, because the only reason
      a voice-agent test needs audio is that STT usually demands it.

    Each frame is one utterance. It reports no usage: there is nothing to bill
    and nothing was measured, and inventing a zero-cost audio-second line would
    pollute cost reports with rows that never happened.
    """

    name = "text_input"

    def __init__(self, **_options: Any) -> None:
        # Accepts and ignores options so a spec can carry provider options for
        # its real providers while swapping this one in for a dev loop.
        pass

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]:
        async for frame in frames:
            text = frame.pcm.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            yield STTResult(text=text, final=True, language=language, usage=[])


@register("stt", "faster_whisper")
class FasterWhisperSTT(STTProvider):
    """Local Whisper transcription via `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_.

    **v0.1 is batch-per-utterance, not streaming.** Frames are buffered until a
    silence-based endpointer decides the caller stopped talking, and only then
    does the whole utterance go through the model, producing a single final
    :class:`~alaap.providers.base.STTResult`. No partial hypotheses are emitted.

    That is a real limitation and worth stating plainly: partials let a runtime
    start the LLM turn before the caller finishes, which is worth a few hundred
    milliseconds of perceived latency. Whisper is an encoder-decoder model over
    a fixed 30-second window, so true streaming means running it repeatedly on
    a sliding buffer and reconciling unstable prefixes. That belongs in its own
    release, behind its own tests.

    The endpointer here is deliberately simple: RMS energy against a fixed
    threshold. It is honest about what it is -- a dependency-free stand-in so
    the local pipeline works out of the box. Production deployments should
    route to a Silero VAD provider, which handles noisy lines and non-speech
    energy that trivially fool an energy gate.

    Usage is reported as exact ``audio_seconds``: we know precisely how many
    bytes we buffered, so nothing is estimated.
    """

    name = "faster_whisper"

    def __init__(
        self,
        model: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        silence_threshold: float = 0.01,
        silence_seconds: float = 0.6,
        min_utterance_seconds: float = 0.3,
        max_utterance_seconds: float = 30.0,
        **_options: Any,
    ) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.silence_threshold = silence_threshold
        self.silence_seconds = silence_seconds
        self.min_utterance_seconds = min_utterance_seconds
        self.max_utterance_seconds = max_utterance_seconds
        self._model: Any | None = None

    def _load_model(self) -> Any:
        """Import and construct the Whisper model on first use.

        Construction is cached on the instance: loading weights costs seconds
        and hundreds of megabytes, and a session may transcribe many turns.
        """
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - needs the real package
                raise _missing("faster-whisper", "run local Whisper transcription") from exc
            self._model = WhisperModel(
                self.model_name, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def _transcribe_utterance(
        self, pcm: bytes, sample_rate: int, channels: int, language: str | None
    ) -> tuple[str, str | None]:
        """Blocking transcription of one buffered utterance.

        Kept synchronous and separate so the async generator can hand it to a
        worker thread: faster-whisper is CPU-bound C++ underneath and would
        otherwise stall the event loop, which in a voice call means the caller
        hears the pipeline freeze.
        """
        model = self._load_model()
        wav = _pcm_to_wav(pcm, sample_rate, channels)
        segments, info = model.transcribe(wav, language=language, beam_size=1)
        text = "".join(segment.text for segment in segments).strip()
        detected = getattr(info, "language", None) or language
        return text, detected

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]:
        buffer = bytearray()
        trailing_silence = 0.0
        sample_rate = WIRE_SAMPLE_RATE
        channels = 1

        def seconds(nbytes: int) -> float:
            return nbytes / (sample_rate * _BYTES_PER_SAMPLE * channels)

        async def flush() -> STTResult | None:
            nonlocal trailing_silence
            pcm = bytes(buffer)
            buffer.clear()
            trailing_silence = 0.0
            duration = seconds(len(pcm))
            if duration < self.min_utterance_seconds:
                return None
            text, detected = await asyncio.to_thread(
                self._transcribe_utterance, pcm, sample_rate, channels, language
            )
            if not text:
                return None
            return STTResult(
                text=text,
                final=True,
                language=detected,
                usage=[
                    Usage(
                        units=duration,
                        unit_name="audio_seconds",
                        estimated=False,
                        model=self.model_name,
                    )
                ],
            )

        async for frame in frames:
            sample_rate = frame.sample_rate
            channels = frame.channels
            buffer.extend(frame.pcm)
            frame_seconds = seconds(len(frame.pcm))

            if _pcm_rms(frame.pcm) < self.silence_threshold:
                trailing_silence += frame_seconds
            else:
                trailing_silence = 0.0

            endpointed = trailing_silence >= self.silence_seconds
            overlong = seconds(len(buffer)) >= self.max_utterance_seconds
            if endpointed or overlong:
                result = await flush()
                if result is not None:
                    yield result

        # The frame stream ended (call hung up). Transcribe whatever is left
        # rather than dropping the caller's last sentence on the floor.
        if buffer:
            result = await flush()
            if result is not None:
                yield result


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------


@register("llm", "ollama")
class OllamaLLM(LLMProvider):
    """Streaming chat completions from a local `Ollama <https://ollama.com>`_ server.

    Ollama's ``/api/chat`` endpoint streams newline-delimited JSON objects, one
    per token-ish delta, and a final object with ``done: true`` that carries
    exact counters: ``prompt_eval_count`` and ``eval_count``. Those are real
    measurements from the runtime that did the work, so the final
    :class:`~alaap.providers.base.LLMChunk` reports ``estimated=False``.

    A local model has no dollar rate, but the token counts still matter: they
    are what the cost model turns into compute-seconds, and they are how you
    compare a local run against the cloud provider you are deciding whether to
    pay for. Free is not the same as unmeasured.

    This provider never buffers waiting for complete JSON. The model is asked
    (by the runtime) to emit a ``{"speak": ..., "tool_call": ...}`` envelope,
    and it is the speak-parser's job upstream to pull speakable text out of a
    half-finished object. Buffering here would erase that entire optimization.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.3,
        timeout_seconds: float = 120.0,
        keep_alive: str = "10m",
        **_options: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.keep_alive = keep_alive

    @staticmethod
    def _map_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """Adapt Alaap tool schemas to Ollama's OpenAI-shaped ``tools`` array.

        Schemas that already arrive wrapped (``{"type": "function", ...}``) are
        passed through untouched, so a caller that has its own opinion about
        the wire format keeps it.
        """
        if not tools:
            return None
        mapped: list[dict[str, Any]] = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                mapped.append(tool)
                continue
            mapped.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {}),
                    },
                }
            )
        return mapped

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - needs the real package
            raise _missing("httpx", "talk to a local Ollama server") from exc

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature},
        }
        mapped_tools = self._map_tools(tools)
        if mapped_tools:
            payload["tools"] = mapped_tools

        prompt_tokens = 0.0
        completion_tokens = 0.0

        async with (
            httpx.AsyncClient(timeout=self.timeout_seconds) as client,
            client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                delta = str(event.get("message", {}).get("content", ""))
                if delta:
                    yield LLMChunk(text=delta)
                if event.get("done"):
                    prompt_tokens = float(event.get("prompt_eval_count") or 0)
                    completion_tokens = float(event.get("eval_count") or 0)
                    break

        # One terminal chunk carrying usage, so consumers have a single,
        # unambiguous place to settle the turn's accounting.
        yield LLMChunk(
            text="",
            finish=True,
            usage=[
                Usage(
                    units=prompt_tokens,
                    unit_name="tokens_in",
                    estimated=False,
                    model=self.model,
                ),
                Usage(
                    units=completion_tokens,
                    unit_name="tokens_out",
                    estimated=False,
                    model=self.model,
                ),
            ],
        )


# --------------------------------------------------------------------------
# TTS
# --------------------------------------------------------------------------

# Flush on sentence-ending punctuation followed by whitespace or end-of-text.
# Sentence-ish, not sentence-perfect: "Dr. Rao" will split early and sound
# slightly clipped, which costs far less than the alternative failure -- a
# caller waiting in silence while we hold text back looking for a period.
_SENTENCE_END = re.compile(
    r"[.!?\u3002\uff01\uff1f]+[\"')\]]*\s*$|[,;:\u060c][\"')\]]*\s+$"
)


@register("tts", "kokoro")
class KokoroTTS(TTSProvider):
    """Local neural TTS via `kokoro-onnx <https://github.com/thewh1teagle/kokoro-onnx>`_.

    The interesting part of this class is not the model call, it is the
    *buffering policy*. :meth:`synthesize` consumes an async iterator of text
    deltas produced while the LLM is still generating, and it must decide how
    much text to hold before synthesizing. Too little and prosody collapses
    into word-by-word robot speech; too much and the caller sits in dead air.

    The rule: flush at the first sentence-ish boundary, or when the buffer
    exceeds ``max_buffer_chars`` and a word boundary is available. That puts
    first audio out roughly one clause after the first token, which is the
    perceptual difference between an agent that "answers" and one that "loads".

    Kokoro synthesizes at 24 kHz; the stack's wire format is 16 kHz mono, so
    output is resampled here (see :meth:`_resample`). Usage is exact
    ``tts_chars``: we count the characters we actually sent to the model, not
    the characters the LLM generated, because text the runtime discarded on a
    barge-in was never synthesized and must not be billed.
    """

    name = "kokoro"

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
        model_path: str = "kokoro-v1.0.onnx",
        voices_path: str = "voices-v1.0.bin",
        max_buffer_chars: int = 180,
        min_buffer_chars: int = 12,
        **_options: Any,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.model_path = model_path
        self.voices_path = voices_path
        self.max_buffer_chars = max_buffer_chars
        self.min_buffer_chars = min_buffer_chars
        self._engine: Any | None = None

    def _load_engine(self) -> Any:
        if self._engine is None:
            try:
                from kokoro_onnx import Kokoro
            except ImportError as exc:  # pragma: no cover - needs the real package
                raise _missing("kokoro-onnx", "run local Kokoro speech synthesis") from exc
            self._engine = Kokoro(self.model_path, self.voices_path)
        return self._engine

    @staticmethod
    def _resample(samples: Any, src_rate: int, dst_rate: int) -> Any:
        """Linear-interpolation resample of a float32 numpy array.

        Deliberately the cheap option, and deliberately documented as such.
        Linear interpolation aliases: downsampling 24 kHz to 16 kHz without a
        low-pass filter folds content above 8 kHz back into the audible band as
        a faint metallic edge. On a telephony-grade path (which is bandlimited
        around 3.4 kHz anyway) it is inaudible, and it keeps the local stack
        free of a scipy/soxr dependency. A future ``alaap[hifi]`` extra should
        use a polyphase resampler for anything wideband.
        """
        import numpy as np

        if src_rate == dst_rate:
            return samples
        duration = samples.shape[0] / float(src_rate)
        dst_count = int(duration * dst_rate)
        if dst_count <= 0:
            return samples[:0]
        src_index = np.linspace(0.0, samples.shape[0] - 1, dst_count, dtype=np.float64)
        return np.interp(src_index, np.arange(samples.shape[0]), samples)

    def _synthesize_text(self, text: str) -> bytes:
        """Blocking synthesis of one buffered segment into 16 kHz PCM bytes."""
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - needs the real package
            raise _missing("numpy", "convert Kokoro output to PCM frames") from exc

        engine = self._load_engine()
        samples, sample_rate = engine.create(
            text, voice=self.voice, speed=self.speed, lang=self.lang
        )
        samples = np.asarray(samples, dtype=np.float32)
        samples = self._resample(samples, int(sample_rate), WIRE_SAMPLE_RATE)
        # Clip before scaling: a float sample slightly above 1.0 would wrap to
        # a loud negative value as int16, which is heard as a click.
        clipped = np.clip(samples, -1.0, 1.0)
        pcm: bytes = (clipped * 32767.0).astype("<i2").tobytes()
        return pcm

    def _should_flush(self, buffer: str) -> bool:
        if len(buffer) < self.min_buffer_chars:
            return False
        if _SENTENCE_END.search(buffer):
            return True
        return len(buffer) >= self.max_buffer_chars and buffer.endswith(" ")

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        if voice:
            self.voice = voice
        buffer = ""

        async def speak(segment: str) -> AsyncIterator[TTSChunk]:
            pcm = await asyncio.to_thread(self._synthesize_text, segment)
            frames = _frames_from_pcm(pcm)
            for index, frame in enumerate(frames):
                # Usage rides on the first frame of the segment so a consumer
                # summing usage never double-counts a segment's characters.
                usage = (
                    [
                        Usage(
                            units=float(len(segment)),
                            unit_name="tts_chars",
                            estimated=False,
                            model=self.voice,
                        )
                    ]
                    if index == 0
                    else []
                )
                yield TTSChunk(frame=frame, usage=usage)

        async for delta in text:
            buffer += delta
            if self._should_flush(buffer):
                segment, buffer = buffer.strip(), ""
                if segment:
                    async for chunk in speak(segment):
                        yield chunk

        tail = buffer.strip()
        if tail:
            async for chunk in speak(tail):
                yield chunk


__all__ = [
    "FasterWhisperSTT",
    "KokoroTTS",
    "OllamaLLM",
    "TextInputSTT",
]
