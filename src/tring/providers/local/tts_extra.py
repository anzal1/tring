"""Extra local TTS adapter: Piper.

Same house rules as ``providers/local/__init__.py``: importing this module
must stay cheap and must never fail (``registry._load_builtin`` imports it
eagerly), so the ``piper`` package is imported lazily inside the method that
needs it, and a missing install is re-raised as an ``ImportError`` naming
the pip package -- ``pip install piper-tts`` -- per this task's instruction,
not the extras-group phrasing ``_missing`` uses elsewhere in this package.

**Sentence-ish buffering.** Piper is a local model with per-call overhead
(loading weights once, then a blocking C++ inference call per invocation),
so -- exactly like ``KokoroTTS`` above -- text deltas are buffered into
sentence-ish segments before being handed to the model, rather than calling
it per streamed token.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from typing import Any

from tring.cost.rates import Rate
from tring.events import CostComponent
from tring.providers.base import TTSChunk, TTSProvider, Usage
from tring.providers.registry import register
from tring.runtimes.base import AudioFrame

# Canonical wire format for the whole stack (see runtimes/base.py's
# AudioFrame docstring): 16 kHz, mono, 16-bit linear PCM.
WIRE_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # 16-bit linear PCM

# 20 ms of 16 kHz mono 16-bit audio -- same framing granularity as
# providers/local's _frames_from_pcm, kept fine-grained for the same reason:
# the interruption ledger and barge-in latency both care about small frames.
_FRAME_BYTES = (WIRE_SAMPLE_RATE // 50) * _BYTES_PER_SAMPLE

# Flush on sentence-ending punctuation followed by whitespace or
# end-of-text (identical pattern to KokoroTTS's _SENTENCE_END above --
# duplicated rather than imported so this file stays a self-contained unit
# another agent can edit without touching KokoroTTS's).
_SENTENCE_END = re.compile(
    r"[.!?\u3002\uff01\uff1f]+[\"')\]]*\s*$|[,;:\u060c][\"')\]]*\s+$"
)


def _missing_piper() -> ImportError:
    """Build the ImportError raised when the ``piper`` package is absent.

    Named ``pip install piper-tts`` explicitly (that is the PyPI
    distribution name; the importable package is ``piper``), so the error a
    user sees names exactly what to run.
    """
    return ImportError(
        "piper is required to run local Piper speech synthesis. Install it "
        "with:\n    pip install piper-tts"
    )


def _frames_from_pcm(pcm: bytes, sample_rate: int = WIRE_SAMPLE_RATE) -> list[AudioFrame]:
    """Slice a PCM buffer into fixed-size AudioFrames (last one may be short)."""
    return [
        AudioFrame(pcm=pcm[i : i + _FRAME_BYTES], sample_rate=sample_rate, channels=1)
        for i in range(0, len(pcm), _FRAME_BYTES)
    ]


def _resample_int16(samples: Any, src_rate: int, dst_rate: int) -> Any:
    """Linear-interpolation resample of a 1-D int16 numpy array.

    Same deliberately-cheap approach as ``KokoroTTS._resample`` in
    ``providers/local/__init__.py`` (see that method's docstring for the
    aliasing tradeoff): plain linear interpolation, no low-pass filter,
    acceptable on a telephony-bandlimited path and free of a scipy/soxr
    dependency. Piper voices most commonly emit at 22050 Hz; some emit at
    16000 Hz already, in which case this is a no-op.
    """
    import numpy as np

    if src_rate == dst_rate:
        return samples
    duration = samples.shape[0] / float(src_rate)
    dst_count = int(duration * dst_rate)
    if dst_count <= 0:
        return samples[:0]
    src_index = np.linspace(0.0, samples.shape[0] - 1, dst_count, dtype=np.float64)
    resampled = np.interp(src_index, np.arange(samples.shape[0]), samples.astype(np.float64))
    return np.clip(resampled, -32768.0, 32767.0)


async def _buffer_sentences(
    text: AsyncIterator[str], min_chars: int, max_chars: int
) -> AsyncIterator[str]:
    """Buffer an async stream of text deltas into sentence-ish segments.

    Identical policy to KokoroTTS's ``_should_flush``: flush at the first
    sentence-ish boundary once at least ``min_chars`` have accumulated, or
    once the buffer exceeds ``max_chars`` at a word boundary. Any trailing
    text once the source iterator ends is flushed too.
    """
    buffer = ""
    async for delta in text:
        buffer += delta
        if len(buffer) < min_chars:
            continue
        if _SENTENCE_END.search(buffer) or (
            len(buffer) >= max_chars and buffer.endswith(" ")
        ):
            segment, buffer = buffer.strip(), ""
            if segment:
                yield segment
    tail = buffer.strip()
    if tail:
        yield tail


@register("tts", "piper")
class PiperTTS(TTSProvider):
    """Local neural TTS via `piper <https://github.com/OHF-Voice/piper1-gpl>`_.

    Uses the ``piper`` Python package's own API
    (``docs/API_PYTHON.md`` in that repo, checked 2026-09-18) -- never a
    subprocess/shell-out, per this task's constraint:

    .. code-block:: python

        from piper import PiperVoice
        voice = PiperVoice.load(model_path, use_cuda=False)
        for chunk in voice.synthesize(text):
            chunk.audio_int16_bytes, chunk.sample_rate, ...

    ``PiperVoice.load`` takes the path to a downloaded ``.onnx`` voice model
    (Piper looks for the matching ``<model_path>.json`` config alongside
    it); there is no built-in default model to ship, so ``model_path`` is a
    required option -- inventing a path would just produce a confusing
    "file not found" instead of this class's own clear error.

    ``voice.synthesize`` yields chunks already close to raw PCM
    (``chunk.audio_int16_bytes`` at ``chunk.sample_rate``, commonly 22050 Hz
    for most published voices). Output is resampled to the stack's 16 kHz
    wire format with :func:`_resample_int16` when the voice's native rate
    differs; some voices are trained at 16000 Hz already, in which case
    resampling is a no-op.

    Usage is ``tts_chars``: the exact character count of the segment handed
    to the model (``estimated=False`` -- it is measured, not inferred, same
    as every local provider in this package). The rate is $0: a local model
    has no per-request vendor bill, listed explicitly in
    :data:`~tring.providers.local.tts_extra.TTS_RATES` so a missing-rate
    lookup for this provider is a real bug rather than a coincidence of an
    empty card.
    """

    name = "piper"

    def __init__(
        self,
        model_path: str,
        use_cuda: bool = False,
        length_scale: float = 1.0,
        noise_scale: float = 1.0,
        noise_w_scale: float = 1.0,
        volume: float = 1.0,
        max_buffer_chars: int = 220,
        min_buffer_chars: int = 12,
        **_options: Any,
    ) -> None:
        self.model_path = model_path
        self.use_cuda = use_cuda
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w_scale = noise_w_scale
        self.volume = volume
        self.max_buffer_chars = max_buffer_chars
        self.min_buffer_chars = min_buffer_chars
        self._voice: Any | None = None

    def _load_voice(self) -> Any:
        """Import and load the Piper voice model on first use.

        Cached on the instance: loading a model's weights is not free, and
        one session may synthesize many turns.
        """
        if self._voice is None:
            try:
                from piper import PiperVoice
            except ImportError as exc:  # pragma: no cover - needs the real package
                raise _missing_piper() from exc
            self._voice = PiperVoice.load(self.model_path, use_cuda=self.use_cuda)
        return self._voice

    def _synthesize_text(self, text: str) -> bytes:
        """Blocking synthesis of one buffered segment into 16 kHz PCM bytes.

        Imports ``piper`` before ``numpy`` on purpose: a bare ``pip install
        tring`` (no extras) is missing both, and the more actionable error
        -- "install piper-tts" -- should be the one a user sees, not a
        numpy import error from deep inside the resampling step.
        """
        voice = self._load_voice()  # raises the piper-specific ImportError first
        try:
            from piper import SynthesisConfig
        except ImportError as exc:  # pragma: no cover - needs the real package
            raise _missing_piper() from exc

        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - needs the real package
            raise ImportError(
                "numpy is required to resample Piper's output to the stack's "
                "16 kHz wire format. Install it with:\n"
                '    pip install "tring[local]"'
            ) from exc

        syn_config = SynthesisConfig(
            volume=self.volume,
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
        )

        pcm_chunks: list[bytes] = []
        native_rate = WIRE_SAMPLE_RATE
        for chunk in voice.synthesize(text, syn_config=syn_config):
            native_rate = int(chunk.sample_rate)
            pcm_chunks.append(bytes(chunk.audio_int16_bytes))
        raw = b"".join(pcm_chunks)
        if not raw:
            return b""

        samples = np.frombuffer(raw, dtype="<i2")
        resampled = _resample_int16(samples, native_rate, WIRE_SAMPLE_RATE)
        pcm: bytes = resampled.astype("<i2").tobytes()
        return pcm

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        # Piper is single-voice-per-model; `voice` (a model swap, not a
        # per-call parameter) is accepted for interface compatibility but
        # intentionally not acted on here -- selecting a different Piper
        # voice means loading a different .onnx file, which is a
        # provider-construction concern (`model_path`), not a per-utterance
        # one.
        async for segment in _buffer_sentences(
            text, self.min_buffer_chars, self.max_buffer_chars
        ):
            pcm = await asyncio.to_thread(self._synthesize_text, segment)
            if not pcm:
                continue
            frames = _frames_from_pcm(pcm)
            for index, frame in enumerate(frames):
                # Usage rides on the first frame of the segment, matching
                # KokoroTTS's convention, so a consumer summing usage across
                # the stream never double-counts a segment's characters.
                usage = (
                    [
                        Usage(
                            units=float(len(segment)),
                            unit_name="tts_chars",
                            estimated=False,
                            model="piper",
                        )
                    ]
                    if index == 0
                    else []
                )
                yield TTSChunk(frame=frame, usage=usage)


# ---------------------------------------------------------------------------
# Rates -- a local model has no per-request vendor bill. Listed explicitly,
# as_of 2026-09, so a missing-rate lookup for this provider is a bug, not a
# coincidence of an empty card (same convention as cost/rates.py's
# DEFAULT_RATES for the other local providers).
# ---------------------------------------------------------------------------

TTS_RATES: list[Rate] = [
    Rate(
        component=CostComponent.TTS,
        provider="piper",
        model=None,
        unit_name="tts_chars",
        price_per_unit=0.0,
        currency="USD",
        as_of="2026-09",
    ),
]


__all__ = [
    "TTS_RATES",
    "PiperTTS",
]
