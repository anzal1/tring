"""Incremental local Whisper transcription: ``faster_whisper_streaming``.

``FasterWhisperSTT`` (``providers/local/__init__.py``) is batch-per-utterance:
it buffers audio with a hand-rolled RMS gate and only ever yields one final
``STTResult`` per utterance. That is a real, documented limitation -- nothing
downstream can start reacting until the caller has fully stopped talking.

This module trades some CPU for perceived latency: it still yields exactly
one final per utterance (same accounting contract), but it *also* re-runs the
model on the in-progress buffer every ``partial_interval_s`` seconds of
accumulated speech and yields non-final ``STTResult``s in between. A runtime
that starts drafting a response off partials (or just shows a live caption)
gets a head start; a runtime that only cares about finals can ignore
``final=False`` results entirely and this class behaves exactly like the
batch provider plus some wasted CPU.

**Segmentation is delegated, not reimplemented.** Endpointing (deciding where
one utterance ends and the next begins) is exactly what :mod:`tring.vad`
exists to own, and duplicating an RMS gate here -- the way ``FasterWhisperSTT``
does -- would just be a second, slightly different endpointer to keep in sync
with the real one. This class instead builds a detector with
``tring.vad.make_vad`` and reacts to the ``VADEvent``\\s its ``feed(frame)``
returns: a ``VADEventKind.SPEECH_START`` opens a new utterance buffer, a
``VADEventKind.SPEECH_END`` closes it. ``tring.vad`` is deliberately not a
registry slot (see that module's ``make_vad`` docstring), so the detector is
built directly rather than through ``providers.registry``.

Same house rules as the rest of ``providers/local``: heavy imports
(``faster_whisper``) are lazy and a missing install is re-raised as an
``ImportError`` naming the ``tring[local]`` extra; usage is measured, not
estimated, so every ``Usage`` here carries ``estimated=False``. ``tring.vad``
itself has no heavy dependencies (the default ``"energy"`` engine is pure
stdlib, same as this package's own RMS gate) so it is imported at module
scope rather than lazily; only its optional ``"silero"`` engine reaches for
an ONNX runtime, and that import is lazy inside :mod:`tring.vad` itself.
"""

from __future__ import annotations

import asyncio
import io
import wave
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from tring.providers.base import STTProvider, STTResult, Usage
from tring.providers.registry import register
from tring.runtimes.base import AudioFrame
from tring.vad import VADEventKind, make_vad

_LOCAL_EXTRA = 'pip install "tring[local]"'

# Canonical wire format for the whole stack (see AudioFrame's docstring).
# Duplicated rather than imported from providers/local/__init__.py: that
# module imports *this* one from its very last line (to run this file's
# ``@register`` call), so importing back from it would be a needless
# circular-import hazard for two constants -- the same tradeoff
# providers/local/tts_extra.py documents and makes the same way.
WIRE_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # 16-bit linear PCM


def _missing(package: str, why: str) -> ImportError:
    """Build the ImportError raised when an optional dependency is absent.

    Mirrors ``providers/local/__init__.py``'s ``_missing`` exactly (message
    shape, extra name); duplicated here for the same reason ``WIRE_SAMPLE_RATE``
    is duplicated above.
    """
    return ImportError(
        f"{package} is required to {why}. Install the local provider stack with:"
        f"\n    {_LOCAL_EXTRA}"
    )


def _pcm_to_wav(pcm: bytes, sample_rate: int, channels: int) -> io.BytesIO:
    """Wrap raw PCM in an in-memory WAV container for faster-whisper.

    Identical to ``providers/local/__init__.py``'s helper of the same name;
    duplicated rather than imported for the circular-import reason above.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(_BYTES_PER_SAMPLE)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    buf.seek(0)
    return buf


def _pcm_seconds(nbytes: int, sample_rate: int, channels: int) -> float:
    denom = sample_rate * _BYTES_PER_SAMPLE * channels
    return nbytes / denom if denom else 0.0


class _VoiceActivityDetectorLike(Protocol):
    """The one method this module needs from a VAD.

    Narrowing to a Protocol (rather than typing directly against
    ``tring.vad.VoiceActivityDetector``, an ABC) is what makes the VAD seam
    testable without constructing a real detector's timing state:
    ``tests/test_stt_streaming.py`` drives this class with a scripted fake
    that implements exactly this method and nothing else, on its own schedule
    rather than a real detector's energy/hangover timing. Endpointer
    *correctness* (real thresholds, real timing) is ``tring.vad``'s own test
    suite's job, not this file's -- this file only tests how
    ``FasterWhisperStreamingSTT`` reacts to whatever events a VAD produces.
    ``list[Any]`` rather than ``list[VADEvent]`` sidesteps ``list``'s
    invariance under mypy --strict: a fake's own event type only needs a
    ``.kind`` comparable to :class:`~tring.vad.VADEventKind`'s members, not to
    literally be a ``VADEvent``.
    """

    def feed(self, frame: AudioFrame) -> list[Any]: ...


#: A zero-argument constructor for a VAD instance. Matches the ``_model``
#: factory seam below in spirit: production passes ``None`` and gets
#: ``tring.vad.make_vad(self.vad_engine)``; tests pass a callable that
#: returns a scripted fake.
VADFactory = Callable[[], _VoiceActivityDetectorLike]


@register("stt", "faster_whisper_streaming")
class FasterWhisperStreamingSTT(STTProvider):
    """Local Whisper transcription with VAD-driven partial hypotheses.

    Segmentation comes entirely from the injected VAD (``vad="energy"`` by
    default, built via ``tring.vad.make_vad``): a ``SPEECH_START`` event opens
    a new utterance buffer, a ``SPEECH_END`` event closes it and runs a
    full-context transcription pass whose result is yielded as a *final*
    ``STTResult``. Between those two events, the in-progress buffer is
    re-transcribed every ``partial_interval_s`` seconds of accumulated speech
    and yielded as a *non-final* ``STTResult`` -- the same idea as Whisper
    streaming demos elsewhere in the ecosystem, implemented here as repeated
    whole-buffer batch calls rather than genuine incremental decoding,
    because faster-whisper has no incremental-decode API to hook into.

    **The cost/latency knob: ``partial_interval_s`` (default 1.0s, 0 disables
    partials).** Every partial re-transcribes the *entire* buffer accumulated
    so far, from scratch -- there is no incremental state carried between
    calls. Halving ``partial_interval_s`` roughly doubles how much CPU this
    provider burns per utterance (twice as many full-buffer passes) in
    exchange for the runtime seeing an updated hypothesis twice as often.
    Setting it to ``0`` recovers exactly ``FasterWhisperSTT``'s behaviour
    (one pass per utterance, at the ``SPEECH_END`` boundary) at zero extra
    cost. There is no "right" value: a barge-in detector that wants to react
    to the caller's words mid-utterance wants this small; a deployment that
    only ever consumes the final transcript should set it to ``0`` and not
    pay for hypotheses nobody reads.

    **``condition_on_previous_text``.** Finals run faster-whisper with its
    default (``True``): a finished utterance is decoded once, end to end, and
    letting the model condition later internal segments on earlier ones (for
    utterances long enough to span more than one internal segment) is the
    setting that keeps a single utterance's transcript internally consistent.
    Partials run with ``condition_on_previous_text=False``: each partial call
    is an independent decode of "the buffer so far", not a continuation of
    the *previous partial call*, and there is no earlier segment within that
    call for the model to condition on in the way the flag intends -- see
    `faster-whisper's WhisperModel.transcribe
    <https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py>`_
    (verified 2026-09-19: ``condition_on_previous_text: bool = True`` is a
    real keyword parameter of ``transcribe()``, and it returns
    ``(Iterable[Segment], TranscriptionInfo)`` -- the same shape
    ``FasterWhisperSTT`` already unpacks in ``providers/local/__init__.py``).

    **Usage is exact ``audio_seconds``, and only on finals.** The same audio
    is transcribed once as each partial and again as the eventual final;
    reporting ``audio_seconds`` usage on partials too would bill the same
    spoken second multiple times over for one utterance the caller only ever
    spoke once. Every partial ``STTResult`` therefore carries ``usage=[]``,
    and the honest, exact, ``estimated=False`` audio-seconds line is recorded
    exactly once per utterance, on the final -- see ARCHITECTURE.md's
    "Honest cost accounting" principle.

    Args:
        model_size: faster-whisper model size/name (``"base"``, ``"small.en"``,
            a local path to converted weights, ...).
        device: passed straight to ``WhisperModel`` (``"auto"``, ``"cpu"``,
            ``"cuda"``).
        compute_type: passed straight to ``WhisperModel`` (``"int8"``,
            ``"float16"``, ...).
        language: default language hint; overridden per call by the
            ``language`` argument to :meth:`transcribe` when that argument is
            not ``None``, since the runtime usually knows the call's routed
            language and this constructor default only matters when it does
            not say.
        vad: which ``tring.vad`` engine to build via ``make_vad``.
            ``"energy"`` (the default) is dependency-free; ``"silero"`` needs
            ``pip install silero-vad`` (see ``tring.providers.local.vad_silero``).
            An unrecognised name raises ``ValueError`` -- there is no silent
            fallback to a different endpointer than the one requested.
        partial_interval_s: see above. Must be ``>= 0``.
    """

    name = "faster_whisper_streaming"

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = None,
        vad: str = "energy",
        partial_interval_s: float = 1.0,
        *,
        _model: Callable[[], Any] | None = None,
        _vad: VADFactory | None = None,
        **_options: Any,
    ) -> None:
        if partial_interval_s < 0:
            raise ValueError(f"partial_interval_s must be >= 0, got {partial_interval_s!r}")
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.vad_engine = vad
        self.partial_interval_s = partial_interval_s

        # Test-only seams. Production leaves both None: the model is built by
        # WhisperModel(...) on first use, the VAD by make_vad(vad_engine).
        # Tests inject a factory here instead of monkeypatching an import, so
        # the seam is explicit at the call site (see class docstring's Args
        # and tests/test_stt_streaming.py).
        self._model_factory = _model
        self._vad_factory = _vad

        self._model_instance: Any | None = None
        self._vad_instance: _VoiceActivityDetectorLike | None = None

    # ------------------------------------------------------------- loading

    def _load_model(self) -> Any:
        """Import and construct the Whisper model on first use, then cache it.

        Deliberately not wrapped in ``asyncio.to_thread`` itself: it is only
        ever called from :meth:`_transcribe_blocking`, which the async
        generator below already runs in a worker thread, so wrapping again
        here would just be redundant thread-hopping.
        """
        if self._model_instance is None:
            if self._model_factory is not None:
                self._model_instance = self._model_factory()
            else:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:  # pragma: no cover - needs the real package
                    raise _missing(
                        "faster-whisper", "run local streaming Whisper transcription"
                    ) from exc
                self._model_instance = WhisperModel(
                    self.model_size, device=self.device, compute_type=self.compute_type
                )
        return self._model_instance

    def _load_vad(self) -> _VoiceActivityDetectorLike:
        """Resolve and cache the VAD instance for this provider's lifetime.

        One VAD per provider instance, not per call: a detector is a stateful
        state machine (it tracks a running noise floor and speech/hangover
        timers across frames), and ``transcribe`` is called once per
        session's whole frame stream, exactly like every other provider in
        this package. ``make_vad`` itself raises ``ValueError`` for an
        unrecognised engine name, so that validation is not duplicated here.
        """
        if self._vad_instance is None:
            if self._vad_factory is not None:
                self._vad_instance = self._vad_factory()
            else:
                self._vad_instance = make_vad(self.vad_engine)  # type: ignore[arg-type]
        return self._vad_instance

    # --------------------------------------------------------- transcribing

    def _transcribe_blocking(
        self,
        pcm: bytes,
        sample_rate: int,
        channels: int,
        language: str | None,
        condition_on_previous_text: bool,
    ) -> tuple[str, str | None]:
        """Blocking single-pass transcription of one buffer.

        Shared by both finals and partials -- they differ only in
        ``condition_on_previous_text`` (see class docstring) and in what the
        caller does with the result (final vs. non-final ``STTResult``, and
        whether usage is attached). Kept synchronous so the async generator
        can hand it to a worker thread: faster-whisper is CPU-bound C++ code
        that would otherwise stall the event loop for every partial, which in
        a voice call means dead air on *every other provider's* turn too.
        """
        model = self._load_model()
        wav = _pcm_to_wav(pcm, sample_rate, channels)
        segments, info = model.transcribe(
            wav,
            language=language,
            beam_size=1,
            condition_on_previous_text=condition_on_previous_text,
        )
        text = "".join(segment.text for segment in segments).strip()
        detected = getattr(info, "language", None) or language
        return text, detected

    async def _finalize(
        self, pcm: bytes, sample_rate: int, channels: int, language: str | None
    ) -> STTResult | None:
        if not pcm:
            return None
        duration = _pcm_seconds(len(pcm), sample_rate, channels)
        text, detected = await asyncio.to_thread(
            self._transcribe_blocking, pcm, sample_rate, channels, language, True
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
                    model=self.model_size,
                )
            ],
        )

    async def _partial(
        self, pcm: bytes, sample_rate: int, channels: int, language: str | None
    ) -> STTResult | None:
        if not pcm:
            return None
        text, detected = await asyncio.to_thread(
            self._transcribe_blocking, pcm, sample_rate, channels, language, False
        )
        if not text:
            return None
        # No usage: see the class docstring's "Usage is exact ... and only on
        # finals" note. This buffer will be transcribed again, either as a
        # later partial or as the eventual final; billing it here too would
        # double-count audio the caller only ever spoke once.
        return STTResult(text=text, final=False, language=detected, usage=[])

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]:
        vad = self._load_vad()
        effective_language = language if language is not None else self.language

        buffer = bytearray()
        sample_rate = WIRE_SAMPLE_RATE
        channels = 1
        in_speech = False
        seconds_since_partial = 0.0

        async for frame in frames:
            sample_rate = frame.sample_rate
            channels = frame.channels

            for event in vad.feed(frame):
                if event.kind == VADEventKind.SPEECH_START:
                    # A new utterance starts here. Whatever was in the buffer
                    # (there should be nothing -- the previous utterance, if
                    # any, was already flushed on its own SPEECH_END) is
                    # discarded rather than carried forward, so two
                    # utterances never bleed into one transcription.
                    buffer.clear()
                    in_speech = True
                    seconds_since_partial = 0.0
                elif event.kind == VADEventKind.SPEECH_END:
                    # Finalize on the buffer as it stood *before* this frame:
                    # the frame that tips the VAD's hangover timer over its
                    # threshold is, definitionally, a (mostly) quiet frame,
                    # not part of the utterance. It is simply dropped rather
                    # than transcribed as a fragment of the next utterance.
                    in_speech = False
                    pcm = bytes(buffer)
                    buffer.clear()
                    seconds_since_partial = 0.0
                    result = await self._finalize(
                        pcm, sample_rate, channels, effective_language
                    )
                    if result is not None:
                        yield result

            if not in_speech:
                continue

            buffer.extend(frame.pcm)
            if self.partial_interval_s <= 0:
                continue
            seconds_since_partial += _pcm_seconds(len(frame.pcm), sample_rate, channels)
            if seconds_since_partial >= self.partial_interval_s:
                seconds_since_partial = 0.0
                partial = await self._partial(
                    bytes(buffer), sample_rate, channels, effective_language
                )
                if partial is not None:
                    yield partial

        # The frame stream ended (call hung up) while an utterance was still
        # open -- no SPEECH_END ever arrived. Flush whatever is buffered
        # rather than silently dropping the caller's last words; mirrors
        # FasterWhisperSTT's identical tail-flush in providers/local/__init__.py.
        if in_speech and buffer:
            result = await self._finalize(
                bytes(buffer), sample_rate, channels, effective_language
            )
            if result is not None:
                yield result


__all__ = ["FasterWhisperStreamingSTT", "VADFactory"]
