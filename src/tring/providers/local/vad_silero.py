"""Silero VAD — the production voice-activity tier, on ONNX Runtime.

Unlike :class:`~tring.vad.EnergyVAD`, this one actually classifies speech: a
small recurrent model (about 2 MB, a fraction of a CPU core in real time)
returns a per-window speech probability, so a cough, a door, hold music and a
keyboard are rejected on their *timbre* rather than passed through because they
happened to be loud. On any real line that difference is the difference between
an agent that takes turns and one that interrupts itself.

House rules, same as the rest of ``providers/local``: importing this module
must stay cheap and must never fail, so ``onnxruntime`` and ``numpy`` are
imported inside the method that needs them and a missing install is re-raised
as an :class:`ImportError` naming the extra. Nothing here registers with
:mod:`tring.providers.registry` — VAD is not a pipeline slot; build a detector
with :func:`tring.vad.make_vad` (``make_vad("silero", ...)``).

Where the model comes from
--------------------------

Two options, in the order this class tries them:

1. ``model_path=...`` — any ``.onnx`` file you manage yourself (a pinned copy
   in your image, a mounted volume, an S3-synced cache).
2. The ``silero-vad`` pip package, which ships the weights as package data in
   ``silero_vad/data/silero_vad.onnx``. ``tring[local]`` does not pull it in,
   because most of that extra is a Whisper/Kokoro stack that a VAD-only
   deployment has no use for: install ``pip install silero-vad`` alongside.

The wire protocol below is not guessed. Every shape, name and constant is taken
from silero-vad's own ONNX wrapper and iterator, read on 2026-09-19:

* https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py
  (``OnnxWrapper`` — window sizes, context, state tensor, input names, outputs;
  ``VADIterator`` — the start/stop hysteresis)
* https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/model.py
  (``load_silero_vad`` — package-data path and file name)
"""

from __future__ import annotations

import array
import os
from typing import Any

from tring.runtimes.base import AudioFrame
from tring.vad import VADEvent, VADEventKind, VoiceActivityDetector

#: Samples per inference window, per sample rate. From ``OnnxWrapper.__call__``:
#: ``num_samples = 512 if sr == 16000 else 256``. These are not a tunable
#: buffer size — the model is exported for exactly these lengths and raises
#: "Provided number of samples is N (Supported values: 256 for 8000 sample
#: rate, 512 for 16000)" for anything else. A 16 kHz window is therefore
#: exactly 32 ms, which sets this detector's timing resolution.
_WINDOW_SAMPLES = {16000: 512, 8000: 256}

#: Samples of the *previous* window prepended to each input, per sample rate.
#: From ``OnnxWrapper.__call__``: ``context_size = 64 if sr == 16000 else 32``,
#: with ``x = torch.cat([self._context, x], dim=1)`` before inference and
#: ``self._context = x[..., -context_size:]`` after. The model's first
#: convolution needs the samples immediately preceding the window; without
#: them every window is judged as if speech began at its own first sample,
#: which costs accuracy exactly at the boundaries that matter here.
_CONTEXT_SAMPLES = {16000: 64, 8000: 32}

#: Shape of the recurrent state carried between windows, from
#: ``OnnxWrapper.reset_states``: ``torch.zeros((2, batch_size, 128)).float()``.
#: Batch is always 1 here: this is a live single-caller stream.
_STATE_SHAPE = (2, 1, 128)

#: Package-data location of the weights shipped by ``pip install silero-vad``,
#: from ``load_silero_vad``: the ``silero_vad.data`` package, file name
#: ``silero_vad.onnx`` for the default opset-16 export.
_MODEL_PACKAGE = "silero_vad.data"
_MODEL_FILENAME = "silero_vad.onnx"

_LOCAL_EXTRA = 'pip install "tring[local]"'
_MODEL_PACKAGE_HINT = "pip install silero-vad"

#: Full-scale value for 16-bit PCM. The model consumes floats in [-1, 1), the
#: same normalisation every silero example uses.
_INT16_FULL_SCALE = 32768.0


def _missing_runtime() -> ImportError:
    """The error raised when the ONNX stack is absent.

    Names both packages because they arrive together — ``onnxruntime`` depends
    on ``numpy`` — and names the fallback, because a user who only wanted
    turn-taking should not have to install an ML stack to get it.
    """
    return ImportError(
        "onnxruntime and numpy are required to run Silero VAD. Install the "
        f"local stack with:\n    {_LOCAL_EXTRA}\n"
        "Or use the dependency-free fallback: tring.vad.make_vad('energy')"
    )


def _missing_model(path: str, from_package: bool) -> FileNotFoundError:
    """The error raised when the weights cannot be found on disk.

    Raised in place of letting ``onnxruntime`` fail, whose message for a
    missing file names an ONNX loader internal and neither of the two things a
    user can actually do about it.
    """
    if from_package:
        return FileNotFoundError(
            f"Silero VAD model not found at {path!r}. Install the weights with:"
            f"\n    {_MODEL_PACKAGE_HINT}\n"
            "Or point SileroVAD(model_path=...) at your own .onnx export."
        )
    return FileNotFoundError(
        f"Silero VAD model not found at the model_path you supplied: {path!r}"
    )


class SileroVAD(VoiceActivityDetector):
    """Neural voice activity detection via the silero-vad ONNX model.

    **Re-framing is this class's real job.** A transport delivers whatever
    frame size its stack likes — 10 ms from a WebRTC track, 20 ms from a
    telephony bridge, a whole second from a file replay — and the model accepts
    exactly 512 samples at 16 kHz (256 at 8 kHz) and rejects everything else.
    So incoming PCM is appended to a pending buffer and drained one exact
    window at a time; a partial window is kept for the next frame rather than
    padded, because zero-padding a window would feed the model 20 ms of
    synthetic silence and teach it that every frame boundary is a pause.

    **Hysteresis, not a single threshold.** A turn opens when the probability
    reaches ``threshold`` and closes only below ``threshold - 0.15`` — silero's
    own ``VADIterator`` margin. One threshold with the probability wobbling
    across it produces a burst of open/close transitions per second, each of
    which a runtime would faithfully turn into a barge-in.

    **``min_silence_ms`` diverges from the upstream default on purpose.**
    ``VADIterator`` defaults to 100 ms, which is right for its job (cutting
    speech regions out of a recording, where an over-long region costs nothing
    and a boundary 200 ms late is an error). It is wrong for a live turn-taker:
    100 ms of quiet is a comma, and an agent that answers at every comma talks
    over its caller all call. The default here is 300 ms, matching
    :class:`~tring.vad.EnergyVAD`'s hangover so the two tiers feel the same.

    Args:
        threshold: speech probability at or above which a turn opens.
        neg_threshold_margin: how far below ``threshold`` the probability must
            fall for a turn to start closing. Silero's own value is 0.15.
        min_silence_ms: continuous below-threshold audio required to close a
            turn (see above).
        speech_pad_ms: padding applied to reported boundaries — starts are
            timestamped this much earlier and ends this much later, so a
            consumer slicing audio at those timestamps keeps the onset of the
            first word and the tail of the last. Matches ``VADIterator``'s
            ``speech_pad_ms`` default of 30 ms.
        sample_rate: 16000 or 8000, the only rates the model supports. Frames
            at any other rate are rejected rather than resampled: resampling
            belongs in the transport, which knows the real source format.
        model_path: an explicit ``.onnx`` file. Defaults to the copy shipped by
            the ``silero-vad`` pip package.
        force_cpu: pin the session to ``CPUExecutionProvider``. On by default
            and it should stay on: this model is microseconds of CPU per
            window, and shipping those windows to a GPU costs more in transfer
            latency than the inference itself, on a device the LLM wants.
    """

    name = "silero"

    def __init__(
        self,
        threshold: float = 0.5,
        neg_threshold_margin: float = 0.15,
        min_silence_ms: int = 300,
        speech_pad_ms: int = 30,
        sample_rate: int = 16000,
        model_path: str | None = None,
        force_cpu: bool = True,
        **_options: Any,
    ) -> None:
        if sample_rate not in _WINDOW_SAMPLES:
            raise ValueError(
                f"silero VAD supports sample rates {sorted(_WINDOW_SAMPLES)}, "
                f"got {sample_rate}"
            )
        self.threshold = threshold
        self.neg_threshold_margin = neg_threshold_margin
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.sample_rate = sample_rate
        self.model_path = model_path
        self.force_cpu = force_cpu

        self.window_samples = _WINDOW_SAMPLES[sample_rate]
        self.context_samples = _CONTEXT_SAMPLES[sample_rate]
        self._window_seconds = self.window_samples / sample_rate

        self._pending = bytearray()
        self._context = [0.0] * self.context_samples
        self._position = 0.0  # seconds of audio *analysed* so far
        self._speaking = False
        self._quiet_from: float | None = None

        self._session: Any | None = None
        self._np: Any | None = None
        self._state: Any | None = None

    @property
    def speaking(self) -> bool:
        """Whether a turn is currently open."""
        return self._speaking

    def reset(self) -> None:
        """Drop speech state, the pending partial window and the model's memory.

        The recurrent state and the sample context are part of "what the model
        believes it just heard", so a reset that kept them would carry the old
        audio path's tail into the new one. The analysed-position clock is
        preserved, per the :class:`~tring.vad.VoiceActivityDetector` contract.
        """
        self._pending.clear()
        self._context = [0.0] * self.context_samples
        self._speaking = False
        self._quiet_from = None
        if self._np is not None:
            self._state = self._np.zeros(_STATE_SHAPE, dtype=self._np.float32)

    def feed(self, frame: AudioFrame) -> list[VADEvent]:
        if frame.sample_rate != self.sample_rate or frame.channels != 1:
            raise ValueError(
                f"silero VAD is configured for {self.sample_rate} Hz mono; got "
                f"{frame.sample_rate} Hz / {frame.channels} channel(s). Resample "
                "in the transport, which knows the real source format."
            )
        # Loaded before any window exists so a missing ML stack surfaces on the
        # first frame of the call rather than 32 ms into it, next to the code
        # that chose this detector rather than deep inside a turn.
        np_, session = self._runtime()

        self._pending.extend(frame.pcm)
        window_bytes = self.window_samples * 2

        events: list[VADEvent] = []
        while len(self._pending) >= window_bytes:
            window = array.array("h")
            window.frombytes(bytes(self._pending[:window_bytes]))
            del self._pending[:window_bytes]
            events.extend(self._consume_window(np_, session, window))
        return events

    # ------------------------------------------------------------- internals

    def _consume_window(
        self, np_: Any, session: Any, window: array.array[int]
    ) -> list[VADEvent]:
        """Score one exact-size window and advance the start/stop state machine."""
        window_start = self._position
        self._position = window_start + self._window_seconds
        probability = self._probability(np_, session, window)
        pad = self.speech_pad_ms / 1000.0

        if probability >= self.threshold:
            # Any confident window cancels a pending close, which is what makes
            # a mid-sentence pause a pause rather than two utterances.
            self._quiet_from = None
            if not self._speaking:
                self._speaking = True
                return [
                    VADEvent(
                        kind=VADEventKind.SPEECH_START,
                        at=max(0.0, window_start - pad),
                    )
                ]
            return []

        if not self._speaking or probability >= self.threshold - self.neg_threshold_margin:
            return []

        if self._quiet_from is None:
            self._quiet_from = window_start
        # Silence is measured from the *start* of the first quiet window;
        # VADIterator measures from its end, making it one window (32 ms at
        # 16 kHz) more conservative. Measuring from the start is what the audio
        # actually did, and this detector's whole purpose is deciding when the
        # caller stopped.
        if self._position - self._quiet_from + 1e-9 < self.min_silence_ms / 1000.0:
            return []

        end_at = min(self._quiet_from + pad, self._position)
        self._speaking = False
        self._quiet_from = None
        return [VADEvent(kind=VADEventKind.SPEECH_END, at=end_at)]

    def _probability(self, np_: Any, session: Any, window: array.array[int]) -> float:
        """Run one window through the model and return its speech probability.

        The feed dict is the one ``OnnxWrapper`` builds: ``input`` is the
        context-prefixed window as float32 ``(1, context + window)``, ``state``
        is the recurrent state threaded from the previous call, and ``sr`` is a
        scalar int64 array. The model returns ``[probability, new_state]``, and
        both the state and the trailing samples of this window are carried
        forward — a VAD that dropped either would restart the model's memory
        every 32 ms.
        """
        samples = [sample / _INT16_FULL_SCALE for sample in window]
        feed = {
            "input": np_.array([self._context + samples], dtype=np_.float32),
            "state": self._state,
            "sr": np_.array(self.sample_rate, dtype="int64"),
        }
        probability, self._state = session.run(None, feed)
        self._context = samples[-self.context_samples :]
        return float(probability[0][0])

    def _runtime(self) -> tuple[Any, Any]:
        """Import the ONNX stack and build the session, once per detector.

        Cached on the instance: session construction reads and optimises the
        graph, which is milliseconds a live call should spend exactly once.
        """
        if self._session is None:
            try:
                import numpy as np
                import onnxruntime as ort
            except ImportError as exc:
                raise _missing_runtime() from exc

            # Single-threaded on purpose, as in silero's own wrapper: the model
            # is far too small to amortise a thread pool, and a VAD that
            # grabbed cores would take them from the STT and TTS engines
            # sharing this box.
            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            providers = ["CPUExecutionProvider"] if self.force_cpu else None

            self._session = ort.InferenceSession(
                self._model_file(), sess_options=options, providers=providers
            )
            self._np = np
            self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        return self._np, self._session

    def _model_file(self) -> str:
        """Resolve the ``.onnx`` weights, explicit path first."""
        if self.model_path:
            if not os.path.exists(self.model_path):
                raise _missing_model(self.model_path, from_package=False)
            return self.model_path

        from importlib import resources

        try:
            path = str(resources.files(_MODEL_PACKAGE).joinpath(_MODEL_FILENAME))
        except (ImportError, TypeError) as exc:
            # ModuleNotFoundError (a subclass of ImportError) when the package
            # is absent; TypeError when something named silero_vad.data exists
            # but is not an importable package.
            raise _missing_model(
                f"{_MODEL_PACKAGE}/{_MODEL_FILENAME}", from_package=True
            ) from exc
        if not os.path.exists(path):
            raise _missing_model(path, from_package=True)
        return path


__all__ = ["SileroVAD"]
