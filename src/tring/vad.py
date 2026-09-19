"""Voice activity detection — the substrate turn-taking is built on.

Every conversational decision a voice agent makes is downstream of one
question: *is the caller talking right now?* Endpointing ("they stopped, go
answer"), barge-in ("they started while we were talking, stop talking") and
utterance buffering ("here is the audio to transcribe") are three consumers of
that single signal. Tring keeps the signal in one place instead of letting each
provider grow its own private endpointer, because three endpointers with three
sets of thresholds produce three different opinions about when a turn ended,
and the caller hears all of them.

What a detector is
------------------

A :class:`VoiceActivityDetector` is a pure state machine: feed it frames in
order, get back zero or more :class:`VADEvent` transitions. No audio is
retained, no callbacks fire, nothing is emitted onto the session bus. Buffering
and policy live in :mod:`tring.runtimes.turn_taking`; this module only answers
the question.

VAD transitions are deliberately *not* :mod:`tring.events` ``SessionEvent``\\s.
A busy line produces tens of speech transitions a minute and thousands of
analysed frames; that is sub-conversational telemetry, one layer below the
vocabulary transports, cost meters and analytics consumers subscribe to. What
reaches the session bus is the turn it produced (``UserTranscript``) or the
barge-in it caused (``Interruption``), not the acoustics.

Two tiers, honestly labelled
----------------------------

:class:`EnergyVAD` is the zero-dependency fallback tier. It is a good adaptive
noise gate and it is *not* a speech detector: it cannot tell a cough, a door,
a baby or hold music from a word, because nothing in it models speech. It
exists so a core ``pip install tring`` has working turn-taking out of the box.

:class:`~tring.providers.local.vad_silero.SileroVAD` is the production tier: a
small neural model that actually classifies speech, at the cost of an ONNX
runtime. Use :func:`make_vad` to choose between them.
"""

from __future__ import annotations

import abc
import array
import enum
import math
from dataclasses import dataclass
from typing import Any, Literal

from tring.runtimes.base import AudioFrame

_BYTES_PER_SAMPLE = 2  # 16-bit linear PCM, the AudioFrame contract

#: dBFS reported for a frame of digital silence. ``20*log10(0)`` is negative
#: infinity, which poisons every average it touches, so the scale is floored at
#: a value far below any real room tone.
SILENCE_DBFS = -90.0

#: Slack allowed when comparing accumulated frame durations against a
#: millisecond threshold. Durations are summed as floats, so six 20 ms frames
#: reach 0.11999999999999998 rather than 0.12; without this, a detector
#: configured for exactly six frames of evidence would silently need seven.
_DURATION_EPSILON = 1e-9


class VADEventKind(enum.StrEnum):
    """Which boundary a :class:`VADEvent` marks.

    ``StrEnum`` rather than the ``(str, Enum)`` pair used in
    :mod:`tring.events`: those are pydantic-validated contract anchors whose
    ``str()`` formatting other modules depend on, while this one exists to name
    a transition and read cleanly in a log line.
    """

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True)
class VADEvent:
    """One speech-boundary transition on the caller's audio timeline.

    ``at`` is seconds of caller audio since the detector's first frame, and it
    names *when the transition happened*, not when the detector became sure of
    it. Those differ by design: a detector must see evidence after a boundary
    to trust it (a burst of energy that lasts, a pause that keeps lasting), so
    every event is reported some tens or hundreds of milliseconds after the
    timestamp it carries.

    Keeping the honest timestamp rather than "now" is what lets a consumer
    recover the audio on either side of the boundary — the onset of the first
    word is already in the past when ``SPEECH_START`` arrives, and clipping it
    is how a transcript loses its first syllable.
    """

    kind: VADEventKind
    at: float


def frame_seconds(frame: AudioFrame) -> float:
    """Duration of one PCM frame in seconds.

    The audio timeline every VAD and :class:`~tring.runtimes.turn_taking.TurnTaker`
    works in is derived from this function alone, never from wall-clock time.
    A detector that timestamps against ``time.monotonic()`` reports different
    boundaries depending on how bursty the transport was; one that counts
    samples reports the same boundaries for the same audio, which is what makes
    turn-taking testable with scripted buffers.
    """
    denom = frame.sample_rate * _BYTES_PER_SAMPLE * frame.channels
    return len(frame.pcm) / denom if denom else 0.0


def pcm_rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of 16-bit little-endian PCM, in 0.0 - 1.0.

    Written against the stdlib ``array`` module rather than ``audioop`` (removed
    in Python 3.13) and rather than numpy (not installed in a core install, and
    the fallback VAD must work in a core install).
    """
    if not pcm:
        return 0.0
    samples = array.array("h")
    # Trim a trailing odd byte rather than raising: a transport that splits a
    # sample across two frames is a framing bug, but dropping half a sample is
    # inaudible, whereas a crash mid-call is not.
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    total = sum(float(s) * float(s) for s in samples)
    return math.sqrt(total / len(samples)) / 32768.0


def pcm_dbfs(pcm: bytes) -> float:
    """Frame level in dBFS (0.0 = full scale), floored at :data:`SILENCE_DBFS`.

    Levels are compared in decibels, not in linear amplitude, because the whole
    point of an adaptive gate is that the *ratio* of speech to room tone is
    roughly stable across lines while the absolute levels are not. A caller on
    a cheap headset and a caller on a conference speaker differ by 30 dB of
    absolute level and by almost nothing in speech-over-noise margin.
    """
    rms = pcm_rms(pcm)
    if rms <= 0.0:
        return SILENCE_DBFS
    return max(SILENCE_DBFS, 20.0 * math.log10(rms))


class VoiceActivityDetector(abc.ABC):
    """Frames in, speech-boundary transitions out.

    Implementations are stateful and strictly sequential: frames must be fed in
    capture order, and the timeline in :attr:`VADEvent.at` accumulates from the
    first frame ever fed. There is no threading contract — drive one detector
    from one task.
    """

    name: str = "vad"

    @abc.abstractmethod
    def feed(self, frame: AudioFrame) -> list[VADEvent]:
        """Analyse one frame and return any boundaries it resolved.

        Returns a list rather than an optional event because a single frame can
        close one utterance and be unable to open the next in the same call;
        returning a list keeps the caller's loop identical in every case.
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Forget speech state, keeping the audio clock.

        For a call that was put on hold, transferred, or otherwise had its
        audio path swapped underneath the detector: mid-utterance state is
        meaningless afterwards, but :attr:`VADEvent.at` must stay comparable
        with the timeline a consumer has been buffering against, so the clock
        does not rewind.
        """


class EnergyVAD(VoiceActivityDetector):
    """Adaptive-threshold energy gate. The zero-dependency fallback tier.

    **What it actually detects: loudness, not speech.** A frame counts as
    speech when its level sits ``speech_threshold_db`` above a rolling estimate
    of the line's noise floor. That is enough for a quiet caller on a decent
    line and it is straightforwardly fooled by anything else with energy in it:
    keyboard noise, a passing siren, hold music, a television, a cough. Nothing
    here models speech, so nothing here can reject non-speech. The production
    answer is :class:`~tring.providers.local.vad_silero.SileroVAD`; this class
    exists so that turn-taking works before anyone has installed an ML stack.

    Three mechanisms do the work, and each one is a specific failure it is
    there to prevent:

    *Adaptive noise floor.* A fixed threshold is wrong on the first call it
    meets, because line noise varies by tens of decibels between a mobile in a
    car and a softphone in an office. The floor is re-estimated from frames
    classified as non-speech, and it is *never* updated while speech is in
    progress — a floor that tracked speech would climb until the gate went deaf
    halfway through a long sentence. Adaptation is asymmetric on purpose: a
    line that got quieter is believed quickly (``noise_adapt_down``), a line
    that got louder is believed slowly (``noise_adapt_up``), because the usual
    reason for a sudden rise is somebody talking.

    *Minimum speech duration.* A single loud frame is a click, not a word.
    ``min_speech_ms`` of sustained energy is required before a turn opens,
    which is also why ``SPEECH_START`` is timestamped at the *beginning* of
    that run rather than at the frame that confirmed it.

    *Hangover.* Ordinary speech is full of short gaps: stops, breaths, the
    pause before a digit. Closing a turn at the first quiet frame chops
    "four... five... six" into three utterances. The turn stays open until
    ``hangover_ms`` of continuous quiet has passed, which merges bursts closer
    than that into one utterance. ``SPEECH_END`` is then timestamped at the
    start of that quiet run — the true end of voiced audio, not the end of the
    hangover.

    **The failure this design cannot avoid**, stated plainly because a caller
    will hit it: on a line already loud enough at the first frame to clear the
    seeded gate — a car, a factory floor, hold music — the gate opens and stays
    open, because the floor is not updated during speech and so never learns
    that the "speech" is the line itself. ``max_speech_ms`` bounds the damage
    rather than removing it: a turn nobody could plausibly have spoken is
    closed, and the floor is re-seeded from the *quietest* moment inside it, on
    the reasoning that real speech dips between words within twenty seconds and
    steady noise does not. It recovers the gate within one spurious turn. It
    does not make this a speech detector, and nothing short of a model will.

    Args:
        speech_threshold_db: how far above the estimated noise floor a frame
            must sit to open a turn. 10 dB is a little over three times the
            amplitude of the room tone: high enough to ignore breathing and
            line hiss, low enough for a soft speaker.
        release_margin_db: hysteresis. Once a turn is open, frames are held to
            a threshold this much lower, so speech hovering right at the gate
            does not chatter the state machine open and shut.
        hangover_ms: continuous quiet required to close a turn. 300 ms is the
            usual trade: long enough to survive an inter-word stop, short
            enough that the caller does not feel the agent hesitating.
        min_speech_ms: sustained energy required to open a turn.
        max_speech_ms: how long a turn may stay open before it is treated as a
            latched gate rather than a person (see above). Twenty seconds is
            past any plausible unbroken utterance; a genuine monologue that
            reaches it is split into two turns rather than lost. Set to 0 to
            disable, and accept that a latched gate then stays latched.
        noise_adapt_down: smoothing factor applied when the observed level is
            below the current floor estimate (fast: we over-estimated).
        noise_adapt_up: smoothing factor when it is above (slow: might be
            speech we have not classified yet).
        initial_noise_floor_db: ceiling for the floor estimate seeded from the
            very first frame. A call that opens with the caller already
            mid-sentence must not calibrate its idea of "silence" from speech.
        min_noise_floor_db: floor for the floor estimate. A digitally silent
            line reads as :data:`SILENCE_DBFS`, and taking that literally would
            leave the gate triggering on its own quantisation noise.
    """

    name = "energy"

    def __init__(
        self,
        speech_threshold_db: float = 10.0,
        release_margin_db: float = 3.0,
        hangover_ms: int = 300,
        min_speech_ms: int = 120,
        max_speech_ms: int = 20000,
        noise_adapt_down: float = 0.25,
        noise_adapt_up: float = 0.02,
        initial_noise_floor_db: float = -50.0,
        min_noise_floor_db: float = -65.0,
        **_options: Any,
    ) -> None:
        self.speech_threshold_db = speech_threshold_db
        self.release_margin_db = release_margin_db
        self.hangover_ms = hangover_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.noise_adapt_down = noise_adapt_down
        self.noise_adapt_up = noise_adapt_up
        self.initial_noise_floor_db = initial_noise_floor_db
        self.min_noise_floor_db = min_noise_floor_db

        self._elapsed = 0.0
        self._noise_floor_db: float | None = None
        self._speaking = False
        self._run_from: float | None = None
        self._run_seconds = 0.0
        self._quiet_from: float | None = None
        self._quiet_seconds = 0.0
        self._speech_seconds = 0.0
        self._speech_min_db = math.inf

    @property
    def speaking(self) -> bool:
        """Whether a turn is currently open."""
        return self._speaking

    @property
    def noise_floor_db(self) -> float | None:
        """Current noise-floor estimate, or ``None`` before the first frame.

        Exposed because it is the number to look at when this VAD misbehaves:
        a floor that has drifted up to within a few dB of speech explains every
        missed turn, and a floor stuck at the minimum explains every false one.
        """
        return self._noise_floor_db

    def reset(self) -> None:
        self._noise_floor_db = None
        self._speaking = False
        self._run_from = None
        self._run_seconds = 0.0
        self._quiet_from = None
        self._quiet_seconds = 0.0
        self._speech_seconds = 0.0
        self._speech_min_db = math.inf

    def feed(self, frame: AudioFrame) -> list[VADEvent]:
        duration = frame_seconds(frame)
        if duration <= 0.0:
            return []
        started_at = self._elapsed
        self._elapsed = started_at + duration

        level = pcm_dbfs(frame.pcm)
        floor = self._seeded_floor(level)
        gate = floor + self.speech_threshold_db
        if self._speaking:
            gate -= self.release_margin_db
        loud = level > gate

        if self._speaking:
            return self._feed_while_speaking(loud, level, started_at, duration)
        return self._feed_while_quiet(loud, level, floor, started_at, duration)

    # ------------------------------------------------------------- internals

    def _feed_while_speaking(
        self, loud: bool, level: float, started_at: float, duration: float
    ) -> list[VADEvent]:
        """Advance an open turn; close it on hangover, or on the safety valve."""
        self._speech_seconds += duration
        self._speech_min_db = min(self._speech_min_db, level)

        if loud:
            # Speech resumed inside the hangover window: the gap was a pause
            # within one utterance, so the pending close is discarded and the
            # two bursts merge.
            self._quiet_from = None
            self._quiet_seconds = 0.0
            return self._latch_guard()

        if self._quiet_from is None:
            self._quiet_from = started_at
        self._quiet_seconds += duration
        if self._quiet_seconds + _DURATION_EPSILON < self.hangover_ms / 1000.0:
            return self._latch_guard()

        return self._close_turn(self._quiet_from)

    def _latch_guard(self) -> list[VADEvent]:
        """Close a turn that has outlasted any plausible utterance.

        The floor is re-seeded from the quietest frame of the turn just closed,
        which is the one measurement that separates the two things that produce
        a very long turn: a person, whose quietest moment over twenty seconds
        is a between-words gap near the true floor, and a noisy line, whose
        quietest moment is the noise itself.
        """
        if self.max_speech_ms <= 0:
            return []
        if self._speech_seconds + _DURATION_EPSILON < self.max_speech_ms / 1000.0:
            return []
        self._noise_floor_db = max(self._speech_min_db, self.min_noise_floor_db)
        return self._close_turn(self._elapsed)

    def _close_turn(self, end_at: float) -> list[VADEvent]:
        self._speaking = False
        self._quiet_from = None
        self._quiet_seconds = 0.0
        self._speech_seconds = 0.0
        self._speech_min_db = math.inf
        return [VADEvent(kind=VADEventKind.SPEECH_END, at=end_at)]

    def _feed_while_quiet(
        self, loud: bool, level: float, floor: float, started_at: float, duration: float
    ) -> list[VADEvent]:
        """Accumulate evidence for a new turn, or keep calibrating the floor."""
        if not loud:
            self._run_from = None
            self._run_seconds = 0.0
            self._noise_floor_db = self._adapted_floor(floor, level)
            return []

        if self._run_from is None:
            self._run_from = started_at
            self._run_seconds = 0.0
        self._run_seconds += duration
        if self._run_seconds + _DURATION_EPSILON < self.min_speech_ms / 1000.0:
            return []

        start_at = self._run_from
        self._speaking = True
        self._run_from = None
        self._run_seconds = 0.0
        self._quiet_from = None
        self._quiet_seconds = 0.0
        self._speech_seconds = 0.0
        self._speech_min_db = level
        return [VADEvent(kind=VADEventKind.SPEECH_START, at=start_at)]

    def _seeded_floor(self, level: float) -> float:
        """Return the floor estimate, seeding it from the first frame if needed."""
        if self._noise_floor_db is None:
            seed = min(level, self.initial_noise_floor_db)
            self._noise_floor_db = max(seed, self.min_noise_floor_db)
        return self._noise_floor_db

    def _adapted_floor(self, floor: float, level: float) -> float:
        alpha = self.noise_adapt_down if level < floor else self.noise_adapt_up
        return max((1.0 - alpha) * floor + alpha * level, self.min_noise_floor_db)


#: The detectors :func:`make_vad` knows how to build.
VADKind = Literal["energy", "silero"]


def make_vad(kind: VADKind = "energy", **options: Any) -> VoiceActivityDetector:
    """Build a detector by name.

    VAD is deliberately absent from :mod:`tring.providers.registry`. The
    registry maps *pipeline slots* — one STT, one LLM, one TTS, one S2S — that a
    runtime binds from an ``AgentSpec``'s routing table. A detector is not a
    slot: an S2S runtime has none at all (the model does its own turn-taking),
    a cascade runtime has one shared by endpointing and barge-in, and no spec
    ever needs to route one per language. Registering it would advertise a
    swappability the contract does not actually have.

    ``kind`` is typed as a literal so a mistyped name is a type error rather
    than a runtime surprise, but it is validated at runtime too: the usual
    caller is a YAML config, which no type checker has ever seen.

    Args:
        kind: ``"energy"`` for the dependency-free gate, ``"silero"`` for the
            neural model (needs ``pip install "tring[local]"``).
        options: forwarded to the detector's constructor. Unknown keys are
            absorbed, so one shared config block can carry knobs for both.
    """
    if kind == "energy":
        return EnergyVAD(**options)
    if kind == "silero":
        # Imported here, not at module scope: this module must stay importable
        # in a core install, and the silero module reaches for an ONNX runtime.
        from tring.providers.local.vad_silero import SileroVAD

        return SileroVAD(**options)
    raise ValueError(f"unknown vad kind {kind!r}; available: ['energy', 'silero']")


__all__ = [
    "SILENCE_DBFS",
    "EnergyVAD",
    "VADEvent",
    "VADEventKind",
    "VADKind",
    "VoiceActivityDetector",
    "frame_seconds",
    "make_vad",
    "pcm_dbfs",
    "pcm_rms",
]
