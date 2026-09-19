"""TurnTaker — one VAD, one ledger, and the two decisions a turn is made of.

A voice runtime has to answer two questions about the caller's audio, and they
are the same question asked at two edges of the same signal:

*When did they stop?* Everything after that boundary is the utterance to
transcribe. Batch ASR (Whisper and every model like it) cannot answer it — it
consumes a finished recording — so somebody upstream has to cut one. Today each
provider cuts its own, with its own thresholds, which means the same pause ends
the turn at different moments depending on which STT engine is routed. This
class is where that cut moves to, so swapping STT engines stops changing the
feel of the conversation.

*When did they start, and were we talking?* A caller who starts speaking while
the bot is mid-sentence has barged in; one who starts after it finished is
simply taking their turn. Telling those apart is not a VAD decision — the audio
is identical — it is a playback-state decision, so it is delegated whole to
:class:`~tring.primitives.interruption.PlaybackLedger`, grace window and all.
``TurnTaker`` contributes the trigger and nothing else.

Buffering: nothing is clipped
-----------------------------

Every VAD reports a boundary later than it happened (evidence accumulates after
the fact), so a buffer that started recording when ``SPEECH_START`` *arrived*
would be missing the beginning of the first word — the single most common way a
transcript loses "no" from "no, cancel it". ``TurnTaker`` keeps a rolling
pre-roll of recent frames while idle and rewinds into it using
:attr:`~tring.vad.VADEvent.at`, so the utterance handed to STT begins where the
caller actually did.

The tail is kept deliberately: the buffer runs to the frame that resolved
``SPEECH_END``, which includes the VAD's hangover. Batch ASR transcribes a
clause with a little trailing silence noticeably better than one truncated at
the last voiced sample, and ``SPEECH_END``'s timestamp is still on the
utterance for any consumer that wants to trim.

Standalone use
--------------

``TurnTaker`` imports no runtime and owns no session. With ``ledger=None`` it
is a plain endpointer usable in any pipeline::

    taker = TurnTaker(vad=make_vad("energy"), on_utterance=transcribe)
    async for frame in microphone:
        await taker.push(frame)
    await taker.flush()

Adopting it in a cascade runtime
--------------------------------

``CascadeRuntime`` already exposes the one piece of shared state this needs::

    taker = TurnTaker(
        vad=make_vad("silero"),
        ledger=runtime.ledger,
        on_utterance=lambda frames: runtime.push_audio(concat_frames(frames)),
        on_interrupt=on_interrupt,
    )
    transport.on_caller_audio = taker.push   # instead of runtime.push_audio

Two integration details are worth stating plainly, because both are silent
failures rather than exceptions:

* The STT provider behind ``push_audio`` now receives one pre-segmented
  utterance per frame. Its own endpointer must be disabled (for
  ``FasterWhisperSTT``, ``silence_seconds=0``) or it will re-cut the cut audio.
* ``on_interrupt`` is handed the verdict, and injecting
  ``verdict.context_annotation`` into the next LLM turn is *its* job. By the
  time the runtime's own ``caller_started_speaking()`` runs on the final
  transcript, this class has already adjudicated and retired the utterance, so
  the runtime will correctly see ordinary turn-taking and produce no
  annotation of its own. Exactly one caller must do it; that caller is the
  interrupt handler.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from tring.primitives.interruption import InterruptionVerdict, PlaybackLedger
from tring.runtimes.base import AudioFrame
from tring.vad import VADEvent, VADEventKind, VoiceActivityDetector, frame_seconds

#: Called with the frames of one complete caller utterance, in order.
UtteranceCallback = Callable[[list[AudioFrame]], Awaitable[None]]

#: Called only for a *genuine* barge-in, with the ledger's verdict.
InterruptCallback = Callable[[InterruptionVerdict], Awaitable[None]]


def concat_frames(frames: Sequence[AudioFrame]) -> AudioFrame | None:
    """Join frames into one, or ``None`` if there is nothing to join.

    The first thing almost every ``on_utterance`` handler does, because batch
    ASR wants one buffer rather than a list. Format is taken from the first
    frame; frames that disagree are concatenated anyway, on the grounds that a
    transport mixing formats mid-utterance is a bug worth hearing rather than
    one worth hiding behind a silent drop.
    """
    if not frames:
        return None
    head = frames[0]
    return AudioFrame(
        pcm=b"".join(frame.pcm for frame in frames),
        sample_rate=head.sample_rate,
        channels=head.channels,
    )


@dataclass(frozen=True)
class _TimedFrame:
    """One frame with its span on the caller's audio timeline."""

    start: float
    end: float
    frame: AudioFrame


class TurnTaker:
    """Binds a VAD and a playback ledger into caller turns and barge-ins.

    Args:
        vad: the detector to drive. Its timeline and this class's are both
            derived from :func:`~tring.vad.frame_seconds` over the same frames,
            so :attr:`~tring.vad.VADEvent.at` indexes directly into the buffer
            here; that shared derivation is why no clock synchronisation is
            needed between them.
        ledger: the playback ledger to adjudicate barge-ins against. ``None``
            makes this a pure endpointer: with no notion of what the bot is
            saying there is nothing to interrupt, and ``on_interrupt`` never
            fires.
        on_utterance: awaited at each ``SPEECH_END`` with the buffered frames.
        on_interrupt: awaited at each ``SPEECH_START`` that the ledger judges a
            genuine barge-in. Not called for ordinary turn-taking, and not
            called inside the ledger's grace window — a bot three characters
            from the end of its sentence has effectively finished, and
            cancelling its TTS there would clip a goodbye for nothing.
        preroll_ms: how much audio to keep while idle, so a ``SPEECH_START``
            timestamped in the past can still be honoured. Must exceed the
            VAD's detection lag (``min_speech_ms`` for the energy tier, one
            window plus ``speech_pad_ms`` for silero); the default has room for
            both.
        max_utterance_seconds: safety valve. A VAD held open by sustained noise
            would otherwise buffer the whole call into memory. On reaching this
            length the buffer is delivered as an utterance and collection
            continues, so the audio is transcribed in pieces rather than lost
            or hoarded. Set to 0 to disable.
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        ledger: PlaybackLedger | None = None,
        on_utterance: UtteranceCallback | None = None,
        on_interrupt: InterruptCallback | None = None,
        preroll_ms: int = 400,
        max_utterance_seconds: float = 30.0,
    ) -> None:
        self.vad = vad
        self.ledger = ledger
        self.on_utterance = on_utterance
        self.on_interrupt = on_interrupt
        self.preroll_ms = preroll_ms
        self.max_utterance_seconds = max_utterance_seconds

        self._buffer: list[_TimedFrame] = []
        self._timeline = 0.0
        self._speaking = False
        self._utterance_from = 0.0

    @property
    def speaking(self) -> bool:
        """Whether a caller utterance is currently open."""
        return self._speaking

    @property
    def elapsed(self) -> float:
        """Seconds of caller audio pushed so far."""
        return self._timeline

    @property
    def buffered_seconds(self) -> float:
        """Audio currently held, in seconds.

        The number to watch in production: while idle it should sit around
        ``preroll_ms``, and a process whose turn-takers drift above that is one
        whose VAD is latched open, which is a far more useful signal than the
        memory graph it eventually shows up on.
        """
        if not self._buffer:
            return 0.0
        return self._buffer[-1].end - self._buffer[0].start

    async def push(self, frame: AudioFrame) -> None:
        """Feed one caller frame: buffer it, then act on what the VAD says.

        The frame is buffered *before* the VAD sees it, because a boundary this
        frame resolves may name a moment inside it — a ``SPEECH_END`` whose
        audio has not been stored yet cannot be delivered.
        """
        duration = frame_seconds(frame)
        if duration <= 0.0:
            return
        start = self._timeline
        self._timeline = start + duration
        self._buffer.append(_TimedFrame(start=start, end=self._timeline, frame=frame))

        for event in self.vad.feed(frame):
            await self._apply(event)

        if self._speaking:
            await self._guard_max_length()
        else:
            self._trim_preroll()

    async def flush(self) -> None:
        """Deliver any in-progress utterance. Call this when the call ends.

        A caller who is cut off mid-sentence still said something, and the
        alternative to delivering it is dropping their last words on the floor
        — which is most of a hang-up's information content when the last thing
        they said was why they hung up.
        """
        if self._speaking:
            await self._deliver()
        self._speaking = False
        self._buffer.clear()

    # ------------------------------------------------------------- internals

    async def _apply(self, event: VADEvent) -> None:
        if event.kind is VADEventKind.SPEECH_START:
            await self._begin(event.at)
        else:
            await self._end()

    async def _begin(self, at: float) -> None:
        """Open an utterance at ``at``, rewinding into the pre-roll buffer."""
        if self._speaking:
            return  # a detector that repeats itself must not restart the buffer
        self._speaking = True
        self._utterance_from = at
        # Keep every frame that overlaps the boundary or follows it. Dropping
        # only frames that ended *before* the onset means the frame containing
        # the first syllable is kept whole.
        self._buffer = [timed for timed in self._buffer if timed.end > at]

        if self.ledger is None:
            return
        # The ledger is the sole authority on whether this was a barge-in, and
        # the call has side effects worth having regardless of whether anyone
        # is listening: it emits the ``Interruption`` event and retires the
        # utterance it adjudicated. So it runs once per onset, callback or not.
        verdict = self.ledger.caller_started_speaking()
        if verdict.genuine and self.on_interrupt is not None:
            await self.on_interrupt(verdict)

    async def _end(self) -> None:
        if not self._speaking:
            return
        self._speaking = False
        await self._deliver()
        self._buffer.clear()

    async def _guard_max_length(self) -> None:
        """Split an utterance that has run past ``max_utterance_seconds``."""
        if self.max_utterance_seconds <= 0.0:
            return
        if self._timeline - self._utterance_from < self.max_utterance_seconds:
            return
        await self._deliver()
        self._buffer.clear()
        # Still speaking as far as the VAD is concerned: collection continues
        # into a fresh buffer and the eventual SPEECH_END delivers the rest.
        self._utterance_from = self._timeline

    async def _deliver(self) -> None:
        if self.on_utterance is None or not self._buffer:
            return
        await self.on_utterance([timed.frame for timed in self._buffer])

    def _trim_preroll(self) -> None:
        """Bound the idle buffer to ``preroll_ms`` of recent audio."""
        horizon = self._timeline - self.preroll_ms / 1000.0
        self._buffer = [timed for timed in self._buffer if timed.end > horizon]


__all__ = ["InterruptCallback", "TurnTaker", "UtteranceCallback", "concat_frames"]
