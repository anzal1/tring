"""Interruption handling: what the caller actually heard vs what was generated.

TTS synthesizes faster than audio plays back. By the time a caller barges in,
the LLM's own context may already contain the *full* text of the bot's
current turn — including the tail the caller never heard, because playback
hadn't caught up to generation yet. Left uncorrected, the model's next turn
confidently references things it never actually said out loud ("as I just
mentioned, the deadline is Friday" — the caller only heard "as I just
mentioned, the dead...").

A second failure mode sits right next to the first: ordinary turn-taking (the
caller starts talking once the bot has finished) fires the exact same
low-level "caller started speaking" signal as a genuine barge-in. Annotating
every single turn boundary as if it might be an interruption would poison the
context with noise on the overwhelming majority of turns, which end cleanly.

``PlaybackLedger`` is the single place that reconciles "generated" against
"played" and decides, per caller-speech event, whether it was a real
interruption at all. Only genuine barge-ins get an ``Interruption`` event and
a context annotation; ordinary turn-taking passes through silently.
"""

from __future__ import annotations

from dataclasses import dataclass

from tring.events import BotSpeechPlayed, BotUtterance, Interruption
from tring.session import CallSession

#: Default grace window (characters). A caller who starts speaking this close
#: to the end of the bot's utterance is treated as taking their turn, not
#: cutting the bot off — TTS/transport playback timing is never chars-exact,
#: so treating "essentially done" as "done" avoids manufacturing interruptions
#: out of jitter.
DEFAULT_GRACE_CHARS = 5

#: How many trailing words of the *heard* text to fold into the context
#: annotation. The annotation exists to stop the model from citing unheard
#: content; it does not need the caller's entire heard history to do that, and
#: a short tail keeps the annotation from becoming a second transcript.
DEFAULT_TAIL_WORDS = 12


@dataclass(frozen=True)
class InterruptionVerdict:
    """The result of a single ``caller_started_speaking()`` call.

    ``genuine=False`` covers both ordinary turn-taking (nothing was
    mid-playback) and the grace-window case (playback was essentially done).
    In both cases the remaining fields are ``None``: callers should not
    inject anything into the model's context for a non-interruption.
    """

    genuine: bool
    utterance_id: str | None = None
    heard_text: str | None = None
    unheard_text: str | None = None
    #: Ready-to-inject system/context note. Only set when ``genuine`` is True.
    context_annotation: str | None = None


@dataclass
class _TrackedUtterance:
    """Mutable bookkeeping for the one bot utterance currently in flight."""

    utterance_id: str
    full_text: str
    played_upto: int = 0


class PlaybackLedger:
    """Tracks generated-vs-played text for one ``CallSession`` and adjudicates
    barge-ins.

    Lifecycle for one bot turn::

        ledger.utterance_started(uid, full_text)   # generation begins
        ledger.mark_played(uid, n)                 # ... called repeatedly as
        ledger.mark_played(uid, n2)                #     the transport/TTS
        ledger.mark_played(uid, n3)                #     reports real playout
        ledger.utterance_finished(uid)              # played out naturally

    If the caller instead starts talking mid-playback::

        verdict = ledger.caller_started_speaking()
        if verdict.genuine:
            # feed verdict.context_annotation into the next LLM turn
            ...

    Only one utterance is tracked as "current" at a time, matching how a
    cascade/S2S runtime actually drives a call: a bot turn is either playing
    out or it isn't. Starting a new utterance implicitly retires whatever was
    being tracked before it (mirrors the runtime already having moved on).
    """

    def __init__(
        self,
        session: CallSession,
        grace_chars: int = DEFAULT_GRACE_CHARS,
        tail_words: int = DEFAULT_TAIL_WORDS,
    ) -> None:
        self._session = session
        self._grace_chars = grace_chars
        self._tail_words = tail_words
        self._current: _TrackedUtterance | None = None

    def utterance_started(self, utterance_id: str, full_text: str) -> None:
        """Record that generation for a new bot turn has begun.

        Emits ``BotUtterance`` with the *full* generated text — this is
        deliberately the text that may outrun playback; ``BotSpeechPlayed``
        is what narrows it back down to reality.
        """
        self._current = _TrackedUtterance(utterance_id=utterance_id, full_text=full_text)
        self._session.emit(
            BotUtterance(
                session_id=self._session.session_id,
                at=self._session.elapsed,
                text=full_text,
            )
        )

    def mark_played(self, utterance_id: str, upto_chars: int) -> None:
        """Record real playback progress for ``utterance_id``.

        ``upto_chars`` is the absolute count of characters of ``full_text``
        played so far (not a delta), as reported by the transport/TTS layer.
        Emits ``BotSpeechPlayed`` with only the *newly* played span so
        subscribers see a clean stream rather than repeatedly overlapping
        prefixes.

        Reports for an utterance that is not the current one (stale, or
        already finished/interrupted) are ignored: by the time they arrive
        there is nothing left to reconcile them against.
        """
        current = self._current
        if current is None or current.utterance_id != utterance_id:
            return
        upto = max(0, min(upto_chars, len(current.full_text)))
        if upto <= current.played_upto:
            return  # no new span played (duplicate or out-of-order report)
        newly_played = current.full_text[current.played_upto : upto]
        current.played_upto = upto
        self._session.emit(
            BotSpeechPlayed(
                session_id=self._session.session_id,
                at=self._session.elapsed,
                text=newly_played,
            )
        )

    def utterance_finished(self, utterance_id: str) -> None:
        """Record that ``utterance_id`` played out to completion naturally.

        This is the ordinary end of a bot turn. It stops tracking the
        utterance without emitting anything: a clean finish is not an event
        worth telling the rest of the system about, and it means a
        subsequent ``caller_started_speaking()`` correctly reports ordinary
        turn-taking rather than an interruption.
        """
        if self._current is not None and self._current.utterance_id == utterance_id:
            self._current = None

    def caller_started_speaking(self) -> InterruptionVerdict:
        """Adjudicate a low-level "caller started speaking" signal.

        Returns ``genuine=False`` (no event emitted, no annotation produced)
        when:

        - no utterance is currently mid-playback (ordinary turn-taking), or
        - the tracked utterance has ``grace_chars`` or fewer characters left
          unplayed (the bot was effectively done; timing jitter, not a
          barge-in).

        Otherwise this is a real barge-in: playback is stopped from the
        ledger's point of view, the heard/unheard split is computed (snapped
        to the nearest word boundary at or before the last-played character,
        so "heard" never claims a caller heard half a word), an
        ``Interruption`` event is emitted, and the returned verdict carries a
        ready-to-inject ``context_annotation``.
        """
        current = self._current
        if current is None:
            return InterruptionVerdict(genuine=False)

        # Either branch below ends the bot's turn from the ledger's
        # perspective: the caller has the floor now, tracked or not.
        self._current = None

        remaining = len(current.full_text) - current.played_upto
        if remaining <= self._grace_chars:
            return InterruptionVerdict(genuine=False)

        split = _snap_to_word_boundary(current.full_text, current.played_upto)
        heard_text = current.full_text[:split]
        unheard_text = current.full_text[split:]

        self._session.emit(
            Interruption(
                session_id=self._session.session_id,
                at=self._session.elapsed,
                heard_text=heard_text,
                unheard_text=unheard_text,
            )
        )

        annotation = (
            "[The caller interrupted. They only heard up to: "
            f'"{_last_words(heard_text, self._tail_words)}". '
            f'They did NOT hear: "{unheard_text}". '
            "Do not refer to the unheard part as if spoken.]"
        )
        return InterruptionVerdict(
            genuine=True,
            utterance_id=current.utterance_id,
            heard_text=heard_text,
            unheard_text=unheard_text,
            context_annotation=annotation,
        )


def _snap_to_word_boundary(text: str, idx: int) -> int:
    """Return the largest word boundary in ``text`` at or before ``idx``.

    A position ``j`` is a "word boundary" if it does not fall strictly
    inside a word — i.e. ``j`` is the start/end of ``text``, or the
    character immediately before or after it is whitespace. That covers
    both edges of a word: the character *played through completely* still
    counts as heard (the boundary right after it, before the following
    space, is valid), while a character played only *partway* through does
    not (there is no whitespace on either side of the cut, so the loop below
    keeps walking backward past the whole partial word).

    Snapping backward (never forward) is the part that matters: the caller
    genuinely did not hear a word that was cut off mid-syllable, so that
    trailing fragment must fall on the "unheard" side, never be credited as
    heard.

    Operates on Python ``str`` (code point) indices throughout, so it is
    correct for multibyte text (e.g. Devanagari, CJK, emoji) without any
    byte-length arithmetic leaking in.
    """
    n = len(text)
    if idx <= 0:
        return 0
    if idx >= n:
        return n
    j = idx
    while j > 0 and not (text[j - 1].isspace() or text[j].isspace()):
        j -= 1
    return j


def _last_words(text: str, n: int) -> str:
    """Join the last ``n`` whitespace-delimited words of ``text``."""
    words = text.split()
    return " ".join(words[-n:])
