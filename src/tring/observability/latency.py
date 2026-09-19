"""Per-turn latency waterfalls, computed from a session's own event timestamps.

No clock of its own: every number here comes from ``SessionEvent.at``, the
same monotonic-since-session-start float every runtime already stamps onto
every event (see ``CallSession``). That is what makes this module honest
under test: a fake clock produces exact, hand-computable waterfalls, so the
percentile tests below are exact rather than "close enough".

A voice call has exactly one latency question that matters: how long between
the caller finishing a sentence and hearing the reply start? ``TurnWaterfall``
splits that gap into the two things a team can actually act on separately,
"the model is slow to produce a first word" vs "TTS/transport is slow to turn
that word into audio", instead of a single blended number that tells you
something is wrong without telling you what.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace

from tring.events import BotSpeechPlayed, BotUtterance, SessionEvent, UserTranscript


@dataclass(frozen=True)
class TurnWaterfall:
    """The three timestamps that bound one caller turn, and the gaps between them.

    A turn begins at a *final* ``UserTranscript``: that is the moment the
    caller stopped talking and started waiting. ``first_bot_utterance_at`` and
    ``first_bot_speech_played_at`` are ``None`` when the call ended (or the
    caller cut back in) before that stage happened; a turn that never got a
    reply is not a bug in this module, it is a fact worth reporting, so it is
    represented rather than dropped.
    """

    session_id: str
    turn_index: int
    user_final_at: float
    first_bot_utterance_at: float | None = None
    first_bot_speech_played_at: float | None = None

    @property
    def thinking_seconds(self) -> float | None:
        """User finished talking -> the model's reply text exists.

        This is STT-finalization-to-first-token-of-reply in one number: on the
        cascade runtime it is dominated by LLM time-to-first-character, since
        ``BotUtterance`` fires once ``SpeakToolParser`` has a complete
        utterance queued for synthesis (see ``runtimes/cascade.py``), not once
        the whole completion has finished streaming.
        """
        if self.first_bot_utterance_at is None:
            return None
        return self.first_bot_utterance_at - self.user_final_at

    @property
    def synthesis_seconds(self) -> float | None:
        """Reply text exists -> the first audio of it was actually played.

        TTS-plus-transport time. Kept separate from ``thinking_seconds``
        because the fix for a slow one never fixes the other: a slow LLM
        needs a smaller model or fewer tokens, a slow synthesis stage needs a
        faster voice or a shorter jitter buffer.
        """
        if self.first_bot_utterance_at is None or self.first_bot_speech_played_at is None:
            return None
        return self.first_bot_speech_played_at - self.first_bot_utterance_at

    @property
    def total_seconds(self) -> float | None:
        """The number the caller actually experiences: silence, end to end."""
        if self.first_bot_speech_played_at is None:
            return None
        return self.first_bot_speech_played_at - self.user_final_at


#: Stage names used as the keys of a ``LatencyReport``. Kept as a tuple (not
#: derived from the dataclass fields) so the aggregation order is stable and
#: explicit rather than dependent on dataclass field declaration order.
STAGES = ("thinking", "synthesis", "total")


def turn_waterfalls(events: Sequence[SessionEvent]) -> list[TurnWaterfall]:
    """Reconstruct one session's per-turn waterfalls from its event history.

    Pass ``session.history`` (or any ordered slice of it). Events are walked
    in the order given: callers replaying a stored session must supply them
    already sorted by ``at``, which every runtime already guarantees at
    emission time.

    A new final ``UserTranscript`` always starts a new turn, even if the
    previous one never reached ``first_bot_utterance_at``: that previous turn
    is closed out exactly as it stood (an interrupted or abandoned turn is
    still a data point, typically an interesting one) and returned alongside
    the rest.
    """
    waterfalls: list[TurnWaterfall] = []
    current: TurnWaterfall | None = None

    for event in events:
        if isinstance(event, UserTranscript) and event.final:
            if current is not None:
                waterfalls.append(current)
            current = TurnWaterfall(
                session_id=event.session_id,
                turn_index=len(waterfalls),
                user_final_at=event.at,
            )
        elif isinstance(event, BotUtterance) and current is not None:
            if current.first_bot_utterance_at is None:
                current = replace(current, first_bot_utterance_at=event.at)
        elif isinstance(event, BotSpeechPlayed) and current is not None:
            if (
                current.first_bot_utterance_at is not None
                and current.first_bot_speech_played_at is None
            ):
                current = replace(current, first_bot_speech_played_at=event.at)

    if current is not None:
        waterfalls.append(current)
    return waterfalls


@dataclass(frozen=True)
class StageStats:
    """p50/p95 for one waterfall stage, plus how many turns had data for it.

    ``count`` matters on its own: a stage with three samples and a stage with
    three thousand can report the same p95, and only ``count`` tells a reader
    how much to trust it.
    """

    stage: str
    count: int
    p50: float
    p95: float


@dataclass(frozen=True)
class LatencyReport:
    """Per-stage percentiles aggregated across one or many sessions."""

    stages: dict[str, StageStats]

    def __getitem__(self, stage: str) -> StageStats:
        return self.stages[stage]


def _percentile(values: list[float], p: float) -> float:
    """The nearest-rank percentile of ``values`` (already validated non-empty).

    Nearest-rank rather than linear interpolation: it always returns an
    observed value (never an interpolated one between two samples), which is
    what makes hand-computed test fixtures exact instead of approximate.
    Formula: sort ascending, take the ``ceil(p/100 * n)``-th value, 1-indexed
    (e.g. p95 of 20 samples is the 19th smallest).
    """
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def aggregate(sessions: Iterable[Sequence[SessionEvent]]) -> LatencyReport:
    """Turn many sessions' event histories into one p50/p95-per-stage report.

    ``sessions`` is an iterable of event histories, typically
    ``[s.history for s in sessions]`` or JSON-replayed equivalents. Each is
    reduced to its own :func:`turn_waterfalls` and every turn from every
    session is pooled per stage before computing percentiles, so a session
    with one turn does not get the same weight as one with a hundred.
    """
    samples: dict[str, list[float]] = {stage: [] for stage in STAGES}
    for history in sessions:
        for waterfall in turn_waterfalls(history):
            for stage in STAGES:
                value = getattr(waterfall, f"{stage}_seconds")
                if value is not None:
                    samples[stage].append(value)

    stages: dict[str, StageStats] = {}
    for stage, values in samples.items():
        if not values:
            continue
        stages[stage] = StageStats(
            stage=stage,
            count=len(values),
            p50=_percentile(values, 50),
            p95=_percentile(values, 95),
        )
    return LatencyReport(stages=stages)


__all__ = [
    "STAGES",
    "LatencyReport",
    "StageStats",
    "TurnWaterfall",
    "aggregate",
    "turn_waterfalls",
]
