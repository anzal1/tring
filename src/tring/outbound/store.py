"""Per-callee outcome tracking, persisted as an append-only JSONL log.

A campaign can run for hours across a process that might restart, so its
audit trail is a plain file, not in-memory state: every stage a call reaches
(dialed, connected, conversation, outcome, ...) is one appended line, never
rewritten in place. That makes the log crash-safe (a killed process loses at
most the record currently being written, never a prior one) and trivially
diffable / greppable / tailable in production, the same tradeoff
``cost/meter.py`` makes by emitting one ``CostRecorded`` event per line item
instead of mutating a running total.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

#: The stages one call attempt can be recorded at, in the order a healthy
#: attempt passes through them. Not every attempt reaches every stage:
#: "failed" and "no_outcome" are both terminal without reaching "outcome".
CallStage = Literal[
    "deferred",  # held back by quiet hours; not yet placed
    "dialed",  # place_call() returned a handle
    "connected",  # the dial outcome reported a live connection
    "conversation",  # the caller's session produced a final transcript
    "outcome",  # a business outcome was recorded (tool call or explicit API)
    "no_outcome",  # the call connected but no outcome arrived in time
    "failed",  # every attempt for this callee exhausted its retries
]


class CampaignRecord(BaseModel):
    """One row in a campaign's JSONL log: one callee at one stage.

    ``call_id`` is empty for a ``"deferred"`` record (quiet hours held the
    call back before any dialer call was ever placed, so no id exists yet)
    and the dialer's id for every other stage. Multiple records share a
    ``call_id`` as a call progresses through stages; a retried callee gets a
    *new* ``call_id`` per attempt (from a fresh ``place_call``), so
    ``attempt`` is what threads a callee's attempts together, not ``call_id``.
    """

    campaign: str
    call_id: str
    phone: str
    attempt: int
    status: CallStage
    at: float  # epoch seconds (time.time()), not session-relative
    answered_by: str | None = None
    call_duration_seconds: float | None = None
    outcome_label: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CampaignStore:
    """Append-only JSONL persistence for :class:`CampaignRecord`.

    One physical file, opened and closed on every call rather than held
    open for the runner's lifetime: campaigns run for hours, and a store
    that keeps a file descriptor open that whole time is one unexpected
    ``OSError`` away from losing every record after the failure. Reopening
    in append mode costs a syscall per record, which is nothing next to the
    latency of an actual phone call.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: CampaignRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(record.model_dump_json() + "\n")

    def read_all(self) -> list[CampaignRecord]:
        """Read every record back, in the order they were appended.

        Returns an empty list for a store whose file does not exist yet
        (a campaign that has not placed its first call), rather than
        raising -- the empty case is normal, not exceptional.
        """
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(CampaignRecord.model_validate_json(line))
        return records


__all__ = ["CallStage", "CampaignRecord", "CampaignStore"]
