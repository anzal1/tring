"""Roll up a campaign's :class:`~tring.outbound.store.CampaignRecord` log.

The funnel counted here -- dials, connected, conversations, outcomes -- is
exactly the one :func:`tring.cost.report.denominator_ladder` was written to
price. That function already existed for per-call cost reports; this module
reuses it rather than re-deriving the same four divisions, which is the
whole point of ``docs/V4_PLAN.md`` Track 2 asking for it "computed live from
real counts."
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from tring.cost.report import denominator_ladder
from tring.outbound.store import CampaignRecord


class CampaignReport(BaseModel):
    """A totals-and-funnel summary over one campaign's record log.

    Built once via :meth:`from_records`; plain data afterwards so it can be
    serialized or asserted on directly, the same shape
    ``cost.report.CostReport`` takes.
    """

    campaign: str
    dials: int
    connected: int
    conversations: int
    outcomes: int
    failed: int
    no_outcome: int
    by_outcome: dict[str, int] = Field(default_factory=dict)
    total_cost: float = 0.0
    currency: str = "USD"
    ladder: dict[str, float | None] = Field(default_factory=dict)

    @classmethod
    def from_records(
        cls,
        campaign: str,
        records: list[CampaignRecord],
        total_cost: float = 0.0,
        currency: str = "USD",
    ) -> CampaignReport:
        """Aggregate every record for ``campaign`` into one report.

        Each funnel stage is counted by *distinct* ``call_id`` reaching
        that stage, not by record count: a call that is dialed, connects,
        and then has an outcome recorded produces three records but should
        count once at each of three stages, not thrice at "dialed."
        Retries are still counted correctly under this rule, because each
        retry attempt gets its own fresh ``call_id`` from the dialer (see
        ``CampaignRecord.call_id``'s docstring) -- so two failed attempts
        for one callee are, correctly, two dials.
        """
        dial_ids: set[str] = set()
        connected_ids: set[str] = set()
        conversation_ids: set[str] = set()
        outcome_ids: set[str] = set()
        no_outcome_ids: set[str] = set()
        failed_ids: set[str] = set()
        outcome_labels: Counter[str] = Counter()

        for record in records:
            if record.campaign != campaign:
                continue
            if record.status == "dialed":
                dial_ids.add(record.call_id)
            elif record.status == "connected":
                connected_ids.add(record.call_id)
            elif record.status == "conversation":
                conversation_ids.add(record.call_id)
            elif record.status == "outcome":
                outcome_ids.add(record.call_id)
                if record.outcome_label:
                    outcome_labels[record.outcome_label] += 1
            elif record.status == "no_outcome":
                no_outcome_ids.add(record.call_id)
            elif record.status == "failed":
                failed_ids.add(record.call_id)

        dials = len(dial_ids)
        connected = len(connected_ids)
        conversations = len(conversation_ids)
        outcomes = len(outcome_ids)

        ladder = denominator_ladder(
            total_cost_all_calls=total_cost,
            dials=dials,
            connected=connected,
            conversations=conversations,
            outcomes=outcomes,
        )

        return cls(
            campaign=campaign,
            dials=dials,
            connected=connected,
            conversations=conversations,
            outcomes=outcomes,
            failed=len(failed_ids),
            no_outcome=len(no_outcome_ids),
            by_outcome=dict(outcome_labels),
            total_cost=total_cost,
            currency=currency,
            ladder=ladder,
        )

    def as_text(self) -> str:
        """Render a plain-text table suitable for a terminal or log line."""
        lines = [f"Campaign Report: {self.campaign}", "=" * (17 + len(self.campaign))]
        lines.append(
            f"Dials: {self.dials}  Connected: {self.connected}  "
            f"Conversations: {self.conversations}  Outcomes: {self.outcomes}"
        )
        lines.append(f"Failed: {self.failed}  No outcome (window expired): {self.no_outcome}")
        lines.append(f"Total cost: {self.total_cost:.4f} {self.currency}")

        lines.append("")
        lines.append("Denominator ladder (cost per stage survived):")
        for key, value in self.ladder.items():
            rendered = f"{value:.4f} {self.currency}" if value is not None else "n/a (0 volume)"
            lines.append(f"  {key:<22} {rendered}")

        if self.by_outcome:
            lines.append("")
            lines.append("By outcome label:")
            for label, count in sorted(self.by_outcome.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {label:<22} {count}")

        return "\n".join(lines)


__all__ = ["CampaignReport"]
