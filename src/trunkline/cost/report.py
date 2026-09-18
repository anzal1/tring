"""Roll-ups over a session's (or many sessions') cost events.

Two different questions live here, and conflating them is how cost work goes
wrong:

1. "What did this call cost, and how much of that number is a guess?"
   -> :meth:`CostReport.from_events`

2. "Is the business actually cheaper, or did we just move the expensive part
   from vendor rates to the funnel above it?" -> :func:`denominator_ladder`

Optimizing rate-per-token is the easy, visible lever. It is also usually the
smaller one: a 20% cheaper LLM is worthless if only one dial in five connects,
or one connected call in ten becomes a real conversation. The ladder makes
that funnel show up in the same currency-per-unit terms as the rate card, so
the two are comparable instead of the funnel staying an unstated assumption.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from trunkline.events import CostRecorded, SessionEnded, SessionEvent


class CostReport(BaseModel):
    """A totals-and-breakdown summary over a batch of session events.

    Built once via :meth:`from_events`; the fields themselves are plain data
    so a report can be serialized, diffed, or asserted on directly in tests
    without re-walking the event list.
    """

    total: float
    currency: str
    by_component: dict[str, float] = Field(default_factory=dict)
    by_provider: dict[str, float] = Field(default_factory=dict)
    estimated_amount: float
    estimated_fraction: float
    record_count: int
    duration_seconds: float | None = None

    @classmethod
    def from_events(cls, events: list[SessionEvent]) -> CostReport:
        """Aggregate every ``CostRecorded`` in ``events`` into one report.

        Multiple ``SessionEnded`` events (e.g. events from several calls
        concatenated) use the *last* one's duration — good enough for a
        quick roll-up; for accurate multi-call duration, sum
        ``duration_seconds`` from each call's own report instead of
        concatenating raw events.

        A report over a mix of currencies is technically wrong to sum, but
        we still sum the raw amounts rather than raising: single-currency
        deployments are the overwhelming case, and refusing to produce a
        number is worse than producing one that a caller who *does* mix
        currencies should catch by checking ``currency`` isn't what they
        expect. ``currency`` reports the last currency seen.
        """
        total = 0.0
        estimated_amount = 0.0
        by_component: dict[str, float] = {}
        by_provider: dict[str, float] = {}
        currency = "USD"
        record_count = 0
        duration_seconds: float | None = None

        for event in events:
            if isinstance(event, CostRecorded):
                record_count += 1
                total += event.amount
                currency = event.currency
                if event.estimated:
                    estimated_amount += event.amount
                by_component[event.component.value] = (
                    by_component.get(event.component.value, 0.0) + event.amount
                )
                by_provider[event.provider] = (
                    by_provider.get(event.provider, 0.0) + event.amount
                )
            elif isinstance(event, SessionEnded):
                duration_seconds = event.duration_seconds

        estimated_fraction = (estimated_amount / total) if total else 0.0

        return cls(
            total=total,
            currency=currency,
            by_component=by_component,
            by_provider=by_provider,
            estimated_amount=estimated_amount,
            estimated_fraction=estimated_fraction,
            record_count=record_count,
            duration_seconds=duration_seconds,
        )

    def as_text(self) -> str:
        """Render a plain-text table suitable for a terminal or log line."""
        lines = ["Cost Report", "=" * 11]
        lines.append(
            f"Total: {self.total:.4f} {self.currency}  "
            f"(estimated: {self.estimated_fraction * 100:.1f}%, "
            f"{self.record_count} record(s))"
        )
        if self.duration_seconds is not None:
            lines.append(f"Duration: {self.duration_seconds:.1f}s")

        if self.by_component:
            lines.append("")
            lines.append("By component:")
            for name, amount in sorted(self.by_component.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {name:<12} {amount:.4f} {self.currency}")

        if self.by_provider:
            lines.append("")
            lines.append("By provider:")
            for name, amount in sorted(self.by_provider.items(), key=lambda kv: -kv[1]):
                lines.append(f"  {name:<12} {amount:.4f} {self.currency}")

        return "\n".join(lines)


def denominator_ladder(
    total_cost_all_calls: float,
    dials: int,
    connected: int,
    conversations: int,
    outcomes: int,
) -> dict[str, float | None]:
    """Cost per stage of the funnel a dial has to survive to become value.

    ``dials`` >= ``connected`` >= ``conversations`` >= ``outcomes`` is the
    expected shape (each stage is a subset of callers who cleared the one
    before), but this function does not enforce it — it just divides.

    The reason this function exists at all: shaving the vendor rate card
    optimizes the numerator. Nothing here optimizes it — it only exposes the
    denominator, which is usually the bigger lever. A team that cuts its STT
    bill 30% while its connect rate is 20% has made a rounding error look
    like a strategy.

    A stage with zero volume has no defined per-unit cost, so its entry is
    ``None`` rather than raising ``ZeroDivisionError`` or silently reporting
    ``0.0`` (which would misleadingly read as "free").
    """

    def safe_div(denominator: int) -> float | None:
        if denominator <= 0:
            return None
        return total_cost_all_calls / denominator

    return {
        "cost_per_dial": safe_div(dials),
        "cost_per_connected": safe_div(connected),
        "cost_per_conversation": safe_div(conversations),
        "cost_per_outcome": safe_div(outcomes),
    }
