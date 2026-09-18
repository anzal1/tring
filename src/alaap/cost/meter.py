"""CostMeter — turns provider Usage into CostRecorded events.

The honesty rule (see ARCHITECTURE.md, "Honest cost accounting") is enforced
in one place: if a provider reports exact usage but the rate card has no
price for it, the meter does not guess — it records a zero-amount line with
``estimated=True`` and a note, so a missing rate shows up as a visible gap
instead of a silently wrong total. Providers may also *hand in* estimated
usage (e.g. a token count approximated from character length); that flag is
passed straight through.

Cached vs. uncached tokens are recorded separately (via ``cached_units`` on
the resulting event) rather than netted out, because a silent cache
regression — the same prompt suddenly costing full price — should be
visible in the ledger, not absorbed into a single blended number.
"""

from __future__ import annotations

from dataclasses import replace

from alaap.cost.rates import RateCard
from alaap.events import CostComponent, CostRecorded
from alaap.providers.base import Usage
from alaap.session import CallSession


class CostMeter:
    """Records metered usage against a session's event stream.

    One meter is bound to one ``CallSession`` and one ``RateCard`` for its
    lifetime. It does not own billing logic beyond unit * price — anything
    fancier (volume discounts, committed spend) belongs in a caller-supplied
    RateCard, not in this class.
    """

    def __init__(self, session: CallSession, card: RateCard) -> None:
        self.session = session
        self.card = card
        # Every "no rate found" occurrence, in order, for callers who want to
        # surface a warning (e.g. "gpt-4o-mini tokens_out has no rate: check
        # your card") without having to re-derive it from the event stream.
        self.missing_rate_notes: list[str] = []

    def record(
        self,
        component: CostComponent,
        provider: str,
        usage: Usage,
    ) -> CostRecorded:
        """Price one ``Usage`` line and emit the resulting ``CostRecorded``.

        Returns the emitted event so callers (tests, CLIs) don't have to
        re-read it off the session history.
        """
        rate = self.card.lookup(component, provider, usage.unit_name, model=usage.model)
        estimated = usage.estimated
        if rate is None:
            price_per_unit = 0.0
            currency = "USD"
            estimated = True
            note = (
                f"no rate for component={component.value} provider={provider!r} "
                f"model={usage.model!r} unit_name={usage.unit_name!r}; "
                "recorded as 0.0 and flagged estimated"
            )
            self.missing_rate_notes.append(note)
        else:
            price_per_unit = rate.price_per_unit
            currency = rate.currency

        event = CostRecorded(
            session_id=self.session.session_id,
            at=self.session.elapsed,
            component=component,
            provider=provider,
            units=usage.units,
            unit_name=usage.unit_name,
            amount=usage.units * price_per_unit,
            currency=currency,
            estimated=estimated,
            model=usage.model,
            cached_units=usage.cached_units,
        )
        self.session.emit(event)
        return event

    def record_llm(
        self,
        provider: str,
        model: str | None,
        usage_in: Usage,
        usage_out: Usage,
        cached: float | None = None,
    ) -> tuple[CostRecorded, CostRecorded]:
        """Convenience for the common case: one LLM turn, in and out tokens.

        ``model`` fills in ``usage_in.model`` / ``usage_out.model`` when the
        caller built ``Usage`` without one (the common case when the model
        name is only known at the call site, not where usage was parsed);
        an existing ``.model`` on a ``Usage`` is left untouched.

        ``cached`` is a shorthand for setting ``usage_in.cached_units`` when
        the caller already has cache-hit counts on hand but built ``Usage``
        without them; it never overwrites a ``cached_units`` the caller
        already set on ``usage_in`` — pass it there directly if you have it.
        Cached tokens are still billed as part of ``usage_in.units`` (most
        vendors bill cached tokens at a reduced, not zero, rate baked into
        the same per-token price); ``cached_units`` exists purely so the
        cached share is *visible*, not to change the amount charged here.
        """
        in_usage = usage_in if usage_in.model is not None else replace(usage_in, model=model)
        out_usage = (
            usage_out if usage_out.model is not None else replace(usage_out, model=model)
        )
        if cached is not None and in_usage.cached_units is None:
            in_usage = replace(in_usage, cached_units=cached)

        cost_in = self.record(CostComponent.LLM, provider, in_usage)
        cost_out = self.record(CostComponent.LLM, provider, out_usage)
        return cost_in, cost_out
