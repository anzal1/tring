"""Versioned vendor rate cards.

A ``RateCard`` is plain data: no vendor SDKs, no network calls, no "current
price" lookups. Prices drift; a rate card is a snapshot with an ``as_of`` date
baked into every row, so a stale card is visible in the data itself rather
than silently wrong.

``DEFAULT_RATES`` below is a *starter* card, not a source of truth. It exists
so the meter has something to compute against out of the box and so tests
have a fixture. Anyone running this in anger should pin their own
``RateCard`` (their negotiated prices, their date) rather than relying on
numbers a library shipped months ago.
"""

from __future__ import annotations

from pydantic import BaseModel

from awaaz.events import CostComponent


class Rate(BaseModel):
    """The price of one unit of one metered thing.

    ``model`` is optional: a rate can be provider-wide (e.g. a flat
    telephony-minute rate) or pinned to one model (e.g. gpt-4o-mini input
    tokens, which price differently than gpt-4o-mini output tokens or any
    other model on the same provider).
    """

    component: CostComponent
    provider: str
    model: str | None = None
    unit_name: str
    price_per_unit: float
    currency: str = "USD"
    # Data, not metadata: the date these numbers were true. Lets a stale card
    # be spotted by reading it, not by trusting the library version.
    as_of: str


class RateCard(BaseModel):
    """A named, versioned collection of rates.

    ``lookup`` implements the one precedence rule that matters: a
    model-specific rate always beats a generic provider-wide rate for the
    same (component, provider, unit_name). This lets a card carry both a
    provider default and per-model overrides without the caller having to
    know which one applies.
    """

    rates: list[Rate]
    version: str

    def lookup(
        self,
        component: CostComponent,
        provider: str,
        unit_name: str,
        model: str | None = None,
    ) -> Rate | None:
        """Find the best-matching rate, or None if this card has no opinion.

        A None result is not an error — it is the signal the meter uses to
        fall back to a zero-rate, ``estimated=True`` record with a note
        rather than guessing a price.
        """
        generic: Rate | None = None
        for rate in self.rates:
            if (
                rate.component is not component
                or rate.provider != provider
                or rate.unit_name != unit_name
            ):
                continue
            if model is not None and rate.model == model:
                # Model-specific match: nothing beats this, return immediately.
                return rate
            if rate.model is None and generic is None:
                generic = rate
        return generic


# ---------------------------------------------------------------------------
# DEFAULT_RATES: a small starter card using publicly listed prices.
#
# These are illustrative, not contractual. Vendor pricing changes; pin your
# own RateCard with your own negotiated or currently-listed prices and a
# fresh `as_of` date rather than depending on the numbers below.
# ---------------------------------------------------------------------------
DEFAULT_RATES = RateCard(
    version="0.1.0-starter",
    rates=[
        # Deepgram Nova-2 streaming STT, pay-as-you-go list price.
        Rate(
            component=CostComponent.STT,
            provider="deepgram",
            model="nova-2",
            unit_name="audio_seconds",
            price_per_unit=0.0043,  # ≈ $0.0043/s ($0.258/min) list, PAYG tier
            currency="USD",
            as_of="2025-06-01",
        ),
        # ElevenLabs TTS, per-character, "Creator"-tier list pricing.
        Rate(
            component=CostComponent.TTS,
            provider="elevenlabs",
            model=None,
            unit_name="tts_chars",
            price_per_unit=0.00018,  # ≈ $0.18 / 1k chars
            currency="USD",
            as_of="2025-06-01",
        ),
        # OpenAI gpt-4o-mini, input and output tokens priced separately.
        Rate(
            component=CostComponent.LLM,
            provider="openai",
            model="gpt-4o-mini",
            unit_name="tokens_in",
            price_per_unit=0.00000015,  # $0.15 / 1M input tokens
            currency="USD",
            as_of="2025-06-01",
        ),
        Rate(
            component=CostComponent.LLM,
            provider="openai",
            model="gpt-4o-mini",
            unit_name="tokens_out",
            price_per_unit=0.0000006,  # $0.60 / 1M output tokens
            currency="USD",
            as_of="2025-06-01",
        ),
        # Local providers: zero marginal cost by construction, not an
        # estimate. Listed explicitly so a missing-rate lookup for a local
        # provider is a real bug, not a coincidence of an empty card.
        Rate(
            component=CostComponent.STT,
            provider="local",
            model="faster_whisper",
            unit_name="audio_seconds",
            price_per_unit=0.0,
            currency="USD",
            as_of="2025-06-01",
        ),
        Rate(
            component=CostComponent.LLM,
            provider="local",
            model="ollama",
            unit_name="tokens_in",
            price_per_unit=0.0,
            currency="USD",
            as_of="2025-06-01",
        ),
        Rate(
            component=CostComponent.LLM,
            provider="local",
            model="ollama",
            unit_name="tokens_out",
            price_per_unit=0.0,
            currency="USD",
            as_of="2025-06-01",
        ),
        Rate(
            component=CostComponent.TTS,
            provider="local",
            model="kokoro",
            unit_name="tts_chars",
            price_per_unit=0.0,
            currency="USD",
            as_of="2025-06-01",
        ),
    ],
)
