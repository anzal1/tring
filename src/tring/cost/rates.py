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

from tring.events import CostComponent


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
    version="0.2.0-starter",
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
        # ------------------------------------------------------------------
        # v0.2 provider expansion. These rows are copied *by value* from each
        # provider module's own exported ``*_RATES`` list (see
        # providers/cloud/{llm,stt,tts,s2s}_extra.py and
        # providers/local/tts_extra.py for the citation of the vendor
        # pricing page each number was verified against, as_of "2026-09").
        #
        # They are duplicated here rather than imported, on purpose: this
        # module's own docstring promises "plain data: no vendor SDKs, no
        # network calls" for the starter card, and provider modules import
        # ``Rate`` from *this* file, so importing them back would both be a
        # circular import and would force ``httpx``/``websockets`` to be
        # importable just to build a price list. Keep both copies in sync
        # by hand when a provider's rate changes.
        # ------------------------------------------------------------------
        # -- STT --
        Rate(
            component=CostComponent.STT,
            provider="assemblyai",
            model=None,
            unit_name="audio_seconds",
            price_per_unit=0.15 / 3600,  # $0.15/hr
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.STT,
            provider="openai_stt",
            model="whisper-1",
            unit_name="audio_seconds",
            price_per_unit=0.006 / 60,  # $0.006/min
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.STT,
            provider="openai_stt",
            model="gpt-4o-transcribe",
            unit_name="tokens_in",
            price_per_unit=2.50e-6,  # $2.50 / 1M input tokens
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.STT,
            provider="openai_stt",
            model="gpt-4o-transcribe",
            unit_name="tokens_out",
            price_per_unit=10.0e-6,  # $10.00 / 1M output tokens
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.STT,
            provider="openai_stt",
            model="gpt-4o-mini-transcribe",
            unit_name="tokens_in",
            price_per_unit=1.25e-6,  # $1.25 / 1M input tokens
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.STT,
            provider="openai_stt",
            model="gpt-4o-mini-transcribe",
            unit_name="tokens_out",
            price_per_unit=5.0e-6,  # $5.00 / 1M output tokens
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.STT,
            provider="sarvam",
            model=None,
            unit_name="audio_seconds",
            price_per_unit=30 / 3600,  # ₹30/hr (Sarvam publishes no USD price)
            currency="INR",
            as_of="2026-09",
        ),
        # -- LLM --
        Rate(
            component=CostComponent.LLM,
            provider="anthropic",
            model="claude-sonnet-5",
            unit_name="tokens_in",
            price_per_unit=0.000002,  # $2 / 1M input tokens
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.LLM,
            provider="anthropic",
            model="claude-sonnet-5",
            unit_name="tokens_out",
            price_per_unit=0.00001,  # $10 / 1M output tokens
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.LLM,
            provider="gemini",
            model="gemini-2.5-flash",
            unit_name="tokens_in",
            price_per_unit=0.0000003,  # $0.30 / 1M input tokens (paid tier)
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.LLM,
            provider="gemini",
            model="gemini-2.5-flash",
            unit_name="tokens_out",
            price_per_unit=0.0000025,  # $2.50 / 1M output tokens (incl. thinking)
            currency="USD",
            as_of="2026-09",
        ),
        # -- TTS --
        Rate(
            component=CostComponent.TTS,
            provider="cartesia",
            model="sonic-3",
            unit_name="tts_chars",
            # No verifiable per-character rate is published (plan-level credit
            # allowances only) -- priced 0.0 rather than guessing a
            # credit-to-character ratio. See providers/cloud/tts_extra.py.
            price_per_unit=0.0,
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.TTS,
            provider="openai_tts",
            model="tts-1",
            unit_name="tts_chars",
            price_per_unit=0.000015,  # $15.00 / 1M characters
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.TTS,
            provider="openai_tts",
            model="tts-1-hd",
            unit_name="tts_chars",
            price_per_unit=0.00003,  # $30.00 / 1M characters
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.TTS,
            provider="sarvam_tts",
            model="bulbul:v3",
            unit_name="tts_chars",
            price_per_unit=0.003,  # ₹30 / 10,000 characters
            currency="INR",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.TTS,
            provider="piper",
            model=None,
            unit_name="tts_chars",
            price_per_unit=0.0,  # local, free
            currency="USD",
            as_of="2026-09",
        ),
        # -- S2S --
        Rate(
            component=CostComponent.S2S,
            provider="openai_realtime",
            model="gpt-realtime-2.1",
            unit_name="tokens_in",
            price_per_unit=0.000032,  # $32.00 / 1M audio input tokens
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.S2S,
            provider="openai_realtime",
            model="gpt-realtime-2.1",
            unit_name="tokens_out",
            price_per_unit=0.000064,  # $64.00 / 1M audio output tokens
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.S2S,
            provider="ultravox",
            model=None,  # priced per call-minute, not per model
            unit_name="audio_seconds",
            price_per_unit=0.05 / 60,  # $0.05 / minute
            currency="USD",
            as_of="2026-09",
        ),
        # ------------------------------------------------------------------
        # v0.4 CORE swarm additions. Same by-value-copy rule as the v0.2
        # block above: see providers/cloud/s2s_gemini.py, outbound/dialer.py,
        # and knowledge/{chroma,keyword_local}.py for the vendor-pricing
        # citation each row was verified against, as_of "2026-09".
        # ------------------------------------------------------------------
        # -- S2S (Gemini Live) --
        Rate(
            component=CostComponent.S2S,
            provider="gemini_live",
            model="gemini-3.8-live",
            unit_name="tokens_in",
            price_per_unit=0.000003,  # $3.00 / 1M audio input tokens
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.S2S,
            provider="gemini_live",
            model="gemini-3.8-live",
            unit_name="tokens_out",
            price_per_unit=0.000012,  # $12.00 / 1M audio output tokens
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.S2S,
            provider="gemini_live",
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            unit_name="tokens_in",
            price_per_unit=0.000003,  # $3.00 / 1M audio (or video) input tokens
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.S2S,
            provider="gemini_live",
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            unit_name="tokens_out",
            price_per_unit=0.000012,  # $12.00 / 1M audio output tokens
            currency="USD",
            as_of="2026-09",
        ),
        # -- Telephony (outbound dialing) --
        Rate(
            component=CostComponent.TELEPHONY,
            provider="twilio",
            model=None,
            unit_name="call_seconds",
            price_per_unit=0.0140 / 60,  # $0.0140/min outbound, US, PAYG list price
            currency="USD",
            as_of="2026-09",
        ),
        # -- Knowledge (local, zero marginal cost by construction) --
        Rate(
            component=CostComponent.OTHER,
            provider="chroma_local",
            model=None,
            unit_name="knowledge_queries",
            price_per_unit=0.0,
            currency="USD",
            as_of="2026-09",
        ),
        Rate(
            component=CostComponent.OTHER,
            provider="keyword_local",
            model=None,
            unit_name="knowledge_queries",
            price_per_unit=0.0,
            currency="USD",
            as_of="2026-09",
        ),
    ],
)
