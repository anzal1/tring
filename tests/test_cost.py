"""Tests for awaaz.cost: rate lookup, metering, and reporting.

No network, no vendor SDKs, no GPU — everything here runs against
``DEFAULT_RATES`` and a plain ``CallSession`` with an injected clock.
"""

from __future__ import annotations

from awaaz.agent import AgentSpec
from awaaz.cost.meter import CostMeter
from awaaz.cost.rates import DEFAULT_RATES, Rate, RateCard
from awaaz.cost.report import CostReport, denominator_ladder
from awaaz.events import CostComponent, CostRecorded, SessionEnded
from awaaz.providers.base import Usage
from awaaz.session import CallSession


def make_session(session_id: str = "sess-1") -> CallSession:
    agent = AgentSpec(name="test-agent", persona="You are a test agent.")
    clock = iter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    return CallSession(agent=agent, session_id=session_id, clock=lambda: next(clock, 10.0))


# ---------------------------------------------------------------------------
# RateCard.lookup precedence
# ---------------------------------------------------------------------------


def test_lookup_model_specific_beats_generic() -> None:
    card = RateCard(
        version="test",
        rates=[
            Rate(
                component=CostComponent.TTS,
                provider="acme",
                model=None,
                unit_name="tts_chars",
                price_per_unit=0.001,
                as_of="2025-01-01",
            ),
            Rate(
                component=CostComponent.TTS,
                provider="acme",
                model="acme-hd",
                unit_name="tts_chars",
                price_per_unit=0.002,
                as_of="2025-01-01",
            ),
        ],
    )

    specific = card.lookup(CostComponent.TTS, "acme", "tts_chars", model="acme-hd")
    generic = card.lookup(CostComponent.TTS, "acme", "tts_chars", model="acme-standard")
    no_model = card.lookup(CostComponent.TTS, "acme", "tts_chars")

    assert specific is not None and specific.price_per_unit == 0.002
    assert generic is not None and generic.price_per_unit == 0.001
    assert no_model is not None and no_model.price_per_unit == 0.001


def test_lookup_unknown_combination_returns_none() -> None:
    result = DEFAULT_RATES.lookup(CostComponent.STT, "nonexistent-vendor", "audio_seconds")
    assert result is None


def test_default_rates_local_providers_are_free() -> None:
    rate = DEFAULT_RATES.lookup(
        CostComponent.LLM, "local", "tokens_in", model="ollama"
    )
    assert rate is not None
    assert rate.price_per_unit == 0.0


# ---------------------------------------------------------------------------
# CostMeter
# ---------------------------------------------------------------------------


def test_meter_record_computes_amount_from_rate() -> None:
    session = make_session()
    meter = CostMeter(session, DEFAULT_RATES)

    usage = Usage(units=120.0, unit_name="audio_seconds", model="nova-2")
    event = meter.record(CostComponent.STT, "deepgram", usage)

    rate = DEFAULT_RATES.lookup(CostComponent.STT, "deepgram", "audio_seconds", model="nova-2")
    assert rate is not None
    assert event.amount == 120.0 * rate.price_per_unit
    assert event.estimated is False
    assert event.type == "cost_recorded"
    # Emitted onto the session, not just returned.
    assert event in session.history


def test_meter_record_unknown_rate_is_flagged_and_zero() -> None:
    session = make_session()
    meter = CostMeter(session, DEFAULT_RATES)

    usage = Usage(units=50.0, unit_name="widgets", model=None)
    event = meter.record(CostComponent.OTHER, "mystery-vendor", usage)

    assert event.amount == 0.0
    assert event.estimated is True
    assert len(meter.missing_rate_notes) == 1
    assert "mystery-vendor" in meter.missing_rate_notes[0]


def test_meter_record_passes_through_provider_estimated_flag() -> None:
    """A provider that hands in estimated=True usage stays estimated even
    when a real rate exists — the flag reflects the *measurement*, and a
    priced measurement can still be a guess."""
    session = make_session()
    meter = CostMeter(session, DEFAULT_RATES)

    usage = Usage(units=42.0, unit_name="tts_chars", estimated=True, model=None)
    event = meter.record(CostComponent.TTS, "elevenlabs", usage)

    assert event.estimated is True
    assert event.amount > 0.0  # a real rate was found and applied


def test_meter_record_llm_emits_two_events_with_cached_units() -> None:
    session = make_session()
    meter = CostMeter(session, DEFAULT_RATES)

    usage_in = Usage(units=1000.0, unit_name="tokens_in")
    usage_out = Usage(units=200.0, unit_name="tokens_out")

    cost_in, cost_out = meter.record_llm(
        "openai", "gpt-4o-mini", usage_in, usage_out, cached=400.0
    )

    assert cost_in.unit_name == "tokens_in"
    assert cost_in.cached_units == 400.0
    assert cost_in.model == "gpt-4o-mini"
    assert cost_out.unit_name == "tokens_out"
    # Cached units never leak onto the output-token record.
    assert cost_out.cached_units is None

    rate_in = DEFAULT_RATES.lookup(
        CostComponent.LLM, "openai", "tokens_in", model="gpt-4o-mini"
    )
    rate_out = DEFAULT_RATES.lookup(
        CostComponent.LLM, "openai", "tokens_out", model="gpt-4o-mini"
    )
    assert rate_in is not None and rate_out is not None
    assert cost_in.amount == 1000.0 * rate_in.price_per_unit
    assert cost_out.amount == 200.0 * rate_out.price_per_unit

    both = [e for e in session.history if isinstance(e, CostRecorded)]
    assert len(both) == 2


def test_meter_record_llm_does_not_overwrite_existing_cached_units() -> None:
    session = make_session()
    meter = CostMeter(session, DEFAULT_RATES)

    usage_in = Usage(units=1000.0, unit_name="tokens_in", cached_units=999.0)
    usage_out = Usage(units=200.0, unit_name="tokens_out")

    cost_in, _ = meter.record_llm("openai", "gpt-4o-mini", usage_in, usage_out, cached=1.0)

    assert cost_in.cached_units == 999.0


# ---------------------------------------------------------------------------
# CostReport
# ---------------------------------------------------------------------------


def test_report_from_events_totals_by_component_and_provider() -> None:
    session = make_session()
    meter = CostMeter(session, DEFAULT_RATES)

    stt_usage = Usage(units=60.0, unit_name="audio_seconds", model="nova-2")
    llm_in_usage = Usage(units=1000.0, unit_name="tokens_in", model="gpt-4o-mini")
    llm_out_usage = Usage(units=200.0, unit_name="tokens_out", model="gpt-4o-mini")
    tts_usage = Usage(units=300.0, unit_name="tts_chars")

    meter.record(CostComponent.STT, "deepgram", stt_usage)
    meter.record(CostComponent.LLM, "openai", llm_in_usage)
    meter.record(CostComponent.LLM, "openai", llm_out_usage)
    meter.record(CostComponent.TTS, "elevenlabs", tts_usage)
    session.emit(
        SessionEnded(session_id=session.session_id, at=session.elapsed, duration_seconds=42.0)
    )

    report = CostReport.from_events(session.history)

    expected_stt = 60.0 * 0.0043
    expected_llm_in = 1000.0 * 0.00000015
    expected_llm_out = 200.0 * 0.0000006
    expected_tts = 300.0 * 0.00018
    expected_total = expected_stt + expected_llm_in + expected_llm_out + expected_tts

    assert report.total == expected_total
    assert report.by_component["stt"] == expected_stt
    assert report.by_component["llm"] == expected_llm_in + expected_llm_out
    assert report.by_component["tts"] == expected_tts
    assert report.by_provider["openai"] == expected_llm_in + expected_llm_out
    assert report.duration_seconds == 42.0
    assert report.record_count == 4
    assert report.estimated_fraction == 0.0


def test_report_estimated_fraction_reflects_unpriced_share() -> None:
    session = make_session()
    meter = CostMeter(session, DEFAULT_RATES)

    # Priced, non-estimated.
    meter.record(CostComponent.TTS, "elevenlabs", Usage(units=1000.0, unit_name="tts_chars"))
    # Unpriced -> forced to estimated, amount 0 (contributes nothing to the
    # fraction's numerator here, so exercise a *priced-but-estimated* case
    # too, which is the case that actually moves estimated_fraction).
    meter.record(
        CostComponent.TTS,
        "elevenlabs",
        Usage(units=1000.0, unit_name="tts_chars", estimated=True),
    )

    report = CostReport.from_events(session.history)

    assert report.estimated_fraction == 0.5
    assert report.record_count == 2


def test_report_as_text_contains_key_figures() -> None:
    session = make_session()
    meter = CostMeter(session, DEFAULT_RATES)
    stt_usage = Usage(units=10.0, unit_name="audio_seconds", model="nova-2")
    meter.record(CostComponent.STT, "deepgram", stt_usage)
    session.emit(
        SessionEnded(session_id=session.session_id, at=session.elapsed, duration_seconds=10.0)
    )

    text = CostReport.from_events(session.history).as_text()

    assert "Cost Report" in text
    assert "Total:" in text
    assert "Duration: 10.0s" in text
    assert "deepgram" in text
    assert "stt" in text


def test_report_from_events_empty_is_zero_safe() -> None:
    report = CostReport.from_events([])
    assert report.total == 0.0
    assert report.estimated_fraction == 0.0
    assert report.record_count == 0
    assert report.duration_seconds is None


# ---------------------------------------------------------------------------
# denominator_ladder
# ---------------------------------------------------------------------------


def test_denominator_ladder_basic_math() -> None:
    ladder = denominator_ladder(
        total_cost_all_calls=1000.0, dials=1000, connected=500, conversations=200, outcomes=50
    )
    assert ladder["cost_per_dial"] == 1.0
    assert ladder["cost_per_connected"] == 2.0
    assert ladder["cost_per_conversation"] == 5.0
    assert ladder["cost_per_outcome"] == 20.0


def test_denominator_ladder_zero_denominators_are_none_not_errors() -> None:
    ladder = denominator_ladder(
        total_cost_all_calls=500.0, dials=0, connected=0, conversations=0, outcomes=0
    )
    assert ladder == {
        "cost_per_dial": None,
        "cost_per_connected": None,
        "cost_per_conversation": None,
        "cost_per_outcome": None,
    }


def test_denominator_ladder_illustrates_funnel_dwarfing_rate_optimization() -> None:
    """The whole point of the ladder: a cheaper rate card barely moves
    cost-per-outcome next to a bad connect rate."""
    expensive_rates_good_funnel = denominator_ladder(
        total_cost_all_calls=1000.0, dials=1000, connected=900, conversations=800, outcomes=700
    )
    cheap_rates_bad_funnel = denominator_ladder(
        total_cost_all_calls=700.0,  # 30% cheaper rate card
        dials=1000,
        connected=200,  # but connect rate collapsed
        conversations=100,
        outcomes=20,
    )

    per_outcome_good = expensive_rates_good_funnel["cost_per_outcome"]
    per_outcome_bad = cheap_rates_bad_funnel["cost_per_outcome"]
    assert per_outcome_good is not None and per_outcome_bad is not None
    # Despite the 30% cheaper rate card, cost-per-outcome got far worse.
    assert per_outcome_bad > per_outcome_good
