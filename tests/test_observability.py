"""Tests for tring.observability: otel export, latency waterfalls, cache alarms.

No network, no vendor SDKs, no GPU. The OpenTelemetry tests stub
``opentelemetry``/``opentelemetry.trace`` in ``sys.modules`` with a small
recording fake (house pattern: see ``tests/test_vad.py``'s ``install_fake_onnx``
and ``tests/test_providers_tts.py``'s piper stub) and assert the resulting
span tree; everything else runs against plain ``CallSession`` event histories
with an injected deterministic clock.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

from tring.agent import AgentSpec
from tring.events import (
    BotSpeechPlayed,
    BotUtterance,
    CostComponent,
    CostRecorded,
    DtmfReceived,
    KnowledgeSearched,
    SessionEnded,
    SessionError,
    SessionEvent,
    SessionStarted,
    SessionTransferred,
    ToolCallCompleted,
    ToolCallStarted,
    UserTranscript,
)
from tring.observability.alarms import CacheRegressionAlarm, CacheRegressionDetector
from tring.observability.latency import StageStats, aggregate, turn_waterfalls
from tring.session import CallSession


def make_session(session_id: str = "sess-1") -> CallSession:
    """A session with a deterministic clock: the Nth ``emit`` sees ``at=N-1``.

    Matches the fixture shape in ``tests/test_cost.py`` so latency numbers in
    these tests are exact integers, not timing-dependent floats.
    """
    agent = AgentSpec(name="test-agent", persona="You are a test agent.")
    clock = iter(float(i) for i in range(1000))
    return CallSession(agent=agent, session_id=session_id, clock=lambda: next(clock))


# ---------------------------------------------------------------------------
# tring.observability.latency
# ---------------------------------------------------------------------------


def test_turn_waterfalls_computes_all_three_stages() -> None:
    session = make_session()
    events: list[SessionEvent] = [
        UserTranscript(session_id=session.session_id, at=0.0, text="hi", final=True),
        BotUtterance(session_id=session.session_id, at=1.0, text="hello"),
        BotSpeechPlayed(session_id=session.session_id, at=1.5, text="hello"),
    ]

    waterfalls = turn_waterfalls(events)

    assert len(waterfalls) == 1
    wf = waterfalls[0]
    assert wf.turn_index == 0
    assert wf.user_final_at == 0.0
    assert wf.first_bot_utterance_at == 1.0
    assert wf.first_bot_speech_played_at == 1.5
    assert wf.thinking_seconds == pytest.approx(1.0)
    assert wf.synthesis_seconds == pytest.approx(0.5)
    assert wf.total_seconds == pytest.approx(1.5)


def test_turn_waterfalls_ignores_non_final_transcripts() -> None:
    session = make_session()
    events: list[SessionEvent] = [
        UserTranscript(session_id=session.session_id, at=0.0, text="h", final=False),
        UserTranscript(session_id=session.session_id, at=0.2, text="hi", final=False),
        UserTranscript(session_id=session.session_id, at=0.5, text="hi there", final=True),
        BotUtterance(session_id=session.session_id, at=1.5, text="hello"),
    ]

    waterfalls = turn_waterfalls(events)

    assert len(waterfalls) == 1
    assert waterfalls[0].user_final_at == 0.5


def test_turn_waterfalls_only_records_first_bot_utterance_per_turn() -> None:
    """A tool's waiting-message speech and the eventual reply can both emit
    ``BotUtterance`` in the same turn; only the first one marks the stage."""
    session = make_session()
    events: list[SessionEvent] = [
        UserTranscript(session_id=session.session_id, at=0.0, text="book it", final=True),
        BotUtterance(session_id=session.session_id, at=0.3, text="one moment"),
        BotUtterance(session_id=session.session_id, at=2.0, text="all booked"),
    ]

    waterfalls = turn_waterfalls(events)

    assert waterfalls[0].first_bot_utterance_at == 0.3


def test_turn_waterfalls_interrupted_turn_has_no_completion_timestamps() -> None:
    """A turn cut off by the caller's next final transcript, before any bot
    reply, is still returned: incomplete data, honestly represented as
    ``None`` rather than silently dropped."""
    session = make_session()
    events: list[SessionEvent] = [
        UserTranscript(session_id=session.session_id, at=0.0, text="wait", final=True),
        UserTranscript(session_id=session.session_id, at=0.4, text="never mind", final=True),
        BotUtterance(session_id=session.session_id, at=1.0, text="sure thing"),
    ]

    waterfalls = turn_waterfalls(events)

    assert len(waterfalls) == 2
    assert waterfalls[0].user_final_at == 0.0
    assert waterfalls[0].first_bot_utterance_at is None
    assert waterfalls[0].thinking_seconds is None
    assert waterfalls[1].user_final_at == 0.4
    assert waterfalls[1].first_bot_utterance_at == 1.0


def test_turn_waterfalls_trailing_turn_never_replied() -> None:
    """A call that ends mid-turn: the last waterfall stays incomplete."""
    session = make_session()
    events = [
        UserTranscript(session_id=session.session_id, at=0.0, text="hello?", final=True),
    ]

    waterfalls = turn_waterfalls(events)

    assert len(waterfalls) == 1
    assert waterfalls[0].first_bot_utterance_at is None
    assert waterfalls[0].total_seconds is None


def test_percentile_is_exact_nearest_rank() -> None:
    """20 samples, 0..19 seconds: p50 is the 10th smallest (index 9 -> 9.0),
    p95 is the 19th smallest (index 18 -> 18.0). Hand-computable because the
    module documents nearest-rank, not interpolation."""
    session = make_session()
    events: list[Any] = []
    at = 0.0
    sid = session.session_id
    for total in range(20):
        events.append(UserTranscript(session_id=sid, at=at, text="x", final=True))
        events.append(BotUtterance(session_id=sid, at=at, text="y"))
        events.append(BotSpeechPlayed(session_id=sid, at=at + total, text="y"))
        at += total + 1

    report = aggregate([events])

    total_stats = report["total"]
    assert isinstance(total_stats, StageStats)
    assert total_stats.count == 20
    assert total_stats.p50 == pytest.approx(9.0)
    assert total_stats.p95 == pytest.approx(18.0)


def test_aggregate_pools_turns_across_sessions() -> None:
    events_a: list[SessionEvent] = [
        UserTranscript(session_id="a", at=0.0, text="x", final=True),
        BotUtterance(session_id="a", at=1.0, text="y"),
        BotSpeechPlayed(session_id="a", at=1.0, text="y"),
    ]
    events_b: list[SessionEvent] = [
        UserTranscript(session_id="b", at=0.0, text="x", final=True),
        BotUtterance(session_id="b", at=3.0, text="y"),
        BotSpeechPlayed(session_id="b", at=3.0, text="y"),
    ]

    report = aggregate([events_a, events_b])

    assert report["thinking"].count == 2
    # p50 (nearest-rank, n=2) is the 1st of [1.0, 3.0] sorted -> 1.0.
    assert report["thinking"].p50 == pytest.approx(1.0)
    assert report["thinking"].p95 == pytest.approx(3.0)


def test_aggregate_skips_stages_with_no_samples() -> None:
    """A stage nobody ever reached (e.g. no call was ever fully played out)
    is absent from the report rather than reported as a fabricated zero."""
    events = [UserTranscript(session_id="s", at=0.0, text="hi", final=True)]

    report = aggregate([events])

    assert "total" not in report.stages
    assert "thinking" not in report.stages


# ---------------------------------------------------------------------------
# tring.observability.alarms: CacheRegressionDetector
# ---------------------------------------------------------------------------


def _cost_event(
    at: float, cached: float, units: float = 1000.0, provider: str = "acme"
) -> CostRecorded:
    return CostRecorded(
        session_id="s",
        at=at,
        component=CostComponent.LLM,
        provider=provider,
        units=units,
        unit_name="tokens_in",
        amount=units * 0.001,
        cached_units=cached,
    )


def test_cache_alarm_never_fires_on_a_healthy_steady_ratio() -> None:
    alarms: list[CacheRegressionAlarm] = []
    detector = CacheRegressionDetector(on_alarm=alarms.append, min_samples=3)

    for i in range(30):
        # Healthy, noisy-but-stable ratio around 0.8.
        cached = 800.0 if i % 2 == 0 else 820.0
        detector.handle_event(_cost_event(at=float(i), cached=cached))

    assert alarms == []


def test_cache_alarm_cold_start_does_not_false_positive() -> None:
    """The first ``min_samples`` events never get compared to a baseline,
    even if they happen to look like a steep drop themselves."""
    alarms: list[CacheRegressionAlarm] = []
    detector = CacheRegressionDetector(on_alarm=alarms.append, min_samples=5)

    ratios = [900.0, 100.0, 900.0, 50.0, 900.0]  # wildly noisy, but pre-baseline
    for i, cached in enumerate(ratios):
        detector.handle_event(_cost_event(at=float(i), cached=cached))

    assert alarms == []


def test_cache_alarm_fires_once_on_a_sustained_regression() -> None:
    alarms: list[CacheRegressionAlarm] = []
    detector = CacheRegressionDetector(
        on_alarm=alarms.append, min_samples=5, drop_fraction=0.5
    )

    # A healthy baseline around 0.9 (cached=900/1000).
    for i in range(10):
        detector.handle_event(_cost_event(at=float(i), cached=900.0))

    # A collapse to 0.1: five samples in a row, well past min_samples so the
    # baseline is established. Only the *first* breach should alarm.
    for i in range(10, 15):
        detector.handle_event(_cost_event(at=float(i), cached=100.0))

    assert len(alarms) == 1
    alarm = alarms[0]
    assert alarm.provider == "acme"
    assert alarm.component == CostComponent.LLM
    assert alarm.at == 10.0
    assert alarm.current_ratio == pytest.approx(0.1)
    assert alarm.baseline_ratio == pytest.approx(0.9)


def test_cache_alarm_recovers_and_can_fire_again_on_a_new_episode() -> None:
    alarms: list[CacheRegressionAlarm] = []
    detector = CacheRegressionDetector(
        on_alarm=alarms.append, min_samples=5, drop_fraction=0.5
    )

    for i in range(10):
        detector.handle_event(_cost_event(at=float(i), cached=900.0))
    for i in range(10, 12):  # dips: one breach episode
        detector.handle_event(_cost_event(at=float(i), cached=100.0))
    for i in range(12, 22):  # recovers back to baseline for a long stretch
        detector.handle_event(_cost_event(at=float(i), cached=900.0))
    for i in range(22, 24):  # a second, independent regression
        detector.handle_event(_cost_event(at=float(i), cached=50.0))

    assert len(alarms) == 2
    assert alarms[0].at == 10.0
    assert alarms[1].at == 22.0


def test_cache_alarm_tracks_provider_and_component_independently() -> None:
    alarms: list[CacheRegressionAlarm] = []
    detector = CacheRegressionDetector(on_alarm=alarms.append, min_samples=5)

    for i in range(10):
        detector.handle_event(_cost_event(at=float(i), cached=900.0, provider="acme"))
    # A completely different provider regressing must not be masked by, or
    # confused with, "acme"'s own healthy baseline.
    for i in range(10, 15):
        detector.handle_event(_cost_event(at=float(i), cached=10.0, provider="other-vendor"))

    assert alarms == []  # other-vendor never had enough samples for a baseline
    # acme itself never regressed.
    assert all(a.provider != "acme" for a in alarms)


def test_cache_alarm_ignores_lines_with_no_cached_units() -> None:
    """A component that never reports caching (most STT/TTS usage) must never
    enter a window at all, let alone trip a false alarm."""
    alarms: list[CacheRegressionAlarm] = []
    detector = CacheRegressionDetector(on_alarm=alarms.append, min_samples=3)

    event = CostRecorded(
        session_id="s",
        at=0.0,
        component=CostComponent.TTS,
        provider="elevenlabs",
        units=300.0,
        unit_name="tts_chars",
        amount=1.5,
        cached_units=None,
    )
    for _ in range(20):
        detector.handle_event(event)

    assert alarms == []


def test_cache_alarm_emits_session_error_when_session_attached() -> None:
    session = make_session()
    alarms: list[CacheRegressionAlarm] = []
    detector = CacheRegressionDetector(
        on_alarm=alarms.append, min_samples=5, drop_fraction=0.5, session=session
    )

    for i in range(10):
        detector.handle_event(_cost_event(at=float(i), cached=900.0))
    detector.handle_event(_cost_event(at=10.0, cached=50.0))

    assert len(alarms) == 1
    errors = [e for e in session.history if isinstance(e, SessionError)]
    assert len(errors) == 1
    assert "acme" in errors[0].message
    assert errors[0].recoverable is True


def test_cache_alarm_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError):
        CacheRegressionDetector(on_alarm=lambda _a: None, window_size=0)
    with pytest.raises(ValueError):
        CacheRegressionDetector(on_alarm=lambda _a: None, min_samples=0)
    with pytest.raises(ValueError):
        CacheRegressionDetector(on_alarm=lambda _a: None, drop_fraction=1.5)
    with pytest.raises(ValueError):
        CacheRegressionDetector(on_alarm=lambda _a: None, drop_fraction=0.0)


# ---------------------------------------------------------------------------
# tring.observability.otel: fake OpenTelemetry for span-tree assertions
# ---------------------------------------------------------------------------


class FakeSpan:
    """Records everything a real ``Span`` would do, plus its parent link."""

    def __init__(self, name: str, parent: FakeSpan | None, attributes: dict[str, Any]) -> None:
        self.name = name
        self.parent = parent
        self.attributes: dict[str, Any] = dict(attributes)
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.ended = False

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(
        self, name: str, attributes: dict[str, Any] | None = None, **_kw: Any
    ) -> None:
        self.events.append((name, dict(attributes or {})))

    def end(self, **_kw: Any) -> None:
        self.ended = True


class FakeContext:
    """Stands in for ``opentelemetry.context.Context``: just carries a span."""

    def __init__(self, span: FakeSpan) -> None:
        self.span = span


class FakeTracer:
    def __init__(self) -> None:
        self.all_spans: list[FakeSpan] = []

    def start_span(
        self,
        name: str,
        context: FakeContext | None = None,
        attributes: dict[str, Any] | None = None,
        **_kw: Any,
    ) -> FakeSpan:
        parent = context.span if context is not None else None
        span = FakeSpan(name, parent, attributes or {})
        self.all_spans.append(span)
        return span


def install_fake_otel(monkeypatch: pytest.MonkeyPatch) -> FakeTracer:
    """Stub ``opentelemetry``/``opentelemetry.trace`` for one test.

    Sets both the submodule entry and the parent package's attribute, which
    is what makes ``from opentelemetry import trace`` resolve correctly
    regardless of which lookup path Python's import system takes.
    """
    tracer = FakeTracer()

    trace_mod = ModuleType("opentelemetry.trace")
    trace_mod.get_tracer = lambda _name: tracer  # type: ignore[attr-defined]
    trace_mod.set_span_in_context = lambda span: FakeContext(span)  # type: ignore[attr-defined]

    otel_pkg = ModuleType("opentelemetry")
    otel_pkg.trace = trace_mod  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "opentelemetry", otel_pkg)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_mod)
    return tracer


def test_otel_exporter_raises_helpful_import_error_without_the_extra() -> None:
    from tring.observability.otel import OtelExporter

    assert "opentelemetry" not in sys.modules
    with pytest.raises(ImportError, match=r"tring\[otel\]"):
        OtelExporter()


def test_otel_session_span_is_root_with_agent_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = install_fake_otel(monkeypatch)
    from tring.observability.otel import OtelExporter

    exporter = OtelExporter()
    exporter.handle_event(
        SessionStarted(
            session_id="s1", at=0.0, agent_name="booking-bot", runtime_mode="cascade"
        )
    )
    exporter.handle_event(
        SessionEnded(session_id="s1", at=5.0, reason="completed", duration_seconds=5.0)
    )

    session_spans = [s for s in tracer.all_spans if s.name == "tring.session"]
    assert len(session_spans) == 1
    span = session_spans[0]
    assert span.parent is None
    assert span.attributes["agent.name"] == "booking-bot"
    assert span.attributes["runtime.mode"] == "cascade"
    assert span.ended is True
    assert span.attributes["session.end_reason"] == "completed"
    assert span.attributes["session.duration_seconds"] == 5.0


def test_otel_turn_span_is_child_of_session_and_ends_on_bot_utterance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = install_fake_otel(monkeypatch)
    from tring.observability.otel import OtelExporter

    exporter = OtelExporter()
    exporter.handle_event(
        SessionStarted(session_id="s1", at=0.0, agent_name="a", runtime_mode="cascade")
    )
    exporter.handle_event(
        UserTranscript(session_id="s1", at=1.0, text="book tuesday", final=True)
    )
    exporter.handle_event(BotUtterance(session_id="s1", at=2.0, text="sure, tuesday works"))

    session_span = next(s for s in tracer.all_spans if s.name == "tring.session")
    turn_span = next(s for s in tracer.all_spans if s.name == "tring.turn")
    assert turn_span.parent is session_span
    assert turn_span.ended is True
    assert turn_span.attributes["turn.outcome"] == "completed"


def test_otel_cost_events_land_as_span_events_with_required_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = install_fake_otel(monkeypatch)
    from tring.observability.otel import OtelExporter

    exporter = OtelExporter()
    exporter.handle_event(
        SessionStarted(session_id="s1", at=0.0, agent_name="a", runtime_mode="cascade")
    )
    exporter.handle_event(UserTranscript(session_id="s1", at=1.0, text="hi", final=True))
    exporter.handle_event(
        CostRecorded(
            session_id="s1",
            at=1.2,
            component=CostComponent.LLM,
            provider="openai",
            units=500.0,
            unit_name="tokens_in",
            amount=0.05,
            estimated=False,
            model="gpt-4o-mini",
            cached_units=400.0,
        )
    )

    turn_span = next(s for s in tracer.all_spans if s.name == "tring.turn")
    assert len(turn_span.events) == 1
    name, attrs = turn_span.events[0]
    assert name == "cost_recorded"
    assert attrs["cost.amount"] == 0.05
    assert attrs["cost.units"] == 500.0
    assert attrs["cost.estimated"] is False
    assert attrs["cost.cached_units"] == 400.0


def test_otel_cost_event_before_any_turn_lands_on_session_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = install_fake_otel(monkeypatch)
    from tring.observability.otel import OtelExporter

    exporter = OtelExporter()
    exporter.handle_event(
        SessionStarted(session_id="s1", at=0.0, agent_name="a", runtime_mode="cascade")
    )
    exporter.handle_event(
        CostRecorded(
            session_id="s1",
            at=0.1,
            component=CostComponent.TELEPHONY,
            provider="twilio",
            units=1.0,
            unit_name="call_setup",
            amount=0.005,
        )
    )

    session_span = next(s for s in tracer.all_spans if s.name == "tring.session")
    assert [e[0] for e in session_span.events] == ["cost_recorded"]


def test_otel_tool_call_is_child_span_of_current_turn_with_latency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = install_fake_otel(monkeypatch)
    from tring.observability.otel import OtelExporter

    exporter = OtelExporter()
    exporter.handle_event(
        SessionStarted(session_id="s1", at=0.0, agent_name="a", runtime_mode="cascade")
    )
    exporter.handle_event(UserTranscript(session_id="s1", at=1.0, text="book it", final=True))
    exporter.handle_event(
        ToolCallStarted(
            session_id="s1",
            at=1.1,
            tool_name="check_appointment",
            call_id="call-1",
            arguments={"phone": "123"},
        )
    )
    exporter.handle_event(
        ToolCallCompleted(
            session_id="s1",
            at=1.6,
            tool_name="check_appointment",
            call_id="call-1",
            ok=True,
            latency_seconds=0.5,
        )
    )
    exporter.handle_event(BotUtterance(session_id="s1", at=1.7, text="you're booked"))

    turn_span = next(s for s in tracer.all_spans if s.name == "tring.turn")
    tool_span = next(s for s in tracer.all_spans if s.name == "tring.tool.check_appointment")
    assert tool_span.parent is turn_span
    assert tool_span.ended is True
    assert tool_span.attributes["tool.ok"] is True
    assert tool_span.attributes["tool.latency_seconds"] == 0.5


def test_otel_tool_call_completed_without_start_is_ignored_not_crashed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_otel(monkeypatch)
    from tring.observability.otel import OtelExporter

    exporter = OtelExporter()
    exporter.handle_event(
        SessionStarted(session_id="s1", at=0.0, agent_name="a", runtime_mode="cascade")
    )
    # No matching ToolCallStarted -- must not raise.
    exporter.handle_event(
        ToolCallCompleted(session_id="s1", at=1.0, tool_name="x", call_id="ghost", ok=True)
    )


def test_otel_new_final_transcript_closes_prior_turn_as_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = install_fake_otel(monkeypatch)
    from tring.observability.otel import OtelExporter

    exporter = OtelExporter()
    exporter.handle_event(
        SessionStarted(session_id="s1", at=0.0, agent_name="a", runtime_mode="cascade")
    )
    exporter.handle_event(UserTranscript(session_id="s1", at=1.0, text="wait", final=True))
    # Caller barges in before any bot reply: a second final transcript.
    exporter.handle_event(
        UserTranscript(session_id="s1", at=1.2, text="never mind", final=True)
    )
    exporter.handle_event(BotUtterance(session_id="s1", at=2.0, text="okay"))

    turn_spans = [s for s in tracer.all_spans if s.name == "tring.turn"]
    assert len(turn_spans) == 2
    assert turn_spans[0].attributes["turn.outcome"] == "interrupted"
    assert turn_spans[0].ended is True
    assert turn_spans[1].attributes["turn.outcome"] == "completed"


def test_otel_dtmf_and_transfer_and_knowledge_and_error_are_span_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = install_fake_otel(monkeypatch)
    from tring.observability.otel import OtelExporter

    exporter = OtelExporter()
    exporter.handle_event(
        SessionStarted(session_id="s1", at=0.0, agent_name="a", runtime_mode="cascade")
    )
    exporter.handle_event(DtmfReceived(session_id="s1", at=0.5, digit="4"))
    exporter.handle_event(
        KnowledgeSearched(
            session_id="s1", at=0.6, query="refund policy", result_count=3,
            latency_seconds=0.2, speculative=True,
        )
    )
    exporter.handle_event(
        SessionError(session_id="s1", at=0.7, message="oops", recoverable=True)
    )
    exporter.handle_event(SessionTransferred(session_id="s1", at=0.8, target="+15551234"))

    session_span = next(s for s in tracer.all_spans if s.name == "tring.session")
    event_names = [name for name, _ in session_span.events]
    assert event_names == [
        "dtmf_received",
        "knowledge_searched",
        "session_error",
        "session_transferred",
    ]


def test_otel_session_ended_without_reply_closes_dangling_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracer = install_fake_otel(monkeypatch)
    from tring.observability.otel import OtelExporter

    exporter = OtelExporter()
    exporter.handle_event(
        SessionStarted(session_id="s1", at=0.0, agent_name="a", runtime_mode="cascade")
    )
    exporter.handle_event(UserTranscript(session_id="s1", at=1.0, text="hello?", final=True))
    exporter.handle_event(
        SessionEnded(session_id="s1", at=3.0, reason="caller_hung_up", duration_seconds=3.0)
    )

    turn_span = next(s for s in tracer.all_spans if s.name == "tring.turn")
    assert turn_span.ended is True
    assert turn_span.attributes["turn.outcome"] == "session_ended"


async def test_otel_exporter_run_consumes_a_live_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The async ``run`` path, exercised against a real ``CallSession`` rather
    than by calling ``handle_event`` directly."""
    tracer = install_fake_otel(monkeypatch)
    from tring.observability.otel import OtelExporter

    session = make_session()
    exporter = OtelExporter()

    import asyncio

    task = asyncio.create_task(exporter.run(session))
    await asyncio.sleep(0)  # let the subscriber attach before anything is emitted

    sid = session.session_id
    session.emit(SessionStarted(session_id=sid, at=0.0, agent_name="a", runtime_mode="cascade"))
    session.emit(UserTranscript(session_id=sid, at=1.0, text="hi", final=True))
    session.emit(BotUtterance(session_id=sid, at=2.0, text="hello"))
    session.emit(SessionEnded(session_id=sid, at=3.0, reason="completed", duration_seconds=3.0))
    session.close()

    await asyncio.wait_for(task, timeout=1.0)

    assert any(s.name == "tring.session" and s.ended for s in tracer.all_spans)
    assert any(s.name == "tring.turn" and s.ended for s in tracer.all_spans)
