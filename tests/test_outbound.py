"""Tests for :mod:`tring.outbound`.

Everything here runs on fakes: no network, no Twilio account, no phone
carrier. ``TwilioDialer``'s tests drive its request construction through an
``httpx.MockTransport`` seam -- the exact house pattern
``tests/test_providers_llm.py`` uses for the cloud LLM adapters -- so the
assertions are on the *shape* of the REST call, not on Twilio ever seeing it.
"""

from __future__ import annotations

import asyncio
import base64
import urllib.parse
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pytest
from pydantic import ValidationError

from tring.agent import AgentSpec
from tring.events import (
    SessionEnded,
    ToolCallCompleted,
    ToolCallStarted,
    UserTranscript,
)
from tring.outbound.campaign import Callee, Campaign, DialingPolicy
from tring.outbound.dialer import CallHandle, DialOutcome, FakeDialer, TwilioDialer
from tring.outbound.report import CampaignReport
from tring.outbound.runner import CampaignRunner
from tring.outbound.store import CampaignRecord, CampaignStore
from tring.session import CallSession

# ---------------------------------------------------------------------------
# Campaign / DialingPolicy validation
# ---------------------------------------------------------------------------


def test_campaign_requires_at_least_one_callee() -> None:
    with pytest.raises(ValidationError):
        Campaign(name="c1", callees=[], agent_path="agent.yaml")


def test_dialing_policy_rejects_malformed_quiet_hours() -> None:
    with pytest.raises(ValidationError):
        DialingPolicy(quiet_hours=("22:00", "25:00"))


def test_dialing_policy_rejects_unknown_timezone() -> None:
    with pytest.raises(ValidationError):
        DialingPolicy(timezone="Neverland/Nowhere")


def test_dialing_policy_rejects_non_positive_concurrency() -> None:
    with pytest.raises(ValidationError):
        DialingPolicy(max_concurrency=0)


# ---------------------------------------------------------------------------
# CampaignStore: JSONL append-only persistence
# ---------------------------------------------------------------------------


def test_campaign_store_read_all_is_empty_before_first_write(tmp_path: Path) -> None:
    store = CampaignStore(tmp_path / "missing.jsonl")
    assert store.read_all() == []


def test_campaign_store_round_trips_jsonl(tmp_path: Path) -> None:
    store = CampaignStore(tmp_path / "campaign.jsonl")
    first = CampaignRecord(
        campaign="c1", call_id="CA1", phone="+15550001111", attempt=1, status="dialed", at=1.0
    )
    second = CampaignRecord(
        campaign="c1",
        call_id="CA1",
        phone="+15550001111",
        attempt=1,
        status="connected",
        at=2.0,
        answered_by="human",
        call_duration_seconds=30.0,
    )
    store.append(first)
    store.append(second)

    # A fresh CampaignStore instance pointed at the same path stands in for
    # a reopen after a process restart -- the whole point of JSONL over an
    # in-memory list.
    reopened = CampaignStore(store.path)
    assert reopened.read_all() == [first, second]


# ---------------------------------------------------------------------------
# CampaignRunner + FakeDialer: retries, concurrency, quiet hours
# ---------------------------------------------------------------------------


async def _no_sleep(_seconds: float) -> None:
    """A ``sleep`` stand-in that returns instantly, for tests that assert on
    backoff *durations requested* rather than actually waiting them out."""


async def test_fake_dialer_end_to_end_campaign_with_retries(tmp_path: Path) -> None:
    phone = "+15550001111"
    campaign = Campaign(
        name="c1",
        callees=[Callee(phone=phone)],
        agent_path="agent.yaml",
        policy=DialingPolicy(retries=1, retry_backoff_s=5.0),
    )
    store = CampaignStore(tmp_path / "c1.jsonl")
    # One scripted failure, then the dialer's default (connected) outcome
    # once that one-entry script is exhausted -- exercises exactly one
    # retry succeeding on the second attempt.
    dialer = FakeDialer(scripts={phone: [DialOutcome(connected=False, error="busy")]})
    sleeps: list[float] = []

    async def recording_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    runner = CampaignRunner(campaign, dialer, store, sleep=recording_sleep)
    await runner.run()

    records = store.read_all()
    assert [(r.attempt, r.status) for r in records] == [
        (1, "dialed"),
        (2, "dialed"),
        (2, "connected"),
    ]
    assert sleeps == [5.0]  # exactly one backoff, between attempt 1 and 2
    assert records[-1].answered_by == "human"
    assert len(dialer.calls) == 2


async def test_fake_dialer_campaign_exhausts_retries_and_records_failed(
    tmp_path: Path,
) -> None:
    phone = "+15550002222"
    campaign = Campaign(
        name="c1",
        callees=[Callee(phone=phone)],
        agent_path="agent.yaml",
        policy=DialingPolicy(retries=1, retry_backoff_s=1.0),
    )
    store = CampaignStore(tmp_path / "c1.jsonl")
    dialer = FakeDialer(default=DialOutcome(connected=False, error="no-answer"))
    runner = CampaignRunner(campaign, dialer, store, sleep=_no_sleep)
    await runner.run()

    records = store.read_all()
    assert [(r.attempt, r.status) for r in records] == [
        (1, "dialed"),
        (2, "dialed"),
        (2, "failed"),
    ]
    assert records[-1].error == "no-answer"


async def test_quiet_hours_defers_dialing_until_window_end(tmp_path: Path) -> None:
    campaign = Campaign(
        name="c1",
        callees=[Callee(phone="+15550003333")],
        agent_path="agent.yaml",
        policy=DialingPolicy(quiet_hours=("22:00", "07:00"), timezone="UTC", retries=0),
    )
    store = CampaignStore(tmp_path / "c1.jsonl")
    dialer = FakeDialer()
    fixed_now = datetime(2026, 9, 19, 23, 0, tzinfo=ZoneInfo("UTC"))
    sleeps: list[float] = []

    async def recording_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    runner = CampaignRunner(
        campaign, dialer, store, now=lambda: fixed_now, sleep=recording_sleep
    )
    await runner.run()

    # 23:00 -> 07:00 next day is 8 hours, and it is the *only* wait: no
    # retries are configured, so nothing else should have called sleep.
    assert sleeps == [8 * 3600.0]
    records = store.read_all()
    assert records[0].status == "deferred"
    assert records[-1].status == "connected"


async def test_outside_quiet_hours_dials_without_deferral(tmp_path: Path) -> None:
    campaign = Campaign(
        name="c1",
        callees=[Callee(phone="+15550004444")],
        agent_path="agent.yaml",
        policy=DialingPolicy(quiet_hours=("22:00", "07:00"), timezone="UTC", retries=0),
    )
    store = CampaignStore(tmp_path / "c1.jsonl")
    dialer = FakeDialer()
    fixed_now = datetime(2026, 9, 19, 12, 0, tzinfo=ZoneInfo("UTC"))
    sleeps: list[float] = []

    async def recording_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    runner = CampaignRunner(
        campaign, dialer, store, now=lambda: fixed_now, sleep=recording_sleep
    )
    await runner.run()

    assert sleeps == []
    records = store.read_all()
    assert records[0].status == "dialed"


class _TrackingDialer:
    """A :class:`~tring.outbound.dialer.Dialer` that reports how many calls
    were in flight at once, to prove the runner's concurrency cap for real."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.placed = 0
        self._resolvers: list[asyncio.Task[None]] = []

    async def place_call(self, callee: Callee) -> CallHandle:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.placed += 1
        call_id = f"CA{self.placed}"
        future: asyncio.Future[DialOutcome] = asyncio.get_running_loop().create_future()

        async def _resolve_later() -> None:
            await asyncio.sleep(0.02)
            self.active -= 1
            future.set_result(DialOutcome(connected=True))

        self._resolvers.append(asyncio.create_task(_resolve_later()))
        return CallHandle(call_id=call_id, callee=callee, outcome=future)


async def test_concurrency_cap_is_enforced(tmp_path: Path) -> None:
    callees = [Callee(phone=f"+1555000{i:04d}") for i in range(6)]
    campaign = Campaign(
        name="c1",
        callees=callees,
        agent_path="agent.yaml",
        policy=DialingPolicy(max_concurrency=2, retries=0),
    )
    store = CampaignStore(tmp_path / "c1.jsonl")
    dialer = _TrackingDialer()
    runner = CampaignRunner(campaign, dialer, store)
    await runner.run()

    assert dialer.placed == 6
    assert dialer.max_active == 2  # never more than the policy's cap


# ---------------------------------------------------------------------------
# CampaignRunner: session-driven outcome tracking (watch_session, mark_outcome)
# ---------------------------------------------------------------------------


async def test_watch_session_marks_conversation_and_tool_call_outcome(
    tmp_path: Path,
) -> None:
    phone = "+15550005555"
    campaign = Campaign(
        name="c1",
        callees=[Callee(phone=phone)],
        agent_path="agent.yaml",
        policy=DialingPolicy(retries=0),
    )
    store = CampaignStore(tmp_path / "c1.jsonl")
    dialer = FakeDialer()
    runner = CampaignRunner(campaign, dialer, store)
    await runner.run()

    call_id = next(r.call_id for r in store.read_all() if r.status == "connected")

    agent = AgentSpec(name="outbound-agent", persona="You are a scheduling assistant.")
    session = CallSession(agent, session_id="s1", clock=lambda: 0.0)
    watch_task = runner.watch_session(session, call_id, outcome_tool_name="record_outcome")
    await asyncio.sleep(0)  # let the watcher subscribe before events are emitted

    session.emit(UserTranscript(session_id="s1", at=0.0, text="Yes, Tuesday works", final=True))
    session.emit(
        ToolCallStarted(
            session_id="s1",
            at=1.0,
            tool_name="record_outcome",
            call_id="tc1",
            arguments={"outcome": "booked"},
        )
    )
    session.emit(
        ToolCallCompleted(
            session_id="s1", at=1.5, tool_name="record_outcome", call_id="tc1", ok=True
        )
    )
    session.emit(SessionEnded(session_id="s1", at=2.0))

    await watch_task

    records = [r for r in store.read_all() if r.call_id == call_id]
    statuses = [r.status for r in records]
    assert "conversation" in statuses
    outcome_record = next(r for r in records if r.status == "outcome")
    assert outcome_record.outcome_label == "booked"


async def test_watch_session_records_no_outcome_when_window_expires(
    tmp_path: Path,
) -> None:
    phone = "+15550006666"
    campaign = Campaign(
        name="c1",
        callees=[Callee(phone=phone)],
        agent_path="agent.yaml",
        policy=DialingPolicy(retries=0, callback_window_s=0.0),
    )
    store = CampaignStore(tmp_path / "c1.jsonl")
    dialer = FakeDialer()
    runner = CampaignRunner(campaign, dialer, store)
    await runner.run()
    call_id = next(r.call_id for r in store.read_all() if r.status == "connected")

    agent = AgentSpec(name="outbound-agent", persona="p")
    session = CallSession(agent, session_id="s1", clock=lambda: 0.0)
    watch_task = runner.watch_session(session, call_id)
    await asyncio.sleep(0)

    session.emit(SessionEnded(session_id="s1", at=1.0))
    await watch_task

    records = [r for r in store.read_all() if r.call_id == call_id]
    assert records[-1].status == "no_outcome"


def test_mark_outcome_rejects_unknown_call_id(tmp_path: Path) -> None:
    campaign = Campaign(
        name="c1", callees=[Callee(phone="+1555")], agent_path="agent.yaml"
    )
    store = CampaignStore(tmp_path / "c1.jsonl")
    runner = CampaignRunner(campaign, FakeDialer(), store)
    with pytest.raises(KeyError):
        runner.mark_outcome("no-such-call", "booked")


# ---------------------------------------------------------------------------
# CampaignReport
# ---------------------------------------------------------------------------


def _rec(
    campaign: str, call_id: str, phone: str, status: str, outcome_label: str | None = None
) -> CampaignRecord:
    """Shorthand for a one-off ``CampaignRecord`` in a report test's fixture
    list, where only ``status`` (and occasionally ``outcome_label``) differ
    between rows and the rest is boilerplate."""
    return CampaignRecord(
        campaign=campaign,
        call_id=call_id,
        phone=phone,
        attempt=1,
        status=status,  # type: ignore[arg-type]
        at=0.0,
        outcome_label=outcome_label,
    )


def test_campaign_report_computes_funnel_and_reuses_denominator_ladder() -> None:
    records = [
        _rec("c1", "CA1", "+1", "dialed"),
        _rec("c1", "CA1", "+1", "connected"),
        _rec("c1", "CA1", "+1", "conversation"),
        _rec("c1", "CA1", "+1", "outcome", outcome_label="booked"),
        _rec("c1", "CA2", "+2", "dialed"),
        _rec("c1", "CA2", "+2", "failed"),
        # A record from a different campaign must not leak into this report.
        _rec("other", "CA9", "+9", "dialed"),
    ]
    report = CampaignReport.from_records("c1", records, total_cost=10.0)

    assert report.dials == 2
    assert report.connected == 1
    assert report.conversations == 1
    assert report.outcomes == 1
    assert report.failed == 1
    assert report.by_outcome == {"booked": 1}
    assert report.ladder["cost_per_dial"] == pytest.approx(5.0)
    assert report.ladder["cost_per_outcome"] == pytest.approx(10.0)

    text = report.as_text()
    assert "Campaign Report: c1" in text
    assert "booked" in text


def test_campaign_report_zero_volume_stage_is_none_not_zero() -> None:
    records = [_rec("c1", "CA1", "+1", "dialed"), _rec("c1", "CA1", "+1", "failed")]
    report = CampaignReport.from_records("c1", records, total_cost=3.0)
    assert report.outcomes == 0
    assert report.ladder["cost_per_outcome"] is None


# ---------------------------------------------------------------------------
# TwilioDialer: request construction via httpx.MockTransport, AMD passthrough
# ---------------------------------------------------------------------------


def _twilio_dialer(
    monkeypatch: pytest.MonkeyPatch,
    captured: list[httpx.Request],
    **kwargs: object,
) -> TwilioDialer:
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC_test_account")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "test_auth_token")

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(201, json={"sid": "CA_fake_0001"})

    return TwilioDialer(
        from_number="+15550009999",
        stream_webhook_url="https://example.org/twilio/stream",
        transport=httpx.MockTransport(handler),
        **kwargs,  # type: ignore[arg-type]
    )


async def test_twilio_dialer_builds_create_call_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []
    dialer = _twilio_dialer(monkeypatch, captured)

    handle = await dialer.place_call(Callee(phone="+15551112222"))

    assert handle.call_id == "CA_fake_0001"
    assert len(captured) == 1
    request = captured[0]
    assert request.method == "POST"
    assert request.url.path == "/2010-04-01/Accounts/AC_test_account/Calls.json"

    body = urllib.parse.parse_qs(request.content.decode())
    assert body["To"] == ["+15551112222"]
    assert body["From"] == ["+15550009999"]
    assert body["Url"] == ["https://example.org/twilio/stream"]
    assert "MachineDetection" not in body

    auth_header = request.headers["authorization"]
    assert auth_header.startswith("Basic ")
    decoded = base64.b64decode(auth_header.removeprefix("Basic ").encode()).decode()
    assert decoded == "AC_test_account:test_auth_token"


async def test_twilio_dialer_passes_through_amd_and_callback_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []
    dialer = _twilio_dialer(
        monkeypatch,
        captured,
        machine_detection="DetectMessageEnd",
        machine_detection_timeout_ms=8000,
        status_callback_url="https://example.org/status",
        async_amd_status_callback_url="https://example.org/amd",
    )

    await dialer.place_call(Callee(phone="+15551112222"))

    body = urllib.parse.parse_qs(captured[0].content.decode())
    assert body["MachineDetection"] == ["DetectMessageEnd"]
    assert body["MachineDetectionTimeout"] == ["8000"]
    assert body["StatusCallback"] == ["https://example.org/status"]
    assert body["AsyncAmdStatusCallback"] == ["https://example.org/amd"]
    assert body["AsyncAmd"] == ["true"]


async def test_twilio_dialer_missing_credential_env_var_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TWILIO_ACCOUNT_SID", raising=False)
    monkeypatch.delenv("TWILIO_AUTH_TOKEN", raising=False)
    dialer = TwilioDialer(
        from_number="+1",
        stream_webhook_url="https://example.org/stream",
        transport=httpx.MockTransport(lambda r: httpx.Response(201, json={"sid": "x"})),
    )
    with pytest.raises(RuntimeError):
        await dialer.place_call(Callee(phone="+15551112222"))


async def test_twilio_dialer_report_status_resolves_with_amd_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []
    dialer = _twilio_dialer(monkeypatch, captured)
    handle = await dialer.place_call(Callee(phone="+15551112222"))

    # Async AMD delivers AnsweredBy on its own webhook, ahead of the call's
    # eventual terminal status -- this must not resolve the future yet.
    dialer.report_status({"CallSid": "CA_fake_0001", "AnsweredBy": "machine_start"})
    assert not handle.outcome.done()

    dialer.report_status(
        {"CallSid": "CA_fake_0001", "CallStatus": "completed", "CallDuration": "42"}
    )
    assert handle.outcome.done()
    outcome = handle.outcome.result()
    assert outcome.connected is True
    assert outcome.answered_by == "machine_start"
    assert outcome.call_duration_seconds == 42.0


async def test_twilio_dialer_report_status_no_answer_is_not_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []
    dialer = _twilio_dialer(monkeypatch, captured)
    handle = await dialer.place_call(Callee(phone="+15551112222"))

    dialer.report_status({"CallSid": "CA_fake_0001", "CallStatus": "no-answer"})

    outcome = handle.outcome.result()
    assert outcome.connected is False
    assert outcome.error == "call ended: no-answer"


async def test_twilio_dialer_report_status_ignores_unknown_call_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []
    dialer = _twilio_dialer(monkeypatch, captured)
    # Must not raise: Twilio retries webhooks, and a call this dialer never
    # placed (or already resolved) is a normal, silent no-op.
    dialer.report_status({"CallSid": "never-placed", "CallStatus": "completed"})
