"""``CampaignRunner`` -- the asyncio orchestrator that drives a ``Campaign``.

One runner ties the three other modules together: it paces
:class:`~tring.outbound.dialer.Dialer` calls under
:class:`~tring.outbound.campaign.DialingPolicy`'s concurrency cap, retries
and quiet hours, and writes every stage a callee reaches to a
:class:`~tring.outbound.store.CampaignStore` as it happens -- never buffered
in memory only, so a killed process loses nothing already appended.

Concurrency model
------------------

One short-lived task per callee, gated by one ``asyncio.Semaphore`` sized to
``policy.max_concurrency``. There is no separate scheduler loop: each task
*is* the retry loop for its callee, from first dial to final outcome, which
keeps the state machine for "what happens to one callee" readable in one
function (:meth:`CampaignRunner._run_callee`) instead of spread across a
central dispatcher.

Deterministic tests
--------------------

Both time sources are injectable: ``now`` (a zero-arg clock returning
``datetime``, for quiet-hours arithmetic) and ``sleep`` (an
``asyncio.sleep``-shaped coroutine function, for backoff and quiet-hours
waits). A test supplies a scripted clock and a ``sleep`` that advances it
instantly instead of actually waiting -- the same "inject the clock"
discipline ``CallSession(clock=...)`` uses (see ``docs/EXTENDING.md``,
"Deterministic clocks"), extended to cover the wall-clock wait itself, not
only timestamp labeling.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from tring.events import SessionEnded, ToolCallCompleted, ToolCallStarted, UserTranscript
from tring.outbound.campaign import Callee, Campaign, parse_hhmm
from tring.outbound.dialer import Dialer, DialOutcome
from tring.outbound.store import CallStage, CampaignRecord, CampaignStore
from tring.session import CallSession

#: Clock and sleep are both injectable so quiet-hours and backoff waits are
#: testable without a real asyncio.sleep -- see the module docstring.
NowFn = Callable[[], datetime]
SleepFn = Callable[[float], Awaitable[None]]


@dataclass
class _CallInfo:
    """What :meth:`CampaignRunner.mark_outcome` needs to know about a call
    it did not place itself (e.g. one reported from a webhook route)."""

    phone: str
    attempt: int


class CampaignRunner:
    """Drives one :class:`Campaign` to completion against one :class:`Dialer`.

    Args:
        campaign: what to dial and under what policy.
        dialer: how to dial it (:class:`~tring.outbound.dialer.FakeDialer`
            in tests, :class:`~tring.outbound.dialer.TwilioDialer` live).
        store: where every stage transition is appended.
        now: clock for quiet-hours arithmetic. Defaults to
            ``datetime.now`` in the policy's configured timezone.
        sleep: awaited for both quiet-hours deferral and retry backoff.
            Defaults to ``asyncio.sleep``.
    """

    def __init__(
        self,
        campaign: Campaign,
        dialer: Dialer,
        store: CampaignStore,
        *,
        now: NowFn | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        self.campaign = campaign
        self.dialer = dialer
        self.store = store
        self._tz = ZoneInfo(campaign.policy.timezone)
        self._now: NowFn = now or (lambda: datetime.now(self._tz))
        self._sleep: SleepFn = sleep or asyncio.sleep

        self._calls: dict[str, _CallInfo] = {}
        self._outcome_events: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------------ run

    async def run(self) -> None:
        """Dial every callee to completion. Returns once all are resolved.

        Build the campaign's :class:`~tring.outbound.report.CampaignReport`
        from ``self.store.read_all()`` afterwards -- this method's job is
        orchestration, not reporting, matching the split between
        ``runtimes/cascade.py`` (drives a call) and ``cost/report.py``
        (summarizes one afterwards).
        """
        semaphore = asyncio.Semaphore(self.campaign.policy.max_concurrency)
        # A bounded number of *tasks running at once* is not the same
        # guarantee as a bounded number of *dial attempts in flight*
        # unless the semaphore is held across the whole attempt, retries
        # included -- acquired once per callee task in _run_callee, not
        # re-acquired per retry, so a callee mid-backoff still occupies its
        # concurrency slot rather than freeing it for a different callee's
        # dial to race in.
        await asyncio.gather(
            *(self._run_callee(callee, semaphore) for callee in self.campaign.callees)
        )

    async def _run_callee(self, callee: Callee, semaphore: asyncio.Semaphore) -> None:
        async with semaphore:
            wait = self._quiet_hours_wait_seconds(self._now())
            if wait > 0:
                self._append(call_id="", phone=callee.phone, attempt=0, status="deferred")
                await self._sleep(wait)

            max_attempts = 1 + self.campaign.policy.retries
            last_call_id = ""
            last_outcome: DialOutcome | None = None

            for attempt in range(1, max_attempts + 1):
                handle = await self.dialer.place_call(callee)
                last_call_id = handle.call_id
                self._calls[handle.call_id] = _CallInfo(phone=callee.phone, attempt=attempt)
                self._append(handle.call_id, callee.phone, attempt, "dialed")

                outcome = await handle.outcome
                last_outcome = outcome
                if outcome.connected:
                    self._append(
                        handle.call_id,
                        callee.phone,
                        attempt,
                        "connected",
                        answered_by=outcome.answered_by,
                        call_duration_seconds=outcome.call_duration_seconds,
                    )
                    return

                if attempt < max_attempts:
                    await self._sleep(self.campaign.policy.retry_backoff_s)

            self._append(
                last_call_id,
                callee.phone,
                max_attempts,
                "failed",
                error=last_outcome.error if last_outcome else "no outcome",
            )

    # ------------------------------------------------------ quiet hours gate

    def _quiet_hours_wait_seconds(self, now: datetime) -> float:
        """Seconds to wait before dialing is allowed, or ``0.0`` if it is now.

        A window where ``start > end`` (e.g. ``("22:00", "07:00")``) wraps
        past midnight; everything else is the ordinary same-day case.
        """
        quiet = self.campaign.policy.quiet_hours
        if quiet is None:
            return 0.0

        local = now.astimezone(self._tz)
        start = parse_hhmm(quiet[0])
        end = parse_hhmm(quiet[1])
        window_start = local.replace(
            hour=start.hour, minute=start.minute, second=0, microsecond=0
        )
        window_end = local.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)

        if start <= end:
            in_window = window_start <= local < window_end
            resume_at = window_end
        else:
            in_window = local >= window_start or local < window_end
            resume_at = window_end if local < window_end else window_end + timedelta(days=1)

        if not in_window:
            return 0.0
        return max(0.0, (resume_at - local).total_seconds())

    # -------------------------------------------------------- outcome intake

    def watch_session(
        self, session: CallSession, call_id: str, outcome_tool_name: str = "record_outcome"
    ) -> asyncio.Task[None]:
        """Subscribe to ``session`` and translate its events into stages.

        Two things this call attempt's :class:`CampaignRecord` log learns
        about purely by watching the session's event stream, with no
        change needed to the runtime or the agent's tool handlers:

        - **conversation**: the first ``UserTranscript(final=True)`` means
          the caller said something real, not just that the line picked up
          (an answering machine greeting does not produce one).
        - **outcome**: a call to the tool named ``outcome_tool_name``
          (default ``"record_outcome"`` -- give the agent a tool by this
          name with an ``outcome`` argument to let the model itself close
          out a campaign row, e.g. "booked" vs. "not interested").
          ``ToolCallStarted.arguments`` is captured when the call begins and
          consumed on the matching ``ToolCallCompleted(ok=True)``, so a
          tool that fails or times out never records a false outcome.

        If neither happens before ``policy.callback_window_s`` elapses
        after the session ends, the row is finalized ``"no_outcome"``
        rather than left open forever -- see :meth:`mark_outcome` for the
        alternative, explicit path (a CRM webhook, a human agent's note)
        that can still land within that same window.

        Returns the background task driving this; callers that want to
        await full completion (tests, mainly) can ``await`` it directly.
        """
        return asyncio.create_task(self._watch_session(session, call_id, outcome_tool_name))

    async def _watch_session(
        self, session: CallSession, call_id: str, outcome_tool_name: str
    ) -> None:
        pending_args: dict[str, dict[str, Any]] = {}
        outcome_event = self._outcome_events.setdefault(call_id, asyncio.Event())
        conversation_marked = False

        async for event in session.subscribe():
            if isinstance(event, UserTranscript) and event.final and not conversation_marked:
                conversation_marked = True
                self._append(call_id, self._phone_for(call_id), self._attempt_for(call_id),
                              "conversation")
            elif isinstance(event, ToolCallStarted) and event.tool_name == outcome_tool_name:
                pending_args[event.call_id] = dict(event.arguments)
            elif (
                isinstance(event, ToolCallCompleted)
                and event.tool_name == outcome_tool_name
                and event.ok
            ):
                args = pending_args.pop(event.call_id, {})
                label = str(args.get("outcome") or args.get("label") or "unknown")
                self.mark_outcome(call_id, label)
            elif isinstance(event, SessionEnded):
                break

        if outcome_event.is_set():
            return
        window = self.campaign.policy.callback_window_s
        if window > 0:
            try:
                await asyncio.wait_for(self._wait_forever_or_set(outcome_event), timeout=window)
                return
            except TimeoutError:
                pass
        if not outcome_event.is_set():
            self._append(
                call_id, self._phone_for(call_id), self._attempt_for(call_id), "no_outcome"
            )

    @staticmethod
    async def _wait_forever_or_set(event: asyncio.Event) -> None:
        await event.wait()

    def mark_outcome(self, call_id: str, label: str) -> CampaignRecord:
        """Explicitly record a business outcome for ``call_id``.

        The escape hatch alongside :meth:`watch_session`'s tool-call path:
        use this from a CRM webhook, an agent-assist console, or a batch
        script reconciling outcomes after the fact. Raises ``KeyError`` if
        ``call_id`` was never seen by this runner (from ``place_call`` or a
        prior :meth:`watch_session` attachment) -- silently accepting an
        unknown id would let a typo in a webhook payload attribute an
        outcome to the wrong campaign row.
        """
        info = self._calls.get(call_id)
        if info is None:
            raise KeyError(f"no call known to this runner with call_id={call_id!r}")
        record = self._append(
            call_id, info.phone, info.attempt, "outcome", outcome_label=label
        )
        event = self._outcome_events.get(call_id)
        if event is not None:
            event.set()
        return record

    # ---------------------------------------------------------------- lookup

    def _phone_for(self, call_id: str) -> str:
        info = self._calls.get(call_id)
        return info.phone if info is not None else ""

    def _attempt_for(self, call_id: str) -> int:
        info = self._calls.get(call_id)
        return info.attempt if info is not None else 0

    def _append(
        self,
        call_id: str,
        phone: str,
        attempt: int,
        status: CallStage,
        *,
        answered_by: str | None = None,
        call_duration_seconds: float | None = None,
        outcome_label: str | None = None,
        error: str | None = None,
    ) -> CampaignRecord:
        record = CampaignRecord(
            campaign=self.campaign.name,
            call_id=call_id,
            phone=phone,
            attempt=attempt,
            status=status,
            at=time.time(),
            answered_by=answered_by,
            call_duration_seconds=call_duration_seconds,
            outcome_label=outcome_label,
            error=error,
        )
        self.store.append(record)
        return record


__all__ = ["CampaignRunner", "NowFn", "SleepFn"]
