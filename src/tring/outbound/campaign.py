"""Campaign definition: who to call, with what agent, under what dialing rules.

Like :class:`~tring.agent.AgentSpec`, a :class:`Campaign` is plain data
(pydantic v2, no callables): a list of callees, a reference to the agent that
should answer once a call connects, and a :class:`DialingPolicy` describing
how aggressively and how considerately to place the calls. Nothing here talks
to a dialer or a clock -- that is :mod:`tring.outbound.runner`'s job -- so a
campaign stays serializable, diffable, and reviewable the same way an
``AgentSpec`` YAML file is.
"""

from __future__ import annotations

import re
from datetime import time as dt_time
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

#: ``HH:MM`` 24-hour local time, e.g. "22:00". Matches the format
#: ``datetime.strptime(s, "%H:%M")`` accepts, checked with a regex instead so
#: a malformed policy fails at model-validation time with a field-level
#: error rather than inside the runner mid-campaign.
_HHMM_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def parse_hhmm(value: str) -> dt_time:
    """Parse an ``"HH:MM"`` string into a :class:`datetime.time`.

    Shared by :class:`DialingPolicy`'s validator and
    :mod:`tring.outbound.runner`'s quiet-hours arithmetic, so the accepted
    format is defined exactly once.
    """
    match = _HHMM_RE.match(value)
    if match is None:
        raise ValueError(f"expected 24-hour \"HH:MM\", got {value!r}")
    return dt_time(hour=int(match.group(1)), minute=int(match.group(2)))


class Callee(BaseModel):
    """One person (or number) a campaign will dial.

    ``metadata`` is opaque to the runner and the dialer alike -- it exists so
    an agent's tools (looked up by ``agent_path``) can personalize the call
    ("say Priya's name", "reference order #4021") without the outbound
    package needing to know what a CRM record looks like.
    """

    phone: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("phone")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("phone must not be blank")
        return v


class DialingPolicy(BaseModel):
    """How a campaign should be dialed: pace, retries, courtesy hours.

    Args:
        max_concurrency: at most this many calls in flight at once. This is
            the number the :class:`~tring.outbound.runner.CampaignRunner`
            actually enforces (an ``asyncio.Semaphore``); it is not a request
            to Twilio, which has its own account-level concurrent-call caps
            that this policy does not know about.
        retries: additional attempts after a call fails to connect. ``0``
            means "call once, never retry."
        retry_backoff_s: flat delay between a failed attempt and the next
            retry for the *same* callee. Flat rather than exponential on
            purpose: a human callee is not a flaky server, and a fixed
            "try again in ten minutes" is both easier to reason about and
            easier to keep inside quiet hours than a backoff curve that
            grows unpredictably.
        quiet_hours: an optional ``(start, end)`` pair of ``"HH:MM"`` local
            time strings (in ``timezone``) during which the runner will not
            *place* new calls -- it defers them until ``end``. A window
            where ``start > end`` (e.g. ``("22:00", "07:00")``) is
            understood as wrapping past midnight.
        timezone: an IANA zone name (``zoneinfo`` key) that ``quiet_hours``
            is interpreted in. Defaults to UTC so a policy with no
            ``quiet_hours`` never needs one.
        callback_window_s: how long after a call's session ends the runner
            keeps accepting an out-of-band outcome
            (:meth:`~tring.outbound.runner.CampaignRunner.mark_outcome`) --
            e.g. a CRM webhook that lands a few seconds after the caller
            hangs up -- before giving up and recording ``no_outcome``. See
            :meth:`~tring.outbound.runner.CampaignRunner.watch_session`.
    """

    max_concurrency: int = 4
    retries: int = 2
    retry_backoff_s: float = 30.0
    quiet_hours: tuple[str, str] | None = None
    timezone: str = "UTC"
    callback_window_s: float = 300.0

    @field_validator("quiet_hours")
    @classmethod
    def _valid_hhmm_pair(cls, v: tuple[str, str] | None) -> tuple[str, str] | None:
        if v is not None:
            parse_hhmm(v[0])
            parse_hhmm(v[1])
        return v

    @field_validator("max_concurrency")
    @classmethod
    def _positive_concurrency(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_concurrency must be at least 1")
        return v

    @field_validator("retries")
    @classmethod
    def _non_negative_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("retries must not be negative")
        return v

    @model_validator(mode="after")
    def _timezone_is_loadable(self) -> DialingPolicy:
        # Fail at construction time, not three hours into a campaign: a typo
        # in an IANA zone name (e.g. "Asia/Kolkota") should surface the
        # moment the policy is built, from the same place that already
        # validates everything else about it.
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        try:
            ZoneInfo(self.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone {self.timezone!r}") from exc
        return self


class Campaign(BaseModel):
    """A batch of outbound calls: who to dial, which agent answers, how.

    ``agent_path`` is a path to an ``AgentSpec`` YAML file, not the loaded
    spec itself -- keeping the campaign plain, serializable data the same
    way ``ToolDef.handler`` keeps a tool a name rather than a bound callable
    (see ``agent.py``). Loading it (``AgentSpec.from_yaml(campaign.agent_path)``)
    and wiring the resulting runtime to a connected call is the transport's
    job, not this package's: :mod:`tring.outbound` only gets a call *placed*
    and its outcome tracked, matching the "campaigns" scope in
    ``docs/V4_PLAN.md`` Track 2.
    """

    name: str
    callees: list[Callee]
    agent_path: str
    policy: DialingPolicy = Field(default_factory=DialingPolicy)

    @field_validator("callees")
    @classmethod
    def _at_least_one_callee(cls, v: list[Callee]) -> list[Callee]:
        if not v:
            raise ValueError("campaign must have at least one callee")
        return v


__all__ = ["Callee", "Campaign", "DialingPolicy", "parse_hhmm"]
