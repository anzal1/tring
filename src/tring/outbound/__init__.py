"""Outbound campaigns: dial a list of callees, track outcomes, report the funnel.

See ``docs/V4_PLAN.md`` Track 2. The four pieces, in the order data flows
through them:

    Campaign (who/how) ──▶ CampaignRunner ──▶ Dialer.place_call
                               │                      │
                               │ appends              │ resolves later
                               ▼                      ▼
                        CampaignStore (JSONL)   DialOutcome
                               │
                               ▼
                        CampaignReport.from_records
"""

from __future__ import annotations

from tring.outbound.campaign import Callee, Campaign, DialingPolicy
from tring.outbound.dialer import (
    TELEPHONY_RATES,
    CallHandle,
    Dialer,
    DialOutcome,
    FakeDialer,
    TwilioDialer,
)
from tring.outbound.report import CampaignReport
from tring.outbound.runner import CampaignRunner
from tring.outbound.store import CallStage, CampaignRecord, CampaignStore

__all__ = [
    "TELEPHONY_RATES",
    "CallHandle",
    "CallStage",
    "Callee",
    "Campaign",
    "CampaignRecord",
    "CampaignReport",
    "CampaignRunner",
    "CampaignStore",
    "DialOutcome",
    "Dialer",
    "DialingPolicy",
    "FakeDialer",
    "TwilioDialer",
]
