"""The ``Dialer`` protocol: place one call, learn its outcome asynchronously.

Placing an outbound call and *knowing what happened* are two different
network events, sometimes minutes apart -- the REST call that dials returns
the instant Twilio accepts the request, long before anyone picks up. Every
:class:`Dialer` therefore returns a :class:`CallHandle` immediately and
resolves its ``outcome`` future later, whether "later" means "the next line
of a test" (:class:`FakeDialer`) or "whenever Twilio's status webhook
arrives" (:class:`TwilioDialer`, see :meth:`TwilioDialer.report_status`).
:class:`~tring.outbound.runner.CampaignRunner` awaits that one future and
does not care which kind it is.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from tring.cost.rates import Rate
from tring.events import CostComponent
from tring.outbound.campaign import Callee


@dataclass(frozen=True)
class DialOutcome:
    """What became of one placed call, once it is known.

    ``answered_by`` carries Twilio's answering-machine-detection verdict
    verbatim when AMD was requested (``"human"``, ``"machine_start"``,
    ``"machine_end_beep"``, ``"machine_end_silence"``, ``"machine_end_other"``,
    ``"fax"``, ``"unknown"`` -- see
    https://www.twilio.com/docs/voice/answering-machine-detection, verified
    2026-09) and is ``None`` when AMD was not requested or is not
    applicable (e.g. :class:`FakeDialer`).
    """

    connected: bool
    answered_by: str | None = None
    call_duration_seconds: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class CallHandle:
    """What ``place_call`` hands back: an id and a promise of an outcome."""

    call_id: str
    callee: Callee
    outcome: asyncio.Future[DialOutcome]


@runtime_checkable
class Dialer(Protocol):
    """Anything that can place one outbound call.

    A ``Protocol`` (not an ABC) so a dialer implementation never needs to
    import this module just to be recognized as one -- the same shape-typing
    choice ``providers/base.py``'s ``STTProvider`` et al. make for a
    different reason (there, an ABC with ``@abstractmethod``; here, a bare
    Protocol, because a dialer has exactly one method and no shared state
    worth a base class).
    """

    async def place_call(self, callee: Callee) -> CallHandle: ...


class FakeDialer:
    """A scripted :class:`Dialer` for tests: no network, deterministic outcomes.

    Args:
        scripts: per-phone-number queues of outcomes. Each call to
            ``place_call`` for a phone pops the next entry, so a campaign
            that retries a callee sees a different scripted outcome per
            attempt (e.g. ``["busy", "connected"]`` to exercise a
            retry-then-succeed path). A phone with no script, or one whose
            queue has been exhausted, falls back to ``default``.
        default: the outcome used once a phone's script (if any) runs out.

    ``calls`` records every ``Callee`` passed to ``place_call``, in order,
    so a test can assert on dial order and count without instrumenting the
    runner itself.
    """

    def __init__(
        self,
        scripts: Mapping[str, list[DialOutcome]] | None = None,
        default: DialOutcome | None = None,
    ) -> None:
        self._scripts: dict[str, list[DialOutcome]] = {
            phone: list(outcomes) for phone, outcomes in (scripts or {}).items()
        }
        self._default = default or DialOutcome(connected=True, answered_by="human")
        self.calls: list[Callee] = []

    async def place_call(self, callee: Callee) -> CallHandle:
        self.calls.append(callee)
        # A real dialer's REST call and its answer are two separate network
        # round trips; one ``sleep(0)`` is enough to force this fake through
        # the same two-step (place, then later resolve) shape so a test
        # exercising cancellation or concurrency limits sees a real await
        # point instead of a handle that is already done before anyone
        # could have raced it.
        await asyncio.sleep(0)
        queue = self._scripts.get(callee.phone)
        outcome = queue.pop(0) if queue else self._default
        future: asyncio.Future[DialOutcome] = asyncio.get_running_loop().create_future()
        future.set_result(outcome)
        return CallHandle(call_id=uuid.uuid4().hex, callee=callee, outcome=future)


# https://www.twilio.com/docs/voice/api/call-resource#call-status-values --
# verified 2026-09. "completed" is the only terminal status that means the
# call was actually answered and later ended normally; every other terminal
# status means it never connected.
_TERMINAL_CALL_STATUSES = frozenset({"completed", "busy", "failed", "no-answer", "canceled"})


class TwilioDialer:
    """Places calls through Twilio's REST Calls API.

    Endpoint, method, and every parameter name below were verified live
    against Twilio's own docs (fetched 2026-09), not written from memory:

    - Create-a-call: ``POST /2010-04-01/Accounts/{AccountSid}/Calls.json``,
      https://www.twilio.com/docs/voice/api/call-resource -- ``To``,
      ``From``, and one of ``Url``/``Twiml``/``ApplicationSid`` are
      required; auth is HTTP Basic with the Account SID as username and the
      Auth Token as password.
    - Answering-machine detection: ``MachineDetection`` (``"Enable"`` or
      ``"DetectMessageEnd"``), ``AsyncAmd`` (``"true"``/``"false"``),
      ``AsyncAmdStatusCallback``, ``MachineDetectionTimeout`` (milliseconds)
      -- https://www.twilio.com/docs/voice/answering-machine-detection.
    - Status callbacks: ``StatusCallback``, ``StatusCallbackMethod``,
      ``StatusCallbackEvent`` -- same call-resource page. A ``completed``
      event's payload additionally carries ``CallDuration`` (seconds).

    **``Url`` is the caller's Media Streams webhook, not this dialer's
    concern.** ``docs/TELEPHONY.md`` documents that webhook returning TwiML
    with ``<Connect><Stream>`` pointed at this stack's websocket transport
    (``transports/twilio.py``); this class only needs its address to pass
    as ``Url`` when creating the call.

    **Why ``place_call`` does not resolve its own outcome future.** Whether
    a call connected is something *Twilio* learns and reports back on
    ``StatusCallback`` -- an HTTP request Twilio makes to your server,
    potentially seconds or minutes after ``place_call`` returns. This class
    has no webhook server of its own (that is your application's HTTP
    framework, not this package's job), so the future it returns is
    resolved out-of-band by :meth:`report_status`, which your
    ``StatusCallback``/``AsyncAmdStatusCallback`` route should call with
    the POSTed form fields. A dialer that instead resolved "connected"
    the moment Twilio accepted the create-call request would be lying:
    accepted-for-dialing and answered are not the same event, and
    :class:`~tring.outbound.runner.CampaignRunner`'s retry logic depends on
    the distinction.

    Args:
        from_number: the caller-id number to dial from.
        stream_webhook_url: TwiML webhook URL passed as ``Url``.
        account_sid_env: name of the environment variable holding the
            Account SID. Not a secret, but kept as an env-var name (never
            passed as a literal) for the same reason ``api_key_env`` is:
            see ``providers/cloud/__init__.py``'s module docstring.
        auth_token_env: name of the environment variable holding the Auth
            Token.
        machine_detection: ``None`` to skip AMD, or ``"Enable"`` /
            ``"DetectMessageEnd"`` to request it. Passed straight through
            as ``MachineDetection``.
        machine_detection_timeout_ms: forwarded as ``MachineDetectionTimeout``
            when set; Twilio's own default (30000ms) applies otherwise.
        status_callback_url: forwarded as ``StatusCallback`` when set. Your
            server's route for it should call :meth:`report_status`.
        async_amd_status_callback_url: forwarded as ``AsyncAmdStatusCallback``
            when set, and implies ``AsyncAmd=true`` -- AMD results then
            arrive on this separate webhook instead of (synchronously)
            blocking the call's own ``Url`` request.
        base_url: overridable for testing; production code should leave it.
        transport: an ``httpx.AsyncBaseTransport`` test seam (``None`` in
            production) -- see ``tests/test_outbound.py`` and the identical
            pattern in ``providers/cloud/llm_extra.py``.
    """

    def __init__(
        self,
        from_number: str,
        stream_webhook_url: str,
        account_sid_env: str = "TWILIO_ACCOUNT_SID",
        auth_token_env: str = "TWILIO_AUTH_TOKEN",
        machine_detection: str | None = None,
        machine_detection_timeout_ms: int | None = None,
        status_callback_url: str | None = None,
        async_amd_status_callback_url: str | None = None,
        base_url: str = "https://api.twilio.com",
        *,
        transport: Any = None,
    ) -> None:
        self._from_number = from_number
        self._stream_webhook_url = stream_webhook_url
        self._account_sid_env = account_sid_env
        self._auth_token_env = auth_token_env
        self._machine_detection = machine_detection
        self._machine_detection_timeout_ms = machine_detection_timeout_ms
        self._status_callback_url = status_callback_url
        self._async_amd_status_callback_url = async_amd_status_callback_url
        self._base_url = base_url.rstrip("/")
        self._transport = transport

        # call_id -> the future place_call handed out for it, plus whatever
        # AnsweredBy has arrived so far (which can precede or follow the
        # terminal CallStatus, depending on sync vs. async AMD -- see
        # report_status).
        self._pending: dict[str, asyncio.Future[DialOutcome]] = {}
        self._answered_by: dict[str, str] = {}

    async def place_call(self, callee: Callee) -> CallHandle:
        import httpx

        account_sid = _read_env(self._account_sid_env, "TwilioDialer account SID")
        auth_token = _read_env(self._auth_token_env, "TwilioDialer auth token")

        data: dict[str, str] = {
            "To": callee.phone,
            "From": self._from_number,
            "Url": self._stream_webhook_url,
        }
        if self._machine_detection is not None:
            data["MachineDetection"] = self._machine_detection
        if self._machine_detection_timeout_ms is not None:
            data["MachineDetectionTimeout"] = str(self._machine_detection_timeout_ms)
        if self._status_callback_url is not None:
            data["StatusCallback"] = self._status_callback_url
        if self._async_amd_status_callback_url is not None:
            data["AsyncAmdStatusCallback"] = self._async_amd_status_callback_url
            data["AsyncAmd"] = "true"

        url = f"{self._base_url}/2010-04-01/Accounts/{account_sid}/Calls.json"
        async with httpx.AsyncClient(transport=self._transport, timeout=30.0) as client:
            response = await client.post(url, data=data, auth=(account_sid, auth_token))
            response.raise_for_status()
        call_id = str(response.json()["sid"])

        future: asyncio.Future[DialOutcome] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future
        return CallHandle(call_id=call_id, callee=callee, outcome=future)

    def report_status(self, payload: Mapping[str, str]) -> None:
        """Feed one Twilio webhook POST (form fields) into this dialer.

        Call this from both your ``StatusCallback`` and (if configured)
        ``AsyncAmdStatusCallback`` routes -- both payloads share the
        ``CallSid`` key and this method tells them apart by which of
        ``CallStatus`` / ``AnsweredBy`` is present. It is a no-op for a
        ``call_id`` this dialer never placed (or already resolved), so a
        duplicate or late-arriving webhook -- Twilio retries these -- never
        raises.
        """
        call_id = payload.get("CallSid")
        if not call_id:
            return

        answered_by = payload.get("AnsweredBy")
        if answered_by:
            self._answered_by[call_id] = answered_by

        call_status = payload.get("CallStatus")
        if call_status is None or call_status not in _TERMINAL_CALL_STATUSES:
            return  # an AMD-only callback, or a non-terminal status update

        future = self._pending.pop(call_id, None)
        if future is None or future.done():
            return

        duration_raw = payload.get("CallDuration")
        future.set_result(
            DialOutcome(
                connected=call_status == "completed",
                answered_by=self._answered_by.pop(call_id, None),
                call_duration_seconds=float(duration_raw) if duration_raw else None,
                error=None if call_status == "completed" else f"call ended: {call_status}",
            )
        )


def _read_env(env_var: str, label: str) -> str:
    import os

    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"{label} requires environment variable {env_var!r} to be set. "
            "Dialer options carry the *name* of the variable, never the "
            "credential itself."
        )
    return value


# ---------------------------------------------------------------------------
# TELEPHONY_RATES: Twilio's own list-priced per-minute voice rate.
#
# https://www.twilio.com/en-us/voice/pricing/us -- outbound calls to US
# numbers, pay-as-you-go, fetched 2026-09. Priced per second (rather than
# per minute) so it composes with CostMeter.record's units*price_per_unit,
# matching every other duration-billed rate in cost/rates.py (e.g.
# Deepgram's audio_seconds row).
# ---------------------------------------------------------------------------
TELEPHONY_RATES: list[Rate] = [
    Rate(
        component=CostComponent.TELEPHONY,
        provider="twilio",
        model=None,
        unit_name="call_seconds",
        price_per_unit=0.0140 / 60,  # $0.0140/min outbound, US, PAYG list price
        currency="USD",
        as_of="2026-09",
    ),
]

__all__ = [
    "TELEPHONY_RATES",
    "CallHandle",
    "DialOutcome",
    "Dialer",
    "FakeDialer",
    "TwilioDialer",
]
