"""``transfer_call``: hand a live Twilio call to a human or another number.

A voice agent that has hit the edge of what it can do (an angry caller, a
request outside its scope, an explicit "let me talk to a person") needs a
way to get out of the caller's way instead of stalling. This module ships
that as a first-class tool contract: :data:`TRANSFER_TOOL` is the
``ToolDef`` an ``AgentSpec`` includes, and :func:`make_transfer_handler`
binds it to one live call via Twilio's REST API.

**The REST endpoint, verified live (fetched 2026-09-19), not written from
memory** -- https://www.twilio.com/docs/voice/api/call-resource#update-a-call-resource:

- ``POST https://api.twilio.com/2010-04-01/Accounts/{AccountSid}/Calls/{CallSid}.json``
  redirects (or ends) a call already in progress.
- Exactly one of ``Url`` (a TwiML URL Twilio re-fetches) or ``Twiml``
  (inline TwiML) is required to redirect a live call to new instructions;
  this is the mechanism used here, not the ``Status`` parameter (that field
  only *ends* a call -- ``canceled``/``completed`` -- it does not redirect
  one, so it is out of scope for a transfer).
- Auth is HTTP Basic with the Account SID as username and the Auth Token as
  password; the body is ``application/x-www-form-urlencoded``, Twilio's
  standard REST API request shape everywhere, not anything specific to this
  endpoint.

Redirecting the live call away from its ``<Connect><Stream>`` (see
``tring.transports.twilio``) is what ends this transport's involvement:
Twilio tears down the Media Streams WebSocket once the call's TwiML moves
on, which is what actually stops audio flowing to and from this stack. This
handler cannot reach into the runtime that owns that WebSocket (a
``ToolHandler`` only ever receives its own JSON arguments -- see
``primitives/choreography.py``'s ``ToolHandler`` alias), so it does the one
thing it *can* do to make "the session has ended" true in the event
stream every consumer already watches: emit ``SessionTransferred`` and then
``SessionEnded(reason="transferred")`` itself, the same event ``V4_PLAN.md``
documents cascade emitting on a transfer. The transport's own
``stop(reason="twilio_stream_ended")`` follows moments later once Twilio's
``stop`` message actually arrives; both events landing on the same session
is expected, not a bug -- see ``CallSession.emit``, which never rejects
multiple ``SessionEnded``s on purpose (closing is ``session.close()``,
a separate, transport-owned step).
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any
from xml.sax.saxutils import escape as _xml_escape

import httpx

from tring.agent import ToolDef
from tring.events import SessionEnded, SessionTransferred
from tring.session import CallSession

_TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"

ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


TRANSFER_TOOL = ToolDef(
    name="transfer_call",
    description=(
        "Transfer the caller to a human agent or another phone number. Use "
        "this when the caller explicitly asks for a human, or the "
        "conversation needs help this agent cannot give."
    ),
    parameters={
        "type": "object",
        "properties": {
            "destination": {
                "type": "string",
                "description": (
                    "Where to send the call: an E.164 phone number (e.g. "
                    "'+14155551234') to dial, or a fully-qualified http(s):// "
                    "TwiML URL Twilio should fetch instead of dialing a "
                    "number directly."
                ),
            },
        },
        "required": ["destination"],
    },
    handler="transfer_call",
    # A REST round trip to Twilio plus a live carrier redirect is not a
    # sub-second operation; the choreographed waiting_message this schema
    # already requires (see augment_tool_schema) is what keeps the caller
    # from hearing dead air while it completes.
    timeout_seconds=8.0,
)


class TransferError(RuntimeError):
    """Twilio rejected the call-update request (bad CallSid, expired call, ...).

    The message embeds Twilio's own status code and body verbatim -- this
    is the string ``primitives/choreography.py``'s ``execute`` feeds back to
    the model as the tool's error, so it should say what actually went
    wrong, not just "transfer failed".
    """


def _read_env(env_var: str) -> str:
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"transfer_call requires environment variable {env_var!r} to be "
            "set (Account SID / Auth Token; see make_transfer_handler's "
            "account_sid_env / auth_token_env parameters)."
        )
    return value


def _dial_twiml(destination: str) -> str:
    """Inline TwiML that dials ``destination`` as the new call leg.

    Escaped with the stdlib's own XML escaper (not string formatting) since
    ``destination`` is caller-influenced (the model chose it from the
    conversation) and lands inside an XML text node Twilio parses.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Dial>{_xml_escape(destination)}</Dial></Response>"
    )


def make_transfer_handler(
    session: CallSession,
    call_sid: str,
    *,
    account_sid_env: str = "TWILIO_ACCOUNT_SID",
    auth_token_env: str = "TWILIO_AUTH_TOKEN",
    client: httpx.AsyncClient | None = None,
) -> ToolHandler:
    """Bind :data:`TRANSFER_TOOL` to one live Twilio call.

    ``call_sid`` identifies *which* call to redirect -- it is the Twilio
    ``CallSid`` this transport's ``start`` message carried (see
    ``transports/twilio.py``'s ``_handle_start``), known once per call and
    closed over here rather than asked of the model, which has no way to
    know it. ``session`` is likewise closed over: it is what lets this
    handler emit ``SessionTransferred``/``SessionEnded`` (see the module
    docstring for why the handler, not a runtime, does that here).

    ``client`` is the test seam: pass an ``httpx.AsyncClient`` built on
    ``httpx.MockTransport`` (the same house pattern
    ``providers/cloud/llm_extra.py`` uses) to drive this against a scripted
    response with no network. Left ``None`` in production, where a fresh
    client is opened and closed per call.
    """

    async def handler(arguments: dict[str, Any]) -> dict[str, Any]:
        destination = str(arguments.get("destination", "")).strip()
        if not destination:
            raise ValueError("transfer_call requires a non-empty 'destination'")

        account_sid = _read_env(account_sid_env)
        auth_token = _read_env(auth_token_env)

        is_twiml_url = destination.startswith("http://") or destination.startswith("https://")
        form: dict[str, str] = (
            {"Url": destination} if is_twiml_url else {"Twiml": _dial_twiml(destination)}
        )

        owns_client = client is None
        http_client = client if client is not None else httpx.AsyncClient(timeout=10.0)
        try:
            response = await http_client.post(
                f"{_TWILIO_API_BASE}/Accounts/{account_sid}/Calls/{call_sid}.json",
                data=form,
                auth=(account_sid, auth_token),
            )
        finally:
            if owns_client:
                await http_client.aclose()

        if response.status_code >= 400:
            raise TransferError(
                f"Twilio call-update for {call_sid} failed: "
                f"{response.status_code} {response.text}"
            )

        session.emit(
            SessionTransferred(
                session_id=session.session_id, at=session.elapsed, target=destination
            )
        )
        session.emit(
            SessionEnded(
                session_id=session.session_id,
                at=session.elapsed,
                reason="transferred",
            )
        )
        return {"transferred_to": destination, "call_sid": call_sid}

    return handler


__all__ = ["TRANSFER_TOOL", "TransferError", "make_transfer_handler"]
