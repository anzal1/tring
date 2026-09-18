"""Tool-call choreography.

A voice model that calls a tool has just created a latency gap: the tool
takes real seconds, and dead air during those seconds reads as a hang, a
dropped call, or a rude agent. Text-chat tool-calling has no equivalent
problem because there is no "listening" cost to silence.

The fix is structural, not prompted: we do not *ask* the model to plan the
silence around a tool call, we make it *impossible for the model to emit a
tool call without doing so*. Every choreographed tool's JSON-Schema grows
three required sibling fields alongside its normal arguments:

- ``waiting_message`` — what to speak while the tool runs. Always required,
  even when it will never be spoken (see ``spoken_mode``), because the model
  must have thought about it, and because ``execute`` needs a self-consistent
  fallback line if a downstream ``ok=False`` demands the model explain itself
  before it gets to plan a new sentence.
- ``spoken_mode`` — did the model's speak-turn *already* answer the caller's
  question before it decided to call the tool (``full_answer_given``, e.g.
  "Let me check that for you" is not an answer but "Sure, that's $42" is), or
  is the answer still pending on the tool result (``answer_pending``)? This
  tells ``execute`` whether it's safe to speak ``waiting_message`` immediately
  or whether doing so would talk over/duplicate what was already said.
- ``post_tool_response`` — after the tool result comes back, must the model
  speak again (``respond``), or would that just repeat itself (``silent``)?

This is deliberately redundant with what a careful prompt could ask for,
because prompts are advisory and schemas are enforced: a provider that
returns malformed enum values fails tool-call parsing loudly (see
``ChoreographyError``) instead of the agent silently going quiet or chatty
at the wrong moment.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field

from alaap.agent import ToolDef
from alaap.events import ToolCallCompleted, ToolCallStarted
from alaap.session import CallSession

#: JSON-Schema property name for each choreography field. Centralized so the
#: augmenter, the parser, and error messages can never drift apart.
WAITING_MESSAGE = "waiting_message"
SPOKEN_MODE = "spoken_mode"
POST_TOOL_RESPONSE = "post_tool_response"

SpokenMode = Literal["full_answer_given", "answer_pending"]
PostToolResponse = Literal["respond", "silent"]

_SPOKEN_MODE_VALUES: tuple[str, ...] = ("full_answer_given", "answer_pending")
_POST_TOOL_RESPONSE_VALUES: tuple[str, ...] = ("respond", "silent")

_CHOREOGRAPHY_PROPERTIES: dict[str, Any] = {
    WAITING_MESSAGE: {
        "type": "string",
        "description": (
            "What to say out loud while this tool runs. Required even if it "
            "will not be spoken this turn (see spoken_mode): plan it before "
            "you decide whether to use it. Keep it short and natural, e.g. "
            "'Let me check that.' Never leave it empty."
        ),
    },
    SPOKEN_MODE: {
        "type": "string",
        "enum": list(_SPOKEN_MODE_VALUES),
        "description": (
            "'full_answer_given' if your spoken turn already answered the "
            "caller before calling this tool (the tool result only confirms "
            "or logs it). 'answer_pending' if the caller is still waiting on "
            "the tool result to be answered — in that case waiting_message "
            "is spoken immediately so the caller is never met with silence."
        ),
    },
    POST_TOOL_RESPONSE: {
        "type": "string",
        "enum": list(_POST_TOOL_RESPONSE_VALUES),
        "description": (
            "'respond' if you must speak again once the tool result comes "
            "back (you have new information to deliver). 'silent' if "
            "speaking again would just repeat what the caller already heard."
        ),
    },
}

_CHOREOGRAPHY_REQUIRED: tuple[str, ...] = (
    WAITING_MESSAGE,
    SPOKEN_MODE,
    POST_TOOL_RESPONSE,
)


class ChoreographyError(ValueError):
    """A tool call failed choreography validation.

    Raised with a message written to be fed straight back to the model as
    the tool-call error content (most LLM tool-calling loops resubmit the
    error as the tool's result and let the model retry) — so the wording
    names the exact field and the exact allowed values, not just "invalid".
    """


class ChoreographedCall(BaseModel):
    """A parsed tool call with choreography fields separated from arguments.

    ``arguments`` holds only what the tool itself declared in
    ``ToolDef.parameters`` — never the three choreography keys. Keeping them
    split means a tool handler is written against its own schema and never
    needs to know choreography exists, even though every call it receives
    was validated against the augmented schema.
    """

    tool_name: str
    call_id: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    waiting_message: str
    spoken_mode: SpokenMode
    post_tool_response: PostToolResponse


class ToolOutcome(BaseModel):
    """What ``execute`` produced, and whether the model must speak again.

    ``should_respond`` is the answer ``execute`` computes by combining
    ``post_tool_response`` with the actual outcome: a failed or timed-out
    call always forces a response, regardless of what the model asked for,
    because the model cannot know in advance that the tool would fail and
    "silent" was only ever a safe choice conditional on success.
    """

    ok: bool
    should_respond: bool
    result: Any = None
    error: str | None = None


ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]
Speak = Callable[[str], Awaitable[None]]


def augment_tool_schema(tool: ToolDef) -> dict[str, Any]:
    """Return the JSON-Schema the model must satisfy to call ``tool``.

    When ``tool.choreographed`` is False, this is a pure pass-through of
    ``tool.parameters`` — an explicit opt-out for tools where dead air is a
    non-issue (e.g. a tool that only mutates local state the model already
    described in its speech, with no latency worth narrating). Every other
    tool gets the three choreography fields folded in as required siblings
    of its own parameters, so the model cannot construct a valid call
    without them.
    """
    base = tool.parameters or {"type": "object", "properties": {}}

    if not tool.choreographed:
        return dict(base)

    schema = dict(base)
    schema.setdefault("type", "object")
    properties = dict(schema.get("properties", {}))
    properties.update(_CHOREOGRAPHY_PROPERTIES)
    schema["properties"] = properties

    existing_required = list(schema.get("required", []))
    required = list(existing_required)
    for field in _CHOREOGRAPHY_REQUIRED:
        if field not in required:
            required.append(field)
    schema["required"] = required

    return schema


def parse_choreographed_call(tool: ToolDef, call: dict[str, Any]) -> ChoreographedCall:
    """Split a raw tool-call payload into arguments and choreography fields.

    ``call`` is the model's tool-call arguments dict (already JSON-decoded)
    plus a ``call_id`` the caller assigns to correlate start/complete events
    — most provider SDKs hand back a call id alongside the arguments, so we
    accept it as an optional key here rather than a separate parameter.

    Raises ``ChoreographyError`` when a required field is missing or an enum
    value is invalid. The error message is written to be replayed to the
    model verbatim as the tool result, so it names the field and the exact
    accepted values rather than a generic "bad call".
    """
    call_id = str(call.get("call_id", ""))

    if not tool.choreographed:
        arguments = {k: v for k, v in call.items() if k != "call_id"}
        return ChoreographedCall(
            tool_name=tool.name,
            call_id=call_id,
            arguments=arguments,
            waiting_message="",
            spoken_mode="full_answer_given",
            post_tool_response="silent",
        )

    missing = [
        field
        for field in _CHOREOGRAPHY_REQUIRED
        if field not in call or call[field] in (None, "")
    ]
    if missing:
        raise ChoreographyError(
            f"tool call to '{tool.name}' is missing required choreography "
            f"field(s) {missing}: every choreographed tool call must include "
            f"waiting_message (a non-empty string), spoken_mode "
            f"({list(_SPOKEN_MODE_VALUES)}), and post_tool_response "
            f"({list(_POST_TOOL_RESPONSE_VALUES)})."
        )

    spoken_mode = call[SPOKEN_MODE]
    if spoken_mode not in _SPOKEN_MODE_VALUES:
        raise ChoreographyError(
            f"tool call to '{tool.name}' has invalid spoken_mode {spoken_mode!r}; "
            f"must be one of {list(_SPOKEN_MODE_VALUES)}."
        )

    post_tool_response = call[POST_TOOL_RESPONSE]
    if post_tool_response not in _POST_TOOL_RESPONSE_VALUES:
        raise ChoreographyError(
            f"tool call to '{tool.name}' has invalid post_tool_response "
            f"{post_tool_response!r}; must be one of "
            f"{list(_POST_TOOL_RESPONSE_VALUES)}."
        )

    waiting_message = call[WAITING_MESSAGE]
    if not isinstance(waiting_message, str) or not waiting_message.strip():
        raise ChoreographyError(
            f"tool call to '{tool.name}' has an empty waiting_message; it must "
            "be a non-empty string, even when spoken_mode is 'full_answer_given' "
            "and it will not be spoken this turn."
        )

    arguments = {
        k: v
        for k, v in call.items()
        if k not in (WAITING_MESSAGE, SPOKEN_MODE, POST_TOOL_RESPONSE, "call_id")
    }

    return ChoreographedCall(
        tool_name=tool.name,
        call_id=call_id,
        arguments=arguments,
        waiting_message=waiting_message,
        spoken_mode=spoken_mode,
        post_tool_response=post_tool_response,
    )


async def execute(
    tool_def: ToolDef,
    call: ChoreographedCall,
    handler: ToolHandler,
    session: CallSession,
    speak: Speak,
) -> ToolOutcome:
    """Run a choreographed tool call, keeping the caller from ever hearing dead air.

    Sequence, matched to what a caller actually experiences:

    1. Emit ``ToolCallStarted`` immediately (analytics/cost see the call the
       instant it begins, not after the round trip).
    2. If ``spoken_mode == "answer_pending"``, speak ``waiting_message``
       *before* awaiting the handler — the caller hears something within a
       turn, not after the tool's own latency. If the model already gave the
       full answer (``full_answer_given``), speaking the waiting message too
       would be redundant chatter, so it is skipped.
    3. Run ``handler(call.arguments)`` under ``tool_def.timeout_seconds``.
    4. Emit ``ToolCallCompleted`` with latency, and decide ``should_respond``:
       - success: honor ``call.post_tool_response`` as the model requested.
       - timeout or handler exception: always ``should_respond=True``. A
         model that planned "silent" did so believing the tool would
         succeed; it cannot have accounted for failure, so silence here
         would strand the caller with no explanation for a tool that never
         answered. The model gets a chance to speak and explain instead.
    """
    started_at = session.elapsed
    session.emit(
        ToolCallStarted(
            session_id=session.session_id,
            at=started_at,
            tool_name=tool_def.name,
            call_id=call.call_id,
            arguments=call.arguments,
            waiting_message=call.waiting_message,
            spoken_mode=call.spoken_mode,
        )
    )

    if call.spoken_mode == "answer_pending":
        await speak(call.waiting_message)

    start_perf = time.perf_counter()
    ok: bool
    result: Any = None
    error: str | None = None
    try:
        async with asyncio.timeout(tool_def.timeout_seconds):
            result = await handler(call.arguments)
        ok = True
    except TimeoutError:
        ok = False
        error = (
            f"tool '{tool_def.name}' timed out after {tool_def.timeout_seconds}s"
        )
    except Exception as exc:  # tool handlers are untrusted plugins; must not crash the loop
        ok = False
        error = f"tool '{tool_def.name}' raised {type(exc).__name__}: {exc}"

    latency = time.perf_counter() - start_perf

    session.emit(
        ToolCallCompleted(
            session_id=session.session_id,
            at=session.elapsed,
            tool_name=tool_def.name,
            call_id=call.call_id,
            ok=ok,
            result_summary=None if ok else error,
            latency_seconds=latency,
        )
    )

    if not ok:
        return ToolOutcome(ok=False, should_respond=True, result=None, error=error)

    should_respond = call.post_tool_response == "respond"
    return ToolOutcome(ok=True, should_respond=should_respond, result=result, error=None)
