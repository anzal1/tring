"""Tests for tool-call choreography.

No network, no vendor SDKs: tool handlers and ``speak`` are plain async
fakes, and the session clock is injected so latency assertions are exact
rather than racy against wall-clock time.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from alaap.agent import ToolDef
from alaap.events import ToolCallCompleted, ToolCallStarted
from alaap.primitives.choreography import (
    ChoreographedCall,
    ChoreographyError,
    ToolOutcome,
    augment_tool_schema,
    execute,
    parse_choreographed_call,
)
from alaap.session import CallSession


class FakeClock:
    """A settable monotonic clock for deterministic latency assertions."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = start

    def __call__(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        self._t += seconds


def make_session(clock: FakeClock | None = None) -> CallSession:
    from alaap.agent import AgentSpec

    agent = AgentSpec(name="test-agent", persona="You are a test agent.")
    return CallSession(agent=agent, session_id="sess-1", clock=clock or FakeClock())


def make_tool(*, choreographed: bool = True, timeout_seconds: float = 5.0) -> ToolDef:
    return ToolDef(
        name="lookup_order",
        description="Look up an order by id.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
        handler="fake",
        timeout_seconds=timeout_seconds,
        choreographed=choreographed,
    )


# --- augment_tool_schema ----------------------------------------------------


def test_augment_marks_all_three_fields_required() -> None:
    tool = make_tool()
    schema = augment_tool_schema(tool)

    assert schema["type"] == "object"
    for field in ("waiting_message", "spoken_mode", "post_tool_response"):
        assert field in schema["properties"]
        assert field in schema["required"]

    # original tool parameters survive alongside the injected fields
    assert "order_id" in schema["properties"]
    assert "order_id" in schema["required"]


def test_augment_spoken_mode_and_post_tool_response_are_enums() -> None:
    schema = augment_tool_schema(make_tool())
    assert schema["properties"]["spoken_mode"]["enum"] == [
        "full_answer_given",
        "answer_pending",
    ]
    assert schema["properties"]["post_tool_response"]["enum"] == ["respond", "silent"]


def test_augment_choreographed_false_is_pure_passthrough() -> None:
    tool = make_tool(choreographed=False)
    schema = augment_tool_schema(tool)
    assert schema == tool.parameters
    assert "waiting_message" not in schema.get("properties", {})


# --- parse_choreographed_call: happy path -----------------------------------


def test_parse_separates_arguments_from_choreography_fields() -> None:
    tool = make_tool()
    call = {
        "call_id": "call-1",
        "order_id": "ORD-42",
        "waiting_message": "Let me check that order.",
        "spoken_mode": "answer_pending",
        "post_tool_response": "respond",
    }
    parsed = parse_choreographed_call(tool, call)

    assert isinstance(parsed, ChoreographedCall)
    assert parsed.tool_name == "lookup_order"
    assert parsed.call_id == "call-1"
    assert parsed.arguments == {"order_id": "ORD-42"}
    assert parsed.waiting_message == "Let me check that order."
    assert parsed.spoken_mode == "answer_pending"
    assert parsed.post_tool_response == "respond"


def test_parse_choreographed_false_passthrough() -> None:
    tool = make_tool(choreographed=False)
    call = {"call_id": "call-2", "order_id": "ORD-7"}
    parsed = parse_choreographed_call(tool, call)

    assert parsed.arguments == {"order_id": "ORD-7"}
    # synthetic defaults that make execute() behave as a plain passthrough
    assert parsed.spoken_mode == "full_answer_given"
    assert parsed.post_tool_response == "silent"


# --- parse_choreographed_call: validation failures --------------------------


def test_parse_missing_fields_raises_choreography_error() -> None:
    tool = make_tool()
    call = {"call_id": "call-3", "order_id": "ORD-1"}
    with pytest.raises(ChoreographyError) as exc_info:
        parse_choreographed_call(tool, call)

    message = str(exc_info.value)
    assert "waiting_message" in message
    assert "spoken_mode" in message
    assert "post_tool_response" in message
    assert "lookup_order" in message


def test_parse_invalid_spoken_mode_raises() -> None:
    tool = make_tool()
    call = {
        "call_id": "call-4",
        "order_id": "ORD-1",
        "waiting_message": "One moment.",
        "spoken_mode": "maybe_later",
        "post_tool_response": "respond",
    }
    with pytest.raises(ChoreographyError) as exc_info:
        parse_choreographed_call(tool, call)
    assert "spoken_mode" in str(exc_info.value)
    assert "maybe_later" in str(exc_info.value)


def test_parse_invalid_post_tool_response_raises() -> None:
    tool = make_tool()
    call = {
        "call_id": "call-5",
        "order_id": "ORD-1",
        "waiting_message": "One moment.",
        "spoken_mode": "answer_pending",
        "post_tool_response": "shout",
    }
    with pytest.raises(ChoreographyError) as exc_info:
        parse_choreographed_call(tool, call)
    assert "post_tool_response" in str(exc_info.value)
    assert "shout" in str(exc_info.value)


def test_parse_empty_waiting_message_raises() -> None:
    tool = make_tool()
    call = {
        "call_id": "call-6",
        "order_id": "ORD-1",
        "waiting_message": "   ",
        "spoken_mode": "full_answer_given",
        "post_tool_response": "silent",
    }
    with pytest.raises(ChoreographyError) as exc_info:
        parse_choreographed_call(tool, call)
    assert "waiting_message" in str(exc_info.value)


# --- execute: happy path event sequence -------------------------------------


async def test_execute_happy_path_emits_started_then_completed() -> None:
    clock = FakeClock(start=100.0)
    session = make_session(clock)
    tool = make_tool()

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        clock.advance(0.25)
        return {"status": "shipped", "order_id": args["order_id"]}

    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    call = parse_choreographed_call(
        tool,
        {
            "call_id": "call-7",
            "order_id": "ORD-99",
            "waiting_message": "Let me check that order.",
            "spoken_mode": "answer_pending",
            "post_tool_response": "respond",
        },
    )

    outcome = await execute(tool, call, handler, session, speak)

    assert isinstance(outcome, ToolOutcome)
    assert outcome.ok is True
    assert outcome.should_respond is True
    assert outcome.result == {"status": "shipped", "order_id": "ORD-99"}

    history = session.history
    assert len(history) == 2

    started, completed = history
    assert isinstance(started, ToolCallStarted)
    assert started.tool_name == "lookup_order"
    assert started.call_id == "call-7"
    assert started.arguments == {"order_id": "ORD-99"}
    assert started.waiting_message == "Let me check that order."
    assert started.spoken_mode == "answer_pending"
    assert started.at == 0.0  # session elapsed at call start

    assert isinstance(completed, ToolCallCompleted)
    assert completed.tool_name == "lookup_order"
    assert completed.call_id == "call-7"
    assert completed.ok is True
    assert completed.latency_seconds is not None
    assert completed.latency_seconds >= 0.0

    # answer_pending: the waiting message must be spoken before the result
    assert spoken == ["Let me check that order."]


async def test_execute_full_answer_given_does_not_speak_waiting_message() -> None:
    session = make_session()
    tool = make_tool()

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    call = parse_choreographed_call(
        tool,
        {
            "call_id": "call-8",
            "order_id": "ORD-1",
            "waiting_message": "Confirming that now.",
            "spoken_mode": "full_answer_given",
            "post_tool_response": "silent",
        },
    )

    outcome = await execute(tool, call, handler, session, speak)

    assert spoken == []  # the model already spoke the answer; no waiting message
    assert outcome.ok is True
    assert outcome.should_respond is False  # post_tool_response == "silent"


async def test_execute_post_tool_response_silent_on_success() -> None:
    session = make_session()
    tool = make_tool()

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    async def speak(text: str) -> None:
        pass

    call = parse_choreographed_call(
        tool,
        {
            "call_id": "call-9",
            "order_id": "ORD-1",
            "waiting_message": "One sec.",
            "spoken_mode": "answer_pending",
            "post_tool_response": "silent",
        },
    )

    outcome = await execute(tool, call, handler, session, speak)
    assert outcome.ok is True
    assert outcome.should_respond is False


# --- execute: failure paths --------------------------------------------------


async def test_execute_timeout_forces_response_and_marks_not_ok() -> None:
    session = make_session()
    tool = make_tool(timeout_seconds=0.05)

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(10)
        return {"never": "reached"}

    async def speak(text: str) -> None:
        pass

    call = parse_choreographed_call(
        tool,
        {
            "call_id": "call-10",
            "order_id": "ORD-1",
            "waiting_message": "Checking now.",
            "spoken_mode": "answer_pending",
            "post_tool_response": "silent",  # model's plan; must be overridden
        },
    )

    outcome = await execute(tool, call, handler, session, speak)

    assert outcome.ok is False
    assert outcome.should_respond is True  # timeout always forces a response
    assert outcome.error is not None
    assert "timed out" in outcome.error

    history = session.history
    assert len(history) == 2
    completed = history[1]
    assert isinstance(completed, ToolCallCompleted)
    assert completed.ok is False
    assert completed.result_summary == outcome.error


async def test_execute_handler_exception_forces_response_and_marks_not_ok() -> None:
    session = make_session()
    tool = make_tool()

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("downstream service unavailable")

    async def speak(text: str) -> None:
        pass

    call = parse_choreographed_call(
        tool,
        {
            "call_id": "call-11",
            "order_id": "ORD-1",
            "waiting_message": "Checking now.",
            "spoken_mode": "full_answer_given",
            "post_tool_response": "silent",
        },
    )

    outcome = await execute(tool, call, handler, session, speak)

    assert outcome.ok is False
    assert outcome.should_respond is True
    assert outcome.error is not None
    assert "RuntimeError" in outcome.error
    assert "downstream service unavailable" in outcome.error


# --- execute: choreographed=False passthrough --------------------------------


async def test_execute_choreographed_false_never_speaks_and_respond_matches_default() -> (
    None
):
    session = make_session()
    tool = make_tool(choreographed=False)

    async def handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    spoken: list[str] = []

    async def speak(text: str) -> None:
        spoken.append(text)

    call = parse_choreographed_call(tool, {"call_id": "call-12", "order_id": "ORD-1"})
    outcome = await execute(tool, call, handler, session, speak)

    assert spoken == []  # spoken_mode defaults to full_answer_given
    assert outcome.ok is True
    assert outcome.should_respond is False  # post_tool_response defaults to silent
