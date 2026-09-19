"""Tests for the flow compiler.

Two things are being protected here, and they pull in different directions.

The first is the *contract*: a flow graph is data that a canvas stores and a
server compiles, so it has to round-trip losslessly and it has to reject a
broken graph by naming the node that broke it. Those tests read like a table,
because that is what they are.

The second is the *persona*. A compiled persona is the prompt a real model
reads, so the golden test below asserts on the lines a change would actually
break: the step numbering, the transitions, the loop label, the slot list. It
deliberately does not assert the whole blob byte for byte. Rewording one
sentence of standing instruction should not fail a test; renumbering the steps
or dropping a transition absolutely should.

No fixtures, no network, no providers: ``compile_flow`` is a pure function of
its two arguments.
"""

from __future__ import annotations

from typing import Any

import pytest

from tring.agent import AgentSpec, ProviderSelection, RuntimeConfig, ToolDef
from tring.flow import (
    FLOW_SECTION_HEADER,
    FlowError,
    FlowGraph,
    compile_flow,
    flow_from_spec,
    render_flow_instructions,
    validate_flow,
)

# ---------------------------------------------------------------------------
# The reference flow: six nodes, one tool, one loop back
# ---------------------------------------------------------------------------

BOOKING_FLOW: dict[str, Any] = {
    "name": "northwind-booking",
    "greeting": "Northwind Dental, how can I help?",
    "nodes": [
        {
            "id": "welcome",
            "kind": "say",
            "label": "Open the call",
            "text": "Thanks for calling Northwind Dental.",
        },
        {
            "id": "details",
            "kind": "ask",
            "prompt": "Who am I speaking with, and what number are you on?",
            "slots": [
                {"name": "full_name", "description": "the caller's full name"},
                {"name": "phone", "description": "a ten digit mobile number"},
                {
                    "name": "notes",
                    "description": "anything the dentist should know",
                    "required": False,
                },
            ],
        },
        {
            "id": "lookup",
            "kind": "tool",
            "tool": {
                "name": "check_availability",
                "description": "Look up open appointment slots for a date.",
                "parameters": {
                    "type": "object",
                    "properties": {"date": {"type": "string"}},
                },
            },
        },
        {
            "id": "triage",
            "kind": "branch",
            "label": "Did we find a slot?",
            "condition": "what the lookup returned",
        },
        {"id": "confirm", "kind": "say", "text": "You are booked. See you then."},
        {"id": "done", "kind": "end", "text": "Thanks for calling. Goodbye."},
    ],
    "edges": [
        {"from": "welcome", "to": "details"},
        {"from": "details", "to": "lookup"},
        {"from": "lookup", "to": "triage"},
        {"from": "triage", "to": "confirm", "when": "a slot the caller accepted is free"},
        {"from": "triage", "to": "details", "when": "the caller wants to try other details"},
        {"from": "confirm", "to": "done"},
    ],
}


def booking_graph(**overrides: Any) -> FlowGraph:
    return FlowGraph.model_validate({**BOOKING_FLOW, **overrides})


# ---------------------------------------------------------------------------
# The golden persona
# ---------------------------------------------------------------------------


def test_the_booking_flow_compiles_to_a_stable_numbered_persona() -> None:
    lines = render_flow_instructions(booking_graph()).splitlines()

    assert lines[0] == FLOW_SECTION_HEADER
    # Topological numbering: no step is introduced before the step that leads
    # to it, which is the whole reason "go to step 5" means anything.
    assert "1. SAY (welcome): Open the call" in lines
    assert '   Say: "Thanks for calling Northwind Dental."' in lines
    assert "   Then go to step 2 (details)." in lines

    assert "2. ASK (details)" in lines
    assert '   Ask: "Who am I speaking with, and what number are you on?"' in lines
    assert "   Collect all of these before moving on:" in lines
    assert "     - full_name: the caller's full name" in lines
    assert "     - notes: anything the dentist should know (optional)" in lines

    assert "3. TOOL (lookup)" in lines
    assert (
        '   Call the tool "check_availability": Look up open appointment slots for a date.'
        in lines
    )

    assert "4. BRANCH (triage): Did we find a slot?" in lines
    assert "   Decide, based on what the lookup returned:" in lines
    assert "     - If a slot the caller accepted is free, go to step 5 (confirm)." in lines

    assert "5. SAY (confirm)" in lines
    assert "6. END (done)" in lines
    assert '   Close with: "Thanks for calling. Goodbye."' in lines
    assert "   Then stop. Do not open a new topic after this step." in lines


def test_a_cycle_is_labeled_as_a_loop_instead_of_breaking_the_numbering() -> None:
    lines = render_flow_instructions(booking_graph()).splitlines()

    # The edge back to step 2 closes a cycle. It is still rendered -- a flow
    # that silently dropped it would compile to an agent that cannot retry --
    # and it is rendered as a loop so the reader knows the numbering went
    # backwards on purpose.
    assert (
        "     - If the caller wants to try other details, loop back to step 2 (details)."
        in lines
    )
    assert sum(line.startswith("2. ") for line in lines) == 1


def test_a_handoff_renders_a_transfer_instruction_and_stops() -> None:
    graph = FlowGraph.model_validate(
        {
            "nodes": [
                {"id": "open", "kind": "say", "text": "Front desk."},
                {
                    "id": "escalate",
                    "kind": "handoff",
                    "target": "the practice manager",
                    "text": "Let me put you through to the manager.",
                },
            ],
            "edges": [{"from": "open", "to": "escalate"}],
        }
    )

    lines = render_flow_instructions(graph).splitlines()

    assert "2. HANDOFF (escalate)" in lines
    assert '   Say: "Let me put you through to the manager."' in lines
    assert "   Hand the call over to the practice manager." in lines
    assert "   Then stop talking and wait for the transfer." in lines


# ---------------------------------------------------------------------------
# The compiled spec
# ---------------------------------------------------------------------------


def test_compiling_without_a_base_produces_a_runnable_spec() -> None:
    spec = compile_flow(booking_graph())

    assert spec.name == "northwind-booking"
    assert spec.greeting == "Northwind Dental, how can I help?"
    assert spec.persona.startswith(FLOW_SECTION_HEADER)
    assert [tool.name for tool in spec.tools] == ["check_availability"]
    assert spec.tools[0].description == "Look up open appointment slots for a date."


def test_the_graph_round_trips_through_metadata() -> None:
    graph = booking_graph()

    spec = compile_flow(graph)

    # Lossless both ways: the canvas reopens the flow that built the spec
    # rather than reverse-engineering one out of the rendered prose.
    assert flow_from_spec(spec) == graph
    assert spec.metadata["flow"]["edges"][0] == {"from": "welcome", "to": "details"}
    assert AgentSpec.model_validate(spec.model_dump(mode="json")) == spec


def test_a_hand_written_spec_has_no_flow_to_read_back() -> None:
    assert flow_from_spec(AgentSpec(name="plain", persona="Be brief.")) is None


def test_the_base_spec_keeps_everything_a_flow_has_no_opinion_about() -> None:
    base = AgentSpec(
        name="front-desk",
        persona="Speak warmly and keep every answer under two sentences.",
        greeting="Front desk.",
        runtime=RuntimeConfig(
            routing={"default": ProviderSelection(stt="faster_whisper", llm="ollama")}
        ),
        tools=[ToolDef(name="open_hours", description="Opening hours for a day.")],
        metadata={"owner": "reception"},
    )

    spec = compile_flow(booking_graph(name=None, greeting=None), base)

    assert spec.name == "front-desk"
    assert spec.greeting == "Front desk."
    assert spec.runtime.routing["default"].llm == "ollama"
    assert spec.metadata["owner"] == "reception"
    # Persona: the base's voice first, the generated steps after it.
    assert spec.persona.startswith("Speak warmly")
    assert FLOW_SECTION_HEADER in spec.persona
    # Tools: the base's first, then the flow's, in step order.
    assert [tool.name for tool in spec.tools] == ["open_hours", "check_availability"]
    # And the base itself is untouched.
    assert base.persona == "Speak warmly and keep every answer under two sentences."
    assert [tool.name for tool in base.tools] == ["open_hours"]


def test_recompiling_replaces_the_generated_section_instead_of_stacking_copies() -> None:
    once = compile_flow(booking_graph(), AgentSpec(name="x", persona="Be brief."))

    twice = compile_flow(booking_graph(), once)
    three_times = compile_flow(booking_graph(), twice)

    assert twice.persona == once.persona
    assert three_times.persona == once.persona
    assert once.persona.count(FLOW_SECTION_HEADER) == 1
    assert once.persona.startswith("Be brief.")


def test_one_tool_called_from_two_nodes_is_bound_once() -> None:
    tool = {"name": "lookup", "description": "Find the caller."}
    graph = FlowGraph.model_validate(
        {
            "nodes": [
                {"id": "first", "kind": "tool", "tool": tool},
                {"id": "second", "kind": "tool", "tool": tool},
                {"id": "done", "kind": "end"},
            ],
            "edges": [
                {"from": "first", "to": "second"},
                {"from": "second", "to": "done"},
            ],
        }
    )

    spec = compile_flow(graph)

    assert [t.name for t in spec.tools] == ["lookup"]


def test_a_flow_tool_that_contradicts_the_base_spec_is_refused() -> None:
    base = AgentSpec(
        name="x",
        persona="p",
        tools=[ToolDef(name="check_availability", description="Something else entirely.")],
    )

    with pytest.raises(FlowError) as raised:
        compile_flow(booking_graph(), base)

    assert raised.value.node_id == "lookup"
    assert "check_availability" in str(raised.value)


# ---------------------------------------------------------------------------
# Rejections: every one names the node and says what to do about it
# ---------------------------------------------------------------------------


def _reject(graph: dict[str, Any]) -> FlowError:
    with pytest.raises(FlowError) as raised:
        validate_flow(FlowGraph.model_validate(graph))
    return raised.value


def test_an_edge_pointing_at_nothing_is_reported_as_an_edge() -> None:
    error = _reject(
        {
            "nodes": [
                {"id": "a", "kind": "say", "text": "hi"},
                {"id": "b", "kind": "end"},
            ],
            "edges": [{"from": "a", "to": "typo"}],
        }
    )

    # Reported as the typo it is, not as "node b unreachable" three checks
    # later: the first error a user sees has to be the root cause.
    assert "'typo'" in str(error)
    assert "no node has that id" in str(error)


@pytest.mark.parametrize(
    ("node", "edges", "node_id", "fragment"),
    [
        pytest.param(
            {"id": "a", "kind": "say"},
            [{"from": "a", "to": "z"}],
            "a",
            "needs 'text'",
            id="say-without-text",
        ),
        pytest.param(
            {"id": "a", "kind": "ask", "prompt": "your name?"},
            [{"from": "a", "to": "z"}],
            "a",
            "at least one slot",
            id="ask-without-slots",
        ),
        pytest.param(
            {
                "id": "a",
                "kind": "say",
                "text": "hi",
                "slots": [{"name": "n", "description": "d"}],
            },
            [{"from": "a", "to": "z"}],
            "a",
            "means nothing on a 'say' node",
            id="field-of-another-kind",
        ),
        pytest.param(
            {"id": "a", "kind": "branch"},
            [{"from": "a", "to": "z", "when": "always"}],
            "a",
            "at least two transitions",
            id="branch-with-one-way-out",
        ),
        pytest.param(
            {"id": "a", "kind": "say", "text": "hi"},
            [{"from": "a", "to": "z"}, {"from": "a", "to": "z"}],
            "a",
            "exactly one transition",
            id="two-transitions-from-a-say",
        ),
        pytest.param(
            {"id": "a", "kind": "say", "text": "hi"},
            [],
            "a",
            "exactly one transition",
            id="dead-end",
        ),
        pytest.param(
            {"id": "a", "kind": "say", "text": "hi"},
            [{"from": "a", "to": "z", "when": "maybe"}],
            "a",
            "'when' only applies",
            id="condition-outside-a-branch",
        ),
        pytest.param(
            {"id": "a", "kind": "tool", "tool": {"name": "t", "description": "d"}},
            [{"from": "a", "to": "z"}, {"from": "z", "to": "a"}],
            "z",
            "no transitions",
            id="transition-out-of-an-end",
        ),
    ],
)
def test_a_node_the_runtime_could_not_run_is_refused_by_name(
    node: dict[str, Any], edges: list[dict[str, Any]], node_id: str, fragment: str
) -> None:
    error = _reject({"nodes": [node, {"id": "z", "kind": "end"}], "edges": edges})

    assert error.node_id == node_id
    assert fragment in str(error)
    assert str(error).startswith(f"node {node_id!r}: ")


def test_two_branch_transitions_without_a_condition_are_refused() -> None:
    error = _reject(
        {
            "nodes": [
                {"id": "a", "kind": "branch"},
                {"id": "y", "kind": "end"},
                {"id": "z", "kind": "end"},
            ],
            "edges": [{"from": "a", "to": "y"}, {"from": "a", "to": "z"}],
        }
    )

    assert error.node_id == "a"
    assert "at most one default transition" in str(error)


def test_a_branch_may_have_exactly_one_default_and_it_renders_last() -> None:
    graph = FlowGraph.model_validate(
        {
            "nodes": [
                {"id": "a", "kind": "branch"},
                {"id": "y", "kind": "end", "text": "Booked."},
                {"id": "z", "kind": "end", "text": "Sorry."},
            ],
            "edges": [
                {"from": "a", "to": "y"},
                {"from": "a", "to": "z", "when": "the caller is in a hurry"},
            ],
        }
    )

    lines = [line for line in render_flow_instructions(graph).splitlines() if "- " in line]

    # Declared first, rendered last: a fallback in the middle of the list reads
    # as one more condition and gets taken as soon as the model reaches it.
    assert lines == [
        "     - If the caller is in a hurry, go to step 3 (z).",
        "     - Otherwise, go to step 2 (y).",
    ]


def test_two_nodes_disagreeing_about_one_tool_are_refused() -> None:
    error = _reject(
        {
            "nodes": [
                {"id": "a", "kind": "tool", "tool": {"name": "t", "description": "one"}},
                {"id": "b", "kind": "tool", "tool": {"name": "t", "description": "another"}},
                {"id": "z", "kind": "end"},
            ],
            "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "z"}],
        }
    )

    assert error.node_id == "b"
    assert "defined differently" in str(error)


def test_a_graph_with_nowhere_to_start_asks_for_an_entry_node() -> None:
    error = _reject(
        {
            "nodes": [
                {"id": "a", "kind": "say", "text": "hi"},
                {"id": "b", "kind": "say", "text": "again"},
            ],
            "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}],
        }
    )

    assert error.node_id is None
    assert str(error).startswith("flow: ")
    assert "set 'entry'" in str(error)


def test_two_possible_starts_are_ambiguous_until_entry_says_which() -> None:
    graph: dict[str, Any] = {
        "nodes": [
            {"id": "a", "kind": "say", "text": "hi"},
            {"id": "b", "kind": "say", "text": "also hi"},
            {"id": "z", "kind": "end"},
        ],
        "edges": [{"from": "a", "to": "z"}, {"from": "b", "to": "z"}],
    }

    assert "2 nodes have nothing pointing at them" in str(_reject(graph))

    # Naming the entry resolves the ambiguity and makes the other start
    # unreachable, which is the next honest complaint rather than a silent
    # half-compiled flow.
    error = _reject({**graph, "entry": "a"})
    assert error.node_id == "b"
    assert "nothing leads here from the entry node 'a'" in str(error)


def test_a_flow_with_no_way_to_finish_is_refused() -> None:
    error = _reject(
        {
            "nodes": [
                {"id": "a", "kind": "say", "text": "hi"},
                {"id": "b", "kind": "say", "text": "bye"},
            ],
            "edges": [{"from": "a", "to": "b"}, {"from": "b", "to": "a"}],
            "entry": "a",
        }
    )

    assert "'end' or 'handoff'" in str(error)


@pytest.mark.parametrize(
    ("graph", "fragment"),
    [
        pytest.param({"nodes": []}, "at least one node", id="empty"),
        pytest.param(
            {
                "nodes": [
                    {"id": "a", "kind": "end"},
                    {"id": "a", "kind": "end"},
                ]
            },
            "ids must be unique",
            id="duplicate-ids",
        ),
        pytest.param(
            {"nodes": [{"id": "a", "kind": "end"}], "entry": "ghost"},
            "not in this flow",
            id="entry-that-does-not-exist",
        ),
    ],
)
def test_a_graph_that_is_not_a_graph_is_refused(graph: dict[str, Any], fragment: str) -> None:
    assert fragment in str(_reject(graph))


def test_an_unknown_node_kind_never_reaches_the_compiler() -> None:
    # pydantic owns the shape; the compiler owns the meaning. Keeping the
    # boundary there is why every FlowError below can assume a valid enum.
    with pytest.raises(ValueError, match="kind"):
        FlowGraph.model_validate({"nodes": [{"id": "a", "kind": "sing"}]})
