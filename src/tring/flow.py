"""Flow graphs: a node canvas compiled into an :class:`~tring.agent.AgentSpec`.

A flow is the shape most voice agents actually have: greet, collect a few
details, look something up, branch on the answer, hand off or hang up. Writing
that as free prose in a persona works until the fourth revision, at which point
nobody can tell which sentence is the step and which is the tone.

So Studio draws it as a graph, and this module turns the graph into a spec:

.. code-block:: text

    FlowGraph            compile_flow()              AgentSpec
    ---------            --------------              ---------
    nodes + edges  --->  topological walk      --->  persona  (numbered steps)
                         per-kind rendering         tools    (from tool nodes)
                         tool collection            metadata["flow"] (the graph)

Three decisions are worth knowing before reading the code.

**The compiler is server side and UI free.** Nothing here imports React, a
canvas library or a layout engine, and no node carries an ``x``/``y``. The
graph is *data*: the same JSON compiles the same spec from the studio, from a
test, or from a script that never opens a browser. The UI owns pixels; this
module owns meaning.

**The walk is topological, and loops are labeled rather than forbidden.** Steps
are numbered in an order where a step is never introduced before the step that
leads to it, which is what makes "go to step 5" readable to a model. Real
conversations loop ("that time does not work, pick another"), so edges that
close a cycle are detected in a depth-first pass, removed from the ordering,
and rendered as an explicit *loop back to step N* instead of quietly breaking
the numbering.

**The graph survives the compile.** ``spec.metadata["flow"]`` holds the graph
verbatim, so opening a compiled spec in the canvas again shows the flow that
built it rather than an attempt to reverse-engineer one out of prose. The
rendered section is re-generated on every compile and is marked as generated
text (:data:`FLOW_SECTION_HEADER`), so recompiling is idempotent instead of
stacking one copy of the flow on top of the last.

Validation is loud and specific. Every rejection names the node it is about and
what is wrong with it, because a flow is edited on a canvas where "invalid
graph" points at nothing at all.
"""

from __future__ import annotations

import enum
import heapq
import textwrap
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from tring.agent import AgentSpec, ToolDef

#: First line of the generated persona section. Two jobs, both load bearing:
#: it tells a human reading ``agent.yaml`` that the text below is machine
#: written, and it is the exact marker :func:`compile_flow` splits a base
#: persona on so a second compile replaces the section instead of appending a
#: second copy of it.
FLOW_SECTION_HEADER = "CONVERSATION FLOW (compiled from the flow graph; do not edit by hand)"

#: The standing rules rendered under the header, once, for the whole flow.
_FLOW_PREAMBLE = """\
Work through the numbered steps below. They are the conversation you are here
to have, in the order it normally happens. Each step ends by naming where to go
next: take that transition as soon as the step's work is done, never skip a
step the flow did not send you to, and never invent a step that is not listed.
Text in quotation marks is scripted: say it as written."""


class NodeKind(enum.StrEnum):
    """What one node in the flow does.

    Six kinds, and the list is closed on purpose. Every conversational move a
    scripted agent makes is one of: say a thing, collect a thing, decide, call
    something, hand the call over, or finish. A seventh kind almost always
    turns out to be one of these with different prose.
    """

    SAY = "say"
    ASK = "ask"
    BRANCH = "branch"
    TOOL = "tool"
    HANDOFF = "handoff"
    END = "end"


#: Kinds after which the conversation is over, so they carry no transitions.
_TERMINAL_KINDS = frozenset({NodeKind.HANDOFF, NodeKind.END})

#: Which optional fields each kind may carry. Anything else set on a node is a
#: rejection rather than a silent no-op: a ``say`` node with ``slots`` on it is
#: a node someone edited into the wrong kind, and finding that out at compile
#: time is much cheaper than finding it out mid-call.
_FIELDS_BY_KIND: Mapping[NodeKind, frozenset[str]] = {
    NodeKind.SAY: frozenset({"text"}),
    NodeKind.ASK: frozenset({"prompt", "slots"}),
    NodeKind.BRANCH: frozenset({"condition"}),
    NodeKind.TOOL: frozenset({"tool"}),
    NodeKind.HANDOFF: frozenset({"target", "text"}),
    NodeKind.END: frozenset({"text"}),
}

#: Fields that must be present for the node to mean anything at all.
_REQUIRED_BY_KIND: Mapping[NodeKind, tuple[str, ...]] = {
    NodeKind.SAY: ("text",),
    NodeKind.ASK: ("prompt",),
    NodeKind.BRANCH: (),
    NodeKind.TOOL: ("tool",),
    NodeKind.HANDOFF: ("target",),
    NodeKind.END: (),
}

_OPTIONAL_FIELDS: tuple[str, ...] = ("text", "prompt", "slots", "condition", "tool", "target")

#: Where the compiler's own prose wraps. Narrow enough to stay readable in the
#: studio's editor pane next to the canvas.
_WRAP_WIDTH = 78


class FlowError(ValueError):
    """A flow the compiler refuses, with the node responsible attached.

    ``node_id`` is ``None`` for whole-graph problems (no entry node, an edge
    pointing nowhere). The rendered message always leads with the subject, so
    a UI can show ``str(error)`` unmodified and a canvas can highlight
    ``error.node_id``.
    """

    def __init__(self, reason: str, node_id: str | None = None) -> None:
        self.node_id = node_id
        self.reason = reason
        subject = f"node {node_id!r}" if node_id else "flow"
        super().__init__(f"{subject}: {reason}")


class FlowSlot(BaseModel):
    """One piece of information an ``ask`` node must collect.

    ``description`` is what the model is told to listen for, so it is written
    for the model ("a ten digit mobile number"), not for a form label.
    """

    name: str
    description: str
    required: bool = True


class FlowNode(BaseModel):
    """One step on the canvas.

    The per-kind fields are all optional on the model and required by
    :func:`validate_flow`, rather than split across six pydantic classes. That
    keeps the JSON the canvas stores flat and diffable, and it keeps every
    rejection in one place, phrased the same way, with the node id in it.
    """

    id: str
    kind: NodeKind
    #: Human label shown on the canvas. Rendered into the persona when set,
    #: because "BRANCH (triage): what the caller wants" reads better to a model
    #: than a bare node id, and it costs one line to carry through.
    label: str | None = None

    #: ``say``: the line to speak. ``end``: the closing line. ``handoff``: what
    #: the caller hears before the transfer.
    text: str | None = None
    #: ``ask``: the question to put to the caller.
    prompt: str | None = None
    #: ``ask``: everything that must be filled before the step is done.
    slots: list[FlowSlot] = Field(default_factory=list)
    #: ``branch``: what the decision is about ("what the caller wants").
    condition: str | None = None
    #: ``tool``: the tool definition, carried on the node so a flow is a
    #: complete, self-contained description of the agent it compiles to.
    tool: ToolDef | None = None
    #: ``handoff``: who the call goes to.
    target: str | None = None


class FlowEdge(BaseModel):
    """A transition. ``when`` is the condition, and only branches have one.

    Serialized as ``{"from": ..., "to": ...}`` because that is the wire shape
    in ``docs/STUDIO_PROTOCOL.md``; ``from`` is a Python keyword, so the fields
    are named ``source``/``target`` and aliased. Both spellings validate.
    """

    model_config = ConfigDict(populate_by_name=True)

    source: str = Field(alias="from")
    target: str = Field(alias="to")
    when: str | None = None


class FlowGraph(BaseModel):
    """The whole canvas: nodes, transitions, and where the conversation starts.

    ``entry`` may be left unset when exactly one node has nothing pointing at
    it, which is the ordinary case. It becomes required as soon as that is
    ambiguous, rather than the compiler guessing.
    """

    name: str | None = None
    greeting: str | None = None
    entry: str | None = None
    nodes: list[FlowNode] = Field(default_factory=list)
    edges: list[FlowEdge] = Field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        """The JSON form stored in ``spec.metadata["flow"]``.

        ``by_alias`` so edges keep their ``from``/``to`` spelling, and
        ``exclude_none`` so a six-node flow does not land in ``agent.yaml`` as
        sixty lines of ``null``. Both are round-trip safe: every excluded field
        has ``None`` as its default.
        """
        return self.model_dump(mode="json", by_alias=True, exclude_none=True)


@dataclass(frozen=True)
class _Walk:
    """The validated reading order of a graph.

    Args:
        order: node ids, topologically sorted with loop edges removed.
        step: node id to 1-based step number, the numbering the persona uses.
        loop_edges: indices into ``graph.edges`` of the edges that close a
            cycle. They are real transitions, just rendered as "loop back".
    """

    order: tuple[str, ...]
    step: Mapping[str, int]
    loop_edges: frozenset[int]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_flow(graph: FlowGraph) -> None:
    """Raise :class:`FlowError` if ``graph`` would not compile. Cheap enough
    to run on every canvas edit."""
    _plan(graph)


def _plan(graph: FlowGraph) -> _Walk:
    """Validate the graph and return the order its steps are numbered in.

    The checks run cheapest-and-most-fundamental first, so the error a user
    sees is the root cause and not a downstream symptom of it: a typo in an
    edge target is reported as a typo, never as "node unreachable".
    """
    nodes = _index_nodes(graph)
    _check_edges(graph, nodes)
    out_edges = _out_edges(graph)
    for node in graph.nodes:
        _check_node(node, [graph.edges[i] for i in out_edges[node.id]])
    _check_tool_conflicts(graph)
    entry = _entry_id(graph, nodes)
    discovery, loop_edges = _explore(graph, out_edges, entry)
    _check_reachable(graph, discovery, entry)
    _check_has_an_ending(graph)
    order = _topological(graph, out_edges, loop_edges, discovery)
    return _Walk(
        order=order,
        step={node_id: position + 1 for position, node_id in enumerate(order)},
        loop_edges=loop_edges,
    )


def _index_nodes(graph: FlowGraph) -> dict[str, FlowNode]:
    if not graph.nodes:
        raise FlowError("a flow needs at least one node")
    nodes: dict[str, FlowNode] = {}
    for node in graph.nodes:
        if not node.id.strip():
            raise FlowError("every node needs a non-empty id")
        if node.id in nodes:
            raise FlowError("two nodes share this id; ids must be unique", node.id)
        nodes[node.id] = node
    return nodes


def _check_edges(graph: FlowGraph, nodes: Mapping[str, FlowNode]) -> None:
    for position, edge in enumerate(graph.edges):
        for role, node_id in (("from", edge.source), ("to", edge.target)):
            if node_id not in nodes:
                raise FlowError(
                    f"edge {position} ({edge.source!r} -> {edge.target!r}) points at "
                    f"{node_id!r} as its {role!r}, and no node has that id"
                )


def _out_edges(graph: FlowGraph) -> dict[str, list[int]]:
    """Edge indices leaving each node, in the order the graph declares them.

    Declaration order is the tie-breaker everywhere downstream (which branch
    reads first, which path the walk explores first), so the compile is stable:
    the same graph always produces the same persona, byte for byte.
    """
    out: dict[str, list[int]] = defaultdict(list)
    for position, edge in enumerate(graph.edges):
        out[edge.source].append(position)
    return out


def _check_node(node: FlowNode, outgoing: Sequence[FlowEdge]) -> None:
    allowed = _FIELDS_BY_KIND[node.kind]
    for field in _OPTIONAL_FIELDS:
        if field in allowed or not getattr(node, field):
            continue
        used_by = sorted(k.value for k, fields in _FIELDS_BY_KIND.items() if field in fields)
        raise FlowError(
            f"{field!r} means nothing on a {node.kind.value!r} node "
            f"(it belongs to: {', '.join(used_by)})",
            node.id,
        )
    for field in _REQUIRED_BY_KIND[node.kind]:
        if not getattr(node, field):
            raise FlowError(f"a {node.kind.value!r} node needs {field!r}", node.id)
    if node.kind is NodeKind.ASK and not node.slots:
        raise FlowError(
            "an 'ask' node needs at least one slot to collect; a question with "
            "nothing to fill is a 'say' node",
            node.id,
        )
    _check_transitions(node, outgoing)


def _check_transitions(node: FlowNode, outgoing: Sequence[FlowEdge]) -> None:
    """One rule per kind, and the rules are the reason the render is simple."""
    if node.kind in _TERMINAL_KINDS:
        if outgoing:
            raise FlowError(
                f"a {node.kind.value!r} node is where this path stops, so it can have "
                f"no transitions; remove the one to {outgoing[0].target!r}",
                node.id,
            )
        return

    if node.kind is NodeKind.BRANCH:
        if len(outgoing) < 2:
            raise FlowError(
                f"a 'branch' node needs at least two transitions to choose between; "
                f"found {len(outgoing)}",
                node.id,
            )
        defaults = [edge for edge in outgoing if edge.when is None]
        if len(defaults) > 1:
            named = ", ".join(repr(edge.target) for edge in defaults)
            raise FlowError(
                f"transitions to {named} have no 'when' condition; a branch can have "
                "at most one default transition",
                node.id,
            )
        return

    if len(outgoing) != 1:
        raise FlowError(
            f"a {node.kind.value!r} node has exactly one transition; found "
            f"{len(outgoing)} (use a 'branch' node to choose between paths, or an "
            "'end' node to finish)",
            node.id,
        )
    if outgoing[0].when is not None:
        raise FlowError(
            "'when' only applies to transitions leaving a 'branch' node; this one "
            f"leaves a {node.kind.value!r} node",
            node.id,
        )


def _check_tool_conflicts(graph: FlowGraph) -> None:
    """Two nodes may call the same tool; they may not disagree about it.

    Calling one tool from several points in a flow is normal (look the caller
    up before booking, and again before cancelling). Two *different* schemas
    under one name is not: the runtime binds tools by name, so one of the two
    definitions would silently win.
    """
    seen: dict[str, ToolDef] = {}
    for node in graph.nodes:
        if node.tool is None:
            continue
        existing = seen.setdefault(node.tool.name, node.tool)
        if existing != node.tool:
            raise FlowError(
                f"tool {node.tool.name!r} is defined differently by another node; "
                "make the two definitions identical or rename one",
                node.id,
            )


def _entry_id(graph: FlowGraph, nodes: Mapping[str, FlowNode]) -> str:
    if graph.entry is not None:
        if graph.entry not in nodes:
            raise FlowError(f"the entry node {graph.entry!r} is not in this flow")
        return graph.entry
    targeted = {edge.target for edge in graph.edges}
    roots = [node.id for node in graph.nodes if node.id not in targeted]
    if len(roots) == 1:
        return roots[0]
    if not roots:
        raise FlowError(
            "every node has something pointing at it, so the flow has no obvious "
            "start; set 'entry' to the node the conversation begins on"
        )
    found = ", ".join(repr(node_id) for node_id in roots)
    raise FlowError(
        f"{len(roots)} nodes have nothing pointing at them ({found}); set 'entry' to "
        "the one the conversation begins on"
    )


def _explore(
    graph: FlowGraph, out_edges: Mapping[str, list[int]], entry: str
) -> tuple[dict[str, int], frozenset[int]]:
    """Depth-first from the entry: discovery order, and which edges are loops.

    An edge is a loop edge exactly when it points at a node that is still open
    on the DFS stack, which is the textbook back-edge test. Removing those
    edges leaves a DAG, which is what makes a topological numbering possible at
    all, and keeping them as data is what lets the render say "loop back to
    step 2" instead of dropping the transition on the floor.

    Iterative rather than recursive so a long flow cannot hit the recursion
    limit, and so the traversal order is visible in the code.
    """
    open_now: set[str] = set()
    closed: set[str] = set()
    discovery: dict[str, int] = {entry: 0}
    loops: set[int] = set()
    open_now.add(entry)
    stack: list[tuple[str, int]] = [(entry, 0)]

    while stack:
        node_id, position = stack.pop()
        edges = out_edges[node_id]
        if position >= len(edges):
            open_now.discard(node_id)
            closed.add(node_id)
            continue
        stack.append((node_id, position + 1))
        edge_index = edges[position]
        target = graph.edges[edge_index].target
        if target in open_now:
            loops.add(edge_index)
        elif target not in closed:
            open_now.add(target)
            discovery[target] = len(discovery)
            stack.append((target, 0))
    return discovery, frozenset(loops)


def _check_reachable(graph: FlowGraph, discovery: Mapping[str, int], entry: str) -> None:
    for node in graph.nodes:
        if node.id not in discovery:
            raise FlowError(
                f"nothing leads here from the entry node {entry!r}, so this step can "
                "never run",
                node.id,
            )


def _check_has_an_ending(graph: FlowGraph) -> None:
    if not any(node.kind in _TERMINAL_KINDS for node in graph.nodes):
        raise FlowError(
            "a flow needs at least one 'end' or 'handoff' node, or the conversation "
            "has no way to finish"
        )


def _topological(
    graph: FlowGraph,
    out_edges: Mapping[str, list[int]],
    loop_edges: frozenset[int],
    discovery: Mapping[str, int],
) -> tuple[str, ...]:
    """Kahn's algorithm over the graph minus its loop edges.

    Ties are broken by discovery order, which is what keeps the main path of a
    conversation contiguous in the numbering: a side branch is emitted after
    the path that found it, not interleaved with it.
    """
    incoming: dict[str, int] = {node.id: 0 for node in graph.nodes}
    for position, edge in enumerate(graph.edges):
        if position not in loop_edges:
            incoming[edge.target] += 1

    ready = [(discovery[node_id], node_id) for node_id, count in incoming.items() if not count]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        _, node_id = heapq.heappop(ready)
        order.append(node_id)
        for position in out_edges[node_id]:
            if position in loop_edges:
                continue
            target = graph.edges[position].target
            incoming[target] -= 1
            if incoming[target] == 0:
                heapq.heappush(ready, (discovery[target], target))

    if len(order) != len(graph.nodes):  # pragma: no cover - removing back edges is total
        raise FlowError("the flow contains a cycle the compiler could not unwind")
    return tuple(order)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_flow_instructions(graph: FlowGraph) -> str:
    """The persona section a graph compiles to, on its own.

    Public because it is the half worth reading in a diff: the UI can preview
    it, and a test can assert on the lines that matter without going through a
    whole :class:`~tring.agent.AgentSpec`.
    """
    walk = _plan(graph)
    out_edges = _out_edges(graph)
    by_id = {node.id: node for node in graph.nodes}
    lines = [FLOW_SECTION_HEADER, "", _FLOW_PREAMBLE, ""]
    for node_id in walk.order:
        lines.extend(_render_node(graph, by_id[node_id], walk, out_edges[node_id]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _wrapped(text: str) -> list[str]:
    """Body prose, wrapped and indented under its step.

    Quoted lines are never wrapped: they are what the agent says, and a line
    break inside them would invite the model to read the layout. Only the
    compiler's own instructions go through here, so the persona stays legible
    in the studio's editor instead of running off the pane.
    """
    return textwrap.wrap(text, width=_WRAP_WIDTH, initial_indent="   ", subsequent_indent="   ")


def _render_node(
    graph: FlowGraph, node: FlowNode, walk: _Walk, outgoing: Sequence[int]
) -> list[str]:
    heading = f"{walk.step[node.id]}. {node.kind.value.upper()} ({node.id})"
    if node.label:
        heading = f"{heading}: {node.label}"
    lines = [heading]

    if node.kind is NodeKind.SAY:
        lines.append(f'   Say: "{node.text}"')
    elif node.kind is NodeKind.ASK:
        lines.append(f'   Ask: "{node.prompt}"')
        lines.append("   Collect all of these before moving on:")
        for slot in node.slots:
            suffix = "" if slot.required else " (optional)"
            lines.append(f"     - {slot.name}: {slot.description}{suffix}")
        lines.extend(
            _wrapped(
                "If an answer is missing or unclear, ask again for that one detail "
                "only, and read anything you are unsure of back to the caller."
            )
        )
    elif node.kind is NodeKind.BRANCH:
        subject = node.condition or "what the caller has said"
        lines.append(f"   Decide, based on {subject}:")
        lines.extend(_render_branch_options(graph, walk, outgoing))
        lines.extend(
            _wrapped(
                "If none of these is clearly true yet, ask one short question to "
                "find out, then decide."
            )
        )
    elif node.kind is NodeKind.TOOL:
        assert node.tool is not None  # guaranteed by _check_node
        lines.append(f'   Call the tool "{node.tool.name}": {node.tool.description}')
        lines.extend(
            _wrapped(
                "Fill its arguments from what the caller has already told you. If a "
                "required argument is still unknown, ask for it before calling."
            )
        )
    elif node.kind is NodeKind.HANDOFF:
        if node.text:
            lines.append(f'   Say: "{node.text}"')
        lines.append(f"   Hand the call over to {node.target}.")
        lines.append("   Then stop talking and wait for the transfer.")
    else:  # NodeKind.END
        if node.text:
            lines.append(f'   Close with: "{node.text}"')
        lines.append("   Then stop. Do not open a new topic after this step.")

    if node.kind not in _TERMINAL_KINDS and node.kind is not NodeKind.BRANCH:
        edge_index = outgoing[0]
        lines.append(f"   Then {_transition(graph, walk, edge_index)}.")
    return lines


def _render_branch_options(
    graph: FlowGraph, walk: _Walk, outgoing: Sequence[int]
) -> list[str]:
    """Conditions first, the default last, whatever order the canvas stored.

    A default rendered in the middle reads as one more condition, and the model
    takes it as soon as it gets there. Last, it reads as the fallback it is.
    """
    conditional = [i for i in outgoing if graph.edges[i].when is not None]
    default = [i for i in outgoing if graph.edges[i].when is None]
    lines = [
        f"     - If {graph.edges[i].when}, {_transition(graph, walk, i)}."
        for i in conditional
    ]
    lines.extend(
        f"     - Otherwise, {_transition(graph, walk, i)}." for i in default
    )
    return lines


def _transition(graph: FlowGraph, walk: _Walk, edge_index: int) -> str:
    edge = graph.edges[edge_index]
    verb = "loop back to" if edge_index in walk.loop_edges else "go to"
    return f"{verb} step {walk.step[edge.target]} ({edge.target})"


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def compile_flow(graph: FlowGraph, base: AgentSpec | None = None) -> AgentSpec:
    """Compile ``graph`` into a runnable :class:`~tring.agent.AgentSpec`.

    ``base`` supplies everything a flow has no opinion about: provider routing,
    language policy, limits, and any persona text that describes *how* the
    agent speaks rather than *what* it does. The flow supplies the numbered
    steps, the tools its ``tool`` nodes carry, and the graph itself under
    ``metadata["flow"]``.

    Recompiling is idempotent. The generated section is always the tail of the
    persona and always starts with :data:`FLOW_SECTION_HEADER`, so a base spec
    that was itself compiled from a flow has its old section cut off rather
    than a second one appended. That is what makes "edit the canvas, compile,
    save, edit again" stable instead of quadratic.

    Args:
        graph: the flow to compile.
        base: the spec to build on, usually the one currently saved in Studio.

    Returns:
        A new spec. ``base`` is never mutated.

    Raises:
        FlowError: the graph is not a conversation. The message names the node.
    """
    walk = _plan(graph)
    section = render_flow_instructions(graph)

    spec = base.model_copy(deep=True) if base is not None else None
    prefix = _strip_generated_section(spec.persona) if spec is not None else ""
    persona = f"{prefix}\n\n{section}" if prefix else section

    metadata: dict[str, Any] = dict(spec.metadata) if spec is not None else {}
    metadata["flow"] = graph.to_payload()

    name = graph.name or (spec.name if spec is not None else None) or "flow-agent"
    greeting = graph.greeting or (spec.greeting if spec is not None else None)

    if spec is None:
        return AgentSpec(
            name=name,
            persona=persona,
            greeting=greeting,
            tools=_merge_tools(graph, walk, []),
            metadata=metadata,
        )
    spec.name = name
    spec.persona = persona
    spec.greeting = greeting
    spec.tools = _merge_tools(graph, walk, spec.tools)
    spec.metadata = metadata
    return spec


def flow_from_spec(spec: AgentSpec) -> FlowGraph | None:
    """The graph a spec was compiled from, or ``None`` if it was hand written.

    The round trip that makes the canvas non-destructive: ``compile_flow``
    stores the graph, this reads it back, and nothing in between has to parse
    the rendered prose.
    """
    stored = spec.metadata.get("flow")
    if not isinstance(stored, dict):
        return None
    return FlowGraph.model_validate(stored)


def _strip_generated_section(persona: str) -> str:
    return persona.split(FLOW_SECTION_HEADER, 1)[0].rstrip()


def _merge_tools(graph: FlowGraph, walk: _Walk, base_tools: Sequence[ToolDef]) -> list[ToolDef]:
    """Base tools first, then each tool node's, in step order.

    Step order rather than node declaration order, so the tool list reads in
    the same sequence as the conversation that calls it. A flow tool that
    collides with a differently-defined base tool is an error for the same
    reason two nodes disagreeing is: the runtime binds by name, so one
    definition would silently lose.
    """
    merged: dict[str, ToolDef] = {tool.name: tool for tool in base_tools}
    by_id = {node.id: node for node in graph.nodes}
    for node_id in walk.order:
        tool = by_id[node_id].tool
        if tool is None:
            continue
        existing = merged.get(tool.name)
        if existing is not None and existing != tool:
            raise FlowError(
                f"tool {tool.name!r} is already defined differently by the base spec; "
                "make the two definitions identical or rename one",
                node_id,
            )
        merged[tool.name] = tool
    return list(merged.values())


__all__ = [
    "FLOW_SECTION_HEADER",
    "FlowEdge",
    "FlowError",
    "FlowGraph",
    "FlowNode",
    "FlowSlot",
    "NodeKind",
    "compile_flow",
    "flow_from_spec",
    "render_flow_instructions",
    "validate_flow",
]
