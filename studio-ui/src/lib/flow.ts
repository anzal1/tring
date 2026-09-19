/** Flow graphs, client side: the same JSON `src/tring/flow.py` compiles.
 *
 * The compiler is server side and UI free, which means this module owns
 * exactly two things the compiler refuses to know about: where a node sits on
 * the canvas, and what a node looks like before it is complete enough to
 * compile. Everything else here is translation, in both directions, between
 * the protocol's graph shape and React Flow's node/edge shape.
 *
 * Coordinates never go over the wire (docs/STUDIO_PROTOCOL.md is explicit
 * that they are the UI's own storage), so they live in `localStorage` keyed
 * by agent, and a graph arriving without them is laid out by depth from the
 * entry node. That keeps `metadata.flow` a description of a conversation
 * rather than a description of somebody's screen.
 */

import type { Edge, Node, XYPosition } from "@xyflow/react";
import type { ToolDef } from "@/lib/protocol";

export const NODE_KINDS = ["say", "ask", "branch", "tool", "handoff", "end"] as const;
export type NodeKind = (typeof NODE_KINDS)[number];

export interface FlowSlot {
  name: string;
  description: string;
  required?: boolean;
}

/** One step. The per-kind fields are optional here and required by the
 *  server's validator, so a half-written node is an editable node rather
 *  than an unrepresentable one. */
export interface FlowNode {
  id: string;
  kind: NodeKind;
  label?: string;
  text?: string;
  prompt?: string;
  slots?: FlowSlot[];
  condition?: string;
  tool?: ToolDef;
  target?: string;
}

/** A transition. `from`/`to` is the wire spelling, kept verbatim. */
export interface FlowEdge {
  from: string;
  to: string;
  when?: string | null;
}

export interface FlowGraph {
  name?: string;
  greeting?: string;
  entry?: string;
  nodes: FlowNode[];
  edges: FlowEdge[];
}

export const EMPTY_GRAPH: FlowGraph = { nodes: [], edges: [] };

/** Per-kind presentation. `dot` is the one place colour appears on the
 *  canvas: a 5px dot, reusing the validated series colours, so kind is
 *  scannable without painting six differently-coloured boxes. */
export interface KindMeta {
  dot: string;
  blurb: string;
  /** Terminal kinds end the path, so they have no outgoing handle at all:
   *  the canvas refuses the transition the compiler would reject anyway. */
  terminal: boolean;
}

export const KIND_META: Record<NodeKind, KindMeta> = {
  say: { dot: "var(--chart-1)", blurb: "speak a scripted line", terminal: false },
  ask: { dot: "var(--chart-2)", blurb: "collect slots from the caller", terminal: false },
  branch: { dot: "var(--chart-3)", blurb: "choose a path by condition", terminal: false },
  tool: { dot: "var(--chart-4)", blurb: "call a tool, then continue", terminal: false },
  handoff: { dot: "var(--brand)", blurb: "transfer to a human", terminal: true },
  end: { dot: "var(--muted-foreground)", blurb: "finish the call", terminal: true },
};

export type CanvasNodeData = { node: FlowNode };
export type CanvasEdgeData = { when: string | null };
export type CanvasNode = Node<CanvasNodeData, "flow">;
export type CanvasEdge = Edge<CanvasEdgeData>;

// ---------------------------------------------------------------- authoring

/** A new node of `kind`, with an id nothing else is using.
 *
 * `ask` arrives with one blank slot because an `ask` with no slots is a
 * `say` with a question mark, and the compiler says so; scaffolding the row
 * makes that rule visible before it is enforced.
 */
export function newNode(kind: NodeKind, existing: readonly FlowNode[]): FlowNode {
  const taken = new Set(existing.map((n) => n.id));
  let n = 1;
  while (taken.has(`${kind}_${n}`)) n++;
  const node: FlowNode = { id: `${kind}_${n}`, kind };
  if (kind === "ask") node.slots = [{ name: "", description: "", required: true }];
  if (kind === "tool") {
    node.tool = { name: "", description: "", parameters: {}, choreographed: true };
  }
  return node;
}

/** The one line of body text a node card shows under its kind. */
export function summarize(node: FlowNode): string {
  switch (node.kind) {
    case "say":
    case "end":
      return node.text ? `“${node.text}”` : "";
    case "ask": {
      const names = (node.slots ?? []).map((s) => s.name).filter(Boolean);
      return node.prompt ? `“${node.prompt}”` : names.length ? names.join(", ") : "";
    }
    case "branch":
      return node.condition ?? "";
    case "tool":
      return node.tool?.name ?? "";
    case "handoff":
      return node.target ? `→ ${node.target}` : "";
  }
}

/** Which node the conversation starts on, by the server's own rule: the
 *  explicit `entry`, else the single node nothing points at. Null means the
 *  compiler will ask for an explicit one, which the inspector then offers. */
export function entryOf(graph: FlowGraph): string | null {
  if (graph.entry && graph.nodes.some((n) => n.id === graph.entry)) return graph.entry;
  const targeted = new Set(graph.edges.map((e) => e.to));
  const roots = graph.nodes.filter((n) => !targeted.has(n.id));
  return roots.length === 1 ? roots[0].id : null;
}

// ----------------------------------------------------------------- geometry

const COLUMN = 264;
const ROW = 132;

/** Lay a graph out by distance from the entry node, left to right.
 *
 * Saved positions win: this runs for graphs that arrive from
 * `metadata.flow`, which is every graph that was authored on somebody else's
 * screen, and a first view that is legible beats a first view that is empty.
 */
export function layout(
  graph: FlowGraph,
  saved: Readonly<Record<string, XYPosition>>,
): Record<string, XYPosition> {
  const depth = new Map<string, number>();
  const entry = entryOf(graph) ?? graph.nodes[0]?.id;
  const queue: string[] = entry ? [entry] : [];
  if (entry) depth.set(entry, 0);
  for (let head = 0; head < queue.length; head++) {
    const id = queue[head];
    const from = depth.get(id) ?? 0;
    for (const edge of graph.edges) {
      if (edge.from !== id || depth.has(edge.to)) continue;
      depth.set(edge.to, from + 1);
      queue.push(edge.to);
    }
  }

  const used = new Map<number, number>();
  const out: Record<string, XYPosition> = {};
  for (const node of graph.nodes) {
    const stored = saved[node.id];
    if (stored) {
      out[node.id] = stored;
      continue;
    }
    // Nodes nothing reaches yet (just dropped, or orphaned by an edit) stack
    // in a column of their own past the graph, where they are findable.
    const column = depth.get(node.id) ?? maxDepth(depth) + 1;
    const row = used.get(column) ?? 0;
    used.set(column, row + 1);
    out[node.id] = { x: 40 + column * COLUMN, y: 32 + row * ROW };
  }
  return out;
}

function maxDepth(depth: ReadonlyMap<string, number>): number {
  let max = 0;
  for (const value of depth.values()) max = Math.max(max, value);
  return max;
}

const POSITION_KEY = "tring.studio.flow-positions";

/** Canvas positions for one agent. Storage failures are not errors here: a
 *  browser with storage disabled gets the computed layout every time, which
 *  is a worse canvas, not a broken one. */
export function loadPositions(agent: string): Record<string, XYPosition> {
  try {
    const all = JSON.parse(localStorage.getItem(POSITION_KEY) ?? "{}") as Record<
      string,
      Record<string, XYPosition>
    >;
    return all[agent || "default"] ?? {};
  } catch {
    return {};
  }
}

export function savePositions(agent: string, positions: Record<string, XYPosition>): void {
  try {
    const all = JSON.parse(localStorage.getItem(POSITION_KEY) ?? "{}") as Record<string, unknown>;
    all[agent || "default"] = positions;
    localStorage.setItem(POSITION_KEY, JSON.stringify(all));
  } catch {
    /* storage disabled or full: positions are a convenience, not state */
  }
}

// -------------------------------------------------------------- translation

export function toCanvas(
  graph: FlowGraph,
  positions: Readonly<Record<string, XYPosition>>,
): { nodes: CanvasNode[]; edges: CanvasEdge[] } {
  return {
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      type: "flow" as const,
      position: positions[node.id] ?? { x: 0, y: 0 },
      data: { node },
    })),
    edges: graph.edges.map((edge, i) => ({
      id: `e${i}_${edge.from}_${edge.to}`,
      source: edge.from,
      target: edge.to,
      label: edge.when ?? undefined,
      data: { when: edge.when ?? null },
    })),
  };
}

/** Canvas back to wire shape.
 *
 * Two things are dropped on the way out, both because the graph is about a
 * conversation and not about an editing session: empty strings (the compiler
 * treats "" as "not set", so sending it only adds noise to `agent.yaml`), and
 * a `when` on a transition that does not leave a `branch`. The second is a
 * stale edit rather than an intention, and the compiler rejects it outright.
 */
export function toGraph(
  nodes: readonly CanvasNode[],
  edges: readonly CanvasEdge[],
  meta: { name?: string; greeting?: string; entry?: string },
): FlowGraph {
  const specs = nodes.map((n) => n.data.node);
  const branches = new Set(specs.filter((n) => n.kind === "branch").map((n) => n.id));
  const graph: FlowGraph = {
    nodes: specs.map(clean),
    edges: edges.map((edge) => {
      const when = branches.has(edge.source) ? (edge.data?.when ?? null) : null;
      const out: FlowEdge = { from: edge.source, to: edge.target };
      if (when) out.when = when;
      return out;
    }),
  };
  if (meta.name?.trim()) graph.name = meta.name.trim();
  if (meta.greeting?.trim()) graph.greeting = meta.greeting.trim();
  if (meta.entry && specs.some((n) => n.id === meta.entry)) graph.entry = meta.entry;
  return graph;
}

function clean(node: FlowNode): FlowNode {
  const out: FlowNode = { id: node.id, kind: node.kind };
  if (node.label?.trim()) out.label = node.label.trim();
  if (node.text?.trim()) out.text = node.text;
  if (node.prompt?.trim()) out.prompt = node.prompt;
  if (node.condition?.trim()) out.condition = node.condition;
  if (node.target?.trim()) out.target = node.target;
  const slots = (node.slots ?? []).filter((s) => s.name.trim());
  if (slots.length) out.slots = slots;
  if (node.tool) out.tool = node.tool;
  return out;
}

export function positionsOf(nodes: readonly CanvasNode[]): Record<string, XYPosition> {
  return Object.fromEntries(nodes.map((n) => [n.id, n.position]));
}

/** Is this `metadata.flow` value actually a flow graph?
 *
 * `metadata` is free-form by design, so anything can be sitting under `flow`.
 * The check is shallow on purpose: a graph that parses but does not compile
 * should reach the canvas and be *shown* to be broken, not be hidden by a
 * strict type guard on the way in. */
export function isFlowGraph(value: unknown): value is FlowGraph {
  if (typeof value !== "object" || value === null) return false;
  const candidate = value as Partial<FlowGraph>;
  return Array.isArray(candidate.nodes) && Array.isArray(candidate.edges);
}
