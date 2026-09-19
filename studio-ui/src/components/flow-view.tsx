import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  addEdge,
  Background,
  BackgroundVariant,
  MarkerType,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  useReactFlow,
  type Connection,
  type OnSelectionChangeParams,
  type XYPosition,
} from "@xyflow/react";
import "@xyflow/react/dist/base.css";
import { FlowInspector, type FlowMeta } from "@/components/flow-inspector";
import { FlowMarks } from "@/components/flow-marks";
import { FlowNodeCard } from "@/components/flow-node";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import {
  entryOf,
  KIND_META,
  layout,
  loadPositions,
  newNode,
  NODE_KINDS,
  positionsOf,
  savePositions,
  toCanvas,
  toGraph,
  type CanvasEdge,
  type CanvasNode,
  type FlowGraph,
  type FlowNode,
  type NodeKind,
} from "@/lib/flow";
import type { CompileResult } from "@/lib/protocol";

const NODE_TYPES = { flow: FlowNodeCard };
const DRAG_KIND = "application/tring-node-kind";
const EDGE_DEFAULTS = {
  type: "smoothstep",
  markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
};

interface Props {
  /** The graph this canvas opens on: `spec.metadata.flow`, or nothing. */
  initial: FlowGraph;
  /** Agent name, used only as the key for canvas positions. */
  agent: string;
  onChange: (graph: FlowGraph) => void;
  /** Hand compiled YAML to the agent panel's draft. Saving stays a separate,
   *  deliberate act, exactly as `POST /api/flow/compile` intends. */
  onApply: (yaml: string) => void;
}

export function FlowView(props: Props) {
  // screenToFlowPosition (drag and drop) needs the store, so the canvas lives
  // inside the provider and the view is a thin shell around it.
  return (
    <ReactFlowProvider>
      <FlowCanvas {...props} />
    </ReactFlowProvider>
  );
}

function FlowCanvas({ initial, agent, onChange, onApply }: Props) {
  // Seeded once per mount, deliberately: the parent remounts this view when a
  // different flow arrives, and re-deriving from `initial` on every render
  // would undo every drag the moment the graph changed.
  const [seed] = useState(() => toCanvas(initial, layout(initial, loadPositions(agent))));
  const [nodes, setNodes, onNodesChange] = useNodesState<CanvasNode>(seed.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState<CanvasEdge>(seed.edges);
  const [meta, setMeta] = useState<FlowMeta>({
    name: initial.name ?? "",
    greeting: initial.greeting ?? "",
    entry: initial.entry ?? "",
  });
  const [selection, setSelection] = useState<{ kind: "node" | "edge"; id: string } | null>(null);
  const [result, setResult] = useState<CompileResult | null>(null);
  const [busy, setBusy] = useState(false);
  const wrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition, fitView } = useReactFlow();

  const graph = useMemo(() => toGraph(nodes, edges, meta), [nodes, edges, meta]);

  useEffect(() => {
    onChange(graph);
  }, [graph, onChange]);

  // Positions are stored by their own value, not on every node change: typing
  // in the inspector rewrites the node list many times a second and moves
  // nothing, and localStorage should not hear about any of it.
  const geometry = useMemo(() => JSON.stringify(positionsOf(nodes)), [nodes]);
  useEffect(() => {
    savePositions(agent, JSON.parse(geometry) as Record<string, XYPosition>);
  }, [agent, geometry]);

  // ------------------------------------------------------------- authoring

  const addNode = useCallback(
    (kind: NodeKind, position: XYPosition) => {
      setNodes((prev) => {
        const node = newNode(kind, prev.map((n) => n.data.node));
        return [
          ...prev.map((n) => ({ ...n, selected: false })),
          { id: node.id, type: "flow" as const, position, selected: true, data: { node } },
        ];
      });
    },
    [setNodes],
  );

  /** Clicking a palette entry drops the node in the middle of the canvas.
   *  Dragging is the better gesture and not everyone has one; a keyboard
   *  should be able to build a flow too. */
  const addCentered = useCallback(
    (kind: NodeKind) => {
      const rect = wrapper.current?.getBoundingClientRect();
      if (!rect) return;
      // Stepped down and across, by more than a card, so clicking the palette
      // four times in a row builds a readable stack instead of a pile.
      const step = nodes.length % 4;
      addNode(
        kind,
        screenToFlowPosition({
          x: rect.x + rect.width / 2 - 99 + step * 32,
          y: rect.y + rect.height / 2 - 110 + step * 78,
        }),
      );
    },
    [addNode, nodes.length, screenToFlowPosition],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const kind = e.dataTransfer.getData(DRAG_KIND) as NodeKind;
      if (!(NODE_KINDS as readonly string[]).includes(kind)) return;
      addNode(kind, screenToFlowPosition({ x: e.clientX, y: e.clientY }));
    },
    [addNode, screenToFlowPosition],
  );

  /** A step has exactly one way out (the compiler says so), so connecting a
   *  second one replaces the first rather than building a graph that is
   *  guaranteed to be rejected. Branches keep every transition they are
   *  given: choosing between them is the whole point of a branch. */
  const onConnect = useCallback(
    (connection: Connection) => {
      const source = nodes.find((n) => n.id === connection.source)?.data.node;
      setEdges((prev) => {
        const kept =
          source && source.kind !== "branch"
            ? prev.filter((e) => e.source !== connection.source)
            : prev;
        return addEdge<CanvasEdge>({ ...connection, data: { when: null } }, kept);
      });
    },
    [nodes, setEdges],
  );

  const onSelectionChange = useCallback((params: OnSelectionChangeParams) => {
    if (params.nodes.length === 1) setSelection({ kind: "node", id: params.nodes[0].id });
    else if (params.edges.length === 1) setSelection({ kind: "edge", id: params.edges[0].id });
    else setSelection(null);
  }, []);

  const updateNode = useCallback(
    (next: FlowNode) => {
      setNodes((prev) => prev.map((n) => (n.id === next.id ? { ...n, data: { node: next } } : n)));
    },
    [setNodes],
  );

  const updateEdge = useCallback(
    (id: string, when: string | null) => {
      setEdges((prev) =>
        prev.map((e) => (e.id === id ? { ...e, data: { when }, label: when ?? undefined } : e)),
      );
    },
    [setEdges],
  );

  const removeNode = useCallback(
    (id: string) => {
      setNodes((prev) => prev.filter((n) => n.id !== id));
      setEdges((prev) => prev.filter((e) => e.source !== id && e.target !== id));
      setSelection(null);
    },
    [setEdges, setNodes],
  );

  const removeEdge = useCallback(
    (id: string) => {
      setEdges((prev) => prev.filter((e) => e.id !== id));
      setSelection(null);
    },
    [setEdges],
  );

  // -------------------------------------------------------------- compiling

  const compile = useCallback(async () => {
    setBusy(true);
    setResult(null);
    try {
      const res = await fetch("/api/flow/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ flow: graph }),
      });
      setResult((await res.json()) as CompileResult);
    } catch {
      setResult({ ok: false, error: "network error: could not reach the studio server" });
    } finally {
      setBusy(false);
    }
  }, [graph]);

  // Every compiler message leads with its subject, so the node it is about is
  // the node the canvas should be pointing at.
  const problem = useMemo(
    () => (result && !result.ok ? (/node '([^']+)'/.exec(result.error)?.[1] ?? null) : null),
    [result],
  );

  const selected = selection?.kind === "node" ? nodes.find((n) => n.id === selection.id) : undefined;
  const selectedEdge =
    selection?.kind === "edge" ? edges.find((e) => e.id === selection.id) : undefined;
  const marks = useMemo(
    () => ({ entry: entryOf(graph), problem }),
    [graph, problem],
  );

  return (
    <div className="relative grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-hidden p-4 lg:grid-cols-[minmax(170px,200px)_minmax(360px,1fr)_minmax(300px,340px)]">
      <Palette onAdd={addCentered} />

      <Card className="flex min-h-0 flex-col gap-0 overflow-hidden py-0">
        <CardHeader className="flex-row items-center border-b py-3! [.border-b]:pb-3">
          <CardTitle className="text-[10.5px] font-medium tracking-[0.08em] uppercase text-muted-foreground">
            Flow
          </CardTitle>
          <div className="ml-auto flex items-center gap-2">
            <span className="mr-1 text-[11px] text-muted-foreground tabular-nums">
              {nodes.length} steps · {edges.length} transitions
            </span>
            <Button
              size="sm"
              variant="ghost"
              className="press"
              disabled={nodes.length === 0}
              onClick={() => fitView({ padding: 0.15, maxZoom: 1, minZoom: 0.5, duration: 200 })}
            >
              Fit
            </Button>
            <Button
              size="sm"
              className="press"
              disabled={busy || nodes.length === 0}
              onClick={() => void compile()}
            >
              {busy ? "Compiling…" : "Compile to agent"}
            </Button>
          </div>
        </CardHeader>

        {result && !result.ok && (
          <div className="enter border-b px-4 py-2 font-mono text-[11px] leading-relaxed text-destructive">
            {result.error}
          </div>
        )}

        <div
          ref={wrapper}
          className="flow-canvas relative min-h-0 flex-1"
          onDragOver={(e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = "move";
          }}
          onDrop={onDrop}
        >
          <FlowMarks value={marks}>
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={NODE_TYPES}
              defaultEdgeOptions={EDGE_DEFAULTS}
              onNodesChange={onNodesChange}
              onEdgesChange={onEdgesChange}
              onConnect={onConnect}
              onSelectionChange={onSelectionChange}
              colorMode="dark"
              fitView
              // Fitting must never zoom in past 1 (two nodes at 160% blow up
              // every hairline in the design system) and never below 0.5,
              // where a long flow becomes a diagram of itself. Past that the
              // graph opens partly off-screen, which is what panning is for.
              fitViewOptions={{ padding: 0.15, maxZoom: 1, minZoom: 0.5 }}
              minZoom={0.3}
              maxZoom={1.6}
            >
              <Background variant={BackgroundVariant.Dots} gap={22} size={1} />
            </ReactFlow>
          </FlowMarks>
          {nodes.length === 0 && (
            <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
              <span className="max-w-[260px] text-center text-xs text-muted-foreground">
                Drag a step from the palette, or click one. Steps compile to
                numbered instructions in the agent's persona.
              </span>
            </div>
          )}
        </div>
      </Card>

      <FlowInspector
        node={selected?.data.node ?? null}
        edge={
          selectedEdge
            ? {
                id: selectedEdge.id,
                source: selectedEdge.source,
                target: selectedEdge.target,
                when: selectedEdge.data?.when ?? null,
                branch:
                  nodes.find((n) => n.id === selectedEdge.source)?.data.node.kind === "branch",
              }
            : null
        }
        meta={meta}
        nodes={nodes.map((n) => n.data.node)}
        entryHint={marks.entry}
        onNode={updateNode}
        onEdge={updateEdge}
        onMeta={setMeta}
        onRemoveNode={removeNode}
        onRemoveEdge={removeEdge}
      />

      {result?.ok && (
        <Review
          yaml={result.yaml}
          note={result.note}
          onApply={() => {
            onApply(result.yaml);
            setResult(null);
          }}
          onClose={() => setResult(null)}
        />
      )}
    </div>
  );
}

function Palette({ onAdd }: { onAdd: (kind: NodeKind) => void }) {
  return (
    <Card className="flex min-h-0 flex-col gap-0 overflow-hidden py-0">
      <CardHeader className="border-b py-3! [.border-b]:pb-3">
        <CardTitle className="text-[10.5px] font-medium tracking-[0.08em] uppercase text-muted-foreground">
          Steps
        </CardTitle>
      </CardHeader>
      <div className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto p-3">
        {NODE_KINDS.map((kind) => (
          <button
            key={kind}
            type="button"
            draggable
            onDragStart={(e) => {
              e.dataTransfer.setData(DRAG_KIND, kind);
              e.dataTransfer.effectAllowed = "move";
            }}
            onClick={() => onAdd(kind)}
            className="press lift flex w-full flex-col gap-0.5 rounded-2xl bg-secondary px-2.5 py-2 text-left shadow-[0_0_0_1px_var(--border)] outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
          >
            <span className="flex items-center gap-1.5">
              <span
                aria-hidden
                className="inline-block size-[5px] rounded-full"
                style={{ background: KIND_META[kind].dot }}
              />
              <span className="font-mono text-[11px]">{kind}</span>
            </span>
            <span className="text-[10.5px] leading-snug text-muted-foreground">
              {KIND_META[kind].blurb}
            </span>
          </button>
        ))}
      </div>
    </Card>
  );
}

/** What the flow compiles to, before anything is saved.
 *
 * Reading it is the point: the compiler writes the persona, and a developer
 * should see the prose their graph produced before it becomes their agent.
 * "Apply to editor" only loads the draft; the save button stays where it has
 * always been.
 */
function Review({
  yaml, note, onApply, onClose,
}: {
  yaml: string;
  note?: string;
  onApply: () => void;
  onClose: () => void;
}) {
  return (
    <div
      role="dialog"
      aria-label="Compiled agent"
      className="enter absolute inset-y-4 right-4 z-10 flex w-[min(540px,calc(100%-2rem))] flex-col overflow-hidden rounded-xl bg-popover ring-1 ring-foreground/10"
    >
      <div className="flex items-center gap-3 border-b px-4 py-3">
        <span className="text-[10.5px] font-medium tracking-[0.08em] uppercase text-muted-foreground">
          Compiled agent
        </span>
        <Button size="sm" variant="ghost" className="press ml-auto" onClick={onClose}>
          Close
        </Button>
        <Button size="sm" className="press" onClick={onApply}>
          Apply to editor
        </Button>
      </div>
      {note && (
        <div className="border-b px-4 py-2 text-[11px] leading-relaxed text-brand">{note}</div>
      )}
      <Textarea
        readOnly
        spellCheck={false}
        value={yaml}
        aria-label="Compiled agent spec"
        className="min-h-0 flex-1 resize-none overflow-auto rounded-none border-0 bg-transparent font-mono text-[11px] leading-relaxed field-sizing-fixed focus-visible:ring-0"
      />
      <div className="border-t px-4 py-2 text-[11px] text-muted-foreground">
        Nothing is saved yet. Apply loads this into the agent editor's YAML
        draft; saving stays your call.
      </div>
    </div>
  );
}
