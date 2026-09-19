import { useState } from "react";
import { Field } from "@/components/agent-panel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { KIND_META, type FlowNode, type FlowSlot } from "@/lib/flow";

/** Graph-level fields, which are the ones no node owns. */
export interface FlowMeta {
  name: string;
  greeting: string;
  /** "" means: let the server infer the single node nothing points at. */
  entry: string;
}

export interface EdgeSelection {
  id: string;
  source: string;
  target: string;
  when: string | null;
  /** Only transitions leaving a branch carry a condition; the compiler
   *  rejects a `when` anywhere else, so the field is not offered there. */
  branch: boolean;
}

interface Props {
  node: FlowNode | null;
  edge: EdgeSelection | null;
  meta: FlowMeta;
  nodes: FlowNode[];
  entryHint: string | null;
  onNode: (next: FlowNode) => void;
  onEdge: (id: string, when: string | null) => void;
  onMeta: (next: FlowMeta) => void;
  onRemoveNode: (id: string) => void;
  onRemoveEdge: (id: string) => void;
}

/** The side panel: whatever is selected, edited with the agent panel's own
 *  controls. Nothing is selected on an empty canvas, and the panel shows the
 *  graph's own fields then, so the space is never dead. */
export function FlowInspector(props: Props) {
  const { node, edge } = props;
  return (
    <Card className="flex min-h-0 flex-col gap-0 overflow-hidden py-0">
      <CardHeader className="flex-row items-center border-b py-3! [.border-b]:pb-3">
        <CardTitle className="text-[10.5px] font-medium tracking-[0.08em] uppercase text-muted-foreground">
          {node ? "Node" : edge ? "Transition" : "Flow"}
        </CardTitle>
        {node && (
          <span className="ml-auto font-mono text-[11px] text-muted-foreground">{node.id}</span>
        )}
      </CardHeader>
      <ScrollArea className="min-h-0 flex-1">
        <CardContent className="flex flex-col gap-3 p-4">
          {node ? (
            <NodeFields key={node.id} node={node} onNode={props.onNode} onRemove={props.onRemoveNode} />
          ) : edge ? (
            <EdgeFields edge={edge} onEdge={props.onEdge} onRemove={props.onRemoveEdge} />
          ) : (
            <GraphFields {...props} />
          )}
        </CardContent>
      </ScrollArea>
    </Card>
  );
}

function GraphFields({ meta, nodes, entryHint, onMeta }: Props) {
  return (
    <>
      <p className="text-[11.5px] leading-relaxed text-muted-foreground">
        Drag a step onto the canvas, then drag between the dots to connect.
        Select anything to edit it here.
      </p>
      <Field label="Name">
        <Input
          placeholder="inherits the saved agent's name"
          value={meta.name}
          onChange={(e) => onMeta({ ...meta, name: e.target.value })}
        />
      </Field>
      <Field label="Greeting">
        <Input
          placeholder="inherits the saved agent's greeting"
          value={meta.greeting}
          onChange={(e) => onMeta({ ...meta, greeting: e.target.value })}
        />
      </Field>
      <Field label="Entry node">
        <Select
          value={meta.entry}
          onValueChange={(v) => v != null && onMeta({ ...meta, entry: String(v) })}
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="infer" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="">infer from the graph</SelectItem>
            {nodes.map((n) => (
              <SelectItem key={n.id} value={n.id}>
                {n.id}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </Field>
      <p className="text-[11px] leading-relaxed text-muted-foreground">
        {meta.entry
          ? `The conversation starts on ${meta.entry}.`
          : entryHint
            ? `Inferred: ${entryHint} is the only node nothing points at.`
            : "Nothing points at more than one node, or at none, so the compiler will ask for an explicit entry."}
      </p>
    </>
  );
}

function EdgeFields({
  edge, onEdge, onRemove,
}: {
  edge: EdgeSelection;
  onEdge: (id: string, when: string | null) => void;
  onRemove: (id: string) => void;
}) {
  return (
    <>
      <div className="font-mono text-[11.5px] text-muted-foreground">
        {edge.source} → {edge.target}
      </div>
      {edge.branch ? (
        <>
          <Field label="When">
            <Input
              placeholder="the caller wants to book"
              value={edge.when ?? ""}
              onChange={(e) => onEdge(edge.id, e.target.value || null)}
            />
          </Field>
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Left empty this becomes the branch's default transition, rendered
            last as “Otherwise…”. A branch may have only one of those.
          </p>
        </>
      ) : (
        <p className="text-[11px] leading-relaxed text-muted-foreground">
          Only transitions leaving a <span className="font-mono">branch</span> carry a
          condition. This one runs as soon as its step is done.
        </p>
      )}
      <Button
        variant="ghost"
        size="sm"
        className="press self-start text-muted-foreground active:text-destructive"
        onClick={() => onRemove(edge.id)}
      >
        Remove transition
      </Button>
    </>
  );
}

function NodeFields({
  node, onNode, onRemove,
}: {
  node: FlowNode;
  onNode: (next: FlowNode) => void;
  onRemove: (id: string) => void;
}) {
  const set = (patch: Partial<FlowNode>) => onNode({ ...node, ...patch });
  return (
    <>
      <div className="flex items-center gap-2">
        <span
          aria-hidden
          className="inline-block size-[5px] rounded-full"
          style={{ background: KIND_META[node.kind].dot }}
        />
        <span className="font-mono text-[11px] text-muted-foreground">{node.kind}</span>
        <span className="text-[11px] text-muted-foreground">{KIND_META[node.kind].blurb}</span>
      </div>

      <Field label="Label">
        <Input
          placeholder="shown on the canvas, and to the model"
          value={node.label ?? ""}
          onChange={(e) => set({ label: e.target.value })}
        />
      </Field>

      {(node.kind === "say" || node.kind === "end") && (
        <Field label={node.kind === "say" ? "Line to speak" : "Closing line"}>
          <Textarea
            rows={3}
            value={node.text ?? ""}
            onChange={(e) => set({ text: e.target.value })}
          />
        </Field>
      )}

      {node.kind === "ask" && (
        <>
          <Field label="Question">
            <Textarea
              rows={2}
              value={node.prompt ?? ""}
              onChange={(e) => set({ prompt: e.target.value })}
            />
          </Field>
          <Slots slots={node.slots ?? []} onChange={(slots) => set({ slots })} />
        </>
      )}

      {node.kind === "branch" && (
        <Field label="Decide on">
          <Input
            placeholder="what the caller wants"
            value={node.condition ?? ""}
            onChange={(e) => set({ condition: e.target.value })}
          />
        </Field>
      )}

      {node.kind === "handoff" && (
        <>
          <Field label="Transfer to">
            <Input
              placeholder="the booking desk"
              value={node.target ?? ""}
              onChange={(e) => set({ target: e.target.value })}
            />
          </Field>
          <Field label="What the caller hears first">
            <Textarea
              rows={2}
              value={node.text ?? ""}
              onChange={(e) => set({ text: e.target.value })}
            />
          </Field>
        </>
      )}

      {node.kind === "tool" && <ToolFields node={node} onNode={onNode} />}

      <Button
        variant="ghost"
        size="sm"
        className="press self-start text-muted-foreground active:text-destructive"
        onClick={() => onRemove(node.id)}
      >
        Remove node
      </Button>
    </>
  );
}

/** What an `ask` step must come away with. The compiler refuses an `ask`
 *  with nothing to fill, so the list starts with a row rather than a button. */
function Slots({
  slots, onChange,
}: {
  slots: FlowSlot[];
  onChange: (slots: FlowSlot[]) => void;
}) {
  const replace = (i: number, slot: FlowSlot) =>
    onChange(slots.map((s, j) => (j === i ? slot : s)));
  return (
    <fieldset className="flex flex-col gap-2 rounded-xl border p-3">
      <legend className="px-1 text-[10.5px] tracking-[0.08em] uppercase text-muted-foreground">
        Slots
      </legend>
      {slots.map((slot, i) => (
        <div
          key={i}
          className="flex flex-col gap-2 rounded-2xl bg-secondary p-2.5 shadow-[0_0_0_1px_var(--border)]"
        >
          <div className="flex items-center gap-2">
            <Input
              className="flex-1 font-mono text-xs"
              placeholder="slot_name"
              value={slot.name}
              onChange={(e) => replace(i, { ...slot, name: e.target.value })}
            />
            <Button
              variant="ghost"
              size="sm"
              aria-label="Remove slot"
              className="press text-muted-foreground active:text-destructive"
              onClick={() => onChange(slots.filter((_, j) => j !== i))}
            >
              ✕
            </Button>
          </div>
          <Input
            placeholder="a ten digit mobile number (written for the model)"
            value={slot.description}
            onChange={(e) => replace(i, { ...slot, description: e.target.value })}
          />
          <label className="flex items-center gap-2 text-xs text-muted-foreground">
            <Checkbox
              checked={slot.required !== false}
              onCheckedChange={(v) => replace(i, { ...slot, required: v === true })}
            />
            required
          </label>
        </div>
      ))}
      <Button
        variant="ghost"
        size="sm"
        className="press"
        onClick={() => onChange([...slots, { name: "", description: "", required: true }])}
      >
        + Add slot
      </Button>
    </fieldset>
  );
}

/** The tool a `tool` node calls, carried on the node so a flow is a complete
 *  description of the agent it compiles to. Same fields, same order, as the
 *  agent panel's tool cards. */
function ToolFields({ node, onNode }: { node: FlowNode; onNode: (next: FlowNode) => void }) {
  const tool = node.tool ?? { name: "", description: "", parameters: {}, choreographed: true };
  // Serialised once per selected node, never on every keystroke: re-encoding
  // as the developer types would fight the cursor inside the textarea. The
  // inspector remounts this on selection (it keys the node editor by id), so
  // the initialiser is the whole of the synchronisation.
  const [params, setParams] = useState(() => JSON.stringify(tool.parameters ?? {}, null, 2));
  const [bad, setBad] = useState<string | null>(null);

  function onParams(text: string) {
    setParams(text);
    if (!text.trim()) {
      setBad(null);
      onNode({ ...node, tool: { ...tool, parameters: {} } });
      return;
    }
    try {
      const parsed: unknown = JSON.parse(text);
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        setBad("parameters must be a JSON object");
        return;
      }
      setBad(null);
      onNode({ ...node, tool: { ...tool, parameters: parsed as Record<string, unknown> } });
    } catch (e) {
      // Kept as text, not pushed onto the node: half-typed JSON is not a
      // schema, and the last good one stays until this one parses.
      setBad((e as Error).message);
    }
  }

  return (
    <fieldset className="flex flex-col gap-2 rounded-xl border p-3">
      <legend className="px-1 text-[10.5px] tracking-[0.08em] uppercase text-muted-foreground">
        Tool
      </legend>
      <Input
        className="font-mono text-xs"
        placeholder="tool_name"
        value={tool.name}
        onChange={(e) => onNode({ ...node, tool: { ...tool, name: e.target.value } })}
      />
      <Input
        placeholder="What this tool does (shown to the model)"
        value={tool.description}
        onChange={(e) => onNode({ ...node, tool: { ...tool, description: e.target.value } })}
      />
      <Textarea
        rows={3}
        spellCheck={false}
        className="font-mono text-[11px]"
        placeholder='{"type": "object", "properties": {…}}'
        aria-invalid={bad !== null}
        value={params}
        onChange={(e) => onParams(e.target.value)}
      />
      {bad && <span className="font-mono text-[11px] text-destructive">{bad}</span>}
      <label className="flex items-center gap-2 text-xs text-muted-foreground">
        <Checkbox
          checked={tool.choreographed}
          onCheckedChange={(v) =>
            onNode({ ...node, tool: { ...tool, choreographed: v === true } })
          }
        />
        choreographed (no dead air)
        {tool.choreographed && <Badge variant="outline" className="text-[10px]">schema-enforced</Badge>}
      </label>
    </fieldset>
  );
}
