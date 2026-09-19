import { useContext } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import { FlowMarks } from "@/components/flow-marks";
import { KIND_META, summarize, type CanvasNode } from "@/lib/flow";

/** One step on the canvas: a tool-card in the studio's vocabulary.
 *
 * Same surface, same hairline ring, same radius as the tool cards in the
 * agent panel, because it is the same kind of object: a small editable thing
 * that belongs to a bigger one. The only colour is the 5px kind dot.
 */
export function FlowNodeCard({ id, data, selected }: NodeProps<CanvasNode>) {
  const { entry, problem } = useContext(FlowMarks);
  const node = data.node;
  const meta = KIND_META[node.kind];
  const body = summarize(node);
  const ring =
    problem === id
      ? "var(--destructive)"
      : selected
        ? "var(--ring)"
        : "var(--border)";

  return (
    <div
      className="lift w-[198px] rounded-2xl bg-secondary px-3 py-2"
      style={{ boxShadow: `0 0 0 1px ${ring}` }}
    >
      <Handle type="target" position={Position.Left} />
      <div className="flex items-center gap-1.5">
        <span
          aria-hidden
          className="inline-block size-[5px] shrink-0 rounded-full"
          style={{ background: meta.dot }}
        />
        <span className="font-mono text-[10px] tracking-[0.08em] uppercase text-muted-foreground">
          {node.kind}
        </span>
        {entry === id && (
          <span className="ml-auto text-[10px] tracking-[0.06em] uppercase text-muted-foreground">
            start
          </span>
        )}
      </div>
      <div className="mt-0.5 text-[12.5px] leading-snug">{node.label || node.id}</div>
      {body && (
        <div className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-muted-foreground">
          {body}
        </div>
      )}
      {!meta.terminal && <Handle type="source" position={Position.Right} />}
    </div>
  );
}
