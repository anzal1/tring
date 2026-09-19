import { useEffect, useRef, useState } from "react";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { EVENT_GROUP, GROUPS, summarize, type SessionEvent } from "@/lib/protocol";

const GROUP_DOT: Record<string, string> = {
  speech: "var(--chart-1)",
  tool: "var(--chart-2)",
  cost: "var(--chart-3)",
  session: "var(--muted-foreground)",
  error: "var(--destructive)",
};

export function EventsPanel({ events }: { events: SessionEvent[] }) {
  const [active, setActive] = useState<Set<string>>(new Set(GROUPS));
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const n = scrollRef.current;
    if (n) n.scrollTop = n.scrollHeight;
  }, [events]);

  const toggle = (g: string) =>
    setActive((prev) => {
      const next = new Set(prev);
      if (next.has(g)) next.delete(g);
      else next.add(g);
      return next;
    });

  return (
    <Card className="flex min-h-0 flex-1 flex-col gap-0 overflow-hidden py-0">
      <CardHeader className="flex-row items-center border-b py-3! [.border-b]:pb-3">
        <CardTitle className="text-[10.5px] font-medium tracking-[0.08em] uppercase text-muted-foreground">
          Events
        </CardTitle>
        <span className="ml-auto text-xs tabular-nums text-muted-foreground">
          {events.length > 0 ? events.length : ""}
        </span>
      </CardHeader>

      <div className="flex flex-wrap gap-1.5 px-4 pt-3" role="group" aria-label="Filter events">
        {GROUPS.map((g) => (
          <button
            key={g}
            aria-pressed={active.has(g)}
            onClick={() => toggle(g)}
            className={`rounded-full border px-2.5 py-0.5 text-[11px] transition-colors duration-150 ${
              active.has(g) ? "border-ring text-foreground/80" : "border-border text-muted-foreground"
            }`}
          >
            {g}
          </button>
        ))}
      </div>

      <div ref={scrollRef} className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto p-4 pt-2.5">
        {events.map((e, i) => {
          const g = EVENT_GROUP[e.type] ?? "session";
          if (!active.has(g)) return null;
          return (
            <div key={i} className="grid grid-cols-[40px_128px_1fr] items-baseline gap-2 text-[11.5px]">
              <span className="font-mono text-[10.5px] tabular-nums text-muted-foreground">
                {e.at != null ? `${e.at.toFixed(1)}s` : ""}
              </span>
              <span
                className={`truncate font-mono text-[10.5px] ${
                  g === "error" ? "text-destructive" : "text-muted-foreground"
                }`}
              >
                <span
                  className="mr-1.5 inline-block size-[5px] rounded-full align-[1px]"
                  style={{ background: GROUP_DOT[g] }}
                />
                {e.type}
              </span>
              <span className="text-muted-foreground [overflow-wrap:anywhere]">{summarize(e)}</span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}
