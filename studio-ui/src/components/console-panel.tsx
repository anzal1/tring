import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import type { Session } from "@/hooks/use-session";
import type { TranscriptItem } from "@/lib/protocol";

export function ConsolePanel({ session }: { session: Session }) {
  const { conn, items, speaking, start, reset, send } = session;
  const [text, setText] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);
  const live = conn === "live";

  useEffect(() => {
    const n = scrollRef.current;
    if (n) n.scrollTop = n.scrollHeight;
  }, [items]);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    send(text);
    setText("");
  }

  return (
    <Card className="flex min-h-0 flex-col gap-0 overflow-hidden py-0">
      <CardHeader className="flex-row items-center border-b py-3! [.border-b]:pb-3">
        <CardTitle className="text-[10.5px] font-medium tracking-[0.08em] uppercase text-muted-foreground">
          Test console
        </CardTitle>
        <div className="ml-auto flex gap-2">
          <Button size="sm" variant="secondary" className="press" onClick={start}>
            Start
          </Button>
          <Button size="sm" variant="ghost" className="press" disabled={!live} onClick={reset}>
            Reset
          </Button>
        </div>
      </CardHeader>

      <div ref={scrollRef} className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto p-4">
        {items.length === 0 ? (
          <div className="m-auto flex max-w-[300px] flex-col items-center gap-2 text-center text-xs text-muted-foreground">
            <span className="text-2xl opacity-35">🔔</span>
            <span>
              Save an agent, then <strong className="text-foreground">Start</strong> a session.
              Every event streams into the panels on the right.
            </span>
          </div>
        ) : (
          items.map((item, i) => <Item key={i} item={item} />)
        )}
      </div>

      {/* speech-activity hairline: neutral, motion is not the only cue (dot in header) */}
      <div aria-hidden className="h-0.5 overflow-hidden">
        <div
          className="h-full origin-left bg-muted-foreground transition-transform duration-150"
          style={{ transform: `scaleX(${speaking ? 1 : 0})` }}
        />
      </div>

      <form onSubmit={submit} className="flex gap-2 border-t p-3">
        <Input
          placeholder="Say something…"
          aria-label="Message"
          disabled={!live}
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <Button type="submit" className="press" disabled={!live}>
          Send
        </Button>
      </form>
    </Card>
  );
}

function Item({ item }: { item: TranscriptItem }) {
  switch (item.kind) {
    case "sys":
      return (
        <div className={`enter self-center text-[11.5px] ${item.bad ? "text-destructive" : "text-muted-foreground"}`}>
          {item.text}
        </div>
      );
    case "user":
      return (
        <div className="enter max-w-[82%] self-end rounded-xl bg-accent px-3 py-1.5 text-[13px] whitespace-pre-wrap">
          {item.text}
        </div>
      );
    case "bot":
      return (
        <div className="enter max-w-[82%] self-start rounded-xl bg-secondary px-3 py-1.5 text-[13px] whitespace-pre-wrap shadow-[0_0_0_1px_var(--border)]">
          {item.text}
        </div>
      );
    case "interrupt":
      return (
        <div className="enter self-center rounded-full px-3 py-0.5 text-[11px] text-muted-foreground shadow-[0_0_0_1px_var(--border)]">
          caller interrupted, unheard text annotated
        </div>
      );
    case "tool":
      return (
        <div className="enter flex max-w-[82%] flex-col gap-0.5 self-start rounded-[14px] bg-secondary px-3 py-2 text-xs shadow-[0_0_0_1px_var(--border)]">
          <span className="font-mono text-[11.5px] text-muted-foreground">
            <span
              className="mr-2 inline-block size-1.5 rounded-full align-[1px]"
              style={{ background: item.done && !item.done.ok ? "var(--destructive)" : "var(--chart-2)" }}
            />
            {item.name}
          </span>
          {item.waiting && <span className="italic text-brand">“{item.waiting}”</span>}
          {item.done &&
            (item.done.ok ? (
              <span className="text-muted-foreground">
                done in {item.done.latency?.toFixed(2) ?? "?"}s
                {item.done.summary ? `, ${item.done.summary}` : ""}
              </span>
            ) : (
              <span className="text-destructive">{item.done.summary ?? "failed"}</span>
            ))}
        </div>
      );
  }
}
