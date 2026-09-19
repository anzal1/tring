import { useEffect, useState } from "react";
import { AgentPanel } from "@/components/agent-panel";
import { ConsolePanel } from "@/components/console-panel";
import { CostPanel } from "@/components/cost-panel";
import { EventsPanel } from "@/components/events-panel";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useSession } from "@/hooks/use-session";
import type { AgentSpec, Meta } from "@/lib/protocol";

const CONN_LABEL = {
  idle: "disconnected",
  connecting: "connecting…",
  live: "session live",
  ended: "session ended",
  error: "connection error",
} as const;

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [yaml, setYaml] = useState("");
  const [spec, setSpec] = useState<AgentSpec | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const session = useSession();

  useEffect(() => {
    (async () => {
      try {
        setMeta(await (await fetch("/api/meta")).json());
        const { yaml: y } = await (await fetch("/api/agent")).json();
        setYaml(y);
        try {
          setSpec(JSON.parse(y) as AgentSpec);
        } catch {
          setSpec(null); // hand-written YAML: raw editor mode
        }
      } catch {
        setUnreachable(true);
      }
    })();
  }, []);

  async function save(body: string): Promise<{ ok: boolean; error?: string }> {
    try {
      const res = await (
        await fetch("/api/agent", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ yaml: body }),
        })
      ).json();
      if (res.ok) {
        setYaml(body);
        try {
          setSpec(JSON.parse(body) as AgentSpec);
        } catch {
          /* stays raw */
        }
      }
      return res;
    } catch {
      return { ok: false, error: "network error: could not reach the studio server" };
    }
  }

  const agentName = spec?.name ?? /^\s*name:\s*(.+)$/m.exec(yaml)?.[1]?.trim() ?? "";
  const conn = unreachable ? "error" : session.conn;

  return (
    <TooltipProvider delay={300}>
      <div className="flex h-screen flex-col">
        <header className="flex items-center gap-3 border-b px-5 py-2.5">
          <span className="text-[13.5px] font-semibold tracking-tight">
            <span className="mr-1.5 opacity-90">🔔</span>tring
            <span className="ml-0.5 font-normal text-muted-foreground">studio</span>
          </span>
          {agentName && (
            <span className="font-mono text-[11.5px] text-muted-foreground">{agentName}</span>
          )}
          <span className="ml-auto flex items-center gap-2 text-xs text-muted-foreground">
            <span
              className="size-[7px] rounded-full transition-colors duration-150"
              style={{
                background:
                  conn === "live"
                    ? "var(--good)"
                    : conn === "error"
                      ? "var(--destructive)"
                      : "var(--muted-foreground)",
              }}
            />
            {CONN_LABEL[conn]}
          </span>
        </header>

        <main className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-y-auto p-4 lg:grid-cols-[minmax(320px,370px)_minmax(380px,1fr)_minmax(300px,350px)] lg:overflow-hidden">
          <AgentPanel key={yaml || "empty"} meta={meta} yaml={yaml} spec={spec} onSave={save} />
          <ConsolePanel session={session} />
          <div className="flex min-h-0 flex-col gap-3">
            <CostPanel costs={session.costs} />
            <EventsPanel events={session.events} />
          </div>
        </main>
      </div>
    </TooltipProvider>
  );
}
