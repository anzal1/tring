import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { AgentPanel } from "@/components/agent-panel";
import { ConsolePanel } from "@/components/console-panel";
import { CostPanel } from "@/components/cost-panel";
import { EventsPanel } from "@/components/events-panel";
import { SessionsView } from "@/components/sessions-view";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useSession } from "@/hooks/use-session";
import { EMPTY_GRAPH, isFlowGraph, type FlowGraph } from "@/lib/flow";
import type { AgentSpec, Meta } from "@/lib/protocol";

const CONN_LABEL = {
  idle: "disconnected",
  connecting: "connecting…",
  live: "session live",
  ended: "session ended",
  error: "connection error",
} as const;

/** The canvas is a third of the built bundle and is not on the path anyone
 *  takes to test an agent, so it loads when it is first opened. */
const FlowView = lazy(() =>
  import("@/components/flow-view").then((m) => ({ default: m.FlowView })),
);

const VIEWS = ["build", "flow", "sessions"] as const;
type View = (typeof VIEWS)[number];
const VIEW_LABEL: Record<View, string> = { build: "Build", flow: "Flow", sessions: "Sessions" };

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [yaml, setYaml] = useState("");
  const [spec, setSpec] = useState<AgentSpec | null>(null);
  const [unreachable, setUnreachable] = useState(false);
  const [view, setView] = useState<View>("build");
  const session = useSession();

  // The canvas is a view, not a page: it keeps its graph while the developer
  // goes off to test the agent it compiled, and the live session keeps
  // running while they are looking at it.
  const [flow, setFlow] = useState<FlowGraph>(EMPTY_GRAPH);
  const [flowSeed, setFlowSeed] = useState(0);
  const seeded = useRef<string | null>(null);
  const [incoming, setIncoming] = useState<{ text: string; nonce: number } | undefined>();

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

  // `metadata.flow` is what makes a compiled agent reopenable as the flow that
  // built it. The re-seed is keyed on the stored graph's own text, so saving
  // an unrelated persona edit does not throw away canvas work.
  useEffect(() => {
    const stored = spec?.metadata?.flow;
    if (!isFlowGraph(stored)) return;
    const json = JSON.stringify(stored);
    if (json === seeded.current) return;
    seeded.current = json;
    setFlow(stored);
    setFlowSeed((n) => n + 1);
  }, [spec]);

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
        setIncoming(undefined); // the compiled draft is the saved agent now
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
          <Tabs value={view} onValueChange={(v) => setView(String(v) as View)} className="ml-1">
            <TabsList className="h-7">
              {VIEWS.map((v) => (
                <TabsTrigger key={v} value={v} className="px-3 text-[11px]">
                  {VIEW_LABEL[v]}
                </TabsTrigger>
              ))}
            </TabsList>
          </Tabs>
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

        {view === "build" && (
          <main className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-y-auto p-4 lg:grid-cols-[minmax(320px,370px)_minmax(380px,1fr)_minmax(300px,350px)] lg:overflow-hidden">
            <AgentPanel
              key={yaml || "empty"}
              meta={meta}
              yaml={yaml}
              spec={spec}
              onSave={save}
              incoming={incoming}
            />
            <ConsolePanel session={session} />
            <div className="flex min-h-0 flex-col gap-3">
              <CostPanel costs={session.costs} />
              <EventsPanel events={session.events} />
            </div>
          </main>
        )}

        {view === "flow" && (
          <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <Suspense
              fallback={
                <span className="m-auto text-xs text-muted-foreground">loading the canvas…</span>
              }
            >
              <FlowView
                key={flowSeed}
                initial={flow}
                agent={agentName}
                onChange={setFlow}
                onApply={(text) => {
                  setIncoming({ text, nonce: Date.now() });
                  setView("build");
                }}
              />
            </Suspense>
          </main>
        )}

        {view === "sessions" && (
          <main className="flex min-h-0 flex-1 flex-col overflow-hidden">
            <SessionsView />
          </main>
        )}
      </div>
    </TooltipProvider>
  );
}
