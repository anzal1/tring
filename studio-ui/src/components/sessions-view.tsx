import { useCallback, useEffect, useMemo, useState } from "react";
import { CostPanel } from "@/components/cost-panel";
import { EventsPanel } from "@/components/events-panel";
import { TranscriptPanel } from "@/components/transcript-panel";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import { REPLAY_RATES, useReplay } from "@/hooks/use-replay";
import { fmtUSD, type SessionDetail, type SessionSummary } from "@/lib/protocol";

/** Past sessions, replayed through the live panels.
 *
 * The replay is not a recording: it is the stored event list run back through
 * the same fold and the same components the live session uses, at the pacing
 * the events' own `at` timestamps describe. Nothing here talks to a socket,
 * and nothing here renders a transcript of its own.
 */
export function SessionsView() {
  const [rows, setRows] = useState<SessionSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detail, setDetail] = useState<SessionDetail | null>(null);
  const [loading, setLoading] = useState<string | null>(null);

  const list = useCallback(async () => {
    try {
      const res = await fetch("/api/sessions");
      setRows((await res.json()) as SessionSummary[]);
      setError(null);
    } catch {
      setError("could not reach the studio server");
    }
  }, []);

  useEffect(() => {
    (async () => {
      await list();
    })();
  }, [list]);

  const open = useCallback(async (id: string) => {
    setLoading(id);
    try {
      const res = await fetch(`/api/sessions/${id}`);
      if (!res.ok) {
        setError(`session ${id} is no longer on disk`);
        return;
      }
      setDetail((await res.json()) as SessionDetail);
      setError(null);
    } catch {
      setError("could not reach the studio server");
    } finally {
      setLoading(null);
    }
  }, []);

  const events = useMemo(() => detail?.events ?? [], [detail]);
  const replay = useReplay(events);

  // The clock ticks every frame while playing, and these panels only change
  // when an event actually lands: memoising on the cursor's output keeps a
  // moving scrubber from re-rendering the whole transcript sixty times a
  // second.
  const panels = useMemo(
    () => ({
      cost: <CostPanel costs={replay.derived.costs} />,
      events: <EventsPanel events={replay.visible} />,
    }),
    [replay.derived, replay.visible],
  );

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 gap-3 overflow-y-auto p-4 lg:grid-cols-[minmax(320px,370px)_minmax(380px,1fr)_minmax(300px,350px)] lg:overflow-hidden">
      <SessionList
        rows={rows}
        error={error}
        selected={detail?.id ?? null}
        loading={loading}
        onRefresh={() => void list()}
        onOpen={(id) => void open(id)}
      />

      <TranscriptPanel
        title={detail ? `Replay · ${detail.agent ?? "unknown agent"}` : "Replay"}
        items={replay.derived.items}
        level={replay.level}
        empty={
          <div className="m-auto max-w-[280px] text-center text-xs text-muted-foreground">
            {!detail
              ? "Pick a session on the left. It replays through these panels with no server involved."
              : events.length === 0
                ? "This session stored no events before it ended."
                : "Press play, or drag the scrubber, to replay this session at its own pacing."}
          </div>
        }
        footer={<Transport replay={replay} enabled={detail !== null} />}
      />

      <div className="flex min-h-0 flex-col gap-3">
        {panels.cost}
        {panels.events}
      </div>
    </div>
  );
}

function SessionList({
  rows, error, selected, loading, onRefresh, onOpen,
}: {
  rows: SessionSummary[] | null;
  error: string | null;
  selected: string | null;
  loading: string | null;
  onRefresh: () => void;
  onOpen: (id: string) => void;
}) {
  return (
    <Card className="flex min-h-0 flex-col gap-0 overflow-hidden py-0">
      <CardHeader className="flex-row items-center border-b py-3! [.border-b]:pb-3">
        <CardTitle className="text-[10.5px] font-medium tracking-[0.08em] uppercase text-muted-foreground">
          Sessions
        </CardTitle>
        <Button size="sm" variant="ghost" className="press ml-auto" onClick={onRefresh}>
          Refresh
        </Button>
      </CardHeader>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {error && <div className="px-4 py-3 text-xs text-destructive">{error}</div>}
        {rows !== null && rows.length === 0 && !error && (
          <div className="px-4 py-3 text-xs leading-relaxed text-muted-foreground">
            No sessions stored yet. Start one in the Build view: every session
            is written to the studio's sessions directory as it happens.
          </div>
        )}
        {rows !== null && rows.length > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="h-7 text-[11px]">started</TableHead>
                <TableHead className="h-7 text-[11px]">agent</TableHead>
                <TableHead className="h-7 text-right text-[11px]">turns</TableHead>
                <TableHead className="h-7 text-right text-[11px]">cost</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((row) => (
                <TableRow
                  key={row.id}
                  role="button"
                  tabIndex={0}
                  aria-current={row.id === selected}
                  data-selected={row.id === selected ? "" : undefined}
                  onClick={() => onOpen(row.id)}
                  onKeyDown={(e) => {
                    if (e.key !== "Enter" && e.key !== " ") return;
                    e.preventDefault();
                    onOpen(row.id);
                  }}
                  className="lift cursor-pointer outline-none data-selected:bg-accent focus-visible:ring-3 focus-visible:ring-ring/50"
                >
                  <TableCell className="py-1.5 text-[11.5px] text-muted-foreground">
                    {when(row.started)}
                    {loading === row.id && <span className="ml-1.5 opacity-60">…</span>}
                  </TableCell>
                  <TableCell className="py-1.5 text-[11.5px]">{row.agent ?? "—"}</TableCell>
                  <TableCell className="py-1.5 text-right font-mono text-[11px] tabular-nums">
                    {row.turns}
                  </TableCell>
                  <TableCell className="py-1.5 text-right font-mono text-[11px] tabular-nums text-muted-foreground">
                    {fmtUSD(row.cost)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </div>
    </Card>
  );
}

function Transport({
  replay, enabled,
}: {
  replay: ReturnType<typeof useReplay>;
  enabled: boolean;
}) {
  const { t, duration, playing, rate, toggle, seek, setRate } = replay;
  return (
    <div className="flex items-center gap-2.5 border-t p-3">
      <Button
        size="sm"
        variant="secondary"
        className="press w-[62px]"
        disabled={!enabled || duration === 0}
        onClick={toggle}
      >
        {playing ? "Pause" : "Play"}
      </Button>
      <input
        type="range"
        className="scrub min-w-0 flex-1"
        aria-label="Replay position"
        min={0}
        max={duration || 1}
        step={0.05}
        disabled={!enabled || duration === 0}
        value={t}
        onChange={(e) => seek(Number(e.target.value))}
      />
      <span className="font-mono text-[11px] tabular-nums text-muted-foreground">
        {t.toFixed(1)}/{duration.toFixed(1)}s
      </span>
      <div className="flex gap-1" role="group" aria-label="Replay speed">
        {REPLAY_RATES.map((r) => (
          <button
            key={r}
            type="button"
            aria-pressed={rate === r}
            disabled={!enabled}
            onClick={() => setRate(r)}
            className={`rounded-full border px-2 py-0.5 text-[11px] transition-colors duration-150 disabled:opacity-50 ${
              rate === r ? "border-ring text-foreground/80" : "border-border text-muted-foreground"
            }`}
          >
            {r}x
          </button>
        ))}
      </div>
    </div>
  );
}

/** Sessions are stamped in UTC ISO-8601; developers read them in their own
 *  timezone, which is where the call actually happened for them. */
function when(iso: string): string {
  const at = new Date(iso);
  return Number.isNaN(at.getTime())
    ? iso
    : at.toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
}
