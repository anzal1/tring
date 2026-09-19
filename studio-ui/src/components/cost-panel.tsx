import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
import {
  Tooltip, TooltipContent, TooltipTrigger,
} from "@/components/ui/tooltip";
import { fmtUSD, type SessionEvent } from "@/lib/protocol";

const SLOTS = ["stt", "llm", "tts", "other"] as const;
const SLOT_COLOR: Record<(typeof SLOTS)[number], string> = {
  stt: "var(--chart-1)",
  llm: "var(--chart-2)",
  tts: "var(--chart-3)",
  other: "var(--chart-4)",
};
const slotOf = (c: string | undefined) =>
  (SLOTS as readonly string[]).includes(c ?? "") ? (c as (typeof SLOTS)[number]) : "other";

export function CostPanel({ costs }: { costs: SessionEvent[] }) {
  const total = costs.reduce((a, c) => a + (c.amount ?? 0), 0);
  const estAmt = costs.filter((c) => c.estimated).reduce((a, c) => a + (c.amount ?? 0), 0);

  const byComponent = new Map<string, number>();
  for (const c of costs) {
    const k = slotOf(c.component);
    byComponent.set(k, (byComponent.get(k) ?? 0) + (c.amount ?? 0));
  }
  const max = Math.max(...byComponent.values(), 1e-12);

  const byProvider = new Map<string, { units: number; unit: string; amount: number; est: boolean }>();
  for (const c of costs) {
    const key = `${c.provider} · ${c.unit_name}`;
    const cur = byProvider.get(key) ?? { units: 0, unit: c.unit_name ?? "", amount: 0, est: false };
    cur.units += c.units ?? 0;
    cur.amount += c.amount ?? 0;
    cur.est ||= c.estimated ?? false;
    byProvider.set(key, cur);
  }

  return (
    <Card className="gap-0 py-0">
      <CardHeader className="border-b py-3! [.border-b]:pb-3">
        <CardTitle className="text-[10.5px] font-medium tracking-[0.08em] uppercase text-muted-foreground">
          Cost
        </CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-4 p-4">
        <div>
          <div className="flex items-baseline gap-2">
            <span className="text-[26px] font-semibold tracking-tight tabular-nums">
              {fmtUSD(total, costs[0]?.currency)}
            </span>
            <span className="text-[11.5px] text-muted-foreground">this session</span>
          </div>
          {estAmt > 0 && (
            <span className="mt-2 inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] text-brand shadow-[0_0_0_1px_var(--border)]">
              ⚠︎ {Math.round((estAmt / total) * 100)}% of this total is estimated, not vendor-reported
            </span>
          )}
        </div>

        {byComponent.size > 0 && (
          <div className="flex flex-col gap-2">
            {SLOTS.filter((s) => byComponent.has(s)).map((slot) => (
              <Tooltip key={slot}>
                <TooltipTrigger
                  render={<div className="grid grid-cols-[64px_1fr_72px] items-center gap-2.5" />}
                >
                  <span className="text-[11.5px] text-muted-foreground">{slot}</span>
                  <div className="h-2">
                    <div
                      className="h-full min-w-0.5 rounded-r-[3px] transition-[width] duration-200"
                      style={{
                        width: `${((byComponent.get(slot) ?? 0) / max) * 100}%`,
                        background: SLOT_COLOR[slot],
                      }}
                    />
                  </div>
                  <span className="text-right font-mono text-[11px] tabular-nums text-muted-foreground">
                    {fmtUSD(byComponent.get(slot) ?? 0)}
                  </span>
                </TooltipTrigger>
                <TooltipContent side="left" className="font-mono text-[11px]">
                  {costs
                    .filter((c) => slotOf(c.component) === slot)
                    .map((c, i) => (
                      <div key={i}>
                        {c.provider}: {c.units?.toFixed(1)} {c.unit_name} → {fmtUSD(c.amount ?? 0)}
                      </div>
                    ))}
                </TooltipContent>
              </Tooltip>
            ))}
          </div>
        )}

        {byProvider.size > 0 && (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="h-7 text-[11px]">provider</TableHead>
                <TableHead className="h-7 text-[11px]">units</TableHead>
                <TableHead className="h-7 text-right text-[11px]">amount</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {[...byProvider.entries()].map(([key, v]) => (
                <TableRow key={key}>
                  <TableCell className="py-1.5 text-[11.5px] text-muted-foreground">
                    {key.split(" · ")[0]}
                  </TableCell>
                  <TableCell className="py-1.5 font-mono text-[11px] tabular-nums">
                    {v.units.toFixed(1)} {v.unit}
                    {v.est && (
                      <span className="text-brand" title="estimated, not vendor-reported"> ~est</span>
                    )}
                  </TableCell>
                  <TableCell className="py-1.5 text-right font-mono text-[11px] tabular-nums">
                    {fmtUSD(v.amount)}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
