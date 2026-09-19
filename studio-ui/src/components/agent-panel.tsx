import { useEffect, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import type { AgentSpec, Meta, ToolDef } from "@/lib/protocol";

interface Props {
  meta: Meta | null;
  yaml: string;
  spec: AgentSpec | null; // null => hand-written YAML, raw editor only
  onSave: (yamlBody: string) => Promise<{ ok: boolean; error?: string }>;
}

const EMPTY_TOOL: ToolDef = { name: "", description: "", parameters: {}, choreographed: true };

/** The form serializes to JSON, which is valid YAML, so the server's
 *  yaml.safe_load path is untouched. Hand-written YAML gets the raw editor
 *  rather than a lossy round-trip through a JS YAML parser. */
export function AgentPanel({ meta, yaml, spec, onSave }: Props) {
  const yamlOnly = spec === null;
  const [tab, setTab] = useState(yamlOnly ? "yaml" : "form");
  const [draft, setDraft] = useState<AgentSpec>(
    spec ?? {
      name: "", persona: "",
      language: { primary: "en", lock: true },
      runtime: { mode: "cascade", routing: { default: {} } },
      tools: [],
    },
  );
  const [paramDrafts, setParamDrafts] = useState<string[]>(
    (spec?.tools ?? []).map((t) => JSON.stringify(t.parameters, null, 2)),
  );
  const [yamlDraft, setYamlDraft] = useState(yaml);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => setYamlDraft(yaml), [yaml]);
  useEffect(() => {
    if (spec) {
      setDraft(spec);
      setParamDrafts(spec.tools.map((t) => JSON.stringify(t.parameters, null, 2)));
    }
  }, [spec]);

  function buildBody(): string {
    if (tab === "yaml" || yamlOnly) return yamlDraft;
    const tools = draft.tools
      .map((t, i) => {
        if (!t.name.trim()) return null;
        const params = paramDrafts[i]?.trim() ? JSON.parse(paramDrafts[i]) : {};
        return { ...t, name: t.name.trim(), parameters: params };
      })
      .filter(Boolean);
    return JSON.stringify({ ...draft, tools }, null, 2);
  }

  async function save() {
    setMsg(null);
    let body: string;
    try {
      body = buildBody();
    } catch (e) {
      setMsg({ ok: false, text: `tool parameters: ${(e as Error).message}` });
      return;
    }
    const res = await onSave(body);
    setMsg(res.ok ? { ok: true, text: "saved" } : { ok: false, text: res.error ?? "rejected" });
  }

  const routing = draft.runtime.routing.default;
  const setRouting = (slot: "stt" | "llm" | "tts", v: string) =>
    setDraft({
      ...draft,
      runtime: { ...draft.runtime, routing: { default: { ...routing, [slot]: v } } },
    });

  return (
    <Card className="flex min-h-0 flex-col gap-0 overflow-hidden py-0">
      <CardHeader className="flex-row items-center border-b py-3! [.border-b]:pb-3">
        <CardTitle className="text-[10.5px] font-medium tracking-[0.08em] uppercase text-muted-foreground">
          Agent
        </CardTitle>
        <Tabs value={tab} onValueChange={setTab} className="ml-auto">
          <TabsList className="h-7">
            <TabsTrigger value="form" disabled={yamlOnly} className="px-3 text-[11px]">Form</TabsTrigger>
            <TabsTrigger value="yaml" className="px-3 text-[11px]">YAML</TabsTrigger>
          </TabsList>
        </Tabs>
      </CardHeader>

      {yamlOnly && (
        <div className="mx-4 mt-3 rounded-lg border px-3 py-2 text-xs text-muted-foreground">
          Hand-written YAML detected. Edit it as text; the form view activates
          after the next save from Studio.
        </div>
      )}

      {tab === "form" && !yamlOnly ? (
        <ScrollArea className="min-h-0 flex-1">
          <CardContent className="flex flex-col gap-4 p-4">
            <div className="grid grid-cols-2 gap-3">
              <Field label="Name">
                <Input value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} />
              </Field>
              <Field label="Greeting">
                <Input
                  placeholder="optional"
                  value={draft.greeting ?? ""}
                  onChange={(e) => setDraft({ ...draft, greeting: e.target.value || undefined })}
                />
              </Field>
            </div>
            <Field label="Persona">
              <Textarea
                rows={3}
                value={draft.persona}
                onChange={(e) => setDraft({ ...draft, persona: e.target.value })}
              />
            </Field>
            <div className="grid grid-cols-3 items-end gap-3">
              <Field label="Language">
                <Input
                  value={draft.language.primary}
                  onChange={(e) =>
                    setDraft({ ...draft, language: { ...draft.language, primary: e.target.value } })
                  }
                />
              </Field>
              <Field label="Mode">
                <Select
                  value={draft.runtime.mode}
                  onValueChange={(v) =>
                    v != null &&
                    setDraft({ ...draft, runtime: { ...draft.runtime, mode: v as AgentSpec["runtime"]["mode"] } })
                  }
                >
                  <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {["cascade", "s2s", "hybrid"].map((m) => (
                      <SelectItem key={m} value={m}>{m}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
              <label className="flex items-center gap-2 pb-2 text-xs text-muted-foreground">
                <Checkbox
                  checked={draft.language.lock}
                  onCheckedChange={(v) =>
                    setDraft({ ...draft, language: { ...draft.language, lock: v === true } })
                  }
                />
                Language lock
              </label>
            </div>

            <fieldset className="rounded-xl border p-3">
              <legend className="px-1 text-[10.5px] tracking-[0.08em] uppercase text-muted-foreground">
                Default routing
              </legend>
              <div className="grid grid-cols-3 gap-2">
                {(["stt", "llm", "tts"] as const).map((slot) => (
                  <Field key={slot} label={slot.toUpperCase()}>
                    <Select value={routing[slot] ?? ""} onValueChange={(v) => v != null && setRouting(slot, String(v))}>
                      <SelectTrigger className="w-full"><SelectValue placeholder="—" /></SelectTrigger>
                      <SelectContent>
                        {(meta?.providers[slot] ?? []).map((p) => (
                          <SelectItem key={p} value={p}>{p}</SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </Field>
                ))}
              </div>
            </fieldset>

            <fieldset className="flex flex-col gap-2 rounded-xl border p-3">
              <legend className="px-1 text-[10.5px] tracking-[0.08em] uppercase text-muted-foreground">
                Tools
              </legend>
              {draft.tools.map((tool, i) => (
                <ToolCard
                  key={i}
                  tool={tool}
                  params={paramDrafts[i] ?? ""}
                  onChange={(t) =>
                    setDraft({ ...draft, tools: draft.tools.map((x, j) => (j === i ? t : x)) })
                  }
                  onParams={(v) =>
                    setParamDrafts(paramDrafts.map((x, j) => (j === i ? v : x)))
                  }
                  onRemove={() => {
                    setDraft({ ...draft, tools: draft.tools.filter((_, j) => j !== i) });
                    setParamDrafts(paramDrafts.filter((_, j) => j !== i));
                  }}
                />
              ))}
              <Button
                variant="ghost"
                size="sm"
                className="press"
                onClick={() => {
                  setDraft({ ...draft, tools: [...draft.tools, { ...EMPTY_TOOL }] });
                  setParamDrafts([...paramDrafts, ""]);
                }}
              >
                + Add tool
              </Button>
            </fieldset>
          </CardContent>
        </ScrollArea>
      ) : (
        <Textarea
          value={yamlDraft}
          onChange={(e) => setYamlDraft(e.target.value)}
          spellCheck={false}
          className="min-h-[320px] flex-1 resize-none rounded-none border-0 font-mono text-xs leading-relaxed focus-visible:ring-0"
          aria-label="Agent YAML"
        />
      )}

      <CardFooter className="gap-3 border-t py-3! [.border-t]:pt-3">
        <Button className="press" onClick={save}>Save agent</Button>
        <span
          role="status"
          className={`text-xs ${msg ? (msg.ok ? "text-good" : "font-mono text-destructive") : "text-muted-foreground"}`}
        >
          {msg?.text ?? ""}
        </span>
      </CardFooter>
    </Card>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5 text-[11.5px] text-muted-foreground">
      {label}
      {children}
    </label>
  );
}

function ToolCard({
  tool, params, onChange, onParams, onRemove,
}: {
  tool: ToolDef;
  params: string;
  onChange: (t: ToolDef) => void;
  onParams: (v: string) => void;
  onRemove: () => void;
}) {
  /* concentric: inner inputs r6(+focus) + p-2.5 (10px) = outer rounded-2xl (16px) */
  return (
    <div className="flex flex-col gap-2 rounded-2xl bg-secondary p-2.5 shadow-[0_0_0_1px_var(--border)]">
      <div className="flex items-center gap-2">
        <Input
          className="flex-1 font-mono text-xs"
          placeholder="tool_name"
          value={tool.name}
          onChange={(e) => onChange({ ...tool, name: e.target.value })}
        />
        <Button
          variant="ghost" size="sm" aria-label="Remove tool"
          className="press text-muted-foreground active:text-destructive"
          onClick={onRemove}
        >
          ✕
        </Button>
      </div>
      <Input
        placeholder="What this tool does (shown to the model)"
        value={tool.description}
        onChange={(e) => onChange({ ...tool, description: e.target.value })}
      />
      <Textarea
        rows={3}
        spellCheck={false}
        className="font-mono text-[11px]"
        placeholder='{"type": "object", "properties": {…}}'
        value={params}
        onChange={(e) => onParams(e.target.value)}
      />
      <label className="flex items-center gap-2 text-xs text-muted-foreground">
        <Checkbox
          checked={tool.choreographed}
          onCheckedChange={(v) => onChange({ ...tool, choreographed: v === true })}
        />
        choreographed (no dead air)
        {tool.choreographed && <Badge variant="outline" className="text-[10px]">schema-enforced</Badge>}
      </label>
    </div>
  );
}
