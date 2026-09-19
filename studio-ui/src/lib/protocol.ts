/** Types mirroring docs/STUDIO_PROTOCOL.md and tring/events.py. */

export interface Meta {
  version: string;
  providers: Record<"stt" | "llm" | "tts" | "s2s", string[]>;
}

export interface ToolDef {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  choreographed: boolean;
}

export interface AgentSpec {
  name: string;
  greeting?: string;
  persona: string;
  language: { primary: string; lock: boolean };
  runtime: {
    mode: "cascade" | "s2s" | "hybrid";
    routing: { default: { stt?: string; llm?: string; tts?: string } };
  };
  tools: ToolDef[];
}

export interface SessionEvent {
  type: string;
  at?: number;
  session_id?: string;
  // union payload fields, narrowed at use sites
  text?: string;
  final?: boolean;
  tool_name?: string;
  call_id?: string;
  waiting_message?: string | null;
  spoken_mode?: string | null;
  ok?: boolean;
  latency_seconds?: number | null;
  result_summary?: string | null;
  component?: string;
  provider?: string;
  units?: number;
  unit_name?: string;
  amount?: number;
  currency?: string;
  estimated?: boolean;
  agent_name?: string;
  runtime_mode?: string;
  reason?: string;
  duration_seconds?: number;
  message?: string;
  unheard_text?: string;
}

export type ServerMsg =
  | { type: "ready"; session_id: string; capabilities: Record<string, unknown> }
  | { type: "event"; event: SessionEvent }
  | { type: "audio_progress"; bytes: number }
  | { type: "error"; message: string };

export type TranscriptItem =
  | { kind: "sys"; text: string; bad?: boolean }
  | { kind: "user"; text: string }
  | { kind: "bot"; text: string }
  | { kind: "interrupt" }
  | {
      kind: "tool";
      callId: string;
      name: string;
      waiting?: string | null;
      done?: { ok: boolean; latency?: number | null; summary?: string | null };
    };

export const EVENT_GROUP: Record<string, string> = {
  user_transcript: "speech",
  bot_utterance: "speech",
  bot_speech_played: "speech",
  interruption: "speech",
  tool_call_started: "tool",
  tool_call_completed: "tool",
  cost_recorded: "cost",
  session_started: "session",
  session_ended: "session",
  error: "error",
};

export const GROUPS = ["speech", "tool", "cost", "session", "error"] as const;

export function fmtUSD(v: number, currency?: string): string {
  const sym = currency && currency !== "USD" ? `${currency} ` : "$";
  return sym + (v >= 0.01 ? v.toFixed(4) : v.toFixed(5));
}

export function summarize(e: SessionEvent): string {
  switch (e.type) {
    case "user_transcript":
      return `"${e.text}"`;
    case "bot_utterance":
      return `"${(e.text ?? "").slice(0, 90)}${(e.text ?? "").length > 90 ? "…" : ""}"`;
    case "bot_speech_played":
      return `played: "${(e.text ?? "").slice(0, 60)}…"`;
    case "interruption":
      return `unheard: "${(e.unheard_text ?? "").slice(0, 60)}"`;
    case "tool_call_started":
      return `${e.tool_name} (${e.spoken_mode ?? "?"})`;
    case "tool_call_completed":
      return `${e.tool_name} ${e.ok ? "ok" : "FAILED"}${
        e.latency_seconds != null ? ` in ${e.latency_seconds.toFixed(2)}s` : ""
      }`;
    case "cost_recorded":
      return `${e.provider} ${e.units?.toFixed(1)} ${e.unit_name} → ${fmtUSD(
        e.amount ?? 0,
        e.currency,
      )}${e.estimated ? " (est)" : ""}`;
    case "session_started":
      return `${e.agent_name} on ${e.runtime_mode}`;
    case "session_ended":
      return `${e.reason}, ${e.duration_seconds?.toFixed(1)}s`;
    case "error":
      return e.message ?? "";
    default:
      return "";
  }
}
