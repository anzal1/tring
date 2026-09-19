/** Folding an event stream into what the panels render.
 *
 * There are two ways to watch a tring session: live over the websocket, and
 * after the fact over `GET /api/sessions/{id}`. Both are the *same* list of
 * `SessionEvent`s, so both must produce the same transcript, the same tool
 * timeline and the same cost table, or a replay would quietly be a different
 * product from the thing it replays.
 *
 * So the derivation lives here, as a pure fold, and both callers use it:
 * `use-session` applies one event at a time as it arrives, `use-replay`
 * re-folds the first N events whenever the scrubber moves. Same function,
 * same result, no second renderer to keep in step.
 */

import type { SessionEvent, TranscriptItem } from "@/lib/protocol";

/** Everything the panels need that is not simply "the events". */
export interface Derived {
  items: TranscriptItem[];
  costs: SessionEvent[];
}

export const EMPTY_DERIVED: Derived = { items: [], costs: [] };

/** A line the transport wrote, not the agent: "session started", a socket
 *  error. Kept in the transcript so the order of what happened survives. */
export function note(prev: Derived, text: string, bad = false): Derived {
  return { ...prev, items: [...prev.items, { kind: "sys", text, bad }] };
}

/** Fold one event in. Unknown event types change nothing here on purpose:
 *  the events panel still lists them verbatim, so a new event type from a
 *  newer server shows up as itself rather than as a rendering crash. */
export function applyEvent(prev: Derived, e: SessionEvent): Derived {
  switch (e.type) {
    case "user_transcript":
      // Partials are for latency, not for the record: only finals land.
      return e.final ? push(prev, { kind: "user", text: e.text ?? "" }) : prev;
    case "bot_utterance":
      return push(prev, { kind: "bot", text: e.text ?? "" });
    case "interruption":
      return push(prev, { kind: "interrupt" });
    case "tool_call_started":
      return push(prev, {
        kind: "tool",
        callId: e.call_id ?? "",
        name: e.tool_name ?? "",
        waiting: e.waiting_message,
      });
    case "tool_call_completed":
      return {
        ...prev,
        items: prev.items.map((it) =>
          it.kind === "tool" && it.callId === e.call_id && !it.done
            ? {
                ...it,
                done: {
                  ok: e.ok ?? false,
                  latency: e.latency_seconds,
                  summary: e.result_summary,
                },
              }
            : it,
        ),
      };
    case "cost_recorded":
      return { ...prev, costs: [...prev.costs, e] };
    case "session_ended":
      return note(prev, `session ended (${e.reason})`);
    case "error":
      return note(prev, e.message ?? "error", true);
    default:
      return prev;
  }
}

/** Fold a whole list. Used by replay, where the cursor can move backwards and
 *  re-deriving from zero is both simpler and cheaper than undoing events. */
export function derive(events: readonly SessionEvent[]): Derived {
  return events.reduce(applyEvent, EMPTY_DERIVED);
}

/** How loud the meter should read at replay time `t`.
 *
 * A stored session carries no audio, so this is honest about what it is: the
 * windows where the agent *was* speaking, drawn at full deflection, rather
 * than a waveform invented from nothing. Live sessions in microphone mode
 * drive the same meter from real playback amplitude instead.
 */
export function speechAt(events: readonly SessionEvent[], t: number): number {
  const WINDOW = 0.6;
  for (let i = events.length - 1; i >= 0; i--) {
    const e = events[i];
    const at = e.at ?? 0;
    if (at > t) continue;
    if (t - at > WINDOW) return 0;
    return e.type === "bot_utterance" || e.type === "bot_speech_played" ? 1 : 0;
  }
  return 0;
}

function push(prev: Derived, item: TranscriptItem): Derived {
  return { ...prev, items: [...prev.items, item] };
}
