import { useCallback, useRef, useState } from "react";
import type { ServerMsg, SessionEvent, TranscriptItem } from "@/lib/protocol";

export type ConnState = "idle" | "connecting" | "live" | "ended" | "error";

export interface Session {
  conn: ConnState;
  sessionId: string | null;
  items: TranscriptItem[];
  events: SessionEvent[];
  costs: SessionEvent[];
  speaking: boolean;
  start: () => void;
  reset: () => void;
  send: (text: string) => void;
}

/** One live studio session over /ws. All UI panels derive from this state,
 *  mirroring how any tring event consumer subscribes to a CallSession. */
export function useSession(): Session {
  const ws = useRef<WebSocket | null>(null);
  const speakTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [conn, setConn] = useState<ConnState>("idle");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [items, setItems] = useState<TranscriptItem[]>([]);
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [costs, setCosts] = useState<SessionEvent[]>([]);
  const [speaking, setSpeaking] = useState(false);

  const clear = useCallback(() => {
    setItems([]);
    setEvents([]);
    setCosts([]);
  }, []);

  const onEvent = useCallback((e: SessionEvent) => {
    setEvents((prev) => [...prev, e]);
    switch (e.type) {
      case "user_transcript":
        if (e.final) setItems((p) => [...p, { kind: "user", text: e.text ?? "" }]);
        break;
      case "bot_utterance":
        setItems((p) => [...p, { kind: "bot", text: e.text ?? "" }]);
        break;
      case "interruption":
        setItems((p) => [...p, { kind: "interrupt" }]);
        break;
      case "tool_call_started":
        setItems((p) => [
          ...p,
          {
            kind: "tool",
            callId: e.call_id ?? "",
            name: e.tool_name ?? "",
            waiting: e.waiting_message,
          },
        ]);
        break;
      case "tool_call_completed":
        setItems((p) =>
          p.map((it) =>
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
        );
        break;
      case "cost_recorded":
        setCosts((p) => [...p, e]);
        break;
      case "session_ended":
        setItems((p) => [...p, { kind: "sys", text: `session ended (${e.reason})` }]);
        setConn("ended");
        break;
      case "error":
        setItems((p) => [...p, { kind: "sys", text: e.message ?? "error", bad: true }]);
        break;
    }
  }, []);

  const handle = useCallback(
    (msg: ServerMsg) => {
      if (msg.type === "ready") {
        clear();
        setSessionId(msg.session_id);
        setConn("live");
        setItems([{ kind: "sys", text: `session ${msg.session_id.slice(0, 8)} started` }]);
        return;
      }
      if (msg.type === "audio_progress") {
        setSpeaking(true);
        if (speakTimer.current) clearTimeout(speakTimer.current);
        speakTimer.current = setTimeout(() => setSpeaking(false), 400);
        return;
      }
      if (msg.type === "error") {
        setItems((p) => [...p, { kind: "sys", text: msg.message, bad: true }]);
        return;
      }
      if (msg.type === "event") onEvent(msg.event);
    },
    [clear, onEvent],
  );

  const start = useCallback(() => {
    ws.current?.close();
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/ws`);
    ws.current = sock;
    setConn("connecting");
    sock.onopen = () => sock.send(JSON.stringify({ type: "start" }));
    sock.onclose = () => {
      setConn((c) => (c === "ended" ? c : "idle"));
      ws.current = null;
    };
    sock.onerror = () => setConn("error");
    sock.onmessage = (m) => handle(JSON.parse(m.data) as ServerMsg);
  }, [handle]);

  const reset = useCallback(() => {
    ws.current?.send(JSON.stringify({ type: "reset" }));
    clear();
  }, [clear]);

  const send = useCallback((text: string) => {
    const t = text.trim();
    if (!t || !ws.current) return;
    ws.current.send(JSON.stringify({ type: "user_text", text: t }));
  }, []);

  return { conn, sessionId, items, events, costs, speaking, start, reset, send };
}
