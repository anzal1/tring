import { useCallback, useEffect, useRef, useState } from "react";
import { MicCapture, micUnsupportedReason, Playback } from "@/lib/audio";
import type { ServerMsg, SessionEvent, TranscriptItem } from "@/lib/protocol";
import { applyEvent, EMPTY_DERIVED, note, type Derived } from "@/lib/transcript";

export type ConnState = "idle" | "connecting" | "live" | "ended" | "error";

/** Where the microphone is in its life, and why.
 *
 * `refused` is the honest one: the browser handed over a microphone and the
 * *server* could not bind a speech recognizer, so the session stayed on typed
 * turns. That is a different problem from a denied permission, and it gets a
 * different word.
 */
export type MicState = "unsupported" | "idle" | "requesting" | "armed" | "denied" | "refused";

export interface Mic {
  state: MicState;
  /** One sentence a developer can act on; null while nothing is wrong. */
  reason: string | null;
  /** True while the button or the spacebar is held down. */
  holding: boolean;
  /** Which STT the server actually bound, once it has said. */
  stt: string | null;
  press: () => void;
  release: () => void;
  /** Hand the device back and return the session to typed turns. */
  disarm: () => void;
}

export interface Session {
  conn: ConnState;
  sessionId: string | null;
  items: TranscriptItem[];
  events: SessionEvent[];
  costs: SessionEvent[];
  /** Speech-meter deflection, 0–1. In microphone mode this is measured off
   *  the audio actually reaching the speakers; in text mode, where no PCM
   *  crosses the socket, it is the byte counter's on/off. */
  level: number;
  mic: Mic;
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
  const [derived, setDerived] = useState<Derived>(EMPTY_DERIVED);
  const [events, setEvents] = useState<SessionEvent[]>([]);
  const [speaking, setSpeaking] = useState(false);

  // ----------------------------------------------------------------- audio
  const capture = useRef<MicCapture | null>(null);
  const playback = useRef<Playback | null>(null);
  const holding = useRef(false);
  const [playLevel, setPlayLevel] = useState(0);
  const [micState, setMicState] = useState<MicState>(() =>
    micUnsupportedReason() === null ? "idle" : "unsupported",
  );
  const [micReason, setMicReason] = useState<string | null>(() => micUnsupportedReason());
  const [micHolding, setMicHolding] = useState(false);
  const [sttName, setSttName] = useState<string | null>(null);

  const clear = useCallback(() => {
    setDerived(EMPTY_DERIVED);
    setEvents([]);
  }, []);

  const speakers = useCallback((): Playback => {
    playback.current ??= new Playback(setPlayLevel);
    return playback.current;
  }, []);

  /** Stop capturing and forget the device. `tell` sends the server back to
   *  typed turns; it is skipped when the server is the one that said no. */
  const release = useCallback((reason: string | null, next: MicState, tell: boolean) => {
    capture.current?.stop();
    capture.current = null;
    holding.current = false;
    setMicHolding(false);
    setMicState(next);
    setMicReason(reason);
    if (tell && ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ type: "mode", audio: false }));
    }
  }, []);

  const onEvent = useCallback((e: SessionEvent) => {
    setEvents((prev) => [...prev, e]);
    setDerived((prev) => applyEvent(prev, e));
    if (e.type === "session_ended") setConn("ended");
  }, []);

  const handle = useCallback(
    (msg: ServerMsg) => {
      switch (msg.type) {
        case "ready":
          clear();
          setSessionId(msg.session_id);
          setConn("live");
          setDerived(note(EMPTY_DERIVED, `session ${msg.session_id.slice(0, 8)} started`));
          setSttName(msg.mode?.stt ?? null);
          playback.current?.flush(); // queued audio belonged to the old session
          return;
        case "audio_progress":
          setSpeaking(true);
          if (speakTimer.current) clearTimeout(speakTimer.current);
          speakTimer.current = setTimeout(() => setSpeaking(false), 400);
          return;
        case "mode":
          setSttName(msg.stt);
          if (!msg.audio && capture.current) {
            // The device is ours and useless: the server is transcribing
            // nothing from it. Hand it back rather than leave the browser's
            // recording indicator lit over a dead uplink.
            release(
              "the server could not run the microphone for this agent, so the session " +
                `stayed on typed turns (stt slot: ${msg.stt ?? "none"})`,
              "refused",
              false,
            );
          }
          return;
        case "audio_format":
          speakers().setFormat(msg.sample_rate, msg.channels);
          return;
        case "error":
          setDerived((prev) => note(prev, msg.message, true));
          return;
        case "event":
          onEvent(msg.event);
      }
    },
    [clear, onEvent, release, speakers],
  );

  const start = useCallback(() => {
    ws.current?.close();
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const sock = new WebSocket(`${proto}://${location.host}/ws`);
    sock.binaryType = "arraybuffer";
    ws.current = sock;
    setConn("connecting");
    sock.onopen = () => {
      sock.send(JSON.stringify({ type: "start" }));
      // A socket replaced while the mic was armed starts its new session in
      // the mode the user still has a finger on.
      if (capture.current) sock.send(JSON.stringify({ type: "mode", audio: true }));
    };
    sock.onclose = () => {
      setConn((c) => (c === "ended" ? c : "idle"));
      ws.current = null;
      if (capture.current) {
        release("the session ended, so the microphone was released", "idle", false);
      }
    };
    sock.onerror = () => setConn("error");
    sock.onmessage = (m) => {
      if (typeof m.data === "string") handle(JSON.parse(m.data) as ServerMsg);
      else if (m.data instanceof ArrayBuffer) speakers().push(m.data);
    };
  }, [handle, release, speakers]);

  const reset = useCallback(() => {
    ws.current?.send(JSON.stringify({ type: "reset" }));
    playback.current?.flush();
    clear();
  }, [clear]);

  const send = useCallback((text: string) => {
    const t = text.trim();
    if (!t || !ws.current) return;
    ws.current.send(JSON.stringify({ type: "user_text", text: t }));
  }, []);

  /** Open the device, then tell the server to switch the STT slot.
   *
   * In that order, and once per session: asking the server first would
   * restart the session for a permission the user may be about to deny.
   * Switching the slot restarts the session by design (the slot is bound when
   * the runtime starts), which is why arming is a deliberate first press
   * rather than something the console does on load.
   */
  const arm = useCallback(async () => {
    const mic = new MicCapture();
    try {
      await mic.start((pcm) => {
        if (!holding.current) return; // push to talk: released means silent
        const sock = ws.current;
        if (sock?.readyState === WebSocket.OPEN) sock.send(pcm);
      });
    } catch (err) {
      mic.stop();
      const denied = err instanceof DOMException && err.name === "NotAllowedError";
      holding.current = false;
      setMicHolding(false);
      setMicState(denied ? "denied" : "idle");
      setMicReason(
        denied
          ? "microphone permission was denied; allow it in the browser's site settings"
          : `could not open a microphone: ${err instanceof Error ? err.message : String(err)}`,
      );
      return;
    }
    capture.current = mic;
    setMicState("armed");
    setMicReason(null);
    ws.current?.send(JSON.stringify({ type: "mode", audio: true }));
  }, []);

  const press = useCallback(() => {
    if (holding.current || micState === "unsupported" || micState === "denied") return;
    holding.current = true;
    setMicHolding(true);
    playback.current?.flush(); // whatever the agent was saying, it is talked over
    if (capture.current) return;
    setMicState("requesting");
    void arm();
  }, [arm, micState]);

  const releaseHold = useCallback(() => {
    if (!holding.current) return;
    holding.current = false;
    setMicHolding(false);
  }, []);

  const disarm = useCallback(() => release(null, "idle", true), [release]);

  useEffect(
    () => () => {
      capture.current?.stop();
      playback.current?.close();
      ws.current?.close();
      if (speakTimer.current) clearTimeout(speakTimer.current);
    },
    [],
  );

  const armed = micState === "armed" || micState === "requesting";
  return {
    conn,
    sessionId,
    items: derived.items,
    events,
    costs: derived.costs,
    level: armed ? playLevel : speaking ? 1 : 0,
    mic: {
      state: micState,
      reason: micReason,
      holding: micHolding,
      stt: sttName,
      press,
      release: releaseHold,
      disarm,
    },
    start,
    reset,
    send,
  };
}
