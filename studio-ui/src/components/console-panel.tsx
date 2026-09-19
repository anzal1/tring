import { useState } from "react";
import { TranscriptPanel } from "@/components/transcript-panel";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { Mic, Session } from "@/hooks/use-session";

export function ConsolePanel({ session }: { session: Session }) {
  const { conn, items, level, mic, start, reset, send } = session;
  const [text, setText] = useState("");
  const live = conn === "live";

  function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!text.trim()) return;
    send(text);
    setText("");
  }

  return (
    <TranscriptPanel
      title="Test console"
      items={items}
      level={level}
      actions={
        <>
          <Button size="sm" variant="secondary" className="press" onClick={start}>
            Start
          </Button>
          <Button size="sm" variant="ghost" className="press" disabled={!live} onClick={reset}>
            Reset
          </Button>
        </>
      }
      empty={
        <div className="m-auto flex max-w-[300px] flex-col items-center gap-2 text-center text-xs text-muted-foreground">
          <span className="text-2xl opacity-35">🔔</span>
          <span>
            Save an agent, then <strong className="text-foreground">Start</strong> a session.
            Every event streams into the panels on the right.
          </span>
        </div>
      }
      footer={
        <>
          <MicBar mic={mic} live={live} />
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
        </>
      }
    />
  );
}

/** Push to talk, and an honest sentence about why it is or is not available.
 *
 * The row renders even when the microphone cannot work: a control that
 * disappears teaches nothing, and every reason this degrades (no secure
 * context, no AudioWorklet, a denied permission, an agent whose STT slot
 * would not bind) is something the developer can act on once they can read
 * it.
 */
function MicBar({ mic, live }: { mic: Mic; live: boolean }) {
  const blocked = mic.state === "unsupported" || mic.state === "denied";
  const disabled = blocked || !live;
  const armed = mic.state === "armed";

  const status = blocked
    ? (mic.reason ?? "microphone unavailable")
    : !live
      ? "start a session to speak to the agent"
      : mic.state === "requesting"
        ? "waiting for microphone permission…"
        : mic.state === "refused"
          ? (mic.reason ?? "the server stayed on typed turns")
          : armed
            ? `16 kHz mono uplink${mic.stt ? ` → ${mic.stt}` : ""}`
            : "arming restarts the session, because the STT slot is bound at start";

  /** Space means "click me" to a button and "scroll" to a document. Both
   *  defaults are wrong for a key being *held*, so both are cancelled and the
   *  hold is driven from keydown/keyup directly. */
  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key !== " " && e.key !== "Spacebar") return;
    e.preventDefault();
    if (!e.repeat) mic.press();
  }

  function onKeyUp(e: React.KeyboardEvent) {
    if (e.key !== " " && e.key !== "Spacebar") return;
    e.preventDefault();
    mic.release();
  }

  return (
    <div className="flex items-center gap-2 border-t px-3 py-2">
      <Tooltip>
        <TooltipTrigger render={<span className="inline-flex" />}>
          <Button
            size="sm"
            variant={mic.holding ? "secondary" : "ghost"}
            className="press"
            disabled={disabled}
            aria-pressed={mic.holding}
            onPointerDown={(e) => {
              // preventDefault stops the press from selecting text or
              // starting a drag, and takes the focus move with it, so focus
              // is put back by hand: holding space is only offered to a
              // focused button, and clicking one is how people focus it.
              e.preventDefault();
              e.currentTarget.focus();
              e.currentTarget.setPointerCapture(e.pointerId);
              mic.press();
            }}
            onPointerUp={() => mic.release()}
            onPointerCancel={() => mic.release()}
            onKeyDown={onKeyDown}
            onKeyUp={onKeyUp}
            onBlur={() => mic.release()}
          >
            <span
              className="mr-1.5 inline-block size-[7px] rounded-full transition-colors duration-150"
              style={{ background: mic.holding ? "var(--good)" : "var(--muted-foreground)" }}
            />
            {mic.holding ? "Listening…" : "Hold to talk"}
          </Button>
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-[280px] text-[11px]">
          {blocked
            ? (mic.reason ?? "microphone unavailable")
            : "Hold the button, or focus it and hold space. Your microphone is downsampled to 16 kHz mono and streamed only while you hold."}
        </TooltipContent>
      </Tooltip>

      <span className="min-w-0 flex-1 truncate text-[11px] text-muted-foreground">{status}</span>

      {armed && (
        <Button size="sm" variant="ghost" className="press text-muted-foreground" onClick={mic.disarm}>
          Release mic
        </Button>
      )}
    </div>
  );
}
