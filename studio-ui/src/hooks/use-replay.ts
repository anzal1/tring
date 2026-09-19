import { useCallback, useEffect, useMemo, useState } from "react";
import type { SessionEvent } from "@/lib/protocol";
import { derive, speechAt, type Derived } from "@/lib/transcript";

/** A clock over a stored event list.
 *
 * Replay is pure state reconstruction: there is no socket, no runtime and no
 * second copy of the render code. The clock moves, the cursor follows it, and
 * the same fold the live session uses produces the same panels. Everything
 * here is therefore about *time*, because time is the only thing a stored
 * session is missing.
 *
 * `at` is seconds since the session started, which is what makes this work at
 * all: the deltas are the session's own pacing, so 1x is really 1x, and 4x is
 * the same call with the silences taken out.
 */
export interface Replay {
  /** Replay clock, in session seconds. */
  t: number;
  duration: number;
  playing: boolean;
  rate: number;
  /** Events up to `t`, and what they fold into. */
  visible: SessionEvent[];
  derived: Derived;
  /** Meter deflection: the windows where the agent was speaking. */
  level: number;
  toggle: () => void;
  seek: (t: number) => void;
  setRate: (rate: number) => void;
}

export const REPLAY_RATES = [1, 4] as const;

export function useReplay(events: readonly SessionEvent[]): Replay {
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState<number>(1);

  const duration = useMemo(
    () => events.reduce((max, e) => Math.max(max, e.at ?? 0), 0),
    [events],
  );

  // A different session is a different clock: never inherit the last one's
  // position, and never keep playing into a list that changed underneath.
  // Adjusted during render rather than in an effect, so the first paint of a
  // newly opened session is already at zero instead of flashing the old one.
  const [source, setSource] = useState(events);
  if (source !== events) {
    setSource(events);
    setT(0);
    setPlaying(false);
  }

  // "Playing" is the intent; running is whether the clock is actually moving.
  // Deriving the second from the first is what lets the end of a replay stop
  // the loop without a state update chasing another state update.
  const running = playing && t < duration;

  useEffect(() => {
    if (!running) return;
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const step = ((now - last) / 1000) * rate;
      last = now;
      setT((prev) => Math.min(duration, prev + step));
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [running, rate, duration]);

  const seek = useCallback((next: number) => setT(next), []);

  const toggle = useCallback(() => {
    if (running) {
      setPlaying(false);
      return;
    }
    // Pressing play at the end starts the session again rather than doing
    // nothing at the right-hand end of the bar.
    if (t >= duration) setT(0);
    setPlaying(true);
  }, [duration, running, t]);

  // The cursor is an integer, and the slice hangs off the integer rather than
  // off `t`: the clock ticks sixty times a second, and the panels should only
  // see new arrays on the frames where an event actually arrived.
  const cursor = useMemo(() => {
    let n = 0;
    while (n < events.length && (events[n].at ?? 0) <= t) n++;
    return n;
  }, [events, t]);
  const visible = useMemo(() => events.slice(0, cursor), [events, cursor]);
  const derived = useMemo(() => derive(visible), [visible]);

  return {
    t,
    duration,
    playing: running,
    rate,
    visible,
    derived,
    level: speechAt(events, t),
    toggle,
    seek,
    setRate,
  };
}
