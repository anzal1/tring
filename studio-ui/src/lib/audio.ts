/** Microphone capture and bot-audio playback, in the browser.
 *
 * The wire contract is in docs/STUDIO_PROTOCOL.md and it is deliberately
 * small: what goes up is 16 kHz mono 16-bit little-endian PCM in binary
 * frames, and what comes down is PCM in the format the server announced with
 * `audio_format` just before the first frame. Everything in this file exists
 * to meet that contract honestly.
 *
 * Two decisions are worth knowing.
 *
 * **Resampling is the browser's job when it can be.** The capture graph asks
 * for an `AudioContext` at 16 kHz, which is the rate the wire wants, and lets
 * the browser's own resampler do the work on the way in. When a browser
 * refuses the rate (it is allowed to), the worklet falls back to a box filter
 * that averages every input sample inside an output sample's window. That is
 * not a great anti-aliasing filter, but it is a real one: plain decimation,
 * the usual shortcut, folds everything above 8 kHz back into the speech band
 * and makes an STT provider look worse than it is.
 *
 * **Playback is scheduled, not fired.** Bot audio arrives in whatever chunks
 * the TTS provider produced. Calling `start()` on each one as it lands leaves
 * a click between every chunk, so each buffer is scheduled at the end of the
 * last one, with a small lead, and the queue resynchronises if the network
 * falls far enough behind that the schedule is already in the past.
 */

/** The only rate the studio's binary uplink accepts. */
export const MIC_SAMPLE_RATE = 16000;

/** 100 ms of 16 kHz mono audio: the frame size the protocol doc describes. */
const FRAME_SAMPLES = 1600;

/** Scheduling lead for playback: enough to survive a slow frame, short
 *  enough that a barge-in still feels immediate. */
const PLAYBACK_LEAD = 0.06;

/** True on every platform the studio runs on, but assumed nowhere: a wrong
 *  guess here turns speech into noise, which is the hardest bug to read. */
const LITTLE_ENDIAN = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;

/** The capture worklet, as source text.
 *
 * It ships inline rather than as a separate file because the studio's built
 * assets are served by a Python server out of a package directory, and an
 * extra worklet URL is one more thing that can 404 in a wheel. A Blob URL
 * cannot.
 */
const WORKLET_SOURCE = `
class MicDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = (options && options.processorOptions) || {};
    this.ratio = (opts.inputRate || sampleRate) / ${MIC_SAMPLE_RATE};
    this.frame = new Int16Array(${FRAME_SAMPLES});
    this.filled = 0;
    this.carry = 0;
    this.sum = 0;
    this.count = 0;
    this.open = true;
    this.port.onmessage = (event) => {
      if (event.data === "stop") this.open = false;
    };
  }

  process(inputs) {
    if (!this.open) return false;
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      this.sum += channel[i];
      this.count++;
      this.carry++;
      if (this.carry < this.ratio) continue;
      this.carry -= this.ratio;
      const mean = this.count > 0 ? this.sum / this.count : 0;
      this.sum = 0;
      this.count = 0;
      const clamped = Math.max(-1, Math.min(1, mean));
      this.frame[this.filled++] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
      if (this.filled === this.frame.length) {
        const full = this.frame;
        this.frame = new Int16Array(${FRAME_SAMPLES});
        this.filled = 0;
        this.port.postMessage(full.buffer, [full.buffer]);
      }
    }
    return true;
  }
}
registerProcessor("mic-downsampler", MicDownsampler);
`;

/** Why the microphone is not available, in words a developer can act on. */
export function micUnsupportedReason(): string | null {
  if (typeof window === "undefined") return "no browser environment";
  if (!window.isSecureContext) {
    return "microphone capture needs a secure context: use https, or localhost";
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    return "this browser has no getUserMedia";
  }
  if (typeof AudioWorkletNode !== "function") {
    return "this browser has no AudioWorklet";
  }
  return null;
}

/** Turns the default input device into 100 ms frames of wire-format PCM. */
export class MicCapture {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private moduleUrl: string | null = null;

  /** Open the device and start emitting frames. Rejects with the DOMException
   *  `getUserMedia` produced, so callers can tell "denied" from "no device". */
  async start(onFrame: (pcm: ArrayBuffer) => void): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const ctx = new AudioContext({ sampleRate: MIC_SAMPLE_RATE });
    this.ctx = ctx;
    if (ctx.state === "suspended") await ctx.resume();

    const url = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "text/javascript" }));
    this.moduleUrl = url;
    await ctx.audioWorklet.addModule(url);

    const node = new AudioWorkletNode(ctx, "mic-downsampler", {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      processorOptions: { inputRate: ctx.sampleRate },
    });
    node.port.onmessage = (event: MessageEvent<ArrayBuffer>) => onFrame(event.data);
    this.node = node;

    // A worklet with nothing downstream is not guaranteed to be pulled, so the
    // graph ends at the destination through a muted gain: the processor runs,
    // and the developer does not hear themselves.
    const silence = ctx.createGain();
    silence.gain.value = 0;
    ctx.createMediaStreamSource(this.stream).connect(node);
    node.connect(silence).connect(ctx.destination);
  }

  /** Release the device. The browser's recording indicator goes out here, so
   *  this is called on every exit path, not only the tidy one. */
  stop(): void {
    this.node?.port.postMessage("stop");
    this.node?.disconnect();
    this.node = null;
    for (const track of this.stream?.getTracks() ?? []) track.stop();
    this.stream = null;
    void this.ctx?.close().catch(() => undefined);
    this.ctx = null;
    if (this.moduleUrl) URL.revokeObjectURL(this.moduleUrl);
    this.moduleUrl = null;
  }
}

/** Plays the server's binary bot audio and reports how loud it actually is.
 *
 * The level is measured off the node that feeds the speakers, not guessed
 * from "a frame arrived", which is what lets the console's speech meter be a
 * statement about sound rather than about traffic.
 */
export class Playback {
  private ctx: AudioContext | null = null;
  private analyser: AnalyserNode | null = null;
  private probe = new Float32Array(0);
  private sources = new Set<AudioBufferSourceNode>();
  private nextAt = 0;
  private rate = MIC_SAMPLE_RATE;
  private channels = 1;
  private raf: number | null = null;
  private readonly onLevel: (level: number) => void;

  constructor(onLevel: (level: number) => void) {
    this.onLevel = onLevel;
  }

  /** What the server's `audio_format` message said. */
  setFormat(sampleRate: number, channels: number): void {
    if (sampleRate > 0) this.rate = sampleRate;
    if (channels > 0) this.channels = channels;
  }

  /** Queue one binary frame. */
  push(pcm: ArrayBuffer): void {
    const ctx = this.ensure();
    const frames = Math.floor(pcm.byteLength / 2 / this.channels);
    if (frames === 0) return;

    const buffer = ctx.createBuffer(this.channels, frames, this.rate);
    const view = new DataView(pcm);
    for (let ch = 0; ch < this.channels; ch++) {
      const out = buffer.getChannelData(ch);
      for (let i = 0; i < frames; i++) {
        out[i] = view.getInt16((i * this.channels + ch) * 2, LITTLE_ENDIAN) / 0x8000;
      }
    }

    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(this.analyser ?? ctx.destination);
    // Behind the clock means the queue underran: restart from now rather than
    // scheduling into the past, where the browser plays everything at once.
    const start = Math.max(ctx.currentTime + PLAYBACK_LEAD, this.nextAt);
    source.start(start);
    this.nextAt = start + buffer.duration;
    this.sources.add(source);
    source.onended = () => this.sources.delete(source);
    this.watch();
  }

  /** Drop whatever is queued: a reset, or the caller barging in. */
  flush(): void {
    for (const source of this.sources) {
      try {
        source.stop();
      } catch {
        /* already finished */
      }
    }
    this.sources.clear();
    this.nextAt = 0;
    this.onLevel(0);
  }

  close(): void {
    this.flush();
    if (this.raf !== null) cancelAnimationFrame(this.raf);
    this.raf = null;
    void this.ctx?.close().catch(() => undefined);
    this.ctx = null;
    this.analyser = null;
  }

  private ensure(): AudioContext {
    if (this.ctx) {
      // An autoplay policy can suspend a context between frames; currentTime
      // stops advancing there, and every buffer would be scheduled into a
      // clock that never arrives. Arming the mic is a user gesture, so by the
      // time bot audio exists there is activation to resume with.
      if (this.ctx.state === "suspended") void this.ctx.resume().catch(() => undefined);
      return this.ctx;
    }
    const ctx = new AudioContext();
    if (ctx.state === "suspended") void ctx.resume().catch(() => undefined);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 1024;
    analyser.connect(ctx.destination);
    this.probe = new Float32Array(analyser.fftSize);
    this.ctx = ctx;
    this.analyser = analyser;
    this.nextAt = 0;
    return ctx;
  }

  /** Report the real amplitude while audio is queued, then fall back to zero
   *  once the schedule has drained. Only runs while there is sound. */
  private watch(): void {
    if (this.raf !== null) return;
    const tick = () => {
      const ctx = this.ctx;
      const analyser = this.analyser;
      if (!ctx || !analyser) {
        this.raf = null;
        return;
      }
      analyser.getFloatTimeDomainData(this.probe);
      let sum = 0;
      for (const sample of this.probe) sum += sample * sample;
      const rms = Math.sqrt(sum / this.probe.length);
      // Speech RMS sits near 0.1: x4 puts an ordinary sentence mid-scale
      // instead of pinning the meter at a hairline.
      this.onLevel(Math.min(1, rms * 4));
      if (ctx.currentTime > this.nextAt + 0.15) {
        this.raf = null;
        this.onLevel(0);
        return;
      }
      this.raf = requestAnimationFrame(tick);
    };
    this.raf = requestAnimationFrame(tick);
  }
}
