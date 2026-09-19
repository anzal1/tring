# Tring Roadmap

## v0.1 (shipped)

- Core agent contract: `AgentSpec` (pydantic, YAML-loadable, serializable).
- Event model: unified `SessionEvent` vocabulary for all runtimes.
- Three runtime adapters: cascade (STT -> LLM -> TTS), speech-to-speech, hybrid.
- Four latency primitives: choreography (tool-call waiting messages), speak-while-thinking parser, interruption tracking, language locking.
- Cost meter with denominator reporting (per-call, per-minute, per-outcome).
- Local cascade stack: faster-whisper STT, Ollama LLM, Kokoro TTS (no paid services required).
- Console transport for text-based dev loops.
- WebSocket transport for raw audio (16kHz mono linear16).
- Initial cloud providers: Deepgram (STT), ElevenLabs (TTS), OpenAI-compatible (LLM).

## v0.2 (shipped)

- Expanded provider matrix: 2 local and 4 cloud STT engines, 1 local and 3 cloud LLM providers, 2 local and 4 cloud TTS engines, 2 cloud S2S models.
- S2S providers: Ultralow latency with tool support (OpenAI Realtime, Ultravox).
- Barge-in over real audio: caller can interrupt the bot without waiting for TTS to finish, with latency budgets that make it structurally possible.

## v0.3 (shipped)

- Streaming STT: true incremental transcription (character by character) instead of buffered phrases via `faster_whisper_streaming`.
- Silero VAD: local voice activity detection for natural turn-taking without vendor lock-in.
- Tring Studio: single-page app for interactive agent development and testing, no audio hardware required.
- Twilio transport: MediaStreams ingress with exact playout marks for perfect interruption reconciliation.

## v0.4 (shipped)

- Studio flow builder: canvas of say/ask/branch/tool/handoff/end nodes, compiled server-side into transition-explicit agent instructions; lossless graph round-trip via spec metadata.
- Studio session replay: every call stored as its event stream, scrubbable through the live panels.
- Studio microphone mode: browser push-to-talk over websocket binary frames, honest degraded states.
- Eval harness: YAML conversation tests (expected tools, arguments, replies, latency) plus LLM judges; `tring eval` with CI exit codes.
- Outbound campaigns: concurrency-bounded dialing, retries, quiet hours, Twilio answering-machine detection, live cost-per-outcome ladder.
- Observability: OpenTelemetry span export, per-turn latency waterfalls, prompt-cache regression alarm.
- Telephony breadth: FreeSWITCH dialplan generator for generic SIP, DTMF events, transfer-to-human tool.
- S2S maturity: Gemini Live adapter; hybrid v2 downgrades mid-call from S2S to cascade with context carried over.
- Knowledge slot: local BM25, Chroma, and Qdrant retrieval with speculative prefetch from partial transcripts.
- `tring` CLI, Dockerfile and docker-compose onboarding, deployment guide.

## v0.5+

- Local speech-to-speech when open models mature (the day the zero-dollar pipeline covers every runtime).
- Studio authentication and a hosted dashboard tier.
- Streaming partials surfaced in Studio; latency waterfalls in the UI.
- Docs site and examples gallery.

---

Each item is scoped to one semantic unit. Dates are intentionally absent; we ship when ready. Honest over optimistic.
