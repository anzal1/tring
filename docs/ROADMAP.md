# Tring Roadmap

## v0.1 (current)

- Core agent contract: `AgentSpec` (pydantic, YAML-loadable, serializable).
- Event model: unified `SessionEvent` vocabulary for all runtimes.
- Three runtime adapters: cascade (STT -> LLM -> TTS), speech-to-speech, hybrid.
- Four latency primitives: choreography (tool-call waiting messages), speak-while-thinking parser, interruption tracking, language locking.
- Cost meter with denominator reporting (per-call, per-minute, per-outcome).
- Local cascade stack: faster-whisper STT, Ollama LLM, Kokoro TTS (no paid services required).
- Console transport for text-based dev loops.
- WebSocket transport for raw audio (16kHz mono linear16).
- Initial cloud providers: Deepgram (STT), ElevenLabs (TTS), OpenAI-compatible (LLM).

## v0.2

- Expanded provider matrix: 2 local and 4 cloud STT engines, 1 local and 3 cloud LLM providers, 2 local and 4 cloud TTS engines, 2 cloud S2S models.
- Streaming STT: true incremental transcription (character by character) instead of buffered phrases.
- Silero VAD: local voice activity detection for natural turn-taking without vendor lock-in.
- S2S providers: Ultralow latency with tool support (OpenAI Realtime, Ultravox).
- Hybrid maturity: seamless S2S-to-cascade fallback for tool calling and advanced reasoning.
- SIP ingress: FreeSWITCH gateway pattern for telephone call integration.
- Barge-in over real audio: caller can interrupt the bot without waiting for TTS to finish, with latency budgets that make it structurally possible.

## v0.3

- Dashboard: session monitoring, cost drill-down, provider comparison.
- Campaign and outbound mode: batch call scheduling, outcome tracking, cost per campaign.
- PyPI package and documentation site: easy install, reference docs, tutorial walkthrough.

---

Each item is scoped to one semantic unit. Dates are intentionally absent; we ship when ready. Honest over optimistic.
