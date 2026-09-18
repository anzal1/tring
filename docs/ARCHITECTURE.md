# Alaap Architecture

Alaap (आलाप, "the opening of a conversation") is an open-source, production-grade voice agent stack.
One agent definition, any runtime, any provider — or no paid provider at all —
and you always know what a call costs.

## Design principles

1. **One contract, three runtimes.** An `AgentSpec` runs unchanged on a
   cascaded pipeline (STT → LLM → TTS), a speech-to-speech model, or a hybrid
   of the two. Switching runtime is a config change, never a rewrite.
2. **Local-first.** The default pipeline runs entirely on open models
   (faster-whisper STT, Ollama LLM, Kokoro/Piper TTS, Silero VAD) at $0 per
   minute. Cloud providers are opt-in upgrades, not requirements.
3. **Latency is a correctness property.** In voice, two seconds of silence is
   a conversational error, not a performance metric. The primitives layer
   exists to make dead air structurally impossible, not to patch it.
4. **Honest cost accounting.** Every metered unit is recorded with an
   `estimated` flag. Exact vendor-reported usage is preferred; estimates are
   never silently substituted for measurements.
5. **Capabilities are declared, never assumed.** Each runtime leaks its own
   assumptions (S2S has no transcripts until call end; cascade has them
   mid-turn). Consumers query `RuntimeCapabilities` instead of guessing.

## Package layout

```
src/alaap/
  agent.py          AgentSpec — the single agent contract (pydantic, YAML-loadable)
  events.py         Unified session event model (all runtimes emit these)
  session.py        CallSession — one live conversation, event bus, lifecycle
  runtimes/
    base.py         RuntimeAdapter ABC + RuntimeCapabilities
    cascade.py      STT → LLM → TTS pipeline runtime
    s2s.py          Speech-to-speech runtime (e.g. Ultravox, Gemini Live)
    hybrid.py       S2S front + cascade fallback/tooling
  providers/
    base.py         STTProvider / LLMProvider / TTSProvider ABCs (streaming)
    registry.py     Provider registry + per-agent, per-language routing
    local/          faster-whisper, ollama, kokoro, piper, silero (lazy imports)
    cloud/          deepgram, elevenlabs, cartesia, openai, anthropic, ... (lazy)
  primitives/
    choreography.py Tool-call choreography: schema-required waiting_message /
                    spoken_mode / post_tool_response so the model cannot emit
                    a tool call without planning the silence around it
    speak_parser.py Character-level streaming parser for {"speak": ..., "tool_call": ...}
                    that flushes speak tokens to TTS while the tool call is
                    still generating (first-token-to-audio ~= 0)
    interruption.py Tracks what audio was actually played vs generated; on a
                    genuine barge-in, annotates context so the model never
                    references words the caller never heard
    language_lock.py Re-asserts the language directive every turn WITHOUT
                    breaking prompt-cache byte-stability (directive folds into
                    a single trailing message; the prior array stays identical)
  cost/
    meter.py        CostMeter — records CostEvent per component with estimated flag
    rates.py        Versioned vendor rate cards + local-compute cost model
    report.py       Per-call and per-outcome roll-ups (cost per call, per
                    connected minute, per conversation, per outcome)
  transports/
    console.py      Text/dev-loop transport for local iteration without audio
    websocket.py    Raw audio-over-websocket transport (16kHz mono linear16)
```

## The contract chain

```
AgentSpec ──▶ RuntimeAdapter.start(session)
                 │ emits
                 ▼
           SessionEvent stream (events.py)
                 │ consumed by
                 ▼
   CostMeter · analytics · transports · tests
```

Everything integrates through `SessionEvent`. A module that needs runtime
behavior subscribes to the session's event stream; it never imports a concrete
runtime.

## Rules for contributors (and swarm agents)

- Python 3.11+, pydantic v2, full type hints, `ruff` + `mypy --strict` clean.
- Core (`agent.py`, `events.py`, `session.py`, primitives, cost) depends only
  on stdlib + pydantic. Heavy deps (audio, ML, vendor SDKs) live behind lazy
  imports in `providers/` and `transports/` and are optional extras.
- Every module ships unit tests. Provider wrappers are tested with fakes;
  no test may require a network call, an API key, or a GPU.
- No references to any prior proprietary system, employer, or client. This is
  a clean-room implementation of publicly describable patterns.
