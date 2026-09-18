# Trunkline

**Open-source voice agent stack. One agent contract, any runtime, honest costs.**

In telephony, a trunk line is the backbone circuit that carries every call. Trunkline lets you define a voice agent once and run it on any of
three architectures — a cascaded pipeline (STT → LLM → TTS), a speech-to-speech
model, or a hybrid of the two — without rewriting anything. It ships with a
fully local, $0-per-minute pipeline as a first-class citizen, and it tells you
exactly what every call costs.

> Built from lessons learned running voice agents across hundreds of thousands
> of production telephony calls, in multiple languages, where callers
> code-switch mid-sentence and two seconds of silence means a hangup.

## Why Trunkline

- **One contract, three runtimes.** `AgentSpec` is plain YAML. Switching a
  customer from a cascade to speech-to-speech is a config change: same tools,
  same analytics, same cost attribution.
- **Local-first.** faster-whisper for STT, Ollama for the LLM, Kokoro for TTS.
  Production-tuned turn-taking and interruption handling with zero cloud
  dependencies. Cloud providers are opt-in upgrades.
- **Latency as correctness.** In voice, dead air is a conversational error.
  Trunkline's primitives make it structurally impossible rather than patching it:
  - *Tool-call choreography* — the model cannot emit a tool call without
    declaring what to say while it runs (`waiting_message`, `spoken_mode`,
    `post_tool_response` are schema-required fields).
  - *Speak-while-thinking* — a character-level streaming parser flushes the
    `speak` field to TTS while the tool call is still being generated.
  - *Interruption reconciliation* — TTS generates faster than audio plays; on
    a barge-in, Trunkline annotates context so the model never references words
    the caller never actually heard.
  - *Cache-safe language lock* — the language directive is re-asserted every
    turn without breaking prompt-cache byte-stability.
- **Honest cost accounting.** Every metered unit carries an `estimated` flag.
  Per-call, per-connected-minute, per-conversation, and per-outcome roll-ups,
  so you optimize the denominator instead of the rate.
- **Per-language provider routing.** Real callers code-switch. The best STT
  engine for one language is often the wrong one for another; Trunkline routes
  STT/TTS/LLM per agent *and* per detected language.

## Quickstart

```bash
pip install trunkline[local]
```

```yaml
# agent.yaml
name: front-desk
persona: |
  You are a friendly front-desk assistant for a dental clinic.
language:
  primary: en
runtime:
  mode: cascade
  routing:
    default: { stt: faster_whisper, llm: ollama, tts: kokoro }
```

```python
from trunkline import AgentSpec, CallSession
from trunkline.runtimes.cascade import CascadeRuntime

agent = AgentSpec.from_yaml("agent.yaml")
session = CallSession(agent)
runtime = CascadeRuntime(session)
```

See [`examples/`](examples/) for the full local quickstart, the console
dev loop, and cost reporting.

## Status

Early alpha. The core contracts, primitives, cost meter, and local cascade
runtime are usable; speech-to-speech and hybrid adapters, telephony ingress
(SIP), and the hosted dashboard are on the [roadmap](docs/ROADMAP.md).

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT © Anzal Hussain Abidi
