# Tring Providers

This document lists all speech and language providers available in Tring. Each slot (STT, LLM, TTS, S2S) supports multiple implementations, allowing you to mix and match based on language, latency requirements, and cost constraints.

API keys are always stored in environment variables (never in specs). Each cloud provider specifies a default env var name but accepts an `api_key_env` option to override it.

---

## Speech-to-Text (STT)

Converts audio to text. Input: 16 kHz mono PCM frames. Output: incrementally streamed `STTResult` objects with `text`, `final: bool`, optional `language`, and usage metrics.

### Local Providers

| Provider | Vendor | Key Env Var | Notable Options | Usage Exactness |
|---|---|---|---|---|
| `faster_whisper` | OpenAI Whisper (run locally) | (none) | `model` (default: "base"), `device` ("auto", "cpu", "cuda"), `silence_threshold`, `silence_seconds` | Exact: audio_seconds from byte count |
| `faster_whisper_streaming` | OpenAI Whisper (run locally, true streaming) | (none) | `model` (default: "base"), `device` ("auto", "cpu", "cuda"), `silence_threshold`, `silence_seconds` | Exact: audio_seconds from byte count |
| `text_input` | (dev/test) | (none) | (accepts and ignores all options) | No usage reported |

### Cloud Providers

| Provider | Vendor | Key Env Var | Notable Options | Usage Exactness | Added in v0.2 |
|---|---|---|---|---|---|
| `deepgram` | Deepgram | `DEEPGRAM_API_KEY` | `model` (default: "nova-2"), `sample_rate` (default: 16000) | Exact: audio_seconds from bytes sent | |
| `assemblyai` | AssemblyAI | `ASSEMBLYAI_API_KEY` | `speech_model` (default: "universal-3-5-pro"), `sample_rate`, `format_turns` | Exact: audio_seconds (billed wall-clock connection time) | Yes |
| `openai_stt` | OpenAI (also reaches Groq's OpenAI-compatible endpoint via `base_url`) | `OPENAI_API_KEY` | `model` (default: "whisper-1"), `base_url`, `language`, `response_format` | Exact: vendor `usage` block when present (tokens for gpt-4o-transcribe*, duration for whisper-1), else exact byte-computed audio_seconds | Yes |
| `sarvam` | Sarvam AI | `SARVAM_API_KEY` | `model` (default: "saaras:v3"), `base_url`, `language_code` | Exact: byte-computed audio_seconds (no usage block in the documented response) | Yes |

`google_stt` (Google Cloud Speech-to-Text v2) was evaluated for v0.2 and **not implemented**: its `recognize` endpoint only accepts OAuth2/service-account credentials, not a simple API key, so it does not fit this repo's `api_key_env`-names-an-env-var pattern without an unverified auth flow.

---

## Language Model (LLM)

Generates response text from conversation context. Input: message array and optional tool definitions. Output: streamed text deltas and final token usage.

### Local Providers

| Provider | Vendor | Key Env Var | Notable Options | Usage Exactness |
|---|---|---|---|---|
| `ollama` | Ollama | (none) | `model` (default: "llama3.1"), `base_url` (default: "http://localhost:11434"), `temperature`, `timeout_seconds`, `keep_alive` | Exact: tokens_in, tokens_out from model |

### Cloud Providers

| Provider | Vendor | Key Env Var | Notable Options | Usage Exactness | Cached Token Support | Added in v0.2 |
|---|---|---|---|---|---|---|
| `openai_compatible` | Any OpenAI-Chat-Completions-shaped API (OpenAI, vLLM, Groq, Together, etc.) | (no default; set `api_key_env` yourself) | `base_url` (required), `model` (required) | Exact: tokens_in, tokens_out with cached awareness | Yes | |
| `anthropic` | Anthropic | `ANTHROPIC_API_KEY` | `model` (default: "claude-sonnet-5"), `max_tokens`, `base_url` | Exact: tokens_in, tokens_out | Yes (`cache_read_input_tokens`) | Yes |
| `gemini` | Google Gemini | `GEMINI_API_KEY` | `model` (default: "gemini-2.5-flash"), `base_url` | Exact: tokens_in, tokens_out | Yes (`cachedContentTokenCount`) | Yes |

See [docs/PROVIDER_ALIASES.md](PROVIDER_ALIASES.md) for verified `base_url`/`api_key_env` values to point `openai_compatible` at Groq, Cerebras, Mistral, Together AI, Fireworks AI, DeepSeek, xAI, OpenRouter, Ollama, LM Studio, or vLLM.

---

## Text-to-Speech (TTS)

Generates audio from text. Input: async iterator of text deltas (speak-while-thinking). Output: streamed 16 kHz mono 16-bit PCM audio frames with usage metrics.

### Local Providers

| Provider | Vendor | Key Env Var | Notable Options | Usage Exactness |
|---|---|---|---|---|
| `kokoro` | kokoro-onnx | (none) | `voice` (default: "af_heart"), `speed`, `lang`, `model_path`, `voices_path` | Exact: tts_chars from input text |
| `piper` | Piper (`piper-tts` package) | (none) | `model_path` (required, no default), `use_cuda`, `length_scale`, `noise_scale`, `noise_w_scale`, `volume` | Exact: tts_chars from input text |

### Cloud Providers

| Provider | Vendor | Key Env Var | Notable Options | Usage Exactness | Added in v0.2 |
|---|---|---|---|---|---|
| `elevenlabs` | ElevenLabs | `ELEVENLABS_API_KEY` | `voice_id` (default: "21m00Tcm4TlvDq8ikWAM"), `model_id` (default: "eleven_turbo_v2_5") | Exact: tts_chars | |
| `openai_tts` | OpenAI | `OPENAI_API_KEY` | `model` (default: "tts-1"), `voice` (default: "alloy"), `base_url` | Exact: tts_chars | Yes |
| `sarvam_tts` | Sarvam AI | `SARVAM_API_KEY` | `model` (default: "bulbul:v3"), `speaker` (default: "shubh"), `language_code` (default: "en-IN") | Exact: tts_chars | Yes |
| `cartesia` | Cartesia | `CARTESIA_API_KEY` | `model_id` (default: "sonic-3"), `voice_id` (required, no safe default), `language` | Exact: tts_chars (priced at $0.0 for now: no verifiable per-character rate is published) | Yes |

---

## Speech-to-Speech (S2S)

End-to-end audio models that listen and respond directly. Input: 16 kHz mono PCM frames. Output: 16 kHz mono 16-bit PCM frames plus optional post-call transcripts.

`sarvam` and `cartesia` are STT/TTS-only providers in this codebase (see the tables above): they are not registered as S2S models.

### Cloud Providers

| Provider | Vendor | Key Env Var | Notable Options | Capabilities | Added in v0.2 |
|---|---|---|---|---|---|
| `openai_realtime` | OpenAI | `OPENAI_API_KEY` | `model` (default: "gpt-realtime-2.1"), `voice` (default: "marin"), `turn_detection` (default: "semantic_vad") | Real-time, tools, server-side turn detection | Yes |
| `ultravox` | Fixie.ai (Ultravox) | `ULTRAVOX_API_KEY` | `model` (default: "ultravox-v0.7"), `voice` (default: "Mark") | Low-latency, tools, transcript events | Yes |
| `gemini_live` | Google | `GEMINI_API_KEY` | `model` (default: "gemini-3.8-live"), `system_instruction`, `tools`, `input_transcription`/`output_transcription` (default: true) | Real-time, tool calls, input/output transcripts, exact end-of-stream token usage | v0.4 |

---

## Routing by Language

Tring supports per-language provider routing. Specify different provider stacks for different languages in your agent spec:

```yaml
runtime:
  mode: cascade
  routing:
    default: { stt: deepgram, llm: openai_compatible, tts: elevenlabs }
    hi:      { stt: assemblyai, llm: anthropic, tts: sarvam_tts }
    mr:      { stt: faster_whisper, llm: ollama, tts: kokoro }
```

---

## Installing Provider Extras

Core install (providers available by name, not their SDKs):
```bash
pip install tring
```

Local providers (requires ML stack):
```bash
pip install "tring[local]"
```

Cloud providers (requires HTTP clients for APIs):
```bash
pip install "tring[cloud]"
```

Both:
```bash
pip install "tring[local,cloud]"
```

Importing `tring.providers.local` or `tring.providers.cloud` never fails and never touches the network on its own, even with no optional dependency installed: every vendor-specific dependency (`websockets`, `numpy`, `faster_whisper`, `kokoro_onnx`, `piper`) is imported lazily inside the method that needs it, and a missing package raises an `ImportError` naming the extra to install. `httpx` is the one exception: it backs every cloud HTTP call and is declared as required by the `local`, `cloud`, and `dev` extras, so it is imported at module scope. `registry.available()` lists every provider name regardless of which extras are actually installed.

---

## Usage Exactness

- **Exact**: vendor-reported counts or measured from actual data (exact byte counts, model outputs). Set `estimated=False`.
- **Estimated**: inferred from proxies (character count as token proxy, fixed duration samples). Set `estimated=True`.

All usage reports include an `estimated` flag. Silently estimated usage is treated as a bug; cost reports aggregate the estimated fraction so you can monitor data quality.
