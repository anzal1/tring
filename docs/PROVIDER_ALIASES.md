# Reaching more vendors through `openai_compatible`

Tring ships one HTTP-shaped LLM adapter for the entire OpenAI-Chat-Completions
family: `openai_compatible` (`src/tring/providers/cloud/__init__.py`). Every
vendor below speaks that same wire format, so none of them needs its own
adapter class -- they need only a `base_url`, an `api_key_env`, and a `model`
in the agent spec's `options`. `anthropic` and `gemini`
(`src/tring/providers/cloud/llm_extra.py`) are the two exceptions: their APIs
are not OpenAI-shaped, so they get dedicated adapters instead.

Every `base_url` below was checked against the vendor's own docs (fetched
2026-09); anything that couldn't be pinned to an official source was left out
rather than guessed. Vendors move their bases occasionally -- if a request
starts 404ing, that's the first thing to re-check.

```yaml
runtime:
  routing:
    default:
      llm: openai_compatible
      options: { base_url: https://api.groq.com/openai/v1, model: llama-3.3-70b-versatile, api_key_env: GROQ_API_KEY }
```

---

## Cloud vendors

| Vendor | `base_url` | `api_key_env` (suggested) | Source |
|---|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | `GROQ_API_KEY` | console.groq.com/docs/openai |
| Cerebras | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` | inference-docs.cerebras.ai/introduction |
| Mistral | `https://api.mistral.ai/v1` | `MISTRAL_API_KEY` | docs.mistral.ai/api |
| Together AI | `https://api.together.ai/v1` | `TOGETHER_API_KEY` | docs.together.ai/docs/openai-api-compatibility |
| Fireworks AI | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` | docs.fireworks.ai/tools-sdks/openai-compatibility |
| DeepSeek | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | api-docs.deepseek.com (no `/v1` suffix -- the docs' own examples post straight to the bare host) |
| xAI (Grok) | `https://api.x.ai/v1` | `XAI_API_KEY` | docs.x.ai/docs/overview |
| OpenRouter | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | openrouter.ai/docs/quickstart |

Each row is a drop-in `options` block:

```yaml
options: { base_url: "<base_url>", model: "<vendor's model id>", api_key_env: "<api_key_env>" }
```

`api_key_env` is only ever the *name* of an environment variable (see
`providers/cloud/__init__.py`'s module docstring) -- pick any name you like,
the suggestions above just avoid collisions when an agent spec routes to more
than one of these vendors at once.

---

## Local servers

Same adapter, no cloud vendor at all -- these are OpenAI-compatible servers
you run yourself. `api_key_env` is typically omitted (`OpenAICompatibleLLM`
skips the `authorization` header entirely when no env var name is given).

| Server | `base_url` | `api_key_env` | Source |
|---|---|---|---|
| Ollama | `http://localhost:11434/v1` | (none) | ollama.com/blog/openai-compatibility |
| LM Studio | `http://localhost:1234/v1` | (none) | lmstudio.ai/docs/app/api/endpoints/openai |
| vLLM | `http://localhost:8000/v1` | (none, unless `--api-key` was passed to `vllm serve`) | docs.vllm.ai/en/latest/serving/openai_compatible_server (documented example host:port; both are configurable via `--host`/`--port`) |

```yaml
options: { base_url: "http://localhost:11434/v1", model: "llama3.1" }
```

---

## Why no per-vendor adapter classes

`OpenAICompatibleLLM` already does the one thing that varies across this
whole list correctly: it reads `usage.prompt_tokens_details.cached_tokens`
when a vendor sends it and passes it through as `Usage.cached_units`, and it
reports zero-guesswork `estimated=False` usage whenever the vendor's `usage`
block is present at all. Adding a same-shaped subclass per vendor would only
duplicate that logic under a different name -- the `base_url`/`api_key_env`
options table above is the entire adapter surface these vendors need. A
vendor earns its own class in `providers/cloud/` only once its wire protocol
actually diverges from OpenAI's -- which is exactly why `anthropic` and
`gemini` have one and this whole list does not.
