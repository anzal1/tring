# Extending Tring

Tring is designed to be built on. This guide covers the five extension points in
depth: providers, tools, event consumers, transports, and runtimes, plus the
testing patterns the repo itself uses.

The stability contract: everything documented here (`AgentSpec`, `SessionEvent`,
`RuntimeAdapter`, the provider ABCs, the registry) is public API. Within a
minor version these do not break. Anything prefixed `_` is not.

---

## 1. Providers

A provider adapts one vendor or model to one of four streaming interfaces in
[`tring/providers/base.py`](../src/tring/providers/base.py): `STTProvider`,
`LLMProvider`, `TTSProvider`, `S2SProvider`.

### Rules

1. **Streaming-first.** Voice latency budgets do not allow request/response
   round trips. `LLMProvider.generate` must yield text deltas as they arrive;
   `TTSProvider.synthesize` consumes an *async iterator* of text (so it can
   start synthesizing before the sentence is finished) and yields audio.
2. **Lazy imports.** Import your SDK inside methods, never at module top.
   Raise a helpful `ImportError` naming the pip extra. The core install must
   stay pydantic + pyyaml only.
3. **Honest usage.** Return `Usage` with vendor-reported numbers and
   `estimated=False` whenever the vendor gives you exact figures. If you must
   infer (chars/4 token guesses, duration-based audio), set `estimated=True`.
   Silently estimated usage is treated as a bug in review.
4. **Report cached tokens.** If your LLM vendor reports cached prompt tokens,
   pass them through as `Usage.cached_units`. This is how a silent
   prompt-cache regression stays visible.
5. **Ship a rate.** Add a `Rate` entry (with `as_of` date) so the cost engine
   can price your provider out of the box.

### Skeleton

```python
from collections.abc import AsyncIterator

from tring.providers.base import LLMChunk, LLMProvider, Usage
from tring.providers.registry import register


@register("llm", "acme")
def _make_acme(**options: object) -> "AcmeLLM":
    return AcmeLLM(
        model=str(options.get("model", "acme-small")),
        api_key_env=str(options.get("api_key_env", "ACME_API_KEY")),
    )


class AcmeLLM(LLMProvider):
    name = "acme"

    def __init__(self, model: str, api_key_env: str) -> None:
        self.model = model
        self.api_key_env = api_key_env  # env var NAME; never a raw key

    async def generate(self, messages, tools=None) -> AsyncIterator[LLMChunk]:
        import os

        import httpx  # lazy: keeps core install light

        key = os.environ[self.api_key_env]
        ...
```

Agent YAML then selects it, with any options passed to your factory:

```yaml
runtime:
  routing:
    default:
      llm: acme
      options: { model: acme-large, api_key_env: ACME_KEY_PROD }
```

### Testing pattern

Never hit the network. For HTTP providers, unit-test request construction with
`httpx.MockTransport`. For behavior, register a fake under a test-only name and
drive the runtime with it; see [`tests/test_cascade.py`](../tests/test_cascade.py)
for the house pattern.

---

## 2. Tools

A tool is a YAML declaration plus an async handler:

```yaml
tools:
  - name: create_lead
    description: Create or update a CRM lead by phone number.
    parameters:
      type: object
      properties:
        phone: { type: string }
        name: { type: string }
      required: [phone]
    timeout_seconds: 8
```

```python
async def create_lead(args: dict) -> dict:
    return await crm.upsert(**args)

runtime = CascadeRuntime(session, handlers={"create_lead": create_lead})
```

What you get for free, from
[`primitives/choreography.py`](../src/tring/primitives/choreography.py):

- The model is *forced* to plan the silence: `waiting_message`, `spoken_mode`,
  and `post_tool_response` are required fields of the call schema.
- Timeouts (`timeout_seconds`) and handler exceptions never produce dead air:
  failures force a spoken explanation (`should_respond=True`).
- `ToolCallStarted` / `ToolCallCompleted` events with latency, for analytics.

Handlers should be idempotent when possible (callers repeat themselves) and
fast (aim for under a second; use the waiting message to buy time, not as an
excuse).

Set `choreographed: false` on a tool to bypass the choreography fields, for
silent background tools that should never affect what the bot says.

---

## 3. Event consumers

`CallSession.subscribe()` returns an async iterator of typed, discriminated
events (see [`tring/events.py`](../src/tring/events.py)). Multiple consumers
can subscribe to the same session; each gets its own queue. `session.history`
holds everything emitted so far, so late-attaching consumers can catch up.

Things people build here:

- **Observability**: latency percentiles per turn, barge-in rates, tool failure
  alerting, cost dashboards with the `estimated` fraction as a data-quality
  metric.
- **Compliance/QA**: transcript stores, PII redaction pipelines, call scoring.
- **Live UX**: agent-assist screens, supervisor barge-in consoles.

Two contract details worth knowing:

- Transcript events carry `availability`: `live` (cascade, mid-turn) or
  `post_call` (S2S reconstructions). Do not assume timing; check the field or
  the runtime's `capabilities.live_transcripts`.
- `CostRecorded.estimated` and `cached_units` are the honesty bits. If you
  aggregate costs, aggregate the estimated fraction too.

---

## 4. Transports

A transport moves audio between the outside world and a runtime. The whole
contract is three touchpoints:

1. Feed caller audio: `await runtime.push_audio(AudioFrame(pcm, 16000, 1))`.
   Canonical wire format is 16 kHz, mono, 16-bit linear PCM.
2. Receive bot audio: set `runtime.on_bot_audio = lambda frame: ...` before
   `start()`.
3. Lifecycle: `await runtime.start()` ... `await runtime.stop(reason)`.

[`transports/websocket.py`](../src/tring/transports/websocket.py) is the
reference implementation (about a hundred lines); a SIP/FreeSWITCH bridge, a
Twilio media-streams adapter, or a WebRTC gateway follows the same shape: your
job is resampling and framing, Tring's job is the conversation.

**Exact interruption reconciliation.** The cascade runtime estimates playout
progress from frame durations. A transport that knows the *real* playout clock
(telephony stacks do) should drive the ledger itself:

```python
runtime.ledger.mark_played(utterance_id, upto_chars)
```

That upgrades heard/unheard splits from estimated to exact.

---

## 5. Runtimes

Subclass [`RuntimeAdapter`](../src/tring/runtimes/base.py) when you have a new
architecture: a new S2S vendor session, an on-device duplex model, a research
pipeline. The contract:

- Lifecycle is `start()` → many `push_audio()` → `stop()`.
- Emit the standard events for everything conversational. If your architecture
  only has transcripts after the call, emit them in `stop()` with
  `availability=POST_CALL`; never fabricate live ones.
- Declare `RuntimeCapabilities` honestly. Consumers branch on these flags;
  optimistic capabilities are bugs in other people's code.
- Route provider `Usage` to the meter if one is attached.

If you do all four, every existing tool, transport, analytics consumer, and
cost report works with your runtime unchanged. That is the point of the
contract. [`runtimes/cascade.py`](../src/tring/runtimes/cascade.py) is written
to be read; start there.

---

## Using the primitives à la carte

All four primitives are dependency-free (pydantic only) and usable in any
pipeline, including Pipecat or LiveKit Agents:

```python
from tring.primitives.speak_parser import SpeakToolParser
from tring.primitives.choreography import augment_tool_schema, parse_choreographed_call
from tring.primitives.interruption import PlaybackLedger
from tring.primitives.language_lock import LanguageLock
```

`SpeakToolParser` in particular is a plain `feed(chunk) -> events` /
`finalize() -> events` state machine with zero Tring coupling: point your LLM
stream at it and your TTS at its `SpeakDelta`s.

---

## House testing patterns

- **Everything offline.** No test may require a network call, an API key, a
  GPU, or audio hardware. CI enforces this by having none of them.
- **Fake providers, real runtime.** Register fakes under test-only names and
  drive the actual `CascadeRuntime` through scripted conversations, asserting
  on the ordered event sequence. This catches contract drift better than
  mocking runtime internals.
- **Deterministic clocks.** `CallSession(clock=...)` injects time. Event `at`
  fields are then exact, so latency assertions are stable.
- **Prefix-stability for anything cache-adjacent.** If your feature touches the
  message array, add a byte-prefix test like the one in
  [`tests/test_language_lock.py`](../tests/test_language_lock.py).
