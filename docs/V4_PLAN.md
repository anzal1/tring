# v0.4 / v0.5 build plan

Approved scope: build all tracks. Ship as 0.4.0 once every track passes the
same bar as prior releases: offline tests (fakes only), ruff clean,
mypy --strict clean, live smoke where keys exist, Studio driven in a real
browser, then the tokenless release pipeline.

Two parallel workstreams with disjoint file trees:

- **Core** (Python): everything under `src/tring/` except `studio/`
- **Studio** (React): `studio-ui/` plus additive changes to `studio/server.py`

## Track 1: Evaluation harness (`src/tring/eval/`)

Scripted conversation tests as YAML, runnable in CI against fake or real
providers.

```yaml
# evals/booking.yaml
agent: agent.yaml
turns:
  - user: "book me for tuesday, 98200 11223"
    expect:
      tool_called: check_appointment
      tool_args_include: { phone: "9820011223" }
      reply_mentions: ["tuesday"]
      max_first_token_s: 2.0
judges:
  - name: politeness
    prompt: "Rate 1-5 whether the assistant stayed polite and concise."
    min_score: 4
```

- `tring.eval.run(path) -> EvalReport`; CLI `python -m tring.eval evals/`.
- Deterministic assertions run without any LLM; judge assertions need an
  `llm` provider config and are skipped (reported as skipped) without one.
- Exit code nonzero on failure: CI-ready. Report renders as terminal table
  and JSON.

## Track 2: Outbound and campaigns (`src/tring/outbound/`)

- `Campaign` model: list of callees (phone, metadata), agent ref, dialing
  policy (max concurrency, retry count/backoff, quiet hours, callback window).
- `CampaignRunner`: asyncio orchestrator placing calls through a `Dialer`
  protocol; ships `TwilioDialer` (REST create-call pointing at the Media
  Streams webhook) and `FakeDialer` for tests.
- Outcome tracking per callee: dialed / connected / conversation / outcome
  (outcome set by a designated tool call or explicit API), persisted as JSONL.
- `CampaignReport`: the denominator ladder computed live from real counts,
  reusing `cost.report.denominator_ladder`.

## Track 3: Observability (`src/tring/observability/`)

- `otel.py`: an event-stream consumer exporting spans via OpenTelemetry
  (lazy import, extra `tring[otel]`): session = trace; each turn = span with
  child spans stt/llm/tts; tool calls as spans; cost and cached_units as
  attributes. No vendor coupling: standard OTLP.
- `latency.py`: per-turn waterfall from event timestamps (user_final ->
  first_token -> first_audio -> playout), aggregation to p50/p95.
- `alarms.py`: cache-regression detector: rolling cached_units ratio per
  provider; a configurable drop triggers a SessionError-level event and a
  callback hook.

## Track 4: Telephony breadth (`src/tring/transports/`, `src/tring/telephony/`)

- `telephony/freeswitch_gen.py`: generate FreeSWITCH dialplan + mod_audio_stream
  config from a YAML description (DID -> agent mapping), rendering the SIP ->
  websocket bridge documented in TELEPHONY.md. Pure text generation, fully
  unit-tested against golden files.
- DTMF: Twilio transport forwards DTMF events as a new `DtmfReceived` session
  event (additive to events.py: see "event model additions" below).
- Transfer-to-human: a built-in `transfer_call` tool contract; Twilio
  implementation via REST call update (verify against docs), cascade emits
  `session_ended(reason="transferred")`.
- Answering-machine detection: Twilio AMD flag passthrough on the dialer;
  outcome recorded on the campaign row.

## Track 5: S2S maturity (`src/tring/providers/cloud/`, `runtimes/`)

- `gemini_live` S2SProvider (websocket, verify protocol against
  ai.google.dev/api/live docs; skip honestly if unverifiable).
- Post-call transcript reconciliation: align provider transcripts to event
  timeline timestamps where offered.
- Hybrid v2: mid-call downgrade: on tool-heavy turns or S2S errors, hand the
  session to a cascade runtime with context carried over; document the seams
  honestly.

## Track 6: Knowledge slot (`src/tring/knowledge/`)

- `KnowledgeProvider` ABC: `search(query, k) -> list[Passage]`; registry kind
  "knowledge".
- Ships: `chroma_local` and `qdrant` (lazy imports, extra `tring[knowledge]`),
  plus `keyword_local` (zero-dep BM25-ish for tests/small KBs).
- Cascade integration: a built-in `search_knowledge` tool; **speculative
  retrieval**: kick off search from partial STT transcripts so results are
  ready before the turn ends; results injected as tool output.
- AgentSpec addition: optional `knowledge:` block (provider + options + top_k).

## Track 7: Studio flow builder + replay + mic (studio-ui + server)

### Protocol additions (merge into STUDIO_PROTOCOL.md after 0.3.0 ships)

- `GET /api/sessions` -> `[{id, started, agent, turns, cost}]`;
  `GET /api/sessions/{id}` -> `{events: [...]}`. Server persists each
  session's event list as JSON under `~/.tring/studio/sessions/`.
- WS binary frames = 16 kHz mono 16-bit PCM mic audio (existing text
  messages unchanged). Server routes binary to `runtime.push_audio` and
  streams bot audio back as binary frames. `{"type":"mode","audio":true}`
  switches STT slot handling from text_input to the configured STT.
- `POST /api/flow/compile`: `{flow: FlowGraph}` -> `{yaml, ok, error}`.

### Flow builder (studio-ui)

- Canvas of nodes: `say`, `ask` (slot filling), `branch` (intent/condition),
  `tool`, `handoff`, `end`. Edges = transitions. Implemented with @xyflow/react
  (React Flow, MIT).
- Compiles to AgentSpec: persona assembled from node graph as structured
  instructions plus per-node tools; the compiler lives in `src/tring/flow.py`
  (server side, unit-tested) so the graph is data, not UI state.
- Flow JSON stored in `AgentSpec.metadata.flow` so round-tripping is lossless.

### Session replay (studio-ui)

- Sessions list from `/api/sessions`; replay view = existing panels fed by a
  cursor over stored events with a time scrubber (events are timestamped;
  playback is pure state reconstruction, no new render code).

### Mic testing (studio-ui)

- getUserMedia -> AudioWorklet downsample to 16k PCM -> ws binary frames;
  bot audio played via WebAudio. Push-to-talk button (spacebar) with the
  speech meter driven by real playback.

## Event model additions (additive only, discriminated union extended)

- `DtmfReceived {digit}`
- `SessionTransferred {target}`
- `KnowledgeSearched {query, results, latency_seconds, speculative}`

## Bar for shipping (unchanged)

Every track: unit tests with fakes; no network in tests; teaching docstrings;
rate-card entries for anything billable; honest capability flags; docs pages.
Release only after: full pytest, ruff, mypy --strict, live smoke, Studio
driven end to end in a real browser against the real server.
