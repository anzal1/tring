# Deploying Tring

Three tiers, smallest first. Every tier runs the same code; pick by how much
infrastructure you want to own today.

## 1. A laptop (onboarding)

```bash
pip install "tring[transports]"
tring studio
```

Opens Studio at http://localhost:8900 with a bundled demo agent. Typed
testing needs an LLM: either a provider key exported by name
(`GEMINI_API_KEY`, `OPENAI_API_KEY`, ...) referenced in the agent's routing,
or a local Ollama (`routing: {default: {llm: ollama}}`) for a zero-key setup.
The fully local voice pipeline adds `pip install "tring[local]"`
(faster-whisper, Kokoro, Silero VAD).

Evals run the same way: `tring eval evals/` exits nonzero on failure, so the
command drops straight into CI.

## 2. A container

```bash
docker compose up        # -> http://localhost:8900
```

The compose file passes provider keys through by name and keeps sessions and
the agent spec on a named volume, so upgrades keep your history. Uncomment
the `ollama` service in `docker-compose.yml` for a fully local stack.

Plain Docker, no compose:

```bash
docker build -t tring .
docker run -p 8900:8900 -v tring-data:/data -e GEMINI_API_KEY=... tring
```

## 3. Production calls

Studio is the workbench; production traffic goes through a transport:

- **Twilio**: point a TwiML `<Connect><Stream>` at
  `tring.transports.twilio.serve_twilio` (see `docs/TELEPHONY.md`); playout
  marks make interruption reconciliation exact on real calls.
- **Generic SIP / PBX**: generate FreeSWITCH configs with
  `tring.telephony.freeswitch_gen` and bridge audio to the websocket
  transport (`docs/TELEPHONY.md`, Generic SIP section).
- **Apps / browsers**: `tring.transports.websocket.serve` speaks 16 kHz mono
  PCM both ways.

Operational notes:

- Run behind any TLS-terminating reverse proxy; the studio and the websocket
  transport are plain HTTP/WS listeners. Studio has no authentication of its
  own yet, so treat it as an internal tool: bind it to localhost or put it
  behind your proxy's auth.
- Keys are only ever read from environment variables named in the agent spec;
  nothing secret lives in YAML, so specs are safe to commit.
- Observability: export sessions as OpenTelemetry spans with
  `tring.observability.otel`, and watch prompt-cache health with
  `tring.observability.alarms`.
