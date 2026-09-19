# Tring Studio protocol

Studio is a single-page app served by `python -m tring.studio` (default port
8900). The server uses only the `websockets` package (the `transports` extra):
its `process_request` hook serves the static page and JSON endpoints; the same
port carries the live-session websocket. No new dependencies.

## HTTP endpoints (same port as the websocket)

| Method + path | Body | Returns |
|---|---|---|
| `GET /` | | `index.html` (embedded in the package as a resource) |
| `GET /api/meta` | | `{"version": str, "providers": {"stt": [...], "llm": [...], "tts": [...], "s2s": [...]}}` from the registry |
| `GET /api/agent` | | `{"yaml": str}` current agent spec as YAML text |
| `POST /api/agent` | `{"yaml": str}` | `{"ok": true}` or `{"ok": false, "error": str}`; validates via `AgentSpec` before accepting, persists to the `--agent` path |

All JSON responses carry `Content-Type: application/json`. Validation errors
return HTTP 200 with `ok: false` so the UI owns the presentation.

## WebSocket `/ws`

One live `CallSession` per connection, torn down on close.

Client to server (JSON text frames):

```json
{"type": "start"}                       // build session+runtime from the saved agent
{"type": "user_text", "text": "hello"}  // one user turn (text_input STT path)
{"type": "reset"}                       // stop and rebuild the session
```

Server to client:

```json
{"type": "ready", "session_id": "...", "capabilities": {...}}
{"type": "event", "event": { ...SessionEvent.model_dump(mode="json")... }}
{"type": "error", "message": "..."}
```

Every `SessionEvent` is forwarded verbatim under `"event"`; the UI derives the
chat transcript, the tool timeline, and the cost panel from that one stream,
exactly like any other event consumer. Bot audio frames are counted, not sent:
the runtime's `on_bot_audio` increments a byte counter surfaced via a
synthetic `{"type": "audio_progress", "bytes": n}` message so the UI can show
speech activity without shipping PCM to the browser.

For text testing the studio forces the STT slot to `text_input` and, when no
TTS key or local model is available, the TTS slot to `studio_silent` — a
studio-registered TTSProvider that emits no frames but reports `tts_chars`
usage, so the cost panel stays honest even in silent mode.

## CLI

```
python -m tring.studio [--agent examples/agent.yaml] [--host 127.0.0.1] [--port 8900]
                       [--sessions-dir ~/.tring/studio/sessions]
```

`--agent` defaults to `./agent.yaml`, falling back to a bundled demo spec if
absent (written to the path on first save).

---

# v0.4 additions

Everything above still holds byte for byte: a v0.3 client talks to a v0.4
server unchanged, gets the same `ready` / `event` / `audio_progress` messages,
and never sees a binary frame unless it asks for one. The additions are session
history, the flow compiler, and microphone mode.

## Session history

Every session's events are appended to `~/.tring/studio/sessions/<id>.jsonl`,
one `SessionEvent` JSON object per line, in the order they were emitted. The
line is written as the event happens, so a studio that is killed mid-call still
leaves the call on disk up to the moment it died. `index.json` in the same
directory holds the summaries, newest first, and is rewritten when a session
ends (which includes ending by disconnecting the browser).

Summaries are *derived from the events*, never supplied alongside them:
`agent` comes from `session_started`, `turns` counts final `user_transcript`
events, `cost` sums `cost_recorded.amount`. The one thing the event stream does
not carry is wall-clock time — `at` is seconds since session start, by design,
so replays are deterministic — so the server stamps `started` (ISO-8601, UTC)
when it writes the session's first line.

| Method + path | Body | Returns |
|---|---|---|
| `GET /api/sessions` | | `[{id, started, agent, turns, cost}]`, newest first (a JSON **array**) |
| `GET /api/sessions/{id}` | | `{id, started, agent, turns, cost, events: [...]}`, or `404` |
| `POST /api/flow/compile` | `{"flow": FlowGraph}` | `{ok: true, yaml, note?}` or `{ok: false, error}` |

`GET /api/sessions/{id}` re-reads the file rather than serving the index, so a
session that is still running (or that was interrupted) reads back with
everything it has so far. An id outside `[A-Za-z0-9][A-Za-z0-9_-]{0,63}` is a
404 rather than a sanitized path.

The directory is `--sessions-dir` on the CLI (and `StudioServer(sessions_dir=)`
in code), and the startup banner names it, because a tool that writes every
session to your home directory should say so the first time you run it rather
than the first time you go looking for the files.

## `POST /api/flow/compile`

Compiles a flow graph into an `AgentSpec` and returns it as text. It **saves
nothing**: seeing what a flow compiles to is not the same act as replacing your
agent with it, so the UI posts the returned text to `POST /api/agent` when the
user means the second thing.

The currently saved spec is the compile's *base*: it supplies provider routing,
language policy, limits, and any persona text that describes how the agent
speaks. The flow supplies the numbered steps, the tools its `tool` nodes carry,
and the graph itself. A saved spec that will not load does not block the
compile; the flow compiles against no base and the response carries a `note`
saying so.

`yaml` is the compiled spec as JSON text. JSON is a subset of YAML, so it loads
as a spec, and it is what lets the frontend open its structured editor on the
result (the same reason `GET /api/agent` serves the bundled demo as JSON).

Errors are HTTP 200 with `ok: false`, exactly like `POST /api/agent`: the text
is UI copy, shown next to the offending node. Every message from the compiler
leads with its subject, `node '<id>': <reason>` or `flow: <reason>`.

## Flow graphs

The compiler is `src/tring/flow.py`, server side and UI free: no node carries a
coordinate, and the same JSON compiles the same spec from the studio, from a
test, or from a script. Canvas positions belong to the UI's own storage, not to
this document.

```json
{
  "name": "northwind-booking",
  "greeting": "Northwind Dental, how can I help?",
  "entry": "welcome",
  "nodes": [ ... ],
  "edges": [ {"from": "welcome", "to": "details", "when": null} ]
}
```

`entry` may be omitted when exactly one node has nothing pointing at it. It
becomes required as soon as that is ambiguous; the compiler does not guess.

Every node is `{id, kind, label?}` plus the fields its kind uses. A field
belonging to another kind is a rejection, not a silent no-op.

| `kind` | Fields | Renders as |
|---|---|---|
| `say` | `text` (required) | `Say: "<text>"` |
| `ask` | `prompt` (required), `slots` (at least one) | the question, then a slot checklist and a re-ask instruction |
| `branch` | `condition` | `Decide, based on <condition>:` and one bullet per outgoing edge |
| `tool` | `tool` (a full `ToolDef`, required) | a call instruction; the tool is bound into `spec.tools` |
| `handoff` | `target` (required), `text` | what to say, who to transfer to, then stop |
| `end` | `text` | the closing line, then stop |

`slots` are `{name, description, required?}`; `description` is written for the
model ("a ten digit mobile number"), not as a form label.

Transitions:

- `say`, `ask`, `tool`: exactly one outgoing edge, and it carries no `when`.
- `branch`: two or more, each with a `when`, plus at most one default edge
  (no `when`) which always renders last as `Otherwise, ...`.
- `handoff`, `end`: none. This is where the path stops.

The flow must be reachable in full from the entry node and must contain at
least one `end` or `handoff`.

### What the compile produces

- **Persona**: the base persona, then a generated section that always begins
  `CONVERSATION FLOW (compiled from the flow graph; do not edit by hand)`.
  Steps are numbered by a topological walk, so a step is never introduced
  before the step that leads to it. Edges that close a cycle are kept and
  rendered as `loop back to step N`. The header is also the marker the
  compiler splits on, so recompiling replaces the section instead of stacking
  a second copy: compile, save, edit, compile again is stable.
- **Tools**: base tools first, then each `tool` node's, in step order. One tool
  called from several nodes is bound once. Two definitions of one name that
  disagree are a rejection, because the runtime binds tools by name and one of
  them would silently lose.
- **`metadata.flow`**: the graph, verbatim, so the canvas reopens the flow that
  built the spec rather than reverse-engineering one out of the prose.
- **`name` / `greeting`**: the graph's when it sets them, otherwise the base's.

## WebSocket: microphone mode

Client to server, additive:

```json
{"type": "mode", "audio": true}   // switch the STT slot; false returns to typed turns
<binary frame>                    // 16 kHz mono 16-bit little-endian PCM
```

Server to client, additive:

```json
{"type": "mode", "audio": true, "stt": "faster_whisper"}
{"type": "audio_format", "sample_rate": 24000, "channels": 1, "encoding": "pcm_s16le"}
<binary frame>                    // bot audio, in the announced format
```

`ready` also gains `"mode": {"audio": bool, "stt": str | null}`, so a client
that only ever reads `ready` still knows which slot it got.

Rules, and the reasons:

- **`audio` is a request, not a setting.** The STT slot is bound when the
  runtime starts, so `mode` with a live session rebuilds it: the browser gets a
  fresh `ready` exactly as it would from `reset`, then a `mode` message. With
  no session live the request is recorded and answered with `stt: null`.
- **The spec's own STT provider is kept only if it constructs.** Otherwise the
  slot falls back to `text_input`, `mode.audio` comes back `false`, and the
  reason arrives as a `SessionError` on the event stream, the same way an
  unavailable TTS provider already does. `text_input` itself is refused for
  microphone mode: it reads UTF-8 out of the `pcm` field, so feeding it
  microphone frames would produce a transcript of mojibake rather than an
  error anyone could diagnose.
- **Binary frames sent while microphone mode is off are dropped**, and the
  client is told once per session. A microphone produces fifty frames a second
  and an error per frame is a flood, not a diagnosis.
- **Bot audio is sent as binary frames only in microphone mode.** Text mode
  keeps the v0.3 behaviour exactly: the audio is counted, not shipped.
  `audio_progress` is still sent in both modes, still coalesced, still
  cumulative.
- **`audio_format` precedes the first frame** of a session and is re-sent only
  if the format changes. Providers do not all synthesize at the 16 kHz wire
  rate, and the browser cannot play PCM it has no sample rate for.
- **Bot audio frames are written without back-pressure.** The studio is a
  loopback tool; a client that cannot keep up with 32 kB/s of PCM has a bigger
  problem than frame pacing.
