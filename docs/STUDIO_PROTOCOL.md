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
```

`--agent` defaults to `./agent.yaml`, falling back to a bundled demo spec if
absent (written to the path on first save).
