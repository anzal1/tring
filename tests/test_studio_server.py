"""Tests for the Tring Studio server.

Everything here runs against a real :class:`~tring.studio.server.StudioServer`
bound to an ephemeral loopback port, driven by real clients: ``httpx`` for the
JSON and static endpoints, the ``websockets`` client for the session socket.
That is deliberate. The whole point of this module is the wire -- header
parsing, path-traversal refusal, content types, the WebSocket handshake, the
message shapes ``docs/STUDIO_PROTOCOL.md`` promises the frontend -- and a test
that calls the routing methods directly would assert on none of it.

No network, no keys, no GPU, no audio: the loopback socket is the only I/O, the
LLM is a fake registered under a test-only name (the house pattern from
``tests/test_cascade.py``), and the TTS slot is ``studio_silent``.

``websockets`` is an optional extra (``pip install "tring[transports]"``), so
the whole module skips when it is absent rather than failing a core install.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

pytest.importorskip("websockets", reason="tring[transports] is not installed")

from websockets.asyncio.client import connect

from tring import __version__
from tring.agent import AgentSpec
from tring.providers.base import (
    LLMChunk,
    LLMProvider,
    STTProvider,
    STTResult,
    TTSChunk,
    TTSProvider,
    Usage,
)
from tring.providers.registry import register
from tring.runtimes.base import AudioFrame
from tring.studio.server import DEMO_AGENT, StudioServer
from tring.studio.silent_tts import SILENT_TTS_NAME

# ---------------------------------------------------------------------------
# Fake providers, registered in the real registry so specs resolve them
# normally. The house pattern from tests/test_cascade.py.
# ---------------------------------------------------------------------------

STUDIO_LLM_NAME = "test_studio_llm"
STUDIO_STT_NAME = "test_studio_stt"
STUDIO_TTS_NAME = "test_studio_tts"

#: What the fake says, in the envelope the cascade runtime asks for.
SCRIPTED_REPLY = '{"speak": "We open at nine.", "tool_call": null}'

#: What the fake recognizer hears, whatever PCM it is handed.
SCRIPTED_TRANSCRIPT = "when do you open"

#: One frame of bot audio. Short, non-empty, and recognisable on the wire.
SPOKEN_PCM = b"\x01\x02\x03\x04"


@register("llm", STUDIO_LLM_NAME)
class ScriptedStudioLLM(LLMProvider):
    """Replies with one fixed envelope, streamed in small fragments."""

    name = STUDIO_LLM_NAME

    def __init__(self, **_options: Any) -> None:
        pass

    async def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMChunk]:
        for index in range(0, len(SCRIPTED_REPLY), 11):
            yield LLMChunk(text=SCRIPTED_REPLY[index : index + 11])
        yield LLMChunk(text="", finish=True)


@register("stt", STUDIO_STT_NAME)
class ScriptedStudioSTT(STTProvider):
    """A recognizer that reports one fixed final transcript per audio frame.

    Deliberately indifferent to what the PCM contains: the thing under test is
    the microphone *path* (binary frame to ``push_audio`` to a turn), not
    anybody's acoustic model.
    """

    name = STUDIO_STT_NAME

    def __init__(self, **_options: Any) -> None:
        pass

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]:
        async for frame in frames:
            if frame.pcm:
                yield STTResult(text=SCRIPTED_TRANSCRIPT, final=True, language=language)


@register("tts", STUDIO_TTS_NAME)
class ScriptedStudioTTS(TTSProvider):
    """Emits one small frame of audio per utterance, at a non-default rate.

    24 kHz on purpose: several real engines synthesize above the 16 kHz wire
    rate, and the studio has to tell the browser which one it is getting
    rather than letting it assume.
    """

    name = STUDIO_TTS_NAME

    def __init__(self, **_options: Any) -> None:
        pass

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        spoken = "".join([chunk async for chunk in text])
        if not spoken:
            return
        yield TTSChunk(
            frame=AudioFrame(pcm=SPOKEN_PCM, sample_rate=24000, channels=1),
            usage=[Usage(units=float(len(spoken)), unit_name="tts_chars")],
        )


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

STUDIO_AGENT_YAML = f"""
name: studio-test
persona: You are a terse front-desk assistant.
greeting: Front desk.
runtime:
  mode: cascade
  routing:
    default:
      stt: text_input
      llm: {STUDIO_LLM_NAME}
      tts: {SILENT_TTS_NAME}
"""


MIC_AGENT_YAML = f"""
name: studio-mic-test
persona: You are a terse front-desk assistant.
greeting: Front desk.
runtime:
  mode: cascade
  routing:
    default:
      stt: {STUDIO_STT_NAME}
      llm: {STUDIO_LLM_NAME}
      tts: {STUDIO_TTS_NAME}
"""


async def _serve(tmp_path: Path, **kwargs: Any) -> AsyncIterator[StudioServer]:
    # Session history goes under tmp_path, never under the developer's real
    # ~/.tring: a test suite that writes to a home directory is a test suite
    # that fails differently on the machine that has already run it.
    kwargs.setdefault("sessions_dir", tmp_path / "sessions")
    server = StudioServer(
        agent_path=tmp_path / "agent.yaml", host="127.0.0.1", port=0, **kwargs
    )
    await server.start()
    try:
        yield server
    finally:
        await server.close()


@pytest.fixture
async def server(tmp_path: Path) -> AsyncIterator[StudioServer]:
    """A studio with no agent file yet, serving the packaged frontend."""
    async for running in _serve(tmp_path):
        yield running


@pytest.fixture
async def agent_server(tmp_path: Path) -> AsyncIterator[StudioServer]:
    """A studio whose agent spec is the fake-provider one above."""
    (tmp_path / "agent.yaml").write_text(STUDIO_AGENT_YAML, encoding="utf-8")
    async for running in _serve(tmp_path):
        yield running


@pytest.fixture
async def mic_server(tmp_path: Path) -> AsyncIterator[StudioServer]:
    """A studio whose spec has a real (fake, but real-shaped) STT and TTS."""
    (tmp_path / "agent.yaml").write_text(MIC_AGENT_YAML, encoding="utf-8")
    async for running in _serve(tmp_path):
        yield running


async def _collect(
    socket: Any, until: str, limit: int = 60, timeout: float = 5.0
) -> list[dict[str, Any]]:
    """Read protocol messages until a SessionEvent of type ``until`` arrives.

    Binary frames (bot audio) are recorded as ``{"type": "audio", "pcm": ...}``
    so one helper drives both modes and every assertion reads off one list.
    """
    received: list[dict[str, Any]] = []
    async with asyncio.timeout(timeout):
        while len(received) < limit:
            frame = await socket.recv()
            if isinstance(frame, bytes):
                received.append({"type": "audio", "pcm": frame})
                continue
            message = json.loads(frame)
            received.append(message)
            if message.get("type") == "event" and message["event"]["type"] == until:
                return received
    raise AssertionError(f"never saw a {until!r} event; got {received!r}")


def _events(messages: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [
        m["event"]
        for m in messages
        if m.get("type") == "event" and m["event"]["type"] == event_type
    ]


def _events_of(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every forwarded SessionEvent, in order, with the envelope stripped."""
    return [m["event"] for m in messages if m.get("type") == "event"]


# ---------------------------------------------------------------------------
# HTTP: /api/meta
# ---------------------------------------------------------------------------


async def test_meta_reports_the_version_and_every_registry_slot(
    server: StudioServer,
) -> None:
    async with httpx.AsyncClient(base_url=server.url) as client:
        response = await client.get("/api/meta")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["version"] == __version__
    assert set(body["providers"]) == {"stt", "llm", "tts", "s2s"}
    # The built-ins the studio itself depends on must be discoverable, or the
    # editor's dropdowns cannot offer a working configuration.
    assert "text_input" in body["providers"]["stt"]
    assert SILENT_TTS_NAME in body["providers"]["tts"]


# ---------------------------------------------------------------------------
# HTTP: /api/agent
# ---------------------------------------------------------------------------


async def test_get_agent_serves_a_valid_demo_spec_before_any_file_exists(
    server: StudioServer,
) -> None:
    async with httpx.AsyncClient(base_url=server.url) as client:
        body = (await client.get("/api/agent")).json()

    import yaml

    from tring.agent import AgentSpec

    # The contract is "YAML text"; the demo is JSON, which is YAML, and which
    # is also what lets the frontend open its structured editor on first run.
    spec = AgentSpec.model_validate(yaml.safe_load(body["yaml"]))
    assert spec.name == DEMO_AGENT.name
    assert json.loads(body["yaml"])["name"] == DEMO_AGENT.name


async def test_get_agent_returns_an_existing_file_byte_for_byte(
    agent_server: StudioServer,
) -> None:
    async with httpx.AsyncClient(base_url=agent_server.url) as client:
        body = (await client.get("/api/agent")).json()

    assert body["yaml"] == STUDIO_AGENT_YAML


async def test_post_agent_validates_persists_and_round_trips(
    server: StudioServer,
) -> None:
    async with httpx.AsyncClient(base_url=server.url) as client:
        saved = await client.post("/api/agent", json={"yaml": STUDIO_AGENT_YAML})
        reloaded = (await client.get("/api/agent")).json()

    assert saved.json() == {"ok": True}
    assert server.agent_path.read_text(encoding="utf-8") == STUDIO_AGENT_YAML
    assert reloaded["yaml"] == STUDIO_AGENT_YAML


async def test_post_agent_rejects_invalid_yaml_with_ok_false(
    server: StudioServer,
) -> None:
    async with httpx.AsyncClient(base_url=server.url) as client:
        response = await client.post("/api/agent", json={"yaml": "name: [unclosed\n"})

    # HTTP 200 on purpose: the error text is UI copy, not a transport failure.
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is False
    assert "YAML" in body["error"]
    assert not server.agent_path.exists(), "a rejected spec must not be persisted"


async def test_post_agent_rejects_a_spec_the_runtime_could_not_load(
    server: StudioServer,
) -> None:
    async with httpx.AsyncClient(base_url=server.url) as client:
        response = await client.post("/api/agent", json={"yaml": "name: no-persona\n"})

    body = response.json()
    assert body["ok"] is False
    assert "persona" in body["error"]
    assert not server.agent_path.exists()


async def test_post_agent_rejects_a_malformed_envelope(server: StudioServer) -> None:
    async with httpx.AsyncClient(base_url=server.url) as client:
        missing_field = await client.post("/api/agent", json={"spec": "name: x"})
        not_json = await client.post(
            "/api/agent",
            content=b"not json at all",
            headers={"Content-Type": "application/json"},
        )

    assert missing_field.json()["ok"] is False
    assert not_json.json()["ok"] is False


# ---------------------------------------------------------------------------
# HTTP: static files
# ---------------------------------------------------------------------------


@pytest.fixture
async def bundle_server(tmp_path: Path) -> AsyncIterator[StudioServer]:
    """A studio serving a stand-in for the built React bundle."""
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>studio</title>", "utf-8")
    (static / "assets" / "app-a1b2c3.js").write_text("export const x = 1;\n", "utf-8")
    (static / "assets" / "app-a1b2c3.css").write_text(":root{color:red}\n", "utf-8")
    (static / "assets" / "app-a1b2c3.js.map").write_text('{"version":3}\n', "utf-8")
    (static / "logo.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", "utf-8")
    (tmp_path / "secret.txt").write_text("not for the browser", encoding="utf-8")

    async for running in _serve(tmp_path, static_dir=static):
        yield running


async def test_root_serves_index_html(bundle_server: StudioServer) -> None:
    async with httpx.AsyncClient(base_url=bundle_server.url) as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert "<title>studio</title>" in response.text


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/assets/app-a1b2c3.js", "text/javascript; charset=utf-8"),
        ("/assets/app-a1b2c3.css", "text/css; charset=utf-8"),
        ("/assets/app-a1b2c3.js.map", "application/json; charset=utf-8"),
        ("/logo.svg", "image/svg+xml"),
        ("/index.html", "text/html; charset=utf-8"),
    ],
)
async def test_bundle_files_are_served_with_correct_content_types(
    bundle_server: StudioServer, path: str, content_type: str
) -> None:
    async with httpx.AsyncClient(base_url=bundle_server.url) as client:
        response = await client.get(path)

    assert response.status_code == 200
    assert response.headers["content-type"] == content_type


@pytest.mark.parametrize(
    "path",
    [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/assets/../../secret.txt",
        "/assets/%2e%2e%2f%2e%2e%2fsecret.txt",
        "//etc/hosts",
    ],
)
async def test_paths_that_escape_the_static_root_are_refused(
    bundle_server: StudioServer, path: str
) -> None:
    # httpx would normalise `..` away client-side, so the raw request is built
    # by hand: traversal has to be tested with the bytes an attacker sends.
    reader, writer = await asyncio.open_connection("127.0.0.1", bundle_server.port)
    writer.write(f"GET {path} HTTP/1.1\r\nHost: studio\r\n\r\n".encode("latin-1"))
    await writer.drain()
    raw = await reader.read(4096)
    writer.close()

    assert raw.startswith(b"HTTP/1.1 404"), raw[:64]
    assert b"not for the browser" not in raw


async def test_missing_index_html_explains_that_the_frontend_is_not_built(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "nothing-built"
    empty.mkdir()
    async for running in _serve(tmp_path, static_dir=empty):
        async with httpx.AsyncClient(base_url=running.url) as client:
            response = await client.get("/")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert "frontend is not built" in response.text
        assert str(empty) in response.text


async def test_head_returns_the_headers_of_the_get_and_no_body(
    bundle_server: StudioServer,
) -> None:
    async with httpx.AsyncClient(base_url=bundle_server.url) as client:
        head = await client.head("/assets/app-a1b2c3.js")
        get = await client.get("/assets/app-a1b2c3.js")

    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == get.headers["content-length"]
    assert head.headers["content-type"] == get.headers["content-type"]


async def test_a_failing_handler_answers_500_instead_of_hanging_the_browser(
    bundle_server: StudioServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bug in the studio must still produce a response, not a dead socket."""

    def boom(self: StudioServer) -> dict[str, Any]:
        raise RuntimeError("simulated studio bug")

    monkeypatch.setattr(StudioServer, "meta", boom)
    async with httpx.AsyncClient(base_url=bundle_server.url) as client:
        response = await client.get("/api/meta")

    assert response.status_code == 500


async def test_unknown_paths_and_methods_are_refused(bundle_server: StudioServer) -> None:
    async with httpx.AsyncClient(base_url=bundle_server.url) as client:
        missing = await client.get("/assets/does-not-exist.js")
        wrong_method = await client.put("/api/agent", json={"yaml": "x"})

    assert missing.status_code == 404
    assert wrong_method.status_code == 405


# ---------------------------------------------------------------------------
# WebSocket: the live session
# ---------------------------------------------------------------------------


async def test_a_typed_turn_runs_the_whole_cascade_and_streams_its_events(
    agent_server: StudioServer,
) -> None:
    async with connect(f"ws://127.0.0.1:{agent_server.port}/ws") as socket:
        await socket.send(json.dumps({"type": "start"}))
        ready = json.loads(await socket.recv())

        assert ready["type"] == "ready"
        assert ready["session_id"]
        # Capabilities are declared, never assumed: the studio forwards the
        # runtime's own answer rather than describing the runtime itself.
        assert ready["capabilities"]["live_transcripts"] is True

        # Drain the greeting all the way through playback, so the next collect
        # cannot stop on a leftover event from it.
        greeting = await _collect(socket, until="bot_speech_played")
        assert _events(greeting, "session_started")[0]["agent_name"] == "studio-test"
        assert _events(greeting, "bot_utterance")[0]["text"] == "Front desk."

        await socket.send(json.dumps({"type": "user_text", "text": "when do you open?"}))
        turn = await _collect(socket, until="bot_speech_played")

        assert _events(turn, "user_transcript")[0]["text"] == "when do you open?"
        assert _events(turn, "bot_utterance")[0]["text"] == "We open at nine."

        # The silent TTS still meters: exact characters, measured not estimated.
        costs = _events(turn, "cost_recorded")
        spoken = [c for c in costs if c["unit_name"] == "tts_chars"]
        assert spoken, "studio_silent must still report tts_chars"
        assert spoken[0]["units"] == float(len("We open at nine."))
        assert spoken[0]["provider"] == SILENT_TTS_NAME


async def test_audio_progress_reports_counted_bytes_rather_than_pcm(
    agent_server: StudioServer,
) -> None:
    async with connect(f"ws://127.0.0.1:{agent_server.port}/ws") as socket:
        await socket.send(json.dumps({"type": "start"}))
        assert json.loads(await socket.recv())["type"] == "ready"
        messages = await _collect(socket, until="bot_speech_played")

    progress = [m for m in messages if m["type"] == "audio_progress"]
    assert progress, "the UI's speech meter has nothing to render without these"
    # studio_silent produces no PCM, so the honest cumulative count is zero --
    # and it is a number, not a stream of audio frames shipped to the browser.
    assert all(isinstance(m["bytes"], int) for m in progress)
    assert progress[-1]["bytes"] == 0


async def test_reset_stops_the_session_and_builds_a_new_one(
    agent_server: StudioServer,
) -> None:
    async with connect(f"ws://127.0.0.1:{agent_server.port}/ws") as socket:
        await socket.send(json.dumps({"type": "start"}))
        first = json.loads(await socket.recv())
        await _collect(socket, until="bot_utterance")

        await socket.send(json.dumps({"type": "reset"}))
        async with asyncio.timeout(5.0):
            while True:
                message = json.loads(await socket.recv())
                if message["type"] == "ready":
                    break

    assert message["session_id"] != first["session_id"]


async def test_an_unavailable_tts_provider_falls_back_and_says_why(
    tmp_path: Path,
) -> None:
    (tmp_path / "agent.yaml").write_text(
        STUDIO_AGENT_YAML.replace(f"tts: {SILENT_TTS_NAME}", "tts: no_such_engine"),
        encoding="utf-8",
    )
    async for running in _serve(tmp_path):
        async with connect(f"ws://127.0.0.1:{running.port}/ws") as socket:
            await socket.send(json.dumps({"type": "start"}))
            assert json.loads(await socket.recv())["type"] == "ready"
            messages = await _collect(socket, until="bot_utterance")

    notes = _events(messages, "error")
    assert notes, "the client must be told the configured TTS was swapped out"
    assert SILENT_TTS_NAME in notes[0]["message"]
    assert "no_such_engine" in notes[0]["message"]
    # And the session still ran to a spoken greeting rather than dying.
    assert _events(messages, "bot_utterance")[0]["text"] == "Front desk."


async def test_user_text_before_start_and_junk_frames_are_reported_not_fatal(
    agent_server: StudioServer,
) -> None:
    async with connect(f"ws://127.0.0.1:{agent_server.port}/ws") as socket:
        await socket.send(json.dumps({"type": "user_text", "text": "hello?"}))
        no_session = json.loads(await socket.recv())

        await socket.send("this is not json")
        bad_json = json.loads(await socket.recv())

        await socket.send(json.dumps({"type": "fly_to_the_moon"}))
        unknown = json.loads(await socket.recv())

        # Still usable afterwards: none of the above tore the socket down.
        await socket.send(json.dumps({"type": "start"}))
        ready = json.loads(await socket.recv())

    assert no_session["type"] == "error" and "start" in no_session["message"]
    assert bad_json["type"] == "error" and "JSON" in bad_json["message"]
    assert unknown["type"] == "error" and "fly_to_the_moon" in unknown["message"]
    assert ready["type"] == "ready"


async def test_disconnecting_tears_the_session_down_without_leaking_tasks(
    agent_server: StudioServer,
) -> None:
    baseline = len(asyncio.all_tasks())

    for _ in range(3):
        async with connect(f"ws://127.0.0.1:{agent_server.port}/ws") as socket:
            await socket.send(json.dumps({"type": "start"}))
            assert json.loads(await socket.recv())["type"] == "ready"
            await _collect(socket, until="bot_utterance")

    # The runtime's STT task, the TTS pump, the event forwarder and the
    # connection handler are all created per session; a disconnect that leaves
    # any of them running is a leak that only shows up under a real UI.
    async with asyncio.timeout(5.0):
        while len(asyncio.all_tasks()) > baseline:
            await asyncio.sleep(0.01)

    assert len(asyncio.all_tasks()) <= baseline


async def test_a_websocket_upgrade_to_the_wrong_path_is_refused(
    agent_server: StudioServer,
) -> None:
    from websockets.exceptions import InvalidStatus

    with pytest.raises(InvalidStatus):
        async with connect(f"ws://127.0.0.1:{agent_server.port}/nope"):
            pass  # pragma: no cover - the connect above must raise


# ---------------------------------------------------------------------------
# v0.4: session persistence and the sessions endpoints
# ---------------------------------------------------------------------------


async def _await_sessions(server: StudioServer, timeout: float = 5.0) -> list[dict[str, Any]]:
    """Poll ``GET /api/sessions`` until the indexed session shows up.

    A session is indexed during the server's own teardown, which runs after
    the client's ``close()`` returns; polling is the honest way to wait for
    the other side of a socket without reaching into the server's internals.
    """
    async with httpx.AsyncClient(base_url=server.url) as client:
        async with asyncio.timeout(timeout):
            while True:
                listed = (await client.get("/api/sessions")).json()
                if listed:
                    return list(listed)
                await asyncio.sleep(0.02)


async def test_a_finished_session_is_listed_and_replayable(
    agent_server: StudioServer,
) -> None:
    async with connect(f"ws://127.0.0.1:{agent_server.port}/ws") as socket:
        await socket.send(json.dumps({"type": "start"}))
        session_id = json.loads(await socket.recv())["session_id"]
        await _collect(socket, until="bot_speech_played")
        await socket.send(json.dumps({"type": "user_text", "text": "when do you open?"}))
        await _collect(socket, until="bot_speech_played")

    listed = await _await_sessions(agent_server)

    assert [row["id"] for row in listed] == [session_id]
    summary = listed[0]
    assert summary["agent"] == "studio-test"
    assert summary["turns"] == 1, "one user turn, counted off the event stream"
    assert isinstance(summary["cost"], float)
    assert summary["started"].startswith("20")  # ISO-8601, UTC, set on the first event

    async with httpx.AsyncClient(base_url=agent_server.url) as client:
        replay = (await client.get(f"/api/sessions/{session_id}")).json()

    kinds = [event["type"] for event in replay["events"]]
    assert kinds[0] == "session_started"
    # The tail matters most: these are emitted after the event forwarder has
    # been cancelled, so they only reach the file if teardown syncs the rest.
    assert kinds[-1] == "session_ended"
    assert replay["turns"] == summary["turns"]
    assert [e["text"] for e in replay["events"] if e["type"] == "bot_utterance"] == [
        "Front desk.",
        "We open at nine.",
    ]


async def test_the_stored_session_is_the_event_stream_the_socket_showed(
    agent_server: StudioServer,
) -> None:
    async with connect(f"ws://127.0.0.1:{agent_server.port}/ws") as socket:
        await socket.send(json.dumps({"type": "start"}))
        session_id = json.loads(await socket.recv())["session_id"]
        live = _events_of(await _collect(socket, until="bot_speech_played"))

    await _await_sessions(agent_server)
    async with httpx.AsyncClient(base_url=agent_server.url) as client:
        stored = (await client.get(f"/api/sessions/{session_id}")).json()["events"]

    # Byte for byte the same dumps, in the same order: replay is a cursor over
    # stored events, so anything the file drops is a panel the UI cannot draw.
    assert stored[: len(live)] == live


async def test_an_unknown_or_unsafe_session_id_is_a_404(server: StudioServer) -> None:
    async with httpx.AsyncClient(base_url=server.url) as client:
        listed = await client.get("/api/sessions")
        missing = await client.get("/api/sessions/deadbeef")
        traversal = await client.get("/api/sessions/..%2f..%2fagent.yaml")

    assert listed.json() == []
    assert missing.status_code == 404
    assert traversal.status_code == 404


# ---------------------------------------------------------------------------
# v0.4: POST /api/flow/compile
# ---------------------------------------------------------------------------

SMALL_FLOW: dict[str, Any] = {
    "nodes": [
        {"id": "welcome", "kind": "say", "text": "Front desk."},
        {
            "id": "ask_name",
            "kind": "ask",
            "prompt": "Who am I speaking with?",
            "slots": [{"name": "full_name", "description": "the caller's full name"}],
        },
        {"id": "bye", "kind": "end", "text": "Thanks, goodbye."},
    ],
    "edges": [
        {"from": "welcome", "to": "ask_name"},
        {"from": "ask_name", "to": "bye"},
    ],
}


async def test_flow_compile_returns_a_spec_without_saving_it(
    agent_server: StudioServer,
) -> None:
    before = agent_server.agent_path.read_text(encoding="utf-8")

    async with httpx.AsyncClient(base_url=agent_server.url) as client:
        body = (await client.post("/api/flow/compile", json={"flow": SMALL_FLOW})).json()

    assert body["ok"] is True
    compiled = AgentSpec.model_validate(yaml.safe_load(body["yaml"]))
    assert "1. SAY (welcome)" in compiled.persona
    assert compiled.metadata["flow"]["nodes"][0]["id"] == "welcome"
    # The saved spec supplies what a flow has no opinion about...
    assert compiled.runtime.routing["default"].llm == STUDIO_LLM_NAME
    # ...and compiling is not saving.
    assert agent_server.agent_path.read_text(encoding="utf-8") == before


async def test_flow_compile_reports_a_broken_graph_as_ui_copy(
    agent_server: StudioServer,
) -> None:
    broken = {
        "nodes": [
            {"id": "welcome", "kind": "say"},  # no text
            {"id": "bye", "kind": "end"},
        ],
        "edges": [{"from": "welcome", "to": "bye"}],
    }

    async with httpx.AsyncClient(base_url=agent_server.url) as client:
        rejected = (await client.post("/api/flow/compile", json={"flow": broken})).json()
        malformed = (await client.post("/api/flow/compile", json={"nodes": []})).json()

    # HTTP 200 with ok:false, exactly like POST /api/agent: the text is copy
    # the canvas shows next to the offending node.
    assert rejected["ok"] is False
    assert "node 'welcome'" in rejected["error"]
    assert "'text'" in rejected["error"]
    assert malformed["ok"] is False


# ---------------------------------------------------------------------------
# v0.4: microphone mode
# ---------------------------------------------------------------------------


class _RecordingRuntime:
    """A stand-in runtime that records the frames the bridge pushes into it.

    The binary path is worth isolating from the cascade: what this test asks
    is "did the exact bytes of one binary frame reach ``push_audio`` with the
    wire format the protocol promises", and a real runtime would answer that
    question through three providers and a turn loop.
    """

    def __init__(self, session: Any, **_kwargs: Any) -> None:
        self.session = session
        self.on_bot_audio: Any = None
        self.frames: list[AudioFrame] = []

    @property
    def capabilities(self) -> Any:
        from tring.runtimes.base import RuntimeCapabilities

        return RuntimeCapabilities(
            live_transcripts=True,
            mid_call_tool_calls=False,
            barge_in=False,
            exact_usage_reporting=False,
        )

    async def start(self) -> None:
        pass

    async def push_audio(self, frame: AudioFrame) -> None:
        from tring.events import UserTranscript

        self.frames.append(frame)
        # Echoed back as an event so the test can wait on the socket instead
        # of sleeping: the transcript names the frame it came from.
        self.session.emit(
            UserTranscript(
                session_id=self.session.session_id,
                at=self.session.elapsed,
                text=f"{len(frame.pcm)} bytes at {frame.sample_rate} Hz",
            )
        )

    async def stop(self, reason: str = "completed") -> None:
        pass


async def test_a_binary_frame_reaches_push_audio_as_16k_mono_pcm(
    mic_server: StudioServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tring.studio import server as server_module

    recorded: list[_RecordingRuntime] = []

    def build(session: Any, **kwargs: Any) -> _RecordingRuntime:
        runtime = _RecordingRuntime(session, **kwargs)
        recorded.append(runtime)
        return runtime

    monkeypatch.setattr(server_module, "CascadeRuntime", build)

    async with connect(f"ws://127.0.0.1:{mic_server.port}/ws") as socket:
        await socket.send(json.dumps({"type": "mode", "audio": True}))
        assert json.loads(await socket.recv()) == {"type": "mode", "audio": True, "stt": None}

        await socket.send(json.dumps({"type": "start"}))
        ready = json.loads(await socket.recv())
        assert ready["mode"] == {"audio": True, "stt": STUDIO_STT_NAME}

        await socket.send(b"\x00\x01" * 160)  # 10 ms of 16 kHz mono PCM
        heard = await _collect(socket, until="user_transcript")

    assert _events(heard, "user_transcript")[0]["text"] == "320 bytes at 16000 Hz"
    assert recorded[0].frames[0].pcm == b"\x00\x01" * 160
    assert recorded[0].frames[0].channels == 1


async def test_microphone_mode_runs_a_real_turn_and_streams_bot_audio_back(
    mic_server: StudioServer,
) -> None:
    async with connect(f"ws://127.0.0.1:{mic_server.port}/ws") as socket:
        await socket.send(json.dumps({"type": "mode", "audio": True}))
        await socket.recv()
        await socket.send(json.dumps({"type": "start"}))
        assert json.loads(await socket.recv())["type"] == "ready"
        greeting = await _collect(socket, until="bot_speech_played")

        await socket.send(b"\x00\x01" * 160)
        turn = await _collect(socket, until="bot_speech_played")

    # The browser cannot play PCM without being told the rate, and this engine
    # is not at the wire rate, so the format is announced before the audio.
    formats = [m for m in greeting if m["type"] == "audio_format"]
    assert formats[0] == {
        "type": "audio_format",
        "sample_rate": 24000,
        "channels": 1,
        "encoding": "pcm_s16le",
    }
    assert [m["pcm"] for m in greeting if m["type"] == "audio"] == [SPOKEN_PCM]
    # Announced once, not once per frame.
    assert [m for m in turn if m["type"] == "audio_format"] == []

    assert _events(turn, "user_transcript")[0]["text"] == SCRIPTED_TRANSCRIPT
    assert _events(turn, "bot_utterance")[0]["text"] == "We open at nine."
    assert [m["pcm"] for m in turn if m["type"] == "audio"] == [SPOKEN_PCM]
    # audio_progress survives alongside the frames, still coalesced and still
    # cumulative: the speech meter works the same in both modes.
    progress = [m for m in greeting if m["type"] == "audio_progress"]
    assert progress and progress[-1]["bytes"] == len(SPOKEN_PCM)


@pytest.mark.parametrize(
    ("configured", "fragment"),
    [
        pytest.param("no_such_engine", "no stt provider named", id="unknown-engine"),
        pytest.param("text_input", "not a speech recognizer", id="typed-text-stand-in"),
    ],
)
async def test_microphone_mode_falls_back_to_text_and_says_why(
    tmp_path: Path, configured: str, fragment: str
) -> None:
    (tmp_path / "agent.yaml").write_text(
        MIC_AGENT_YAML.replace(f"stt: {STUDIO_STT_NAME}", f"stt: {configured}"),
        encoding="utf-8",
    )

    async for running in _serve(tmp_path):
        async with connect(f"ws://127.0.0.1:{running.port}/ws") as socket:
            await socket.send(json.dumps({"type": "mode", "audio": True}))
            await socket.recv()
            await socket.send(json.dumps({"type": "start"}))
            ready = json.loads(await socket.recv())
            messages = await _collect(socket, until="bot_utterance")

            # A frame sent anyway is dropped rather than decoded as text, and
            # the client is told once instead of once per 20 ms of speech.
            await socket.send(b"\x00\x01" * 160)
            await socket.send(b"\x00\x01" * 160)
            await socket.send(json.dumps({"type": "user_text", "text": "typed instead"}))
            after = await _collect(socket, until="user_transcript")

    assert ready["mode"] == {"audio": False, "stt": "text_input"}
    notes = [e["message"].lower() for e in _events(messages, "error")]
    assert any(fragment in note for note in notes), notes
    assert any("falling back to 'text_input'" in note for note in notes)

    dropped = [m for m in after if m["type"] == "error"]
    assert len(dropped) == 1, "one warning per session, not one per frame"
    assert _events(after, "user_transcript")[0]["text"] == "typed instead"


async def test_switching_mode_mid_session_rebuilds_it(mic_server: StudioServer) -> None:
    async with connect(f"ws://127.0.0.1:{mic_server.port}/ws") as socket:
        await socket.send(json.dumps({"type": "start"}))
        typed = json.loads(await socket.recv())
        await _collect(socket, until="bot_utterance")

        await socket.send(json.dumps({"type": "mode", "audio": True}))
        rebuilt = json.loads(await socket.recv())
        while rebuilt["type"] != "ready":  # pragma: no cover - ready comes first
            rebuilt = json.loads(await socket.recv())

    # The STT slot is bound when the runtime starts, so switching means a new
    # session, announced exactly the way `reset` announces one.
    assert typed["mode"] == {"audio": False, "stt": "text_input"}
    assert rebuilt["mode"] == {"audio": True, "stt": STUDIO_STT_NAME}
    assert rebuilt["session_id"] != typed["session_id"]


async def test_a_mode_message_without_a_boolean_is_reported_not_fatal(
    agent_server: StudioServer,
) -> None:
    async with connect(f"ws://127.0.0.1:{agent_server.port}/ws") as socket:
        await socket.send(json.dumps({"type": "mode", "audio": "yes please"}))
        complaint = json.loads(await socket.recv())

        await socket.send(json.dumps({"type": "start"}))
        ready = json.loads(await socket.recv())

    assert complaint["type"] == "error" and "audio" in complaint["message"]
    assert ready["type"] == "ready"
