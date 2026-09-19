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

pytest.importorskip("websockets", reason="tring[transports] is not installed")

from websockets.asyncio.client import connect

from tring import __version__
from tring.providers.base import LLMChunk, LLMProvider
from tring.providers.registry import register
from tring.studio.server import DEMO_AGENT, StudioServer
from tring.studio.silent_tts import SILENT_TTS_NAME

# ---------------------------------------------------------------------------
# A fake LLM, registered in the real registry so specs resolve it normally
# ---------------------------------------------------------------------------

STUDIO_LLM_NAME = "test_studio_llm"

#: What the fake says, in the envelope the cascade runtime asks for.
SCRIPTED_REPLY = '{"speak": "We open at nine.", "tool_call": null}'


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


async def _serve(tmp_path: Path, **kwargs: Any) -> AsyncIterator[StudioServer]:
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


async def _collect(
    socket: Any, until: str, limit: int = 60, timeout: float = 5.0
) -> list[dict[str, Any]]:
    """Read protocol messages until a SessionEvent of type ``until`` arrives."""
    received: list[dict[str, Any]] = []
    async with asyncio.timeout(timeout):
        while len(received) < limit:
            message = json.loads(await socket.recv())
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
