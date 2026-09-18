"""Tests for S2SRuntime, HybridRuntime, and the cloud provider adapters.

No network, no API keys, no GPU: the S2S/S2S-adjacent tests use a scripted
fake ``S2SProvider``; the LLM test drives ``httpx.MockTransport`` instead of
a real endpoint.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from tring.agent import AgentSpec, ProviderSelection, RuntimeConfig, RuntimeMode, ToolDef
from tring.events import (
    BotUtterance,
    CostRecorded,
    SessionEnded,
    SessionStarted,
    ToolCallCompleted,
    ToolCallStarted,
    TranscriptAvailability,
    UserTranscript,
)
from tring.providers.base import S2SProvider, TTSChunk, Usage
from tring.runtimes.base import AudioFrame
from tring.runtimes.hybrid import HybridRuntime
from tring.runtimes.s2s import S2SRuntime
from tring.session import CallSession


def _agent(mode: RuntimeMode = RuntimeMode.S2S) -> AgentSpec:
    return AgentSpec(
        name="test-agent",
        persona="You are a test agent.",
        runtime=RuntimeConfig(
            mode=mode,
            routing={"default": ProviderSelection(s2s="fake")},
        ),
    )


class FakeS2SProvider(S2SProvider):
    """A scripted S2SProvider: yields two audio chunks (one with usage),
    then a fixed post-call transcript. No provider under test ever touches
    real audio bytes or a network socket."""

    name = "fake-s2s"

    def __init__(self) -> None:
        self.frames_received: list[AudioFrame] = []
        self.post_call_calls = 0

    async def converse(
        self, frames: AsyncIterator[AudioFrame]
    ) -> AsyncIterator[TTSChunk]:
        async for frame in frames:
            self.frames_received.append(frame)
        yield TTSChunk(frame=AudioFrame(pcm=b"bot-audio-1"))
        yield TTSChunk(
            frame=AudioFrame(pcm=b"bot-audio-2"),
            usage=[Usage(units=3.5, unit_name="audio_seconds", estimated=False)],
        )

    async def post_call_transcript(self) -> list[dict]:
        self.post_call_calls += 1
        return [
            {"role": "user", "text": "hello there", "language": "en"},
            {"role": "bot", "text": "hi, how can I help?", "language": "en"},
        ]


@pytest.mark.asyncio
async def test_s2s_runtime_declares_no_live_transcripts() -> None:
    session = CallSession(agent=_agent())
    runtime = S2SRuntime(session, provider=FakeS2SProvider())
    caps = runtime.capabilities
    assert caps.live_transcripts is False
    assert caps.mid_call_tool_calls is True
    assert caps.barge_in is True
    assert caps.exact_usage_reporting is False
    assert caps.local_capable is False


@pytest.mark.asyncio
async def test_s2s_runtime_full_session_event_order_and_post_call_transcript() -> None:
    session = CallSession(agent=_agent())
    provider = FakeS2SProvider()
    runtime = S2SRuntime(session, provider=provider)

    received_audio: list[bytes] = []
    runtime.on_bot_audio = lambda frame: received_audio.append(frame.pcm)  # type: ignore[assignment]

    await runtime.start()
    await runtime.push_audio(AudioFrame(pcm=b"caller-audio"))
    await runtime.stop(reason="completed")

    assert provider.frames_received == [AudioFrame(pcm=b"caller-audio")]
    assert received_audio == [b"bot-audio-1", b"bot-audio-2"]
    assert provider.post_call_calls == 1

    history = session.history
    types = [type(e) for e in history]

    # SessionStarted first, SessionEnded last -- everything else in between.
    assert types[0] is SessionStarted
    assert types[-1] is SessionEnded
    assert history[0].runtime_mode == "s2s"  # type: ignore[union-attr]

    cost_events = [e for e in history if isinstance(e, CostRecorded)]
    assert len(cost_events) == 1
    assert cost_events[0].units == 3.5
    assert cost_events[0].unit_name == "audio_seconds"
    assert cost_events[0].estimated is False
    assert cost_events[0].amount == 0.0  # pricing is cost/rates.py's job, not ours

    user_transcripts = [e for e in history if isinstance(e, UserTranscript)]
    assert len(user_transcripts) == 1
    assert user_transcripts[0].text == "hello there"
    assert user_transcripts[0].availability is TranscriptAvailability.POST_CALL

    bot_utterances = [e for e in history if isinstance(e, BotUtterance)]
    assert len(bot_utterances) == 1
    assert bot_utterances[0].text == "hi, how can I help?"

    # Post-call transcript events land before SessionEnded, per contract.
    assert history.index(user_transcripts[0]) < history.index(history[-1])
    assert history.index(bot_utterances[0]) < history.index(history[-1])
    assert types.count(SessionStarted) == 1
    assert types.count(SessionEnded) == 1


@pytest.mark.asyncio
async def test_s2s_runtime_stop_is_idempotent() -> None:
    session = CallSession(agent=_agent())
    runtime = S2SRuntime(session, provider=FakeS2SProvider())
    await runtime.start()
    await runtime.stop()
    ended_count_before = sum(1 for e in session.history if isinstance(e, SessionEnded))
    await runtime.stop()  # should be a no-op, not double-emit
    ended_count_after = sum(1 for e in session.history if isinstance(e, SessionEnded))
    assert ended_count_before == ended_count_after == 1


@pytest.mark.asyncio
async def test_s2s_runtime_resolves_provider_from_routing() -> None:
    """When no provider is injected, S2SRuntime resolves one from the
    registry using the spec's routing -- the real (non-test) path."""
    from tring.providers import registry

    registry.register("s2s", "fake-registered")(lambda **_: FakeS2SProvider())
    session = CallSession(
        agent=AgentSpec(
            name="a",
            persona="p",
            runtime=RuntimeConfig(
                routing={"default": ProviderSelection(s2s="fake-registered")}
            ),
        )
    )
    runtime = S2SRuntime(session)
    await runtime.start()
    await runtime.stop()
    assert any(isinstance(e, SessionStarted) for e in session.history)


@pytest.mark.asyncio
async def test_s2s_runtime_missing_routing_raises() -> None:
    session = CallSession(
        agent=AgentSpec(name="a", persona="p")  # default routing has no s2s slot
    )
    runtime = S2SRuntime(session)
    with pytest.raises(ValueError, match="routing"):
        await runtime.start()


# ---------------------------------------------------------------------------
# HybridRuntime
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hybrid_runtime_delegates_audio_path_to_s2s() -> None:
    session = CallSession(agent=_agent(RuntimeMode.HYBRID))
    provider = FakeS2SProvider()
    # HybridRuntime always builds its own inner S2SRuntime; reach in to swap
    # the provider the same way a real deployment would via routing.
    runtime = HybridRuntime(session)
    runtime._s2s._provider = provider  # test seam: inject the fake directly

    received: list[bytes] = []
    runtime.on_bot_audio = lambda frame: received.append(frame.pcm)  # type: ignore[assignment]

    await runtime.start()
    await runtime.push_audio(AudioFrame(pcm=b"caller-audio"))
    await runtime.stop()

    assert received == [b"bot-audio-1", b"bot-audio-2"]
    assert any(
        isinstance(e, SessionStarted) and e.runtime_mode == "hybrid" for e in session.history
    )
    assert runtime.capabilities.live_transcripts is False
    assert runtime.capabilities.mid_call_tool_calls is True


def _agent_with_tool(tool: ToolDef) -> AgentSpec:
    return AgentSpec(
        name="test-agent",
        persona="You are a test agent.",
        runtime=RuntimeConfig(
            mode=RuntimeMode.HYBRID,
            routing={"default": ProviderSelection(s2s="fake")},
        ),
        tools=[tool],
    )


_BOOK_SLOT_TOOL = ToolDef(
    name="book_slot",
    description="Book a slot on a given date.",
    parameters={
        "type": "object",
        "properties": {"date": {"type": "string"}},
        "required": ["date"],
    },
)


@pytest.mark.asyncio
async def test_hybrid_tool_call_hook_invokes_handler_and_returns_choreographed_outcome() -> (
    None
):
    session = CallSession(agent=_agent_with_tool(_BOOK_SLOT_TOOL))
    calls: list[dict] = []

    async def book_slot(args: dict) -> dict:
        calls.append(args)
        return {"confirmation_id": "abc123"}

    runtime = HybridRuntime(session, handlers={"book_slot": book_slot})

    outcome = await runtime.on_provider_tool_call(
        "book_slot",
        {
            "date": "2026-09-20",
            "waiting_message": "One moment...",
            "spoken_mode": "answer_pending",
            "post_tool_response": "respond",
        },
    )

    assert outcome["ok"] is True
    assert outcome["result"] == {"confirmation_id": "abc123"}
    assert outcome["should_respond"] is True
    # Choreography keys were stripped before reaching the handler -- it only
    # ever sees the tool's own declared parameters.
    assert calls == [{"date": "2026-09-20"}]

    started = [e for e in session.history if isinstance(e, ToolCallStarted)]
    completed = [e for e in session.history if isinstance(e, ToolCallCompleted)]
    assert len(started) == 1 and len(completed) == 1
    assert started[0].tool_name == "book_slot"
    assert started[0].waiting_message == "One moment..."
    assert started[0].spoken_mode == "answer_pending"
    assert started[0].arguments == {"date": "2026-09-20"}
    assert completed[0].ok is True
    assert completed[0].call_id == started[0].call_id


@pytest.mark.asyncio
async def test_hybrid_tool_call_hook_unknown_tool_reports_failure() -> None:
    session = CallSession(agent=_agent_with_tool(_BOOK_SLOT_TOOL))
    runtime = HybridRuntime(session, handlers={})

    outcome = await runtime.on_provider_tool_call("unknown_tool", {"x": 1})

    assert outcome["ok"] is False
    assert "unknown_tool" in outcome["error"]
    completed = [e for e in session.history if isinstance(e, ToolCallCompleted)]
    assert completed[0].ok is False


@pytest.mark.asyncio
async def test_hybrid_tool_call_hook_declared_tool_without_bound_handler_reports_failure() -> (
    None
):
    session = CallSession(agent=_agent_with_tool(_BOOK_SLOT_TOOL))
    # Tool is declared on the agent, but nothing bound it to a handler.
    runtime = HybridRuntime(session, handlers={})

    outcome = await runtime.on_provider_tool_call(
        "book_slot",
        {
            "date": "2026-09-20",
            "waiting_message": "One moment...",
            "spoken_mode": "answer_pending",
            "post_tool_response": "respond",
        },
    )

    assert outcome["ok"] is False
    assert "no handler bound" in outcome["error"]


@pytest.mark.asyncio
async def test_hybrid_tool_call_hook_missing_choreography_field_is_rejected() -> None:
    session = CallSession(agent=_agent_with_tool(_BOOK_SLOT_TOOL))
    ran = False

    async def book_slot(args: dict) -> dict:
        nonlocal ran
        ran = True
        return {}

    runtime = HybridRuntime(session, handlers={"book_slot": book_slot})

    # post_tool_response is missing -- a schema violation, not a runtime error.
    outcome = await runtime.on_provider_tool_call(
        "book_slot",
        {
            "date": "2026-09-20",
            "waiting_message": "One moment...",
            "spoken_mode": "answer_pending",
        },
    )

    assert outcome["ok"] is False
    assert "post_tool_response" in outcome["error"]
    assert ran is False  # the handler must never run against an unvalidated call


@pytest.mark.asyncio
async def test_hybrid_tool_call_hook_handler_exception_is_contained_and_forces_response() -> (
    None
):
    session = CallSession(agent=_agent_with_tool(_BOOK_SLOT_TOOL))

    async def book_slot(args: dict) -> dict:
        raise RuntimeError("downstream API is down")

    runtime = HybridRuntime(session, handlers={"book_slot": book_slot})
    outcome = await runtime.on_provider_tool_call(
        "book_slot",
        {
            "date": "2026-09-20",
            "waiting_message": "One moment...",
            "spoken_mode": "answer_pending",
            # The model planned to stay silent, believing the tool would
            # succeed -- choreography.execute forces should_respond=True
            # anyway once it actually fails.
            "post_tool_response": "silent",
        },
    )

    assert outcome["ok"] is False
    assert "downstream API is down" in outcome["error"]
    assert outcome["should_respond"] is True
    completed = [e for e in session.history if isinstance(e, ToolCallCompleted)]
    assert completed[0].ok is False


@pytest.mark.asyncio
async def test_hybrid_tool_call_hook_non_choreographed_tool_skips_the_three_fields() -> None:
    """A tool with choreographed=False is a pure pass-through: no
    waiting_message / spoken_mode / post_tool_response required, matching
    primitives.choreography's own opt-out."""
    tool = ToolDef(
        name="log_event",
        description="Log an event locally; no latency worth narrating.",
        parameters={"type": "object", "properties": {"event": {"type": "string"}}},
        choreographed=False,
    )
    session = CallSession(agent=_agent_with_tool(tool))

    async def log_event(args: dict) -> dict:
        return {"logged": args["event"]}

    runtime = HybridRuntime(session, handlers={"log_event": log_event})
    outcome = await runtime.on_provider_tool_call("log_event", {"event": "call_started"})

    assert outcome["ok"] is True
    assert outcome["result"] == {"logged": "call_started"}


# ---------------------------------------------------------------------------
# providers/cloud
# ---------------------------------------------------------------------------


def test_cloud_module_registers_expected_provider_names() -> None:
    import tring.providers.cloud  # noqa: F401 -- import is the point of the test
    from tring.providers import registry

    available = registry.available()
    assert ("stt", "deepgram") in available
    assert ("tts", "elevenlabs") in available
    assert ("llm", "openai_compatible") in available


@pytest.mark.asyncio
async def test_openai_compatible_llm_builds_request_and_parses_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tring.providers.cloud import OpenAICompatibleLLM

    monkeypatch.setenv("MY_LLM_KEY", "sk-test-123")

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)

        sse_body = (
            'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
            'data: {"choices":[{"delta":{}}],'
            '"usage":{"prompt_tokens":10,"completion_tokens":2,'
            '"prompt_tokens_details":{"cached_tokens":4}}}\n\n'
            "data: [DONE]\n\n"
        )
        return httpx.Response(200, content=sse_body.encode())

    transport = httpx.MockTransport(handler)
    llm = OpenAICompatibleLLM(
        base_url="https://example.test/v1",
        model="test-model",
        api_key_env="MY_LLM_KEY",
        transport=transport,
    )

    chunks = []
    async for chunk in llm.generate([{"role": "user", "content": "hi"}]):
        chunks.append(chunk)

    assert captured["method"] == "POST"
    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"]["authorization"] == "Bearer sk-test-123"

    text_so_far = "".join(c.text for c in chunks)
    assert text_so_far == "Hello"

    final = chunks[-1]
    assert final.finish is True
    usage_by_name = {u.unit_name: u for u in final.usage}
    assert usage_by_name["tokens_in"].units == 10
    assert usage_by_name["tokens_in"].estimated is False
    assert usage_by_name["tokens_in"].cached_units == 4
    assert usage_by_name["tokens_out"].units == 2


@pytest.mark.asyncio
async def test_openai_compatible_llm_without_api_key_env_omits_auth_header() -> None:
    from tring.providers.cloud import OpenAICompatibleLLM

    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, content=b"data: [DONE]\n\n")

    llm = OpenAICompatibleLLM(
        base_url="https://example.test/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )
    async for _ in llm.generate([{"role": "user", "content": "hi"}]):
        pass

    assert "authorization" not in captured["headers"]


def test_openai_compatible_llm_missing_env_var_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tring.providers.cloud import OpenAICompatibleLLM

    monkeypatch.delenv("SOME_MISSING_KEY", raising=False)
    llm = OpenAICompatibleLLM(
        base_url="https://example.test/v1", model="m", api_key_env="SOME_MISSING_KEY"
    )
    with pytest.raises(RuntimeError, match="SOME_MISSING_KEY"):
        llm._headers()
