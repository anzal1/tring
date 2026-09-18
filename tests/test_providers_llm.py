"""Tests for tring.providers.cloud.llm_extra: AnthropicLLM and GeminiLLM.

No network, no API keys: every HTTP-shaped assertion drives the request
through an ``httpx.MockTransport`` seam and asserts on the exact URL,
headers, and payload built, plus the ``LLMChunk``/``Usage`` parsed back out
of a scripted SSE body -- the same house pattern as
``tests/test_cost.py``'s ``OpenAICompatibleLLM`` tests.
"""

from __future__ import annotations

import httpx
import pytest

from tring.cost.rates import CostComponent, RateCard
from tring.providers.cloud.llm_extra import LLM_RATES, AnthropicLLM, GeminiLLM


def test_llm_extra_module_registers_expected_provider_names() -> None:
    import tring.providers.cloud  # noqa: F401 -- import is the point of the test
    from tring.providers import registry

    available = registry.available()
    assert ("llm", "anthropic") in available
    assert ("llm", "gemini") in available


# ---------------------------------------------------------------------------
# AnthropicLLM
# ---------------------------------------------------------------------------


_ANTHROPIC_SSE_BODY = (
    'event: message_start\n'
    'data: {"type":"message_start","message":{"id":"msg_1","type":"message",'
    '"role":"assistant","model":"claude-sonnet-5","content":[],'
    '"stop_reason":null,"usage":{"input_tokens":25,"output_tokens":1}}}\n\n'
    'event: content_block_start\n'
    'data: {"type":"content_block_start","index":0,'
    '"content_block":{"type":"text","text":""}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"Hel"}}\n\n'
    'event: content_block_delta\n'
    'data: {"type":"content_block_delta","index":0,'
    '"delta":{"type":"text_delta","text":"lo"}}\n\n'
    'event: content_block_stop\n'
    'data: {"type":"content_block_stop","index":0}\n\n'
    'event: message_delta\n'
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
    '"usage":{"input_tokens":25,"output_tokens":15,"cache_read_input_tokens":10}}\n\n'
    'event: message_stop\n'
    'data: {"type":"message_stop"}\n\n'
)


@pytest.mark.asyncio
async def test_anthropic_llm_builds_request_and_streams_text_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_ANTHROPIC_KEY", "sk-ant-test")
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = __import__("json").loads(request.content)
        return httpx.Response(200, content=_ANTHROPIC_SSE_BODY.encode())

    llm = AnthropicLLM(
        model="claude-sonnet-5",
        api_key_env="MY_ANTHROPIC_KEY",
        transport=httpx.MockTransport(handler),
    )

    chunks = []
    async for chunk in llm.generate(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "hi"},
        ]
    ):
        chunks.append(chunk)

    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    assert captured["headers"]["x-api-key"] == "sk-ant-test"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"

    body = captured["json"]
    assert body["model"] == "claude-sonnet-5"
    assert body["stream"] is True
    assert body["system"] == "You are terse."
    # The system turn was pulled out -- it must not also linger in `messages`.
    assert body["messages"] == [{"role": "user", "content": "hi"}]

    # Text deltas are yielded as their own chunks, not buffered until the
    # end -- this is the whole point of the streaming contract.
    text_chunks = [c for c in chunks if c.text]
    assert [c.text for c in text_chunks] == ["Hel", "lo"]
    assert all(c.finish is False for c in text_chunks)

    final = chunks[-1]
    assert final.finish is True
    usage_by_name = {u.unit_name: u for u in final.usage}
    assert usage_by_name["tokens_in"].units == 25
    assert usage_by_name["tokens_in"].estimated is False
    assert usage_by_name["tokens_in"].cached_units == 10
    assert usage_by_name["tokens_out"].units == 15
    assert usage_by_name["tokens_out"].cached_units is None


def test_anthropic_llm_missing_env_var_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOME_MISSING_ANTHROPIC_KEY", raising=False)
    llm = AnthropicLLM(api_key_env="SOME_MISSING_ANTHROPIC_KEY")

    async def _drive() -> None:
        async for _ in llm.generate([{"role": "user", "content": "hi"}]):
            pass

    import asyncio

    with pytest.raises(RuntimeError, match="SOME_MISSING_ANTHROPIC_KEY"):
        asyncio.run(_drive())


# ---------------------------------------------------------------------------
# GeminiLLM
# ---------------------------------------------------------------------------


_GEMINI_SSE_BODY = (
    'data: {"candidates":[{"content":{"parts":[{"text":"Hel"}]}}],'
    '"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":1}}\n\n'
    'data: {"candidates":[{"content":{"parts":[{"text":"lo"}]}}],'
    '"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":2,'
    '"cachedContentTokenCount":5}}\n\n'
)


@pytest.mark.asyncio
async def test_gemini_llm_builds_request_and_streams_text_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_GEMINI_KEY", "gk-test")
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["json"] = __import__("json").loads(request.content)
        return httpx.Response(200, content=_GEMINI_SSE_BODY.encode())

    llm = GeminiLLM(
        model="gemini-2.5-flash",
        api_key_env="MY_GEMINI_KEY",
        transport=httpx.MockTransport(handler),
    )

    chunks = []
    async for chunk in llm.generate(
        [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
        ]
    ):
        chunks.append(chunk)

    assert captured["method"] == "POST"
    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.5-flash:streamGenerateContent?alt=sse"
    )
    assert captured["headers"]["x-goog-api-key"] == "gk-test"
    # The key must never travel in the URL/query string.
    assert "key=" not in captured["url"]

    body = captured["json"]
    assert body["systemInstruction"] == {"parts": [{"text": "Be terse."}]}
    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "yo"}]},
    ]

    text_chunks = [c for c in chunks if c.text]
    assert [c.text for c in text_chunks] == ["Hel", "lo"]

    final = chunks[-1]
    assert final.finish is True
    usage_by_name = {u.unit_name: u for u in final.usage}
    assert usage_by_name["tokens_in"].units == 12
    assert usage_by_name["tokens_in"].estimated is False
    assert usage_by_name["tokens_in"].cached_units == 5
    assert usage_by_name["tokens_out"].units == 2


def test_gemini_llm_missing_env_var_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOME_MISSING_GEMINI_KEY", raising=False)
    llm = GeminiLLM(api_key_env="SOME_MISSING_GEMINI_KEY")

    async def _drive() -> None:
        async for _ in llm.generate([{"role": "user", "content": "hi"}]):
            pass

    import asyncio

    with pytest.raises(RuntimeError, match="SOME_MISSING_GEMINI_KEY"):
        asyncio.run(_drive())


# ---------------------------------------------------------------------------
# LLM_RATES
# ---------------------------------------------------------------------------


def test_llm_rates_cover_both_default_models_in_and_out() -> None:
    card = RateCard(version="test", rates=LLM_RATES)

    anthropic_in = card.lookup(
        CostComponent.LLM, "anthropic", "tokens_in", model="claude-sonnet-5"
    )
    anthropic_out = card.lookup(
        CostComponent.LLM, "anthropic", "tokens_out", model="claude-sonnet-5"
    )
    gemini_in = card.lookup(CostComponent.LLM, "gemini", "tokens_in", model="gemini-2.5-flash")
    gemini_out = card.lookup(
        CostComponent.LLM, "gemini", "tokens_out", model="gemini-2.5-flash"
    )

    assert anthropic_in is not None and anthropic_in.price_per_unit == 0.000002
    assert anthropic_out is not None and anthropic_out.price_per_unit == 0.00001
    assert gemini_in is not None and gemini_in.price_per_unit == 0.0000003
    assert gemini_out is not None and gemini_out.price_per_unit == 0.0000025
    assert all(rate.as_of == "2026-09" for rate in LLM_RATES)
