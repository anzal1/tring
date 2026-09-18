"""LLM adapters: Anthropic Messages API and Google Gemini API.

Endpoints, headers, event shapes, and prices were verified live against the
vendors' own docs (fetched 2026-09) rather than written from memory:

- Anthropic Messages API:    https://platform.claude.com/docs/en/api/messages
- Anthropic streaming:       https://platform.claude.com/docs/en/build-with-claude/streaming
- Anthropic API versioning:  https://platform.claude.com/docs/en/api/versioning
- Anthropic pricing:         https://claude.com/pricing
- Anthropic model IDs:       https://platform.claude.com/docs/en/models/overview
- Gemini generateContent:    https://ai.google.dev/api/generate-content
- Gemini API key handling:   https://ai.google.dev/gemini-api/docs/api-key
- Gemini pricing:            https://ai.google.dev/gemini-api/docs/pricing

No vendor SDKs: plain ``httpx`` with manual SSE parsing, matching the house
style set by ``providers/cloud/__init__.py`` (see that module's docstring
for the "why no SDK" rationale). Both providers below stream text the
instant it arrives -- ``LLMProvider.generate`` (providers/base.py) forbids
buffering output waiting for a complete sentence or a complete tool-call
JSON object, since that is exactly the round-trip voice latency cannot
afford.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from tring.cost.rates import Rate
from tring.events import CostComponent
from tring.providers.base import LLMChunk, LLMProvider, Usage
from tring.providers.registry import register


def _read_api_key(env_var: str, provider_label: str) -> str:
    """Dereference an environment-variable *name* into its value.

    Deliberately duplicated (it is eight lines) from the identically-named
    helper in ``providers/cloud/__init__.py`` instead of imported from it --
    this module is appended to that package's namespace at the very end of
    its file (see the two lines appended there), and importing back from a
    module that has not finished initializing is a needless coupling this
    file has no reason to take on. The contract it enforces is the same
    one documented there: options carry the *name* of an env var, never a
    raw key, so a key never ends up sitting in a YAML-loadable, diffable
    ``ProviderSelection``.
    """
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"{provider_label} requires environment variable {env_var!r} to be "
            "set. Provider options carry the *name* of the variable holding "
            "the key, never the key itself -- see providers/cloud module docs."
        )
    return value


# https://platform.claude.com/docs/en/api/versioning -- version history shows
# a single entry since 2023-06-01; there is no newer value to pin.
_ANTHROPIC_VERSION = "2023-06-01"


@register("llm", "anthropic")
class AnthropicLLM(LLMProvider):
    """Anthropic's Messages API (``POST /v1/messages``), streamed over SSE.

    Verified against
    https://platform.claude.com/docs/en/api/messages and
    https://platform.claude.com/docs/en/build-with-claude/streaming.
    The stream is a strict event sequence -- ``message_start``, then for
    each content block a ``content_block_start`` / ``content_block_delta``\\*
    / ``content_block_stop`` run, then one or more ``message_delta`` events,
    then ``message_stop``. Only a ``content_block_delta`` whose
    ``delta.type`` is ``"text_delta"`` carries output text (the other delta
    kinds -- ``input_json_delta`` for tool calls, ``thinking_delta`` /
    ``signature_delta`` for extended thinking -- are out of scope here per
    ``LLMProvider``'s contract that tool-call framing belongs to the
    speak_parser, not the provider).

    Usage is exact, never estimated. Both ``message_start.message.usage``
    and each ``message_delta.usage`` carry ``input_tokens``/``output_tokens``
    (the docs warn the latter is *cumulative*, so the latest value seen
    always wins over an earlier one) and, once a prompt-cache read actually
    happens, ``cache_read_input_tokens`` -- forwarded verbatim as
    ``Usage.cached_units`` on the input-token record, never the output one.

    ``transport`` is a test-only seam (see ``tests/test_providers_llm.py``);
    production code should leave it ``None``.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 1024,
        base_url: str = "https://api.anthropic.com",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        **_: Any,
    ) -> None:
        self._model = model
        self._api_key_env = api_key_env
        self._max_tokens = max_tokens
        self._base_url = base_url.rstrip("/")
        self._transport = transport

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        api_key = _read_api_key(self._api_key_env, "AnthropicLLM")

        # The Messages API has no "system" role inside `messages` -- any
        # system turn is pulled out into the top-level `system` param
        # instead (docs: "there is no 'system' role for input messages").
        system_parts: list[str] = []
        chat_messages: list[dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_parts.append(str(msg.get("content", "")))
            else:
                chat_messages.append(msg)

        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": chat_messages,
            "stream": True,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if tools:
            payload["tools"] = tools

        headers = {
            "x-api-key": api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        url = f"{self._base_url}/v1/messages"

        # Running usage snapshot. message_start seeds it; every later
        # message_delta overwrites with the cumulative totals the docs say
        # it carries, so the last value observed before message_stop is
        # always the right one to report.
        input_tokens = 0.0
        output_tokens = 0.0
        cached_tokens: float | None = None

        async with (
            httpx.AsyncClient(transport=self._transport, timeout=60.0) as client,
            client.stream("POST", url, headers=headers, json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                event = json.loads(data)
                event_type = event.get("type")

                if event_type == "message_start":
                    usage_block = event.get("message", {}).get("usage") or {}
                elif event_type == "message_delta":
                    usage_block = event.get("usage") or {}
                else:
                    usage_block = {}
                if "input_tokens" in usage_block:
                    input_tokens = usage_block["input_tokens"]
                if "output_tokens" in usage_block:
                    output_tokens = usage_block["output_tokens"]
                if "cache_read_input_tokens" in usage_block:
                    cached_tokens = usage_block["cache_read_input_tokens"]

                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        yield LLMChunk(text=delta["text"])

                if event_type == "message_stop":
                    yield LLMChunk(
                        text="",
                        finish=True,
                        usage=[
                            Usage(
                                units=input_tokens,
                                unit_name="tokens_in",
                                estimated=False,
                                model=self._model,
                                cached_units=cached_tokens,
                            ),
                            Usage(
                                units=output_tokens,
                                unit_name="tokens_out",
                                estimated=False,
                                model=self._model,
                            ),
                        ],
                    )


@register("llm", "gemini")
class GeminiLLM(LLMProvider):
    """Google's Gemini API, ``streamGenerateContent`` over SSE.

    Verified against https://ai.google.dev/api/generate-content and
    https://ai.google.dev/gemini-api/docs/api-key: ``POST
    {base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse`` with
    the key sent as the ``x-goog-api-key`` header. The docs also show a
    ``?key=`` query-parameter form; the header form is used here instead
    on purpose, since a key in the URL ends up in proxy and access logs
    that a header does not.

    Each SSE ``data:`` line is one complete ``GenerateContentResponse``.
    Text arrives incrementally across ``candidates[].content.parts[].text``
    -- every part is flushed the moment it is seen, never buffered into a
    full candidate or a full response.

    Usage is exact, never estimated: ``usageMetadata.promptTokenCount`` /
    ``candidatesTokenCount`` map to tokens_in/tokens_out, and
    ``cachedContentTokenCount`` (present once a cached-content prefix is
    actually hit) maps straight through to ``Usage.cached_units`` on the
    input-token record.

    ``transport`` is a test-only seam (see ``tests/test_providers_llm.py``);
    production code should leave it ``None``.
    """

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key_env: str = "GEMINI_API_KEY",
        base_url: str = "https://generativelanguage.googleapis.com",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        **_: Any,
    ) -> None:
        self._model = model
        self._api_key_env = api_key_env
        self._base_url = base_url.rstrip("/")
        self._transport = transport

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        api_key = _read_api_key(self._api_key_env, "GeminiLLM")

        # Gemini has no "system" role inside `contents` either -- system
        # turns go into the top-level `systemInstruction` param. Its two
        # remaining roles are "user" and "model" (not "assistant"); a role
        # this internal message log doesn't know about (e.g. a "tool" turn)
        # folds onto "user" rather than being silently dropped.
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role")
            text = str(msg.get("content", ""))
            if role == "system":
                system_parts.append(text)
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": text}]})

        payload: dict[str, Any] = {"contents": contents}
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if tools:
            payload["tools"] = tools

        url = f"{self._base_url}/v1beta/models/{self._model}:streamGenerateContent?alt=sse"
        headers = {"x-goog-api-key": api_key, "content-type": "application/json"}

        # usageMetadata is repeated (cumulatively) on later chunks in
        # practice; keep the latest value seen rather than the first.
        prompt_tokens = 0.0
        output_tokens = 0.0
        cached_tokens: float | None = None

        async with (
            httpx.AsyncClient(transport=self._transport, timeout=60.0) as client,
            client.stream("POST", url, headers=headers, json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if not data:
                    continue
                chunk = json.loads(data)

                for candidate in chunk.get("candidates", []):
                    for part in candidate.get("content", {}).get("parts", []):
                        part_text = part.get("text")
                        if part_text:
                            yield LLMChunk(text=part_text)

                usage_block = chunk.get("usageMetadata") or {}
                if "promptTokenCount" in usage_block:
                    prompt_tokens = usage_block["promptTokenCount"]
                if "candidatesTokenCount" in usage_block:
                    output_tokens = usage_block["candidatesTokenCount"]
                if "cachedContentTokenCount" in usage_block:
                    cached_tokens = usage_block["cachedContentTokenCount"]

        # streamGenerateContent has no explicit "final chunk" marker of its
        # own (no [DONE], no message_stop) -- the HTTP stream simply ends,
        # so the finish chunk is emitted once the loop above falls through.
        yield LLMChunk(
            text="",
            finish=True,
            usage=[
                Usage(
                    units=prompt_tokens,
                    unit_name="tokens_in",
                    estimated=False,
                    model=self._model,
                    cached_units=cached_tokens,
                ),
                Usage(
                    units=output_tokens,
                    unit_name="tokens_out",
                    estimated=False,
                    model=self._model,
                ),
            ],
        )


# ---------------------------------------------------------------------------
# LLM_RATES: verified list prices for this module's default models.
#
# Anthropic pricing: https://claude.com/pricing (Claude Sonnet 5 row).
# Gemini pricing:    https://ai.google.dev/gemini-api/docs/pricing
#                    (Gemini 2.5 Flash row, paid tier, text/image/video).
# Both fetched 2026-09; prices drift, pin your own RateCard for anything
# that matters financially rather than trusting numbers a library shipped.
# ---------------------------------------------------------------------------
LLM_RATES: list[Rate] = [
    Rate(
        component=CostComponent.LLM,
        provider="anthropic",
        model="claude-sonnet-5",
        unit_name="tokens_in",
        price_per_unit=0.000002,  # $2 / 1M input tokens
        as_of="2026-09",
    ),
    Rate(
        component=CostComponent.LLM,
        provider="anthropic",
        model="claude-sonnet-5",
        unit_name="tokens_out",
        price_per_unit=0.00001,  # $10 / 1M output tokens
        as_of="2026-09",
    ),
    Rate(
        component=CostComponent.LLM,
        provider="gemini",
        model="gemini-2.5-flash",
        unit_name="tokens_in",
        price_per_unit=0.0000003,  # $0.30 / 1M input tokens (text/image/video, paid tier)
        as_of="2026-09",
    ),
    Rate(
        component=CostComponent.LLM,
        provider="gemini",
        model="gemini-2.5-flash",
        unit_name="tokens_out",
        price_per_unit=0.0000025,  # $2.50 / 1M output tokens (incl. thinking tokens)
        as_of="2026-09",
    ),
]
