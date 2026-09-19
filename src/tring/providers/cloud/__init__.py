"""Cloud provider adapters: plain ``httpx`` (and, for one, ``websockets``),
no vendor SDKs.

Every class here is registered at import time via ``@register(...)`` (see
``providers/registry.py``), and registration itself is cheap: it only
stores a reference to the class, so importing this module never touches
the network and never requires an optional dependency to be installed.
The one genuinely heavy import -- ``websockets``, needed for Deepgram's
streaming socket -- happens lazily inside the method that needs it, exactly
per ``docs/ARCHITECTURE.md``'s rule that heavy deps live behind lazy imports
in ``providers/``. ``httpx`` is a core dependency (see ``pyproject.toml``)
so it is imported normally at module scope.

**API keys.** Every provider here takes an ``api_key_env`` option: the
*name* of an environment variable, never the key itself. ``AgentSpec`` and
``ProviderSelection`` are plain, YAML-loadable, diffable data (see
``agent.py``'s module docstring) -- a raw key in an ``options`` dict would
turn every agent spec into a secret, defeating the entire point of that
design. The key is only dereferenced at call time, inside the provider.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

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


def _read_api_key(env_var: str, provider_label: str) -> str:
    """Dereference an environment-variable *name* into its value.

    This is the one place a key is ever read out of the environment; every
    provider below goes through it instead of touching ``os.environ``
    itself, so there is exactly one error message to get right.
    """
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"{provider_label} requires environment variable {env_var!r} to be "
            "set. Provider options carry the *name* of the variable holding "
            "the key, never the key itself -- see providers/cloud module docs."
        )
    return value


@register("stt", "deepgram")
class DeepgramSTT(STTProvider):
    """Deepgram real-time STT over its raw websocket API.

    No Deepgram SDK: just ``websockets`` (an optional dependency, the
    ``cloud`` extra) speaking Deepgram's documented streaming protocol --
    send linear16 frames, receive JSON transcript events. That keeps the
    only new dependency generic instead of vendor-specific.

    Usage is reported as ``audio_seconds``, computed from the exact number
    of bytes of audio we sent (``bytes / 2 (16-bit) / sample_rate``) rather
    than estimated from anything -- Deepgram's socket protocol does not
    return a billing block, so the honest number available client-side is
    "how much audio did we actually stream", and it is exact
    (``estimated=False``), not guessed.
    """

    name = "deepgram"

    def __init__(
        self,
        api_key_env: str = "DEEPGRAM_API_KEY",
        model: str = "nova-2",
        sample_rate: int = 16000,
        **_: Any,
    ) -> None:
        self._api_key_env = api_key_env
        self._model = model
        self._sample_rate = sample_rate

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]:
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "DeepgramSTT needs the 'websockets' package. Install the "
                "optional cloud extra: pip install 'tring[cloud]'"
            ) from exc
        import asyncio

        api_key = _read_api_key(self._api_key_env, "DeepgramSTT")
        query = f"model={self._model}&encoding=linear16&sample_rate={self._sample_rate}"
        if language:
            query += f"&language={language}"
        url = f"wss://api.deepgram.com/v1/listen?{query}"

        total_bytes_sent = 0

        async with websockets.connect(
            url, additional_headers={"Authorization": f"Token {api_key}"}
        ) as ws:

            async def _pump() -> None:
                nonlocal total_bytes_sent
                async for frame in frames:
                    total_bytes_sent += len(frame.pcm)
                    await ws.send(frame.pcm)
                await ws.send(json.dumps({"type": "CloseStream"}))

            pump_task = asyncio.create_task(_pump())
            try:
                async for raw in ws:
                    message = json.loads(raw)
                    alternatives = message.get("channel", {}).get("alternatives", [])
                    text = alternatives[0].get("transcript", "") if alternatives else ""
                    if not text:
                        continue
                    is_final = bool(message.get("is_final", False))
                    usage = (
                        [
                            Usage(
                                units=total_bytes_sent / 2 / self._sample_rate,
                                unit_name="audio_seconds",
                                estimated=False,
                                model=self._model,
                            )
                        ]
                        if is_final
                        else []
                    )
                    yield STTResult(text=text, final=is_final, language=language, usage=usage)
            finally:
                pump_task.cancel()


@register("tts", "elevenlabs")
class ElevenLabsTTS(TTSProvider):
    """ElevenLabs TTS via its streaming HTTP endpoint (no SDK).

    Usage is metered in ``tts_chars``: the exact character count of each
    text segment handed to the API. ElevenLabs bills by character, so this
    is a real measurement (``estimated=False``), not a proxy for tokens --
    unlike an LLM, there is no ambiguity about what "chars sent" means
    here, which is why no estimated variant of this metric exists.
    """

    name = "elevenlabs"

    def __init__(
        self,
        api_key_env: str = "ELEVENLABS_API_KEY",
        voice_id: str = "21m00Tcm4TlvDq8ikWAM",
        model_id: str = "eleven_turbo_v2_5",
        output_format: str = "pcm_16000",
        base_url: str = "https://api.elevenlabs.io",
        **_: Any,
    ) -> None:
        self._api_key_env = api_key_env
        self._voice_id = voice_id
        self._model_id = model_id
        self._output_format = output_format
        self._base_url = base_url.rstrip("/")

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        api_key = _read_api_key(self._api_key_env, "ElevenLabsTTS")
        voice_id = voice or self._voice_id
        url = (
            f"{self._base_url}/v1/text-to-speech/{voice_id}/stream"
            f"?output_format={self._output_format}"
        )
        headers = {"xi-api-key": api_key, "content-type": "application/json"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            async for segment in text:
                if not segment:
                    continue
                payload = {"text": segment, "model_id": self._model_id}
                # Usage is attached to the first audio chunk of this
                # segment rather than summed at the end, so a consumer that
                # sums every CostRecorded/Usage it sees across the whole
                # stream gets the right total without buffering anything.
                usage_pending = [
                    Usage(
                        units=len(segment),
                        unit_name="tts_chars",
                        estimated=False,
                        model=self._model_id,
                    )
                ]
                async with client.stream(
                    "POST", url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    async for raw in response.aiter_bytes():
                        if not raw:
                            continue
                        yield TTSChunk(
                            frame=AudioFrame(pcm=raw, sample_rate=16000, channels=1),
                            usage=usage_pending,
                        )
                        usage_pending = []


@register("llm", "openai_compatible")
class OpenAICompatibleLLM(LLMProvider):
    """Any OpenAI-Chat-Completions-shaped endpoint: OpenAI itself, vLLM,
    Together, Groq, a local llama.cpp server, etc.

    Deliberately no SDK: a raw ``httpx`` POST with ``stream=True`` and
    manual SSE parsing. ``LLMProvider.generate`` (providers/base.py) exists
    so text can be flushed to the speak-parser the instant it arrives;
    most SDKs wrap streaming in enough buffering/retry machinery that
    "manual SSE over httpx" ends up the more predictable latency contract
    for a module this timing-sensitive.

    ``transport`` is a test-only seam: pass an ``httpx.MockTransport`` to
    exercise request-building without a network call (see
    ``tests/test_s2s_hybrid.py``). Production code should never set it --
    leaving it ``None`` gets a normal network-backed client.
    """

    name = "openai_compatible"

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        api_key_env: str | None = "OPENAI_API_KEY",
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        **_: Any,
    ) -> None:
        # Defaults target OpenAI itself so `llm: openai_compatible` works with
        # zero options; any compatible vendor (Groq, Together, vLLM, ...) is a
        # base_url + api_key_env override away (docs/PROVIDER_ALIASES.md).
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key_env = api_key_env
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._api_key_env:
            key = _read_api_key(self._api_key_env, "OpenAICompatibleLLM")
            headers["authorization"] = f"Bearer {key}"
        return headers

    async def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[LLMChunk]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            # Ask for the trailing usage-only SSE event every OpenAI-shaped
            # endpoint that supports it will send when this is set -- it's
            # the only source of exact (non-estimated) token counts.
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools

        url = f"{self._base_url}/chat/completions"
        async with (
            httpx.AsyncClient(transport=self._transport, timeout=60.0) as client,
            client.stream("POST", url, headers=self._headers(), json=payload) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                choices = event.get("choices") or []
                delta_text = ""
                if choices:
                    delta_text = choices[0].get("delta", {}).get("content") or ""

                usage_block = event.get("usage")
                usage: list[Usage] = []
                if usage_block:
                    cached = (usage_block.get("prompt_tokens_details") or {}).get(
                        "cached_tokens"
                    )
                    usage = [
                        Usage(
                            units=usage_block.get("prompt_tokens", 0),
                            unit_name="tokens_in",
                            estimated=False,
                            model=self._model,
                            cached_units=cached,
                        ),
                        Usage(
                            units=usage_block.get("completion_tokens", 0),
                            unit_name="tokens_out",
                            estimated=False,
                            model=self._model,
                        ),
                    ]

                if delta_text or usage:
                    yield LLMChunk(text=delta_text, usage=usage, finish=bool(usage))


# Registers AnthropicLLM ("llm","anthropic") and GeminiLLM ("llm","gemini");
# see that module's docstring for endpoint/usage verification sources.
from tring.providers.cloud import llm_extra as _llm_extra  # noqa: F401,E402

# Registers OpenAIRealtimeS2S ("s2s","openai_realtime") and UltravoxS2S
# ("s2s","ultravox"); see that module's docstring for endpoint/usage
# verification sources.
from tring.providers.cloud import s2s_extra as _s2s_extra  # noqa: F401,E402

# Registers GeminiLiveS2S ("s2s","gemini_live"); see that module's docstring
# for endpoint/message/pricing verification sources and its documented skips.
from tring.providers.cloud import s2s_gemini as _s2s_gemini  # noqa: F401,E402

# Registers AssemblyAISTT ("stt","assemblyai"), OpenAISTT ("stt","openai_stt"),
# and SarvamSTT ("stt","sarvam"); see that module's docstring for
# endpoint/usage verification sources.
from tring.providers.cloud import stt_extra as _stt_extra  # noqa: F401,E402

# Registers CartesiaTTS ("tts","cartesia"), OpenAITTS ("tts","openai_tts"),
# and SarvamTTS ("tts","sarvam_tts"); see that module's docstring for
# endpoint/usage verification sources.
from tring.providers.cloud import tts_extra as _tts_extra  # noqa: F401,E402
