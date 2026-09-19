"""Additional cloud STT adapters: AssemblyAI (streaming), OpenAI-compatible audio
transcriptions (batch), and Sarvam AI (batch, Indian languages).

Registered here rather than in ``providers/cloud/__init__.py`` so this file can be
owned and tested independently; ``providers/cloud/__init__.py`` imports this module
for its side effect (running the ``@register`` decorators below) via a single line
appended at the very end of that file, per ``docs/EXTENDING.md``.

Same two rules as every other file under ``providers/cloud``:

1. **Lazy imports.** ``httpx`` is a core dependency of the ``cloud`` extra and is
   imported at module scope (matching ``providers/cloud/__init__.py``); the one
   genuinely optional dependency here -- ``websockets``, for AssemblyAI's
   streaming socket -- is imported inside the method that needs it, and a missing
   install raises an ``ImportError`` naming ``tring[cloud]``.
2. **API keys are env var *names*, never raw keys.** Every provider takes an
   ``api_key_env`` option; :func:`_api_key` is the one place a key is actually
   read out of the environment (a self-contained copy of
   ``providers/cloud/__init__.py``'s ``_read_api_key`` -- duplicated rather than
   imported, so this file has no dependency on another agent's private helper).

Endpointing note. AssemblyAI's streaming API is real streaming: frames go straight
onto its WebSocket as they arrive, exactly like ``DeepgramSTT`` in
``providers/cloud/__init__.py``. OpenAI's and Sarvam's transcription endpoints are
plain batch REST -- one HTTP call per utterance, no partials. But
``STTProvider.transcribe`` is handed one long-lived frame stream for the *whole
call* (see ``runtimes/cascade.py``), not one per utterance, so a batch vendor still
needs something to decide where one utterance ends and the next begins.
:func:`_endpoint_utterances` is a small RMS-energy gate that does that -- the same
technique, and the same honest limitations (a fixed threshold, easily fooled by
noisy lines), as ``providers/local``'s ``FasterWhisperSTT`` endpointer. It is
duplicated here rather than imported from ``providers/local`` so the local and
cloud provider trees stay independent of each other's private helpers.
"""

from __future__ import annotations

import array
import io
import math
import os
import wave
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlencode

import httpx

from tring.cost.rates import Rate
from tring.events import CostComponent
from tring.providers.base import STTProvider, STTResult, Usage
from tring.providers.registry import register
from tring.runtimes.base import AudioFrame

# Canonical wire format for the whole stack (see AudioFrame / runtimes/base.py).
_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # 16-bit linear PCM


def _api_key(env_var: str, provider_label: str) -> str:
    """Dereference an environment-variable *name* into its value.

    See ``providers/cloud/__init__.py``'s ``_read_api_key`` -- same contract,
    duplicated locally per this module's docstring.
    """
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"{provider_label} requires environment variable {env_var!r} to be "
            "set. Provider options carry the *name* of the variable holding the "
            "key, never the key itself."
        )
    return value


def _pcm_rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of 16-bit little-endian PCM, 0.0 - 1.0."""
    if not pcm:
        return 0.0
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    if not samples:
        return 0.0
    total = sum(float(s) * float(s) for s in samples)
    return math.sqrt(total / len(samples)) / 32768.0


def _pcm_to_wav_bytes(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    """Wrap raw PCM in an in-memory WAV container for multipart upload."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(_BYTES_PER_SAMPLE)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()


async def _endpoint_utterances(
    frames: AsyncIterator[AudioFrame],
    silence_threshold: float,
    silence_seconds: float,
    min_utterance_seconds: float,
    max_utterance_seconds: float,
) -> AsyncIterator[tuple[bytes, int, int]]:
    """Slice one continuous caller-audio stream into utterance-sized PCM buffers.

    Yields ``(pcm, sample_rate, channels)`` once trailing silence exceeds
    ``silence_seconds`` (or the buffer has run past ``max_utterance_seconds``),
    skipping anything shorter than ``min_utterance_seconds`` as noise. Whatever
    is left when ``frames`` ends (the caller hung up mid-utterance) is flushed
    too, so the caller's last sentence is never dropped on the floor.

    A flushed buffer must contain at least one above-threshold frame
    (``has_speech``): without that guard, a long dead-air tail -- the caller
    went quiet and then hung up, or simply never spoke again after the last
    utterance -- would itself look "long enough" and get flushed as a bogus
    silence-only utterance once the stream ends.
    """
    buffer = bytearray()
    trailing_silence = 0.0
    has_speech = False
    sample_rate = _SAMPLE_RATE
    channels = 1

    def seconds(nbytes: int) -> float:
        return nbytes / (sample_rate * _BYTES_PER_SAMPLE * channels)

    async for frame in frames:
        sample_rate = frame.sample_rate
        channels = frame.channels
        buffer.extend(frame.pcm)
        frame_seconds = seconds(len(frame.pcm))

        if _pcm_rms(frame.pcm) < silence_threshold:
            trailing_silence += frame_seconds
        else:
            trailing_silence = 0.0
            has_speech = True

        endpointed = trailing_silence >= silence_seconds
        overlong = seconds(len(buffer)) >= max_utterance_seconds
        if endpointed or overlong:
            if has_speech and seconds(len(buffer)) >= min_utterance_seconds:
                yield bytes(buffer), sample_rate, channels
            buffer.clear()
            trailing_silence = 0.0
            has_speech = False

    if buffer and has_speech and seconds(len(buffer)) >= min_utterance_seconds:
        yield bytes(buffer), sample_rate, channels


# ---------------------------------------------------------------------------
# AssemblyAI -- real-time streaming
# ---------------------------------------------------------------------------


@register("stt", "assemblyai")
class AssemblyAISTT(STTProvider):
    """AssemblyAI real-time STT over its v3 Universal-Streaming WebSocket API.

    Docs verified 2026-09:
    https://www.assemblyai.com/docs/speech-to-text/universal-streaming
    https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api

    No AssemblyAI SDK: just ``websockets`` (the ``cloud`` extra, same dependency
    ``DeepgramSTT`` uses) speaking the documented protocol -- connect to
    ``wss://streaming.assemblyai.com/v3/ws`` with ``sample_rate``/``encoding``/
    ``speech_model``/``format_turns`` query parameters, authenticate with the raw
    API key in the ``Authorization`` header (no ``Bearer`` prefix -- that is
    AssemblyAI's documented, non-standard scheme), send 16-bit PCM as binary
    frames, and receive JSON ``Turn`` events (``transcript``, ``end_of_turn``)
    until a ``{"type": "Terminate"}`` control message closes the session out.

    The streaming API's documented parameters have no per-connection language
    override, so ``language`` is accepted (to satisfy ``STTProvider``'s
    signature) and passed straight through onto each ``STTResult`` for callers
    that want it on the record, but it is not sent on the wire -- there is
    nothing verified to send it as.

    Usage/billing note: AssemblyAI's docs state streaming is "billed on the
    total duration that your WebSocket connection stays open, not on the amount
    of audio you send" -- unlike ``DeepgramSTT``, which bills (and therefore
    measures) bytes sent. So this provider reports ``audio_seconds`` as the
    exact wall-clock time the socket has been open at the moment of each final
    Turn: that is the quantity actually billed, measured exactly rather than
    inferred, hence ``estimated=False``.
    """

    name = "assemblyai"

    def __init__(
        self,
        api_key_env: str = "ASSEMBLYAI_API_KEY",
        speech_model: str = "universal-3-5-pro",
        sample_rate: int = _SAMPLE_RATE,
        format_turns: bool = True,
        **_: Any,
    ) -> None:
        self._api_key_env = api_key_env
        self._speech_model = speech_model
        self._sample_rate = sample_rate
        self._format_turns = format_turns

    def _ws_url(self) -> str:
        """Build the streaming connection URL. A pure function, on purpose --
        it is the seam the tests exercise instead of opening a real socket."""
        params = {
            "sample_rate": self._sample_rate,
            "encoding": "pcm_s16le",
            "speech_model": self._speech_model,
            "format_turns": "true" if self._format_turns else "false",
        }
        return f"wss://streaming.assemblyai.com/v3/ws?{urlencode(params)}"

    def _ws_headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": api_key}

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]:
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "AssemblyAISTT needs the 'websockets' package. Install the "
                "optional cloud extra: pip install 'tring[cloud]'"
            ) from exc
        import asyncio
        import json

        api_key = _api_key(self._api_key_env, "AssemblyAISTT")
        loop = asyncio.get_event_loop()
        start = loop.time()

        async with websockets.connect(
            self._ws_url(), additional_headers=self._ws_headers(api_key)
        ) as ws:

            async def _pump() -> None:
                async for frame in frames:
                    await ws.send(frame.pcm)
                await ws.send(json.dumps({"type": "Terminate"}))

            pump_task = asyncio.create_task(_pump())
            try:
                async for raw in ws:
                    message = json.loads(raw)
                    if message.get("type") != "Turn":
                        continue
                    text = message.get("transcript", "")
                    if not text:
                        continue
                    is_final = bool(message.get("end_of_turn", False))
                    usage = (
                        [
                            Usage(
                                units=loop.time() - start,
                                unit_name="audio_seconds",
                                estimated=False,
                                model=self._speech_model,
                            )
                        ]
                        if is_final
                        else []
                    )
                    yield STTResult(text=text, final=is_final, language=language, usage=usage)
            finally:
                pump_task.cancel()


# ---------------------------------------------------------------------------
# OpenAI (and OpenAI-shaped) audio transcriptions -- batch REST
# ---------------------------------------------------------------------------


@register("stt", "openai_stt")
class OpenAISTT(STTProvider):
    """Batch STT via OpenAI's ``/v1/audio/transcriptions`` REST endpoint.

    Docs verified 2026-09:
    https://platform.openai.com/docs/api-reference/audio/createTranscription
    https://platform.openai.com/docs/guides/speech-to-text
    https://platform.openai.com/docs/pricing

    ``base_url`` defaults to ``https://api.openai.com`` but is a first-class
    option specifically so this same class works, unmodified, against any
    vendor mirroring OpenAI's transcription shape -- Groq's Whisper endpoint
    is the main one in practice: point ``base_url`` at
    ``https://api.groq.com/openai`` and ``api_key_env`` at your Groq key's env
    var name (and ``model`` at Groq's own model id); nothing else changes.

    Not streaming: OpenAI's transcription endpoint is one request in, one
    transcript out, so this buffers the runtime's continuous frame stream into
    utterance-sized chunks via :func:`_endpoint_utterances` and issues one POST
    per utterance -- see this module's docstring for why that buffering exists
    at all.

    Usage: prefers whatever ``usage`` block the response actually carries.
    ``gpt-4o-transcribe``/``gpt-4o-mini-transcribe`` report
    ``{"type": "tokens", "input_tokens": ..., "output_tokens": ...}``;
    ``whisper-1`` under ``verbose_json`` reports
    ``{"type": "duration", "seconds": ...}``. Both are vendor-reported, hence
    ``estimated=False``. When the response carries no ``usage`` block at all
    (``whisper-1``'s plain ``json`` format), this falls back to the exact
    ``audio_seconds`` computed from the utterance buffer actually sent --
    exact, not guessed, so still ``estimated=False`` (the same reasoning
    ``DeepgramSTT`` documents for its own client-side measurement).
    """

    name = "openai_stt"

    def __init__(
        self,
        model: str = "whisper-1",
        base_url: str = "https://api.openai.com",
        api_key_env: str = "OPENAI_API_KEY",
        language: str | None = None,
        response_format: str = "json",
        silence_threshold: float = 0.01,
        silence_seconds: float = 0.6,
        min_utterance_seconds: float = 0.3,
        max_utterance_seconds: float = 30.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        **_: Any,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._language = language
        self._response_format = response_format
        self._silence_threshold = silence_threshold
        self._silence_seconds = silence_seconds
        self._min_utterance_seconds = min_utterance_seconds
        self._max_utterance_seconds = max_utterance_seconds
        self._transport = transport

    async def _transcribe_utterance(
        self,
        client: httpx.AsyncClient,
        pcm: bytes,
        sample_rate: int,
        channels: int,
        language: str | None,
    ) -> STTResult | None:
        api_key = _api_key(self._api_key_env, "OpenAISTT")
        wav_bytes = _pcm_to_wav_bytes(pcm, sample_rate, channels)
        lang = language or self._language
        data: dict[str, str] = {"model": self._model, "response_format": self._response_format}
        if lang:
            data["language"] = lang

        response = await client.post(
            f"{self._base_url}/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data=data,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        )
        response.raise_for_status()
        body = response.json()
        text = str(body.get("text", "")).strip()
        if not text:
            return None

        exact_seconds = len(pcm) / (sample_rate * _BYTES_PER_SAMPLE * channels)
        usage_block = body.get("usage")
        usage: list[Usage]
        if usage_block and usage_block.get("type") == "tokens":
            usage = [
                Usage(
                    units=float(usage_block.get("input_tokens", 0)),
                    unit_name="tokens_in",
                    estimated=False,
                    model=self._model,
                ),
                Usage(
                    units=float(usage_block.get("output_tokens", 0)),
                    unit_name="tokens_out",
                    estimated=False,
                    model=self._model,
                ),
            ]
        elif usage_block and usage_block.get("type") == "duration":
            usage = [
                Usage(
                    units=float(usage_block.get("seconds", exact_seconds)),
                    unit_name="audio_seconds",
                    estimated=False,
                    model=self._model,
                )
            ]
        else:
            usage = [
                Usage(
                    units=exact_seconds,
                    unit_name="audio_seconds",
                    estimated=False,
                    model=self._model,
                )
            ]
        return STTResult(text=text, final=True, language=lang, usage=usage)

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]:
        async with httpx.AsyncClient(transport=self._transport, timeout=60.0) as client:
            async for pcm, sample_rate, channels in _endpoint_utterances(
                frames,
                self._silence_threshold,
                self._silence_seconds,
                self._min_utterance_seconds,
                self._max_utterance_seconds,
            ):
                result = await self._transcribe_utterance(
                    client, pcm, sample_rate, channels, language
                )
                if result is not None:
                    yield result


# ---------------------------------------------------------------------------
# Sarvam AI -- batch REST, Indian languages
# ---------------------------------------------------------------------------

# The endpoint accepts ONLY these language_code values (plus "unknown"); a bare
# ISO-639 code like "en" is a 400 invalid_request_error, discovered live on
# 2026-09-18 and confirmed against the enum at
# https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe.
_SARVAM_LANGUAGE_CODES = frozenset(
    {
        "unknown", "hi-IN", "bn-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN",
        "ta-IN", "te-IN", "en-IN", "gu-IN", "as-IN", "ur-IN", "ne-IN", "kok-IN",
        "ks-IN", "sd-IN", "sa-IN", "sat-IN", "mni-IN", "brx-IN", "mai-IN", "doi-IN",
    }
)


def _sarvam_language(code: str | None) -> str | None:
    """Normalize a caller-supplied language to Sarvam's accepted enum.

    Users routing per-language pass bare ISO codes ("en", "hi", "mr"); Sarvam
    only accepts its "-IN" variants. Exact matches pass through, bare codes
    get the "-IN" suffix when that lands in the enum, and anything else
    becomes "unknown" (auto-detect) rather than a guaranteed 400.
    """
    if code is None:
        return None
    if code in _SARVAM_LANGUAGE_CODES:
        return code
    suffixed = f"{code}-IN"
    if suffixed in _SARVAM_LANGUAGE_CODES:
        return suffixed
    return "unknown"


@register("stt", "sarvam")
class SarvamSTT(STTProvider):
    """Batch STT via Sarvam AI's ``/speech-to-text`` REST endpoint.

    Docs verified 2026-09:
    https://docs.sarvam.ai/api-reference-docs/speech-to-text/speech-to-text
    https://docs.sarvam.ai/

    Naming note: Sarvam's speech-to-text model family is documented today as
    ``saaras`` (``saaras:v3`` the default, ``saaras:v4`` the latest). Older
    public material refers to a ``saarika`` model, but the live docs fetched
    above no longer list it under this endpoint or anywhere else, so this
    adapter defaults ``model`` to the currently-documented ``saaras:v3``
    rather than a name the docs no longer confirm.

    Specialized for Sarvam's 22 Indian languages plus English: ``language_code``
    takes Sarvam's BCP-47-style codes (``hi-IN``, ``ta-IN``, ``en-IN``, ...) or
    ``"unknown"`` for auto-detection. Auth is the ``api-subscription-key``
    header (not ``Authorization``) -- Sarvam's documented, non-standard scheme.

    Not streaming: same batch-per-utterance treatment as :class:`OpenAISTT`,
    via :func:`_endpoint_utterances` (see this module's docstring).

    Usage: the documented response has no duration or usage field at all, so
    ``audio_seconds`` is computed from the exact number of PCM bytes sent for
    the utterance -- exact, not guessed, hence ``estimated=False`` (see
    ``DeepgramSTT`` and :class:`OpenAISTT` above for the same reasoning applied
    elsewhere in this module).
    """

    name = "sarvam"

    def __init__(
        self,
        model: str = "saaras:v3",
        api_key_env: str = "SARVAM_API_KEY",
        base_url: str = "https://api.sarvam.ai",
        language_code: str | None = None,
        silence_threshold: float = 0.01,
        silence_seconds: float = 0.6,
        min_utterance_seconds: float = 0.3,
        max_utterance_seconds: float = 30.0,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        **_: Any,
    ) -> None:
        self._model = model
        self._api_key_env = api_key_env
        self._base_url = base_url.rstrip("/")
        self._language_code = language_code
        self._silence_threshold = silence_threshold
        self._silence_seconds = silence_seconds
        self._min_utterance_seconds = min_utterance_seconds
        self._max_utterance_seconds = max_utterance_seconds
        self._transport = transport

    async def _transcribe_utterance(
        self,
        client: httpx.AsyncClient,
        pcm: bytes,
        sample_rate: int,
        channels: int,
        language: str | None,
    ) -> STTResult | None:
        api_key = _api_key(self._api_key_env, "SarvamSTT")
        wav_bytes = _pcm_to_wav_bytes(pcm, sample_rate, channels)
        lang = _sarvam_language(language or self._language_code)
        data: dict[str, str] = {"model": self._model}
        if lang:
            data["language_code"] = lang

        response = await client.post(
            f"{self._base_url}/speech-to-text",
            headers={"api-subscription-key": api_key},
            data=data,
            files={"file": ("audio.wav", wav_bytes, "audio/wav")},
        )
        response.raise_for_status()
        body = response.json()
        text = str(body.get("transcript", "")).strip()
        if not text:
            return None

        exact_seconds = len(pcm) / (sample_rate * _BYTES_PER_SAMPLE * channels)
        detected_language = body.get("language_code") or lang
        return STTResult(
            text=text,
            final=True,
            language=detected_language,
            usage=[
                Usage(
                    units=exact_seconds,
                    unit_name="audio_seconds",
                    estimated=False,
                    model=self._model,
                )
            ],
        )

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]:
        async with httpx.AsyncClient(transport=self._transport, timeout=60.0) as client:
            async for pcm, sample_rate, channels in _endpoint_utterances(
                frames,
                self._silence_threshold,
                self._silence_seconds,
                self._min_utterance_seconds,
                self._max_utterance_seconds,
            ):
                result = await self._transcribe_utterance(
                    client, pcm, sample_rate, channels, language
                )
                if result is not None:
                    yield result


# ---------------------------------------------------------------------------
# Rate card entries
#
# Google Cloud Speech-to-Text v2 is deliberately not implemented in this
# module: its `recognize` REST endpoint only accepts OAuth2 bearer tokens or
# service-account credentials (verified 2026-09 against
# https://cloud.google.com/speech-to-text/v2/docs/reference/rest/v2/projects.locations.recognizers/recognize
# -- "Simple API key parameters (`?key=`) are not supported"), which does not
# fit this repo's `api_key_env`-names-an-env-var-holding-a-bearer-token
# pattern without a service-account JSON flow this file has no verified,
# SDK-free way to drive. Per the task's own instruction, skipped rather than
# guessed.
# ---------------------------------------------------------------------------

STT_RATES: list[Rate] = [
    # AssemblyAI Universal-Streaming, both English and Multilingual variants
    # list at the same rate. Verified 2026-09 against assemblyai.com/pricing.
    Rate(
        component=CostComponent.STT,
        provider="assemblyai",
        model=None,
        unit_name="audio_seconds",
        price_per_unit=0.15 / 3600,  # $0.15/hr
        currency="USD",
        as_of="2026-09",
    ),
    # OpenAI whisper-1, billed per minute of audio. Verified 2026-09 against
    # platform.openai.com/docs/pricing.
    Rate(
        component=CostComponent.STT,
        provider="openai_stt",
        model="whisper-1",
        unit_name="audio_seconds",
        price_per_unit=0.006 / 60,  # $0.006/min
        currency="USD",
        as_of="2026-09",
    ),
    # gpt-4o-transcribe / gpt-4o-mini-transcribe are billed per token, priced
    # like their chat-completions namesakes. Verified 2026-09 against
    # platform.openai.com/docs/pricing.
    Rate(
        component=CostComponent.STT,
        provider="openai_stt",
        model="gpt-4o-transcribe",
        unit_name="tokens_in",
        price_per_unit=2.50e-6,  # $2.50 / 1M input tokens
        currency="USD",
        as_of="2026-09",
    ),
    Rate(
        component=CostComponent.STT,
        provider="openai_stt",
        model="gpt-4o-transcribe",
        unit_name="tokens_out",
        price_per_unit=10.0e-6,  # $10.00 / 1M output tokens
        currency="USD",
        as_of="2026-09",
    ),
    Rate(
        component=CostComponent.STT,
        provider="openai_stt",
        model="gpt-4o-mini-transcribe",
        unit_name="tokens_in",
        price_per_unit=1.25e-6,  # $1.25 / 1M input tokens
        currency="USD",
        as_of="2026-09",
    ),
    Rate(
        component=CostComponent.STT,
        provider="openai_stt",
        model="gpt-4o-mini-transcribe",
        unit_name="tokens_out",
        price_per_unit=5.0e-6,  # $5.00 / 1M output tokens
        currency="USD",
        as_of="2026-09",
    ),
    # Sarvam AI speech-to-text, listed in INR (Sarvam publishes no USD price).
    # Verified 2026-09 against docs.sarvam.ai's pricing page.
    Rate(
        component=CostComponent.STT,
        provider="sarvam",
        model=None,
        unit_name="audio_seconds",
        price_per_unit=30 / 3600,  # ₹30/hr
        currency="INR",
        as_of="2026-09",
    ),
]


__all__ = [
    "STT_RATES",
    "AssemblyAISTT",
    "OpenAISTT",
    "SarvamSTT",
]
