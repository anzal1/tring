"""Extra cloud TTS adapters: Cartesia, OpenAI, Sarvam -- plain ``httpx``, no SDKs.

Same house rules as ``providers/cloud/__init__.py``: every class here is
registered at import time via ``@register(...)``, importing this module
never touches the network, and ``api_key_env`` options carry the *name* of
an environment variable, never a raw key (see that module's docstring for
why -- ``AgentSpec``/``ProviderSelection`` must stay plain, diffable data).

**Endpoint accuracy.** Every endpoint, header, and field name below was
checked two ways before being written down: against the vendor's published
API reference, and (where the reference was ambiguous or gated behind a
login wall) by sending a real unauthenticated/bogus-credential request and
reading the error the live endpoint actually returns -- a 400/401/403 body
reveals the exact header name and required fields without needing a real
API key. Each class docstring below cites both. Prices in :data:`TTS_RATES`
are equally verified against the vendor's public pricing page as of
2026-09; where a vendor's page does not expose a stable, per-unit rate the
entry is priced ``0.0`` with a comment saying so, rather than guessed.

**Sentence-ish buffering.** :class:`~tring.providers.base.TTSProvider`
consumes a *stream* of text deltas (the runtime hands it whatever the
speak-parser has decoded so far, which is typically word- or clause-sized,
not sentence-sized). Every REST vendor in this file bills and rate-limits
per HTTP call, so calling out on every delta would both throttle hard and
butcher prosody by synthesizing sentence fragments. :func:`_buffer_sentences`
below is the same buffering policy ``KokoroTTS`` uses in
``providers/local``: flush at a sentence-ish boundary once a minimum has
accumulated, or at a word boundary once a maximum is exceeded.
"""

from __future__ import annotations

import array
import base64
import io
import os
import re
import wave
from collections.abc import AsyncIterator
from typing import Any

import httpx

from tring.cost.rates import Rate
from tring.events import CostComponent
from tring.providers.base import TTSChunk, TTSProvider, Usage
from tring.providers.registry import register
from tring.runtimes.base import AudioFrame

# Canonical wire format for the whole stack (see runtimes/base.py's
# AudioFrame docstring): 16 kHz, mono, 16-bit linear PCM.
_WIRE_SAMPLE_RATE = 16000


def _require_api_key(env_var: str, provider_label: str) -> str:
    """Dereference an environment-variable *name* into its value.

    A private copy of ``providers.cloud._read_api_key``'s logic rather than
    an import of it: this module is appended to ``providers/cloud/__init__``
    at the very end of that file specifically so two agents editing cloud
    providers never have to touch each other's code, and reaching back into
    the sibling module for a helper would recreate that coupling for no
    real benefit -- the function is four lines.
    """
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"{provider_label} requires environment variable {env_var!r} to be "
            "set. Provider options carry the *name* of the variable holding "
            "the key, never the key itself."
        )
    return value


# Flush on sentence-ending punctuation followed by whitespace or end-of-text
# (same pattern as providers/local's KokoroTTS -- see that module's comment
# for why "Dr. Rao" splitting early is an acceptable, documented tradeoff).
_SENTENCE_END = re.compile(
    r"[.!?\u3002\uff01\uff1f]+[\"')\]]*\s*$|[,;:\u060c][\"')\]]*\s+$"
)


async def _buffer_sentences(
    text: AsyncIterator[str], min_chars: int, max_chars: int
) -> AsyncIterator[str]:
    """Buffer an async stream of text deltas into sentence-ish segments.

    Flushes at the first sentence-ish boundary once at least ``min_chars``
    have accumulated, or once the buffer exceeds ``max_chars`` at a word
    boundary -- whichever comes first. Any trailing text once the source
    iterator ends is flushed too, so the caller's last clause is never
    dropped on the floor.
    """
    buffer = ""
    async for delta in text:
        buffer += delta
        if len(buffer) < min_chars:
            continue
        if _SENTENCE_END.search(buffer) or (
            len(buffer) >= max_chars and buffer.endswith(" ")
        ):
            segment, buffer = buffer.strip(), ""
            if segment:
                yield segment
    tail = buffer.strip()
    if tail:
        yield tail


def _resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of little-endian 16-bit PCM.

    Pure ``array``-module arithmetic, not numpy: the ``cloud`` extra
    (pyproject.toml) intentionally does not pull in numpy, and this module
    must not add a new dependency just to resample OpenAI's fixed-24kHz TTS
    output down to the stack's 16kHz wire format. Like ``KokoroTTS``'s
    resampler in providers/local, this is a deliberately cheap choice: it
    aliases content above the destination Nyquist frequency into a faint
    metallic edge, inaudible on a telephony-bandlimited path. A future
    ``tring[hifi]`` extra should use a real polyphase resampler instead.
    """
    if src_rate == dst_rate or not pcm:
        return pcm
    src = array.array("h")
    src.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    src_n = len(src)
    if src_n < 2:
        return b""
    dst_n = max(int(src_n * dst_rate / src_rate), 0)
    if dst_n <= 0:
        return b""
    last_index = src_n - 1
    step = last_index / max(dst_n - 1, 1)
    out = array.array("h", bytes(2 * dst_n))
    for i in range(dst_n):
        pos = i * step
        lo = int(pos)
        hi = min(lo + 1, last_index)
        frac = pos - lo
        value = src[lo] * (1.0 - frac) + src[hi] * frac
        out[i] = max(-32768, min(32767, int(value)))
    return out.tobytes()


def _unwrap_wav(data: bytes) -> tuple[bytes, int, int]:
    """Strip a WAV container down to ``(pcm_bytes, sample_rate, channels)``.

    Used for vendors (Sarvam) that hand back whole WAV files rather than a
    raw PCM stream. The stdlib ``wave`` module parses the RIFF header so
    nothing here hand-rolls chunk parsing.
    """
    with wave.open(io.BytesIO(data), "rb") as wav:
        pcm = wav.readframes(wav.getnframes())
        return pcm, wav.getframerate(), wav.getnchannels()


# ---------------------------------------------------------------------------
# Cartesia
# ---------------------------------------------------------------------------

CARTESIA_BASE_URL = "https://api.cartesia.ai"
# Cartesia requires a dated version pin on every request. This is the
# latest version as verified live against the endpoint on 2026-09-18 (an
# unversioned POST to /tts/bytes returns 400 naming this exact value as
# "latest version").
CARTESIA_API_VERSION = "2026-08-14"


@register("tts", "cartesia")
class CartesiaTTS(TTSProvider):
    """Cartesia Sonic TTS over the REST "bytes" endpoint (no SDK).

    Endpoint, auth header, and the required version header were verified
    live against ``https://api.cartesia.ai/tts/bytes`` on 2026-09-18: an
    unauthenticated POST returns HTTP 400 naming the missing
    ``Cartesia-Version`` header ("... header is required in YYYY-MM-DD
    form, latest version is 2026-08-14"), and a POST that adds a bogus
    ``X-API-Key`` moves the error from "No credentials were provided" to
    "Invalid credentials. Check your API key or access token." -- proof
    that header, not ``Authorization: Bearer``, is what the endpoint reads.
    Cross-referenced against
    https://docs.cartesia.ai/api-reference/tts/bytes for the request/
    response body shape.

    Cartesia also offers ``wss://api.cartesia.ai/tts/websocket``
    (https://docs.cartesia.ai/api-reference/tts/tts) for lower-latency
    multi-turn generation that shares synthesis context across calls. This
    adapter deliberately uses the REST bytes endpoint instead: one HTTP
    POST per sentence-ish segment is simple, testable with
    ``httpx.MockTransport`` per house convention, and -- because
    ``output_format.sample_rate`` can be set to 16000 directly in the
    request -- needs no resampling step, unlike every other TTS adapter in
    this file. A future ``cartesia_ws`` provider can add the websocket path
    for callers who need the extra latency headroom.

    Usage is ``tts_chars``, counted from the exact segment text handed to
    the API (``estimated=False``). Cartesia bills by credits and does not
    report a usage figure in this endpoint's response body, so -- exactly
    like ``ElevenLabsTTS`` in ``providers/cloud`` -- "characters we actually
    sent" is the honest, unambiguous number, not a guess.
    """

    name = "cartesia"

    def __init__(
        self,
        api_key_env: str = "CARTESIA_API_KEY",
        model_id: str = "sonic-3",
        voice_id: str | None = None,
        language: str | None = None,
        api_version: str = CARTESIA_API_VERSION,
        base_url: str = CARTESIA_BASE_URL,
        max_buffer_chars: int = 180,
        min_buffer_chars: int = 12,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        **_: Any,
    ) -> None:
        self._api_key_env = api_key_env
        self._model_id = model_id
        self._voice_id = voice_id
        self._language = language
        self._api_version = api_version
        self._base_url = base_url.rstrip("/")
        self._max_buffer_chars = max_buffer_chars
        self._min_buffer_chars = min_buffer_chars
        # Test-only seam (an httpx.MockTransport): production code should
        # never set this, matching OpenAICompatibleLLM's convention in
        # providers/cloud/__init__.py.
        self._transport = transport

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        voice_id = voice or self._voice_id
        if not voice_id:
            raise ValueError(
                "CartesiaTTS requires a voice id: pass `voice_id` in provider "
                "options, or a `voice` argument to synthesize(). There is no "
                "safe universal default -- pick one from "
                "https://docs.cartesia.ai/api-reference/tts/bytes."
            )
        api_key = _require_api_key(self._api_key_env, "CartesiaTTS")
        url = f"{self._base_url}/tts/bytes"
        headers = {
            "X-API-Key": api_key,
            "Cartesia-Version": self._api_version,
            "content-type": "application/json",
        }

        async with httpx.AsyncClient(transport=self._transport, timeout=30.0) as client:
            async for segment in _buffer_sentences(
                text, self._min_buffer_chars, self._max_buffer_chars
            ):
                payload: dict[str, Any] = {
                    "model_id": self._model_id,
                    "transcript": segment,
                    "voice": {"id": voice_id},
                    "output_format": {
                        "container": "raw",
                        "encoding": "pcm_s16le",
                        "sample_rate": _WIRE_SAMPLE_RATE,
                    },
                }
                if self._language:
                    payload["language"] = self._language

                # Usage rides on the first audio chunk of the segment, per
                # ElevenLabsTTS's convention, so a consumer summing usage
                # across the stream never double-counts one segment.
                usage_pending = [
                    Usage(
                        units=float(len(segment)),
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
                            frame=AudioFrame(
                                pcm=raw, sample_rate=_WIRE_SAMPLE_RATE, channels=1
                            ),
                            usage=usage_pending,
                        )
                        usage_pending = []


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

OPENAI_BASE_URL = "https://api.openai.com"
# response_format=pcm is fixed at this rate per OpenAI's own docs; there is
# no sample-rate parameter to request something else.
_OPENAI_TTS_NATIVE_SAMPLE_RATE = 24000


@register("tts", "openai_tts")
class OpenAITTS(TTSProvider):
    """OpenAI's ``POST /v1/audio/speech`` endpoint (no SDK), over httpx.

    Endpoint and auth verified live on 2026-09-18: an unauthenticated POST
    to ``https://api.openai.com/v1/audio/speech`` returns HTTP 401 with
    "You need to provide your API key in an Authorization header using
    Bearer auth (i.e. Authorization: Bearer YOUR_KEY)". Request/response
    shape cross-referenced against
    https://developers.openai.com/api/docs/guides/text-to-speech, which
    documents ``response_format=pcm`` as "raw samples in 24kHz (16-bit
    signed, low-endian), without the header" and streaming via chunked
    transfer encoding for the ``wav``/``pcm`` formats.

    OpenAI's PCM is fixed at 24 kHz; the stack's wire format is 16 kHz mono,
    so each segment's audio is resampled with :func:`_resample_pcm16` --
    plain ``array``-module linear interpolation, not numpy, because the
    ``cloud`` extra (pyproject.toml) does not carry numpy and this module
    must not add a dependency to reach it. That means, unlike
    :class:`CartesiaTTS`, a segment's bytes are fully buffered before being
    resampled and emitted as one frame rather than streamed chunk-by-chunk
    -- the latency cost is bounded by one sentence-ish segment's audio, the
    same granularity ``KokoroTTS`` already accepts in providers/local.

    Usage is ``tts_chars`` from the exact segment sent, ``estimated=False``
    -- OpenAI's ``tts-1``/``tts-1-hd`` models bill per character
    (https://openai.com/api/pricing: "$15.00 / 1M characters" for
    ``tts-1``), so this is a real meter, not a token proxy.
    """

    name = "openai_tts"

    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "tts-1",
        voice: str = "alloy",
        base_url: str = OPENAI_BASE_URL,
        max_buffer_chars: int = 300,
        min_buffer_chars: int = 12,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        **_: Any,
    ) -> None:
        self._api_key_env = api_key_env
        self._model = model
        self._voice = voice
        self._base_url = base_url.rstrip("/")
        self._max_buffer_chars = max_buffer_chars
        self._min_buffer_chars = min_buffer_chars
        self._transport = transport

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        api_key = _require_api_key(self._api_key_env, "OpenAITTS")
        url = f"{self._base_url}/v1/audio/speech"
        headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        }
        voice_name = voice or self._voice

        async with httpx.AsyncClient(transport=self._transport, timeout=30.0) as client:
            async for segment in _buffer_sentences(
                text, self._min_buffer_chars, self._max_buffer_chars
            ):
                payload = {
                    "model": self._model,
                    "input": segment,
                    "voice": voice_name,
                    "response_format": "pcm",
                }
                pcm_24k = bytearray()
                async with client.stream(
                    "POST", url, headers=headers, json=payload
                ) as response:
                    response.raise_for_status()
                    async for raw in response.aiter_bytes():
                        pcm_24k.extend(raw)

                pcm_16k = _resample_pcm16(
                    bytes(pcm_24k), _OPENAI_TTS_NATIVE_SAMPLE_RATE, _WIRE_SAMPLE_RATE
                )
                if not pcm_16k:
                    continue
                yield TTSChunk(
                    frame=AudioFrame(
                        pcm=pcm_16k, sample_rate=_WIRE_SAMPLE_RATE, channels=1
                    ),
                    usage=[
                        Usage(
                            units=float(len(segment)),
                            unit_name="tts_chars",
                            estimated=False,
                            model=self._model,
                        )
                    ],
                )


# ---------------------------------------------------------------------------
# Sarvam
# ---------------------------------------------------------------------------

SARVAM_BASE_URL = "https://api.sarvam.ai"


@register("tts", "sarvam_tts")
class SarvamTTS(TTSProvider):
    """Sarvam AI's Bulbul TTS REST API for Indian languages (no SDK).

    Endpoint and auth header verified live on 2026-09-18: a POST to
    ``https://api.sarvam.ai/text-to-speech`` with a bogus
    ``api-subscription-key`` returns HTTP 403 "Invalid or missing
    authentication credentials" (confirming that header, not
    ``Authorization``, is what is read; an entirely missing key returns a
    distinct "No credentials were provided" 401 for comparison).
    Request/response field names cross-referenced against
    https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert, which
    shows the body as ``{"text": ..., "language_code": ...}`` and the
    response as ``{"audios": ["<base64 wav>"], "request_id": ...}``.

    Bulbul returns whole base64-encoded WAV files, not a raw PCM stream, so
    each segment is one blocking HTTP call followed by unwrapping the WAV
    container with the stdlib ``wave`` module (:func:`_unwrap_wav`) -- no
    numpy needed. The request pins ``speech_sample_rate: 16000`` (a
    documented enum value for this field), so -- like :class:`CartesiaTTS`
    -- no resampling step is needed: Bulbul hands back audio already at the
    stack's wire rate.

    Usage is ``tts_chars`` from the exact segment sent, ``estimated=False``
    -- Sarvam bills bulbul:v3 at a flat ₹30 per 10,000 characters
    (https://docs.sarvam.ai/api-reference-docs/pricing), a real
    per-character meter.
    """

    name = "sarvam_tts"

    def __init__(
        self,
        api_key_env: str = "SARVAM_API_KEY",
        model: str = "bulbul:v3",
        speaker: str = "shubh",
        language_code: str = "en-IN",
        base_url: str = SARVAM_BASE_URL,
        max_buffer_chars: int = 300,
        min_buffer_chars: int = 12,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        **_: Any,
    ) -> None:
        self._api_key_env = api_key_env
        self._model = model
        self._speaker = speaker
        self._language_code = language_code
        self._base_url = base_url.rstrip("/")
        self._max_buffer_chars = max_buffer_chars
        self._min_buffer_chars = min_buffer_chars
        self._transport = transport

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        api_key = _require_api_key(self._api_key_env, "SarvamTTS")
        url = f"{self._base_url}/text-to-speech"
        headers = {
            "api-subscription-key": api_key,
            "content-type": "application/json",
        }
        speaker = voice or self._speaker

        async with httpx.AsyncClient(transport=self._transport, timeout=30.0) as client:
            async for segment in _buffer_sentences(
                text, self._min_buffer_chars, self._max_buffer_chars
            ):
                payload = {
                    "text": segment,
                    "language_code": self._language_code,
                    "model": self._model,
                    "speaker": speaker,
                    "speech_sample_rate": _WIRE_SAMPLE_RATE,
                }
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                body = response.json()
                usage = [
                    Usage(
                        units=float(len(segment)),
                        unit_name="tts_chars",
                        estimated=False,
                        model=self._model,
                    )
                ]
                for wav_b64 in body.get("audios", []):
                    pcm, sample_rate, channels = _unwrap_wav(base64.b64decode(wav_b64))
                    if not pcm:
                        continue
                    yield TTSChunk(
                        frame=AudioFrame(pcm=pcm, sample_rate=sample_rate, channels=channels),
                        usage=usage,
                    )
                    usage = []


# ---------------------------------------------------------------------------
# Rates -- verified against each vendor's public pricing page, as_of 2026-09.
# ---------------------------------------------------------------------------

TTS_RATES: list[Rate] = [
    # Cartesia: the public pricing page (cartesia.ai/pricing, checked
    # 2026-09-18) lists only plan-level monthly credit allowances (e.g.
    # "Startup: $49/mo for 1.25M credits") and per-minute *voice agent*
    # overage, not a documented credits-per-character (or $-per-character)
    # rate for the Sonic TTS bytes/websocket API itself. Computing a rate
    # from the plan numbers would require guessing the credit-to-character
    # ratio, which the "never guess a price" rule forbids -- priced 0.0
    # until a verifiable per-unit rate is published.
    Rate(
        component=CostComponent.TTS,
        provider="cartesia",
        model="sonic-3",
        unit_name="tts_chars",
        price_per_unit=0.0,
        currency="USD",
        as_of="2026-09",
    ),
    # OpenAI tts-1: "$15.00 / 1M characters" per openai.com/api/pricing,
    # checked 2026-09-18.
    Rate(
        component=CostComponent.TTS,
        provider="openai_tts",
        model="tts-1",
        unit_name="tts_chars",
        price_per_unit=0.000015,
        currency="USD",
        as_of="2026-09",
    ),
    # OpenAI tts-1-hd: "$30.00 / 1M characters", same source.
    Rate(
        component=CostComponent.TTS,
        provider="openai_tts",
        model="tts-1-hd",
        unit_name="tts_chars",
        price_per_unit=0.00003,
        currency="USD",
        as_of="2026-09",
    ),
    # Sarvam bulbul:v3: "₹30/10K characters" per
    # docs.sarvam.ai/api-reference-docs/pricing, checked 2026-09-18.
    # Priced in INR (Rate.currency), not converted to USD, so the number
    # matches what the vendor's own page states rather than introducing a
    # second, drifting exchange-rate assumption.
    Rate(
        component=CostComponent.TTS,
        provider="sarvam_tts",
        model="bulbul:v3",
        unit_name="tts_chars",
        price_per_unit=0.003,
        currency="INR",
        as_of="2026-09",
    ),
]


__all__ = [
    "TTS_RATES",
    "CartesiaTTS",
    "OpenAITTS",
    "SarvamTTS",
]
