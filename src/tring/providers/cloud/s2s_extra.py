"""Speech-to-speech provider adapters: OpenAI Realtime and Ultravox.

Both vendors here are *stateful websocket* protocols, which makes them a
different animal from the request/response and one-shot-stream providers in
``providers/cloud/__init__.py``. Two consequences shape every line below.

**One socket, two directions, one generator.** ``S2SProvider.converse``
(``providers/base.py``) is a single async generator: caller audio goes in via
the ``frames`` iterator, bot audio comes out as ``TTSChunk``. A websocket has
no such symmetry -- it is one duplex pipe. So each provider runs a *pump task*
that drains ``frames`` into the socket while the generator body reads the
socket and yields. When ``frames`` is exhausted the pump closes the socket,
which is what ends the read loop and therefore the generator. That is the
entire lifecycle; there is no separate ``stop()`` to coordinate with.

**Usage has nowhere else to go.** ``TTSChunk`` is the only thing ``converse``
can yield, and vendor usage arrives *after* the last audio byte (OpenAI's
``response.done``; Ultravox's socket simply closing). Rather than drop it or
attach it to a stale chunk, each provider yields a final ``TTSChunk`` whose
frame is **zero-length PCM** purely as a usage carrier. ``S2SRuntime``
(``runtimes/s2s.py``) forwards the frame to ``on_bot_audio`` and records the
usage; a zero-length frame is a no-op for any transport, so the cost event
survives without inventing audio.

**Sample rates.** The stack's canonical wire format is 16 kHz mono 16-bit PCM
(``runtimes/base.py``). Neither vendor speaks it natively at the default
settings, so ``_resample_pcm16`` below converts on both edges and every frame
this module yields is stamped 16 kHz. See the per-class docstrings for what
each vendor actually requires and why.

**API keys** follow the same rule as the rest of ``providers/cloud``: options
carry the *name* of an environment variable, never a key.
"""

from __future__ import annotations

import array
import asyncio
import base64
import contextlib
import json
import os
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol

from tring.cost.rates import Rate
from tring.events import CostComponent
from tring.providers.base import S2SProvider, TTSChunk, Usage
from tring.providers.registry import register
from tring.runtimes.base import AudioFrame

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import httpx

# Canonical wire format for the whole stack (see runtimes/base.AudioFrame).
WIRE_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # 16-bit linear PCM

_CLOUD_EXTRA = 'pip install "tring[cloud]"'

# A zero-length frame: the usage carrier described in the module docstring.
# AudioFrame is a frozen dataclass, so one shared instance is safe.
_USAGE_ONLY_FRAME = AudioFrame(pcm=b"", sample_rate=WIRE_SAMPLE_RATE, channels=1)


class _WebSocketLike(Protocol):
    """The three things this module needs from a websocket connection.

    Narrowing the surface to a Protocol is what makes the ``connect`` seam
    testable: a scripted fake in ``tests/test_providers_s2s.py`` implements
    exactly these three members, with no ``websockets`` install and no socket.
    """

    async def send(self, message: str | bytes) -> None: ...

    async def close(self) -> None: ...

    def __aiter__(self) -> AsyncIterator[str | bytes]: ...


# (url, headers) -> connected socket. Injected in tests; ``None`` in production
# means "use _websockets_connect below".
WebSocketConnect = Callable[[str, dict[str, str]], Awaitable[_WebSocketLike]]


async def _websockets_connect(url: str, headers: dict[str, str]) -> _WebSocketLike:
    """Default ``WebSocketConnect``: the ``websockets`` library, imported lazily.

    Lazy because ``registry._load_builtin`` imports this module eagerly just to
    run the ``@register`` decorators, and a core install has no ``websockets``.
    Import-time failure here would break provider *discovery* for everyone,
    including users who only ever wanted a local provider.
    """
    try:
        import websockets
    except ImportError as exc:  # pragma: no cover - exercised by install shape
        raise ImportError(
            "Speech-to-speech providers need the 'websockets' package. "
            f"Install the optional cloud extra: {_CLOUD_EXTRA}"
        ) from exc

    connection: _WebSocketLike = await websockets.connect(
        url, additional_headers=headers or None
    )
    return connection


def _require_env(env_var: str, provider_label: str) -> str:
    """Dereference an environment-variable *name* into its value.

    Deliberately a local copy of the helper in ``providers/cloud/__init__.py``
    rather than an import of it: this module is imported *from the bottom of*
    that package's ``__init__``, so importing a private name back out of a
    half-initialised package would be a needless circular-import hazard for a
    six-line function.
    """
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(
            f"{provider_label} requires environment variable {env_var!r} to be "
            "set. Provider options carry the *name* of the variable holding the "
            "key, never the key itself -- see the providers/cloud module docs."
        )
    return value


def _resample_pcm16(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of mono 16-bit little-endian PCM.

    Deliberately dependency-free: ``numpy`` lives in the ``local`` extra and
    ``audioop`` was removed from the stdlib in 3.13, so neither is available to
    a ``tring[cloud]`` install. Linear interpolation is not a polyphase filter
    and it resamples each chunk independently, so it can alias slightly and can
    leave a sub-sample discontinuity at chunk boundaries. For 16 kHz <-> 24 kHz
    speech that is inaudible; a transport that already owns a real resampler
    should hand frames in at the vendor's native rate instead (both providers
    take the rate as an option) and this becomes a no-op.
    """
    if src_rate == dst_rate or not pcm:
        return pcm

    samples = array.array("h")
    # Guard against a half-sample tail: a websocket chunk boundary is not
    # obliged to land on a sample boundary.
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % _BYTES_PER_SAMPLE)])
    if sys.byteorder == "big":  # pragma: no cover - CI is little-endian
        samples.byteswap()
    n_src = len(samples)
    if n_src == 0:
        return b""

    n_dst = max(1, round(n_src * dst_rate / src_rate))
    out = array.array("h", bytes(n_dst * _BYTES_PER_SAMPLE))
    step = (n_src - 1) / (n_dst - 1) if n_dst > 1 else 0.0
    for i in range(n_dst):
        position = i * step
        left = int(position)
        right = min(left + 1, n_src - 1)
        fraction = position - left
        out[i] = int(samples[left] + (samples[right] - samples[left]) * fraction)

    if sys.byteorder == "big":  # pragma: no cover - CI is little-endian
        out.byteswap()
    return out.tobytes()


# ---------------------------------------------------------------------------
# OpenAI Realtime
#
# Docs verified 2026-09:
#   https://developers.openai.com/api/docs/guides/realtime-conversations
#   https://developers.openai.com/api/reference/resources/realtime/client-events
#   https://developers.openai.com/api/reference/resources/realtime/server-events
#   https://developers.openai.com/api/docs/guides/realtime-transcription
# ---------------------------------------------------------------------------

OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime"

# The Realtime API accepts exactly one PCM sample rate: 24 kHz, mono, 16-bit
# ({"type": "audio/pcm", "rate": 24000}). Not a default we picked -- the only
# alternatives are the telephony codecs audio/pcmu and audio/pcma. This is why
# every frame is resampled on the way in and on the way out.
OPENAI_REALTIME_RATE = 24000


@register("s2s", "openai_realtime")
class OpenAIRealtimeS2S(S2SProvider):
    """OpenAI's Realtime API over a raw websocket -- no ``openai`` SDK.

    Protocol, as verified against the docs cited above:

    1. Connect to ``wss://api.openai.com/v1/realtime?model=<model>`` with an
       ``Authorization: Bearer <key>`` header. (The old ``OpenAI-Beta:
       realtime=v1`` header is deprecated post-GA and is not sent.)
    2. Send one ``session.update`` naming the audio formats, the voice, the
       turn-detection mode, and the input-transcription model. Input
       transcription is opt-in and is the *only* way the caller's words ever
       become text -- without it ``post_call_transcript`` would return the
       bot's half of the conversation and nothing else.
    3. Stream caller audio as ``input_audio_buffer.append`` events carrying
       base64 PCM. With server/semantic VAD enabled the server commits the
       buffer and starts responses on its own, so no manual
       ``input_audio_buffer.commit`` or ``response.create`` is sent.
    4. Read ``response.output_audio.delta`` events (base64 PCM) and yield them
       as ``TTSChunk``s.

    Transcripts come from two server events, collected for
    ``post_call_transcript``: ``conversation.item.input_audio_transcription
    .completed`` (the caller) and ``response.output_audio_transcript.done``
    (the bot).

    Usage is **exact** (``estimated=False``): ``response.done`` carries a
    ``response.usage`` block with ``input_tokens`` / ``output_tokens`` and
    ``input_token_details.cached_tokens``, which is passed through as
    ``Usage.cached_units`` so a prompt-cache regression stays visible. One
    usage-carrier chunk is yielded per response, not one for the whole call,
    so a multi-turn call reports each turn as the vendor priced it.
    """

    name = "openai_realtime"

    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        model: str = "gpt-realtime-2.1",
        voice: str = "marin",
        instructions: str | None = None,
        transcription_model: str | None = "whisper-1",
        turn_detection: str = "semantic_vad",
        base_url: str = OPENAI_REALTIME_URL,
        *,
        connect: WebSocketConnect | None = None,
        **_: Any,
    ) -> None:
        self._api_key_env = api_key_env
        self._model = model
        self._voice = voice
        self._instructions = instructions
        self._transcription_model = transcription_model
        self._turn_detection = turn_detection
        self._base_url = base_url
        # Test-only seam (see _WebSocketLike). Production leaves this None.
        self._connect: WebSocketConnect = connect or _websockets_connect
        self._transcript: list[dict[str, Any]] = []

    # -- request construction -------------------------------------------------

    def build_url(self) -> str:
        return f"{self._base_url}?model={self._model}"

    def build_headers(self, api_key: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {api_key}"}

    def build_session_update(self) -> dict[str, Any]:
        """The one configuration event, split out so tests can assert on it
        without standing up a socket."""
        audio_input: dict[str, Any] = {
            "format": {"type": "audio/pcm", "rate": OPENAI_REALTIME_RATE},
            "turn_detection": {"type": self._turn_detection},
        }
        if self._transcription_model:
            audio_input["transcription"] = {"model": self._transcription_model}

        session: dict[str, Any] = {
            "type": "realtime",
            "model": self._model,
            "output_modalities": ["audio"],
            "audio": {
                "input": audio_input,
                "output": {
                    "format": {"type": "audio/pcm", "rate": OPENAI_REALTIME_RATE},
                    "voice": self._voice,
                },
            },
        }
        if self._instructions:
            session["instructions"] = self._instructions
        return {"type": "session.update", "session": session}

    def build_audio_append(self, frame: AudioFrame) -> dict[str, Any]:
        pcm = _resample_pcm16(frame.pcm, frame.sample_rate, OPENAI_REALTIME_RATE)
        return {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(pcm).decode("ascii"),
        }

    # -- event handling -------------------------------------------------------

    def _handle_event(self, event: dict[str, Any]) -> list[TTSChunk]:
        """Map one server event to zero or more ``TTSChunk``s.

        A pure function of the event plus the transcript accumulator, so the
        event-to-chunk mapping is unit-testable without any socket at all.
        """
        kind = event.get("type")

        if kind == "response.output_audio.delta":
            raw = base64.b64decode(event.get("delta") or "")
            pcm = _resample_pcm16(raw, OPENAI_REALTIME_RATE, WIRE_SAMPLE_RATE)
            frame = AudioFrame(pcm=pcm, sample_rate=WIRE_SAMPLE_RATE, channels=1)
            return [TTSChunk(frame=frame)]

        if kind == "conversation.item.input_audio_transcription.completed":
            text = event.get("transcript") or ""
            if text:
                self._transcript.append({"role": "user", "text": text})
            return []

        if kind == "response.output_audio_transcript.done":
            text = event.get("transcript") or ""
            if text:
                self._transcript.append({"role": "bot", "text": text})
            return []

        if kind == "response.done":
            usage = self._usage_from_response_done(event)
            return [TTSChunk(frame=_USAGE_ONLY_FRAME, usage=usage)] if usage else []

        if kind == "error":
            detail = event.get("error") or {}
            raise RuntimeError(
                "OpenAI Realtime API returned an error event: "
                f"{detail.get('type')}: {detail.get('message')}"
            )

        return []

    def _usage_from_response_done(self, event: dict[str, Any]) -> list[Usage]:
        block = (event.get("response") or {}).get("usage") or {}
        if not block:
            # Honesty over completeness: no usage block means no usage record,
            # rather than a plausible-looking guess.
            return []
        cached = (block.get("input_token_details") or {}).get("cached_tokens")
        return [
            Usage(
                units=float(block.get("input_tokens", 0)),
                unit_name="tokens_in",
                estimated=False,
                model=self._model,
                cached_units=None if cached is None else float(cached),
            ),
            Usage(
                units=float(block.get("output_tokens", 0)),
                unit_name="tokens_out",
                estimated=False,
                model=self._model,
            ),
        ]

    # -- the session ----------------------------------------------------------

    async def converse(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[TTSChunk]:
        api_key = _require_env(self._api_key_env, "OpenAIRealtimeS2S")
        socket = await self._connect(self.build_url(), self.build_headers(api_key))
        await socket.send(json.dumps(self.build_session_update()))

        async def _pump() -> None:
            async for frame in frames:
                await socket.send(json.dumps(self.build_audio_append(frame)))
            # Closing is how the read loop below terminates: the Realtime API
            # has no "I am done talking" client event to send instead.
            await socket.close()

        pump_task = asyncio.create_task(_pump())
        try:
            async for raw in socket:
                if isinstance(raw, bytes):
                    continue  # the Realtime protocol is JSON text only
                for chunk in self._handle_event(json.loads(raw)):
                    yield chunk
        finally:
            pump_task.cancel()
            with contextlib.suppress(Exception):
                await socket.close()

    async def post_call_transcript(self) -> list[dict[str, Any]]:
        return list(self._transcript)


# ---------------------------------------------------------------------------
# Ultravox
#
# Docs verified 2026-09:
#   https://docs.ultravox.ai/api-reference/calls/calls-post
#   https://docs.ultravox.ai/apps/websockets
#   https://docs.ultravox.ai/apps/datamessages
# ---------------------------------------------------------------------------

ULTRAVOX_BASE_URL = "https://api.ultravox.ai"


@register("s2s", "ultravox")
class UltravoxS2S(S2SProvider):
    """Ultravox over its ``serverWebSocket`` medium -- no vendor SDK.

    Ultravox is a two-step handshake, which is the part worth reading twice:

    1. ``POST https://api.ultravox.ai/api/calls`` with an ``X-API-Key`` header
       and ``medium: {"serverWebSocket": {...}}`` in the body. Unlike the
       WebRTC path, the sample rates are *negotiated here*, not on the socket:
       ``inputSampleRate`` is required and ``outputSampleRate`` defaults to it.
       This adapter asks for 16 kHz on both sides, which is the stack's
       canonical wire format, so in the common case no resampling happens at
       all. The response carries ``joinUrl`` and ``callId``.
    2. Connect to ``joinUrl``. It is already authenticated (it embeds the
       call's credentials), so no auth header goes on the socket.

    On the wire, audio is **raw binary** s16le PCM in both directions -- no
    base64, no envelope. Text frames are "data messages": JSON with camelCase
    keys and a snake_case ``type``. This adapter handles ``transcript`` (the
    transcript accumulator), ``ping`` (answered with ``pong``, keeping the
    socket alive), and ignores the rest. ``playback_clear_buffer`` is
    deliberately *not* acted on here: it means "drop un-played bot audio
    because the caller interrupted", and this provider does not own the
    playback buffer -- ``PlaybackLedger`` and the transport do.

    ``clientBufferSizeMs`` is set high (30 s) per Ultravox's own guidance,
    which is only safe *because* barge-in is reconciled upstream.

    Usage is **estimated** (``estimated=True``). Ultravox bills per minute of
    call time, but nothing on the socket reports the billed duration, so the
    honest client-side number is socket lifetime -- a proxy, not a
    measurement, and it is labelled as one. An operator who needs the exact
    figure should read ``billingStatus`` from ``GET /api/calls/{callId}``
    after the call; ``call_id`` below is kept for exactly that.
    """

    name = "ultravox"

    def __init__(
        self,
        api_key_env: str = "ULTRAVOX_API_KEY",
        model: str = "ultravox-v0.7",
        voice: str = "Mark",
        system_prompt: str | None = None,
        language_hint: str | None = None,
        temperature: float | None = None,
        input_sample_rate: int = WIRE_SAMPLE_RATE,
        output_sample_rate: int = WIRE_SAMPLE_RATE,
        client_buffer_size_ms: int = 30000,
        base_url: str = ULTRAVOX_BASE_URL,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        connect: WebSocketConnect | None = None,
        **_: Any,
    ) -> None:
        self._api_key_env = api_key_env
        self._model = model
        self._voice = voice
        self._system_prompt = system_prompt
        self._language_hint = language_hint
        self._temperature = temperature
        self._input_sample_rate = input_sample_rate
        self._output_sample_rate = output_sample_rate
        self._client_buffer_size_ms = client_buffer_size_ms
        self._base_url = base_url.rstrip("/")
        # Test-only seams: an httpx.MockTransport for the REST leg, a scripted
        # socket for the websocket leg. Production leaves both None.
        self._transport = transport
        self._connect: WebSocketConnect = connect or _websockets_connect
        self.call_id: str | None = None
        # Keyed by the vendor's ``ordinal`` so a later delta or a corrected
        # full ``text`` lands on the right turn regardless of arrival order.
        self._turns: dict[int, dict[str, Any]] = {}

    # -- request construction -------------------------------------------------

    def build_call_request(self) -> dict[str, Any]:
        """The ``POST /api/calls`` body. Split out so a test can assert the
        exact ``serverWebSocket`` medium without a network call."""
        body: dict[str, Any] = {
            "model": self._model,
            "voice": self._voice,
            "medium": {
                "serverWebSocket": {
                    "inputSampleRate": self._input_sample_rate,
                    "outputSampleRate": self._output_sample_rate,
                    "clientBufferSizeMs": self._client_buffer_size_ms,
                }
            },
        }
        if self._system_prompt:
            body["systemPrompt"] = self._system_prompt
        if self._language_hint:
            body["languageHint"] = self._language_hint
        if self._temperature is not None:
            body["temperature"] = self._temperature
        return body

    def build_call_headers(self, api_key: str) -> dict[str, str]:
        return {"X-API-Key": api_key, "content-type": "application/json"}

    async def create_call(self) -> str:
        """Run the REST leg and return the ``joinUrl``."""
        try:
            import httpx as _httpx
        except ImportError as exc:  # pragma: no cover - exercised by install shape
            raise ImportError(
                "UltravoxS2S needs the 'httpx' package. Install the optional "
                f"cloud extra: {_CLOUD_EXTRA}"
            ) from exc

        api_key = _require_env(self._api_key_env, "UltravoxS2S")
        url = f"{self._base_url}/api/calls"
        async with _httpx.AsyncClient(transport=self._transport, timeout=30.0) as client:
            response = await client.post(
                url, headers=self.build_call_headers(api_key), json=self.build_call_request()
            )
            response.raise_for_status()
            payload = response.json()

        join_url = payload.get("joinUrl")
        if not join_url:
            raise RuntimeError(
                f"Ultravox POST {url} returned no 'joinUrl'; got keys "
                f"{sorted(payload)}. Cannot open the audio socket."
            )
        self.call_id = payload.get("callId")
        return str(join_url)

    # -- message handling -----------------------------------------------------

    def _handle_binary(self, raw: bytes) -> list[TTSChunk]:
        pcm = _resample_pcm16(raw, self._output_sample_rate, WIRE_SAMPLE_RATE)
        if not pcm:
            return []
        return [TTSChunk(frame=AudioFrame(pcm=pcm, sample_rate=WIRE_SAMPLE_RATE, channels=1))]

    def _handle_data_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Fold one JSON data message into state; return a reply to send, if any."""
        kind = message.get("type")

        if kind == "transcript":
            self._apply_transcript(message)
            return None

        if kind == "ping":
            # Answering keeps the socket from idling out on a quiet call, and
            # echoing the timestamp is what lets Ultravox measure round-trip.
            return {"type": "pong", "timestamp": message.get("timestamp")}

        if kind == "call_started":
            self.call_id = message.get("callId") or self.call_id
            return None

        # state / playback_clear_buffer / debug / client_tool_invocation are
        # real message types we deliberately do not act on here; see the class
        # docstring for why playback_clear_buffer in particular is upstream's.
        return None

    def _apply_transcript(self, message: dict[str, Any]) -> None:
        ordinal = message.get("ordinal")
        key = ordinal if isinstance(ordinal, int) else len(self._turns)
        turn = self._turns.setdefault(key, {"role": "user", "text": "", "final": False})
        # Ultravox says "agent"; the event contract (events.py) says "bot".
        turn["role"] = "bot" if message.get("role") == "agent" else "user"
        text = message.get("text")
        if text is not None:
            turn["text"] = str(text)  # a full-text message replaces
        elif message.get("delta"):
            turn["text"] = str(turn["text"]) + str(message["delta"])  # a delta appends
        turn["final"] = bool(message.get("final", turn["final"]))

    # -- the session ----------------------------------------------------------

    async def converse(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[TTSChunk]:
        join_url = await self.create_call()
        socket = await self._connect(join_url, {})
        started_at = time.monotonic()

        async def _pump() -> None:
            async for frame in frames:
                pcm = _resample_pcm16(frame.pcm, frame.sample_rate, self._input_sample_rate)
                if pcm:
                    await socket.send(pcm)  # raw s16le, no envelope
            await socket.close()

        pump_task = asyncio.create_task(_pump())
        try:
            async for raw in socket:
                if isinstance(raw, bytes):
                    for chunk in self._handle_binary(raw):
                        yield chunk
                    continue
                reply = self._handle_data_message(json.loads(raw))
                if reply is not None:
                    await socket.send(json.dumps(reply))
        finally:
            pump_task.cancel()
            with contextlib.suppress(Exception):
                await socket.close()

        yield TTSChunk(
            frame=_USAGE_ONLY_FRAME,
            usage=[
                Usage(
                    units=time.monotonic() - started_at,
                    unit_name="audio_seconds",
                    # Socket lifetime, not a vendor-reported billed duration.
                    estimated=True,
                    model=self._model,
                )
            ],
        )

    async def post_call_transcript(self) -> list[dict[str, Any]]:
        return [
            {"role": turn["role"], "text": turn["text"]}
            for _, turn in sorted(self._turns.items())
            if turn["text"]
        ]


# ---------------------------------------------------------------------------
# S2S_RATES
#
# Public list prices verified 2026-09 from the vendors' own pricing pages:
#   https://developers.openai.com/api/docs/pricing
#   https://www.ultravox.ai/pricing
#
# Snapshot, not a contract -- same caveat as cost/rates.DEFAULT_RATES. Pin your
# own RateCard with your own negotiated prices and a fresh as_of date.
# ---------------------------------------------------------------------------
S2S_RATES: list[Rate] = [
    # gpt-realtime-2.1 audio tokens. The Realtime usage block does not split
    # audio from text tokens at the top level, and an audio-only session
    # (output_modalities=["audio"], as configured above) is overwhelmingly
    # audio tokens, so the audio price is the honest single rate to carry.
    # Cached audio input is listed at $0.40/1M; Rate has no cached-price field,
    # so cached_units is reported by the provider and priced downstream.
    Rate(
        component=CostComponent.S2S,
        provider="openai_realtime",
        model="gpt-realtime-2.1",
        unit_name="tokens_in",
        price_per_unit=0.000032,  # $32.00 / 1M audio input tokens
        currency="USD",
        as_of="2026-09",
    ),
    Rate(
        component=CostComponent.S2S,
        provider="openai_realtime",
        model="gpt-realtime-2.1",
        unit_name="tokens_out",
        price_per_unit=0.000064,  # $64.00 / 1M audio output tokens
        currency="USD",
        as_of="2026-09",
    ),
    Rate(
        component=CostComponent.S2S,
        provider="ultravox",
        model=None,  # Ultravox prices per call-minute, not per model.
        unit_name="audio_seconds",
        price_per_unit=0.05 / 60,  # $0.05 / minute, pay-as-you-go and Pro
        currency="USD",
        as_of="2026-09",
    ),
]
