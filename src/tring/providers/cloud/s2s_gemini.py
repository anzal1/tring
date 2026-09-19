"""Gemini Live API speech-to-speech adapter.

Google's Live API is a bidirectional websocket ("BidiGenerateContent"): one
socket, audio in, audio out, tool calls inline. It is the third stateful
websocket provider in this package, so it follows the same shape as the two
in ``providers/cloud/s2s_extra.py`` -- a pump task drains the caller frames
into the socket while the generator body reads the socket and yields -- and
imports that module's connect seam, env-var helper and resampler rather than
re-deriving them. Those names are underscore-private because they are package
internals, not public API; importing them across two sibling modules in the
same package is the point of that privacy, and a second copy of a resampler
is a second thing to keep in sync.

Protocol, verified 2026-09 against the public docs:

* Endpoint and handshake --
  https://ai.google.dev/gemini-api/docs/live-api/get-started-websocket
  ``wss://generativelanguage.googleapis.com/ws/
  google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent
  ?key=<API_KEY>``. The key travels as a query parameter because that is the
  only authentication the documented websocket handshake accepts; there is no
  documented header form for this endpoint.
* Setup message -- https://ai.google.dev/api/live (BidiGenerateContentSetup).
  ``{"setup": {"model": "models/<model>", "generationConfig": {...},
  "systemInstruction": {...}, "tools": [...], "inputAudioTranscription": {},
  "outputAudioTranscription": {}}}``. ``responseModalities`` is a field of
  ``generationConfig``, not of ``setup``. The server answers ``setupComplete``.
* Audio in -- https://ai.google.dev/gemini-api/docs/live-guide
  ``{"realtimeInput": {"audio": {"mimeType": "audio/pcm;rate=16000",
  "data": "<base64>"}}}``. "Audio data in the Live API is always raw,
  little-endian, 16-bit PCM. Input audio is natively 16kHz" -- which is this
  stack's canonical wire format, so the input edge needs no resampling at all.
* Audio out -- same guide: "Audio output always uses a sample rate of 24kHz",
  delivered as base64 in ``serverContent.modelTurn.parts[].inlineData.data``.
  That edge *is* resampled, to 16 kHz, so every frame this module yields is
  stamped with the canonical rate like every other provider's.
* Transcripts -- ``serverContent.inputTranscription.text`` (the caller) and
  ``serverContent.outputTranscription.text`` (the model), both opt-in via the
  two ``...AudioTranscription`` setup fields.
* Tools -- ``toolCall.functionCalls[]`` in, ``{"toolResponse":
  {"functionResponses": [{"id", "name", "response"}]}}`` out.
* Usage -- ``usageMetadata`` (https://ai.google.dev/api/live#UsageMetadata).

**Deliberate skips**, because the public docs do not pin them down:

* *Voice selection.* ``generationConfig.speechConfig`` exists, but the API
  reference does not document its fields and the speech-generation guide shows
  a shape (``"speech_config": [{"voice": "Kore"}]``) that contradicts the
  older nested ``voiceConfig.prebuiltVoiceConfig.voiceName`` form. Rather than
  guess and ship a silently-ignored voice option, ``speech_config`` is a
  verbatim passthrough: whatever dict you supply is placed at
  ``generationConfig.speechConfig`` unchanged.
* *Timed transcripts.* ``BidiGenerateContentTranscription`` carries ``text``
  and ``languageCode`` and no timestamps, so ``post_call_transcript`` turns
  carry no ``at`` key and ``S2SRuntime`` stamps them with its own clock (see
  ``runtimes/s2s.reconcile_transcript``). Inventing an offset from local
  arrival time would be a measurement of our socket, labelled as the vendor's.
* *Error events.* The documented server message union has no error member;
  Live API failures close the socket with a status code, which surfaces as the
  ``websockets`` exception from the read loop rather than as a parsed event.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any
from urllib.parse import quote

from tring.cost.rates import Rate
from tring.events import CostComponent
from tring.providers.base import S2SProvider, TTSChunk, Usage
from tring.providers.cloud.s2s_extra import (
    _USAGE_ONLY_FRAME,
    WIRE_SAMPLE_RATE,
    WebSocketConnect,
    _require_env,
    _resample_pcm16,
    _websockets_connect,
)
from tring.providers.registry import register
from tring.runtimes.base import AudioFrame

GEMINI_LIVE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)

#: "Input audio is natively 16kHz" -- identical to this stack's wire format.
GEMINI_LIVE_INPUT_RATE = 16000

#: "Audio output always uses a sample rate of 24kHz."
GEMINI_LIVE_OUTPUT_RATE = 24000

#: The exact mime type the realtimeInput Blob must carry.
GEMINI_LIVE_INPUT_MIME = f"audio/pcm;rate={GEMINI_LIVE_INPUT_RATE}"

#: Called with ``(tool_name, arguments)``, returns the object to put in the
#: ``response`` field of the matching FunctionResponse. The signature is
#: exactly ``HybridRuntime.on_provider_tool_call``'s, which is the point: a
#: hybrid session wires the two together and every Gemini tool call goes
#: through the same choreography enforcement as every other vendor's.
ToolCallBridge = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]

#: Called once per completed model turn (``serverContent.turnComplete``).
#: ``HybridRuntime.on_provider_turn_complete`` is the intended consumer: turn
#: boundaries are what make its "n tool calls in m turns" policy measurable.
TurnCompleteBridge = Callable[[], Awaitable[None]]


@register("s2s", "gemini_live")
class GeminiLiveS2S(S2SProvider):
    """Google's Gemini Live API over a raw websocket -- no ``google-genai`` SDK.

    Two public callbacks extend the ``S2SProvider`` ABC without changing it,
    because the ABC's ``converse()`` can only yield audio and a live session
    has two other things to say. Both default to ``None`` and both are
    optional:

    ``on_tool_call``
        Bridges ``toolCall`` to a caller-owned handler. Left unset, a tool
        call is answered with an explicit error object rather than ignored:
        an unanswered function call leaves the model waiting on a response
        that will never arrive, which the caller hears as a dead line.

    ``on_turn_complete``
        Fires on each ``serverContent.turnComplete``.

    Usage is **exact** and reported **once**, at the end of the stream. The
    Live API "will periodically send messages that include UsageMetadata"
    whose ``totalTokenCount`` is documented as the total "for the generation
    request" -- i.e. periodic snapshots of one running total, not per-turn
    deltas. Summing every snapshot would therefore multiply-count a long call,
    so this adapter keeps the most recent block and emits that. If a future
    revision of the docs pins the semantics the other way, the fix is one
    method (:meth:`_usage_lines`) and not a redesign.
    """

    name = "gemini_live"

    def __init__(
        self,
        api_key_env: str = "GEMINI_API_KEY",
        model: str = "gemini-3.8-live",
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        input_transcription: bool = True,
        output_transcription: bool = True,
        temperature: float | None = None,
        speech_config: dict[str, Any] | None = None,
        base_url: str = GEMINI_LIVE_URL,
        *,
        connect: WebSocketConnect | None = None,
        **_: Any,
    ) -> None:
        self._api_key_env = api_key_env
        self._model = model
        self._system_instruction = system_instruction
        self._tools = list(tools) if tools else None
        self._input_transcription = input_transcription
        self._output_transcription = output_transcription
        self._temperature = temperature
        self._speech_config = speech_config
        self._base_url = base_url
        # Test-only seam (see _WebSocketLike in s2s_extra). Production leaves
        # this None and gets the lazily imported ``websockets`` connector.
        self._connect: WebSocketConnect = connect or _websockets_connect

        self.on_tool_call: ToolCallBridge | None = None
        self.on_turn_complete: TurnCompleteBridge | None = None

        #: True once the server has acknowledged the setup message.
        self.setup_complete = False
        #: Seconds of connection left, when the server sent a ``goAway``.
        self.go_away_seconds: float | None = None

        self._transcript: list[dict[str, Any]] = []
        # The turn dict currently accumulating text for each role. Transcripts
        # stream in as fragments; a turn is appended to ``_transcript`` the
        # moment its first fragment lands, so arrival order is preserved, and
        # later fragments mutate that same dict in place.
        self._open: dict[str, dict[str, Any]] = {}
        self._usage_metadata: dict[str, Any] | None = None

    # -- request construction -------------------------------------------------

    def build_url(self, api_key: str) -> str:
        """The websocket URL, key included.

        A key in a query string is not this adapter's preference; it is the
        only form the documented handshake accepts. ``quote`` is applied so a
        key containing a URL-significant character cannot silently truncate
        the query.
        """
        return f"{self._base_url}?key={quote(api_key, safe='')}"

    def build_setup(self) -> dict[str, Any]:
        """The one configuration message, split out so tests can assert on it
        without standing up a socket."""
        generation_config: dict[str, Any] = {"responseModalities": ["AUDIO"]}
        if self._temperature is not None:
            generation_config["temperature"] = self._temperature
        if self._speech_config is not None:
            # Verbatim passthrough; see the module docstring's skip note.
            generation_config["speechConfig"] = self._speech_config

        model = self._model if self._model.startswith("models/") else f"models/{self._model}"
        setup: dict[str, Any] = {"model": model, "generationConfig": generation_config}
        if self._system_instruction:
            setup["systemInstruction"] = {"parts": [{"text": self._system_instruction}]}
        if self._tools:
            setup["tools"] = list(self._tools)
        # AudioTranscriptionConfig is an empty message: its presence is the
        # opt-in. Without it the caller's half of the call never becomes text
        # and ``post_call_transcript`` returns only what the model said.
        if self._input_transcription:
            setup["inputAudioTranscription"] = {}
        if self._output_transcription:
            setup["outputAudioTranscription"] = {}
        return {"setup": setup}

    def build_realtime_input(self, frame: AudioFrame) -> dict[str, Any]:
        pcm = _resample_pcm16(frame.pcm, frame.sample_rate, GEMINI_LIVE_INPUT_RATE)
        return {
            "realtimeInput": {
                "audio": {
                    "mimeType": GEMINI_LIVE_INPUT_MIME,
                    "data": base64.b64encode(pcm).decode("ascii"),
                }
            }
        }

    def build_audio_stream_end(self) -> dict[str, Any]:
        """The documented end-of-audio marker.

        Sent before closing so the server can finish the caller's last
        utterance instead of treating the disconnect as a truncation.
        """
        return {"realtimeInput": {"audioStreamEnd": True}}

    # -- message handling -----------------------------------------------------

    def _handle_message(self, message: dict[str, Any]) -> list[TTSChunk]:
        """Fold one server message into state; return any audio it carried.

        Pure apart from the accumulators, and free of ``await``, so the whole
        message-to-chunk mapping is unit-testable with no socket. The parts
        that must talk back to the server (tool responses) live in
        :meth:`_dispatch` instead.
        """
        if "setupComplete" in message:
            self.setup_complete = True

        usage = message.get("usageMetadata")
        if isinstance(usage, dict) and usage:
            self._usage_metadata = usage

        go_away = message.get("goAway")
        if isinstance(go_away, dict):
            # "3.2s" style duration strings are the proto JSON encoding; the
            # raw value is kept as a float only when it parses cleanly, so a
            # format this adapter has not verified never becomes a fake number.
            self.go_away_seconds = _parse_duration(go_away.get("timeLeft"))

        server_content = message.get("serverContent")
        if not isinstance(server_content, dict):
            return []

        self._apply_transcription(server_content.get("inputTranscription"), "user")
        self._apply_transcription(server_content.get("outputTranscription"), "bot")

        if server_content.get("turnComplete") or server_content.get("interrupted"):
            # An interruption ends the model's turn just as firmly as
            # completion does: whatever it was mid-sentence about is over.
            self._open.clear()

        return self._audio_chunks(server_content.get("modelTurn"))

    def _apply_transcription(self, block: Any, role: str) -> None:
        if not isinstance(block, dict):
            return
        text = block.get("text")
        if not text:
            return
        turn = self._open.get(role)
        if turn is None:
            turn = {"role": role, "text": "", "language": block.get("languageCode")}
            self._open[role] = turn
            self._transcript.append(turn)
        turn["text"] = str(turn["text"]) + str(text)
        turn["language"] = block.get("languageCode") or turn["language"]

    def _audio_chunks(self, model_turn: Any) -> list[TTSChunk]:
        if not isinstance(model_turn, dict):
            return []
        chunks: list[TTSChunk] = []
        for part in model_turn.get("parts") or []:
            if not isinstance(part, dict):
                continue
            inline = part.get("inlineData")
            if not isinstance(inline, dict) or not inline.get("data"):
                continue
            raw = base64.b64decode(inline["data"])
            pcm = _resample_pcm16(raw, GEMINI_LIVE_OUTPUT_RATE, WIRE_SAMPLE_RATE)
            if pcm:
                chunks.append(
                    TTSChunk(
                        frame=AudioFrame(
                            pcm=pcm, sample_rate=WIRE_SAMPLE_RATE, channels=1
                        )
                    )
                )
        return chunks

    async def _dispatch(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Run the callbacks one message may trigger; return a reply to send."""
        server_content = message.get("serverContent")
        if (
            isinstance(server_content, dict)
            and server_content.get("turnComplete")
            and self.on_turn_complete is not None
        ):
            await self.on_turn_complete()

        tool_call = message.get("toolCall")
        if not isinstance(tool_call, dict):
            return None

        responses: list[dict[str, Any]] = []
        for call in tool_call.get("functionCalls") or []:
            if not isinstance(call, dict):
                continue
            responses.append(await self._respond_to_function_call(call))
        return {"toolResponse": {"functionResponses": responses}} if responses else None

    async def _respond_to_function_call(self, call: dict[str, Any]) -> dict[str, Any]:
        name = str(call.get("name") or "")
        arguments = call.get("args")
        bridge = self.on_tool_call
        if bridge is None:
            result: dict[str, Any] = {
                "error": (
                    f"no tool handler is bound to this {self.name} session, so "
                    f"{name!r} cannot be executed; tell the caller you cannot do it."
                )
            }
        else:
            result = await bridge(name, dict(arguments) if isinstance(arguments, dict) else {})
        response: dict[str, Any] = {"name": name, "response": result}
        if call.get("id") is not None:
            # Present only for async function calling; echoing it back is how
            # the server matches the response to the call it came from.
            response["id"] = call["id"]
        return response

    def _usage_lines(self) -> list[Usage]:
        block = self._usage_metadata
        if not block:
            # Honesty over completeness: no usage block means no usage record,
            # rather than a plausible-looking guess.
            return []
        cached = block.get("cachedContentTokenCount")
        return [
            Usage(
                units=float(block.get("promptTokenCount", 0)),
                unit_name="tokens_in",
                estimated=False,
                model=self._model,
                cached_units=None if cached is None else float(cached),
            ),
            Usage(
                units=float(block.get("responseTokenCount", 0)),
                unit_name="tokens_out",
                estimated=False,
                model=self._model,
            ),
        ]

    # -- the session ----------------------------------------------------------

    async def converse(self, frames: AsyncIterator[AudioFrame]) -> AsyncIterator[TTSChunk]:
        api_key = _require_env(self._api_key_env, "GeminiLiveS2S")
        socket = await self._connect(self.build_url(api_key), {})
        await socket.send(json.dumps(self.build_setup()))

        async def _pump() -> None:
            async for frame in frames:
                await socket.send(json.dumps(self.build_realtime_input(frame)))
            await socket.send(json.dumps(self.build_audio_stream_end()))
            # Closing is how the read loop below terminates.
            await socket.close()

        pump_task = asyncio.create_task(_pump())
        try:
            async for raw in socket:
                # The Live API frames its JSON as binary on the wire as often
                # as text, so both are decoded rather than one being skipped.
                text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                message = json.loads(text)
                for chunk in self._handle_message(message):
                    yield chunk
                reply = await self._dispatch(message)
                if reply is not None:
                    await socket.send(json.dumps(reply))
        finally:
            pump_task.cancel()
            with contextlib.suppress(Exception):
                await socket.close()

        usage = self._usage_lines()
        if usage:
            # A zero-length frame as a usage carrier: see the "Usage has
            # nowhere else to go" section of s2s_extra's module docstring.
            yield TTSChunk(frame=_USAGE_ONLY_FRAME, usage=usage)

    async def post_call_transcript(self) -> list[dict[str, Any]]:
        return [
            {"role": turn["role"], "text": turn["text"], "language": turn["language"]}
            for turn in self._transcript
            if turn["text"]
        ]


def _parse_duration(value: Any) -> float | None:
    """Parse a proto JSON duration (``"3.5s"``) into seconds, or give up.

    Giving up returns ``None``: a ``goAway`` whose encoding this adapter has
    not verified is better reported as "unknown" than as a number pulled out
    of a string that might not mean what we assumed.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str) and value.endswith("s"):
        with contextlib.suppress(ValueError):
            return float(value[:-1])
    return None


# ---------------------------------------------------------------------------
# GEMINI_LIVE_RATES
#
# Public list prices verified 2026-09 from Google's own pricing page:
#   https://ai.google.dev/gemini-api/docs/pricing
#
# Snapshot, not a contract -- same caveat as cost/rates.DEFAULT_RATES. Pin your
# own RateCard with your own negotiated prices and a fresh as_of date.
#
# Only the *audio* prices are carried. UsageMetadata's top-level counts do not
# split audio from text, and a Live session configured here for
# responseModalities=["AUDIO"] is overwhelmingly audio tokens, so the audio
# price is the honest single rate per unit_name. (Google also publishes a
# per-minute alternative for these models; Rate prices one unit, and the token
# counts are what the socket actually reports, so tokens are what is priced.)
# ---------------------------------------------------------------------------
GEMINI_LIVE_RATES: list[Rate] = [
    Rate(
        component=CostComponent.S2S,
        provider="gemini_live",
        model="gemini-3.8-live",
        unit_name="tokens_in",
        price_per_unit=0.000003,  # $3.00 / 1M audio input tokens
        currency="USD",
        as_of="2026-09",
    ),
    Rate(
        component=CostComponent.S2S,
        provider="gemini_live",
        model="gemini-3.8-live",
        unit_name="tokens_out",
        price_per_unit=0.000012,  # $12.00 / 1M audio output tokens
        currency="USD",
        as_of="2026-09",
    ),
    Rate(
        component=CostComponent.S2S,
        provider="gemini_live",
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        unit_name="tokens_in",
        price_per_unit=0.000003,  # $3.00 / 1M audio (or video) input tokens
        currency="USD",
        as_of="2026-09",
    ),
    Rate(
        component=CostComponent.S2S,
        provider="gemini_live",
        model="gemini-2.5-flash-native-audio-preview-12-2025",
        unit_name="tokens_out",
        price_per_unit=0.000012,  # $12.00 / 1M audio output tokens
        currency="USD",
        as_of="2026-09",
    ),
]
