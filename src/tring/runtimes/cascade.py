"""CascadeRuntime — the reference STT -> LLM -> TTS pipeline.

This is the runtime everything else in Tring is measured against, and it is
written to be read. There is no framework here: an ``asyncio.Queue`` of audio
frames, three provider streams, and four primitives wired together in a way
that makes the latency behaviour visible rather than buried.

The shape of one turn
---------------------

::

    push_audio ──▶ frame queue ──▶ STTProvider.transcribe
                                        │  final STTResult
                                        ▼
                             UserTranscript · barge-in adjudication
                                        │
                                        ▼
                              LLMProvider.generate  (streaming)
                                        │  raw characters
                                        ▼
                                 SpeakToolParser
                                   │           │
                          SpeakDelta         ToolCallReady
                                   │           │
                                   ▼           ▼
                        TTSProvider      choreography.execute
                          .synthesize          │ result
                                   │           ▼
                                   │    (one follow-up LLM turn)
                                   ▼
                          on_bot_audio + PlaybackLedger

Four decisions carry most of the weight, and each is a primitive rather than a
line of prompt:

1. **The model answers in an envelope** (:data:`ENVELOPE_INSTRUCTION`), so the
   thing to say and the thing to do arrive in one stream and the parser can
   start speaking before the tool call has finished generating.
2. **Tool schemas are augmented** by ``choreography.augment_tool_schema``, so a
   tool call physically cannot be emitted without a plan for the silence it
   creates.
3. **The language directive is appended, never rewritten**, by ``LanguageLock``,
   so re-asserting it every turn costs nothing in prompt-cache hits.
4. **Generated text and played text are tracked separately** by
   ``PlaybackLedger``, so a barge-in does not leave the model believing it said
   words the caller never heard.

Concurrency model
-----------------

Two long-lived tasks, and that is all:

* ``_stt_task`` drives the transcript stream for the whole call.
* ``_turn_task`` is the *current* bot turn. A new final transcript cancels it,
  which is what makes barge-in real: the caller talking over the bot stops
  generation and synthesis instead of queueing behind them.

Everything else is a short-lived child of one of those two.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any

from tring.agent import ProviderSelection, ToolDef
from tring.cost.meter import CostMeter
from tring.events import (
    CostComponent,
    SessionEnded,
    SessionError,
    SessionStarted,
    UserTranscript,
)
from tring.primitives.choreography import (
    ChoreographyError,
    augment_tool_schema,
    execute,
    parse_choreographed_call,
)
from tring.primitives.interruption import InterruptionVerdict, PlaybackLedger
from tring.primitives.language_lock import LanguageLock
from tring.primitives.speak_parser import (
    FallbackText,
    ParserEvent,
    SpeakDelta,
    SpeakToolParser,
    ToolCallReady,
)
from tring.providers import registry
from tring.providers.base import LLMProvider, STTProvider, TTSProvider, Usage
from tring.runtimes.base import AudioFrame, RuntimeAdapter, RuntimeCapabilities
from tring.runtimes.turn_taking import TurnTaker, concat_frames
from tring.session import CallSession
from tring.vad import VoiceActivityDetector

#: The output contract every cascade turn is generated against.
#:
#: This lives here, next to the parser that consumes it, because an envelope
#: instruction and its parser are one artifact split across two files: change
#: the wording of one and you have silently changed the other's input. Keeping
#: them apart is how prompt/parser drift happens.
#:
#: The last bullet is the load-bearing one. Everything the model emits before
#: ``speak`` is latency the caller experiences as dead air, because no audio
#: can exist until the first speakable character does.
ENVELOPE_INSTRUCTION = """\
OUTPUT FORMAT (strict)

Reply with exactly one JSON object and nothing else: no prose before it, no
markdown fence around it, no second object after it.

{"speak": "<exactly what the caller should hear>", "tool_call": null}

- "speak" is read aloud verbatim by a speech engine. Write words, not markup:
  no bullet points, no asterisks, no emoji, no URLs, no code. Write numbers,
  dates and abbreviations the way a person would say them out loud.
- "speak" is never empty while the caller is waiting on you. Silence on a
  phone call reads as a dropped line, not as thinking.
- Set "tool_call" to null unless you are calling a tool this turn. To call one:

  {"speak": "<what the caller hears around the tool running>",
   "tool_call": {"name": "<tool name>", "arguments": {...}}}

- "arguments" must satisfy that tool's schema. Every tool schema includes
  waiting_message, spoken_mode and post_tool_response, and all three are
  required: a tool call with no plan for what the caller hears while the tool
  runs is not a valid tool call.
- Put "speak" first in the object. Its characters are streamed to the speaker
  as you generate them, so every character you emit before it is a character
  of delay the caller hears as silence."""

#: Assumed speech rate, used to convert generated audio back into "how many
#: characters has the caller actually heard by now". See
#: :meth:`_BotSpeech._played_chars` for why this approximation exists and what
#: replaces it when a transport can report its real playout clock.
SPOKEN_CHARS_PER_SECOND = 15.0

_BYTES_PER_SAMPLE = 2  # 16-bit linear PCM, the AudioFrame contract


def _frame_seconds(frame: AudioFrame) -> float:
    """Duration of one PCM frame in seconds."""
    denom = frame.sample_rate * _BYTES_PER_SAMPLE * frame.channels
    return len(frame.pcm) / denom if denom else 0.0


def _options_for(selection: ProviderSelection, slot: str) -> dict[str, Any]:
    """Split a ``ProviderSelection.options`` dict into per-slot options.

    One ``options`` dict serves three providers, so it needs a rule for who
    gets what. The rule: a key named after a slot (``stt`` / ``llm`` / ``tts``)
    whose value is a dict belongs to that slot alone; everything else is shared
    by all three.

    That keeps the common case trivial -- ``options: {language: "hi"}`` reaches
    every provider -- while still letting a spec say ``options: {llm: {model:
    "llama3.1"}}`` without accidentally handing ``model`` to the STT engine,
    where it would mean something completely different.
    """
    slots = ("stt", "llm", "tts", "s2s")
    shared = {k: v for k, v in selection.options.items() if k not in slots}
    specific = selection.options.get(slot, {})
    if not isinstance(specific, dict):
        return shared
    return {**shared, **specific}


class _BotSpeech:
    """One bot utterance in flight: text streams in, audio frames stream out.

    This is the async-generator bridge between two stages that disagree about
    shape. The parser hands over *discrete events* as they are decoded; the TTS
    provider wants a single ``AsyncIterator[str]`` it can consume for the whole
    utterance. A queue in the middle reconciles them, and a pump task keeps the
    audio side draining while the LLM side is still generating -- which is the
    entire point. Collecting the text first and synthesizing after would be
    three lines shorter and would add the model's full generation time to every
    caller's wait.

    **Ledger timing.** The ledger is armed in :meth:`finish`, once the ``speak``
    string is known to be complete, not at the first delta -- ``BotUtterance``
    carries the whole generated utterance by definition. Frames emitted before
    that point are simply unaccounted: ``mark_played`` takes an *absolute*
    character count and ignores unknown utterance ids, so the first call after
    arming catches the ledger up in one step with no bookkeeping of its own.
    """

    def __init__(self, runtime: CascadeRuntime, utterance_id: str) -> None:
        self._runtime = runtime
        self.utterance_id = utterance_id
        self.text = ""
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._audio_seconds = 0.0
        self._pump: asyncio.Task[None] = asyncio.create_task(self._drain_audio())

    def push(self, text: str) -> None:
        """Hand newly decoded speech to the synthesizer immediately."""
        if not text:
            return
        self.text += text
        self._queue.put_nowait(text)

    async def finish(self) -> None:
        """Close the text side, arm the ledger, and let the audio drain out."""
        if self.text:
            self._runtime.ledger.utterance_started(self.utterance_id, self.text)
        self._queue.put_nowait(None)
        await self._pump
        if self.text:
            # Nothing cancelled us, so every character was synthesized and
            # handed to the transport: the caller heard the whole utterance.
            self._runtime.ledger.mark_played(self.utterance_id, len(self.text))
            self._runtime.ledger.utterance_finished(self.utterance_id)

    async def abort(self) -> None:
        """Stop synthesis mid-utterance (the caller barged in, or we are shutting down).

        The ledger is deliberately *not* closed out here: whatever was marked
        played stays marked, and the utterance stays "current" so the next
        ``caller_started_speaking()`` can adjudicate it as a real barge-in.
        """
        self._pump.cancel()
        with suppress(asyncio.CancelledError):
            await self._pump

    async def _text_stream(self) -> AsyncIterator[str]:
        """The provider-facing view of the queue: one async iterator of text."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            yield item

    async def _drain_audio(self) -> None:
        runtime = self._runtime
        assert runtime._tts is not None  # started before any utterance exists
        async for chunk in runtime._tts.synthesize(self._text_stream(), runtime.voice):
            runtime._record_usage(CostComponent.TTS, runtime._tts_name, chunk.usage)
            runtime._emit_audio(chunk.frame)
            self._audio_seconds += _frame_seconds(chunk.frame)
            runtime.ledger.mark_played(self.utterance_id, self._played_chars())

    def _played_chars(self) -> int:
        """Approximate how much of ``text`` the caller has actually heard.

        The honest version of this number comes from the transport: it knows
        when a frame left the speaker, and it can call ``mark_played`` itself
        (``runtime.ledger`` is public for exactly that). With no playout clock
        available, v0.1 assumes a frame handed to ``on_bot_audio`` is a frame
        heard, and converts audio time to characters at a fixed speech rate.

        This is the same ratio as ``len(text) * played_seconds / total_seconds``
        with the total eliminated -- which matters, because the total duration
        of an utterance is not known until its last frame exists, and we need
        this number *during* synthesis, including on the cancellation path
        where the last frame never arrives.

        The error is real: speech rate varies by language, voice and speed, so
        the split point can be off by a word. It is snapped to a word boundary
        downstream by the ledger, and being off by a word in what the model
        believes the caller heard is a far smaller error than assuming the
        caller heard everything that was generated.
        """
        return min(len(self.text), round(self._audio_seconds * SPOKEN_CHARS_PER_SECOND))


class CascadeRuntime(RuntimeAdapter):
    """STT -> LLM -> TTS conversation runtime.

    Args:
        session: the live call this runtime drives; it owns the event bus.
        handlers: tool implementations keyed by ``ToolDef.handler`` (falling
            back to ``ToolDef.name``). Specs stay serializable precisely
            because they carry a handler *name*; binding the callable happens
            here, at the last possible moment.
        meter: optional :class:`~tring.cost.meter.CostMeter`. Every provider
            ``Usage`` is routed to it as the stage that produced it completes,
            so the cost ledger is correct even on a call that ends mid-turn.
        language: language to route providers and the language lock with.
            Defaults to the agent's primary language.
        vad: optional :class:`~tring.vad.VoiceActivityDetector`. ``None`` (the
            default) keeps the old behaviour: barge-in is adjudicated only
            once a full caller utterance has come back as a *final*
            ``STTResult``, so the bot keeps talking for however long the STT
            provider takes to recognise it was cut off. Passing a VAD hands
            caller audio to a :class:`~tring.runtimes.turn_taking.TurnTaker`
            first: it segments utterances itself (feed the STT provider's own
            endpointer disabled, e.g. ``silence_seconds=0`` for
            ``FasterWhisperSTT``, or it will re-cut audio already cut) and
            fires ``on_interrupt`` the instant speech starts, which is what
            makes barge-in cut the bot off *while the caller is still
            talking* instead of after they finish. The eventual final
            transcript still runs the old ``caller_started_speaking()`` call
            in :meth:`_begin_turn`; by then the ledger has already retired
            the utterance the VAD adjudicated, so that second call is a
            harmless no-op that reports ordinary turn-taking rather than a
            second interruption (see ``PlaybackLedger.caller_started_speaking``
            and the "Adopting it in a cascade runtime" note on
            :class:`~tring.runtimes.turn_taking.TurnTaker`).

    Bot audio is delivered through ``on_bot_audio``, set by the transport.
    """

    def __init__(
        self,
        session: CallSession,
        handlers: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]] | None = None,
        meter: CostMeter | None = None,
        language: str | None = None,
        vad: VoiceActivityDetector | None = None,
    ) -> None:
        super().__init__(session)
        self.handlers = dict(handlers or {})
        self.meter = meter
        self.language = language or session.agent.language.primary
        self.voice: str | None = None

        #: Public so a transport with a real playout clock can drive
        #: ``mark_played`` itself instead of relying on this runtime's estimate.
        self.ledger = PlaybackLedger(session)

        #: ``None`` unless ``vad`` was supplied; see the class docstring.
        #: Ownership of caller-audio segmentation moves to it wholesale so
        #: there is exactly one endpointer per call, never the VAD's and the
        #: STT provider's both racing to decide where an utterance ends.
        self._turn_taker: TurnTaker | None = None
        if vad is not None:
            self._turn_taker = TurnTaker(
                vad=vad,
                ledger=self.ledger,
                on_utterance=self._deliver_vad_utterance,
                on_interrupt=self._on_vad_interrupt,
            )

        self._lock = LanguageLock(session.agent.language)
        self._tools: dict[str, ToolDef] = {t.name: t for t in session.agent.tools}
        self._tool_schemas: list[dict[str, Any]] = []
        self._messages: list[dict[str, Any]] = []

        self._stt: STTProvider | None = None
        self._llm: LLMProvider | None = None
        self._tts: TTSProvider | None = None
        self._stt_name = "stt"
        self._llm_name = "llm"
        self._tts_name = "tts"

        # ``None`` is the shutdown sentinel: it unblocks the frame iterator so
        # the STT provider's stream can end normally rather than be cancelled
        # mid-utterance, which would throw away the caller's last sentence.
        self._frames: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._stt_task: asyncio.Task[None] | None = None
        self._turn_task: asyncio.Task[None] | None = None
        self._speech: _BotSpeech | None = None
        self._tool_calls = 0
        self._stopped = False

    # ------------------------------------------------------------- contract

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            live_transcripts=True,  # STT produces transcripts mid-turn
            mid_call_tool_calls=True,
            barge_in=True,  # a new final transcript cancels the bot turn
            # Every local provider reports vendor/runtime-measured units, and
            # the cloud wrappers are held to the same contract. A provider that
            # can only estimate says so on its own Usage lines; it does not
            # change what this runtime is capable of reporting.
            exact_usage_reporting=True,
            local_capable=True,  # faster_whisper + ollama + kokoro, no cloud
        )

    async def start(self) -> None:
        """Bind providers, build the stable prompt prefix, greet the caller."""
        registry._load_builtin()
        agent = self.session.agent
        selection = agent.runtime.select(self.language)
        self._bind_providers(selection)
        self._build_base_messages()

        self.session.emit(
            SessionStarted(
                session_id=self.session.session_id,
                at=self.session.elapsed,
                agent_name=agent.name,
                runtime_mode=agent.runtime.mode.value,
            )
        )

        self._stt_task = asyncio.create_task(self._consume_transcripts())

        if agent.greeting:
            # Spoken before the caller has said anything, so it is a plain
            # utterance rather than a turn: no LLM call, no parser, no cost
            # beyond TTS. Awaited here so ``start()`` returning means the
            # caller has been greeted.
            await self._speak_text(agent.greeting)

    async def push_audio(self, frame: AudioFrame) -> None:
        """Feed caller audio in.

        With no VAD this only queues the frame, and never blocks: the queue
        absorbs the jitter. With a VAD the frame is routed through the
        :class:`~tring.runtimes.turn_taking.TurnTaker` instead, which does its
        own buffering (pre-roll while idle, the in-progress utterance while
        speaking) and only reaches the queue once a whole utterance is ready;
        that buffering can itself await, unlike the plain queue path.
        """
        if self._stopped:
            return
        if self._turn_taker is not None:
            await self._turn_taker.push(frame)
        else:
            self._frames.put_nowait(frame)

    async def _deliver_vad_utterance(self, frames: list[AudioFrame]) -> None:
        """``TurnTaker.on_utterance``: hand one segmented utterance to STT.

        Joined into a single frame because the STT provider still reads a
        plain frame queue; the join is where a stream of raw mic frames turns
        into a stream of pre-segmented utterances without changing anything
        downstream of ``_audio_frames``.
        """
        frame = concat_frames(frames)
        if frame is not None:
            self._frames.put_nowait(frame)

    async def _on_vad_interrupt(self, verdict: InterruptionVerdict) -> None:
        """``TurnTaker.on_interrupt``: cut the bot off the instant speech starts.

        This is the entire point of wiring a VAD in: without it, the earliest
        anything can react to a barge-in is ``_begin_turn``, which only runs
        once the STT provider has finished recognising a *final* transcript.
        The VAD's ``SPEECH_START`` fires as soon as the caller opens their
        mouth, so the annotation this produces is already in ``_messages`` by
        the time that final transcript arrives and calls
        ``caller_started_speaking()`` again on an already-retired utterance.
        """
        await self._cancel_turn()
        if verdict.context_annotation:
            self._messages.append({"role": "system", "content": verdict.context_annotation})

    async def stop(self, reason: str = "completed") -> None:
        """Cancel in-flight work and emit ``SessionEnded``. Idempotent."""
        if self._stopped:
            return
        self._stopped = True

        await self._cancel_turn()
        if self._turn_taker is not None:
            # A caller cut off mid-word still said something; flush delivers
            # whatever the VAD had open instead of dropping it on the floor.
            await self._turn_taker.flush()
        self._frames.put_nowait(None)  # let the STT stream end, not abort
        if self._stt_task is not None:
            with suppress(asyncio.CancelledError):
                await self._stt_task
            self._stt_task = None

        duration = self.session.elapsed
        self.session.emit(
            SessionEnded(
                session_id=self.session.session_id,
                at=duration,
                reason=reason,
                duration_seconds=duration,
            )
        )

    async def drain(self) -> None:
        """Wait until pushed audio has been transcribed and its bot turn finished.

        Not part of the ``RuntimeAdapter`` contract -- a real call never drains,
        it just keeps going -- but a console dev loop and a test both need to
        know "the agent has finished replying to what I just said".

        It is an exact barrier rather than a poll, and the reason is worth
        knowing: ``task_done`` for a frame fires only when the STT provider
        comes back for the *next* frame, which it can only do after this
        runtime has already handled the transcript that frame produced. So
        ``join()`` returning implies the turn task for the last frame exists.
        """
        await self._frames.join()
        task = self._turn_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    # ------------------------------------------------------------- start-up

    def _bind_providers(self, selection: ProviderSelection) -> None:
        for slot in ("stt", "llm", "tts"):
            if getattr(selection, slot) is None:
                raise ValueError(
                    f"cascade runtime needs a {slot} provider: set "
                    f"runtime.routing[...].{slot} in the agent spec "
                    f"(registered {slot} providers: "
                    f"{[name for _, name in registry.available(slot)]})"
                )
        self._stt_name = str(selection.stt)
        self._llm_name = str(selection.llm)
        self._tts_name = str(selection.tts)
        self._stt = registry.create("stt", self._stt_name, **_options_for(selection, "stt"))
        self._llm = registry.create("llm", self._llm_name, **_options_for(selection, "llm"))
        self._tts = registry.create("tts", self._tts_name, **_options_for(selection, "tts"))
        self.voice = _options_for(selection, "tts").get("voice")

    def _build_base_messages(self) -> None:
        """Assemble the prompt prefix that never changes for the rest of the call.

        Order matters for prompt caching, not for the model: persona, then
        output contract, then tool schemas, all fixed for the session. The
        language directive is deliberately *absent* -- ``LanguageLock`` appends
        it after the conversation on every turn, so this prefix stays
        byte-identical and cacheable while still being re-asserted each time.
        """
        agent = self.session.agent
        self._tool_schemas = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": augment_tool_schema(tool),
            }
            for tool in agent.tools
        ]

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": agent.persona},
            {"role": "system", "content": ENVELOPE_INSTRUCTION},
        ]
        if self._tool_schemas:
            # Also rendered into the prompt, not only passed as native tool
            # definitions: the envelope means the model emits tool calls as
            # ordinary JSON text, and plenty of small local models have no
            # native tool-calling channel at all.
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "TOOLS AVAILABLE (JSON Schema for each tool's "
                        '"arguments"):\n'
                        + json.dumps(self._tool_schemas, indent=2, ensure_ascii=False)
                    ),
                }
            )
        self._messages = messages

    # --------------------------------------------------------- the STT side

    async def _audio_frames(self) -> AsyncIterator[AudioFrame]:
        """The provider-facing view of the frame queue."""
        while True:
            frame = await self._frames.get()
            if frame is None:
                self._frames.task_done()
                return
            try:
                yield frame
            finally:
                self._frames.task_done()

    async def _consume_transcripts(self) -> None:
        """Drive the transcript stream for the whole call."""
        assert self._stt is not None
        try:
            async for result in self._stt.transcribe(self._audio_frames(), self.language):
                self._record_usage(CostComponent.STT, self._stt_name, result.usage)
                self.session.emit(
                    UserTranscript(
                        session_id=self.session.session_id,
                        at=self.session.elapsed,
                        text=result.text,
                        final=result.final,
                        language=result.language or self.language,
                    )
                )
                if result.final:
                    await self._begin_turn(result.text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # a provider fault must not kill the call silently
            self._emit_error(f"speech recognition failed: {exc}")

    async def _begin_turn(self, text: str) -> None:
        """Adjudicate the barge-in, extend the history, start the bot's reply.

        The order here is the whole interruption story. The previous turn is
        cancelled *first*, so nothing more is generated or spoken over the
        caller. Only then is the ledger asked whether that was a genuine
        barge-in or ordinary turn-taking, and only a genuine one puts an
        annotation into the context.
        """
        await self._cancel_turn()

        verdict = self.ledger.caller_started_speaking()
        if verdict.genuine and verdict.context_annotation:
            self._messages.append({"role": "system", "content": verdict.context_annotation})

        self._messages.append({"role": "user", "content": text})
        self._turn_task = asyncio.create_task(self._run_turn())

    async def _cancel_turn(self) -> None:
        """Stop the in-flight bot turn, if any, and its synthesis with it."""
        if self._speech is not None:
            await self._speech.abort()
            self._speech = None
        task, self._turn_task = self._turn_task, None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    # -------------------------------------------------------- the turn loop

    async def _run_turn(self, follow_up: bool = False) -> None:
        """Generate one bot turn: stream to speech, then run any tool it asked for.

        ``follow_up`` marks the single extra turn a tool result is allowed to
        trigger. It is a hard stop rather than a loop: a model that keeps
        calling tools has plenty of ways to talk itself into an unbounded chain,
        and on a phone call the caller pays for every link in real time. Deeper
        chains belong to an explicit agent loop, not to a turn handler.
        """
        try:
            tool_call = await self._generate_and_speak()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit_error(f"language model turn failed: {exc}")
            return

        if tool_call is None or follow_up:
            return
        await self._run_tool_call(tool_call)

    async def _generate_and_speak(self) -> dict[str, Any] | None:
        """Stream one completion through the parser into TTS.

        Returns the ``tool_call`` object if the model emitted one.
        """
        assert self._llm is not None
        speech = _BotSpeech(self, uuid.uuid4().hex)
        self._speech = speech
        parser = SpeakToolParser()
        tool_call: dict[str, Any] | None = None

        try:
            messages = self._lock.apply(self._messages, self.language)
            # tools=None on purpose: the envelope IS the tool channel. Schemas
            # are already rendered into the system prompt, and the model emits
            # tool calls as ordinary text for the speak-parser. Passing our
            # bare schema dicts through a provider's native `tools` parameter
            # is rejected by every vendor (each expects its own wrapper shape,
            # verified live against Groq and Gemini on 2026-09-18), and even
            # where it worked it would race a second, unparsed tool channel
            # against the envelope.
            async for chunk in self._llm.generate(messages, None):
                self._record_usage(CostComponent.LLM, self._llm_name, chunk.usage)
                for event in parser.feed(chunk.text):
                    tool_call = self._apply_parser_event(event, speech) or tool_call
            for event in parser.finalize():
                tool_call = self._apply_parser_event(event, speech) or tool_call
            await speech.finish()
        finally:
            # Cleared even on cancellation so a barge-in's ``_cancel_turn``
            # does not try to abort an utterance that already finished.
            if self._speech is speech:
                self._speech = None

        # The history keeps the same envelope shape the model was asked to
        # produce. Storing only the spoken half would teach it, turn by turn,
        # that the format is optional.
        self._messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"speak": speech.text, "tool_call": tool_call}, ensure_ascii=False
                ),
            }
        )
        return tool_call

    def _apply_parser_event(
        self, event: ParserEvent, speech: _BotSpeech
    ) -> dict[str, Any] | None:
        """Route one parser event. Returns a tool call when the event is one.

        ``FallbackText`` is spoken like ordinary speech on purpose: it means the
        model ignored the envelope, and a slightly odd-sounding reply beats a
        silent call while someone is holding a phone to their ear.
        """
        if isinstance(event, SpeakDelta | FallbackText):
            speech.push(event.text)
            return None
        if isinstance(event, ToolCallReady):
            return event.call
        return None

    # ------------------------------------------------------------ the tools

    async def _run_tool_call(self, raw_call: dict[str, Any]) -> None:
        """Validate, execute and (if the choreography says so) respond to a tool call."""
        name, arguments = _split_tool_call(raw_call)
        tool = self._tools.get(name)
        if tool is None:
            self._emit_error(f"model called unknown tool {name!r}")
            await self._respond_to_tool(
                name, {"ok": False, "error": f"no such tool: {name}"}
            )
            return

        limit = self.session.agent.limits.max_tool_calls
        if limit is not None and self._tool_calls >= limit:
            self._emit_error(f"tool-call limit of {limit} reached; refusing {name!r}")
            await self._respond_to_tool(
                name,
                {"ok": False, "error": f"tool-call limit of {limit} reached for this call"},
            )
            return
        self._tool_calls += 1

        arguments.setdefault("call_id", uuid.uuid4().hex)
        try:
            call = parse_choreographed_call(tool, arguments)
        except ChoreographyError as exc:
            # The message is written to be replayed to the model verbatim, so
            # the follow-up turn is the model's chance to correct itself.
            self._emit_error(str(exc))
            await self._respond_to_tool(name, {"ok": False, "error": str(exc)})
            return

        outcome = await execute(
            tool_def=tool,
            call=call,
            handler=self._resolve_handler(tool),
            session=self.session,
            speak=self._speak_text,
        )
        if outcome.should_respond:
            await self._respond_to_tool(
                tool.name,
                {"ok": outcome.ok, "result": outcome.result, "error": outcome.error},
            )

    def _resolve_handler(self, tool: ToolDef) -> Callable[[dict[str, Any]], Awaitable[Any]]:
        """Bind a spec's handler *name* to a real callable.

        An unbound handler returns a raising stub rather than blowing up here,
        so the miss travels the normal tool-failure path: ``ToolCallCompleted``
        with ``ok=False``, ``should_respond=True``, and the model gets to
        apologize instead of the caller hearing the line go dead.
        """
        handler = self.handlers.get(tool.handler or tool.name)
        if handler is not None:
            return handler

        async def _unbound(_arguments: dict[str, Any]) -> Any:
            raise LookupError(
                f"no handler bound for tool {tool.name!r}; pass "
                f'CascadeRuntime(..., handlers={{"{tool.handler or tool.name}": fn}})'
            )

        return _unbound

    async def _respond_to_tool(self, tool_name: str, payload: dict[str, Any]) -> None:
        """Append a tool result and run the one permitted follow-up turn."""
        self._messages.append(
            {
                "role": "tool",
                "name": tool_name,
                "content": json.dumps(payload, ensure_ascii=False, default=str),
            }
        )
        await self._run_turn(follow_up=True)

    # ----------------------------------------------------------- small bits

    async def _speak_text(self, text: str) -> None:
        """Speak a fixed line (greeting, waiting message) as its own utterance.

        Fixed lines skip the LLM and the parser entirely but keep every other
        stage, so they show up in the event stream, the ledger and the cost
        report exactly like generated speech does. A waiting message the caller
        heard is speech, whoever wrote it.
        """
        if not text.strip():
            return
        speech = _BotSpeech(self, uuid.uuid4().hex)
        speech.push(text)
        await speech.finish()

    def _emit_audio(self, frame: AudioFrame) -> None:
        callback = self.on_bot_audio
        if callable(callback):
            callback(frame)

    def _record_usage(
        self, component: CostComponent, provider: str, usage: list[Usage]
    ) -> None:
        """Route provider-reported usage to the meter, if one is attached.

        Called as each stage completes rather than at end of call, so a call
        that drops mid-turn still has an accurate ledger for everything that
        actually ran.
        """
        if self.meter is None:
            return
        for line in usage:
            self.meter.record(component, provider, line)

    def _emit_error(self, message: str, recoverable: bool = True) -> None:
        self.session.emit(
            SessionError(
                session_id=self.session.session_id,
                at=self.session.elapsed,
                message=message,
                recoverable=recoverable,
            )
        )


def _split_tool_call(raw: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Pull ``(name, arguments)`` out of the model's ``tool_call`` object.

    The documented shape is ``{"name": ..., "arguments": {...}}``, but models
    routinely flatten it to ``{"name": ..., <args inline>}``. Both are accepted:
    rejecting the flattened form would trade a working call for a retry, and
    the distinction carries no information we need.
    """
    name = str(raw.get("name") or raw.get("tool") or raw.get("tool_name") or "")
    arguments = raw.get("arguments")
    if isinstance(arguments, dict):
        return name, dict(arguments)
    return name, {k: v for k, v in raw.items() if k not in ("name", "tool", "tool_name")}


__all__ = ["ENVELOPE_INSTRUCTION", "SPOKEN_CHARS_PER_SECOND", "CascadeRuntime"]
