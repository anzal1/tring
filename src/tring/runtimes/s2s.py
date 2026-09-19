"""S2SRuntime — speech-to-speech runtime.

A speech-to-speech model (Ultravox, Gemini Live, GPT-4o Realtime, ...) takes
caller audio in and produces bot audio out in one hop, with no separate STT
or TTS stage the runtime can see. That collapse buys latency but costs
visibility: there is no mid-turn transcript to show a live-monitoring UI,
and usage is whatever the vendor chooses to report (often nothing exact
until the call closes). ``S2SRuntime`` does not paper over that; it is the
runtime-layer expression of design principle #5 in ``docs/ARCHITECTURE.md``
("capabilities are declared, never assumed") applied to the one architecture
where the gap is structural rather than a missing feature.

Concretely:

- ``push_audio`` feeds caller frames straight into ``S2SProvider.converse``;
  there is no STT step to intercept.
- Every ``TTSChunk`` the provider yields goes to ``on_bot_audio`` immediately
  (this is the whole point of S2S: first-token-to-audio is the provider's
  problem, not a pipeline this runtime assembles) and any ``Usage`` on it is
  turned into a ``CostRecorded`` event on the session bus.
- ``stop()`` is where transcripts, if any, appear at all: it calls
  ``provider.post_call_transcript()`` and replays it as ``UserTranscript`` /
  ``BotUtterance`` events stamped ``TranscriptAvailability.POST_CALL`` *before*
  ``SessionEnded``, so a subscriber that only cares about the transcript can
  simply wait for those events instead of polling or guessing when they will
  show up.

**Transcript reconciliation.** A post-call transcript arrives as a flat list
with no relationship to the event timeline, so by default every replayed turn
is stamped with the one timestamp that is true of all of them: the moment the
call ended. Some vendors do better and report *when* each turn happened.
:func:`reconcile_transcript` is the seam for that: a provider that knows the
timing puts it on the turn dict, and those turns are then stamped with the
vendor's own offsets and replayed in time order instead of arrival order. It
is strictly opt-in -- a turn with no timing key behaves exactly as before, and
a transcript with no timings at all sorts to the identical order it arrived
in, because the sort is stable and every key is the same fallback value.

Turning raw ``Usage`` into a priced ``CostRecorded.amount`` is deliberately
*not* done here: that requires a vendor rate card, which is ``cost/rates.py``'s
job, not a runtime's. Emitting a made-up dollar amount would be exactly the
kind of silent estimate design principle #4 forbids — so ``amount`` is left
at ``0.0`` and every other field (``units``, ``unit_name``, ``estimated``,
``model``, ``cached_units``) is passed through exactly as the provider
reported it. A ``CostMeter`` (or any other subscriber) that owns pricing can
listen for these events and attach real amounts downstream.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Any

from tring.events import (
    BotUtterance,
    CostComponent,
    CostRecorded,
    SessionEnded,
    SessionStarted,
    TranscriptAvailability,
    UserTranscript,
)
from tring.providers import registry
from tring.providers.base import S2SProvider, Usage
from tring.runtimes.base import AudioFrame, RuntimeAdapter, RuntimeCapabilities
from tring.session import CallSession

#: Turn-dict keys a provider may use to report when a transcript turn happened,
#: in preference order. The value is seconds since session start -- the same
#: clock ``BaseEvent.at`` uses, because the whole point is to line the
#: transcript up against the rest of the event timeline. A provider that only
#: knows wall-clock or socket-relative times must convert before reporting;
#: handing over a number on a different clock would look exactly like a
#: correct answer and be wrong by the length of the connect handshake.
TRANSCRIPT_TIME_KEYS = ("at", "start")


def transcript_offset(turn: Mapping[str, Any], fallback: float) -> float:
    """When to stamp one transcript turn, or ``fallback`` if the provider
    did not say."""
    for key in TRANSCRIPT_TIME_KEYS:
        value = turn.get(key)
        # bool is an int subclass; a stray ``{"at": True}`` is not a timestamp.
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    return fallback


def reconcile_transcript(
    turns: Sequence[Mapping[str, Any]], fallback: float
) -> list[tuple[float, Mapping[str, Any]]]:
    """Pair each turn with the time to stamp it, ordered along the timeline.

    Ordering is the reconciliation half: a provider that reports the caller's
    and the model's halves of a call as two independently accumulated lists
    hands them over concatenated, not interleaved. Sorting by the reported
    time puts the conversation back in the order it actually happened. The
    sort is stable and every untimed turn shares one fallback key, so a
    transcript with no timing information comes back in exactly the order it
    went in -- this function is a no-op for every provider that has nothing
    to add.
    """
    timed = [(transcript_offset(turn, fallback), turn) for turn in turns]
    timed.sort(key=lambda pair: pair[0])
    return timed


_CAPABILITIES = RuntimeCapabilities(
    live_transcripts=False,
    mid_call_tool_calls=True,
    barge_in=True,
    exact_usage_reporting=False,
    voice_cloning=False,
    local_capable=False,
)


class S2SRuntime(RuntimeAdapter):
    """Runs one conversation entirely through a single ``S2SProvider``.

    The provider is resolved from ``AgentSpec.runtime.routing`` (the ``s2s``
    slot) the first time ``start()`` runs, exactly like every other runtime
    resolves its providers from the same routing table — switching from
    cascade to S2S is a config change to ``runtime.mode`` and ``routing``,
    never a code change. Tests (and ``HybridRuntime``, which owns its own
    ``S2SRuntime`` instance) may instead pass a provider instance directly,
    which skips registry resolution entirely.
    """

    def __init__(
        self,
        session: CallSession,
        *,
        provider: S2SProvider | None = None,
        runtime_mode_label: str = "s2s",
        on_stream_error: Callable[[BaseException], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(session)
        self._provider = provider
        # Label stamped on the SessionStarted event. HybridRuntime reuses this
        # class for its live conversation but wants "hybrid" in the event
        # stream, not "s2s" — see runtimes/hybrid.py.
        self._runtime_mode_label = runtime_mode_label
        #: Optional supervisor for a provider stream that fails mid-call.
        #: Left ``None`` (the default) an exception propagates out of the
        #: consume task exactly as before, surfacing when ``stop()`` awaits
        #: it. Set, it is handed the exception instead, which is what lets
        #: ``HybridRuntime`` downgrade a dead S2S leg to a cascade while the
        #: caller is still on the line rather than at hang-up. Public so it
        #: can be attached after construction: the runtime that wants it
        #: builds its ``S2SRuntime`` before it can bind its own methods.
        self.on_stream_error = on_stream_error
        self._frame_queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self._consume_task: asyncio.Task[None] | None = None
        self._stopped = False

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return _CAPABILITIES

    async def start(self) -> None:
        if self._provider is None:
            selection = self.session.agent.runtime.select(self.session.agent.language.primary)
            if not selection.s2s:
                raise ValueError(
                    "S2SRuntime requires routing.<lang>.s2s (or routing.default.s2s) "
                    "to name a registered s2s provider; got no s2s selection for "
                    f"language {self.session.agent.language.primary!r}"
                )
            self._provider = registry.create("s2s", selection.s2s, **selection.options)

        self.session.emit(
            SessionStarted(
                session_id=self.session.session_id,
                at=self.session.elapsed,
                agent_name=self.session.agent.name,
                runtime_mode=self._runtime_mode_label,
            )
        )
        # The greeting is deliberately not spoken here: AgentSpec.greeting is
        # plain text, and S2SProvider (providers/base.py) has no "speak this
        # text" method to hand it to -- only `converse(frames)`. Faking a
        # greeting by synthesizing it through a *different* voice pipeline
        # would contradict the whole point of S2S (one model, one voice,
        # everything in-band). Providers that support an initial system
        # utterance should do so on their own, via `options`.
        self._consume_task = asyncio.create_task(self._consume_bot_audio())

    async def _frames(self) -> AsyncIterator[AudioFrame]:
        while True:
            frame = await self._frame_queue.get()
            if frame is None:
                return
            yield frame

    async def _consume_bot_audio(self) -> None:
        assert self._provider is not None  # start() always sets this first
        try:
            async for chunk in self._provider.converse(self._frames()):
                callback = self.on_bot_audio
                if callback is not None:
                    callback(chunk.frame)  # type: ignore[operator]
                for usage in chunk.usage:
                    self._record_usage(usage)
        except asyncio.CancelledError:
            # Shutdown, not a provider fault: never route it to the supervisor.
            raise
        except Exception as exc:
            if self.on_stream_error is None:
                raise
            await self.on_stream_error(exc)

    def _record_usage(self, usage: Usage) -> None:
        assert self._provider is not None
        self.session.emit(
            CostRecorded(
                session_id=self.session.session_id,
                at=self.session.elapsed,
                component=CostComponent.S2S,
                provider=self._provider.name,
                units=usage.units,
                unit_name=usage.unit_name,
                amount=0.0,  # see module docstring: pricing is cost/rates.py's job
                estimated=usage.estimated,
                model=usage.model,
                cached_units=usage.cached_units,
            )
        )

    async def push_audio(self, frame: AudioFrame) -> None:
        if self._stopped:
            raise RuntimeError("S2SRuntime.push_audio called after stop()")
        await self._frame_queue.put(frame)

    async def halt(self) -> list[dict[str, Any]]:
        """Shut the provider leg down and hand back its transcript, *without*
        ending the session.

        ``stop()`` is this plus the two things that mean "the call is over":
        replaying the transcript onto the event bus and emitting
        ``SessionEnded``. Splitting them apart is what makes a mid-call
        handover possible -- ``HybridRuntime``'s downgrade needs the S2S leg
        dead and its transcript in hand while the session keeps going and the
        caller stays on the line. After this the runtime is ``_stopped``, so a
        later ``stop()`` is the same no-op a second ``stop()`` always was, and
        the leg that took over owns ``SessionEnded``.

        Returns ``[]`` when there is nothing to hand over (already stopped, or
        ``start()`` never ran).
        """
        if self._stopped:
            return []
        self._stopped = True
        # Sentinel closes the frame iterator handed to provider.converse(),
        # which lets the provider finish its stream on its own terms.
        await self._frame_queue.put(None)
        consume, self._consume_task = self._consume_task, None
        if consume is not None and consume is not asyncio.current_task():
            # The identity check matters on exactly one path: an
            # ``on_stream_error`` supervisor runs *inside* the consume task,
            # so a downgrade triggered from there would be this task awaiting
            # itself. There is also nothing to wait for in that case -- the
            # loop we would be draining is the one calling us, and it returns
            # the moment the supervisor does.
            await consume

        if self._provider is None:
            return []
        return list(await self._provider.post_call_transcript())

    def emit_transcript(self, turns: Sequence[Mapping[str, Any]]) -> None:
        """Replay provider transcript turns onto the session event bus.

        Every turn is stamped with the time :func:`reconcile_transcript`
        resolves for it: the provider's own offset when it reported one, and
        otherwise a single snapshot of the session clock shared by all of them
        (one snapshot, not one read per turn, so an untimed transcript keeps a
        stable order instead of being reshuffled by microseconds of drift).
        """
        fallback = self.session.elapsed
        for at, turn in reconcile_transcript(turns, fallback):
            text = str(turn.get("text", ""))
            language = turn.get("language")
            if turn.get("role") == "bot":
                # BotUtterance (events.py) has no availability field --
                # it always means "text the bot generated", live or not.
                # The post-call marker only matters for what the *caller*
                # said, which is why UserTranscript carries it below.
                self.session.emit(
                    BotUtterance(
                        session_id=self.session.session_id,
                        at=at,
                        text=text,
                        language=language,
                    )
                )
            else:
                self.session.emit(
                    UserTranscript(
                        session_id=self.session.session_id,
                        at=at,
                        text=text,
                        final=True,
                        language=language,
                        availability=TranscriptAvailability.POST_CALL,
                    )
                )

    async def stop(self, reason: str = "completed") -> None:
        if self._stopped:
            return
        self.emit_transcript(await self.halt())
        self.session.emit(
            SessionEnded(
                session_id=self.session.session_id,
                at=self.session.elapsed,
                reason=reason,
                duration_seconds=self.session.elapsed,
            )
        )
