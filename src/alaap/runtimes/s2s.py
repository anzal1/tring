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
from collections.abc import AsyncIterator

from alaap.events import (
    BotUtterance,
    CostComponent,
    CostRecorded,
    SessionEnded,
    SessionStarted,
    TranscriptAvailability,
    UserTranscript,
)
from alaap.providers import registry
from alaap.providers.base import S2SProvider, Usage
from alaap.runtimes.base import AudioFrame, RuntimeAdapter, RuntimeCapabilities
from alaap.session import CallSession

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
    ) -> None:
        super().__init__(session)
        self._provider = provider
        # Label stamped on the SessionStarted event. HybridRuntime reuses this
        # class for its live conversation but wants "hybrid" in the event
        # stream, not "s2s" — see runtimes/hybrid.py.
        self._runtime_mode_label = runtime_mode_label
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
        async for chunk in self._provider.converse(self._frames()):
            callback = self.on_bot_audio
            if callback is not None:
                callback(chunk.frame)  # type: ignore[operator]
            for usage in chunk.usage:
                self._record_usage(usage)

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

    async def stop(self, reason: str = "completed") -> None:
        if self._stopped:
            return
        self._stopped = True
        # Sentinel closes the frame iterator handed to provider.converse(),
        # which lets the provider finish its stream on its own terms.
        await self._frame_queue.put(None)
        if self._consume_task is not None:
            await self._consume_task

        if self._provider is not None:
            for turn in await self._provider.post_call_transcript():
                text = turn.get("text", "")
                language = turn.get("language")
                if turn.get("role") == "bot":
                    # BotUtterance (events.py) has no availability field --
                    # it always means "text the bot generated", live or not.
                    # The post-call marker only matters for what the *caller*
                    # said, which is why UserTranscript carries it below.
                    self.session.emit(
                        BotUtterance(
                            session_id=self.session.session_id,
                            at=self.session.elapsed,
                            text=text,
                            language=language,
                        )
                    )
                else:
                    self.session.emit(
                        UserTranscript(
                            session_id=self.session.session_id,
                            at=self.session.elapsed,
                            text=text,
                            final=True,
                            language=language,
                            availability=TranscriptAvailability.POST_CALL,
                        )
                    )

        self.session.emit(
            SessionEnded(
                session_id=self.session.session_id,
                at=self.session.elapsed,
                reason=reason,
                duration_seconds=self.session.elapsed,
            )
        )
