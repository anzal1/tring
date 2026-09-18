"""HybridRuntime — S2S conversation, runtime-owned tool choreography.

Read this docstring before reaching for ``HybridRuntime``: it is the
runtime this swarm is being the most honest about, not the most capable
about.

``docs/ARCHITECTURE.md`` describes hybrid mode as "S2S front + cascade
fallback/tooling". As of v0.1 this module delivers exactly the "tooling"
half and none of the "fallback" half:

- **What it does.** It composes an ``S2SRuntime`` for the entire live
  conversation (audio path, transcript replay, usage events -- all
  delegated wholesale, see ``runtimes/s2s.py``) and adds one thing on top:
  a single, runtime-owned choke point for tool calls, so every tool call
  gets the same ``primitives.choreography`` enforcement (schema-validated
  ``waiting_message`` / ``spoken_mode`` / ``post_tool_response``,
  ``ToolCallStarted`` / ``ToolCallCompleted`` events) regardless of which
  S2S vendor is underneath. That matters because a pure ``S2SRuntime``
  leaves tool-calling entirely inside the provider's own opaque
  implementation -- fine when you trust one vendor's tool semantics,
  useless when you want one consistent choreographed experience across
  several.
- **What it does not do.** There is no cascade fallback. If the S2S leg
  drops or the vendor model degrades, this runtime does not transparently
  reconnect through a cascade pipeline mid-call. Advertising that here,
  before it is built, would be exactly the "fake feature" this codebase's
  contributor rules forbid. When that lands it will show up as a real,
  tested code path -- not a docstring promise.

**The tool-call hook.** ``providers/base.py``'s ``S2SProvider`` ABC does not
(yet) define a generic channel for a provider to *surface* a tool call to
its caller -- ``converse()`` only yields ``TTSChunk``. That is a real gap in
the v0.1 provider contract, not an oversight hidden here. Instead of
inventing an unused abstract method no concrete provider implements yet,
this module exposes one clear extension point: whatever provider-specific
mechanism a vendor integration uses to notice a tool call (a callback, a
side-channel message, polling a vendor SDK object -- all vendor-specific)
is expected to end by calling
``await hybrid_runtime.on_provider_tool_call(name, args)``. That method is
the actual contract; "the provider surfaces a tool call" is deliberately
left to whoever wires up a concrete S2S provider next.

**Choreography and "speak".** ``on_provider_tool_call`` parses ``args``
against the tool's augmented schema and runs it through
``primitives.choreography.execute`` -- the same function a cascade runtime
would use, so a tool handler behaves identically under either mode. That
function's contract includes a ``speak(text) -> Awaitable[None]`` callback
for playing ``waiting_message`` live while the tool runs. In cascade mode
that drives the TTS pipeline; in S2S/hybrid mode there is no separate TTS
stage to drive -- the S2S model already produced whatever speech happens
around the tool call as part of its own audio output. Passing anything
other than a no-op here would risk synthesizing the waiting message a
*second* time through a channel the caller never hears blended with the
model's own audio. So the callback passed to ``execute`` is a documented
no-op: choreography's bookkeeping (events, timeout, forced response on
failure) still applies in full; only the "narrate it out loud ourselves"
half is inapplicable in this mode.
"""

from __future__ import annotations

import uuid
from typing import Any

from tring.events import ToolCallCompleted, ToolCallStarted
from tring.primitives.choreography import (
    ChoreographyError,
    ToolHandler,
    execute,
    parse_choreographed_call,
)
from tring.runtimes.base import AudioFrame, RuntimeAdapter, RuntimeCapabilities
from tring.runtimes.s2s import S2SRuntime
from tring.session import CallSession

_CAPABILITIES = RuntimeCapabilities(
    live_transcripts=False,
    mid_call_tool_calls=True,
    barge_in=True,
    exact_usage_reporting=False,
    voice_cloning=False,
    local_capable=False,
)


async def _speak_is_already_in_the_audio(_: str) -> None:
    """The ``speak`` callback handed to ``choreography.execute``.

    A no-op, on purpose: see the "Choreography and speak" section of this
    module's docstring. The S2S model's own audio output already carries
    whatever it said around the tool call; this hook exists only so
    ``execute``'s signature (shared with cascade) does not need a
    hybrid-specific branch.
    """
    return None


class HybridRuntime(RuntimeAdapter):
    """S2S conversation with runtime-owned tool choreography.

    ``handlers`` maps tool name -> async handler, exactly like a cascade
    runtime's tool dispatch would: the mapping is what session start-up
    binds ``AgentSpec.tools[i].handler`` names to, keeping ``AgentSpec``
    itself free of callables (see ``agent.py``'s module docstring). The
    tool must also appear in ``session.agent.tools`` by name -- that is
    where its JSON-Schema and ``choreographed`` flag live, and
    choreography cannot validate a call against a schema it does not have.
    """

    def __init__(
        self,
        session: CallSession,
        *,
        handlers: dict[str, ToolHandler] | None = None,
    ) -> None:
        # Built before super().__init__() because the on_bot_audio property
        # setter below forwards to it, and RuntimeAdapter.__init__ assigns
        # self.on_bot_audio = None as part of construction.
        self._s2s = S2SRuntime(session, runtime_mode_label="hybrid")
        super().__init__(session)
        self._handlers: dict[str, ToolHandler] = dict(handlers or {})

    @property
    def on_bot_audio(self) -> object | None:
        return self._s2s.on_bot_audio

    @on_bot_audio.setter
    def on_bot_audio(self, callback: object | None) -> None:
        self._s2s.on_bot_audio = callback

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return _CAPABILITIES

    async def start(self) -> None:
        await self._s2s.start()

    async def push_audio(self, frame: AudioFrame) -> None:
        await self._s2s.push_audio(frame)

    async def stop(self, reason: str = "completed") -> None:
        await self._s2s.stop(reason)

    async def on_provider_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Route one provider-surfaced tool call through choreography.

        ``args`` is the raw tool-call payload the provider produced: the
        tool's own arguments plus the three choreography keys
        (``waiting_message``, ``spoken_mode``, ``post_tool_response``) the
        model was required to include by the tool's augmented schema (see
        ``primitives.choreography.augment_tool_schema``), plus an optional
        ``call_id`` for correlating events if the provider assigns one.

        Two failure modes never reach ``choreography.execute`` at all, and
        are reported the same shape either way (``{"ok": False, "call_id":
        ..., "error": ...}``) so a caller does not need to distinguish
        them: the tool name is not declared on this agent at all, or no
        handler was bound for it. Both are configuration errors, not
        runtime ones, so they are reported without ever starting a timer.
        A schema violation (missing/invalid choreography field) is
        ``ChoreographyError`` raised by ``parse_choreographed_call`` and is
        reported the same way -- the model gets the exact error text back
        so a real vendor integration can resubmit it as the tool result
        and let the model retry.

        Never raises: a broken tool handler, or a malformed call, ends the
        *tool call*, not the conversation.
        """
        call_id = str(args.get("call_id") or uuid.uuid4().hex)
        tool_def = next((t for t in self.session.agent.tools if t.name == name), None)
        handler = self._handlers.get(name)

        if tool_def is None or handler is None:
            error = (
                f"tool {name!r} is not declared on agent {self.session.agent.name!r}"
                if tool_def is None
                else f"no handler bound for tool {name!r}"
            )
            return self._report_pre_execute_failure(name, call_id, args, error)

        try:
            call = parse_choreographed_call(tool_def, args)
        except ChoreographyError as exc:
            return self._report_pre_execute_failure(name, call_id, args, str(exc))

        outcome = await execute(
            tool_def, call, handler, self.session, speak=_speak_is_already_in_the_audio
        )
        return {
            "ok": outcome.ok,
            "call_id": call.call_id,
            "result": outcome.result,
            "error": outcome.error,
            "should_respond": outcome.should_respond,
        }

    def _report_pre_execute_failure(
        self, name: str, call_id: str, args: dict[str, Any], error: str
    ) -> dict[str, Any]:
        """Emit start/complete events for a call that never reaches ``execute``.

        Configuration errors (unknown tool, unbound handler, schema
        violation) still deserve a ``ToolCallStarted``/``ToolCallCompleted``
        pair -- analytics and cost tooling should be able to see that a
        call was attempted even when it never got as far as running a
        handler.
        """
        started_at = self.session.elapsed
        arguments = {k: v for k, v in args.items() if k != "call_id"}
        self.session.emit(
            ToolCallStarted(
                session_id=self.session.session_id,
                at=started_at,
                tool_name=name,
                call_id=call_id,
                arguments=arguments,
            )
        )
        self.session.emit(
            ToolCallCompleted(
                session_id=self.session.session_id,
                at=self.session.elapsed,
                tool_name=name,
                call_id=call_id,
                ok=False,
                result_summary=error,
                latency_seconds=self.session.elapsed - started_at,
            )
        )
        return {"ok": False, "call_id": call_id, "error": error}
