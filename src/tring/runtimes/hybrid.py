"""HybridRuntime — S2S conversation, runtime-owned tool choreography.

Read this docstring before reaching for ``HybridRuntime``: it is the
runtime this swarm is being the most honest about, not the most capable
about.

``docs/ARCHITECTURE.md`` describes hybrid mode as "S2S front + cascade
fallback/tooling". v0.1 delivered the "tooling" half and, deliberately,
none of the "fallback" half; v0.4 adds the fallback half as an opt-in
policy. Both are described below, in that order, because the second one
only makes sense once the first is clear:

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
- **What it does now do (v0.4): mid-call downgrade.** The "fallback" half
  that v0.1 deliberately refused to advertise exists, opt-in, behind
  :class:`DowngradePolicy`. Pass one and the runtime watches for two
  conditions -- the S2S provider's stream failing, and a stretch of turns
  heavy enough with tool calls that a cascade would serve the caller better
  -- and on either one it stops the S2S leg, stands up a
  :class:`~tring.runtimes.cascade.CascadeRuntime` on the *same session* from
  the *same* ``AgentSpec``, replays the transcript accumulated so far into
  that runtime's message history, and routes the rest of the call through it.
  Pass nothing (the default) and none of that machinery is armed: the
  behaviour is v0.1's exactly.

  Read :meth:`_downgrade` before relying on this. The handover is honest but
  it is not free, and the seams are documented there rather than smoothed
  over: the caller hears a gap while the cascade binds its providers, the
  event stream gains a second ``SessionStarted`` (preceded by the
  ``SessionError`` that says why), the greeting is suppressed because
  re-greeting a caller mid-call is worse than silence, and what carries over
  is the *transcript*, not the S2S model's internal state -- prosody,
  hesitations and whatever the model inferred from the caller's tone are gone,
  because text is the only thing a provider hands over.

  ``capabilities`` tracks the change rather than describing an average: before
  a downgrade it reports the S2S truth (no live transcripts, inexact usage),
  after one it reports whatever the cascade reports. Design principle #5 says
  capabilities are declared, never assumed; a runtime that can change shape
  mid-call has to declare the shape it is actually in.

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

import json
import uuid
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, Field

from tring.events import SessionError, ToolCallCompleted, ToolCallStarted
from tring.primitives.choreography import (
    ChoreographyError,
    ToolHandler,
    execute,
    parse_choreographed_call,
)
from tring.runtimes.base import AudioFrame, RuntimeAdapter, RuntimeCapabilities
from tring.runtimes.cascade import CascadeRuntime
from tring.runtimes.s2s import S2SRuntime, reconcile_transcript
from tring.session import CallSession

_CAPABILITIES = RuntimeCapabilities(
    live_transcripts=False,
    mid_call_tool_calls=True,
    barge_in=True,
    exact_usage_reporting=False,
    voice_cloning=False,
    local_capable=False,
)


class DowngradePolicy(BaseModel):
    """When a hybrid session should stop being one.

    Both triggers are about the same thing from opposite directions: an S2S
    model is the right choice while the conversation is *conversation*, and
    the wrong one once it is mostly machinery.

    ``tool_calls`` / ``within_turns``
        The tool-heaviness threshold: downgrade once ``tool_calls`` tool calls
        have been attempted within the last ``within_turns`` model turns. A
        turn-heavy stretch of tool calls means the caller is spending the call
        listening to waiting messages, and that is a workload where a cascade
        wins -- its transcripts are live, its tool loop is the runtime's own,
        and a per-turn LLM is cheaper than per-second audio tokens. Attempted
        calls count, failures included: a model retrying a broken tool three
        times is exactly the case worth escaping.

        Turns are counted from :meth:`HybridRuntime.on_provider_turn_complete`.
        If an integration never reports turn boundaries, the turn index stays
        at zero, every tool call lands in one window, and the threshold
        degenerates to "downgrade after ``tool_calls`` tool calls" -- a
        defensible reading of a provider that cannot tell us where its turns
        end, not a silent failure.

    ``on_error``
        Downgrade when the S2S provider's stream raises. This is the failure
        the v0.1 docstring called out as unhandled: without it the exception
        surfaces at ``stop()``, i.e. after the caller has already hung up on a
        dead line.
    """

    tool_calls: int = Field(default=3, ge=1)
    within_turns: int = Field(default=2, ge=1)
    on_error: bool = True


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
        downgrade: DowngradePolicy | None = None,
    ) -> None:
        # Both legs are declared before super().__init__() because the
        # on_bot_audio property setter below forwards to them, and
        # RuntimeAdapter.__init__ assigns self.on_bot_audio = None as part of
        # construction -- which runs that setter before any other assignment
        # in this body would have happened.
        self._s2s = S2SRuntime(session, runtime_mode_label="hybrid")
        #: The cascade that took over, or ``None`` while this is still a
        #: hybrid session. Doubles as the "have we downgraded" flag: there is
        #: no separate boolean to keep in sync with it.
        self._cascade: CascadeRuntime | None = None
        super().__init__(session)
        self._handlers: dict[str, ToolHandler] = dict(handlers or {})
        self._policy = downgrade
        self._turn_index = 0
        #: The turn index of each recent tool call, pruned to the policy's
        #: window. A deque rather than a counter because the threshold is
        #: "within the last m turns", which needs the calls themselves, not
        #: a total that can never come back down.
        self._tool_call_turns: deque[int] = deque()

        if downgrade is not None and downgrade.on_error:
            self._s2s.on_stream_error = self._downgrade_after_stream_error

    @property
    def downgraded(self) -> bool:
        """True once the cascade has taken the call over."""
        return self._cascade is not None

    @property
    def _live(self) -> RuntimeAdapter:
        """Whichever leg currently owns the call."""
        return self._cascade if self._cascade is not None else self._s2s

    @property
    def on_bot_audio(self) -> object | None:
        return self._live.on_bot_audio

    @on_bot_audio.setter
    def on_bot_audio(self, callback: object | None) -> None:
        # Written to both legs, not just the live one, so a transport that
        # attaches its sink once at start-up keeps receiving audio across a
        # downgrade without knowing one happened.
        self._s2s.on_bot_audio = callback
        if self._cascade is not None:
            self._cascade.on_bot_audio = callback

    @property
    def capabilities(self) -> RuntimeCapabilities:
        cascade = self._cascade
        return cascade.capabilities if cascade is not None else _CAPABILITIES

    async def start(self) -> None:
        await self._s2s.start()

    async def push_audio(self, frame: AudioFrame) -> None:
        await self._live.push_audio(frame)

    async def stop(self, reason: str = "completed") -> None:
        # Whichever leg is live emits SessionEnded. The other one, if a
        # downgrade happened, was halted rather than stopped and therefore
        # never owed the session an ending (see S2SRuntime.halt).
        await self._live.stop(reason)

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

        Every call through here also counts against
        :class:`DowngradePolicy`'s tool-heaviness threshold, which is why the
        counting happens in this wrapper rather than beside one of the
        ``return`` statements below: a tool call that failed configuration
        validation is still a tool call the caller waited through.
        """
        self._note_tool_call()
        outcome = await self._dispatch_tool_call(name, args)
        await self._maybe_downgrade_for_tool_load()
        return outcome

    async def _dispatch_tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """The choreography half of :meth:`on_provider_tool_call`."""
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

    # ------------------------------------------------------- mid-call downgrade

    async def on_provider_turn_complete(self) -> None:
        """One model turn finished on the S2S leg.

        The second vendor-integration seam, and the same kind of thing as
        ``on_provider_tool_call``: ``S2SProvider.converse()`` can only yield
        audio, so a turn boundary (Gemini Live's
        ``serverContent.turnComplete``, OpenAI Realtime's ``response.done``)
        has to be reported by whoever is reading the vendor's socket. Calling
        it is optional -- see :class:`DowngradePolicy` for what the threshold
        degenerates to when nobody does.
        """
        self._turn_index += 1

    def _note_tool_call(self) -> None:
        """Record one attempted tool call against the policy's window."""
        policy = self._policy
        if policy is None or self._cascade is not None:
            return
        self._tool_call_turns.append(self._turn_index)
        oldest_in_window = self._turn_index - (policy.within_turns - 1)
        while self._tool_call_turns and self._tool_call_turns[0] < oldest_in_window:
            self._tool_call_turns.popleft()

    async def _maybe_downgrade_for_tool_load(self) -> None:
        policy = self._policy
        if policy is None or len(self._tool_call_turns) < policy.tool_calls:
            return
        await self._downgrade(
            f"{len(self._tool_call_turns)} tool calls within "
            f"{policy.within_turns} turn(s) exceeds the configured "
            f"threshold of {policy.tool_calls}"
        )

    async def _downgrade_after_stream_error(self, exc: BaseException) -> None:
        """``S2SRuntime.on_stream_error``: the provider leg died mid-call."""
        await self._downgrade(f"speech-to-speech provider failed: {exc!r}")

    async def _downgrade(self, reason: str) -> None:
        """Hand the live call from the S2S leg to a fresh cascade. Idempotent.

        The order is the whole story, and each step has a seam worth naming:

        1. **Announce it.** A ``SessionError`` with ``recoverable=True`` goes
           out *first*, so a subscriber reading the stream sees the reason
           before it sees the consequences. Recoverable is the literal truth:
           the call is continuing, just not the way it started.
        2. **Halt, do not stop, the S2S leg.** ``S2SRuntime.halt`` kills the
           provider stream and hands back its transcript without emitting
           ``SessionEnded`` -- the session is not over, and exactly one leg
           may end it. The transcript is replayed onto the bus through the
           same ``emit_transcript`` path ``stop()`` would have used, so the
           caller's words are not lost just because the call took a detour.
        3. **Build the cascade from the same spec.** Same ``CallSession``,
           same ``AgentSpec``, same handlers -- re-keyed onto cascade's
           ``ToolDef.handler`` lookup, since hybrid binds handlers by tool
           name and cascade by handler name.
        4. **Suppress the greeting, only for start-up.** ``CascadeRuntime``
           speaks ``AgentSpec.greeting`` in ``start()``; greeting a caller who
           is already mid-conversation is worse than the silence it would
           replace. The spec is swapped for a greeting-less copy for the
           duration of that one call and restored in a ``finally``. This is
           the ugliest seam here and it is deliberate: the clean fix is a
           ``CascadeRuntime(..., greet=False)`` argument, and inventing one
           would mean editing cascade.py to serve hybrid's convenience.
        5. **Replay the transcript as message history.** Also reaching past a
           public API (``cascade._messages``), for the same reason and with
           the same intended fix: a ``history=`` constructor argument. It has
           to happen *after* ``start()``, which builds the prompt prefix by
           assignment and would otherwise overwrite whatever was appended.

        What does not carry over: everything the S2S model knew that was not
        words. That is a property of transcripts, not a gap in this code.
        """
        if self._cascade is not None:
            return

        self.session.emit(
            SessionError(
                session_id=self.session.session_id,
                at=self.session.elapsed,
                message=f"downgrading hybrid session to cascade: {reason}",
                recoverable=True,
            )
        )

        turns = await self._s2s.halt()
        self._s2s.emit_transcript(turns)

        cascade = CascadeRuntime(self.session, handlers=self._cascade_handlers())
        agent = self.session.agent
        self.session.agent = agent.model_copy(update={"greeting": None})
        try:
            await cascade.start()
        finally:
            self.session.agent = agent

        cascade._messages.extend(_replayed_messages(turns))
        cascade.on_bot_audio = self._s2s.on_bot_audio
        self._cascade = cascade

    def _cascade_handlers(self) -> dict[str, ToolHandler]:
        """Re-key this runtime's handler map onto cascade's lookup.

        ``HybridRuntime`` binds handlers by tool *name* (the name the provider
        surfaces); ``CascadeRuntime`` looks them up by ``ToolDef.handler``,
        falling back to the name. For every spec where the two are the same
        string this is a copy; where a spec names a distinct handler, this is
        what stops the downgraded call from losing its tools.
        """
        mapping = dict(self._handlers)
        for tool in self.session.agent.tools:
            handler = self._handlers.get(tool.name)
            if handler is not None and tool.handler:
                mapping[tool.handler] = handler
        return mapping


def _replayed_messages(turns: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Turn an S2S transcript into cascade message history.

    Bot turns are re-encoded into the ``{"speak": ..., "tool_call": null}``
    envelope rather than stored as bare text. The cascade's history is also a
    worked example of the output contract the model is being held to (see
    ``cascade.ENVELOPE_INSTRUCTION``); dropping a dozen bare-text assistant
    messages into it would teach the model, turn by turn, that the format is
    optional -- and the first turn after a downgrade is the worst possible
    moment to have the envelope parser fall back.

    Ordering goes through ``reconcile_transcript`` for the same reason the
    event replay does: a provider that reports the two halves of a call
    separately would otherwise hand the model a conversation in which nobody
    answers anybody.
    """
    messages: list[dict[str, Any]] = []
    for _at, turn in reconcile_transcript(turns, 0.0):
        text = str(turn.get("text") or "")
        if not text:
            continue
        if turn.get("role") == "bot":
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"speak": text, "tool_call": None}, ensure_ascii=False
                    ),
                }
            )
        else:
            messages.append({"role": "user", "content": text})
    return messages
