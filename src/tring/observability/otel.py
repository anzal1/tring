"""OpenTelemetry export: turn a CallSession's event stream into a span tree.

Tring's event stream (``events.py``) already carries everything a trace
needs: who started talking, when the reply existed, when a tool ran and how
long it took, what each turn cost. This module is a thin, one-directional
translation from that vocabulary into standard OTLP spans, so a call shows up
in whatever tracing backend a deployment already runs (Jaeger, Tempo, Honeycomb,
a vendor's hosted OTLP endpoint) without Tring depending on any of them.

Span shape, one call::

    tring.session                    (root: SessionStarted .. SessionEnded)
      tring.turn                     (final UserTranscript .. next BotUtterance)
        tring.tool.<name>            (ToolCallStarted .. ToolCallCompleted)
      tring.turn
        ...

``CostRecorded`` lines (stt/llm/tts/... usage) are not spans of their own: a
span per token would be pure noise, and the amount/units/estimated/
cached_units numbers matter as annotations on the turn they happened during,
not as timed operations in their own right. They land as span *events*
(``add_event``) on whichever span is open when they arrive, and so do
``DtmfReceived``, ``SessionTransferred``, ``KnowledgeSearched`` and
``SessionError``: real things that happened during a turn (or the call as a
whole), not new timed operations.

No vendor coupling: only the standard ``opentelemetry.trace`` API is used
(``get_tracer``, ``start_span``, ``set_span_in_context``, and the ``Span``
methods). Whichever ``TracerProvider`` and exporter the host process has
configured is what actually receives these spans; this module never imports
a vendor SDK, and never sets one up itself, that configuration is the host
application's job, same as any other OTel-instrumented library.
"""

from __future__ import annotations

from typing import Any

from tring.events import (
    BotUtterance,
    CostRecorded,
    DtmfReceived,
    KnowledgeSearched,
    SessionEnded,
    SessionError,
    SessionEvent,
    SessionStarted,
    SessionTransferred,
    ToolCallCompleted,
    ToolCallStarted,
    UserTranscript,
)
from tring.session import CallSession


def _load_trace() -> Any:
    """Import ``opentelemetry.trace``, lazily, with a helpful failure.

    The core install stays free of ``opentelemetry-api``: nothing in Tring
    needs it unless this exporter is actually used, and plenty of deployments
    never touch tracing at all.
    """
    try:
        from opentelemetry import trace
    except ImportError as exc:
        raise ImportError(
            "OtelExporter needs the OpenTelemetry API. Install the optional "
            "extra: pip install 'tring[otel]'"
        ) from exc
    return trace


class OtelExporter:
    """Consumes one ``CallSession``'s event stream and emits an OTLP span tree.

    Timestamps: OpenTelemetry's ``start_time``/``end_time`` are wall-clock
    epoch nanoseconds, while ``SessionEvent.at`` is seconds since the
    session's own clock started (often a fake clock in tests), with no epoch
    reference at all. Rather than fabricate one, every span here is started
    and ended with the timestamp omitted, which is the SDK's own documented
    behaviour for "use the current wall-clock time". That is exactly correct
    for the intended use, consuming the live event stream as a call actually
    happens, and it is the only honest choice when replaying a stored session
    whose ``at`` values carry no wall-clock meaning at all.

    Args:
        service_name: passed to ``get_tracer``; most backends show this as
            the instrumentation scope name on every span it produces.
    """

    def __init__(self, service_name: str = "tring") -> None:
        trace = _load_trace()
        self._trace = trace
        self._tracer = trace.get_tracer(service_name)

        self._session_span: Any | None = None
        self._turn_span: Any | None = None
        self._tool_spans: dict[str, Any] = {}

    async def run(self, session: CallSession) -> None:
        """Subscribe to ``session`` and export events until it closes.

        Typical wiring: ``asyncio.create_task(exporter.run(session))``
        alongside the runtime. Returns on its own once the session closes
        (``subscribe()``'s sentinel), closing out any span left open rather
        than leaking one past the call's end.
        """
        async for event in session.subscribe():
            self.handle_event(event)
        self._close_all()

    def handle_event(self, event: SessionEvent) -> None:
        """Apply one event to the span tree.

        Synchronous and side-effect-only: this is the seam tests drive
        directly with a plain list of events, no event loop or live session
        required, to assert the resulting span tree.
        """
        if isinstance(event, SessionStarted):
            self._start_session(event)
        elif isinstance(event, UserTranscript):
            if event.final:
                self._start_turn(event)
        elif isinstance(event, BotUtterance):
            self._end_turn(outcome="completed")
        elif isinstance(event, ToolCallStarted):
            self._start_tool(event)
        elif isinstance(event, ToolCallCompleted):
            self._end_tool(event)
        elif isinstance(event, CostRecorded):
            self._add_cost_event(event)
        elif isinstance(event, DtmfReceived):
            self._add_event("dtmf_received", {"digit": event.digit})
        elif isinstance(event, SessionTransferred):
            self._add_event("session_transferred", {"target": event.target})
        elif isinstance(event, KnowledgeSearched):
            self._add_event(
                "knowledge_searched",
                {
                    "query": event.query,
                    "result_count": event.result_count,
                    "latency_seconds": event.latency_seconds,
                    "speculative": event.speculative,
                },
            )
        elif isinstance(event, SessionError):
            self._add_event(
                "session_error",
                {"message": event.message, "recoverable": event.recoverable},
            )
        elif isinstance(event, SessionEnded):
            self._end_session(event)

    # ------------------------------------------------------------- session

    def _start_session(self, event: SessionStarted) -> None:
        self._session_span = self._tracer.start_span(
            "tring.session",
            attributes={
                "session.id": event.session_id,
                "agent.name": event.agent_name,
                "runtime.mode": event.runtime_mode,
            },
        )

    def _end_session(self, event: SessionEnded) -> None:
        # A turn still open at call end never reached a bot reply: close it
        # out as such rather than leaving it dangling past the root span.
        self._end_turn(outcome="session_ended")
        if self._session_span is not None:
            self._session_span.set_attribute("session.end_reason", event.reason)
            self._session_span.set_attribute(
                "session.duration_seconds", event.duration_seconds
            )
            self._session_span.end()
            self._session_span = None

    def _close_all(self) -> None:
        """Best-effort cleanup for a stream that ends without ``SessionEnded``
        (a crash, a dropped connection). An exporter task must never leave a
        span open forever just because the call ended abnormally."""
        for span in list(self._tool_spans.values()):
            span.end()
        self._tool_spans.clear()
        if self._turn_span is not None:
            self._turn_span.end()
            self._turn_span = None
        if self._session_span is not None:
            self._session_span.end()
            self._session_span = None

    # --------------------------------------------------------------- turns

    def _start_turn(self, event: UserTranscript) -> None:
        # A new final transcript while a turn is still open means the caller
        # started the next turn before this one produced a reply: a genuine
        # cascade barge-in cancels the in-flight turn task (see
        # ``CascadeRuntime._cancel_turn``), so the span for it closes here as
        # interrupted rather than staying open indefinitely.
        self._end_turn(outcome="interrupted")
        self._turn_span = self._tracer.start_span(
            "tring.turn",
            context=self._context_of(self._session_span),
            attributes={"turn.language": event.language or ""},
        )

    def _end_turn(self, outcome: str) -> None:
        if self._turn_span is None:
            return
        self._turn_span.set_attribute("turn.outcome", outcome)
        self._turn_span.end()
        self._turn_span = None

    # ---------------------------------------------------------------- tools

    def _start_tool(self, event: ToolCallStarted) -> None:
        parent = self._turn_span if self._turn_span is not None else self._session_span
        self._tool_spans[event.call_id] = self._tracer.start_span(
            f"tring.tool.{event.tool_name}",
            context=self._context_of(parent),
            attributes={"tool.name": event.tool_name, "tool.call_id": event.call_id},
        )

    def _end_tool(self, event: ToolCallCompleted) -> None:
        span = self._tool_spans.pop(event.call_id, None)
        if span is None:
            # No matching start, e.g. the exporter attached mid-call. Nothing
            # to close, and inventing a span retroactively would misreport
            # its duration as however long *this* exporter has been running.
            return
        span.set_attribute("tool.ok", event.ok)
        if event.latency_seconds is not None:
            span.set_attribute("tool.latency_seconds", event.latency_seconds)
        if event.result_summary is not None:
            span.set_attribute("tool.result_summary", event.result_summary)
        span.end()

    # ----------------------------------------------------------------- cost

    def _add_cost_event(self, event: CostRecorded) -> None:
        attributes: dict[str, Any] = {
            "cost.component": event.component.value,
            "cost.provider": event.provider,
            "cost.amount": event.amount,
            "cost.units": event.units,
            "cost.unit_name": event.unit_name,
            "cost.estimated": event.estimated,
        }
        if event.model is not None:
            attributes["cost.model"] = event.model
        if event.cached_units is not None:
            attributes["cost.cached_units"] = event.cached_units
        self._add_event("cost_recorded", attributes)

    # --------------------------------------------------------------- helpers

    def _add_event(self, name: str, attributes: dict[str, Any]) -> None:
        """Attach a span event to whatever is currently open: the in-progress
        turn if there is one, otherwise the session span covering the whole
        call."""
        target = self._turn_span if self._turn_span is not None else self._session_span
        if target is not None:
            target.add_event(name, attributes=attributes)

    def _context_of(self, span: Any | None) -> Any | None:
        """``trace.set_span_in_context``, guarded for "no such span yet".

        Passing ``None`` through to ``start_span(context=None)`` is the
        documented way to say "no explicit parent"; it is also exactly what
        this exporter needs defensively if events ever arrive out of the
        expected order (a tool call before any ``SessionStarted``, say).
        """
        if span is None:
            return None
        return self._trace.set_span_in_context(span)


__all__ = ["OtelExporter"]
