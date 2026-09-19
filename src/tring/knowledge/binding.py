"""Wiring a knowledge provider into a live call, including speculative retrieval.

What this binds
---------------

Two things, both handed to the runtime the ordinary way:

* a ``search_knowledge`` :class:`~tring.agent.ToolDef`, appended to the
  agent's tool list so it is rendered into the prompt and validated like any
  other tool;
* the async handler behind it, inserted into the ``handlers`` dict the
  runtime is constructed with.

There is no knowledge-specific code path anywhere in a runtime, which is the
point: the feature is a tool plus an event consumer, so it works on the
cascade runtime, on an S2S runtime with tool support, and on anything a
contributor writes tomorrow.

Speculative retrieval, and why it is an event consumer
------------------------------------------------------

A knowledge answer is two serial waits: the caller finishes speaking, the
model decides to search, *then* the search runs, and only then can the model
start composing the sentence the caller hears. The retrieval is dead air
inside the turn.

Most of that wait is avoidable, because the query is usually knowable before
the turn ends. Partial transcripts arrive mid-sentence; "what time do you
close on sund-" is already enough to retrieve the opening-hours passage. So
this binding subscribes to the session event stream, and every non-final
:class:`~tring.events.UserTranscript` kicks off a background search of the
partial text. When the model calls the tool a moment later, the answer is
already sitting in a cache and the handler returns without waiting for
anything.

This is written as an **event consumer rather than an edit to the cascade
runtime**, and that choice is load-bearing:

* Every runtime emits ``UserTranscript``. A consumer therefore gets
  speculation on the cascade, on any S2S runtime that reports live
  transcripts, and on runtimes that do not exist yet -- where a cascade edit
  would have to be re-implemented once per runtime.
* Zero coupling in the other direction: no runtime imports this module, so a
  knowledge feature can never regress the pipeline's latency path or its
  barge-in behaviour. It cannot even keep the call from ending.
* It is testable without a runtime at all: emit two events on a session and
  assert what the handler returns.

The costs are named honestly rather than hidden. Speculation buys latency
with wasted work: some prefetched searches are never used, and every one of
them is metered and emitted as ``KnowledgeSearched(speculative=True)`` so
the waste is measurable instead of invisible. Cheap retrievers
(``keyword_local``, a local Chroma) make that trade easily; a per-query
billed store may not, which is why ``knowledge.speculative`` is a spec
field.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from typing import Any

from tring.agent import AgentSpec, KnowledgeConfig, ToolDef
from tring.cost.meter import CostMeter
from tring.events import CostComponent, KnowledgeSearched, SessionError, UserTranscript
from tring.knowledge.base import KnowledgeProvider, Passage
from tring.knowledge.keyword_local import tokenize
from tring.providers.base import Usage
from tring.providers.registry import create
from tring.session import CallSession

#: Tool name and handler key. One constant, because the ToolDef, the handlers
#: dict and any spec that wants to describe the tool differently all have to
#: agree on it.
SEARCH_KNOWLEDGE = "search_knowledge"

#: A partial transcript shorter than this is not searched. Two or three
#: characters retrieve whatever passage happens to contain "wh", which wastes
#: a query and poisons the cache for the real one that follows.
DEFAULT_MIN_PREFETCH_CHARS = 12

#: How much of the shorter token set the two queries must share for a
#: prefetched result to answer a tool call. Containment, not Jaccard: the
#: model's query is typically a *cleaned-up subset* of the caller's partial
#: sentence ("what are your opening hours on a sunday" -> "opening hours
#: sunday"), so measuring against the shorter set is what makes a good
#: prefetch count as a hit.
DEFAULT_OVERLAP = 0.6

#: Tool timeout. Generous enough for a cold vector store to answer, short
#: enough that a hung store becomes a spoken apology within a few seconds
#: rather than an abandoned call.
DEFAULT_TIMEOUT_SECONDS = 6.0

#: Cost component for one retrieval. ``OTHER`` rather than ``EMBEDDING``: the
#: metered unit is a whole query (embedding, index scan and payload fetch
#: together), not the tokens an embedding API billed for.
_COST_COMPONENT = CostComponent.OTHER
_COST_UNIT = "knowledge_queries"

ToolHandler = Callable[[dict[str, Any]], Awaitable[Any]]


def _search_tool(timeout_seconds: float) -> ToolDef:
    """The ``search_knowledge`` contract, as the model sees it.

    Choreographed like any other tool: the model must still plan a waiting
    line, because the *first* search of a call usually misses the prefetch
    cache and a store can always be slow. The cost of that is one short line
    the caller sometimes did not need to hear; the cost of the alternative is
    silence on a phone call.
    """
    return ToolDef(
        name=SEARCH_KNOWLEDGE,
        description=(
            "Search the knowledge base and return the passages most likely to "
            "answer the caller. Use it for any question about facts you were "
            "not given in your instructions (hours, prices, policies, "
            "products). Answer only from the passages it returns, and say you "
            "do not know if none of them answers the question."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "What to look up, as a short search phrase in the "
                        "caller's own words. Keep their exact nouns: names, "
                        "plan names and reference numbers are what match."
                    ),
                }
            },
            "required": ["query"],
        },
        handler=SEARCH_KNOWLEDGE,
        timeout_seconds=timeout_seconds,
    )


def _normalize(text: str) -> str:
    """Collapse transcript whitespace so dedup compares content, not spacing."""
    return " ".join(text.split())


def _overlaps(query: str, prefetched: str, threshold: float) -> bool:
    """Is ``prefetched`` close enough to ``query`` to answer it?

    Token containment against the *shorter* set (see ``DEFAULT_OVERLAP``).
    Both sides go through the ``keyword_local`` tokenizer so the comparison
    is Unicode-aware and case-folded, exactly like the retrieval it gates.
    """
    left, right = set(tokenize(query)), set(tokenize(prefetched))
    if not left or not right:
        return False
    return len(left & right) / min(len(left), len(right)) >= threshold


def _attach_tool(agent: AgentSpec, tool: ToolDef) -> None:
    """Add the tool to the spec, idempotently.

    The spec is mutated rather than copied because the spec *is* what a
    runtime reads: ``CascadeRuntime`` builds its tool table and its prompt
    prefix from ``session.agent.tools``, so a tool that is not there is a
    tool the model never learns about. Bind before constructing the runtime.
    """
    if any(existing.name == tool.name for existing in agent.tools):
        return
    agent.tools.append(tool)


def _resolve(config: KnowledgeConfig) -> KnowledgeProvider:
    """Build the configured provider through the registry, kind ``"knowledge"``."""
    provider = create("knowledge", config.provider, **config.options)
    if not isinstance(provider, KnowledgeProvider):
        raise TypeError(
            f"knowledge provider {config.provider!r} returned "
            f"{type(provider).__name__}, which is not a KnowledgeProvider"
        )
    return provider


class KnowledgeBinding:
    """Bind ``spec.knowledge`` to a live session: one tool, one handler, one consumer.

    Usage is three lines, and the order matters::

        handlers: dict[str, ToolHandler] = {}
        binding = KnowledgeBinding(session, spec.knowledge, handlers)
        runtime = CascadeRuntime(session, handlers=handlers)

    The binding must be constructed *before* the runtime, because the runtime
    snapshots the agent's tool list when it is built.

    Args:
        session: the live call. Its event stream feeds speculation and
            receives every ``KnowledgeSearched``.
        config: ``AgentSpec.knowledge``. ``None`` makes the binding inert --
            no tool, no handler, no consumer -- so a caller never has to
            branch on whether the agent has a knowledge base.
        handlers: the dict about to be handed to the runtime. The handler is
            inserted under the tool's name; nothing else is touched.
        meter: optional cost meter. Every executed search, speculative ones
            included, records one ``knowledge_queries`` unit against the
            provider's registry name.
        clock: monotonic source for the ``latency_seconds`` on emitted
            events. Injectable for the same reason ``CallSession`` takes one:
            latency assertions in tests should be exact, not flaky.
        min_prefetch_chars, overlap_threshold, timeout_seconds: the three
            knobs of the speculation policy; see the module constants.
    """

    def __init__(
        self,
        session: CallSession,
        config: KnowledgeConfig | None,
        handlers: dict[str, ToolHandler],
        meter: CostMeter | None = None,
        clock: Callable[[], float] | None = None,
        min_prefetch_chars: int = DEFAULT_MIN_PREFETCH_CHARS,
        overlap_threshold: float = DEFAULT_OVERLAP,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.session = session
        self.config = config
        self.meter = meter
        self.min_prefetch_chars = min_prefetch_chars
        self.overlap_threshold = overlap_threshold
        self._clock = clock or time.perf_counter

        self.provider: KnowledgeProvider | None = None
        self.tool: ToolDef | None = None

        #: The most recently *completed* prefetch: (query, passages).
        self._cache: tuple[str, list[Passage]] | None = None
        #: The query of the in-flight prefetch, and its task. Latest wins.
        self._prefetch_query: str | None = None
        self._prefetch: asyncio.Task[None] | None = None
        self._consumer: asyncio.Task[None] | None = None

        if config is None:
            return

        self.provider = _resolve(config)
        self.tool = _search_tool(timeout_seconds)
        handlers[self.tool.name] = self.handle_search
        _attach_tool(session.agent, self.tool)

        if config.speculative:
            # Start now when there is a loop to start on, so the three-line
            # wiring above needs no fourth line. Constructed outside a loop
            # (a sync factory building the session), speculation waits for
            # the explicit ``await binding.start()``.
            with contextlib.suppress(RuntimeError):
                self._consumer = asyncio.create_task(self._consume())

    # ------------------------------------------------------------- lifecycle

    async def start(self) -> None:
        """Ensure the transcript consumer is running. Idempotent.

        Returns only once the consumer is actually subscribed: creating the
        task is not enough, because ``CallSession.subscribe`` registers its
        queue inside the generator body. One loop pass runs the new task up
        to its first suspension, which is that ``await queue.get()`` -- after
        registration. Without this, events emitted immediately after
        ``start()`` could be missed.
        """
        if self.config is None or not self.config.speculative:
            return
        if self._consumer is None:
            self._consumer = asyncio.create_task(self._consume())
        await asyncio.sleep(0)

    async def aclose(self) -> None:
        """Cancel background work and release the provider. Idempotent."""
        await _cancel(self._prefetch)
        self._prefetch = None
        await _cancel(self._consumer)
        self._consumer = None
        if self.provider is not None:
            await self.provider.aclose()

    # ------------------------------------------------------------- the tool

    async def handle_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Answer a ``search_knowledge`` call, from the prefetch cache when possible.

        Raising on a bad or failing query is deliberate: the runtime's
        choreography turns a tool exception into ``ok=False`` and forces the
        model to speak, so a broken knowledge base becomes an apology the
        caller hears rather than a silent line.
        """
        provider = self.provider
        assert provider is not None  # bound only when a provider exists

        query = _normalize(str(arguments.get("query", "")))
        if not query:
            raise ValueError(
                f"{SEARCH_KNOWLEDGE} requires a non-empty 'query' argument"
            )

        passages = await self._cached_for(query)
        prefetched = passages is not None
        if passages is None:
            passages = await self._search(query, speculative=False)
        return {
            "query": query,
            "prefetched": prefetched,
            "passages": [passage.to_dict() for passage in passages],
        }

    async def _cached_for(self, query: str) -> list[Passage] | None:
        """The prefetched passages for ``query``, or None if this is a miss.

        If a matching prefetch is still running, wait for it instead of
        starting a second identical search -- the whole turn is blocked on
        this answer either way, and the running one is already partway done.
        ``asyncio.wait`` rather than ``await task``: a newer partial
        transcript may cancel that task at any moment, and a cancellation
        meant for the prefetch must not propagate into the caller's turn.
        """
        task = self._prefetch
        pending = self._prefetch_query
        if (
            task is not None
            and not task.done()
            and pending is not None
            and _overlaps(query, pending, self.overlap_threshold)
        ):
            await asyncio.wait({task})

        cached = self._cache
        if cached is None:
            return None
        cached_query, passages = cached
        if not _overlaps(query, cached_query, self.overlap_threshold):
            return None
        return passages

    # ----------------------------------------------------- the event consumer

    async def _consume(self) -> None:
        """Watch the session for partial transcripts and speculate on them."""
        async for event in self.session.subscribe():
            if isinstance(event, UserTranscript) and not event.final:
                self._schedule(event.text)

    def _schedule(self, text: str) -> None:
        """Start a prefetch for a partial transcript, newest wins.

        Partial transcripts arrive several times a second and each one
        supersedes the last, so an in-flight prefetch for the previous
        partial is cancelled rather than left to finish: its answer is for a
        question the caller has already finished asking differently. The
        dedup on identical text matters just as much -- STT providers
        routinely re-emit an unchanged partial while the caller pauses.
        """
        query = _normalize(text)
        if len(query) < self.min_prefetch_chars or query == self._prefetch_query:
            return
        self._prefetch_query = query
        previous = self._prefetch
        if previous is not None and not previous.done():
            previous.cancel()
        self._prefetch = asyncio.create_task(self._speculate(query))

    async def _speculate(self, query: str) -> None:
        """Run one prefetch and cache it. Failures never reach the call.

        A speculative search is best-effort by definition: nobody is waiting
        on it, and the tool handler will happily search again. So a provider
        fault is reported as a recoverable ``SessionError`` and dropped,
        instead of being raised in a background task where it would surface
        as an unretrieved exception at some unrelated moment.
        """
        try:
            passages = await self._search(query, speculative=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._emit(
                SessionError(
                    session_id=self.session.session_id,
                    at=self.session.elapsed,
                    message=f"speculative knowledge search failed: {exc}",
                    recoverable=True,
                )
            )
            return
        self._cache = (query, passages)

    # ----------------------------------------------------------- small bits

    async def _search(self, query: str, speculative: bool) -> list[Passage]:
        """Run one retrieval, then report it as an event and a cost line.

        Exactly one ``KnowledgeSearched`` per retrieval that actually ran. A
        cache hit emits nothing further: the prefetch already reported it,
        and a second event would make "how many searches did this call do"
        double-count the ones speculation was supposed to save.
        """
        provider, config = self.provider, self.config
        assert provider is not None and config is not None

        started = self._clock()
        passages = await provider.search(query, config.top_k)
        latency = self._clock() - started

        self._emit(
            KnowledgeSearched(
                session_id=self.session.session_id,
                at=self.session.elapsed,
                query=query,
                result_count=len(passages),
                latency_seconds=latency,
                speculative=speculative,
            )
        )
        if self.meter is not None:
            # Exact, not estimated: one query ran, and we counted it.
            self.meter.record(
                _COST_COMPONENT,
                provider.name,
                Usage(units=1.0, unit_name=_COST_UNIT, estimated=False),
            )
        return passages

    def _emit(self, event: KnowledgeSearched | SessionError) -> None:
        """Emit, tolerating a session that has already closed.

        A prefetch can land after the caller hangs up. That is a race nobody
        can prevent from here, and it is not worth an exception escaping a
        background task -- the result is simply no longer wanted.
        """
        with contextlib.suppress(RuntimeError):
            self.session.emit(event)


async def _cancel(task: asyncio.Task[None] | None) -> None:
    """Cancel a task and wait for it to actually stop."""
    if task is None or task.done():
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


__all__ = [
    "DEFAULT_MIN_PREFETCH_CHARS",
    "DEFAULT_OVERLAP",
    "DEFAULT_TIMEOUT_SECONDS",
    "SEARCH_KNOWLEDGE",
    "KnowledgeBinding",
    "ToolHandler",
]
