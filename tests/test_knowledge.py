"""Tests for the knowledge slot: BM25, the tool binding, and speculation.

Everything runs offline. The vector-store adapters are exercised against
``sys.modules`` stubs (construction, request shape, result mapping), because
what is worth testing about an adapter is the call it makes and the passages
it builds from the answer -- not whether Chroma and Qdrant work, which is
their maintainers' job.

The binding is tested through the *real* ``CascadeRuntime`` with fake
providers, the house pattern from ``tests/test_cascade.py``: a tool that only
works when the runtime is mocked is a tool that does not work.
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tring.agent import (
    AgentSpec,
    KnowledgeConfig,
    LanguagePolicy,
    Limits,
    ProviderSelection,
    RuntimeConfig,
)
from tring.cost.meter import CostMeter
from tring.cost.rates import RateCard
from tring.events import CostComponent, UserTranscript
from tring.knowledge import (
    CHROMA_RATES,
    KEYWORD_LOCAL_RATES,
    QDRANT_RATES,
    SEARCH_KNOWLEDGE,
    ChromaKnowledge,
    KnowledgeBinding,
    KnowledgeProvider,
    Passage,
    QdrantKnowledge,
)
from tring.knowledge.binding import ToolHandler, _overlaps
from tring.knowledge.keyword_local import (
    DEFAULT_B,
    DEFAULT_K1,
    KeywordLocalKnowledge,
    chunk_text,
    coerce_documents,
    load_directory,
    tokenize,
)
from tring.providers.base import (
    LLMChunk,
    LLMProvider,
    STTProvider,
    STTResult,
    TTSChunk,
    TTSProvider,
)
from tring.providers.registry import UnknownProviderError, create, register
from tring.runtimes.base import AudioFrame
from tring.runtimes.cascade import CascadeRuntime
from tring.session import CallSession

# ---------------------------------------------------------------------------
# Fakes: one knowledge provider and three pipeline providers
# ---------------------------------------------------------------------------

LLM_SCRIPTS: dict[str, list[str]] = {}
FAKE_PROVIDERS: dict[str, FakeKnowledge] = {}

KB = [
    Passage(
        text="We open at nine and close at five on Sunday.",
        score=2.0,
        source="faq.md#0",
        metadata={"topic": "hours"},
    ),
    Passage(
        text="Refunds are issued within fourteen days.",
        score=1.0,
        source="faq.md#1",
        metadata={"topic": "refunds"},
    ),
]


class FakeKnowledge(KnowledgeProvider):
    """Records every query and answers from a fixed list."""

    name = "test_fake"

    def __init__(self) -> None:
        self.queries: list[str] = []
        self.closed = False
        #: When set, ``search`` blocks on it -- lets a test hold a prefetch
        #: open and check what the handler does while one is in flight.
        self.gate: asyncio.Event | None = None
        self.fail = False

    async def search(self, query: str, k: int = 4) -> list[Passage]:
        self.queries.append(query)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("knowledge store unreachable")
        return list(KB[:k])

    async def aclose(self) -> None:
        self.closed = True


@register("knowledge", "test_fake")
def _make_fake(**options: Any) -> FakeKnowledge:
    provider = FakeKnowledge()
    FAKE_PROVIDERS[str(options["instance_id"])] = provider
    return provider


@register("stt", "test_knowledge_stt")
class ScriptedSTT(STTProvider):
    """One final transcript per pushed frame (the text_input trick)."""

    name = "test_knowledge_stt"

    def __init__(self, **_options: Any) -> None:
        pass

    async def transcribe(
        self, frames: AsyncIterator[AudioFrame], language: str | None = None
    ) -> AsyncIterator[STTResult]:
        async for frame in frames:
            text = frame.pcm.decode("utf-8").strip()
            if text:
                yield STTResult(text=text, final=True, language=language)


@register("llm", "test_knowledge_llm")
class ScriptedLLM(LLMProvider):
    """Replays one scripted completion per ``generate`` call, in fragments."""

    name = "test_knowledge_llm"

    def __init__(self, script_id: str = "", **_options: Any) -> None:
        self.script = list(LLM_SCRIPTS[script_id])
        self.calls: list[list[dict[str, Any]]] = []

    async def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMChunk]:
        self.calls.append([dict(m) for m in messages])
        if not self.script:
            raise AssertionError("scripted LLM ran out of responses")
        completion = self.script.pop(0)
        for i in range(0, len(completion), 11):
            yield LLMChunk(text=completion[i : i + 11])
        yield LLMChunk(text="", finish=True)


@register("tts", "test_knowledge_tts")
class ScriptedTTS(TTSProvider):
    """One audio frame per utterance; the audio itself is not under test."""

    name = "test_knowledge_tts"

    def __init__(self, **_options: Any) -> None:
        self.spoken: list[str] = []

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        buffer = ""
        async for delta in text:
            buffer += delta
        if buffer.strip():
            self.spoken.append(buffer.strip())
            yield TTSChunk(frame=AudioFrame(pcm=b"\x00\x00" * 160))


class Ticks:
    """A monotonic fake clock: every reading is one step after the last."""

    def __init__(self, step: float = 0.25) -> None:
        self.step = step
        self.now = 0.0

    def __call__(self) -> float:
        self.now += self.step
        return self.now


def stub_module(name: str, **attributes: Any) -> ModuleType:
    """Build a stand-in module for a vendor SDK that is not installed."""
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


async def settle(passes: int = 5) -> None:
    """Let background tasks (the consumer, a prefetch) reach their next await."""
    for _ in range(passes):
        await asyncio.sleep(0)


def build_agent(instance_id: str, script_id: str, speculative: bool) -> AgentSpec:
    return AgentSpec(
        name="kb-desk",
        persona="You answer questions about the shop.",
        language=LanguagePolicy(primary="en"),
        runtime=RuntimeConfig(
            routing={
                "default": ProviderSelection(
                    stt="test_knowledge_stt",
                    llm="test_knowledge_llm",
                    tts="test_knowledge_tts",
                    options={"llm": {"script_id": script_id}},
                )
            }
        ),
        knowledge=KnowledgeConfig(
            provider="test_fake",
            options={"instance_id": instance_id},
            top_k=2,
            speculative=speculative,
        ),
        limits=Limits(max_tool_calls=4),
    )


# ---------------------------------------------------------------------------
# keyword_local: BM25 math and corpus loading
# ---------------------------------------------------------------------------

CORPUS = [
    "we open at nine",
    "we close at five",
    "parking is free for guests",
]


def reference_bm25(corpus: list[str], query: str, index: int) -> float:
    """Okapi BM25 written out independently, to check the index against.

    Deliberately a second implementation rather than a call into the
    provider: a formula test that reuses the code under test only proves the
    code is self-consistent.
    """
    docs = [tokenize(text) for text in corpus]
    n = len(docs)
    avgdl = sum(len(d) for d in docs) / n
    total = 0.0
    for term in set(tokenize(query)):
        df = sum(1 for d in docs if term in d)
        freq = docs[index].count(term)
        if df == 0 or freq == 0:
            continue
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        norm = 1.0 - DEFAULT_B + DEFAULT_B * len(docs[index]) / avgdl
        total += idf * (freq * (DEFAULT_K1 + 1.0)) / (freq + DEFAULT_K1 * norm)
    return total


async def test_bm25_score_matches_the_formula_exactly() -> None:
    provider = KeywordLocalKnowledge(coerce_documents(CORPUS))
    hits = await provider.search("open")

    # Hand-computed: N=3, df("open")=1 -> idf = ln(1 + 2.5/1.5) = 0.980829;
    # avgdl = 13/3, |D| = 4 -> norm = 0.25 + 0.75*4/(13/3) = 0.942308;
    # tf part = 1*(1.5+1) / (1 + 1.5*0.942308) = 1.035857.
    assert len(hits) == 1
    assert hits[0].score == pytest.approx(1.0159984, abs=1e-6)
    assert hits[0].score == pytest.approx(reference_bm25(CORPUS, "open", 0), abs=1e-12)


async def test_bm25_ranks_by_relevance_and_breaks_ties_deterministically() -> None:
    provider = KeywordLocalKnowledge(coerce_documents(CORPUS))

    # "at" appears in two of three documents, "nine" in one: the document
    # carrying the rarer term must win.
    ranked = await provider.search("at nine")
    assert [p.source for p in ranked] == ["inline:0", "inline:1"]
    assert ranked[0].score > ranked[1].score
    for position, passage in enumerate(ranked):
        assert passage.score == pytest.approx(
            reference_bm25(CORPUS, "at nine", position), abs=1e-12
        )

    # A term both documents share scores them identically; the tie is broken
    # by source, so a replayed call retrieves the same passage every time.
    tied = await provider.search("we")
    assert [p.source for p in tied] == ["inline:0", "inline:1"]
    assert tied[0].score == pytest.approx(tied[1].score)


async def test_repeated_query_words_do_not_inflate_the_score() -> None:
    """A caller repeating themselves is a disfluency, not a relevance signal."""
    provider = KeywordLocalKnowledge(coerce_documents(CORPUS))
    once = await provider.search("nine")
    twice = await provider.search("nine nine nine")
    assert once[0].score == pytest.approx(twice[0].score)


async def test_non_matching_passages_are_dropped_not_padded() -> None:
    provider = KeywordLocalKnowledge(coerce_documents(CORPUS))
    assert await provider.search("helicopter", k=4) == []
    assert len(await provider.search("we", k=4)) == 2  # not padded up to k
    assert await provider.search("we", k=0) == []
    assert await provider.search("", k=4) == []
    assert await KeywordLocalKnowledge([]).search("we") == []


async def test_unicode_corpus_is_tokenized_not_silently_dropped() -> None:
    """An ASCII-only tokenizer would index a Devanagari corpus as nothing."""
    corpus = ["हम नौ बजे खुलते हैं", "we open at nine"]
    provider = KeywordLocalKnowledge(coerce_documents(corpus))
    assert [p.source for p in await provider.search("नौ बजे")] == ["inline:0"]


def test_chunking_merges_headings_into_the_text_they_head() -> None:
    text = (
        "# Hours\n\n"
        "We open at nine and close at five, Monday through Saturday. "
        "On Sunday we are closed all day.\n\n"
        "# Parking\n\n"
        "Parking is free for guests staying overnight, and five pounds "
        "otherwise for the first two hours.\n"
    )
    chunks = chunk_text(text, source="faq.md")

    assert len(chunks) == 2
    assert chunks[0].text.startswith("# Hours\n")
    assert "Monday through Saturday" in chunks[0].text
    assert chunks[1].text.startswith("# Parking\n")
    assert [c.source for c in chunks] == ["faq.md#0", "faq.md#1"]
    assert chunks[0].metadata == {"source_path": "faq.md", "chunk": 0}

    # A trailing fragment with nothing to merge into is kept, not dropped.
    assert [c.text for c in chunk_text("# Hours\n", source="faq.md")] == ["# Hours"]


def test_load_directory_reads_recursively_and_ignores_other_file_types(
    tmp_path: Path,
) -> None:
    (tmp_path / "faq.md").write_text(
        "We open at nine and close at five every day except Sunday.\n", encoding="utf-8"
    )
    (tmp_path / "notes.csv").write_text("a,b,c\n", encoding="utf-8")
    nested = tmp_path / "policies"
    nested.mkdir()
    (nested / "refunds.txt").write_text(
        "Refunds are issued within fourteen days of purchase.\n", encoding="utf-8"
    )

    documents = load_directory(tmp_path)
    assert [d.source for d in documents] == ["faq.md#0", "policies/refunds.txt#0"]

    with pytest.raises(NotADirectoryError):
        load_directory(tmp_path / "faq.md")


async def test_keyword_local_factory_composes_inline_documents_and_a_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "faq.md").write_text(
        "Parking is free for guests staying overnight at the hotel.\n", encoding="utf-8"
    )
    provider = create(
        "knowledge",
        "keyword_local",
        path=str(tmp_path),
        documents=[{"text": "We open at nine.", "source": "hours", "metadata": {"a": 1}}],
    )
    assert isinstance(provider, KeywordLocalKnowledge)
    assert len(provider) == 2

    assert [p.source for p in await provider.search("parking")] == ["faq.md#0"]
    hits = await provider.search("open")
    assert hits[0].source == "hours"
    assert hits[0].metadata == {"a": 1}

    with pytest.raises(ValueError, match="keyword_local needs a corpus"):
        create("knowledge", "keyword_local")
    with pytest.raises(ValueError, match="has no 'text' key"):
        create("knowledge", "keyword_local", documents=[{"source": "x"}])


# ---------------------------------------------------------------------------
# The binding, through the real cascade runtime
# ---------------------------------------------------------------------------

TOOL_TURN = (
    '{"speak": "Let me check that.", "tool_call": {"name": "search_knowledge", '
    '"arguments": {"query": "opening hours sunday", '
    '"waiting_message": "One moment.", "spoken_mode": "answer_pending", '
    '"post_tool_response": "respond"}}}'
)
ANSWER_TURN = '{"speak": "We open at nine and close at five.", "tool_call": null}'


async def test_tool_flow_end_to_end_through_the_cascade() -> None:
    LLM_SCRIPTS["e2e"] = [TOOL_TURN, ANSWER_TURN]
    agent = build_agent("e2e", "e2e", speculative=False)
    session = CallSession(agent)

    handlers: dict[str, ToolHandler] = {}
    binding = KnowledgeBinding(session, agent.knowledge, handlers, clock=Ticks())
    # The binding is what puts the tool on the spec, and it must do so before
    # the runtime reads that list.
    assert [tool.name for tool in agent.tools] == [SEARCH_KNOWLEDGE]
    assert set(handlers) == {SEARCH_KNOWLEDGE}

    runtime = CascadeRuntime(session, handlers=handlers)
    try:
        await runtime.start()
        await runtime.push_audio(AudioFrame(pcm=b"what time do you close on sunday"))
        await runtime.drain()
        await runtime.stop()
    finally:
        await binding.aclose()

    assert [event.type for event in session.history] == [
        "session_started",
        "user_transcript",
        "bot_utterance",  # spoken before the search, not after it
        "bot_speech_played",
        "tool_call_started",
        "bot_utterance",  # the waiting message
        "bot_speech_played",
        "knowledge_searched",
        "tool_call_completed",
        "bot_utterance",  # the answer, composed from the passages
        "bot_speech_played",
        "session_ended",
    ]

    searched = next(e for e in session.history if e.type == "knowledge_searched")
    assert searched.query == "opening hours sunday"
    assert searched.result_count == 2  # top_k from the spec
    assert searched.speculative is False
    assert searched.latency_seconds == pytest.approx(0.25)

    completed = next(e for e in session.history if e.type == "tool_call_completed")
    assert completed.ok is True

    # The passages reached the model: the follow-up turn carries them as the
    # tool result, which is the only way the answer can be grounded.
    llm = runtime._llm
    assert isinstance(llm, ScriptedLLM)
    tool_message = next(m for m in llm.calls[1] if m["role"] == "tool")
    assert "We open at nine and close at five on Sunday." in tool_message["content"]
    assert '"prefetched": false' in tool_message["content"]

    # And the tool was advertised in the (cacheable) prompt prefix, with the
    # choreography fields folded into its schema like any other tool.
    schema_message = llm.calls[0][2]["content"]
    assert SEARCH_KNOWLEDGE in schema_message
    assert "waiting_message" in schema_message


async def test_tool_failure_becomes_a_spoken_apology_not_a_dropped_call() -> None:
    LLM_SCRIPTS["fail"] = [TOOL_TURN, ANSWER_TURN]
    agent = build_agent("fail", "fail", speculative=False)
    session = CallSession(agent)
    handlers: dict[str, ToolHandler] = {}
    binding = KnowledgeBinding(session, agent.knowledge, handlers)
    FAKE_PROVIDERS["fail"].fail = True

    runtime = CascadeRuntime(session, handlers=handlers)
    try:
        await runtime.start()
        await runtime.push_audio(AudioFrame(pcm=b"what time do you close on sunday"))
        await runtime.drain()
        await runtime.stop()
    finally:
        await binding.aclose()

    completed = next(e for e in session.history if e.type == "tool_call_completed")
    assert completed.ok is False
    assert completed.result_summary is not None
    assert "knowledge store unreachable" in completed.result_summary
    # The caller still hears a closing line and the call ends normally.
    assert [e.type for e in session.history][-3:] == [
        "bot_utterance",
        "bot_speech_played",
        "session_ended",
    ]


async def test_binding_without_a_knowledge_config_is_inert() -> None:
    LLM_SCRIPTS["inert"] = [ANSWER_TURN]
    agent = build_agent("inert", "inert", speculative=False)
    agent.knowledge = None
    session = CallSession(agent)

    handlers: dict[str, ToolHandler] = {}
    binding = KnowledgeBinding(session, agent.knowledge, handlers)
    await binding.start()
    await binding.aclose()

    assert handlers == {}
    assert agent.tools == []
    assert binding.tool is None
    assert binding.provider is None


def test_unknown_knowledge_provider_names_the_available_ones() -> None:
    with pytest.raises(UnknownProviderError, match="keyword_local"):
        create("knowledge", "not_a_store")


# ---------------------------------------------------------------------------
# Speculative retrieval
# ---------------------------------------------------------------------------


async def bind_speculative(
    instance_id: str,
) -> tuple[CallSession, KnowledgeBinding, dict[str, ToolHandler], FakeKnowledge]:
    """A session with speculation on, and nothing else running."""
    LLM_SCRIPTS.setdefault(instance_id, [ANSWER_TURN])
    agent = build_agent(instance_id, instance_id, speculative=True)
    session = CallSession(agent)
    handlers: dict[str, ToolHandler] = {}
    binding = KnowledgeBinding(session, agent.knowledge, handlers, clock=Ticks())
    await binding.start()
    return session, binding, handlers, FAKE_PROVIDERS[instance_id]


def emit_partial(session: CallSession, text: str) -> None:
    session.emit(
        UserTranscript(
            session_id=session.session_id, at=session.elapsed, text=text, final=False
        )
    )


async def drain_prefetch(binding: KnowledgeBinding) -> None:
    """Wait for the in-flight prefetch, if any, to finish."""
    await settle()
    task = binding._prefetch
    if task is not None:
        await asyncio.wait({task})


async def test_speculative_prefetch_answers_the_tool_call_without_searching_again() -> None:
    session, binding, handlers, provider = await bind_speculative("spec")
    try:
        emit_partial(session, "what time do you close on sun")
        await drain_prefetch(binding)

        speculative = [e for e in session.history if e.type == "knowledge_searched"]
        assert len(speculative) == 1
        assert speculative[0].speculative is True
        assert speculative[0].query == "what time do you close on sun"
        assert speculative[0].result_count == 2
        assert speculative[0].latency_seconds == pytest.approx(0.25)

        # The model's tidied-up query is a subset of the partial transcript,
        # so the prefetch answers it and no second search runs.
        result = await handlers[SEARCH_KNOWLEDGE]({"query": "what time close sun"})
        assert result["prefetched"] is True
        assert result["passages"][0]["source"] == "faq.md#0"
        assert result["passages"][0]["text"] == KB[0].text
        assert provider.queries == ["what time do you close on sun"]
        assert len([e for e in session.history if e.type == "knowledge_searched"]) == 1
    finally:
        await binding.aclose()
    assert provider.closed is True  # aclose releases the provider too


async def test_unrelated_query_ignores_the_prefetch_and_searches() -> None:
    session, binding, handlers, provider = await bind_speculative("miss")
    try:
        emit_partial(session, "what time do you close on sunday")
        await drain_prefetch(binding)

        result = await handlers[SEARCH_KNOWLEDGE]({"query": "refund policy"})
        assert result["prefetched"] is False
        assert provider.queries == ["what time do you close on sunday", "refund policy"]

        events = [e for e in session.history if e.type == "knowledge_searched"]
        assert [e.speculative for e in events] == [True, False]
    finally:
        await binding.aclose()


async def test_handler_waits_for_a_matching_prefetch_already_in_flight() -> None:
    """The turn is blocked on this answer either way; joining beats duplicating."""
    session, binding, handlers, provider = await bind_speculative("inflight")
    provider.gate = asyncio.Event()
    try:
        emit_partial(session, "what time do you close on sunday")
        await settle()  # the prefetch is now parked on the gate

        async def ask() -> Any:
            return await handlers[SEARCH_KNOWLEDGE]({"query": "close sunday"})

        call = asyncio.create_task(ask())
        await settle()
        assert not call.done()  # waiting on the prefetch, not on a second search

        provider.gate.set()
        result = await call
        assert result["prefetched"] is True
        assert provider.queries == ["what time do you close on sunday"]
    finally:
        provider.gate = None
        await binding.aclose()


async def test_newest_partial_wins_and_identical_partials_are_deduped() -> None:
    session, binding, _handlers, provider = await bind_speculative("latest")
    try:
        emit_partial(session, "what time do you close on")
        emit_partial(session, "what time do you close on sunday")
        await drain_prefetch(binding)
        # The superseded partial's search was cancelled before it ever ran.
        assert provider.queries == ["what time do you close on sunday"]

        # Re-emitted and whitespace-only-different partials are not requeried.
        emit_partial(session, "what time do you close on sunday")
        emit_partial(session, "  what time do you   close on sunday  ")
        await settle()
        assert provider.queries == ["what time do you close on sunday"]

        # Too short to be worth a query: two characters match everything.
        emit_partial(session, "wh")
        await settle()
        assert provider.queries == ["what time do you close on sunday"]
    finally:
        await binding.aclose()


async def test_final_transcripts_do_not_trigger_speculation() -> None:
    """Speculation exists to beat the end of the turn; after it, it is waste."""
    session, binding, _handlers, provider = await bind_speculative("final")
    try:
        session.emit(
            UserTranscript(
                session_id=session.session_id,
                at=session.elapsed,
                text="what time do you close on sunday",
                final=True,
            )
        )
        await settle()
        assert provider.queries == []
    finally:
        await binding.aclose()


async def test_a_failing_prefetch_is_reported_and_never_reaches_the_caller() -> None:
    session, binding, handlers, provider = await bind_speculative("specfail")
    try:
        provider.fail = True
        emit_partial(session, "what time do you close on sunday")
        await drain_prefetch(binding)

        errors = [e for e in session.history if e.type == "error"]
        assert len(errors) == 1
        assert "speculative knowledge search failed" in errors[0].message
        assert errors[0].recoverable is True
        assert not [e for e in session.history if e.type == "knowledge_searched"]

        # The handler still works: it just searches for itself.
        provider.fail = False
        result = await handlers[SEARCH_KNOWLEDGE]({"query": "close sunday"})
        assert result["prefetched"] is False
    finally:
        await binding.aclose()


async def test_empty_query_argument_fails_the_tool_call() -> None:
    _session, binding, handlers, _provider = await bind_speculative("empty")
    try:
        with pytest.raises(ValueError, match="non-empty 'query'"):
            await handlers[SEARCH_KNOWLEDGE]({"query": "   "})
    finally:
        await binding.aclose()


def test_overlap_is_containment_against_the_shorter_query() -> None:
    assert _overlaps("close sunday", "what time do you close on sunday", 0.6)
    assert not _overlaps("refund policy", "what time do you close on sunday", 0.6)
    assert not _overlaps("", "anything", 0.6)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


async def test_every_search_is_metered_including_the_speculative_one() -> None:
    session, binding, handlers, _provider = await bind_speculative("cost")
    # Price the fake with the rate the real keyword_local ships, so the card
    # has an opinion and the missing-rate path stays meaningful.
    card = RateCard(
        version="test-knowledge",
        rates=[KEYWORD_LOCAL_RATES[0].model_copy(update={"provider": "test_fake"})],
    )
    meter = CostMeter(session, card)
    binding.meter = meter
    try:
        emit_partial(session, "what time do you close on sunday")
        await drain_prefetch(binding)
        await handlers[SEARCH_KNOWLEDGE]({"query": "refund policy"})
    finally:
        await binding.aclose()

    costs = [e for e in session.history if e.type == "cost_recorded"]
    assert len(costs) == 2  # speculation is paid for even when it is not used
    assert all(e.component is CostComponent.OTHER for e in costs)
    assert all(e.unit_name == "knowledge_queries" and e.units == 1.0 for e in costs)
    assert all(e.estimated is False and e.amount == 0.0 for e in costs)
    assert meter.missing_rate_notes == []


def test_rate_cards_are_dated_and_honest() -> None:
    for rate in [*KEYWORD_LOCAL_RATES, *CHROMA_RATES]:
        assert rate.as_of == "2026-09"
        assert rate.unit_name == "knowledge_queries"
        assert rate.price_per_unit == 0.0  # local compute, no vendor charge
    # Qdrant publishes no per-query price, so a 0.0 rate would be a guess; the
    # meter's missing-rate note is left to do the reporting instead.
    assert QDRANT_RATES == []


# ---------------------------------------------------------------------------
# Vector-store adapters: lazy imports, request shape, result mapping
# ---------------------------------------------------------------------------


async def test_chroma_missing_dependency_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A None entry in sys.modules is how Python reports "this import is
    # blocked": `import chromadb` then raises ImportError.
    monkeypatch.setitem(sys.modules, "chromadb", None)
    provider = ChromaKnowledge(path="/tmp/chroma", collection="kb")
    with pytest.raises(ImportError, match=r"tring\[knowledge\]"):
        await provider.search("hours")


async def test_qdrant_missing_dependency_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    provider = QdrantKnowledge(
        url="http://localhost:6333", collection="kb", embedding_model="m"
    )
    with pytest.raises(ImportError, match=r"tring\[knowledge\]"):
        await provider.search("hours")


class StubChromaCollection:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] = {}

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.kwargs = kwargs
        return {
            "ids": [["a1", "a2"]],
            "documents": [["We open at nine.", "Refunds take fourteen days."]],
            "metadatas": [[{"topic": "hours"}, None]],
            "distances": [[0.25, 1.0]],
        }


async def test_chroma_builds_the_documented_request_and_maps_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = StubChromaCollection()
    opened: list[tuple[str, str]] = []

    class StubClient:
        def __init__(self, path: str) -> None:
            self.path = path

        def get_collection(self, name: str) -> StubChromaCollection:
            opened.append((self.path, name))
            return collection

    monkeypatch.setitem(
        sys.modules, "chromadb", stub_module("chromadb", PersistentClient=StubClient)
    )

    provider = ChromaKnowledge(path="/tmp/kb", collection="shop")
    passages = await provider.search("opening hours", k=2)

    assert opened == [("/tmp/kb", "shop")]
    assert collection.kwargs == {
        "query_texts": ["opening hours"],
        "n_results": 2,
        "include": ["documents", "metadatas", "distances"],
    }
    assert [p.source for p in passages] == ["shop#a1", "shop#a2"]
    assert passages[0].text == "We open at nine."
    # Distance -> higher-is-better score, ranking preserved, raw distance kept.
    assert passages[0].score == pytest.approx(1.0 / 1.25)
    assert passages[1].score == pytest.approx(0.5)
    assert passages[0].metadata == {"topic": "hours", "distance": 0.25}
    assert passages[1].metadata == {"distance": 1.0}

    # The store is opened once and reused: it loads an embedding model.
    await provider.search("parking", k=1)
    assert opened == [("/tmp/kb", "shop")]


async def test_qdrant_builds_the_documented_request_and_maps_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    built: list[dict[str, Any]] = []
    closed: list[bool] = []

    class StubDocument:
        def __init__(self, text: str, model: str) -> None:
            self.text = text
            self.model = model

    class StubAsyncClient:
        def __init__(self, **kwargs: Any) -> None:
            built.append(kwargs)

        async def query_points(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        id="p1",
                        score=0.87,
                        payload={"text": "We open at nine.", "topic": "hours"},
                    ),
                    SimpleNamespace(id=7, score=0.41, payload=None),
                ]
            )

        async def close(self) -> None:
            closed.append(True)

    monkeypatch.setitem(
        sys.modules,
        "qdrant_client",
        stub_module(
            "qdrant_client",
            AsyncQdrantClient=StubAsyncClient,
            models=stub_module("qdrant_client.models", Document=StubDocument),
        ),
    )
    monkeypatch.setenv("QDRANT_KEY", "secret-value")

    provider = QdrantKnowledge(
        url="https://cluster.example:6333",
        collection="shop",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        api_key_env="QDRANT_KEY",
    )
    passages = await provider.search("opening hours", k=2)

    assert built == [
        {
            "url": "https://cluster.example:6333",
            "api_key": "secret-value",
            "cloud_inference": False,
        }
    ]
    call = calls[0]
    assert call["collection_name"] == "shop"
    assert call["limit"] == 2
    assert call["with_payload"] is True
    assert call["using"] is None  # the collection's default vector
    assert isinstance(call["query"], StubDocument)
    assert call["query"].text == "opening hours"
    assert call["query"].model == "sentence-transformers/all-MiniLM-L6-v2"

    assert [p.source for p in passages] == ["shop#p1", "shop#7"]
    assert passages[0].text == "We open at nine."
    assert passages[0].score == pytest.approx(0.87)
    assert passages[0].metadata == {"topic": "hours"}  # text is not duplicated
    assert passages[1].text == ""  # a payload with no text field is not a crash

    await provider.aclose()
    assert closed == [True]


async def test_qdrant_never_accepts_a_raw_key_only_an_env_var_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "qdrant_client",
        stub_module(
            "qdrant_client",
            AsyncQdrantClient=object,
            models=stub_module("qdrant_client.models"),
        ),
    )
    monkeypatch.delenv("QDRANT_KEY", raising=False)

    provider = QdrantKnowledge(
        url="https://cluster.example:6333",
        collection="shop",
        embedding_model="m",
        api_key_env="QDRANT_KEY",
    )
    with pytest.raises(RuntimeError, match="QDRANT_KEY"):
        await provider.search("hours")


def test_vector_store_factories_require_the_options_that_cannot_be_guessed() -> None:
    with pytest.raises(ValueError, match=r"knowledge\.options\.collection"):
        create("knowledge", "chroma_local", path="/tmp/kb")
    with pytest.raises(ValueError, match=r"knowledge\.options\.collection"):
        create("knowledge", "qdrant")
    # A query embedded by the wrong model returns confident nonsense rather
    # than an error, so the model name has no default.
    with pytest.raises(ValueError, match="embedding_model"):
        create("knowledge", "qdrant", collection="shop")

    provider = create(
        "knowledge",
        "qdrant",
        collection="shop",
        embedding_model="m",
        vector_name="dense",
        text_field="body",
    )
    assert isinstance(provider, QdrantKnowledge)
    assert provider.url == "http://localhost:6333"
    assert provider.vector_name == "dense"
    assert provider.text_field == "body"
    assert provider.api_key_env is None


async def test_vector_store_searches_short_circuit_on_empty_input() -> None:
    """No client is opened at all, so a blank query cannot cost a round trip."""
    chroma = ChromaKnowledge(path="/nonexistent", collection="kb")
    assert await chroma.search("   ") == []
    assert await chroma.search("hours", k=0) == []

    qdrant = QdrantKnowledge(
        url="http://localhost:6333", collection="kb", embedding_model="m"
    )
    assert await qdrant.search("") == []
    await qdrant.aclose()  # nothing to close, and nothing to blow up on
