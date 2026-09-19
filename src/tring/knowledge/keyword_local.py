"""``keyword_local`` -- a zero-dependency BM25 index over a small corpus.

Why ship a lexical retriever at all, in a stack that also adapts Chroma and
Qdrant? Three reasons, in order of how often they matter:

1. **The knowledge slot must be testable offline.** Every other test in this
   repo runs with no network, no model weights and no GPU; a retrieval
   feature whose only implementations need an embedding model would be a
   feature nobody can regression-test.
2. **Most voice knowledge bases are tiny.** Opening hours, the return
   policy, the four plans and what each includes -- a few dozen paragraphs.
   At that size a vector store is operational weight (a service, an
   embedding model, a re-index step) bought for very little recall.
3. **Exact-term questions are the common case on the phone.** Callers ask
   about "the *premium* plan" and "order *A-4417*" -- literal tokens, which
   is precisely where lexical scoring is strong and dense retrieval is
   weakest.

The scoring is Okapi BM25 (Robertson & Sparck Jones), the standard baseline:

.. math::

    score(D, Q) = \\sum_{q \\in Q} IDF(q) \\cdot
        \\frac{f(q, D) \\cdot (k_1 + 1)}
             {f(q, D) + k_1 \\cdot (1 - b + b \\cdot \\frac{|D|}{avgdl})}

with ``IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))``, ``k1 = 1.5`` and
``b = 0.75``. Two deliberate choices in that sum:

* **Query terms are deduplicated.** The classic formulation sums over query
  terms *with* repetition (or weights them with a third ``k3`` saturation
  parameter). A caller who says "hours, what are your hours" would otherwise
  double the weight of one word purely for repeating themselves, which is
  a speech disfluency, not a relevance signal.
* **The ``+1`` inside the IDF log** keeps every IDF positive, so a term that
  appears in *every* document contributes ~0 instead of a negative score
  that would push a matching passage below a non-matching one.

Cost: none. Everything here is pure Python over an in-memory list, which
also bounds where it is sensible -- scoring is linear in corpus size on
every query, so a few thousand chunks is the honest ceiling. Past that,
point the slot at ``chroma_local`` or ``qdrant``.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tring.cost.rates import Rate
from tring.events import CostComponent
from tring.knowledge.base import KnowledgeProvider, Passage
from tring.providers.registry import register

#: Term-frequency saturation. 1.2-2.0 is the range the literature settles on;
#: 1.5 is the middle of it and what most implementations default to.
DEFAULT_K1 = 1.5

#: Length normalization: 0 ignores document length entirely, 1 divides it out
#: completely. 0.75 is the standard compromise, and it matters here because a
#: chunked FAQ mixes one-line answers with long policy paragraphs.
DEFAULT_B = 0.75

#: File types a corpus directory is read from. Deliberately text-only: parsing
#: PDF or HTML is a different job with different dependencies, and doing it
#: badly (silently ingesting nav bars and page furniture) poisons retrieval.
DEFAULT_EXTENSIONS: tuple[str, ...] = (".txt", ".md")

#: A chunk shorter than this is merged into the one that follows it. This is
#: what keeps a Markdown heading ("## Refunds") from becoming a passage of its
#: own -- it would score highly on the word "refunds" and then tell the caller
#: nothing at all.
DEFAULT_MIN_CHUNK_CHARS = 80

#: Unicode-aware word tokenizer. ``\\w+`` rather than ``[a-z0-9]+`` on purpose:
#: this stack routes Hindi and Marathi calls, and an ASCII-only tokenizer would
#: reduce a Devanagari corpus to zero tokens without raising anything.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

#: Paragraph split: one or more blank lines, whatever whitespace is on them.
_PARAGRAPH_RE = re.compile(r"\n\s*\n")

# ---------------------------------------------------------------------------
# Rates
#
# A ``keyword_local`` query costs nothing: no vendor, no network, no model.
# It is listed explicitly rather than left out so that a missing-rate note
# from the cost meter stays meaningful -- an absent rate should mean "nobody
# priced this", never "this one happens to be free".
# ---------------------------------------------------------------------------
KEYWORD_LOCAL_RATES: list[Rate] = [
    Rate(
        component=CostComponent.OTHER,
        provider="keyword_local",
        model=None,
        unit_name="knowledge_queries",
        price_per_unit=0.0,
        currency="USD",
        as_of="2026-09",
    )
]


def tokenize(text: str) -> list[str]:
    """Lowercase, Unicode-aware word tokens. The whole analyzer, on purpose.

    No stemming and no stop-word list. Stemming needs a per-language stemmer
    (and gets Indian languages wrong more often than it helps), and stop
    words are already handled by IDF: a word in every document earns an IDF
    of about zero and stops mattering without anyone maintaining a list.
    """
    return _TOKEN_RE.findall(text.casefold())


@dataclass(frozen=True)
class Document:
    """One corpus entry, before it is indexed.

    This is the *input* shape, distinct from :class:`~tring.knowledge.base.Passage`
    (the output shape) because a corpus entry has no score: a score only
    exists relative to a query.
    """

    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Indexed:
    """A document plus the per-document statistics BM25 needs."""

    document: Document
    freqs: Counter[str]
    length: int


def chunk_text(
    text: str, source: str, min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS
) -> list[Document]:
    """Split one file into retrievable passages on blank lines.

    Paragraphs, not sentences and not fixed-size windows: a paragraph is the
    unit a human wrote as one complete thought, so it is the unit most likely
    to answer a question on its own -- which is the whole requirement for a
    passage that will be read aloud.

    Short fragments are merged forward into the next paragraph
    (``min_chunk_chars``), so headings travel with the text they head and the
    index never contains a passage that matches a query but answers nothing.
    Trailing fragments with nothing to merge into are kept as-is rather than
    dropped: the last line of a file is sometimes the whole answer.
    """
    chunks: list[Document] = []
    pending = ""
    for raw in _PARAGRAPH_RE.split(text):
        part = raw.strip()
        if not part:
            continue
        merged = f"{pending}\n{part}" if pending else part
        if len(merged) < min_chunk_chars:
            pending = merged
            continue
        chunks.append(
            Document(
                text=merged,
                source=f"{source}#{len(chunks)}",
                metadata={"source_path": source, "chunk": len(chunks)},
            )
        )
        pending = ""
    if pending:
        chunks.append(
            Document(
                text=pending,
                source=f"{source}#{len(chunks)}",
                metadata={"source_path": source, "chunk": len(chunks)},
            )
        )
    return chunks


def load_directory(
    path: str | Path,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
) -> list[Document]:
    """Read every matching file under ``path`` (recursively) into chunks.

    Files are visited in sorted order so an index built twice from the same
    directory is byte-identical -- which is what makes a retrieval regression
    reproducible instead of dependent on filesystem iteration order.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(
            f"keyword_local corpus path {str(root)!r} is not a directory; pass "
            "options.path (a folder of .txt/.md files) or options.documents "
            "(an inline list)"
        )
    wanted = {ext.lower() for ext in extensions}
    documents: list[Document] = []
    for file in sorted(root.rglob("*")):
        if not file.is_file() or file.suffix.lower() not in wanted:
            continue
        source = file.relative_to(root).as_posix()
        documents.extend(
            chunk_text(
                file.read_text(encoding="utf-8"),
                source=source,
                min_chunk_chars=min_chunk_chars,
            )
        )
    return documents


def coerce_documents(raw: Iterable[str | dict[str, Any]]) -> list[Document]:
    """Accept an inline corpus as plain strings or ``{text, source, metadata}``.

    Inline documents are *not* chunked. Someone who wrote the list by hand
    already decided where one answer ends and the next begins; re-splitting it
    would be the library overruling them.
    """
    documents: list[Document] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            documents.append(Document(text=item, source=f"inline:{index}"))
            continue
        text = str(item.get("text", ""))
        if not text:
            raise ValueError(
                f"inline knowledge document {index} has no 'text' key: "
                'expected a string or {"text": ..., "source": ..., "metadata": {...}}'
            )
        metadata = item.get("metadata") or {}
        documents.append(
            Document(
                text=text,
                source=str(item.get("source", f"inline:{index}")),
                metadata=dict(metadata),
            )
        )
    return documents


class KeywordLocalKnowledge(KnowledgeProvider):
    """BM25 over an in-memory corpus. No index files, no daemon, no network.

    The index is built once in ``__init__`` and is immutable afterwards: a
    knowledge base that changes mid-call would make two turns of the same
    conversation disagree, and re-reading a directory per query would put
    filesystem latency inside the caller's turn.
    """

    name = "keyword_local"

    def __init__(
        self,
        documents: Sequence[Document],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[_Indexed] = []
        document_frequency: Counter[str] = Counter()
        for document in documents:
            tokens = tokenize(document.text)
            if not tokens:
                continue  # an empty or punctuation-only chunk can never match
            freqs = Counter(tokens)
            self._docs.append(
                _Indexed(document=document, freqs=freqs, length=len(tokens))
            )
            document_frequency.update(freqs.keys())

        total_length = sum(doc.length for doc in self._docs)
        # An empty corpus keeps a harmless avgdl of 1.0 so the scorer has no
        # division-by-zero branch; it never runs, because search() returns
        # early when there is nothing to score.
        self._avgdl = total_length / len(self._docs) if self._docs else 1.0
        corpus_size = len(self._docs)
        self._idf: dict[str, float] = {
            term: math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def __len__(self) -> int:
        """Number of indexed passages -- the number a corpus loader is checked against."""
        return len(self._docs)

    async def search(self, query: str, k: int = 4) -> list[Passage]:
        """Score every passage against ``query`` and return the best ``k``.

        Passages that score zero (they share no term with the query) are
        dropped rather than padded in to reach ``k``. Handing a model four
        passages when only one is relevant is how a confident wrong answer
        gets built.
        """
        terms = set(tokenize(query))
        if not terms or not self._docs or k <= 0:
            return []

        scored: list[tuple[float, str, _Indexed]] = []
        for indexed in self._docs:
            score = self._score(indexed, terms)
            if score > 0.0:
                # The source rides along as the tie-break key: equal scores
                # must order deterministically, or a replayed call can return
                # a different passage than the one the caller heard about.
                scored.append((score, indexed.document.source, indexed))
        scored.sort(key=lambda row: (-row[0], row[1]))

        return [
            Passage(
                text=indexed.document.text,
                score=score,
                source=indexed.document.source,
                metadata=dict(indexed.document.metadata),
            )
            for score, _source, indexed in scored[:k]
        ]

    def _score(self, indexed: _Indexed, terms: set[str]) -> float:
        """BM25 for one document against a set of (deduplicated) query terms."""
        total = 0.0
        norm = 1.0 - self.b + self.b * indexed.length / self._avgdl
        for term in terms:
            freq = indexed.freqs.get(term, 0)
            if not freq:
                continue
            idf = self._idf[term]  # a term with freq > 0 is always in the vocab
            total += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm)
        return total


@register("knowledge", "keyword_local")
def _make_keyword_local(**options: Any) -> KeywordLocalKnowledge:
    """Build the provider from spec options.

    ``path`` (a directory of .txt/.md files) and ``documents`` (an inline
    list) compose rather than exclude each other: a spec can ship a handful
    of inline answers alongside a folder of policy text without a second
    provider.
    """
    documents: list[Document] = []
    inline = options.get("documents")
    if inline is not None:
        if not isinstance(inline, Iterable) or isinstance(inline, str | bytes):
            raise TypeError(
                "keyword_local option 'documents' must be a list of strings or "
                'of {"text": ..., "source": ..., "metadata": {...}} objects'
            )
        documents.extend(coerce_documents(inline))
    path = options.get("path")
    if path is not None:
        documents.extend(
            load_directory(
                str(path),
                extensions=tuple(options.get("extensions", DEFAULT_EXTENSIONS)),
                min_chunk_chars=int(options.get("min_chunk_chars", DEFAULT_MIN_CHUNK_CHARS)),
            )
        )
    if not documents:
        raise ValueError(
            "keyword_local needs a corpus: set knowledge.options.path to a "
            "directory of .txt/.md files, or knowledge.options.documents to an "
            "inline list"
        )
    return KeywordLocalKnowledge(
        documents,
        k1=float(options.get("k1", DEFAULT_K1)),
        b=float(options.get("b", DEFAULT_B)),
    )


__all__ = [
    "DEFAULT_B",
    "DEFAULT_EXTENSIONS",
    "DEFAULT_K1",
    "DEFAULT_MIN_CHUNK_CHARS",
    "KEYWORD_LOCAL_RATES",
    "Document",
    "KeywordLocalKnowledge",
    "chunk_text",
    "coerce_documents",
    "load_directory",
    "tokenize",
]
