"""``chroma_local`` -- knowledge from an on-disk Chroma collection.

Chroma is the shortest path from "I have a folder of documents" to dense
retrieval: it embeds with a bundled local model, stores vectors on disk and
needs no service to run. That makes it the natural upgrade from
``keyword_local`` for a knowledge base that has outgrown lexical scoring but
has not earned a vector database cluster.

This adapter is deliberately **read-only**. Building and re-indexing a
collection is an offline job with its own cadence (a nightly export, a CMS
webhook), and doing it from inside a live call would put embedding latency
and write contention on the path of a caller waiting for an answer. Point
this at a collection something else maintains.

API verified against the official Chroma docs, 2026-09:

* ``chromadb.PersistentClient(path=...)`` and ``get_collection(name=...)``:
  https://docs.trychroma.com/reference/python/client
* ``collection.query(query_texts=..., n_results=..., include=...)`` returning
  column-major lists nested one level per query
  (``ids``/``documents``/``metadatas``/``distances``):
  https://docs.trychroma.com/docs/querying-collections/query-and-get

Threading note. Chroma's Python client is synchronous, and its query does
real work in-process: it embeds the query text with the collection's
embedding function (by default a local ONNX model) before it searches. Run
on the event loop, that would stall audio for every millisecond of it, so
every call into the client here goes through ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tring.cost.rates import Rate
from tring.events import CostComponent
from tring.knowledge.base import KnowledgeProvider, Passage
from tring.providers.registry import register

#: What Chroma is asked to return. ``ids`` comes back regardless of
#: ``include``; the other three are requested explicitly because a passage
#: with no text is useless and a passage with no provenance is unauditable.
_INCLUDE = ["documents", "metadatas", "distances"]

_IMPORT_HINT = (
    "chroma_local requires the 'chromadb' package: pip install 'tring[knowledge]' "
    "(or pip install chromadb). The core install stays pydantic + pyyaml only."
)

# ---------------------------------------------------------------------------
# Rates
#
# This adapter talks to a *local* PersistentClient: the vectors are on the
# caller's disk and the query embedding is computed in-process, so there is no
# vendor charge per query. Listed explicitly (price 0.0) for the same reason
# the local STT/TTS providers are: a missing rate should mean "nobody priced
# this", never "this one is free".
#
# Chroma Cloud is a different product with usage pricing -- $0.0075 per TiB
# queried, $2.50/GiB written, $0.33/GiB-month stored, $0.09/GiB returned
# (https://www.trychroma.com/pricing, checked 2026-09). None of those units is
# a "query", and this adapter cannot reach Cloud anyway, so no Cloud rate is
# guessed here.
# ---------------------------------------------------------------------------
CHROMA_RATES: list[Rate] = [
    Rate(
        component=CostComponent.OTHER,
        provider="chroma_local",
        model=None,
        unit_name="knowledge_queries",
        price_per_unit=0.0,
        currency="USD",
        as_of="2026-09",
    )
]


def _column(result: dict[str, Any], key: str) -> list[Any]:
    """Pull one column of the first (and only) query out of a Chroma result.

    Chroma answers in column-major form, nested one level per query text:
    ``result["documents"][0]`` is the document list for ``query_texts[0]``.
    A column the caller did not ``include`` comes back as ``None`` rather
    than missing, so both shapes collapse to an empty list here.
    """
    column = result.get(key)
    if not column:
        return []
    first = column[0]
    return list(first) if first else []


def _passages(result: dict[str, Any], collection: str) -> list[Passage]:
    """Map one Chroma result into passages, best first.

    Chroma ranks by *distance* (smaller is closer) on whatever space the
    collection was configured with. ``1 / (1 + distance)`` turns that into a
    higher-is-better number without claiming to be a similarity: it is
    strictly rank-preserving for any non-negative distance, which is the only
    property :class:`~tring.knowledge.base.Passage` promises. The raw
    distance is kept in ``metadata["distance"]`` so nothing is lost for a
    consumer that knows which space it configured.
    """
    documents = _column(result, "documents")
    metadatas = _column(result, "metadatas")
    distances = _column(result, "distances")
    ids = _column(result, "ids")

    passages: list[Passage] = []
    for index, text in enumerate(documents):
        distance = float(distances[index]) if index < len(distances) else 0.0
        metadata = dict(metadatas[index] or {}) if index < len(metadatas) else {}
        identifier = str(ids[index]) if index < len(ids) else str(index)
        metadata["distance"] = distance
        passages.append(
            Passage(
                text=str(text),
                score=1.0 / (1.0 + distance),
                source=f"{collection}#{identifier}",
                metadata=metadata,
            )
        )
    return passages


class ChromaKnowledge(KnowledgeProvider):
    """Read one persistent Chroma collection.

    The client and collection handle are opened on first search and reused
    for the rest of the process: opening a PersistentClient loads the
    embedding model, which is far too expensive to repeat per turn. The lock
    exists because the first search of a call and a speculative prefetch of
    the same turn genuinely race, and opening the store twice would double
    that one-time cost at the worst possible moment.
    """

    name = "chroma_local"

    def __init__(self, path: str, collection: str) -> None:
        self.path = path
        self.collection = collection
        self._handle: Any = None
        self._lock = asyncio.Lock()

    async def search(self, query: str, k: int = 4) -> list[Passage]:
        if k <= 0 or not query.strip():
            return []
        handle = await self._collection()
        result = await asyncio.to_thread(
            handle.query,
            query_texts=[query],
            n_results=k,
            include=list(_INCLUDE),
        )
        return _passages(dict(result), self.collection)

    async def _collection(self) -> Any:
        async with self._lock:
            if self._handle is None:
                self._handle = await asyncio.to_thread(self._open)
            return self._handle

    def _open(self) -> Any:
        """Open the on-disk store. Runs in a worker thread, never on the loop."""
        try:
            import chromadb
        except ImportError as exc:  # lazy: the core install has no vector store
            raise ImportError(_IMPORT_HINT) from exc

        client = chromadb.PersistentClient(path=self.path)
        # get_collection, not get_or_create: silently creating an empty
        # collection because of a typo would make every search return nothing
        # and look like a knowledge base with no answers, rather than a
        # misconfiguration.
        return client.get_collection(name=self.collection)


@register("knowledge", "chroma_local")
def _make_chroma(**options: Any) -> ChromaKnowledge:
    collection = options.get("collection")
    if not collection:
        raise ValueError(
            "chroma_local needs knowledge.options.collection (the name of an "
            "existing Chroma collection); this provider reads, it never creates"
        )
    return ChromaKnowledge(
        path=str(options.get("path", "./chroma")),
        collection=str(collection),
    )


__all__ = ["CHROMA_RATES", "ChromaKnowledge"]
