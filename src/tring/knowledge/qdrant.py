"""``qdrant`` -- knowledge from a Qdrant collection, local or hosted.

The step up from an embedded store: a real vector database, the same one
whether it runs in Docker next to the agent or as a managed cluster. Worth
its operational weight once the corpus is large enough that recall is a
tuning problem (filters, named vectors, quantization) rather than a lookup.

Like the Chroma adapter this one is **read-only** -- indexing belongs to an
offline job, not to a live call -- and it never sees a raw API key: options
carry the *name* of the environment variable holding it.

API verified 2026-09 against:

* ``AsyncQdrantClient(url=..., api_key=..., cloud_inference=...)`` and
  ``async query_points(collection_name, query, using, limit, with_payload)
  -> types.QueryResponse``:
  https://github.com/qdrant/qdrant-client/blob/master/qdrant_client/async_qdrant_client.py
* ``response.points``, each point carrying ``id`` / ``score`` / ``payload``,
  ordered by decreasing similarity:
  https://qdrant.tech/documentation/quickstart/
* Text queries via ``models.Document(text=..., model=...)``, embedded by the
  client's FastEmbed integration (``pip install "qdrant-client[fastembed]"``)
  or server-side when ``cloud_inference=True``:
  https://qdrant.tech/documentation/fastembed/fastembed-semantic-search/

``embedding_model`` has no default, on purpose. A query embedded by a
different model than the collection was built with does not fail -- it
returns confidently ranked nonsense, which on a phone call is a wrong answer
delivered in a friendly voice. Making it required turns that into a
configuration error nobody can miss.

Latency note. ``AsyncQdrantClient`` is natively async, so the network round
trip never blocks the loop. Local FastEmbed inference, however, runs
in-process inside ``query_points``: with a large embedding model that is real
CPU time on the event loop. Where that matters, set ``cloud_inference: true``
so the server embeds (Qdrant Cloud only -- the client rejects it against a
local instance).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from tring.cost.rates import Rate
from tring.knowledge.base import KnowledgeProvider, Passage
from tring.providers.registry import register

_IMPORT_HINT = (
    "qdrant requires the 'qdrant-client' package: pip install 'tring[knowledge]' "
    "(or pip install 'qdrant-client[fastembed]' to embed query text locally). "
    "The core install stays pydantic + pyyaml only."
)

# ---------------------------------------------------------------------------
# Rates: intentionally empty.
#
# Qdrant publishes no per-query or per-request price. Qdrant Cloud bills
# hourly for the resources a cluster consumes -- vCPU, memory, storage, backup
# storage, plus inference tokens for paid models -- and does not list a rate
# per search (https://qdrant.tech/pricing/, checked 2026-09); a self-hosted
# instance has no vendor charge at all but is not free either.
#
# Shipping a 0.0 rate here would be a lie in the Cloud case and a guess in the
# self-hosted one. An empty list is the honest answer: the cost meter then
# records the query at 0.0 with ``estimated=True`` and a missing-rate note,
# which is exactly the visible gap that design exists to produce. Price your
# own cluster and pin a Rate in your own RateCard.
# ---------------------------------------------------------------------------
QDRANT_RATES: list[Rate] = []


class QdrantKnowledge(KnowledgeProvider):
    """Query one Qdrant collection with text, embedded by the client.

    The client is built on first search and reused: it owns a connection
    pool, and with local inference it also owns a loaded embedding model.
    The lock guards the same race the Chroma adapter has -- a speculative
    prefetch and the tool handler can both reach an unopened client in the
    same turn.
    """

    name = "qdrant"

    def __init__(
        self,
        url: str,
        collection: str,
        embedding_model: str,
        api_key_env: str | None = None,
        text_field: str = "text",
        vector_name: str | None = None,
        cloud_inference: bool = False,
    ) -> None:
        self.url = url
        self.collection = collection
        self.embedding_model = embedding_model
        self.api_key_env = api_key_env  # env var NAME; never a raw key
        self.text_field = text_field
        self.vector_name = vector_name
        self.cloud_inference = cloud_inference
        self._client: Any = None
        self._models: Any = None
        self._lock = asyncio.Lock()

    async def search(self, query: str, k: int = 4) -> list[Passage]:
        if k <= 0 or not query.strip():
            return []
        client, models = await self._connect()
        response = await client.query_points(
            collection_name=self.collection,
            query=models.Document(text=query, model=self.embedding_model),
            using=self.vector_name,  # None = the collection's default vector
            limit=k,
            with_payload=True,  # the payload *is* the passage; default is True
        )
        return [self._passage(point) for point in response.points]

    def _passage(self, point: Any) -> Passage:
        """Map one scored point. The payload's text field is the passage.

        Qdrant scores are already higher-is-better and come back in
        decreasing order, so the ranking is passed through untouched. A point
        whose payload has no text field yields an empty passage rather than
        an exception: one badly indexed row should not take down the call,
        and an empty passage is visible in the event stream's result count.
        """
        payload = dict(point.payload or {})
        text = str(payload.pop(self.text_field, ""))
        return Passage(
            text=text,
            score=float(point.score),
            source=f"{self.collection}#{point.id}",
            metadata=payload,
        )

    async def aclose(self) -> None:
        """Close the connection so a finished call does not leak a socket."""
        client, self._client = self._client, None
        self._models = None
        if client is not None:
            await client.close()

    async def _connect(self) -> tuple[Any, Any]:
        async with self._lock:
            if self._client is None:
                self._client, self._models = self._build()
            return self._client, self._models

    def _build(self) -> tuple[Any, Any]:
        try:
            from qdrant_client import AsyncQdrantClient, models
        except ImportError as exc:  # lazy: the core install has no vector store
            raise ImportError(_IMPORT_HINT) from exc

        api_key: str | None = None
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(
                    f"qdrant requires environment variable {self.api_key_env!r} to "
                    "be set. Provider options carry the *name* of the variable "
                    "holding the key, never the key itself."
                )

        client = AsyncQdrantClient(
            url=self.url,
            api_key=api_key,
            cloud_inference=self.cloud_inference,
        )
        return client, models


@register("knowledge", "qdrant")
def _make_qdrant(**options: Any) -> QdrantKnowledge:
    collection = options.get("collection")
    if not collection:
        raise ValueError(
            "qdrant needs knowledge.options.collection (the name of an existing "
            "Qdrant collection); this provider reads, it never creates"
        )
    embedding_model = options.get("embedding_model")
    if not embedding_model:
        raise ValueError(
            "qdrant needs knowledge.options.embedding_model: the query must be "
            "embedded by the same model the collection was built with, and a "
            "mismatch returns confidently ranked nonsense rather than an error. "
            'Example: embedding_model: "sentence-transformers/all-MiniLM-L6-v2"'
        )
    api_key_env = options.get("api_key_env")
    vector_name = options.get("vector_name")
    return QdrantKnowledge(
        url=str(options.get("url", "http://localhost:6333")),
        collection=str(collection),
        embedding_model=str(embedding_model),
        api_key_env=str(api_key_env) if api_key_env else None,
        text_field=str(options.get("text_field", "text")),
        vector_name=str(vector_name) if vector_name else None,
        cloud_inference=bool(options.get("cloud_inference", False)),
    )


__all__ = ["QDRANT_RATES", "QdrantKnowledge"]
