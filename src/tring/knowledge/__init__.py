"""The knowledge slot: retrieval as a provider, wired in as a tool.

An agent gets a knowledge base by declaring one in its spec::

    knowledge:
      provider: keyword_local
      options:
        path: ./kb          # a folder of .txt / .md files
      top_k: 4
      speculative: true

and binding it to the session before the runtime is built::

    from tring.knowledge import KnowledgeBinding, ToolHandler
    from tring.runtimes.cascade import CascadeRuntime

    handlers: dict[str, ToolHandler] = {}
    binding = KnowledgeBinding(session, spec.knowledge, handlers)
    runtime = CascadeRuntime(session, handlers=handlers)
    ...
    await binding.aclose()

Those three lines add a ``search_knowledge`` tool to the agent, put its
handler in ``handlers``, and -- when ``speculative`` is on -- start searching
from partial transcripts so the answer is usually ready before the model
asks for it. Order matters: the runtime reads the agent's tool list when it
is constructed, so the binding comes first. ``spec.knowledge`` may be
``None``; the binding is then inert and the same three lines still work.

Providers ship here: ``keyword_local`` (zero-dependency BM25, the one that
needs no install and no service), ``chroma_local`` and ``qdrant`` (lazy
imports, ``pip install 'tring[knowledge]'``). Importing this package is what
registers all three under registry kind ``"knowledge"``; it stays cheap,
because every vendor SDK is imported inside the method that needs it.

See :mod:`tring.knowledge.binding` for why speculative retrieval is built as
an event consumer instead of an edit to the cascade runtime.
"""

from __future__ import annotations

from tring.knowledge.base import KnowledgeProvider, Passage
from tring.knowledge.binding import SEARCH_KNOWLEDGE, KnowledgeBinding, ToolHandler
from tring.knowledge.chroma import CHROMA_RATES, ChromaKnowledge
from tring.knowledge.keyword_local import (
    KEYWORD_LOCAL_RATES,
    Document,
    KeywordLocalKnowledge,
)
from tring.knowledge.qdrant import QDRANT_RATES, QdrantKnowledge

__all__ = [
    "CHROMA_RATES",
    "KEYWORD_LOCAL_RATES",
    "QDRANT_RATES",
    "SEARCH_KNOWLEDGE",
    "ChromaKnowledge",
    "Document",
    "KeywordLocalKnowledge",
    "KnowledgeBinding",
    "KnowledgeProvider",
    "Passage",
    "QdrantKnowledge",
    "ToolHandler",
]
