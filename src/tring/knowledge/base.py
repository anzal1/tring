"""Knowledge provider interface: the retrieval slot of an agent.

A knowledge provider answers one question -- "which passages of the
knowledge base are most likely to answer this?" -- and nothing else. It does
not summarize, it does not call a model, and it does not decide what the bot
says. That separation is what lets the same corpus serve a cascade runtime,
an S2S runtime and an evaluation harness without any of them knowing which
vector store (or none at all) sits underneath.

Why passages and not documents
------------------------------

The consumer of a search result here is a *voice* turn: the passages are
pasted into a tool result that the model must read, reason over and answer
from inside the caller's patience budget. A whole document costs prompt
tokens, costs latency, and buries the one sentence that mattered. Providers
therefore return the smallest self-contained unit they can justify (a
paragraph, a chunk, a row), each carrying the ``source`` a human would need
to audit the answer afterwards.

Scores are comparable *within one provider's result list* and nowhere else.
A BM25 score, a cosine distance folded into ``1/(1+d)`` and a Qdrant dot
product are three different scales; ranking is the only thing the contract
promises, so nothing downstream may threshold on an absolute value.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

#: How many decimal places of ``score`` survive into a tool result. The model
#: only ever uses the score to see *which* passage ranked first; the trailing
#: digits of a float are prompt tokens spent on nothing.
_SCORE_DIGITS = 4


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk of the knowledge base.

    ``source`` is a human-auditable pointer back to where the text came from
    (``"faq.md#2"``, ``"policies#41f0"``). It is not decoration: an answer a
    caller disputes has to be traceable to a line someone can go read, and
    the model is told to name it when the answer matters.

    ``metadata`` is whatever the store had -- provider-specific by design, so
    nothing here has to pretend Chroma's metadata and Qdrant's payload are
    the same shape.
    """

    text: str
    score: float
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Render for a tool result: JSON-safe, and no wider than it needs to be."""
        return {
            "text": self.text,
            "source": self.source,
            "score": round(self.score, _SCORE_DIGITS),
            "metadata": self.metadata,
        }


class KnowledgeProvider(abc.ABC):
    """One knowledge base, adapted to one method.

    Implementations register under registry kind ``"knowledge"`` and are
    selected by ``AgentSpec.knowledge.provider``, exactly like the STT / LLM
    / TTS slots -- so swapping a local keyword index for a hosted vector
    store is a config change, never a rewrite.

    Two rules, inherited from the provider ABCs next door:

    1. **Lazy imports.** Importing the module must stay cheap and must never
       fail; vendor SDKs are imported inside the method that needs them and a
       missing one is re-raised as an ``ImportError`` naming the pip extra.
    2. **No blocking the event loop.** A voice call has audio in flight the
       whole time ``search`` runs. A synchronous vendor client belongs in
       ``asyncio.to_thread``, not on the loop.
    """

    #: Registry name, used as the provider label on cost lines.
    name: str = "knowledge"

    @abc.abstractmethod
    async def search(self, query: str, k: int = 4) -> list[Passage]:
        """Return up to ``k`` passages, best first.

        Returning fewer than ``k`` (including none at all) is a normal
        result, not an error: "the knowledge base has nothing on this" is
        exactly the answer that stops a model from inventing one.
        """

    async def aclose(self) -> None:
        """Release any client or connection the provider opened.

        Default is a no-op, which is correct for in-memory and embedded
        stores; network-backed providers override it so a finished call does
        not leak a socket.
        """
        return None


__all__ = ["KnowledgeProvider", "Passage"]
