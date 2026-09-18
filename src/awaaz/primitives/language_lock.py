"""Language lock: re-assert the language directive every turn, cache-safely.

Multilingual voice models drift to English mid-conversation unless the
language directive is repeated. The naive fix — re-injecting a fresh system
message at the front of the array, or rewriting an existing one in place —
silently destroys provider prompt caches: those caches key on a byte-identical
*prefix* of the message array, and any edit to an already-sent message (or
insertion before it) invalidates everything after it. That turns a latency
optimization into a latency regression on every single turn.

The fix here is structural, not clever: never touch a previously sent
message. Every call to :meth:`LanguageLock.apply` returns the input messages
untouched, plus exactly one *new* trailing message carrying the directive.
Because appends never change the bytes of what came before, a provider cache
keyed on message-array prefixes stays warm turn over turn — the array only
ever grows at the end.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from awaaz.agent import LanguagePolicy

# Template for the enforcement text. Kept as a module-level constant (not a
# method-local f-string) so tooling and tests can inspect/patch it, and so the
# wording used in production is the same wording covered by tests.
DIRECTIVE_TEMPLATE = (
    "LANGUAGE DIRECTIVE: Respond only in {language}. This applies to this "
    "reply and every reply that follows, regardless of what language the "
    "caller used. Do not switch to any other language unless the caller "
    "explicitly asks you to, and even then only switch to one of: {allowed}."
)


class LanguageLock:
    """Re-asserts an agent's language policy every turn without cache breaks.

    Why a class and not a function: the enforcement text depends on
    ``policy`` (primary + allowed languages), which is fixed for the life of
    an agent/session, so it is bound once at construction rather than
    threaded through every call site.
    """

    def __init__(self, policy: LanguagePolicy) -> None:
        self.policy = policy

    def directive(self, language: str | None = None) -> str:
        """Render the enforcement text.

        Args:
            language: override the language to enforce for this turn, used
                when the runtime detects the caller has switched to another
                *allowed* language (e.g. code-switching between two of the
                agent's supported languages). Must be ``policy.primary`` or
                a member of ``policy.allowed``; anything else is rejected so
                a mis-detected language can't smuggle an unsupported
                directive into the prompt. Defaults to ``policy.primary``.
        """
        target = language or self.policy.primary
        permitted = {self.policy.primary, *self.policy.allowed}
        if target not in permitted:
            raise ValueError(
                f"language {target!r} is not in the agent's policy "
                f"(primary={self.policy.primary!r}, allowed={self.policy.allowed!r})"
            )
        allowed_str = ", ".join(permitted) if permitted else target
        return DIRECTIVE_TEMPLATE.format(language=target, allowed=allowed_str)

    def apply(
        self,
        messages: list[dict[str, Any]],
        active_language: str | None = None,
        role: Literal["system", "user"] = "system",
    ) -> list[dict[str, Any]]:
        """Return ``messages`` plus exactly one trailing directive message.

        The input list is never mutated and no message is ever inserted
        before existing content — only appended after. When
        ``policy.lock`` is False the messages are returned unchanged (still
        a new list, matching the "never mutate input" contract, but with no
        directive appended).

        Args:
            messages: the conversation so far, provider wire format
                (``{"role": ..., "content": ...}`` dicts).
            active_language: the language detected as active this turn; see
                :meth:`directive`.
            role: which role carries the directive. Some providers treat a
                trailing ``system`` message as a hint about the *next*
                completion; others require it framed as ``user``. Callers
                pick per-provider; the default is ``system``.
        """
        if not self.policy.lock:
            return list(messages)
        text = self.directive(active_language)
        return [*messages, {"role": role, "content": text}]


def prefix_stable(
    prev_applied: list[dict[str, Any]], next_applied: list[dict[str, Any]]
) -> bool:
    """Assert cache-stability between two consecutive turns' applied arrays.

    ``prev_applied`` and ``next_applied`` are both outputs of
    :meth:`LanguageLock.apply` from consecutive turns, where the underlying
    conversation history only grew by appends between them (the normal case:
    a user turn and a bot turn were added, then the lock re-applied). This
    holds precisely when the JSON serialization of ``prev_applied`` with its
    trailing directive dropped is a byte-for-byte *prefix* of the
    serialization of ``next_applied``.

    This is exactly the property a provider-side prompt cache depends on:
    it can only reuse the KV cache for the longest previously-seen prefix
    of the request bytes. If dropping the old trailing directive is not a
    prefix of the new array's bytes, some earlier message's bytes moved,
    and the cache would have been invalidated.

    Serialization uses compact, deterministic ``json.dumps`` (no key
    reordering, no separator whitespace drift) so "byte prefix" is
    meaningful rather than incidental to formatting.
    """
    prev_bytes = _dumps(prev_applied[:-1])
    next_bytes = _dumps(next_applied)
    return next_bytes.startswith(prev_bytes)


def _dumps(messages: list[dict[str, Any]]) -> str:
    """Serialize messages as a concatenation of independently-encoded objects.

    A JSON *array* is unsuitable for prefix comparison: ``[a,b]`` is not a
    string-prefix of ``[a,b,c]`` because the shorter array's closing ``]``
    sits where the longer one has a ``,``. Encoding each message on its own
    and concatenating (newline-separated) makes "the first N messages are
    unchanged" the same thing as "the string is a prefix" — which is the
    actual property a provider's byte-prefix prompt cache relies on.
    """
    return "\n".join(json.dumps(m, sort_keys=False, separators=(",", ":")) for m in messages)
