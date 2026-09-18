"""Tests for the language_lock primitive.

Core property under test: re-asserting the language directive every turn
must never rewrite or precede previously sent messages, only append after
them — otherwise a provider's prefix-keyed prompt cache goes cold every turn.
"""

from __future__ import annotations

import copy

import pytest

from trunkline.agent import LanguagePolicy
from trunkline.primitives.language_lock import LanguageLock, prefix_stable


def make_lock(**kwargs: object) -> LanguageLock:
    return LanguageLock(LanguagePolicy(**kwargs))


def test_apply_appends_exactly_one_trailing_directive() -> None:
    lock = make_lock(primary="hi", allowed=["en"])
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "namaste"},
    ]
    result = lock.apply(messages)

    assert len(result) == len(messages) + 1
    assert result[:-1] == messages
    trailing = result[-1]
    assert trailing["role"] == "system"
    assert "hi" in trailing["content"]
    assert "LANGUAGE DIRECTIVE" in trailing["content"]


def test_apply_does_not_mutate_input() -> None:
    lock = make_lock(primary="mr", allowed=[])
    messages = [{"role": "user", "content": "hello"}]
    snapshot = copy.deepcopy(messages)

    result = lock.apply(messages)

    assert messages == snapshot
    assert result is not messages


def test_prefix_stability_across_three_growing_turns() -> None:
    """Simulate 3 turns of growing history; each turn's applied array must
    keep the previous turn's non-directive messages byte-prefix-stable."""
    lock = make_lock(primary="hi", allowed=["en"])

    history: list[dict] = [{"role": "system", "content": "Persona text."}]
    applied_turns = []

    # Turn 1
    history.append({"role": "user", "content": "hello"})
    applied_turns.append(lock.apply(history))

    # Turn 2: history grows by append only (bot reply + new user turn)
    history.append({"role": "assistant", "content": "namaste, kaise madad karoon?"})
    history.append({"role": "user", "content": "mujhe cricket score chahiye"})
    applied_turns.append(lock.apply(history))

    # Turn 3
    history.append({"role": "assistant", "content": "abhi score dekh rahe hain."})
    history.append({"role": "user", "content": "thank you"})
    applied_turns.append(lock.apply(history))

    assert prefix_stable(applied_turns[0], applied_turns[1])
    assert prefix_stable(applied_turns[1], applied_turns[2])
    # transitive: turn 1 must also be a prefix-stable ancestor of turn 3
    assert prefix_stable(applied_turns[0], applied_turns[2])


def test_prefix_stability_detects_a_broken_cache() -> None:
    """If an earlier message is rewritten (the naive/broken approach),
    prefix_stable must report False."""
    lock = make_lock(primary="hi", allowed=[])
    history = [{"role": "user", "content": "hello"}]
    turn1 = lock.apply(history)

    # Naive fix: mutate an earlier message instead of only appending.
    broken_history = [{"role": "user", "content": "HELLO (edited)"}]
    turn2 = lock.apply(broken_history)

    assert not prefix_stable(turn1, turn2)


def test_language_override_changes_only_trailing_message() -> None:
    lock = make_lock(primary="hi", allowed=["en", "mr"])
    messages = [{"role": "user", "content": "switching languages now"}]

    default_result = lock.apply(messages)
    override_result = lock.apply(messages, active_language="en")

    assert default_result[:-1] == override_result[:-1] == messages
    assert default_result[-1] != override_result[-1]
    assert "hi" in default_result[-1]["content"]
    assert "en" in override_result[-1]["content"]


def test_override_rejects_language_outside_policy() -> None:
    lock = make_lock(primary="hi", allowed=["en"])
    with pytest.raises(ValueError):
        lock.directive("fr")


def test_lock_false_passthrough() -> None:
    lock = make_lock(primary="hi", allowed=["en"], lock=False)
    messages = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi there"},
    ]

    result = lock.apply(messages)

    assert result == messages
    assert result is not messages  # still a new list, input never mutated


def test_role_parameter_controls_trailing_message_role() -> None:
    lock = make_lock(primary="hi", allowed=[])
    messages = [{"role": "user", "content": "hi"}]

    result = lock.apply(messages, role="user")

    assert result[-1]["role"] == "user"


def test_directive_lists_allowed_languages() -> None:
    lock = make_lock(primary="hi", allowed=["en", "mr"])
    text = lock.directive()
    assert "hi" in text
    assert "en" in text
    assert "mr" in text
