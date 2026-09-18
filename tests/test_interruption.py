"""Tests for trunkline.primitives.interruption.PlaybackLedger.

Uses a bare CallSession bound to a minimal AgentSpec (no runtime, no
providers, no network) and reads back session.history to assert on emitted
events. No fakes beyond that are needed: the ledger's whole job is pure
text/state bookkeeping over the session event bus.
"""

from __future__ import annotations

from trunkline.agent import AgentSpec
from trunkline.events import BotSpeechPlayed, BotUtterance, Interruption
from trunkline.primitives.interruption import PlaybackLedger
from trunkline.session import CallSession


def make_session() -> CallSession:
    agent = AgentSpec(name="test-agent", persona="You are a helpful test agent.")
    # Deterministic clock: tests don't care about wall-clock timing, only order.
    tick = iter(range(10_000))
    return CallSession(agent=agent, clock=lambda: float(next(tick)))


def test_normal_turn_end_emits_no_annotation() -> None:
    session = make_session()
    ledger = PlaybackLedger(session)

    text = "Sure, I can help you with that today."
    ledger.utterance_started("u1", text)
    ledger.mark_played("u1", len(text))
    ledger.utterance_finished("u1")

    verdict = ledger.caller_started_speaking()

    assert verdict.genuine is False
    assert verdict.context_annotation is None
    assert verdict.heard_text is None
    assert verdict.unheard_text is None
    assert not any(isinstance(e, Interruption) for e in session.history)


def test_genuine_interrupt_splits_on_word_boundary() -> None:
    session = make_session()
    ledger = PlaybackLedger(session, grace_chars=5)

    text = "The weather tomorrow will be sunny with a light breeze in the afternoon."
    ledger.utterance_started("u1", text)
    # Play up to partway through the word "sunny" (not on a boundary).
    cut = text.index("sunny") + 2  # lands inside "sunny"
    ledger.mark_played("u1", cut)

    verdict = ledger.caller_started_speaking()

    assert verdict.genuine is True
    assert verdict.heard_text is not None and verdict.unheard_text is not None
    # The split must land on a word boundary: heard_text should not end with
    # a partial word, and heard + unheard must reconstitute the original text.
    assert verdict.heard_text + verdict.unheard_text == text
    assert verdict.heard_text == "The weather tomorrow will be "
    assert verdict.unheard_text == "sunny with a light breeze in the afternoon."
    assert not verdict.heard_text.endswith("sun")

    assert verdict.context_annotation is not None
    assert '"The weather tomorrow will be"' in verdict.context_annotation
    assert '"sunny with a light breeze in the afternoon."' in verdict.context_annotation
    assert "did NOT hear" in verdict.context_annotation

    interruptions = [e for e in session.history if isinstance(e, Interruption)]
    assert len(interruptions) == 1
    assert interruptions[0].heard_text == verdict.heard_text
    assert interruptions[0].unheard_text == verdict.unheard_text

    # Tracking stopped: a second call reports ordinary turn-taking.
    assert ledger.caller_started_speaking().genuine is False


def test_interrupt_split_exactly_on_boundary_is_unchanged() -> None:
    session = make_session()
    ledger = PlaybackLedger(session, grace_chars=2)

    text = "Please hold while I check your account balance right now."
    ledger.utterance_started("u1", text)
    cut = text.index("check")  # already a clean word boundary
    ledger.mark_played("u1", cut)

    verdict = ledger.caller_started_speaking()

    assert verdict.genuine is True
    assert verdict.heard_text == "Please hold while I "
    assert verdict.unheard_text == "check your account balance right now."


def test_heard_tail_truncated_to_last_twelve_words() -> None:
    session = make_session()
    ledger = PlaybackLedger(session, grace_chars=1)

    words = [f"word{i}" for i in range(1, 21)]  # 20 words
    text = " ".join(words) + " tail more content that stays unheard here."
    ledger.utterance_started("u1", text)
    # Play through all 20 words exactly (a clean boundary).
    cut = len(" ".join(words))
    ledger.mark_played("u1", cut)

    verdict = ledger.caller_started_speaking()

    assert verdict.genuine is True
    assert verdict.context_annotation is not None
    expected_tail = " ".join(words[-12:])
    assert f'"{expected_tail}"' in verdict.context_annotation
    # Words before the tail must not leak into the annotation.
    assert "word1 " not in verdict.context_annotation.split("did NOT hear")[0]


def test_multibyte_text_splits_correctly() -> None:
    session = make_session()
    ledger = PlaybackLedger(session, grace_chars=2)

    # Hindi (Devanagari) + an emoji: multi-codepoint / multibyte-in-UTF-8
    # characters that must not corrupt char-index-based splitting.
    text = "नमस्ते आप कैसे हैं आज 🎉 और कल भी अच्छा रहेगा दिन।"
    ledger.utterance_started("u1", text)
    cut = text.index("कैसे") + 2  # inside "कैसे", not a boundary
    ledger.mark_played("u1", cut)

    verdict = ledger.caller_started_speaking()

    assert verdict.genuine is True
    assert verdict.heard_text is not None and verdict.unheard_text is not None
    assert verdict.heard_text + verdict.unheard_text == text
    assert verdict.heard_text == "नमस्ते आप "
    assert verdict.unheard_text.startswith("कैसे")
    assert "🎉" in verdict.unheard_text


def test_grace_window_treated_as_natural_turn_taking() -> None:
    session = make_session()
    ledger = PlaybackLedger(session, grace_chars=5)

    text = "All set, have a great day!"
    ledger.utterance_started("u1", text)
    ledger.mark_played("u1", len(text) - 3)  # 3 chars left, within grace of 5

    verdict = ledger.caller_started_speaking()

    assert verdict.genuine is False
    assert verdict.context_annotation is None
    assert not any(isinstance(e, Interruption) for e in session.history)
    # Grace-window resolution still retires tracking: a follow-up call finds
    # nothing mid-playback rather than re-reporting the same utterance.
    assert ledger.caller_started_speaking().genuine is False


def test_no_utterance_in_flight_is_not_an_interrupt() -> None:
    session = make_session()
    ledger = PlaybackLedger(session)

    verdict = ledger.caller_started_speaking()

    assert verdict.genuine is False
    assert verdict.context_annotation is None
    assert session.history == []


def test_event_emission_order() -> None:
    session = make_session()
    ledger = PlaybackLedger(session, grace_chars=3)

    text = "Let me pull that record up for you right now please wait."
    ledger.utterance_started("u1", text)
    ledger.mark_played("u1", 8)
    ledger.mark_played("u1", 16)
    cut = text.index("right")
    ledger.mark_played("u1", cut)
    ledger.caller_started_speaking()

    kinds = [type(e) for e in session.history]
    assert kinds == [
        BotUtterance,
        BotSpeechPlayed,
        BotSpeechPlayed,
        BotSpeechPlayed,
        Interruption,
    ]

    played_events = [e for e in session.history if isinstance(e, BotSpeechPlayed)]
    # Each BotSpeechPlayed carries only the newly played span, and
    # concatenating them reconstructs everything played before the interrupt.
    assert played_events[0].text == text[0:8]
    assert played_events[1].text == text[8:16]
    assert played_events[2].text == text[16:cut]


def test_mark_played_is_monotonic_and_ignores_stale_reports() -> None:
    session = make_session()
    ledger = PlaybackLedger(session)

    text = "One two three four five six seven eight."
    ledger.utterance_started("u1", text)
    ledger.mark_played("u1", 10)
    ledger.mark_played("u1", 5)  # out-of-order / stale, must not regress or emit

    played_events = [e for e in session.history if isinstance(e, BotSpeechPlayed)]
    assert len(played_events) == 1
    assert played_events[0].text == text[0:10]

    # A report for an utterance_id that isn't current is ignored outright.
    ledger.mark_played("some-other-id", 100)
    played_events = [e for e in session.history if isinstance(e, BotSpeechPlayed)]
    assert len(played_events) == 1


def test_two_sequential_utterances_tracked_independently() -> None:
    session = make_session()
    ledger = PlaybackLedger(session, grace_chars=2)

    first = "Thanks for calling, how can I help you today?"
    ledger.utterance_started("u1", first)
    ledger.mark_played("u1", len(first))
    ledger.utterance_finished("u1")

    # First utterance ended cleanly: no interruption should surface for it.
    assert ledger.caller_started_speaking().genuine is False

    second = "Sure, checking that now, it looks like your order shipped yesterday."
    ledger.utterance_started("u2", second)
    cut = second.index("shipped") + 3  # inside "shipped"
    ledger.mark_played("u2", cut)

    verdict = ledger.caller_started_speaking()

    assert verdict.genuine is True
    assert verdict.utterance_id == "u2"
    assert verdict.heard_text == "Sure, checking that now, it looks like your order "
    assert verdict.unheard_text == "shipped yesterday."

    interruptions = [e for e in session.history if isinstance(e, Interruption)]
    assert len(interruptions) == 1  # only the second utterance produced one

    utterances = [e for e in session.history if isinstance(e, BotUtterance)]
    assert [u.text for u in utterances] == [first, second]
