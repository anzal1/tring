"""Tests for the streaming speak/tool_call envelope parser.

The parser's whole reason to exist is behavior under hostile chunking, so most
tests feed one character at a time: if a rule works at that granularity it
works at every larger one.
"""

from __future__ import annotations

import json

import pytest

from trunkline.primitives.speak_parser import (
    FallbackText,
    ParserEvent,
    SpeakDelta,
    SpeakToolParser,
    ToolCallReady,
    parse_all,
)


def spoken(events: list[ParserEvent]) -> str:
    return "".join(e.text for e in events if isinstance(e, SpeakDelta))


def fallback(events: list[ParserEvent]) -> str:
    return "".join(e.text for e in events if isinstance(e, FallbackText))


def tool_calls(events: list[ParserEvent]) -> list[dict[str, object]]:
    return [e.call for e in events if isinstance(e, ToolCallReady)]


def by_char(text: str) -> list[str]:
    return list(text)


# --------------------------------------------------------------- happy path


def test_full_envelope_one_char_at_a_time() -> None:
    payload = json.dumps(
        {"speak": "Let me check that for you.", "tool_call": {"name": "lookup"}}
    )
    events = parse_all(by_char(payload))

    assert spoken(events) == "Let me check that for you."
    assert tool_calls(events) == [{"name": "lookup"}]
    assert fallback(events) == ""


def test_speak_flushes_before_tool_call_is_generated() -> None:
    """The latency contract: audio starts while the tool call is still coming."""
    parser = SpeakToolParser()
    head = '{"speak": "One moment.", "tool_call": {"name": "lookup", "args": {'
    seen: list[ParserEvent] = []
    for char in by_char(head):
        seen.extend(parser.feed(char))

    # Every speak character is already out, though the tool call is unfinished.
    assert spoken(seen) == "One moment."
    assert tool_calls(seen) == []
    # And the last speak delta landed well before the end of the head.
    last_speak = max(i for i, e in enumerate(seen) if isinstance(e, SpeakDelta))
    assert last_speak == len(seen) - 1  # nothing else has been emitted yet


def test_one_delta_per_feed_call_not_per_character() -> None:
    parser = SpeakToolParser()
    events = parser.feed('{"speak": "hello there"')
    assert events == [SpeakDelta("hello there")]


def test_tool_call_ready_arrives_before_root_object_closes() -> None:
    parser = SpeakToolParser()
    seen: list[ParserEvent] = []
    for char in by_char('{"speak": "ok", "tool_call": {"name": "x"}}'):
        seen.extend(parser.feed(char))
        if tool_calls(seen):
            break
    # The event fired on the tool object's own closing brace, so one character
    # of the stream (the root '}') is still unread.
    assert tool_calls(seen) == [{"name": "x"}]


def test_tolerates_surrounding_whitespace_and_newlines() -> None:
    events = parse_all(["\n\n  ", '{ "speak" : "hi" , ', '"tool_call" : null }', "\n"])
    assert spoken(events) == "hi"
    assert tool_calls(events) == []
    assert fallback(events) == ""


# ------------------------------------------------------------------ escapes


def test_escapes_split_across_chunk_boundaries() -> None:
    # Every escape is deliberately torn in half by a chunk boundary.
    chunks = [
        '{"speak": "line',
        "\\",  # dangling backslash: the escape type is not known yet
        "n\\",  # completes \n, opens the next escape
        '"quoted\\',  # completes \" (a literal quote), opens the next
        '" back\\',  # completes \" again
        "\\slash \\",  # completes \\ (a literal backslash), opens \u....
        "u09",  # ... torn in the middle of its hex digits
        "39",
        '\\u0948"}',
    ]
    events = parse_all(chunks)
    assert spoken(events) == 'line\n"quoted" back\\slash है'


def test_unicode_escape_split_one_char_at_a_time() -> None:
    payload = '{"speak": "\\u0928\\u092e\\u0938\\u094d\\u0924\\u0947"}'
    events = parse_all(by_char(payload))
    assert spoken(events) == "नमस्ते"


def test_literal_devanagari_and_multibyte_text() -> None:
    payload = json.dumps(
        {"speak": "नमस्ते, आपकी कॉल के लिए धन्यवाद 🙏", "tool_call": None},
        ensure_ascii=False,
    )
    events = parse_all(by_char(payload))
    assert spoken(events) == "नमस्ते, आपकी कॉल के लिए धन्यवाद 🙏"
    assert tool_calls(events) == []


def test_surrogate_pair_escape_becomes_one_character() -> None:
    # ensure_ascii=True encodes non-BMP characters as a UTF-16 surrogate pair.
    payload = json.dumps({"speak": "thanks 🙏"})
    assert "\\ud83d" in payload
    events = parse_all(by_char(payload))
    assert spoken(events) == "thanks 🙏"


def test_lone_high_surrogate_does_not_break_the_stream() -> None:
    events = parse_all(by_char('{"speak": "a\\ud83db"}'))
    assert spoken(events) == "a�b"
    # Whatever we emit must be encodable, or TTS downstream would explode.
    spoken(events).encode("utf-8")


def test_escaped_quote_inside_tool_call_does_not_end_the_object() -> None:
    call = {"name": "note", "args": {"text": 'she said "hi" }'}}
    events = parse_all(by_char(json.dumps({"speak": "ok", "tool_call": call})))
    assert tool_calls(events) == [call]


# --------------------------------------------------------------- tool_calls


def test_tool_call_null_yields_no_tool_event() -> None:
    events = parse_all(by_char('{"speak": "All set, goodbye.", "tool_call": null}'))
    assert spoken(events) == "All set, goodbye."
    assert tool_calls(events) == []
    assert fallback(events) == ""


def test_tool_call_with_nested_objects_and_arrays() -> None:
    call = {
        "name": "book_appointment",
        "arguments": {
            "slots": [{"day": "mon", "times": [9, 10.5]}, {"day": "tue", "times": []}],
            "contact": {"phone": "+91-000", "prefs": {"sms": True, "email": None}},
            "notes": ["नमस्ते", "second"],
        },
        "waiting_message": "Checking the calendar now.",
    }
    payload = json.dumps({"speak": "One second.", "tool_call": call})
    events = parse_all(by_char(payload))
    assert spoken(events) == "One second."
    assert tool_calls(events) == [call]


def test_tool_call_first_still_parses_and_speak_arrives_later() -> None:
    """Key order is not guaranteed by the model, only by the prompt."""
    payload = '{"tool_call": {"name": "lookup", "arguments": {"id": 7}}, "speak": "Sure."}'
    parser = SpeakToolParser()
    seen: list[ParserEvent] = []
    for char in by_char(payload):
        seen.extend(parser.feed(char))
    seen.extend(parser.finalize())

    assert spoken(seen) == "Sure."
    assert tool_calls(seen) == [{"name": "lookup", "arguments": {"id": 7}}]
    # Ordering is preserved: the tool call landed before the speech here.
    kinds = [type(e).__name__ for e in seen]
    assert kinds.index("ToolCallReady") < kinds.index("SpeakDelta")


def test_tool_call_closing_at_the_very_end_is_flushed_by_finalize() -> None:
    parser = SpeakToolParser()
    seen: list[ParserEvent] = []
    # No root '}' at all: the stream was cut after the tool object closed.
    for char in by_char('{"speak": "hi", "tool_call": {"name": "x"}'):
        seen.extend(parser.feed(char))
    seen.extend(parser.finalize())
    assert spoken(seen) == "hi"
    assert tool_calls(seen) == [{"name": "x"}]


def test_non_object_tool_call_value_is_ignored() -> None:
    events = parse_all(by_char('{"speak": "hi", "tool_call": "lookup"}'))
    assert spoken(events) == "hi"
    assert tool_calls(events) == []


def test_unknown_keys_are_skipped_without_breaking_speak() -> None:
    payload = '{"thinking": [1, 2], "speak": "Right.", "confidence": 0.93}'
    events = parse_all(by_char(payload))
    assert spoken(events) == "Right."
    assert fallback(events) == ""


# ----------------------------------------------------------------- fences


def test_fenced_json_envelope() -> None:
    payload = '```json\n{"speak": "Fenced but fine.", "tool_call": null}\n```'
    events = parse_all(by_char(payload))
    assert spoken(events) == "Fenced but fine."
    assert fallback(events) == ""


def test_bare_fence_and_same_line_object() -> None:
    events = parse_all(["``", '`json {"speak": "ok"}', "\n```"])
    assert spoken(events) == "ok"
    assert fallback(events) == ""


# ----------------------------------------------------------------- fallback


def test_plain_prose_falls_back_to_text() -> None:
    chunks = ["Sorry, ", "I could not ", "find that booking."]
    events = parse_all(chunks)
    assert fallback(events) == "Sorry, I could not find that booking."
    assert spoken(events) == ""
    assert tool_calls(events) == []


def test_fallback_fires_on_the_first_non_envelope_character() -> None:
    parser = SpeakToolParser()
    assert parser.feed("S") == [FallbackText("S")]
    assert parser.feed("orry") == [FallbackText("orry")]
    assert parser.finalize() == []


def test_fallback_flushes_the_withheld_fence_prefix() -> None:
    # A single backtick is withheld while we wait to see if it is a fence.
    parser = SpeakToolParser()
    assert parser.feed("`") == []
    events = parser.feed("code-ish reply")
    assert fallback(events) == "`code-ish reply"


def test_fallback_keeps_the_unconsumed_tail_of_the_breaking_chunk() -> None:
    events = parse_all(["   Hello there, how can I help?"])
    assert fallback(events) == "   Hello there, how can I help?"


def test_brace_prefixed_prose_is_spoken_rather_than_dropped() -> None:
    events = parse_all(by_char("{not really json at all}"))
    assert fallback(events) == "{not really json at all}"


def test_empty_stream_produces_nothing() -> None:
    parser = SpeakToolParser()
    assert parser.feed("") == []
    assert parser.finalize() == []
    assert parse_all([]) == []
    assert parse_all(["", ""]) == []


def test_whitespace_only_stream_produces_nothing() -> None:
    assert parse_all(by_char("  \n\t ")) == []


# ----------------------------------------------------- malformed / garbage


def test_malformed_json_after_valid_speak_keeps_the_speech_and_stays_quiet() -> None:
    parser = SpeakToolParser()
    seen: list[ParserEvent] = []
    for char in by_char('{"speak": "I can help with that." ??? garbage'):
        seen.extend(parser.feed(char))
    seen.extend(parser.finalize())

    assert spoken(seen) == "I can help with that."
    # Already-spoken words are never replayed as fallback text.
    assert fallback(seen) == ""
    assert tool_calls(seen) == []


def test_truncated_tool_call_json_yields_no_tool_event() -> None:
    events = parse_all(by_char('{"speak": "one sec", "tool_call": {"name": "x"'))
    assert spoken(events) == "one sec"
    assert tool_calls(events) == []


def test_truncated_envelope_is_not_read_aloud_as_json() -> None:
    events = parse_all(by_char('{"spea'))
    assert fallback(events) == ""
    assert spoken(events) == ""


def test_unterminated_speak_string_still_flushed_incrementally() -> None:
    events = parse_all(by_char('{"speak": "half a sentence'))
    assert spoken(events) == "half a sentence"


@pytest.mark.parametrize(
    "garbage",
    [
        "{",
        "}",
        "{}",
        '{"speak"}',
        '{"speak":}',
        '{"speak": "a" "b"}',
        '{"tool_call": {{{{',
        '{"speak": "x", }',
        "[1, 2, 3]",
        '{"speak": "x"}{"speak": "y"}',
        "\\u",
        '{"speak": "\\uZZZZ"}',
        '{"speak": "\\q"}',
        "```",
        "```json",
        "```json\n",
        "```json\nnot json\n```",
        "null",
        '{"speak": 12}',
        '{"speak": tru',
    ],
)
def test_garbage_never_raises(garbage: str) -> None:
    for chunks in (by_char(garbage), [garbage]):
        parser = SpeakToolParser()
        for chunk in chunks:
            parser.feed(chunk)
        parser.finalize()


def test_invalid_hex_escape_is_passed_through_verbatim() -> None:
    events = parse_all(by_char('{"speak": "a\\uZZZZb"}'))
    assert spoken(events) == "a\\uZZZZb"


def test_unknown_escape_keeps_its_character() -> None:
    events = parse_all(by_char('{"speak": "a\\qb"}'))
    assert spoken(events) == "aqb"


def test_finalize_is_idempotent_and_feed_after_finalize_is_inert() -> None:
    parser = SpeakToolParser()
    parser.feed('{"speak": "done"}')
    assert parser.finalize() == []
    assert parser.finalize() == []
    assert parser.feed("trailing") == []


def test_trailing_text_after_the_envelope_is_ignored() -> None:
    events = parse_all(by_char('{"speak": "bye"}\n```\nignore me'))
    assert spoken(events) == "bye"
    assert fallback(events) == ""


def test_parser_instances_do_not_share_state() -> None:
    first = SpeakToolParser()
    second = SpeakToolParser()
    first.feed('{"speak": "one')
    assert spoken(second.feed('{"speak": "two"}')) == "two"
    assert spoken(first.feed(' more"}')) == " more"
