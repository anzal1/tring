"""Character-level streaming parser for the canonical LLM envelope.

The envelope the prompt contract asks the model for is::

    {"speak": "text the caller hears", "tool_call": {...} | null}

The naive way to consume that is to buffer the whole completion and
``json.loads`` it. In a voice call that is a correctness bug, not a style
choice: the caller sits in silence for the entire generation, and the tool
arguments (the slowest part to generate, because they are the part the model
has to *think* about) are what keep them waiting. A 900 ms tool-call tail
becomes 900 ms of dead air before the first phoneme.

This parser instead walks the stream one character at a time and hands each
decoded character of the ``speak`` string to TTS the instant it arrives, while
the ``tool_call`` object is still being generated. First-token-to-audio
collapses to roughly the model's own first-token latency.

Three properties matter more than elegance here, and shape the code below:

1. **Chunk boundaries are arbitrary.** Providers split tokens wherever they
   like; ``\\u0939`` can arrive as six separate chunks. The parser is a
   resumable state machine (written as a generator coroutine fed one char at a
   time) so no rule may ever look ahead into "the rest of the buffer" -- there
   is no rest of the buffer.
2. **Models disobey.** Sooner or later a model answers with bare prose, or
   wraps the JSON in a ``` fence, or emits ``tool_call`` before ``speak``. The
   parser degrades instead of failing: unparseable input is re-emitted as
   :class:`FallbackText` so the call never goes silent, and a malformed tail
   after a valid ``speak`` is simply dropped.
3. **Speech already spoken cannot be unspoken.** Once a ``SpeakDelta`` is out,
   the parser is committed to envelope mode; it will never later re-emit the
   same words as fallback text. Saying something twice is worse than saying a
   truncated version once.

Events are frozen dataclasses rather than pydantic models, matching the
streaming chunk types in ``providers/base.py``: they are hot-path values
created per character-run, carry no untrusted input that needs validation, and
never cross a serialization boundary.
"""

from __future__ import annotations

import json
from collections.abc import Generator, Iterable
from dataclasses import dataclass
from typing import Any, TypeAlias

__all__ = [
    "FallbackText",
    "ParserEvent",
    "SpeakDelta",
    "SpeakToolParser",
    "ToolCallReady",
    "parse_all",
]


@dataclass(frozen=True)
class SpeakDelta:
    """Decoded text from the ``speak`` string, ready to send to TTS now.

    ``text`` is already unescaped: ``\\n`` is a newline, ``\\u0939`` is ``ह``.
    One event carries every speak character decoded during a single
    :meth:`SpeakToolParser.feed` call, so a consumer that awaits per event does
    not pay one await per character.
    """

    text: str


@dataclass(frozen=True)
class ToolCallReady:
    """The ``tool_call`` value parsed into a dict.

    Emitted the moment its closing brace arrives -- mid-stream if the model put
    ``tool_call`` first, at :meth:`SpeakToolParser.finalize` otherwise. Only
    JSON *objects* produce this event: ``null``, strings and numbers mean
    "no tool call" and are dropped.
    """

    call: dict[str, Any]


@dataclass(frozen=True)
class FallbackText:
    """Raw text emitted because the stream is not the expected envelope.

    The escape hatch for a model that ignored the output contract. Speaking
    slightly-wrong text beats a silent call, so this is a recovery path, not an
    error path.
    """

    text: str


ParserEvent: TypeAlias = SpeakDelta | ToolCallReady | FallbackText

# JSON's fixed two-character escapes. Anything else after a backslash is
# invalid JSON; we pass the character through verbatim rather than abort, since
# aborting would cost the caller their audio.
_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}

# Characters that can legally terminate a bare literal (number/true/false/null).
_LITERAL_ENDERS = ",}] \t\r\n"

_REPLACEMENT = "�"


class _NotEnvelope(Exception):
    """Internal signal: the stream cannot be the canonical envelope."""


class SpeakToolParser:
    """Incremental parser turning streamed model text into parser events.

    Usage mirrors any streaming codec::

        parser = SpeakToolParser()
        async for chunk in llm.generate(messages):
            for event in parser.feed(chunk.text):
                ...
        for event in parser.finalize():
            ...

    The instance is single-use and not thread-safe: one parser per model turn.
    """

    # Lifecycle states. Kept as plain strings for readable debugging output.
    _PARSING = "parsing"  # walking the envelope
    _FINISHED = "finished"  # root object closed; trailing text ignored
    _FALLBACK = "fallback"  # not an envelope; everything is speech
    _DEAD = "dead"  # broke after we already spoke; stay quiet

    def __init__(self) -> None:
        self._state = self._PARSING
        self._out: list[ParserEvent] = []
        self._raw: list[str] = []  # every char consumed, for fallback flushing
        self._spoke = False  # at least one SpeakDelta has escaped
        self._confirmed = False  # the root '{' was seen
        self._finalized = False

        # Raw text of the value currently being captured, so that finalize()
        # can still try to salvage a tool_call from a truncated stream.
        self._cap_key: str | None = None
        self._cap_buf: list[str] = []

        self._machine: Generator[None, str, None] | None = self._run()
        next(self._machine)  # advance to the first `yield`

    # ---------------------------------------------------------------- public

    def feed(self, chunk: str) -> list[ParserEvent]:
        """Consume one streamed chunk and return the events it produced.

        Returns an empty list when the chunk only advanced internal state (for
        example a chunk that is entirely inside the ``tool_call`` object).
        """
        if not chunk:
            return []
        if self._state == self._FALLBACK:
            self._raw.append(chunk)
            return [FallbackText(chunk)]
        if self._state in (self._FINISHED, self._DEAD):
            self._raw.append(chunk)
            return []

        self._out = []
        machine = self._machine
        assert machine is not None  # _PARSING implies a live machine
        for i, char in enumerate(chunk):
            self._raw.append(char)
            try:
                machine.send(char)
            except StopIteration:
                # Root object closed. Trailing text (closing ``` fence,
                # stray newline, a second envelope) is deliberately ignored.
                self._state = self._FINISHED
                self._machine = None
                break
            except Exception:  # garbage in must never raise out to the runtime
                self._machine = None
                self._break_out(rest=chunk[i + 1 :])
                break
        return _coalesce(self._out)

    def finalize(self) -> list[ParserEvent]:
        """Flush end-of-stream state. Never raises, whatever was fed.

        Handles the two things only the end of the stream can tell us: that a
        buffered ``tool_call`` is complete (when it was the last key, its
        closing brace may be the final character), and that a stream which
        never looked like an envelope should be spoken as-is.
        """
        if self._finalized:
            return []
        self._finalized = True
        self._out = []

        if self._state == self._FALLBACK:
            return []

        # A tool_call still in the capture buffer either ended exactly at the
        # stream boundary or was truncated; json.loads decides which.
        if self._cap_key == "tool_call" and self._cap_buf:
            self._try_emit_tool("".join(self._cap_buf))
        self._cap_key = None
        self._cap_buf = []

        # Never went silent rule: if we never even confirmed an opening brace
        # and nothing was spoken, whatever we did receive is the reply.
        if not self._confirmed and not self._spoke and self._state != self._DEAD:
            raw = "".join(self._raw)
            if raw.strip():
                self._out.append(FallbackText(raw))

        self._state = self._FINISHED
        self._machine = None
        return _coalesce(self._out)

    # --------------------------------------------------------------- helpers

    def _break_out(self, rest: str) -> None:
        """React to input that cannot be the envelope.

        If we have already spoken, the caller has heard part of this turn;
        replaying the raw JSON would duplicate it, so we go quiet instead and
        drop the malformed tail. If we have not spoken, everything received so
        far -- the withheld preamble, the consumed prefix, and the unconsumed
        tail of this chunk -- becomes speech, braces and all. Slightly wrong
        audio is recoverable for a caller; silence is not.
        """
        if self._spoke:
            self._state = self._DEAD
            return
        if rest:
            self._raw.append(rest)
        self._state = self._FALLBACK
        text = "".join(self._raw)
        if text:
            self._out.append(FallbackText(text))

    def _try_emit_tool(self, raw: str) -> None:
        """Emit ToolCallReady if ``raw`` is a complete JSON object."""
        try:
            value = json.loads(raw)
        except (ValueError, TypeError):
            return
        if isinstance(value, dict):
            self._out.append(ToolCallReady(value))

    def _emit_speak(self, text: str) -> None:
        if not text:
            return
        self._spoke = True
        self._out.append(SpeakDelta(text))

    # ------------------------------------------------------- the state machine
    #
    # Everything below is one coroutine: `char = yield` pulls the next
    # character, and `yield from` composes sub-parsers. Written this way the
    # parser reads top-to-bottom like a recursive descent parser while still
    # suspending at any byte boundary.

    def _run(self) -> Generator[None, str, None]:
        char = yield from self._preamble()
        if char != "{":
            raise _NotEnvelope
        self._confirmed = True
        yield from self._object()

    def _skip_ws(self, char: str | None = None) -> Generator[None, str, str]:
        """Return the next non-whitespace character."""
        while True:
            if char is None:
                char = yield
            if not char.isspace():
                return char
            char = None

    def _preamble(self) -> Generator[None, str, str]:
        """Skip leading whitespace and an optional markdown code fence.

        Models trained on chat habitually wrap JSON in ```json ... ```. The
        fence is withheld rather than emitted: if it turns out not to be a
        fence, :meth:`_break_out` flushes it back out as fallback text, so no
        character is ever lost.
        """
        char = yield from self._skip_ws()
        if char != "`":
            return char
        for _ in range(2):  # a fence is exactly three backticks
            if (yield) != "`":
                raise _NotEnvelope
        while True:  # optional info string ("json", "JSON", "") then newline
            char = yield
            if char == "{":  # ```json {"speak": ...} on one line
                return char
            if char == "\n":
                return (yield from self._skip_ws())
            if not (char.isalnum() or char in " \t-_+."):
                raise _NotEnvelope

    def _object(self) -> Generator[None, str, None]:
        """Parse the top-level object, dispatching on each key."""
        char = yield from self._skip_ws()
        if char == "}":
            return
        while True:
            if char != '"':
                raise _NotEnvelope
            key = yield from self._string(emit=False)
            char = yield from self._skip_ws()
            if char != ":":
                raise _NotEnvelope
            char = yield from self._skip_ws()
            pushback = yield from self._value(key, char)
            char = yield from self._skip_ws(pushback)
            if char == "}":
                return
            if char != ",":
                raise _NotEnvelope
            char = yield from self._skip_ws()

    def _value(self, key: str, first: str) -> Generator[None, str, str | None]:
        """Parse one value. Returns a pushback char when it over-read.

        Only ``speak`` streams; every other value is captured as raw text,
        because a value we are not speaking has no latency budget attached.
        Key order independence falls out of this: a ``tool_call``-first
        envelope simply captures the tool object, then streams ``speak`` when
        it arrives.
        """
        if key == "speak" and first == '"':
            yield from self._string(emit=True)
            return None

        self._cap_key = key
        self._cap_buf = [first]
        pushback = yield from self._capture(first, self._cap_buf)
        raw = "".join(self._cap_buf)
        self._cap_key = None
        self._cap_buf = []
        if key == "tool_call":
            self._try_emit_tool(raw)
        return pushback

    def _string(self, emit: bool) -> Generator[None, str, str]:
        """Decode a JSON string, optionally emitting deltas as it decodes.

        This is where the latency win lives: with ``emit=True`` a character is
        handed to TTS as soon as it is decodable, not when the string closes.
        Escapes are the only thing ever withheld, and only until complete --
        ``\\u0939`` split across six chunks holds back six characters, never a
        word.
        """
        out: list[str] = []
        pending_high: int | None = None  # unpaired UTF-16 high surrogate

        def take(text: str) -> None:
            out.append(text)
            if emit:
                self._emit_speak(text)

        def flush_high() -> None:
            # A high surrogate with no partner is not encodable text; emitting
            # U+FFFD keeps the stream valid UTF-8 for downstream TTS.
            nonlocal pending_high
            if pending_high is not None:
                pending_high = None
                take(_REPLACEMENT)

        while True:
            char = yield
            if char == '"':
                flush_high()
                return "".join(out)
            if char != "\\":
                flush_high()
                take(char)
                continue

            esc = yield
            if esc != "u":
                flush_high()
                take(_SIMPLE_ESCAPES.get(esc, esc))
                continue

            digits = ""
            for _ in range(4):
                digits += yield
            try:
                code = int(digits, 16)
            except ValueError:
                flush_high()
                take("\\u" + digits)  # lenient: keep the text, lose the escape
                continue

            if 0xDC00 <= code <= 0xDFFF and pending_high is not None:
                # Non-BMP characters (emoji, rare scripts) arrive as a UTF-16
                # surrogate pair: two \u escapes that mean one character.
                high, pending_high = pending_high, None
                take(chr(0x10000 + ((high - 0xD800) << 10) + (code - 0xDC00)))
                continue
            if 0xD800 <= code <= 0xDBFF:
                flush_high()
                pending_high = code
                continue
            flush_high()
            take(_REPLACEMENT if 0xD800 <= code <= 0xDFFF else chr(code))

    def _capture(self, first: str, buf: list[str]) -> Generator[None, str, str | None]:
        """Consume one JSON value verbatim into ``buf``.

        Brace counting is string-aware, so ``{"q": "}"}`` does not end early.
        Bare literals (numbers, ``true``, ``null``) have no closing token, so
        the terminating delimiter is returned to the caller as a pushback.
        """
        if first == '"':
            yield from self._capture_string(buf)
            return None
        if first in "{[":
            depth = 1
            while depth:
                char = yield
                buf.append(char)
                if char == '"':
                    yield from self._capture_string(buf)
                elif char in "{[":
                    depth += 1
                elif char in "}]":
                    depth -= 1
            return None
        while True:
            char = yield
            if char in _LITERAL_ENDERS:
                return char
            buf.append(char)

    def _capture_string(self, buf: list[str]) -> Generator[None, str, None]:
        """Copy a JSON string (opening quote already in ``buf``) verbatim."""
        while True:
            char = yield
            buf.append(char)
            if char == "\\":
                buf.append((yield))  # an escaped quote must not end the string
            elif char == '"':
                return


def _coalesce(events: list[ParserEvent]) -> list[ParserEvent]:
    """Merge adjacent SpeakDeltas so one feed() yields at most one per run."""
    merged: list[ParserEvent] = []
    for event in events:
        if isinstance(event, SpeakDelta):
            if not event.text:
                continue
            if merged and isinstance(merged[-1], SpeakDelta):
                merged[-1] = SpeakDelta(merged[-1].text + event.text)
                continue
        merged.append(event)
    return merged


def parse_all(chunks: Iterable[str]) -> list[ParserEvent]:
    """Run a whole stream through a fresh parser. Convenience for tests/CLI."""
    parser = SpeakToolParser()
    events: list[ParserEvent] = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finalize())
    return events
