"""Drives YAML eval files through the real ``CascadeRuntime``.

The whole point of the harness is that it exercises production code, not a
simulation of it: one real ``CallSession``, one real ``CascadeRuntime``, the
real choreography/parser/ledger primitives underneath it. The only
substitution is the STT slot, forced to the shipped ``text_input`` provider
(see :func:`_force_text_input_stt`) so a scripted turn is plain text instead
of a recorded utterance — exactly the substitution Studio's text-testing mode
makes, for the identical reason.

What "first token" means here
------------------------------

``events.py`` has no event for "the model produced its first character": the
runtime intentionally does not expose one (see ``cascade.py``'s
``_BotSpeech`` docstring — ``BotUtterance`` fires only once the whole
``speak`` string is known). ``max_first_token_s`` is therefore measured
against the first ``BotUtterance`` or ``ToolCallStarted`` after the turn's
final transcript — the first moment the event stream shows the caller's wait
ending, whether that is the start of a spoken reply or a tool call's own
waiting message. It is an honest proxy, not the model's true
time-to-first-token, and :func:`_latency_result` says so.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import yaml

from tring.agent import AgentSpec
from tring.eval.judges import run_judge
from tring.eval.models import (
    AssertionResult,
    EvalFile,
    EvalReport,
    EvalResult,
    ExpectBlock,
    TurnResult,
    TurnSpec,
)
from tring.events import BotUtterance, SessionEvent, ToolCallStarted, UserTranscript
from tring.runtimes.base import AudioFrame
from tring.runtimes.cascade import CascadeRuntime
from tring.session import CallSession

#: The STT provider every eval turn is forced onto; see _force_text_input_stt.
_TEXT_INPUT_STT = "text_input"


async def run(path: str | Path, *, clock: Callable[[], float] | None = None) -> EvalReport:
    """Run every eval file found at ``path`` and return one combined report.

    ``path`` is either a single eval YAML file or a directory. Under a
    directory, every ``*.yaml``/``*.yml`` (recursively) that parses as a
    mapping with a top-level ``turns`` key is treated as an eval file — see
    :func:`_looks_like_eval_file` — so an agent spec kept alongside its evals
    is skipped rather than mistaken for one.

    ``clock`` is forwarded to each eval file's ``CallSession`` (see
    ``CallSession.__init__``), letting a caller run every file against the
    same deterministic clock — the house pattern for latency assertions that
    must not depend on real wall-clock jitter.
    """
    target = Path(path)
    files = _discover_eval_files(target)
    if not files:
        raise FileNotFoundError(f"no eval files found under {target}")
    results = [await _run_eval_file(file, clock=clock) for file in files]
    return EvalReport(results=results)


def _discover_eval_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"eval path does not exist: {path}")
    candidates = sorted(set(path.rglob("*.yaml")) | set(path.rglob("*.yml")))
    return [file for file in candidates if _looks_like_eval_file(file)]


def _looks_like_eval_file(path: Path) -> bool:
    """An eval file is any YAML mapping with a top-level ``turns`` key.

    Cheap enough to run over every YAML file under a directory, and it is
    what lets ``evals/`` hold an eval file next to the agent spec it
    references without the agent spec being mistaken for one.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError:
        return False
    return isinstance(data, dict) and "turns" in data


async def _run_eval_file(eval_path: Path, *, clock: Callable[[], float] | None) -> EvalResult:
    eval_file = EvalFile.from_yaml(eval_path)
    # Resolved against the eval file's own directory, matching how
    # AgentSpec.from_yaml/to_yaml treat paths elsewhere in the stack: an eval
    # file is portable together with the agent spec it names, wherever the
    # pair is checked out.
    agent_path = (eval_path.parent / eval_file.agent).resolve()
    agent = AgentSpec.from_yaml(agent_path)
    _force_text_input_stt(agent)

    handlers = {
        name: _make_stub_handler(result) for name, result in eval_file.tool_stubs.items()
    }

    session = CallSession(agent, clock=clock)
    runtime = CascadeRuntime(session, handlers=handlers)
    await runtime.start()

    turns = [
        await _run_turn(runtime, session, index, turn)
        for index, turn in enumerate(eval_file.turns)
    ]

    await runtime.stop()

    judges = [await run_judge(judge, session) for judge in eval_file.judges]

    return EvalResult(path=str(eval_path), agent=str(agent_path), turns=turns, judges=judges)


def _force_text_input_stt(agent: AgentSpec) -> None:
    """Force every routing entry's STT slot to ``text_input``.

    A scripted turn is ``TurnSpec.user`` text, never recorded audio, so the
    agent's own STT choice (``faster_whisper``, ``deepgram``, ...) would
    never see anything to transcribe and has no bearing on what an eval
    actually exercises: persona, tool choreography, and the LLM/TTS wiring an
    agent ships with. Every routing entry is rewritten, not just the one the
    agent's primary language selects, so an eval for a multi-language agent
    cannot silently leave a microphone-only provider wired into a turn spoken
    in its second language.

    Studio's text-testing mode makes the identical substitution for the
    identical reason (``tring.studio.server._force_studio_providers``); it is
    duplicated here rather than imported, because ``tring.eval`` is a core
    package and must not import from ``tring.studio``.
    """
    for selection in agent.runtime.routing.values():
        selection.stt = _TEXT_INPUT_STT


def _make_stub_handler(result: Any) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    """Bind one ``tool_stubs`` entry to a handler that always returns it.

    Ignores the arguments it is called with entirely: it exists so a turn
    does not dead-end on ``LookupError: no handler bound`` when the eval
    cares about *what the model asked for* — checked via ``tool_called`` /
    ``tool_args_include`` against the ``ToolCallStarted`` event, which is
    emitted regardless of whether a handler runs at all — rather than a real
    integration's behavior.
    """

    async def handler(_arguments: dict[str, Any]) -> Any:
        return result

    return handler


async def _run_turn(
    runtime: CascadeRuntime, session: CallSession, index: int, turn: TurnSpec
) -> TurnResult:
    history_before = len(session.history)
    fallback_started_at = session.elapsed
    wall_start = time.perf_counter()

    await runtime.push_audio(AudioFrame(pcm=turn.user.encode("utf-8")))
    await runtime.drain()

    wall_time_s = time.perf_counter() - wall_start
    new_events = session.history[history_before:]
    turn_started_at = _turn_started_at(new_events, fallback_started_at)
    assertions = _evaluate_expect(turn.expect, new_events, turn_started_at)
    return TurnResult(
        index=index, user=turn.user, assertions=assertions, wall_time_s=wall_time_s
    )


def _turn_started_at(events: list[SessionEvent], fallback: float) -> float:
    """The instant this turn's clock starts: the caller's final transcript.

    ``fallback`` covers the pathological case where the pushed text somehow
    produced no transcript at all (e.g. an all-whitespace turn, which
    ``text_input`` drops) — the turn simply started "now" rather than the
    latency assertion crashing on an empty search.
    """
    for event in events:
        if isinstance(event, UserTranscript) and event.final:
            return event.at
    return fallback


def _evaluate_expect(
    expect: ExpectBlock, events: list[SessionEvent], turn_started_at: float
) -> list[AssertionResult]:
    results: list[AssertionResult] = []
    tool_calls = [event for event in events if isinstance(event, ToolCallStarted)]
    bot_texts = [event.text for event in events if isinstance(event, BotUtterance)]

    matched: ToolCallStarted | None = None
    if expect.tool_called is not None:
        matched = next(
            (call for call in tool_calls if call.tool_name == expect.tool_called), None
        )
        results.append(_tool_called_result(expect.tool_called, matched, tool_calls))

    if expect.tool_args_include:
        target = matched if expect.tool_called is not None else _first_or_none(tool_calls)
        results.append(_tool_args_result(expect.tool_args_include, target))

    if expect.reply_mentions:
        results.append(_reply_mentions_result(expect.reply_mentions, bot_texts))

    if expect.max_first_token_s is not None:
        results.append(_latency_result(expect.max_first_token_s, events, turn_started_at))

    return results


def _first_or_none(calls: list[ToolCallStarted]) -> ToolCallStarted | None:
    return calls[0] if calls else None


def _tool_called_result(
    expected: str, matched: ToolCallStarted | None, tool_calls: list[ToolCallStarted]
) -> AssertionResult:
    if matched is not None:
        return AssertionResult(
            kind="tool_called", passed=True, detail=f"'{expected}' was called"
        )
    seen = [call.tool_name for call in tool_calls]
    detail = (
        f"'{expected}' was not called; tools called this turn: {seen}"
        if seen
        else f"'{expected}' was not called; no tool was called this turn"
    )
    return AssertionResult(kind="tool_called", passed=False, detail=detail)


def _tool_args_result(
    expected: dict[str, Any], target: ToolCallStarted | None
) -> AssertionResult:
    if target is None:
        return AssertionResult(
            kind="tool_args_include",
            passed=False,
            detail="no tool call this turn to check arguments against",
        )
    mismatches = [
        f"{key}: expected {value!r}, got {target.arguments.get(key)!r}"
        for key, value in expected.items()
        if target.arguments.get(key) != value
    ]
    if not mismatches:
        return AssertionResult(
            kind="tool_args_include", passed=True, detail=f"arguments included {expected}"
        )
    return AssertionResult(kind="tool_args_include", passed=False, detail="; ".join(mismatches))


def _reply_mentions_result(mentions: list[str], bot_texts: list[str]) -> AssertionResult:
    reply = " ".join(bot_texts)
    joined = reply.lower()
    found = [mention for mention in mentions if mention.lower() in joined]
    if found:
        return AssertionResult(
            kind="reply_mentions", passed=True, detail=f"found {found} in the reply"
        )
    return AssertionResult(
        kind="reply_mentions",
        passed=False,
        detail=f"none of {mentions} found in reply {reply!r}",
    )


def _latency_result(
    limit: float, events: list[SessionEvent], turn_started_at: float
) -> AssertionResult:
    """See the module docstring for what "first" means here, and why."""
    first = next(
        (event for event in events if isinstance(event, BotUtterance | ToolCallStarted)), None
    )
    if first is None:
        return AssertionResult(
            kind="max_first_token_s", passed=False, detail="turn produced no bot output"
        )
    latency = first.at - turn_started_at
    passed = latency <= limit
    return AssertionResult(
        kind="max_first_token_s",
        passed=passed,
        detail=f"first output at {latency:.3f}s (limit {limit:.3f}s)",
    )


__all__ = ["run"]
