"""End-to-end tests for :mod:`tring.eval`.

Fakes are registered in the real provider registry under test-only names --
the same house pattern ``tests/test_cascade.py`` uses -- so the harness runs
through its real path: YAML -> ``AgentSpec`` -> ``registry.create`` ->
``CascadeRuntime``. Nothing here mocks the runner itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import yaml

from tring.agent import AgentSpec, ProviderSelection, RuntimeConfig, ToolDef
from tring.eval import run
from tring.eval.__main__ import main as eval_main
from tring.providers.base import LLMChunk, LLMProvider, TTSChunk, TTSProvider, Usage
from tring.providers.registry import register
from tring.runtimes.base import AudioFrame

# ---------------------------------------------------------------------------
# Fakes, registered under names unique to this file so they cannot collide
# with the (also test-only) names test_cascade.py registers.
# ---------------------------------------------------------------------------

#: Scripted LLM completions, looked up by the ``script_id`` provider option.
EVAL_LLM_SCRIPTS: dict[str, list[str]] = {}

#: Canned judge-model replies, looked up by the ``response_id`` provider option.
JUDGE_REPLIES: dict[str, str] = {}


@register("llm", "eval_test_scripted")
class _ScriptedEvalLLM(LLMProvider):
    """Replays one scripted completion per ``generate`` call, envelope-shaped."""

    name = "eval_test_scripted"

    def __init__(self, script_id: str = "", **_options: Any) -> None:
        self.script = list(EVAL_LLM_SCRIPTS[script_id])

    async def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMChunk]:
        if not self.script:
            raise AssertionError("eval-test LLM ran out of scripted responses")
        completion = self.script.pop(0)
        yield LLMChunk(text=completion)
        yield LLMChunk(
            text="",
            finish=True,
            usage=[
                Usage(units=10, unit_name="tokens_in", model="fake-eval-llm"),
                Usage(units=5, unit_name="tokens_out", model="fake-eval-llm"),
            ],
        )


@register("tts", "eval_test_silent")
class _SilentEvalTTS(TTSProvider):
    """Consumes whatever text arrives and emits one silent frame. No usage."""

    name = "eval_test_silent"

    async def synthesize(
        self, text: AsyncIterator[str], voice: str | None = None
    ) -> AsyncIterator[TTSChunk]:
        buffer = ""
        async for delta in text:
            buffer += delta
        if buffer:
            yield TTSChunk(frame=AudioFrame(pcm=b"\x00\x00", sample_rate=16000, channels=1))


@register("llm", "eval_test_judge")
class _CannedJudgeLLM(LLMProvider):
    """A judge model that always replies with one canned, scripted reply."""

    name = "eval_test_judge"

    def __init__(self, response_id: str = "", **_options: Any) -> None:
        self.reply = JUDGE_REPLIES[response_id]

    async def generate(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> AsyncIterator[LLMChunk]:
        yield LLMChunk(text=self.reply, finish=True)


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

#: Deliberately unregistered: proves the runner overrides the agent's own STT
#: choice rather than merely happening to work with a real one configured.
_UNUSED_STT = "unused_stt_should_be_overridden_by_the_eval_runner"

BOOKING_SCRIPT = [
    (
        '{"speak": "Sure, checking that now.", "tool_call": {"name": '
        '"check_appointment", "arguments": {"phone": "9820011223", "day": '
        '"tuesday", "waiting_message": "One moment.", "spoken_mode": '
        '"answer_pending", "post_tool_response": "respond"}}}'
    ),
    '{"speak": "You are booked for tuesday.", "tool_call": null}',
]


def _write_agent(path: Path, script_id: str) -> None:
    spec = AgentSpec(
        name="front-desk",
        persona="You are a concise, polite front-desk assistant.",
        runtime=RuntimeConfig(
            routing={
                "default": ProviderSelection(
                    stt=_UNUSED_STT,
                    llm="eval_test_scripted",
                    tts="eval_test_silent",
                    options={"llm": {"script_id": script_id}},
                )
            }
        ),
        tools=[
            ToolDef(
                name="check_appointment",
                description="Check whether an appointment slot is open.",
                parameters={
                    "type": "object",
                    "properties": {
                        "phone": {"type": "string"},
                        "day": {"type": "string"},
                    },
                    "required": ["phone"],
                },
                handler="check_appointment",
            )
        ],
    )
    spec.to_yaml(path)


def _write_eval(
    path: Path,
    agent_rel: str,
    turns: list[dict[str, Any]],
    judges: list[dict[str, Any]] | None = None,
    tool_stubs: dict[str, Any] | None = None,
) -> None:
    data: dict[str, Any] = {"agent": agent_rel, "turns": turns}
    if judges is not None:
        data["judges"] = judges
    if tool_stubs is not None:
        data["tool_stubs"] = tool_stubs
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


class _StepClock:
    """A fake clock that advances by a fixed step on every call.

    Standing in for ``time.monotonic``: it makes ``max_first_token_s``
    assertions deterministic instead of dependent on how fast the machine
    running the tests happens to be. Each call returns strictly more time
    than the last, so "did event B happen after event A" checks behave
    exactly like a real clock, just without real waiting.
    """

    def __init__(self, step: float = 0.05) -> None:
        self._t = 0.0
        self._step = step

    def __call__(self) -> float:
        value = self._t
        self._t += self._step
        return value


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_passing_eval_reports_every_assertion_passed(tmp_path: Path) -> None:
    EVAL_LLM_SCRIPTS["passing"] = list(BOOKING_SCRIPT)
    _write_agent(tmp_path / "agent.yaml", "passing")
    _write_eval(
        tmp_path / "booking.yaml",
        "agent.yaml",
        turns=[
            {
                "user": "book me for tuesday, 98200 11223",
                "expect": {
                    "tool_called": "check_appointment",
                    "tool_args_include": {"phone": "9820011223"},
                    "reply_mentions": ["tuesday"],
                },
            }
        ],
        tool_stubs={"check_appointment": {"confirmation": "R-417"}},
    )

    report = await run(tmp_path / "booking.yaml")

    assert report.passed is True
    (result,) = report.results
    (turn,) = result.turns
    assert turn.passed is True
    assert {a.kind for a in turn.assertions} == {
        "tool_called",
        "tool_args_include",
        "reply_mentions",
    }
    assert all(a.passed for a in turn.assertions)


async def test_failing_assertion_is_reported_without_raising(tmp_path: Path) -> None:
    EVAL_LLM_SCRIPTS["failing"] = list(BOOKING_SCRIPT)
    _write_agent(tmp_path / "agent.yaml", "failing")
    _write_eval(
        tmp_path / "booking.yaml",
        "agent.yaml",
        turns=[
            {
                "user": "book me for tuesday, 98200 11223",
                "expect": {
                    "tool_called": "check_appointment",
                    # Wrong on purpose: the script books "tuesday".
                    "reply_mentions": ["wednesday"],
                },
            }
        ],
        tool_stubs={"check_appointment": {"confirmation": "R-417"}},
    )

    report = await run(tmp_path / "booking.yaml")

    assert report.passed is False
    (turn,) = report.results[0].turns
    assert turn.passed is False
    by_kind = {a.kind: a for a in turn.assertions}
    assert by_kind["tool_called"].passed is True
    assert by_kind["reply_mentions"].passed is False
    assert "wednesday" in by_kind["reply_mentions"].detail


async def test_eval_runner_overrides_agent_stt_to_text_input(tmp_path: Path) -> None:
    """The agent spec names an unregistered STT provider; the eval must not care.

    If the runner failed to force ``text_input``, ``CascadeRuntime.start()``
    would raise ``UnknownProviderError`` trying to construct ``_UNUSED_STT``,
    so this test failing to raise is itself the assertion.
    """
    EVAL_LLM_SCRIPTS["override"] = ['{"speak": "Hello there.", "tool_call": null}']
    _write_agent(tmp_path / "agent.yaml", "override")
    _write_eval(
        tmp_path / "greet.yaml",
        "agent.yaml",
        turns=[{"user": "hi", "expect": {"reply_mentions": ["hello"]}}],
    )

    report = await run(tmp_path / "greet.yaml")

    assert report.passed is True


async def test_judge_is_skipped_without_a_configured_provider(tmp_path: Path) -> None:
    EVAL_LLM_SCRIPTS["judge-skip"] = ['{"speak": "Sure thing.", "tool_call": null}']
    _write_agent(tmp_path / "agent.yaml", "judge-skip")
    _write_eval(
        tmp_path / "eval.yaml",
        "agent.yaml",
        turns=[{"user": "hello", "expect": {}}],
        judges=[{"name": "politeness", "prompt": "Rate 1-5 politeness.", "min_score": 4}],
    )

    report = await run(tmp_path / "eval.yaml")

    # A skipped judge is not a failure: the turn passed and nothing else
    # failed, so the file as a whole passes even though a judge was skipped.
    assert report.passed is True
    (judge,) = report.results[0].judges
    assert judge.status == "skipped"
    assert judge.score is None
    assert "llm" in judge.reason


async def test_judge_scores_via_a_registered_llm_provider(tmp_path: Path) -> None:
    EVAL_LLM_SCRIPTS["judge-pass"] = [
        '{"speak": "Thanks, have a nice day!", "tool_call": null}'
    ]
    JUDGE_REPLIES["polite"] = '{"score": 5, "reason": "warm and concise"}'
    _write_agent(tmp_path / "agent.yaml", "judge-pass")
    _write_eval(
        tmp_path / "eval.yaml",
        "agent.yaml",
        turns=[{"user": "hello", "expect": {}}],
        judges=[
            {
                "name": "politeness",
                "prompt": "Rate 1-5 politeness.",
                "min_score": 4,
                "llm": "eval_test_judge",
                "options": {"response_id": "polite"},
            }
        ],
    )

    report = await run(tmp_path / "eval.yaml")

    assert report.passed is True
    (judge,) = report.results[0].judges
    assert judge.status == "passed"
    assert judge.score == 5.0


async def test_judge_fails_below_its_min_score(tmp_path: Path) -> None:
    EVAL_LLM_SCRIPTS["judge-fail"] = ['{"speak": "whatever.", "tool_call": null}']
    JUDGE_REPLIES["rude"] = '{"score": 2, "reason": "curt and dismissive"}'
    _write_agent(tmp_path / "agent.yaml", "judge-fail")
    _write_eval(
        tmp_path / "eval.yaml",
        "agent.yaml",
        turns=[{"user": "hello", "expect": {}}],
        judges=[
            {
                "name": "politeness",
                "prompt": "Rate 1-5 politeness.",
                "min_score": 4,
                "llm": "eval_test_judge",
                "options": {"response_id": "rude"},
            }
        ],
    )

    report = await run(tmp_path / "eval.yaml")

    assert report.passed is False
    (judge,) = report.results[0].judges
    assert judge.status == "failed"
    assert judge.score == 2.0


async def test_latency_assertion_passes_within_bound_and_fails_when_tight(
    tmp_path: Path,
) -> None:
    EVAL_LLM_SCRIPTS["latency-a"] = ['{"speak": "Right away.", "tool_call": null}']
    EVAL_LLM_SCRIPTS["latency-b"] = ['{"speak": "Right away.", "tool_call": null}']
    _write_agent(tmp_path / "agent_a.yaml", "latency-a")
    _write_agent(tmp_path / "agent_b.yaml", "latency-b")
    _write_eval(
        tmp_path / "generous.yaml",
        "agent_a.yaml",
        turns=[{"user": "hi", "expect": {"max_first_token_s": 100.0}}],
    )
    _write_eval(
        tmp_path / "tight.yaml",
        "agent_b.yaml",
        turns=[{"user": "hi", "expect": {"max_first_token_s": 0.0001}}],
    )

    generous = await run(tmp_path / "generous.yaml", clock=_StepClock())
    tight = await run(tmp_path / "tight.yaml", clock=_StepClock())

    assert generous.passed is True
    (latency_ok,) = generous.results[0].turns[0].assertions
    assert latency_ok.kind == "max_first_token_s"
    assert latency_ok.passed is True

    assert tight.passed is False
    (latency_bad,) = tight.results[0].turns[0].assertions
    assert latency_bad.passed is False
    assert "limit 0.000" in latency_bad.detail


async def test_directory_target_skips_non_eval_yaml_files(tmp_path: Path) -> None:
    """An agent.yaml with no ``turns`` key sitting in an evals/ dir is not an eval."""
    EVAL_LLM_SCRIPTS["dir"] = ['{"speak": "hi there.", "tool_call": null}']
    _write_agent(tmp_path / "agent.yaml", "dir")
    _write_eval(
        tmp_path / "one.yaml", "agent.yaml", turns=[{"user": "hi", "expect": {}}]
    )

    report = await run(tmp_path)

    assert len(report.results) == 1
    assert report.results[0].path.endswith("one.yaml")


async def test_missing_eval_path_raises() -> None:
    try:
        await run("/no/such/eval/path/at/all")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError for a missing eval path")


def test_cli_exit_code_reflects_report_pass_fail(tmp_path: Path, capsys: Any) -> None:
    EVAL_LLM_SCRIPTS["cli-pass"] = ['{"speak": "hello there.", "tool_call": null}']
    _write_agent(tmp_path / "agent.yaml", "cli-pass")
    _write_eval(
        tmp_path / "eval.yaml",
        "agent.yaml",
        turns=[{"user": "hi", "expect": {"reply_mentions": ["hello"]}}],
    )

    exit_code = eval_main([str(tmp_path / "eval.yaml")])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Overall: PASS" in out

    EVAL_LLM_SCRIPTS["cli-fail"] = ['{"speak": "hello there.", "tool_call": null}']
    _write_agent(tmp_path / "agent2.yaml", "cli-fail")
    _write_eval(
        tmp_path / "eval2.yaml",
        "agent2.yaml",
        turns=[{"user": "hi", "expect": {"reply_mentions": ["goodbye"]}}],
    )
    exit_code = eval_main([str(tmp_path / "eval2.yaml"), "--json"])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert '"passed": false' in out
