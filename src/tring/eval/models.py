"""The eval-file schema and the report schema it produces.

An eval file is plain YAML, loaded into :class:`EvalFile` the same way an
agent spec loads into :class:`~tring.agent.AgentSpec`: no callables, nothing
that cannot round-trip through ``yaml.safe_load``. Running one produces an
:class:`EvalReport`, built once by :mod:`tring.eval.runner` and read many
times (printed, diffed in CI logs, asserted on directly in tests).

See ``docs/V4_PLAN.md`` ("Track 1: Evaluation harness") for the worked
example this schema implements::

    agent: agent.yaml
    turns:
      - user: "book me for tuesday, 98200 11223"
        expect:
          tool_called: check_appointment
          tool_args_include: { phone: "9820011223" }
          reply_mentions: ["tuesday"]
          max_first_token_s: 2.0
    judges:
      - name: politeness
        prompt: "Rate 1-5 whether the assistant stayed polite and concise."
        min_score: 4
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# The eval file itself
# ---------------------------------------------------------------------------


class ExpectBlock(BaseModel):
    """What one turn's assertions check, all of them optional and independent.

    An eval file states only the assertions it cares about for a given turn;
    a field left unset is simply not checked, not checked-and-passed.
    """

    #: Name of a tool that must have been called this turn (a
    #: ``ToolCallStarted`` event with this ``tool_name``).
    tool_called: str | None = None

    #: Arguments that must appear in the matched tool call's own arguments
    #: (never the choreography fields; see ``ToolCallStarted.arguments``).
    #: Checked against the ``tool_called`` match if that field is also set,
    #: otherwise against the turn's first tool call.
    tool_args_include: dict[str, Any] = Field(default_factory=dict)

    #: Case-insensitive any-of: the turn's spoken reply (every ``BotUtterance``
    #: this turn, joined) must contain at least one of these substrings.
    reply_mentions: list[str] = Field(default_factory=list)

    #: Upper bound, in seconds, on the delay between the caller's final
    #: transcript and the first thing the caller hears back this turn. See
    #: ``tring.eval.runner._latency_result`` for exactly what "first" means
    #: given the events the shipped runtime actually emits.
    max_first_token_s: float | None = None


class TurnSpec(BaseModel):
    """One scripted line from the caller, and what the agent must do with it."""

    user: str
    expect: ExpectBlock = Field(default_factory=ExpectBlock)


class JudgeSpec(BaseModel):
    """One LLM-graded rubric, run once per eval file against the full transcript.

    ``llm`` names a provider already registered in the real provider registry
    (``tring.providers.registry``) — the same registry an ``AgentSpec`` selects
    providers from — so a judge model is configured exactly like any other
    provider, never hard-coded to one vendor. Leaving it unset is a valid,
    honest choice: the judge is reported ``skipped`` with a reason rather than
    silently omitted or run against a guessed default.
    """

    name: str
    prompt: str
    min_score: float
    llm: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class EvalFile(BaseModel):
    """One YAML eval file: an agent to run, a scripted conversation, judges."""

    #: Path to the ``AgentSpec`` YAML this eval drives, resolved relative to
    #: the eval file's own directory (see ``runner._run_eval_file``).
    agent: str
    turns: list[TurnSpec]
    judges: list[JudgeSpec] = Field(default_factory=list)

    #: Static results for tool handlers the eval file wants stubbed, keyed by
    #: tool name: ``{tool_name: <result the handler always returns>}``. A tool
    #: not listed here has no handler bound at all, which is a legitimate
    #: eval too — ``tool_called``/``tool_args_include`` only need the model's
    #: own ``ToolCallStarted`` event, never a successful handler run.
    tool_stubs: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> EvalFile:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

AssertionKind = Literal[
    "tool_called", "tool_args_include", "reply_mentions", "max_first_token_s"
]
JudgeStatus = Literal["passed", "failed", "skipped"]


class AssertionResult(BaseModel):
    """One evaluated assertion from one turn's ``ExpectBlock``."""

    kind: AssertionKind
    passed: bool
    #: Human-readable explanation, written to be read straight off a failing
    #: CI log: what was expected, what actually happened.
    detail: str


class TurnResult(BaseModel):
    """One turn's outcome: what was said, what was checked, how long it took."""

    index: int
    user: str
    assertions: list[AssertionResult] = Field(default_factory=list)
    #: Real wall-clock time the turn took to drain, independent of whatever
    #: clock the session was constructed with — useful even when a test
    #: injects a fake clock for deterministic ``max_first_token_s`` checks.
    wall_time_s: float

    @property
    def passed(self) -> bool:
        return all(assertion.passed for assertion in self.assertions)


class JudgeResult(BaseModel):
    """One judge's outcome: scored, failed the bar, or honestly skipped."""

    name: str
    status: JudgeStatus
    score: float | None = None
    reason: str = ""


class EvalResult(BaseModel):
    """The full outcome of running one eval file."""

    path: str
    agent: str
    turns: list[TurnResult] = Field(default_factory=list)
    judges: list[JudgeResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """A skipped judge does not fail the file; an unscored one is not a defect."""
        return all(turn.passed for turn in self.turns) and all(
            judge.status != "failed" for judge in self.judges
        )


class EvalReport(BaseModel):
    """The combined outcome of every eval file a ``run()`` call covered."""

    results: list[EvalResult] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    def _tallies(self) -> dict[str, int]:
        assertions_passed = assertions_failed = 0
        judges_passed = judges_failed = judges_skipped = 0
        for result in self.results:
            for turn in result.turns:
                for assertion in turn.assertions:
                    if assertion.passed:
                        assertions_passed += 1
                    else:
                        assertions_failed += 1
            for judge in result.judges:
                if judge.status == "passed":
                    judges_passed += 1
                elif judge.status == "failed":
                    judges_failed += 1
                else:
                    judges_skipped += 1
        return {
            "assertions_passed": assertions_passed,
            "assertions_failed": assertions_failed,
            "judges_passed": judges_passed,
            "judges_failed": judges_failed,
            "judges_skipped": judges_skipped,
            "files_passed": sum(1 for result in self.results if result.passed),
            "files_failed": sum(1 for result in self.results if not result.passed),
        }

    def as_text(self) -> str:
        """Render a plain-text table suitable for a terminal or a CI log."""
        tallies = self._tallies()
        lines = ["Eval Report", "=" * 11]
        lines.append(
            f"{tallies['assertions_passed']} assertion(s) passed, "
            f"{tallies['assertions_failed']} failed  |  "
            f"{tallies['judges_passed']} judge(s) passed, "
            f"{tallies['judges_failed']} failed, "
            f"{tallies['judges_skipped']} skipped"
        )

        for result in self.results:
            lines.append("")
            lines.append(f"[{'PASS' if result.passed else 'FAIL'}] {result.path}")
            for turn in result.turns:
                lines.append(
                    f"  [{'PASS' if turn.passed else 'FAIL'}] turn {turn.index}: "
                    f"{turn.user!r}  ({turn.wall_time_s:.3f}s)"
                )
                for assertion in turn.assertions:
                    mark = "ok" if assertion.passed else "FAIL"
                    lines.append(f"        {mark:<4} {assertion.kind}: {assertion.detail}")
            for judge in result.judges:
                marker = {"passed": "ok", "failed": "FAIL", "skipped": "skip"}[judge.status]
                lines.append(f"  [{marker}] judge {judge.name}: {judge.reason}")

        lines.append("")
        lines.append(f"Overall: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)

    def as_json(self) -> str:
        """Render the same information as machine-readable JSON."""
        payload = {
            "passed": self.passed,
            "tallies": self._tallies(),
            "results": [result.model_dump(mode="json") for result in self.results],
        }
        return json.dumps(payload, indent=2)


__all__ = [
    "AssertionKind",
    "AssertionResult",
    "EvalFile",
    "EvalReport",
    "EvalResult",
    "ExpectBlock",
    "JudgeResult",
    "JudgeSpec",
    "JudgeStatus",
    "TurnResult",
    "TurnSpec",
]
