"""LLM judges: a free-text rubric scored by a registered LLM provider.

Deterministic assertions (``tool_called``, ``reply_mentions``, ...) need no
model in the loop; a judge exists for the assertions that do — "did the
assistant stay polite", "did it avoid making up a price" — where the check
is a matter of degree, not a fixed string. A judge is therefore optional and
honestly so: an eval file that names no ``llm`` provider for a judge gets a
``skipped`` result with a reason, never a silent pass or a fabricated score.

Judges reuse the same provider registry an ``AgentSpec`` selects real
providers from (``tring.providers.registry``), so grading an eval costs
nothing new to configure beyond naming a provider that already exists —
including a scripted test fake registered under a ``test_*`` name, which is
exactly how this module's own tests exercise it end to end.
"""

from __future__ import annotations

import json
import re
from typing import cast

from tring.eval.models import JudgeResult, JudgeSpec
from tring.events import BotUtterance, UserTranscript
from tring.providers import registry
from tring.providers.base import LLMProvider
from tring.providers.registry import UnknownProviderError
from tring.session import CallSession

#: The instruction wrapped around every judge's own rubric prompt. Asking for
#: exactly one JSON object (rather than "a score from 1 to 5") is what makes
#: parsing the reply a JSON decode instead of a guess at the model's prose.
_JUDGE_INSTRUCTION = (
    "You are grading one voice-agent phone conversation against a single "
    "rubric line. Read the rubric and the transcript, then reply with "
    'exactly one JSON object and nothing else: {"score": <integer 1-5>, '
    '"reason": "<one short sentence explaining the score>"}. 1 means the '
    "rubric was clearly violated; 5 means it was clearly satisfied."
)

#: Fallback for a judge model that does not follow the JSON-only instruction
#: verbatim (wraps it in a sentence, fences it in markdown, ...). A score is
#: still worth extracting from that reply rather than discarding it, since a
#: judge that occasionally prefaces its JSON with a sentence is not the same
#: failure as a judge that produced no score at all.
_SCORE_PATTERN = re.compile(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)')


def render_transcript(session: CallSession) -> str:
    """Render the conversation so far as plain ``role: text`` lines.

    Only what a human reading a transcript would expect goes in: what the
    caller said, what the bot said. Tool calls, cost lines and choreography
    detail are noise for a rubric about conversational quality, and would
    only spend the judge model's context on events it was not asked to grade.
    """
    lines: list[str] = []
    for event in session.history:
        if isinstance(event, UserTranscript) and event.final:
            lines.append(f"user: {event.text}")
        elif isinstance(event, BotUtterance):
            lines.append(f"bot: {event.text}")
    return "\n".join(lines)


async def run_judge(judge: JudgeSpec, session: CallSession) -> JudgeResult:
    """Score one judge rubric against ``session``'s transcript so far.

    Skipped, never failed, whenever grading cannot happen: no provider was
    named, the named provider is not registered, or the provider's reply had
    no parseable score. Every one of those is a setup gap, not a
    conversational defect the agent under test caused, so it must not read as
    a failing assertion in the report.
    """
    if not judge.llm:
        return JudgeResult(
            name=judge.name,
            status="skipped",
            reason="no llm provider configured for this judge (set judges[].llm)",
        )

    registry._load_builtin()
    try:
        provider = cast(LLMProvider, registry.create("llm", judge.llm, **judge.options))
    except UnknownProviderError as exc:
        return JudgeResult(name=judge.name, status="skipped", reason=str(exc))

    transcript = render_transcript(session)
    messages = [
        {"role": "system", "content": _JUDGE_INSTRUCTION},
        {
            "role": "user",
            "content": f"RUBRIC: {judge.prompt}\n\nTRANSCRIPT:\n{transcript}",
        },
    ]

    reply = ""
    async for chunk in provider.generate(messages, None):
        reply += chunk.text

    score = _parse_score(reply)
    if score is None:
        return JudgeResult(
            name=judge.name,
            status="skipped",
            reason=f"judge model reply had no parseable score: {reply!r}",
        )

    passed = score >= judge.min_score
    return JudgeResult(
        name=judge.name,
        status="passed" if passed else "failed",
        score=score,
        reason=f"scored {score:g} against a minimum of {judge.min_score:g}",
    )


def _parse_score(reply: str) -> float | None:
    """Pull the numeric score out of a judge model's reply.

    Tries strict JSON first — the format ``_JUDGE_INSTRUCTION`` asks for —
    then falls back to a regex so a model that wraps the object in a
    sentence or a markdown fence still yields a usable score.
    """
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
        score = data.get("score")
        if isinstance(score, int | float) and not isinstance(score, bool):
            return float(score)

    match = _SCORE_PATTERN.search(reply)
    return float(match.group(1)) if match else None


__all__ = ["render_transcript", "run_judge"]
