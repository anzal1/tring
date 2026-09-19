"""Tring's evaluation harness: scripted conversation tests as YAML.

Public surface:

* :func:`run` drives every eval file under a path (or a single file) through
  the real ``CascadeRuntime`` and returns an :class:`~tring.eval.models.EvalReport`.
* :mod:`tring.eval.models` is the eval-file schema (``EvalFile``, ``TurnSpec``,
  ``ExpectBlock``, ``JudgeSpec``) and the report schema (``EvalReport``,
  ``EvalResult``, ``TurnResult``, ``AssertionResult``, ``JudgeResult``).
* ``python -m tring.eval <path> [--json]`` is the CI entry point: it exits 1
  on any assertion or judge failure, 0 otherwise.

See ``docs/V4_PLAN.md`` ("Track 1: Evaluation harness") for the eval YAML
shape this package implements, and ``tring.eval.runner`` for how a scripted
turn actually drives the runtime.
"""

from __future__ import annotations

from tring.eval.models import (
    AssertionResult,
    EvalFile,
    EvalReport,
    EvalResult,
    ExpectBlock,
    JudgeResult,
    JudgeSpec,
    TurnResult,
    TurnSpec,
)
from tring.eval.runner import run

__all__ = [
    "AssertionResult",
    "EvalFile",
    "EvalReport",
    "EvalResult",
    "ExpectBlock",
    "JudgeResult",
    "JudgeSpec",
    "TurnResult",
    "TurnSpec",
    "run",
]
