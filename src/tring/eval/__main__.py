"""``python -m tring.eval`` — run YAML eval files as a CI check.

    python -m tring.eval evals/              # every eval file under a dir
    python -m tring.eval evals/booking.yaml   # a single eval file
    python -m tring.eval evals/ --json        # machine-readable report

Exit code is 1 if any assertion or judge failed, 0 otherwise — this is meant
to be the last step of a CI job, with no extra glue around it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from tring.eval.runner import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tring.eval",
        description="Run Tring eval YAML files against the real CascadeRuntime.",
    )
    parser.add_argument(
        "path", type=Path, help="an eval YAML file, or a directory of eval files"
    )
    parser.add_argument(
        "--json", action="store_true", help="print the report as JSON instead of a table"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = asyncio.run(run(args.path))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(report.as_json() if args.json else report.as_text())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
