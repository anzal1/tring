"""The ``tring`` command: one front door to the stack's runnable pieces.

Each subcommand delegates to a module that also works as ``python -m ...``;
this multiplexer exists so that onboarding is one installed command with
discoverable help, not a scavenger hunt through module paths.
"""

from __future__ import annotations

import sys

from tring import __version__

_USAGE = f"""tring {__version__} — the open-source voice agent stack

usage: tring <command> [args]

commands:
  studio    open Tring Studio in the browser (agent designer, live testing,
            flow builder, session replay).  tring studio --help
  eval      run YAML conversation evals against an agent, CI exit codes.
            tring eval evals/ --help
  version   print the installed version

docs: https://github.com/anzal1/tring
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help", "help"}:
        print(_USAGE)
        return 0
    command, rest = args[0], args[1:]
    if command == "version" or command == "--version":
        print(__version__)
        return 0
    if command == "studio":
        from tring.studio.__main__ import main as studio_main

        return studio_main(rest)
    if command == "eval":
        from tring.eval.__main__ import main as eval_main

        return eval_main(rest)
    print(f"tring: unknown command {command!r}\n", file=sys.stderr)
    print(_USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
