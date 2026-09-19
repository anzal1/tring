"""``python -m tring.studio`` — run the studio server.

CLI per ``docs/STUDIO_PROTOCOL.md``::

    python -m tring.studio [--agent agent.yaml] [--host 127.0.0.1] [--port 8900]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from tring import __version__
from tring.studio.server import (
    DEFAULT_AGENT_PATH,
    DEFAULT_HOST,
    DEFAULT_PORT,
    StudioServer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tring.studio",
        description=(
            "Tring Studio: edit an agent spec and run live text sessions "
            "against it in the browser."
        ),
    )
    parser.add_argument(
        "--agent",
        type=Path,
        default=DEFAULT_AGENT_PATH,
        metavar="PATH",
        help=(
            "agent spec to edit and run (default: %(default)s). The file need "
            "not exist yet: a bundled demo spec is served until the first save."
        ),
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=(
            "interface to bind (default: %(default)s). The studio runs agent "
            "code and has no authentication, so binding beyond loopback exposes "
            "it to everyone who can reach the port."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="TCP port (default: %(default)s); 0 picks a free one.",
    )
    parser.add_argument("--version", action="version", version=f"tring {__version__}")
    return parser


def _banner(server: StudioServer) -> str:
    if server.agent_path.is_file():
        agent_line = str(server.agent_path)
    else:
        agent_line = f"{server.agent_path}  (not created yet: serving the demo spec)"
    return "\n".join(
        [
            f"Tring Studio {__version__}",
            f"  agent  {agent_line}",
            f"  open   {server.url}",
            "",
            "Press Ctrl-C to stop.",
        ]
    )


async def _serve(agent: Path, host: str, port: int) -> None:
    server = StudioServer(agent_path=agent, host=host, port=port)
    await server.start()
    # Printed after binding, so `--port 0` reports the port it actually got.
    print(_banner(server), flush=True)
    try:
        await server.serve_forever()
    finally:
        await server.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        asyncio.run(_serve(args.agent, args.host, args.port))
    except KeyboardInterrupt:
        print("\nstudio stopped.", flush=True)
    except ImportError as exc:
        # The one dependency the studio cannot do without; say so plainly
        # instead of unwinding a traceback at the user.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: could not bind {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
