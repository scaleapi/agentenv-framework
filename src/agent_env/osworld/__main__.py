"""``python -m agent_env.osworld stage --source <guest server source> --out <folder>``: the ``osworld-server`` app's
folder with the guest server's source, for ``agent-env env put``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_env.osworld import stage_server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent_env.osworld")
    commands = parser.add_subparsers(dest="command", required=True)
    stage = commands.add_parser("stage", help="stage the osworld-server app with the guest server's source")
    stage.add_argument("--source", required=True, type=Path, help="the folder holding the guest server's main.py")
    stage.add_argument("--out", required=True, type=Path, help="the folder to write, which must not exist")
    args = parser.parse_args(argv)
    try:
        out = stage_server(args.source, args.out)
    except (FileNotFoundError, FileExistsError) as e:
        print(e, file=sys.stderr)
        return 1
    print(f"Staged {out}; put it with: agent-env env put {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
