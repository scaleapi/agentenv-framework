"""``python -m agentenv_protocol.yatf schema`` prints the strict JSON schema, as ``yatf-1.schema.json`` is checked in."""
from __future__ import annotations

import argparse
import sys

from .schema import schema_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentenv_protocol.yatf")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schema", help="print the strict JSON schema of one yatf/1 event")
    parser.parse_args(argv)
    sys.stdout.write(schema_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
