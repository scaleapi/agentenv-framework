"""``python -m agentenv_protocol.atif validate FILE [--media DIR] [--reader]``: check an ATIF document, printing each
finding; exit 1 when any is an error, 2 when the file can't be read."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .validator import errors, parse, validate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentenv_protocol.atif",
                                     description="Check ATIF documents against ATIF and the agentenv profile.")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="check one document; exit 1 when it has an error")
    check.add_argument("file", type=Path, help="the trajectory.json to check")
    check.add_argument("--media", type=Path, metavar="DIR",
                       help="the folder the document's media paths are relative to, which holds media/; "
                            "given, each file is checked against its name and profile")
    check.add_argument("--reader", action="store_true",
                       help="check someone else's document for reading: what a producer must not write warns")
    args = parser.parse_args(argv)
    try:
        document, findings = parse(args.file)
    except OSError as exc:
        print(f"can't read {args.file}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    if not findings:
        findings = validate(document, reader=args.reader, media=args.media)
    for finding in findings:
        print(finding)
    failed = errors(findings)
    print(f"{args.file}: {'invalid' if failed else 'valid'}, {len(failed)} errors, "
          f"{len(findings) - len(failed)} warnings")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
