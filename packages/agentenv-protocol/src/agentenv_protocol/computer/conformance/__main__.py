"""``python -m agentenv_protocol.computer.conformance <driver url>``: probe a running driver, print each check, and
exit 1 when one failed or 2 when the driver couldn't be reached. ``--token`` is its control token, by default
``CONTROL_TOKEN_ENV``'s."""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from ...types import CONTROL_TOKEN_ENV
from . import run_probe

_MARK = {"passed": "PASS", "failed": "FAIL", "unverified": "----", "not_applicable": "n/a "}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agentenv_protocol.computer.conformance",
                                     description="Probe a running computer driver against the contract.")
    parser.add_argument("url", help="where the driver serves its card, such as http://localhost:18765")
    parser.add_argument("--level", choices=("basic", "full"), default="full")
    parser.add_argument("--json", type=Path, help="also write the report to this file")
    parser.add_argument("--timeout", type=float, default=60, help="seconds for each call, and to report ready")
    parser.add_argument("--token", default=os.environ.get(CONTROL_TOKEN_ENV),
                        help=f"the driver's control token, if it was given one (default: ${CONTROL_TOKEN_ENV})")
    args = parser.parse_args(argv)
    try:
        report = asyncio.run(run_probe(args.url, args.level, timeout_s=args.timeout, ready_timeout_s=args.timeout,
                                       token=args.token))
    except Exception as e:
        print(f"couldn't probe {args.url}: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    for result in report.checks:
        detail = f": {result.detail}" if result.detail else ""
        print(f"{_MARK[result.status]} {result.check} {result.name}{detail}")
    counts = report.counts()
    print(f"{report.driver}: {counts['passed']} passed, {counts['failed']} failed, {counts['unverified']} unverified, "
          f"{counts['not_applicable']} not applicable at {report.level}")
    for uri, verdict in report.extensions.items():
        print(f"  {verdict:10} {uri}")
    if args.json:
        args.json.write_text(report.model_dump_json(indent=1))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
