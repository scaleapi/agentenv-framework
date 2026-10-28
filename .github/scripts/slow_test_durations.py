"""Rewrite tst/integration/slow_durations.json from the slow tier's JUnit reports, so the CI shards stay balanced.

    gh run download <run id> --pattern 'junit-integration-local-*'
    python .github/scripts/slow_test_durations.py junit-integration-local-*/junit-integration-slow.xml

Use a green ``main`` run: it has a ``junit-integration-local-*`` artifact for every ``integration-local-slow`` and
``integration-local-gateway`` shard, which ``gh run download`` puts in a folder each. Each test's time counts toward its file,
or toward the test itself for a file the table names in ``split_by_test``; that list is kept as it is.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "tst" / "integration" / "slow_durations.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", nargs="+", help="JUnit XML reports written by pytest --junitxml")
    parser.add_argument("--table", type=Path, default=TABLE, help="the durations table to rewrite")
    parser.add_argument("--root", type=Path, default=ROOT, help="the checkout the reports' test files are in")
    args = parser.parse_args()

    table = json.loads(args.table.read_text())
    split = set(table.get("split_by_test", []))
    seconds: dict[str, float] = defaultdict(float)
    for report in args.reports:
        for case in ET.parse(report).getroot().iter("testcase"):
            nodeid = _nodeid(case.get("classname", ""), case.get("name", ""), args.root)
            if nodeid is None:
                print(f"slow test durations: no test file for {case.get('classname')!r}; left out")
                continue
            path = nodeid.split("::", 1)[0]
            seconds[nodeid if path in split else path] += float(case.get("time") or 0)
    if not seconds:
        print("slow test durations: the reports have no test cases; the table is unchanged")
        return 1
    table["seconds"] = {unit: round(value, 1) for unit, value in sorted(seconds.items())}
    args.table.write_text(json.dumps(table, indent=1) + "\n")
    print(f"slow test durations: {len(seconds)} units, {sum(seconds.values()) / 60:.1f} min, written to {args.table}")
    return 0


def _nodeid(classname: str, name: str, root: Path) -> str | None:
    """The pytest node id of a JUnit test case: the longest dotted prefix of its classname that is a test
    file, then any classes, then the test's name."""
    parts = classname.split(".")
    for cut in range(len(parts), 0, -1):
        path = "/".join(parts[:cut]) + ".py"
        if (root / path).is_file():
            return "::".join([path, *parts[cut:], name])
    return None


if __name__ == "__main__":
    sys.exit(main())
