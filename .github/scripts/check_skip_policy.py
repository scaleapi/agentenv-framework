"""Fail a CI job whose JUnit report contains a skip the job did not declare.

A skip is allowed only when its reason is ``agentenv-capability-missing: <name>`` for a
capability passed with ``--allow``; any other skip, or a skip under a ``--no-skips-under``
path, fails the job. A report with no test cases fails too. An ``xfail`` is an expected
failure the test declares, not a skip, and is ignored.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET

PREFIX = "agentenv-capability-missing: "


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit", help="JUnit XML report written by pytest --junitxml")
    parser.add_argument("--allow", nargs="*", default=[], metavar="CAPABILITY")
    parser.add_argument("--no-skips-under", nargs="*", default=[], metavar="PATH_PREFIX")
    args = parser.parse_args()

    cases = list(ET.parse(args.junit).getroot().iter("testcase"))
    if not cases:
        print("skip policy: the report has no test cases; nothing ran")
        return 1

    problems = []
    skips = 0
    for case in cases:
        skipped = case.find("skipped")
        if skipped is None or skipped.get("type") == "pytest.xfail":
            continue
        skips += 1
        path = case.get("classname", "").replace(".", "/")
        node = f"{path}::{case.get('name')}"
        reason = skipped.get("message") or ""
        capability = reason[len(PREFIX):].strip() if reason.startswith(PREFIX) else None
        if any(path.startswith(prefix.rstrip("/")) for prefix in args.no_skips_under):
            problems.append(f"{node}: skips are not allowed here ({reason!r})")
        elif capability is None:
            problems.append(f"{node}: reason is not a declared capability gap ({reason!r})")
        elif capability not in args.allow:
            problems.append(f"{node}: capability {capability!r} is not allowed in this job")

    for problem in problems:
        print(f"skip policy: {problem}")
    print(f"skip policy: {len(cases)} tests, {skips} skipped, {len(problems)} violations")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
