"""Fail unless a built agentenv-framework wheel carries the explorer's web UI: ``agent_env/explorer/static/index.html``
and every ``/_next/`` script, stylesheet and font it references, so ``agent-env up`` serves a page that loads.

Usage: ``python .github/scripts/check_explorer_ui.py WHEEL``. Run by the release and by the ``ui`` job on pull requests.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import unquote

STATIC = "agent_env/explorer/static/"
_ASSET = re.compile(r'(?:src|href)="/(_next/[^"?#]+)')


def ui_problems(wheel: Path) -> list[str]:
    """What the wheel's explorer UI is missing; empty when index.html and every asset it references are there."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        if STATIC + "index.html" not in names:
            return [f"{STATIC}index.html is missing"]
        # The catch-all page's chunk is referenced percent-encoded (%5B%5B...slug%5D%5D) but stored as [[...slug]].
        assets = sorted({unquote(a) for a in _ASSET.findall(archive.read(STATIC + "index.html").decode())})
    if not assets:
        return [f"{STATIC}index.html references no /_next/ assets"]
    return [f"{STATIC}{asset} is missing" for asset in assets if STATIC + asset not in names]


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: check_explorer_ui.py WHEEL", file=sys.stderr)
        return 2
    problems = ui_problems(Path(argv[0]))
    for problem in problems:
        print(f"::error::the wheel's explorer UI is incomplete: {problem}")
    if not problems:
        print(f"{argv[0]}: the explorer UI's index.html and every asset it references are in the wheel")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
