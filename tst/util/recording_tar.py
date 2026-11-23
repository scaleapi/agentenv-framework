"""A stand-in for ``tar`` that records each run's arguments and whether it was started on real time (agent-env's
libfaketime marker in its environment), and writes the archive and snapshot files it's named, empty."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPT = """#!{python}
import json, os, sys
args = sys.argv[1:]
with open({log!r}, "a") as log:
    log.write(json.dumps({{"real_time": os.environ.get("AGENTENV_REAL_TIME"), "args": args}}) + "\\n")
for arg, value in zip(args, args[1:] + [None]):
    named = arg.partition("=")[2] if arg.startswith("--listed-incremental=") else value if arg in ("-cf", "-czf") else None
    if named and named not in ("-", "/dev/null"):
        open(named, "a").close()
"""


def install(directory: Path) -> Path:
    """A ``tar`` in ``directory``, to put first on PATH; returns the log it adds a line to per run."""
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / "tar-runs.jsonl"
    tar = directory / "tar"
    tar.write_text(_SCRIPT.format(python=sys.executable, log=str(log)))
    tar.chmod(0o755)
    return log


def incremental_runs(log: Path) -> list[dict]:
    """The recorded runs that read or wrote an incremental snapshot."""
    runs = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return [run for run in runs if any(arg.startswith("--listed-incremental=") for arg in run["args"])]
