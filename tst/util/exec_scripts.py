"""What a recording fake sandbox saw a ``sudo bash -c`` exec run, and what that script does when bash runs it."""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path


def script_run(args: tuple[str, ...]) -> str:
    """The script of ``("sudo", "bash", "-c", script, name, *params)``, each ``"$n"`` replaced by its param."""
    script = args[3]
    for n, param in enumerate(args[5:], 1):
        script = script.replace(f'"${n}"', param)
    return script


def run_with_stubs(script: str, directory: Path, *commands: str) -> dict[str, list[list[str]]]:
    """Run ``script`` under bash in ``directory``, each of ``commands`` a stub on PATH that only records its arguments,
    and return each command's calls."""
    bin_dir = directory / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in commands:
        log = shlex.quote(str(directory / f"{name}.log"))
        (bin_dir / name).write_text(f'#!/bin/sh\nfor a in "$@"; do printf "%s\\n" "$a"; done >> {log}\necho --- >> {log}\n')
        (bin_dir / name).chmod(0o755)
    subprocess.run(["bash", "-c", script], cwd=directory, check=True,
                   env={**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"})
    return {name: _calls(directory / f"{name}.log") for name in commands}


def _calls(log: Path) -> list[list[str]]:
    return [call.strip("\n").split("\n") for call in log.read_text().split("---\n")[:-1]] if log.exists() else []
