"""Open files in LibreOffice through the app's own server, as every LibreOffice launcher on the computer does (its
install step points them here): ``open.py [--writer|--impress|--calc|...] [FILE...]``.

The server stops and relaunches the app around a load and a reset, and only the copy it started, so this never starts
one itself: with none running it has the server launch it and waits until it's ready; then ``soffice`` hands the files,
or the window to open, to that copy, which LibreOffice runs one of per user. A CSV opens comma-separated in UTF-8,
without Calc's text import dialog."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from agentenv_protocol.computer.apps import endpoint
from agentenv_protocol.computer.served import served
from agentenv_protocol.types import WELL_KNOWN_PATH

FOLDER = Path(__file__).resolve().parent
SOFFICE = "/usr/lib/libreoffice/program/soffice"
#: Comma-separated, double-quoted, UTF-8 (76), from the first line: what a CSV the agent opens most likely is.
CSV = "--infilter=CSV:44,34,76,1"
READY_S = 60.0


def running() -> bool:
    """Whether the session user's LibreOffice runs."""
    return subprocess.run(["pgrep", "-u", str(os.getuid()), "-x", "soffice.bin"], capture_output=True).returncode == 0


def launched_by_its_server(timeout_s: float = READY_S) -> None:
    """The app launched by its server, once it's ready."""
    with served(FOLDER).sync_client(timeout=10.0) as http:
        card = http.get(WELL_KNOWN_PATH).raise_for_status().json()
        http.post(endpoint(card, "launch"), json={}).raise_for_status()
        deadline = time.monotonic() + timeout_s
        while not http.get(endpoint(card, "ready")).raise_for_status().json().get("ready"):
            if time.monotonic() > deadline:
                raise SystemExit(f"LibreOffice wasn't ready within {timeout_s:g} s of its server launching it")
            time.sleep(0.25)


def invocations(argv: list[str]) -> list[list[str]]:
    """The ``soffice`` arguments ``argv`` opens with: CSV files on their own, with ``CSV``'s filter, after the rest."""
    csv = [arg for arg in argv if arg.lower().endswith(".csv")]
    rest = [arg for arg in argv if arg not in csv]
    return ([rest] if rest or not csv else []) + ([[CSV, *csv]] if csv else [])


def main(argv: list[str]) -> None:
    if not running():
        launched_by_its_server()
        if not argv:
            return  # the Start Center the server's launch shows is what was asked for
    calls = invocations(argv)
    for arguments in calls[:-1]:
        subprocess.run([SOFFICE, *arguments], check=False)
    os.execv(SOFFICE, [SOFFICE, *calls[-1]])


if __name__ == "__main__":
    main(sys.argv[1:])
