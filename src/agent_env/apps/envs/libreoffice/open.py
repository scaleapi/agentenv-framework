"""Open files in LibreOffice through the app's own server, as every LibreOffice launcher on the computer does (its
install step points them here): ``open.py [--writer|--impress|--calc|...] [FILE...]``.

The server stops and relaunches the app around a load and a reset, and only the copy it started, so this never runs
``soffice`` itself: it leaves what it opens in ``OPENS`` and has the server launch the app, whose launch hands it to the
copy that runs, LibreOffice running one per user, in the turn a load or a reset takes too, so neither stops that copy
in between. With none running, the server starts one, and its launch once that's ready hands it what waits. A CSV opens
comma-separated in UTF-8, without Calc's text import dialog."""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from agentenv_protocol.computer.apps import endpoint
from agentenv_protocol.computer.served import served
from agentenv_protocol.types import WELL_KNOWN_PATH

FOLDER = Path(__file__).resolve().parent
#: Where an open waits for the server's launch: a JSON list of ``soffice`` argument lists, named so the oldest sorts
#: first.
OPENS = FOLDER / ".agentenv" / "opens"
#: Comma-separated, double-quoted, UTF-8 (76), from the first line: what a CSV the agent opens most likely is.
CSV = "--infilter=CSV:44,34,76,1"
READY_S = 60.0


def invocations(argv: list[str]) -> list[list[str]]:
    """The ``soffice`` arguments ``argv`` opens with, each file by its absolute path, as the server's folder isn't
    this one's: CSV files on their own, with ``CSV``'s filter, after the rest."""
    argv = [arg if arg.startswith("-") or "://" in arg else os.path.abspath(arg) for arg in argv]
    csv = [arg for arg in argv if arg.lower().endswith(".csv")]
    rest = [arg for arg in argv if arg not in csv]
    return ([rest] if rest or not csv else []) + ([[CSV, *csv]] if csv else [])


def waiting(calls: list[list[str]]) -> Path:
    """``calls`` left for the server's launch, written beside their place and renamed into it, so it never reads half
    of them."""
    OPENS.mkdir(parents=True, exist_ok=True)
    request = OPENS / f"{time.time_ns()}-{uuid.uuid4().hex}.json"
    staged = request.with_suffix(".staged")
    staged.write_text(json.dumps(calls), encoding="utf-8")
    staged.replace(request)
    return request


def handed_on(request: Optional[Path], timeout_s: float = READY_S) -> None:
    """The server's launch of the app, again once it's ready while ``request`` still waits, as it does when the launch
    had to start the app."""
    with served(FOLDER).sync_client(timeout=timeout_s) as http:
        card = http.get(WELL_KNOWN_PATH).raise_for_status().json()
        deadline = time.monotonic() + timeout_s
        while True:
            http.post(endpoint(card, "launch"), json={}).raise_for_status()
            if request is None or not request.exists():
                return
            while not http.get(endpoint(card, "ready")).raise_for_status().json().get("ready"):
                if time.monotonic() > deadline:
                    raise SystemExit(f"LibreOffice wasn't ready within {timeout_s:g} s of its server launching it")
                time.sleep(0.25)
            if time.monotonic() > deadline:
                raise SystemExit(f"LibreOffice's server didn't hand it {request.name} within {timeout_s:g} s")


def main(argv: list[str]) -> None:
    request = waiting(invocations(argv)) if argv else None
    try:
        handed_on(request)
    finally:
        if request is not None:
            request.unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[1:])
