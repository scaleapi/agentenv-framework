"""Launch a computer app through its own server, as its menu launcher does:
``python -m agentenv_protocol.computer.launch <app folder>`` asks the folder's server (``served``) to launch it, so the
server starts every copy of the app."""
from __future__ import annotations

import sys
from pathlib import Path

from ..types import WELL_KNOWN_PATH
from .apps import endpoint
from .served import served


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m agentenv_protocol.computer.launch <app folder>")
    with served(Path(argv[0])).sync_client(timeout=5.0) as http:
        card = http.get(WELL_KNOWN_PATH).raise_for_status().json()
        http.post(endpoint(card, "launch"), json={}).raise_for_status()


if __name__ == "__main__":
    main(sys.argv[1:])
