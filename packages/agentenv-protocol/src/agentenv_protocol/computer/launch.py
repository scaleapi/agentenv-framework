"""Launch a computer app through its own server, as its menu launcher does:
``python -m agentenv_protocol.computer.launch <app folder>`` asks the server on the folder's socket to launch it, so
the server starts every copy of the app."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

from ..types import WELL_KNOWN_PATH
from .app import SOCKET
from .apps import endpoint


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m agentenv_protocol.computer.launch <app folder>")
    with httpx.Client(transport=httpx.HTTPTransport(uds=str(Path(argv[0]) / SOCKET)), base_url="http://app") as http:
        card = http.get(WELL_KNOWN_PATH).raise_for_status().json()
        http.post(endpoint(card, "launch"), json={}).raise_for_status()


if __name__ == "__main__":
    main(sys.argv[1:])
