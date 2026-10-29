"""Launch a computer app through its own server, as its menu launcher does:
``python -m agentenv_protocol.computer.launch <app folder>`` asks the server on the folder's socket to launch it, so
the server starts every copy of the app."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

from ..types import RPC_PATH
from .app import SOCKET


def main(argv: list[str]) -> None:
    if len(argv) != 1:
        raise SystemExit("usage: python -m agentenv_protocol.computer.launch <app folder>")
    with httpx.Client(transport=httpx.HTTPTransport(uds=str(Path(argv[0]) / SOCKET)), base_url="http://app") as http:
        http.post(f"{RPC_PATH}/ext/computer-app/launch", json={}).raise_for_status()


if __name__ == "__main__":
    main(sys.argv[1:])
