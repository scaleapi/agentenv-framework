"""Serve a computer app's class from its file, as its computer's program starts it:
``python -m agentenv_protocol.computer.serve_app <file>:<Class>``, run in the app's folder, serves the class's card and
``computer-app/v1`` on the Unix socket ``.agentenv/server.sock`` there. Once it serves, the app's server adds the app's
launcher and launches a boot app (``ComputerAppEnvironment._computer_app_start``). Needs the ``app`` extra."""
from __future__ import annotations

import asyncio
import contextlib
import sys
from pathlib import Path

import uvicorn

from ._load import load_class
from .app import SOCKET, ComputerAppEnvironment

USAGE = "usage: python -m agentenv_protocol.computer.serve_app <file>:<Class>"


def main(argv: list[str]) -> None:
    if len(argv) != 1 or ":" not in argv[0]:
        raise SystemExit(USAGE)
    file, _, name = argv[0].partition(":")
    app = load_class(Path(file), name, ComputerAppEnvironment)()
    asgi = app.create_app().streamable_http_app()
    serving = asgi.router.lifespan_context

    @contextlib.asynccontextmanager
    async def lifespan(scope):
        async with serving(scope):
            started = asyncio.ensure_future(app._computer_app_start())
            yield
            started.cancel()

    asgi.router.lifespan_context = lifespan
    SOCKET.unlink(missing_ok=True)
    uvicorn.run(asgi, uds=str(SOCKET), log_level="warning")


if __name__ == "__main__":
    main(sys.argv[1:])
