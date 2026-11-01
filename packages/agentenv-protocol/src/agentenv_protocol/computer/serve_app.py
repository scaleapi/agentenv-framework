"""Serve a computer app's class from its file, as its computer's program starts it:
``python -m agentenv_protocol.computer.serve_app <file>:<Class>``, run in the app's folder, serves the class's card and
``computer-app/v1`` on the Unix socket ``.agentenv/server.sock`` there. Before it serves, it keeps a copy of the app's
data paths as installed, which a reset puts back; once it serves, it adds the app's launcher and launches a boot app
(``AppRuntime.start``). A server that can't start, or whose start fails once it serves,
writes why to ``.agentenv/server.failure``, which its computer's driver reports, and exits. Needs the ``app`` extra."""
from __future__ import annotations

import asyncio
import contextlib
import sys
import traceback
from pathlib import Path

import uvicorn

from ._load import load_class
from .app import FAILURE, SOCKET, ComputerAppEnvironment

USAGE = "usage: python -m agentenv_protocol.computer.serve_app <file>:<Class>"


def main(argv: list[str]) -> None:
    FAILURE.unlink(missing_ok=True)
    try:
        _serve(argv)
    except (Exception, SystemExit) as e:
        FAILURE.write_text(traceback.format_exception_only(e)[-1].strip())
        raise


def _serve(argv: list[str]) -> None:
    """Serve the class ``argv`` names, starting the app once it serves; a start that fails stops the server and is
    raised once it has."""
    if len(argv) != 1 or ":" not in argv[0]:
        raise SystemExit(USAGE)
    file, _, name = argv[0].partition(":")
    app = load_class(Path(file), name, ComputerAppEnvironment)()
    runtime = app._computer_app_runtime()
    app._computer_app_data_plane().keep_initial()
    asgi = app.create_app().streamable_http_app()
    server = uvicorn.Server(uvicorn.Config(asgi, uds=str(SOCKET), log_level="warning"))
    serving, failed = asgi.router.lifespan_context, []

    def stop(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is not None:
            failed.append(task.exception())
            server.should_exit = True

    @contextlib.asynccontextmanager
    async def lifespan(scope):
        async with serving(scope):
            started = asyncio.ensure_future(runtime.start())
            started.add_done_callback(stop)
            yield
            started.cancel()

    asgi.router.lifespan_context = lifespan
    SOCKET.unlink(missing_ok=True)
    server.run()
    if failed:
        raise failed[0]


if __name__ == "__main__":
    main(sys.argv[1:])
