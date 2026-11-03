"""The front door of a computer app that runs alone, with no computer around it, as a website browser does:
``python -m agentenv_protocol.computer.front_door <folder> <file>:<Class> [--site-proxy COMMAND]``, run from the app's
venv as its container's command. It runs the app's own server (``serve_app``) in ``<folder>``, and the site proxy
command when it's given one, and starts each again whenever it ends, after a delay that grows while it keeps ending
soon after it starts. On ``PORT`` it serves the app's card, with ``computer-ready/v1`` added and no interface for
agents, since nothing here serves them yet; ``computer-ready/v1``'s ``status``, which says whether the app is ready, or
``failed`` for good once its launch log says it ended on its own, since nothing launches it again; and every other
request, sent on to the app's server at its path, its answer streamed back. Everything but the card takes the env's
control token (``CONTROL_TOKEN_ENV``), which no request it sends on and none of the programs it runs is given. Needs
the ``app`` extra."""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
import shlex
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from ..agent_env_environment import _error_response, _json_response, control_guard
from ..types import CONTROL_TOKEN_ENV, RPC_PATH, WELL_KNOWN_PATH, EnvironmentCard
from . import apps
from .app import SOCKET
from .app_runtime import LAUNCHES
from .errors import DeliveryFailed
from .operations import COMPUTER_READY, ReadyCheck

logger = logging.getLogger(__name__)

#: The env's one port, where the front door serves.
PORT = 8000
#: Where it serves ``computer-ready/v1``'s status, as a driver does.
STATUS_PATH = f"{RPC_PATH}/ext/computer-ready/status"
#: The delays before a program that ended starts again: each in turn while it keeps ending within ``STABLE_S`` of
#: starting, then the last, and the first again once a run has lasted that long.
RESTART_DELAYS_S = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
STABLE_S = 60.0
#: How long a program asked to stop may take before it's killed.
STOP_TIMEOUT_S = 10.0
#: The headers of one hop, and the control token's, which no request or answer it sends on carries.
UNFORWARDED = frozenset({"authorization", "connection", "host", "keep-alive", "proxy-authenticate",
                         "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade"})
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]
READY_EXTENSION = {
    "uri": COMPUTER_READY,
    "description": "Whether the app running alone is ready, by its own checks and its front door's, or failed for "
                   "good, and why.",
    "required": False,
    "params": {"methods": {"status": {"method": "GET", "endpoint": STATUS_PATH,
                                      "request": {"type": "object", "properties": {}}}}},
}
USAGE = "python -m agentenv_protocol.computer.front_door <folder> <file>:<Class> [--site-proxy COMMAND] [--port PORT]"


class Program:
    """A command the front door runs in ``cwd``, started again whenever it ends, until it's stopped."""

    def __init__(self, name: str, argv: list[str], cwd: Path) -> None:
        self.name, self.argv, self.cwd = name, argv, cwd
        self.process: Optional[asyncio.subprocess.Process] = None
        self.stopping = False

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def run(self) -> None:
        """Run it, and start it again each time it ends, with the environment the front door has."""
        ending_soon = 0
        while not self.stopping:
            started = time.monotonic()
            try:
                self.process = await asyncio.create_subprocess_exec(*self.argv, cwd=self.cwd,
                                                                    stdin=asyncio.subprocess.DEVNULL)
                how = _how(await self.process.wait())
            except OSError as e:
                how = f"couldn't start: {e}"
            except asyncio.CancelledError:
                if self.running:
                    self.process.kill()
                raise
            if self.stopping:
                return
            ran = time.monotonic() - started
            if ran >= STABLE_S:
                ending_soon = 0
            delay = RESTART_DELAYS_S[min(ending_soon, len(RESTART_DELAYS_S) - 1)]
            ending_soon += 1
            logger.warning(f"{self.name} ended after {ran:.1f} s ({how}); starting it again in {delay:g} s")
            await asyncio.sleep(delay)

    async def stop(self) -> None:
        """Ask it to end, and kill it after ``STOP_TIMEOUT_S``; it isn't started again."""
        self.stopping = True
        if not self.running:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), STOP_TIMEOUT_S)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()


class FrontDoor:
    """A computer app alone: the app's server in ``folder``, serving the class ``served`` names, and the site proxy
    command, the programs it runs; and what it serves."""

    def __init__(self, folder: Path, served: str, site_proxy: Optional[str] = None) -> None:
        self.folder = folder
        self.name = os.environ.get("ENVIRONMENT_NAME", folder.name)
        self.server = Program("the app's server", [sys.executable, "-m", "agentenv_protocol.computer.serve_app", served],
                              folder)
        self.site_proxy = Program("the site proxy", shlex.split(site_proxy), folder) if site_proxy else None
        self.programs = [self.server, *([self.site_proxy] if self.site_proxy else [])]
        self.http: Optional[httpx.AsyncClient] = None

    def app(self) -> Starlette:
        """What it serves; its guard takes the control token the environment has now."""
        guard = control_guard()
        return Starlette(routes=[Route(WELL_KNOWN_PATH, self.card, methods=["GET"]),
                                 Route(STATUS_PATH, guard(self.status), methods=["GET"]),
                                 Route("/{path:path}", guard(self.forward), methods=METHODS)],
                         lifespan=self.lifespan)

    @contextlib.asynccontextmanager
    async def lifespan(self, app: Starlette) -> AsyncIterator[None]:
        self.http = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(self.folder / SOCKET)),
                                      base_url="http://app", timeout=None)
        runs = [asyncio.ensure_future(program.run()) for program in self.programs]
        try:
            yield
        finally:
            await asyncio.gather(*(program.stop() for program in self.programs))
            for run in runs:
                run.cancel()
            await self.http.aclose()

    async def card(self, request: Request) -> Response:
        """The app's card as its server serves it, with no interface for agents and with ``computer-ready/v1``; while
        its server isn't answering, its name and ``computer-ready/v1`` alone."""
        card = await apps.served_card(self.folder) or EnvironmentCard(name=self.name, url=RPC_PATH).model_dump()
        card["additionalInterfaces"] = []
        capabilities = card.get("capabilities") or {}
        capabilities["extensions"] = [*(capabilities.get("extensions") or []), READY_EXTENSION]
        card["capabilities"] = capabilities
        return _json_response(card)

    async def status(self, request: Request) -> Response:
        return _json_response(await self.readiness())

    async def readiness(self) -> dict[str, Any]:
        """``computer-ready/v1``'s status: the app's readiness as its server says, or why its server isn't answering,
        and whether the site proxy runs; ``failed`` too, once the app can't be ready again."""
        checks = [await apps.check(self.folder)]
        if self.site_proxy is not None:
            checks.append(ReadyCheck("site proxy", self.site_proxy.running,
                                     "running" if self.site_proxy.running else "not running; it starts again"))
        failed = self.failure()
        if failed is not None:
            checks[0] = ReadyCheck(checks[0].name, False, failed)
        return {"ready": failed is None and all(check.ok for check in checks),
                "checks": [asdict(check) for check in checks], **({"failed": failed} if failed else {})}

    def failure(self) -> Optional[str]:
        """Why the app can't be ready again: its launch log says it ended on its own, and alone nothing launches it
        again, an infrastructure failure."""
        for entry in _launch_log(self.folder):
            if entry.get("event") == "exit":
                how = (f"by signal {entry['signal']}" if "signal" in entry else
                       f"with code {entry['code']}" if "code" in entry else "in a way that wasn't recorded")
                return (f"it ended on its own at {entry.get('at')}, {how}, and nothing launches it again: an "
                        "infrastructure failure")
        return None

    async def forward(self, request: Request) -> Response:
        """``request`` sent on to the app's server at its path, but for its hop and token headers, and its answer
        streamed back; it takes as long as the app does."""
        sent = self.http.build_request(
            request.method, httpx.URL(path=request.url.path, query=request.url.query.encode()),
            headers=_forwarded(request.headers.items()), content=await request.body())
        try:
            answer = await self.http.send(sent, stream=True)
        except (httpx.HTTPError, OSError) as e:
            return _error_response(DeliveryFailed(f"the app's server isn't answering: {type(e).__name__}: {e}"))
        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in answer.aiter_raw():
                    yield chunk
            finally:
                await answer.aclose()

        response = StreamingResponse(body(), status_code=answer.status_code)
        response.raw_headers = [(name.encode("latin-1"), value.encode("latin-1"))
                                for name, value in _forwarded(answer.headers.multi_items())]
        return response


def _forwarded(headers: Any) -> list[tuple[str, str]]:
    """``headers`` but those of one hop: the standard ones, the control token's and any its ``Connection`` names."""
    headers = list(headers)
    named = {token.strip().lower() for name, value in headers if name.lower() == "connection"
             for token in value.split(",")}
    return [(name, value) for name, value in headers if name.lower() not in UNFORWARDED | named]


def _launch_log(folder: Path) -> list[dict]:
    """Each entry of the app's launch log that's whole, in order."""
    try:
        lines = (folder / LAUNCHES).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    entries = []
    for line in lines:
        with contextlib.suppress(ValueError):
            entries.append(json.loads(line))
    return entries


def _how(code: int) -> str:
    return f"signal {signal.Signals(-code).name}" if code < 0 else f"code {code}"


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="front_door", usage=USAGE)
    parser.add_argument("folder", type=Path)
    parser.add_argument("served")
    parser.add_argument("--site-proxy")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s front door: %(message)s")
    folder = args.folder.resolve()
    os.environ.setdefault("ENVIRONMENT_NAME", folder.name)
    served = FrontDoor(folder, args.served, args.site_proxy).app()
    os.environ.pop(CONTROL_TOKEN_ENV, None)
    uvicorn.run(served, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main(sys.argv[1:])
