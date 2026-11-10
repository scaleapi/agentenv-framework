"""The ``osworld-server`` computer app: the OSWorld guest server, which a CUA VM runs on port 5000, run on a computer
as the session user from the source staged under ``payload/`` (``agent_env.osworld.stage_server``), so the
``osworld_*`` steps set up and grade OSWorld tasks there. Its own server forwards ``/osworld/<route>`` to the guest
server, and its card advertises that prefix as the endpoint of the ``osworld-guest/v1`` extension, which the computer
roots where it serves the app and guards with the env's control token. The OSWorld controller, deployed beside the
computer by ``osworld_deploy_controller``, reads the endpoint from the computer's card; an agent can't reach it there.

This file runs on the computer, in the app's folder: it imports only what the app's server has."""
from __future__ import annotations

import socket
from typing import Any

import httpx
from agentenv_protocol import extension
from agentenv_protocol.computer import (ComputerAppEnvironment, Packages, ReadyCheck, Service, computer_app, linux,
                                        ready_check)
from starlette.requests import Request
from starlette.responses import Response

GUEST_EXTENSION = "urn:agentenv:osworld-guest/v1"
PREFIX = "/osworld"
HOST, PORT = "127.0.0.1", 5000
UPSTREAM = f"http://{HOST}:{PORT}"
METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE"]
#: Setup downloads task assets and evaluation runs long checks; the controller allows each up to 1800 s.
TIMEOUT_S = 1800.0
#: Not forwarded: the hop's own headers, and the control token, which is the computer's.
_DROPPED_REQUEST = frozenset({"host", "content-length", "connection", "transfer-encoding", "authorization"})
_DROPPED_RESPONSE = frozenset({"content-length", "connection", "transfer-encoding", "content-encoding"})
#: What a CUA VM installs for the guest server that Ubuntu 24.04's desktop computer lacks: AT-SPI's Python bindings
#: for its accessibility tree, Tk for pyautogui, the window and input tools its setup routes run, ffmpeg for recordings.
PACKAGES = ["python3-pyatspi", "python3-gi", "gir1.2-atspi-2.0", "at-spi2-core", "python3-tk", "ffmpeg", "wmctrl",
            "xdotool", "x11-xkb-utils", "x11-xserver-utils"]


async def forward(request: Request, client: httpx.AsyncClient, upstream: str = UPSTREAM) -> Response:
    """``request`` under ``PREFIX``, sent to the guest server with its method, query and body, and its answer
    returned as the guest server gave it."""
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROPPED_REQUEST}
    try:
        answer = await client.request(request.method, f"{upstream}/{request.path_params.get('path', '')}",
                                      params=list(request.query_params.multi_items()), headers=headers,
                                      content=await request.body(), timeout=TIMEOUT_S)
    except httpx.HTTPError as e:
        return Response(f"osworld-server unreachable: {type(e).__name__}", status_code=502)
    return Response(answer.content, status_code=answer.status_code,
                    headers={k: v for k, v in answer.headers.items() if k.lower() not in _DROPPED_RESPONSE})


def guest_server_check() -> ReadyCheck:
    try:
        with socket.create_connection((HOST, PORT), timeout=2):
            return ReadyCheck("osworld-server", True)
    except OSError as e:
        return ReadyCheck("osworld-server", False, f"{HOST}:{PORT}: {e}")


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
@linux(packages=Packages(apt=PACKAGES), python="requirements.txt",
       services=[Service(name="osworld-server", command="start-osworld-server.sh")])
class OSWorldServer(ComputerAppEnvironment):
    """The OSWorld guest server, forwarded under ``/osworld/`` to the controller beside the computer."""

    def create_app(self) -> Any:
        app = super().create_app()
        client = httpx.AsyncClient()

        async def _forward(request: Request) -> Response:
            return await forward(request, client)

        app.custom_route(f"{PREFIX}/{{path:path}}", methods=METHODS)(_forward)
        return app

    @extension(GUEST_EXTENSION, method="GET", path=PREFIX,
               description="The OSWorld guest server's routes, under this endpoint; a GET says whether it's up.")
    async def guest(self) -> dict:
        check = guest_server_check()
        return {"ready": check.ok, "detail": check.detail}

    @ready_check(os="linux")
    def guest_server(self) -> ReadyCheck:
        return guest_server_check()
