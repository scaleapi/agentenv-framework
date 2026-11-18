"""Where an app's own server answers and how its computer reaches it: on its folder's Unix socket, or on Windows on
the loopback port and token it wrote there, nothing answering till it has. On loopback it answers only a request or WebSocket carrying its
token, once, which it takes out before the app sees the request, and a client reaches it through no proxy the
environment names."""
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import httpx
import pytest

uvicorn = pytest.importorskip("uvicorn")

from starlette.applications import Starlette  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route, WebSocketRoute  # noqa: E402
from starlette.websockets import WebSocket  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402
from websockets.exceptions import InvalidStatus  # noqa: E402

import agentenv_protocol.computer.served as transports  # noqa: E402
from agentenv_protocol.computer.app import ADDRESS, SOCKET  # noqa: E402
from agentenv_protocol.computer.served import TOKEN_HEADER, Served, bind, guarded, served  # noqa: E402


async def _headers(request: Request) -> JSONResponse:
    return JSONResponse(dict(request.headers.items()))


async def _hello(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_text("hello")
    await websocket.close()


@contextlib.asynccontextmanager
async def _serving(folder: Path):
    """An app answering its headers and a WebSocket hello, served on loopback with its token, as on Windows."""
    (folder / ADDRESS).parent.mkdir(parents=True, exist_ok=True)
    listening, token = bind(folder)
    app = Starlette(routes=[Route("/headers", _headers), WebSocketRoute("/hello", _hello)])
    server = uvicorn.Server(uvicorn.Config(guarded(app, token), log_level="warning", lifespan="off"))
    serving = asyncio.create_task(server.serve(sockets=[listening]))
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        yield token
    finally:
        server.should_exit = True
        await serving


def test_an_app_is_reached_on_its_socket_but_on_windows_none_answers_till_its_server_wrote_its_address(tmp_path,
                                                                                                       monkeypatch):
    monkeypatch.setattr(transports, "LOOPBACK", False)
    on_its_socket = served(tmp_path)
    monkeypatch.setattr(transports, "LOOPBACK", True)

    with pytest.raises(FileNotFoundError):
        served(tmp_path)
    assert on_its_socket == Served(socket=tmp_path / SOCKET)


@pytest.mark.asyncio
async def test_an_app_on_loopback_answers_its_token_alone_and_the_app_never_sees_it(tmp_path, monkeypatch):
    monkeypatch.setattr(transports, "LOOPBACK", True)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9")
    async with _serving(tmp_path) as token:
        reached = served(tmp_path)
        async with reached.client(timeout=5) as http:
            seen = (await http.get("/headers", headers={TOKEN_HEADER: "forged"})).json()
        port = reached.port
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False) as bare:
            missing = await bare.get("/headers")
            wrong = await bare.get("/headers", headers={TOKEN_HEADER: "nope"})
            twice = await bare.get("/headers", headers=[(TOKEN_HEADER, token), (TOKEN_HEADER, token)])
        async with connect(f"ws://127.0.0.1:{port}/hello", additional_headers=[(TOKEN_HEADER, token)],
                           proxy=None) as websocket:
            hello = await websocket.recv()
        with pytest.raises(InvalidStatus) as refused:
            async with connect(f"ws://127.0.0.1:{port}/hello", proxy=None):
                pass

    assert TOKEN_HEADER not in seen and seen["host"] == f"127.0.0.1:{port}"
    assert (missing.status_code, wrong.status_code, twice.status_code) == (403, 403, 403)
    assert hello == "hello" and refused.value.response.status_code == 403


def test_a_loopback_address_is_written_whole_with_the_port_listened_on(tmp_path):
    (tmp_path / ADDRESS).parent.mkdir()
    listening, token = bind(tmp_path)
    try:
        written = json.loads((tmp_path / ADDRESS).read_text())
        assert written == {"port": listening.getsockname()[1], "token": token} and len(token) >= 32
        assert listening.getsockname()[0] == "127.0.0.1"
        assert sorted(path.name for path in (tmp_path / ADDRESS).parent.iterdir()) == [ADDRESS.name]
    finally:
        listening.close()
