"""A slot's door, in front of a live upstream on loopback: it lets in only requests with the slot's token, takes the
token off them, and passes everything else through, streams and WebSockets included."""

from __future__ import annotations

import asyncio
import json
import socket
import time

import httpx
import pytest
import pytest_asyncio
import uvicorn
from pytest_socket import enable_socket
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from agent_env.env.gateway import door
from agent_env.env.gateway.constants import DOOR_ROUTES_ENV, DOOR_TOKEN_ENV, SLOT_TOKEN_HEADER

TOKEN = "the-slots-token"
STREAM_GAP_S = 0.6


async def _echo(request: Request) -> JSONResponse:
    return JSONResponse({"method": request.method, "path": request.url.path, "query": request.url.query,
                         "headers": dict(request.headers), "body": (await request.body()).decode()})


async def _stream(_: Request) -> StreamingResponse:
    async def chunks():
        yield b"data: first\n\n"
        await asyncio.sleep(STREAM_GAP_S)
        yield b"data: second\n\n"
    return StreamingResponse(chunks(), media_type="text/event-stream")


async def _ws(websocket: WebSocket) -> None:
    await websocket.accept(subprotocol=(websocket.scope.get("subprotocols") or [None])[0])
    await websocket.send_json({"headers": dict(websocket.headers)})
    async for message in websocket.iter_text():
        await websocket.send_text(f"echo {message}")


_UPSTREAM = Starlette(routes=[Route("/echo", _echo, methods=["GET", "POST"]), Route("/stream", _stream),
                              WebSocketRoute("/ws", _ws)])


async def _serve(app) -> tuple[uvicorn.Server, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(app, log_level="critical", access_log=False, lifespan="off")
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return server, sock.getsockname()[1]


@pytest_asyncio.fixture
async def door_port():
    enable_socket()  # loopback only
    upstream, upstream_port = await _serve(_UPSTREAM)
    async with door.door_client() as client:
        front, port = await _serve(door.Door(f"127.0.0.1:{upstream_port}", TOKEN, client))
        try:
            yield port
        finally:
            await front.shutdown()
            await upstream.shutdown()


def _token(value: str = TOKEN) -> dict[str, str]:
    return {SLOT_TOKEN_HEADER: value}


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, _token("wrong"), _token(TOKEN[:-1]), {"Authorization": f"Bearer {TOKEN}"}])
async def test_a_request_without_the_slots_token_gets_401(door_port, headers):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://127.0.0.1:{door_port}/echo", headers=headers)
    assert response.status_code == 401
    assert SLOT_TOKEN_HEADER in response.text and "Browser sign-in" in response.text


@pytest.mark.asyncio
async def test_the_token_twice_is_refused(door_port):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"http://127.0.0.1:{door_port}/echo",
                                    headers=[(SLOT_TOKEN_HEADER, TOKEN), (SLOT_TOKEN_HEADER, TOKEN)])
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_with_the_token_a_request_passes_without_it_and_with_everything_else(door_port):
    async with httpx.AsyncClient() as client:
        response = await client.post(f"http://127.0.0.1:{door_port}/echo?a=1&b=2", content=b"payload",
                                     headers={**_token(), "Authorization": "Bearer control", "Host": "shop.example.com",
                                              "X-Other": "kept"})
    assert response.status_code == 200
    seen = response.json()
    assert (seen["method"], seen["path"], seen["query"], seen["body"]) == ("POST", "/echo", "a=1&b=2", "payload")
    assert SLOT_TOKEN_HEADER.lower() not in seen["headers"]
    assert seen["headers"]["authorization"] == "Bearer control"
    assert seen["headers"]["host"] == "shop.example.com"
    assert seen["headers"]["x-other"] == "kept"


@pytest.mark.asyncio
async def test_a_stream_passes_as_it_comes(door_port):
    async with httpx.AsyncClient() as client:
        async with client.stream("GET", f"http://127.0.0.1:{door_port}/stream", headers=_token()) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            started = time.monotonic()
            chunks = []
            async for chunk in response.aiter_text():
                chunks.append((chunk, time.monotonic() - started))
    assert "".join(chunk for chunk, _ in chunks) == "data: first\n\ndata: second\n\n"
    assert chunks[0][1] < STREAM_GAP_S / 2


@pytest.mark.asyncio
async def test_a_websocket_is_relayed_without_the_token(door_port):
    async with connect(f"ws://127.0.0.1:{door_port}/ws", additional_headers={**_token(), "Authorization": "Bearer c"},
                       subprotocols=["mcp"]) as ws:
        assert ws.subprotocol == "mcp"
        hello = json.loads(await ws.recv())
        assert SLOT_TOKEN_HEADER.lower() not in hello["headers"]
        assert hello["headers"]["authorization"] == "Bearer c"
        await ws.send("ping")
        assert await ws.recv() == "echo ping"


@pytest.mark.asyncio
async def test_a_websocket_without_the_token_gets_401(door_port):
    with pytest.raises(InvalidStatus) as refused:
        async with connect(f"ws://127.0.0.1:{door_port}/ws"):
            pass
    assert refused.value.response.status_code == 401


@pytest.mark.asyncio
async def test_an_upstream_that_isnt_up_is_a_502_not_a_pass():
    enable_socket()
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        dead = free.getsockname()[1]
    async with door.door_client() as client:
        front, port = await _serve(door.Door(f"127.0.0.1:{dead}", TOKEN, client))
        try:
            async with httpx.AsyncClient() as caller:
                response = await caller.get(f"http://127.0.0.1:{port}/", headers=_token())
        finally:
            await front.shutdown()
    assert response.status_code == 502


def test_routes_parse_and_bad_ones_are_refused():
    assert door.parse_routes("19765=gateway:18765,9081=pgweb:8081") == {19765: "gateway:18765", 9081: "pgweb:8081"}
    for bad in ("", "19765", "x=gateway:1", "1=gateway", "1=a:2,1=b:3"):
        with pytest.raises(ValueError):
            door.parse_routes(bad)


def test_a_door_never_starts_without_a_token(monkeypatch):
    monkeypatch.setenv(DOOR_ROUTES_ENV, "19765=gateway:18765")
    monkeypatch.delenv(DOOR_TOKEN_ENV, raising=False)
    with pytest.raises(SystemExit, match=DOOR_TOKEN_ENV):
        door.main()
    with pytest.raises(ValueError):
        door.Door("gateway:18765", "", httpx.AsyncClient())

