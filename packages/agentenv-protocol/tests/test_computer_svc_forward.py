"""The driver's /svc/<app>/ in front of an app's own server, on its socket as on Linux and macOS and on a loopback port
with its token as on Windows, alike: a WebSocket is relayed both ways, with the
subprotocol the app picks, until the app closes it with its code, and only with the control token, off the routes the
app opens to agents, and a Host that's a host; an answer streams through as the app sends it, 20 MB whole; every
method and body goes on to the path as it came, still escaped, and the caller's headers, and only those, pass on as
their bytes but for the hop-by-hop ones, the control token and any address the caller forwarded, which the driver
sets as the one it was reached at. An agent's call on a route the app opens that the app answers, a WebSocket it takes, and
any tool call, note in the app's folder that an agent has acted; agentenv's own calls, refused ones and those the app
answers with an error don't."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import tempfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import FakeDriver  # noqa: E402
from probe_fake import served  # noqa: E402
from starlette.requests import Request  # noqa: E402
from starlette.responses import JSONResponse, Response, StreamingResponse  # noqa: E402
from starlette.routing import WebSocketRoute  # noqa: E402
from starlette.websockets import WebSocket  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402
from websockets.exceptions import ConnectionClosedError, InvalidStatus  # noqa: E402

from agentenv_protocol import CONTROL_TOKEN_ENV, client  # noqa: E402
from agentenv_protocol.computer import ComputerAppEnvironment, apps, linux, svc  # noqa: E402
from agentenv_protocol.computer.app import AGENT_ACTED, SOCKET  # noqa: E402
import agentenv_protocol.computer.served as transports  # noqa: E402
from agentenv_protocol.computer.served import TOKEN_HEADER, bind, guarded  # noqa: E402
from agentenv_protocol.tool_context import EMPTY  # noqa: E402

MB = 1 << 20
TOKEN = "s3cret"


@linux()
class Echo(ComputerAppEnvironment):
    """An app whose own server echoes what a WebSocket sends it until "bye", which it closes with 4001; answers with
    the headers it got and the size of what it was sent; and streams 20 MB, its first MB at once and the rest once
    ``released``."""

    agent_routes = ("/raw/",)

    def __init__(self) -> None:
        super().__init__()
        self.released = asyncio.Event()

    def create_app(self):
        mcp = super().create_app()
        mcp.custom_route("/headers", methods=["GET"])(self._headers)
        mcp.custom_route("/upload", methods=["PUT", "DELETE"])(self._upload)
        mcp.custom_route("/big", methods=["GET"])(self._big)
        mcp.custom_route("/raw/{rest:path}", methods=["GET"])(self._raw)
        mcp._custom_starlette_routes += [WebSocketRoute("/echo", self._echo), WebSocketRoute("/raw/{rest:path}",
                                                                                             self._raw_socket)]
        return mcp

    async def _raw(self, request: Request) -> Response:
        scope = request.scope
        if scope["raw_path"].decode().endswith("/missing"):
            return JSONResponse({"error": "no such thing"}, status_code=404)
        return JSONResponse({"path": scope["raw_path"].decode(), "query": scope["query_string"].decode()})

    async def _raw_socket(self, websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text(websocket.scope["raw_path"].decode())
        await websocket.close()

    async def _headers(self, request: Request) -> Response:
        answer = JSONResponse({"host": request.headers.get("host"), "headers": dict(request.headers.items())})
        for header in ("set-cookie: a=1", "set-cookie: b=2", "connection: x-secret", "x-secret: 1", "x-kept: 1",
                       'content-disposition: attachment; filename="报告.zip"'):
            name, _, value = header.partition(": ")
            answer.raw_headers.append((name.encode(), value.encode()))
        return answer

    async def _upload(self, request: Request) -> Response:
        body = await request.body()
        return JSONResponse({"method": request.method, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})

    async def _big(self, request: Request) -> Response:
        async def chunks():
            yield b"a" * MB
            await self.released.wait()
            for _ in range(19):
                yield b"b" * MB
        return StreamingResponse(chunks(), media_type="application/octet-stream")

    async def _echo(self, websocket: WebSocket) -> None:
        await websocket.accept(subprotocol="cdp" if "cdp" in websocket.scope.get("subprotocols", []) else None)
        await websocket.send_text(json.dumps({"forwarded": websocket.headers.get("x-forwarded-prefix"),
                                              "authorization": websocket.headers.get("authorization")}))
        while (message := await websocket.receive())["type"] != "websocket.disconnect":
            if message.get("text") == "bye":
                await websocket.close(code=4001)
                return
            await (websocket.send_text(message["text"]) if message.get("text") is not None
                   else websocket.send_bytes(message["bytes"][::-1]))


@pytest.fixture(params=["socket", "loopback"])
def transport(request) -> str:
    """Where the app's server answers: its folder's Unix socket, as on Linux and macOS, or a loopback port with a
    token, as on Windows."""
    return request.param


@contextlib.asynccontextmanager
async def _computer(monkeypatch, transport: str):
    """The Echo app served in a computer's apps folder over ``transport``, and a driver in front of it on loopback, with
    a control token; the app, and the driver's URL."""
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="apps-") as root:
        folder = Path(root) / "echo"
        (folder / SOCKET).parent.mkdir(parents=True)
        monkeypatch.setattr(apps, "APPS", Path(root))
        monkeypatch.setattr(transports, "LOOPBACK", transport == "loopback")
        monkeypatch.setenv("ENVIRONMENT_NAME", "echo")
        app = Echo()
        asgi = app.create_app().streamable_http_app()
        if transport == "loopback":
            listening, token = bind(folder)
            server = uvicorn.Server(uvicorn.Config(guarded(asgi, token), log_level="warning", lifespan="off"))
            serving = asyncio.create_task(server.serve(sockets=[listening]))
        else:
            server = uvicorn.Server(uvicorn.Config(asgi, uds=str(folder / SOCKET), log_level="warning",
                                                   lifespan="off"))
            serving = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        monkeypatch.delenv("ENVIRONMENT_NAME")
        monkeypatch.setenv(CONTROL_TOKEN_ENV, TOKEN)
        try:
            with served(FakeDriver()) as url:
                yield app, url
        finally:
            server.should_exit = True
            await serving


@pytest.mark.asyncio
async def test_a_websocket_through_svc_reaches_the_app_both_ways_until_it_closes_and_only_with_the_token(monkeypatch, transport):
    async with _computer(monkeypatch, transport) as (_, url):
        ws = url.replace("http", "ws") + "/svc/echo/echo"
        async with connect(ws, additional_headers=client.control_headers(TOKEN), subprotocols=["cdp"]) as socket:
            hello = json.loads(await socket.recv())
            await socket.send("Page.navigate")
            text = await socket.recv()
            await socket.send(b"\x01\x02\x03")
            data = await socket.recv()
            await socket.send("bye")
            with pytest.raises(ConnectionClosedError) as closed:
                await socket.recv()
            subprotocol = socket.subprotocol
        refused = []
        for headers, path in (({}, "/svc/echo/echo"), (client.control_headers("guess"), "/svc/echo/echo"),
                              (client.control_headers(TOKEN), "/svc/notes/echo")):
            with pytest.raises(InvalidStatus) as status:
                async with connect(url.replace("http", "ws") + path, additional_headers=headers):
                    pass
            refused.append(status.value.response.status_code)
        malformed = [await _handshake(url, host) for host in ("x/y", "[::1")]
        async with connect(url.replace("http", "ws") + "/svc/echo/raw/a%3Fb",
                           additional_headers=client.control_headers(TOKEN)) as socket:
            raw = await socket.recv()
        async with connect(url.replace("http", "ws") + "/svc/echo/raw/open") as socket:
            opened = await socket.recv()

    assert hello == {"forwarded": "/svc/echo", "authorization": None} and subprotocol == "cdp"
    assert (text, data) == ("Page.navigate", b"\x03\x02\x01") and closed.value.rcvd.code == 4001
    assert refused == [403, 403, 403] and malformed == [b"HTTP/1.1 403 Forbidden", b"HTTP/1.1 403 Forbidden"]
    assert raw == "/raw/a%3Fb" and opened == "/raw/open"


@pytest.mark.asyncio
async def test_an_agents_call_on_a_route_the_app_opens_and_a_tool_call_note_in_its_folder_that_an_agent_acted(
        monkeypatch, transport):
    async with _computer(monkeypatch, transport) as (_, url):
        marker = apps.APPS / "echo" / AGENT_ACTED
        async with httpx.AsyncClient() as http:
            await http.get(url + "/svc/echo/headers", headers=client.control_headers(TOKEN))
            by_agentenv = marker.exists()
            refused = (await http.get(url + "/svc/echo/headers")).status_code
            by_refused = marker.exists()
            missing = (await http.get(url + "/svc/echo/raw/missing")).status_code
            by_error = marker.exists()
            await http.get(url + "/svc/echo/raw/open")
            by_agent = marker.exists()
            marker.unlink()
        async with connect(url.replace("http", "ws") + "/svc/echo/raw/open") as socket:
            await socket.recv()
        by_socket = marker.exists()
        marker.unlink()
        FakeDriver().on_tool_call(EMPTY)
        by_tool = marker.exists()

    assert (by_agentenv, refused, by_refused, missing, by_error) == (False, 401, False, 404, False)
    assert (by_agent, by_socket, by_tool) == (True, True, True)


@pytest.mark.asyncio
async def test_an_answer_streams_through_svc_as_the_app_sends_it_20_mb_whole(monkeypatch, transport):
    async with _computer(monkeypatch, transport) as (app, url):
        async with httpx.AsyncClient(base_url=url, headers=client.control_headers(TOKEN), timeout=30) as http:
            async with http.stream("GET", "/svc/echo/big") as answer:
                chunks, first = answer.aiter_raw(), []

                async def first_mb() -> None:
                    while sum(map(len, first)) < MB:
                        first.append(await chunks.__anext__())
                await asyncio.wait_for(first_mb(), 10)
                app.released.set()
                body = b"".join(first) + b"".join([chunk async for chunk in chunks])

    assert answer.status_code == 200 and len(body) == 20 * MB
    assert body[:MB] == b"a" * MB and body[MB:] == b"b" * (19 * MB)


@pytest.mark.asyncio
async def test_every_method_and_body_goes_on_and_headers_pass_but_the_hop_ones_token_and_forwarded_address(
        monkeypatch, transport):
    payload = b"x" * (3 * MB)
    async with _computer(monkeypatch, transport) as (_, url):
        async with httpx.AsyncClient(base_url=url, headers=client.control_headers(TOKEN), timeout=30) as http:
            seen = await http.get("/svc/echo/headers", headers={
                "x-forwarded-host": "evil.test", "x-forwarded-prefix": "/elsewhere", "connection": "x-drop",
                "x-drop": "1", "x-kept": "1", TOKEN_HEADER: "forged"})
            put = (await http.put("/svc/echo/upload", content=payload)).json()
            deleted = (await http.delete("/svc/echo/upload")).json()
            tokenless = httpx.AsyncClient(base_url=url)
            refused = await tokenless.get("/svc/echo/headers")
            opened = await tokenless.get("/svc/echo/raw/open")
            stepped_out = await tokenless.get(httpx.URL(url, raw_path=b"/svc/echo/raw/%2E%2E/headers"))
            utf8 = (await http.get("/svc/echo/headers", headers=[(b"x-name", "报告".encode())])).json()["headers"]
            raw = (await http.get("/svc/echo/raw/a%3Fb%2Fc?q=1")).json()
            escaped = (await http.get(httpx.URL(url, raw_path=b"/%73vc/echo/raw/a%3Fb?q=1"))).json()
            http.headers.clear()
            bare = (await http.get("/svc/echo/headers", headers=client.control_headers(TOKEN))).json()["headers"]
    got, host = seen.json()["headers"], url.removeprefix("http://")

    assert seen.json()["host"] == host and got["x-kept"] == "1"
    assert {key: got[key] for key in svc.FORWARDED} == {
        "x-forwarded-host": host, "x-forwarded-proto": "http", "x-forwarded-prefix": "/svc/echo"}
    assert "authorization" not in got and "x-drop" not in got and TOKEN_HEADER not in got
    assert seen.headers.get_list("set-cookie") == ["a=1", "b=2"] and "x-secret" not in seen.headers
    assert seen.headers["x-kept"] == "1"
    assert put == {"method": "PUT", "bytes": 3 * MB, "sha256": hashlib.sha256(payload).hexdigest()}
    assert deleted["method"] == "DELETE" and deleted["bytes"] == 0 and refused.status_code == 401
    assert opened.json()["path"] == "/raw/open" and stepped_out.status_code == 401
    assert not {"accept", "accept-encoding", "user-agent"} & set(bare)
    assert (b"content-disposition", 'attachment; filename="报告.zip"'.encode()) in seen.headers.raw
    assert utf8["x-name"] == "报告".encode().decode("latin-1")
    assert raw == {"path": "/raw/a%3Fb%2Fc", "query": "q=1"} and escaped == {"path": "/raw/a%3Fb", "query": "q=1"}


@pytest.mark.asyncio
async def test_the_drivers_server_logs_a_path_under_a_route_open_to_agents_only_up_to_the_route(monkeypatch, transport, caplog):
    for logger in svc.SERVER_LOGGERS:
        monkeypatch.setattr(logging.getLogger(logger), "filters", [])
    async with _computer(monkeypatch, transport) as (_, url):
        async with httpx.AsyncClient(base_url=url, headers=client.control_headers(TOKEN)) as http:
            with caplog.at_level(logging.INFO):
                opened = await http.get("/svc/echo/raw/a-secret/open")
        forwarded = [record.getMessage() for record in caplog.records if "a-secret" in record.getMessage()]
        logged = []
        for logger, message, path in [("uvicorn.access", '%s - "%s %s HTTP/%s" %d', "/svc/echo/raw/a-secret/open"),
                                      ("uvicorn.error", '%s - "WebSocket %s" [accepted]', "/svc/echo/raw/a-secret"),
                                      ("uvicorn.access", '%s - "%s %s HTTP/%s" %d', "/svc/echo/headers")]:
            args = ("127.0.0.1:1", "GET", path, "1.1", 200) if "HTTP" in message else ("127.0.0.1:1", path)
            record = logging.LogRecord(logger, logging.INFO, __file__, 0, message, args, None)
            logging.getLogger(logger).filter(record)
            logged.append(record.getMessage())

    assert opened.status_code == 200 and forwarded == []
    assert logged == ['127.0.0.1:1 - "GET /svc/echo/raw/<redacted> HTTP/1.1" 200',
                      '127.0.0.1:1 - "WebSocket /svc/echo/raw/<redacted>" [accepted]',
                      '127.0.0.1:1 - "GET /svc/echo/headers HTTP/1.1" 200']


@pytest.mark.parametrize(("raw", "root", "routed"), [(b"/svc/echo/raw/a%2Fb", "", "raw/a/b"),
                                                     (b"/gw/svc/echo/raw/a%2Fb", "/gw", "raw/a/b")])
def test_the_apps_path_is_the_raw_one_past_svc_and_its_name_wherever_the_driver_is_mounted(raw, root, routed):
    connection = SimpleNamespace(scope={"raw_path": raw, "root_path": root, "query_string": b"q=1"},
                                 path_params={"name": "echo", "path": routed})

    assert svc.target(connection) == "/raw/a%2Fb?q=1"


async def _handshake(url: str, host: str) -> bytes:
    """The status line the driver answers a WebSocket handshake for the app's echo route with, sent with ``host`` as
    its Host and the control token."""
    reader, writer = await asyncio.open_connection(*url.removeprefix("http://").split(":"))
    writer.write((f"GET /svc/echo/echo HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                  "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n"
                  f"Authorization: Bearer {TOKEN}\r\n\r\n").encode())
    status = (await asyncio.wait_for(reader.readline(), 10)).strip()
    writer.close()
    return status
