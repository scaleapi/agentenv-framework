"""A WebSocket on a site's route reaches the site's server through the real gateway, under the HTTP route's rules: its
data plane and card are refused, the role and forwarded address go upstream as the gateway writes them, and each
message and either side's close is relayed."""

from __future__ import annotations

import asyncio
import contextlib
import socket

import httpx
import pytest
import pytest_asyncio
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus

from agent_env.env.gateway.gateway import Gateway

pytestmark = pytest.mark.usefixtures("socket_enabled")


class EchoSite:
    """A site whose ``/live`` echoes each message, closes with 4001 on ``bye``, and records each handshake's headers
    and the code its client closed with."""

    def __init__(self):
        self.handshakes, self.closed = [], []
        self.app = Starlette(routes=[WebSocketRoute("/live", self.live), WebSocketRoute("/session", self.session),
                                     WebSocketRoute("/{path:path}", self.live)])

    async def session(self, websocket: WebSocket) -> None:
        await websocket.accept(headers=[(b"set-cookie", b"s=2; Path=/"), (b"set-cookie", b"t=3; Path=/"),
                                        (b"x-site", b"yes")])
        await websocket.close(1000)

    async def live(self, websocket: WebSocket) -> None:
        self.handshakes.append((websocket.url.path, dict(websocket.headers)))
        offered = websocket.headers.get("sec-websocket-protocol")
        await websocket.accept(subprotocol="chat.v2" if offered and "chat.v2" in offered else None)
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    self.closed.append(message.get("code"))
                    return
                if message.get("text") == "bye":
                    await websocket.close(4001, "done")
                    return
                if message.get("text") is not None:
                    await websocket.send_text(message["text"].upper())
                else:
                    await websocket.send_bytes(message["bytes"][::-1])
        except WebSocketDisconnect as e:
            self.closed.append(e.code)


async def serve(app) -> tuple[uvicorn.Server, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(app, lifespan="off", log_level="critical", access_log=False)
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return server, sock.getsockname()[1]


async def _until(condition, timeout_s: float = 5.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not condition():
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.02)


@pytest_asyncio.fixture
async def stack():
    site = EchoSite()
    site_server, site_port = await serve(site.app)
    gateway = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                      rest_proxy_urls={"notes": f"http://127.0.0.1:{site_port}"},
                      website_urls={"notes": f"http://127.0.0.1:{site_port}"})
    gateway_server, gateway_port = await serve(gateway._asgi_app())
    try:
        yield site, f"ws://127.0.0.1:{gateway_port}/website/notes"
    finally:
        await gateway_server.shutdown()
        await site_server.shutdown()


@pytest.mark.asyncio
async def test_messages_and_the_sites_close_are_relayed_with_the_gateways_headers(stack):
    site, route = stack
    async with connect(f"{route}/live?room=1", subprotocols=["chat.v1", "chat.v2"],
                       additional_headers=[("AgentEnv-Role", "grader"), ("X-Forwarded-Prefix", "/elsewhere"),
                                           ("Cookie", "s=1")]) as websocket:
        await websocket.send("hello")
        text = await asyncio.wait_for(websocket.recv(), 5)
        await websocket.send(b"\x01\x02\x03")
        data = await asyncio.wait_for(websocket.recv(), 5)
        await websocket.send("bye")
        with pytest.raises(ConnectionClosed):
            await asyncio.wait_for(websocket.recv(), 5)
        subprotocol, close = websocket.subprotocol, websocket.close_code

    ((path, headers),) = site.handshakes
    assert (text, data, subprotocol, close) == ("HELLO", b"\x03\x02\x01", "chat.v2", 4001)
    assert path == "/live" and headers["cookie"] == "s=1"
    assert headers["agentenv-role"] == "grader"
    assert (headers["x-forwarded-proto"], headers["x-forwarded-prefix"]) == ("http", "/website/notes")


@pytest.mark.asyncio
async def test_a_websocket_reached_at_the_sites_name_takes_it_as_its_host_and_at_any_other_keeps_its_own():
    site = EchoSite()
    site_server, site_port = await serve(site.app)
    gateway = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                      rest_proxy_urls={"notes": f"http://127.0.0.1:{site_port}"},
                      website_urls={"notes": f"http://127.0.0.1:{site_port}"}, site_names={"notes": ["notes.test"]})
    gateway_server, gateway_port = await serve(gateway._asgi_app())
    try:
        for root in ("https://notes.test", "https://evil.example"):
            async with connect(f"ws://127.0.0.1:{gateway_port}/website/notes/live",
                               additional_headers=[("AgentEnv-Site-Root", root)]) as websocket:
                await websocket.send("hi")
                assert await asyncio.wait_for(websocket.recv(), 5) == "HI"
    finally:
        await gateway_server.shutdown()
        await site_server.shutdown()
    assert [(headers["host"], headers.get("x-forwarded-host")) for _, headers in site.handshakes] == [
        ("notes.test", "notes.test"), (f"127.0.0.1:{site_port}", "evil.example")]


@pytest.mark.asyncio
async def test_the_sites_handshake_answer_reaches_the_client_with_each_cookie_apart(stack):
    _, route = stack
    async with connect(f"{route}/session") as websocket:
        headers = websocket.response.headers

    assert headers.get_all("set-cookie") == ["s=2; Path=/", "t=3; Path=/"] and headers["x-site"] == "yes"
    assert len(headers.get_all("sec-websocket-accept")) == 1 and len(headers.get_all("server")) <= 1


@pytest.mark.asyncio
async def test_the_clients_close_reaches_the_site(stack):
    site, route = stack
    async with connect(f"{route}/live") as websocket:
        await websocket.send("ping")
        assert await asyncio.wait_for(websocket.recv(), 5) == "PING"
        await websocket.close(4002, "leaving")

    await _until(lambda: site.closed)
    assert site.closed == [4002]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/.well-known/agent-env.json", "/agentenv", "/%61gentenv", "/x/..%2Fagentenv"])
async def test_a_websocket_to_the_sites_data_plane_or_card_is_refused(stack, path):
    site, route = stack
    with pytest.raises(InvalidStatus) as refused:
        async with connect(f"{route}{path}"):
            pass

    assert refused.value.response.status_code == 404 and site.handshakes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/grade", "/grade/", "/x/..%2Fgrade", "/grading/now", "/agentenv/ext/website/ready"])
async def test_a_websocket_to_a_path_the_sites_card_advertises_is_refused(path):
    site = EchoSite()
    card = {"name": "notes", "url": "/agentenv", "capabilities": {"extensions": [{
        "uri": "urn:agentenv:website/v1", "params": {"methods": {
            "ready": {"method": "GET", "endpoint": "/agentenv/ext/website/ready"},
            "grade": {"method": "GET", "endpoint": "/grade"}}}},
        {"uri": "urn:example:grading/v1", "params": {"endpoint": "/grading"}}]}}
    site.app.router.routes.insert(0, Route("/.well-known/agent-env.json", lambda request: JSONResponse(card)))
    site_server, site_port = await serve(site.app)
    gateway = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                      website_urls={"notes": f"http://127.0.0.1:{site_port}"})
    gateway_server, gateway_port = await serve(gateway._asgi_app())
    route = f"ws://127.0.0.1:{gateway_port}/website/notes"
    try:
        with pytest.raises(InvalidStatus) as refused:
            async with connect(f"{route}{path}"):
                pass
        async with connect(f"{route}/live") as websocket:
            await websocket.send("ping")
            assert await asyncio.wait_for(websocket.recv(), 5) == "PING"
    finally:
        await gateway_server.shutdown()
        await site_server.shutdown()

    assert refused.value.response.status_code == 404 and [p for p, _ in site.handshakes] == ["/live"]


@pytest.mark.asyncio
async def test_a_message_from_the_site_over_the_limit_closes_both_sides_with_1009(stack, monkeypatch):
    from agent_env.env.gateway import rest_proxy

    site, route = stack
    monkeypatch.setattr(rest_proxy, "WEBSOCKET_MESSAGE_MAX_BYTES", 1024)

    async def big(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_bytes(b"x" * 2048)
        with contextlib.suppress(WebSocketDisconnect):
            while (await websocket.receive())["type"] != "websocket.disconnect":
                pass

    site.app.router.routes.insert(0, WebSocketRoute("/big", big))
    async with connect(f"{route}/big") as websocket:
        with pytest.raises(ConnectionClosed) as closed:
            await asyncio.wait_for(websocket.recv(), 5)

    assert closed.value.rcvd.code == 1009


@pytest.mark.asyncio
async def test_a_websocket_is_refused_while_the_sites_card_cant_be_read():
    site = EchoSite()
    site.app.router.routes.insert(0, Route("/.well-known/agent-env.json", lambda request: JSONResponse({}, 503)))
    site_server, site_port = await serve(site.app)
    gateway = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                      website_urls={"notes": f"http://127.0.0.1:{site_port}"})
    gateway_server, gateway_port = await serve(gateway._asgi_app())
    try:
        with pytest.raises(InvalidStatus) as refused:
            async with connect(f"ws://127.0.0.1:{gateway_port}/website/notes/live"):
                pass
    finally:
        await gateway_server.shutdown()
        await site_server.shutdown()

    assert refused.value.response.status_code == 503 and site.handshakes == []


@pytest.mark.asyncio
async def test_the_providers_headers_never_reach_the_site_over_http_or_a_websocket():
    site, seen = EchoSite(), []

    async def echo_headers(request):
        seen.append(dict(request.headers))
        return JSONResponse(dict(request.headers))

    site.app.router.routes.insert(0, Route("/whoami", echo_headers))
    site_server, site_port = await serve(site.app)
    gateway = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                      website_urls={"notes": f"http://127.0.0.1:{site_port}"},
                      provider_headers=["CF-Access-Client-Id", "cf-access-client-secret", " "])
    gateway_server, gateway_port = await serve(gateway._asgi_app())
    sent = {"CF-Access-Client-Id": "id", "CF-Access-Client-Secret": "secret", "X-Site": "kept"}
    try:
        async with httpx.AsyncClient() as client:
            answer = await client.get(f"http://127.0.0.1:{gateway_port}/website/notes/whoami", headers=sent)
        async with connect(f"ws://127.0.0.1:{gateway_port}/website/notes/live", additional_headers=sent) as websocket:
            await websocket.send("ping")
            assert await asyncio.wait_for(websocket.recv(), 5) == "PING"
    finally:
        await gateway_server.shutdown()
        await site_server.shutdown()

    [(_, handshake)] = site.handshakes
    for headers in (seen[0], handshake, answer.json()):
        assert "cf-access-client-id" not in headers and "cf-access-client-secret" not in headers
        assert headers["x-site"] == "kept"


@pytest.mark.asyncio
async def test_a_refused_handshake_keeps_the_sites_headers_each_cookie_apart():
    async def refuse(websocket: WebSocket) -> None:
        answer = JSONResponse({"error": "signed out"}, status_code=401)
        answer.raw_headers += [(b"set-cookie", b"s=; Max-Age=0; Path=/"), (b"set-cookie", b"t=; Max-Age=0; Path=/"),
                               (b"x-site", b"yes")]
        await websocket.send_denial_response(answer)

    site_server, site_port = await serve(Starlette(routes=[WebSocketRoute("/live", refuse)]))
    gateway = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                      website_urls={"notes": f"http://127.0.0.1:{site_port}"})
    gateway_server, gateway_port = await serve(gateway._asgi_app())
    try:
        with pytest.raises(InvalidStatus) as refused:
            async with connect(f"ws://127.0.0.1:{gateway_port}/website/notes/live"):
                pass
    finally:
        await gateway_server.shutdown()
        await site_server.shutdown()

    answer = refused.value.response
    assert answer.status_code == 401 and answer.headers.get_all("set-cookie") == [
        "s=; Max-Age=0; Path=/", "t=; Max-Age=0; Path=/"]
    assert answer.headers["x-site"] == "yes" and answer.headers["content-type"] == "application/json"
    assert answer.headers.get_all("content-length") == [str(len(answer.body))]


@pytest.mark.asyncio
async def test_a_site_that_refuses_the_handshake_has_its_answer_relayed():
    site_server, site_port = await serve(Starlette(routes=[]))
    gateway = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                      website_urls={"notes": f"http://127.0.0.1:{site_port}"})
    gateway_server, gateway_port = await serve(gateway._mcp.streamable_http_app())
    try:
        with pytest.raises(InvalidStatus) as refused:
            async with connect(f"ws://127.0.0.1:{gateway_port}/website/notes/missing"):
                pass
        with pytest.raises(InvalidStatus) as unknown:
            async with connect(f"ws://127.0.0.1:{gateway_port}/website/other/live"):
                pass
    finally:
        await gateway_server.shutdown()
        await site_server.shutdown()

    assert refused.value.response.status_code == 403 and unknown.value.response.status_code == 404
