"""The gateway's CDP route over real loopback sockets, in front of a stand-in browser computer whose card, ``cdp``
answer, discovery and WebSocket are shaped like the real one's: every address the browser hands out comes back naming
the gateway, a WebSocket is relayed both ways with its close codes, an address the browser didn't mint is refused,
and each connection is a trajectory event without its token."""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
import websockets
from agentenv_protocol.browser_cdp import ROUTE
from agentenv_protocol.computer.app import COMPUTER_APP
from agentenv_protocol.website_browser import BROWSER
from pytest_socket import enable_socket
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket
from websockets.asyncio.client import connect

from agent_env.env.gateway import browser_proxy
from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.browser_proxy import (CDP_ROUTE, TokenRedaction, at_gateway, browsers, public_base,
                                                 rewritten)
from agent_env.env.gateway.constants import EXT_BROWSER_URI, EXT_COMPUTER_APP_URI
from agent_env.env.gateway.gateway import Gateway

TOKEN = "t0k3n"
# The address the browser minted, at a host the gateway's caller never reached it at.
MINTED = "http://browser-host:7000/svc/chromium/cdp/t0k3n/"
CDP_PATH = "/svc/chromium/agentenv/ext/browser/cdp"


def _app_card(name: str = "chromium", agent_routes: list[str] | None = None) -> dict:
    """A website browser's app card as a computer's driver lists it: its endpoints under /svc/<name>."""
    return {"name": name, "url": f"/svc/{name}/agentenv", "capabilities": {"extensions": [
        {"uri": EXT_BROWSER_URI, "params": {"endpoint": f"/svc/{name}/agentenv/ext/browser", "methods": {
            "cdp": {"endpoint": f"/svc/{name}/agentenv/ext/browser/cdp"},
            "record": {"method": "GET", "endpoint": f"/svc/{name}/agentenv/ext/browser/record"}}}},
        {"uri": EXT_COMPUTER_APP_URI, "params": {"paths": {}, "agent_routes": ["/cdp/"] if agent_routes is None
                                                 else agent_routes}}]}}


def _computer_card(*apps: dict) -> dict:
    return {"name": "chromium", "url": "/agentenv", "capabilities": {"extensions": []},
            "children_environments": list(apps)}


def _browser_computer(card: dict, seen: list[dict] | None = None) -> Starlette:
    """A computer serving ``card`` and a browser's CDP routes; ``seen`` gets the headers of each request to a CDP route."""
    def saw(connection) -> None:
        if seen is not None:
            seen.append(dict(connection.headers))

    async def served_card(_: Request) -> JSONResponse:
        return JSONResponse(card)

    named: dict[str, str] = {}

    async def cdp(request: Request) -> JSONResponse:
        saw(request)
        if origin := (await request.json()).get("origin"):  # minted under it, as a browser taking one does
            root = f"{origin.rstrip('/')}{request.headers.get('x-forwarded-prefix', '')}{ROUTE}{TOKEN}"
            named["ws_url"] = "wss" + root.removeprefix("https") + "/devtools/browser"
            return JSONResponse({"http_url": f"{root}/", "ws_url": named["ws_url"]})
        return JSONResponse({"http_url": MINTED, "ws_url": "ws://browser-host:7000/svc/chromium/cdp/t0k3n/devtools/browser"})

    async def discovery(request: Request) -> JSONResponse:
        saw(request)
        if request.path_params["token"] != TOKEN:
            return JSONResponse({"error": "agentenv didn't mint this CDP address"}, status_code=403)
        if request.path_params["rest"].rstrip("/") == "json/list":
            return JSONResponse([{"id": "1", "type": "page", "title": "Home", "url": "about:blank",
                                  "webSocketDebuggerUrl": f"ws://browser-host:7000/svc/chromium/cdp/{TOKEN}/devtools/page/1",
                                  "devtoolsFrontendUrl": ""}])
        return JSONResponse({"Browser": "Chrome/156", "webSocketDebuggerUrl": named.get("ws_url") or
                             f"ws://browser-host:7000/svc/chromium/cdp/{TOKEN}/devtools/browser"})

    async def browser_socket(websocket: WebSocket) -> None:
        saw(websocket)
        if websocket.path_params["token"] != TOKEN:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return
            text = message.get("text")
            if text == "close-4000":
                await websocket.close(code=4000)
                return
            await websocket.send_text(f"echo:{text}")

    return Starlette(routes=[
        Route("/.well-known/agent-env.json", served_card),
        Route(CDP_PATH, cdp, methods=["POST"]),
        Route("/svc/chromium/cdp/{token}/{rest:path}", discovery),
        WebSocketRoute("/svc/chromium/cdp/{token}/{rest:path}", browser_socket),
    ])


@pytest.fixture(autouse=True)
def _loopback():
    """Override the suite-wide socket block (tst/unit/conftest.py) for this module: loopback is the point here."""
    enable_socket()
    yield


async def _serve(app) -> tuple[uvicorn.Server, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.set_inheritable(True)
    config = uvicorn.Config(app, lifespan="off", log_level="critical", access_log=False)
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return server, f"http://127.0.0.1:{sock.getsockname()[1]}"


@asynccontextmanager
async def _gateway(tmp_path, monkeypatch, card: dict | None = None, seen: list[dict] | None = None,
                   provider_headers: tuple[str, ...] = ()):
    """The real gateway app on loopback, its /svc child ``web`` a browser computer serving ``card``; yields the
    gateway's base URL and a reader of its browser_connection events, which waits for ``at_least``: each is written
    as its connection ends, just after the client sees the close."""
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    upstream, upstream_url = await _serve(_browser_computer(_computer_card(_app_card()) if card is None else card, seen))
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[],
                 rest_proxy_urls={"web": upstream_url}, provider_headers=provider_headers)
    gateway, gateway_url = await _serve(gw._asgi_app())
    async def events(at_least: int) -> list[dict]:
        for _ in range(50):
            found = [event for event in map(json.loads, (tmp_path / "trajectory.jsonl").read_text().splitlines())
                     if event["event_type"] == "browser_connection"]
            if len(found) >= at_least:
                break
            await asyncio.sleep(0.1)
        return found

    try:
        yield gateway_url, events
    finally:
        await gateway.shutdown()
        await upstream.shutdown()


@pytest.mark.asyncio
async def test_the_cdp_answer_and_discovery_name_the_gateway_however_the_browser_minted_them(tmp_path, monkeypatch):
    async with _gateway(tmp_path, monkeypatch) as (url, _), httpx.AsyncClient() as client:
        answer = (await client.post(f"{url}/svc/web{CDP_PATH}", json={"user": "solver"})).json()
        version = (await client.get(f"{url}/browser/web/cdp/{TOKEN}/json/version")).json()
        listed = (await client.get(f"{url}/browser/web/cdp/{TOKEN}/json/list/")).json()
        tunnelled = (await client.get(f"{url}/browser/web/cdp/{TOKEN}/json/version",
                                      headers={"X-Forwarded-Proto": "https"})).json()

    ws = "ws" + url.removeprefix("http")
    assert answer == {"http_url": f"{url}/browser/web/cdp/{TOKEN}/",
                      "ws_url": f"{ws}/browser/web/cdp/{TOKEN}/devtools/browser"}
    assert version == {"Browser": "Chrome/156", "webSocketDebuggerUrl": f"{ws}/browser/web/cdp/{TOKEN}/devtools/browser"}
    assert [(entry["webSocketDebuggerUrl"], entry["devtoolsFrontendUrl"]) for entry in listed] == [
        (f"{ws}/browser/web/cdp/{TOKEN}/devtools/page/1", "")]
    assert tunnelled["webSocketDebuggerUrl"] == f"wss://{url.removeprefix('http://')}/browser/web/cdp/{TOKEN}/devtools/browser"


@pytest.mark.asyncio
async def test_behind_a_tunnel_that_says_no_scheme_an_address_minted_under_an_https_origin_stays_https_and_wss(
        tmp_path, monkeypatch):
    async with _gateway(tmp_path, monkeypatch) as (url, _), httpx.AsyncClient() as client:
        answer = (await client.post(f"{url}/svc/web{CDP_PATH}",
                                    json={"user": "solver", "origin": "https://gw-18765.w.modal.host"})).json()
        version = (await client.get(f"{url}/browser/web/cdp/{TOKEN}/json/version")).json()
    at = url.removeprefix("http://")
    assert answer == {"http_url": f"https://{at}/browser/web/cdp/{TOKEN}/",
                      "ws_url": f"wss://{at}/browser/web/cdp/{TOKEN}/devtools/browser"}
    assert version["webSocketDebuggerUrl"] == f"wss://{at}/browser/web/cdp/{TOKEN}/devtools/browser"


@pytest.mark.asyncio
async def test_a_websocket_is_relayed_both_ways_and_each_connection_is_an_event_without_its_token(tmp_path, monkeypatch):
    async with _gateway(tmp_path, monkeypatch) as (url, events), httpx.AsyncClient() as client:
        discovered = (await client.get(f"{url}/browser/web/cdp/{TOKEN}/json/version")).json()["webSocketDebuggerUrl"]
        async with connect(discovered, additional_headers={"AgentEnv-Role": "buyer"}) as conn:
            await conn.send("hi")
            echoed = await conn.recv()
        async with connect(discovered) as conn:
            await conn.send("close-4000")
            with pytest.raises(websockets.ConnectionClosed) as closed:
                await conn.recv()
        with pytest.raises(websockets.InvalidStatus) as refused:
            async with connect(discovered.replace(TOKEN, "forged")):
                pass
        recorded = await events(3)

    assert echoed == "echo:hi"
    assert closed.value.rcvd.code == 4000
    assert refused.value.response.status_code == 403
    assert sorted((e["role"], e["closed_by"], e["close_code"], e.get("status") or 0) for e in recorded) == [
        ("buyer", "client", 1000, 0), ("default", "browser", 1008, 403), ("default", "browser", 4000, 0)]
    assert all(e["browser"] == "web" and e["path"] == "devtools/browser" and e["closed_at"] for e in recorded)
    assert all(TOKEN not in json.dumps(e) and "forged" not in json.dumps(e) for e in recorded)


@pytest.mark.asyncio
async def test_the_providers_headers_reach_the_browser_on_no_cdp_route(tmp_path, monkeypatch):
    seen: list[dict] = []
    sent = {"CF-Access-Client-Id": "id", "X-Kept": "yes"}
    async with (_gateway(tmp_path, monkeypatch, seen=seen, provider_headers=("CF-Access-Client-Id",)) as (url, _),
                httpx.AsyncClient() as client):
        await client.post(f"{url}/svc/web{CDP_PATH}", json={}, headers=sent)
        discovered = (await client.get(f"{url}/browser/web/cdp/{TOKEN}/json/version", headers=sent)).json()
        async with connect(discovered["webSocketDebuggerUrl"], additional_headers=sent) as conn:
            await conn.send("hi")
            await conn.recv()

    assert len(seen) == 3
    assert [("cf-access-client-id" in headers, headers.get("x-kept")) for headers in seen] == [(False, "yes")] * 3


@pytest.mark.asyncio
async def test_a_relay_that_fails_in_between_closes_both_sides_and_records_the_failures_type(tmp_path, monkeypatch):
    real = browser_proxy.connect

    async def failing_upstream(*args, **kwargs):
        upstream = await real(*args, **kwargs)

        async def send(message):
            raise RuntimeError(f"lost {message}")
        upstream.send = send
        return upstream

    monkeypatch.setattr(browser_proxy, "connect", failing_upstream)
    async with _gateway(tmp_path, monkeypatch) as (url, events):
        async with connect(f"ws{url.removeprefix('http')}/browser/web/cdp/{TOKEN}/devtools/browser") as conn:
            await conn.send("secret page data")
            with pytest.raises(websockets.ConnectionClosed) as closed:
                await conn.recv()
        recorded = await events(1)

    assert closed.value.rcvd.code == 1011
    assert [(e["closed_by"], e["close_code"], e["error"]) for e in recorded] == [("gateway", 1011, "RuntimeError")]
    assert "secret" not in json.dumps(recorded)


@pytest.mark.asyncio
@pytest.mark.parametrize(("card", "status"), [
    (_computer_card(), 404),
    (_computer_card(_app_card(agent_routes=[])), 404),
    (_computer_card(_app_card("chromium"), _app_card("brave")), 409),
])
async def test_a_child_with_no_browser_or_several_is_refused_and_other_svc_requests_go_through(
        tmp_path, monkeypatch, card, status):
    async with _gateway(tmp_path, monkeypatch, card) as (url, events), httpx.AsyncClient() as client:
        discovered = await client.get(f"{url}/browser/web/cdp/{TOKEN}/json/version")
        unknown = await client.get(f"{url}/browser/nowhere/cdp/{TOKEN}/json/version")
        svc = await client.post(f"{url}/svc/web{CDP_PATH}", json={"user": "solver"})
        with pytest.raises(websockets.InvalidStatus):
            async with connect(f"ws{url.removeprefix('http')}/browser/web/cdp/{TOKEN}/devtools/browser"):
                pass
        recorded = await events(1)

    assert (discovered.status_code, unknown.status_code) == (status, 404)
    assert svc.json()["http_url"] == MINTED  # no single browser, so no rewrite: the answer goes through as it came
    assert [(e["closed_by"], e["status"]) for e in recorded] == [("gateway", status)]


@pytest.mark.asyncio
async def test_a_browser_whose_server_wasnt_answering_at_the_first_read_is_found_once_its_card_names_it(
        tmp_path, monkeypatch):
    card = _computer_card({"name": "chromium", "url": "/svc/chromium/agentenv"})  # how a driver lists a down app
    async with _gateway(tmp_path, monkeypatch, card) as (url, _), httpx.AsyncClient() as client:
        before = await client.get(f"{url}/browser/web/cdp/{TOKEN}/json/version")
        card["children_environments"] = [_app_card()]
        after = await client.get(f"{url}/browser/web/cdp/{TOKEN}/json/version")

    assert (before.status_code, after.status_code) == (404, 200)
    assert after.json()["webSocketDebuggerUrl"] == f"ws{url.removeprefix('http')}/browser/web/cdp/{TOKEN}/devtools/browser"


def test_an_address_is_rebased_by_its_path_and_anything_else_is_kept():
    base = "https://gw.example:8443"
    assert at_gateway("http://10.0.0.5:7000/svc/chromium/cdp/abc/", base, "b") == "https://gw.example:8443/browser/b/cdp/abc/"
    assert at_gateway("ws://x/svc/chromium/cdp/abc/devtools/browser?y=1", base, "b") == (
        "wss://gw.example:8443/browser/b/cdp/abc/devtools/browser?y=1")
    assert at_gateway("ws://x/cdp/abc", "http://gw", "b") == "ws://gw/browser/b/cdp/abc"
    assert at_gateway("https://named.example/svc/chromium/cdp/abc/", "http://gw", "b") == (
        "https://gw/browser/b/cdp/abc/")
    assert at_gateway("wss://named.example/cdp/abc/devtools/browser", "http://gw", "b") == (
        "wss://gw/browser/b/cdp/abc/devtools/browser")
    for kept in ("", "/devtools/inspector.html?ws=x/cdp/abc", "http://x/devtools/browser", "file:///cdp/abc/"):
        assert at_gateway(kept, base, "b") == kept
    assert rewritten({"webSocketDebuggerUrl": "ws://x/cdp/a/", "other": "ws://x/cdp/a/"}, frozenset({"webSocketDebuggerUrl"}),
                     "http://gw", "b") == {"webSocketDebuggerUrl": "ws://gw/browser/b/cdp/a/", "other": "ws://x/cdp/a/"}


def test_a_browser_is_found_in_the_card_or_among_its_apps_only_when_it_opens_its_cdp_route_to_agents():
    alone = _app_card()
    alone["capabilities"]["extensions"][0]["params"]["methods"]["cdp"]["endpoint"] = "/agentenv/ext/browser/cdp"
    assert [(b.prefix, b.cdp_path) for b in browsers(_computer_card(_app_card()))] == [("/svc/chromium", CDP_PATH)]
    assert [(b.prefix, b.cdp_path) for b in browsers(alone)] == [("", "/agentenv/ext/browser/cdp")]
    assert browsers(_computer_card(_app_card(agent_routes=["/raw/"]))) == []


def test_the_public_base_is_the_host_the_client_sent_a_bad_one_none():
    def request(headers: dict, scheme: str = "http") -> Request:
        return Request({"type": "http", "scheme": scheme, "path": "/", "headers":
                        [(k.lower().encode(), v.encode()) for k, v in headers.items()]})

    assert public_base(request({"host": "host.docker.internal:41000"})) == "http://host.docker.internal:41000"
    assert public_base(request({"host": "gw", "x-forwarded-proto": "https, http"})) == "https://gw"
    assert public_base(request({"host": "[::1]:80"}, scheme="https")) == "https://[::1]:80"
    for bad in ("", "evil.com/path", "a b", "evil.com@x"):
        assert public_base(request({"host": bad})) is None


def test_no_log_line_shows_a_cdp_addresss_token():
    record = logging.LogRecord("uvicorn.error", logging.INFO, __file__, 1, '%s - "WebSocket %s" [accepted]',
                               ("127.0.0.1:5", f"/browser/web/cdp/{TOKEN}/devtools/browser"), None)
    TokenRedaction().filter(record)
    assert record.getMessage() == '127.0.0.1:5 - "WebSocket /browser/web/cdp/<redacted>/devtools/browser" [accepted]'


def test_the_mirrored_names_match_the_protocols():
    assert (EXT_BROWSER_URI, EXT_COMPUTER_APP_URI, CDP_ROUTE) == (BROWSER, COMPUTER_APP, ROUTE)
