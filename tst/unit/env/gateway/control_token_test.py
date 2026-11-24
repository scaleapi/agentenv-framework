"""The gateway's control routes, and behind its proxy the data plane, an MCP server's own endpoint and the older REST
routes that reset, load or export state, take the deploy's control token when it has one; every other route stays
open."""
from __future__ import annotations

import asyncio
import logging
from urllib.parse import unquote, urlsplit

import httpx
import pytest
from starlette.responses import JSONResponse

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.constants import CONTROL_ROUTES, LEGACY_STATE_ROUTES, WELL_KNOWN_PATH
from agent_env.env.gateway.gateway import Gateway

TOKEN = "s3cret"
CONTROL = {
    "/tools/disable": "POST", "/tools/enable": "POST",
    "/triggers/register": "POST", "/triggers/remove": "POST", "/triggers/clear": "POST",
    "/notifications/config": "POST", "/notifications/take": "POST", "/notifications/clear": "POST",
    "/clock/followers": "PUT",
}
OPEN = {"/state": "GET", "/step": "POST", "/clock/time": "GET", "/clock/state": "GET", "/triggers/state": "GET",
        "/notifications/state": "GET", "/trajectory": "GET", "/changes": "GET", "/.well-known/agent-env.json": "GET",
        "/clock/set-time": "PUT", "/clock/advance": "PUT", "/clock/clear": "POST"}


LEGACY = {"/api/reset": "POST", "/api/add": "POST", "/api/bridge/apply": "POST", "/export-state": "GET",
          "/export-snapshot": "GET", "/snapshot-dirty": "GET", "/reset-memory": "POST", "/validate-rows": "POST",
          "/api/_state": "GET", "/api/_reset": "POST", "/api/debug/seed": "POST"}

# Behind the proxy: the data plane, its extensions, each service's included, an MCP server's own endpoint, and the
# older REST routes on an MCP server's route and a site's, however the path is spelt on its way to the gateway.
GUARDED = [("POST", "/agentenv"), ("POST", "/svc/mcp-svc/agentenv"), ("POST", "/svc/mcp-svc/agentenv/ext/grant"),
           ("POST", "/svc/mcp-svc/x/../agentenv"), ("POST", "/svc/mcp-svc/%61gentenv"), ("POST", "/svc/shop/agentenv"),
           ("POST", "/svc/mcp-svc/mcp"), ("GET", "/svc/mcp-svc/mcp/"),
           ("DELETE", "/svc/mcp-svc/./mcp"), ("POST", "/svc/mcp-svc//mcp"), ("POST", "/svc/mcp-svc/public%2Fx/../agentenv"),
           ("POST", "/svc/mcp-svc/public%2Fx/../mcp"), ("POST", "/svc/mcp-svc/public%2Fx/../agentenv/ext/set_acting_user"),
           ("POST", "/svc/mcp-svc/AgentEnv"), ("POST", "/svc/mcp-svc/MCP"),
           *((method, f"/svc/{key}{path}") for key in ("mcp-svc", "shop") for path, method in LEGACY.items()),
           ("GET", "/svc/mcp-svc/export-snapshot?stream=1"), ("POST", "/svc/shop/api/reset/x"),
           ("GET", "/svc/mcp-svc/x/../export-state"), ("GET", "/svc/mcp-svc/%65xport-state"),
           ("POST", "/svc/mcp-svc/public%2Fx/../api/reset"), ("POST", "/svc/shop/public%2Fx/../api/add"),
           ("POST", "/svc/mcp-svc/API/Reset"), ("GET", "/svc/shop/Export-Snapshot"), ("POST", "/svc/mcp-svc/api//reset"),
           ("POST", "/svc/mcp-svc/api/bridge/../bridge/apply"), ("GET", "/svc/shop/api/_state/"),
           ("GET", "/svc/shop/API/_State"), ("GET", "/svc/shop/x/../api/_state"), ("GET", "/svc/shop/api/%5Fstate")]
PROXIED_OPEN = [("GET", "/svc/mcp-svc/.well-known/agent-env.json"), ("GET", "/svc/mcp-svc/openapi.yaml"),
                ("POST", "/svc/shop/api/orders"), ("POST", "/svc/shop/mcp"), ("GET", "/website/shop/index.html"),
                ("GET", "/svc/mcp-svc/export-statex"), ("POST", "/svc/mcp-svc/api/resetx"),
                ("POST", "/svc/shop/api/address"), ("GET", "/svc/mcp-svc/x/export-state"),
                ("POST", "/svc/mcp-svc/api/bridge"), ("GET", "/website/shop/export-statex"),
                ("GET", "/svc/shop/api/_statex"), ("GET", "/svc/shop/api/state"), ("GET", "/website/shop/api/state"),
                ("GET", "/website/shop/api/_statex"), ("GET", "/svc/shop/api/debug/inspect")]
# A site's route never reaches these on a gateway with a token, however they are spelt, with the token or without.
SITE_REFUSED = [*(f"/website/shop{path}" for path in LEGACY), "/website/shop/api/reset/x",
                "/website/shop/export-snapshot?stream=1", "/website/shop/x/../export-state",
                "/website/shop/x/../api/add", "/website/shop/public%2Fx/../api/reset",
                "/website/shop/x/../api/_state", "/website/shop/api/%5Fstate"]
_RealAsyncClient = httpx.AsyncClient


def _client(token: str | None) -> httpx.AsyncClient:
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], control_token=token)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw._asgi_app()), base_url="http://gateway")


def test_the_routes_under_test_are_the_ones_the_gateway_covers():
    assert set(CONTROL) == CONTROL_ROUTES
    assert set(LEGACY) == set(LEGACY_STATE_ROUTES)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(CONTROL))
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}],
                         ids=["none", "wrong", "not-bearer"])
async def test_a_control_route_refuses_a_call_without_the_token(path, headers):
    async with _client(TOKEN) as client:
        before = (await client.get("/triggers/state")).json(), (await client.get("/notifications/state")).json()
        response = await client.request(CONTROL[path], path, json={}, headers=headers)
        after = (await client.get("/triggers/state")).json(), (await client.get("/notifications/state")).json()
    assert response.status_code == 401
    assert response.json() == {"ok": False, "error": "this route takes the env's control token"}
    assert after == before


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(CONTROL))
async def test_a_control_route_answers_a_call_with_the_token(path):
    async with _client(TOKEN) as client:
        response = await client.request(CONTROL[path], path, json={}, headers={"Authorization": f"Bearer {TOKEN}"})
    assert response.status_code != 401


@pytest.mark.asyncio
async def test_the_clock_is_armed_and_read_without_the_token():
    async with _client(TOKEN) as client:
        armed = await client.put("/clock/set-time", json={"virtual_time": "2019-06-15T12:00:00Z"})
        read = await client.get("/clock/time")
    assert (armed.status_code, read.status_code) == (200, 200)
    assert read.json()["virtual_time"].startswith("2019-06-15T12:00")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(OPEN))
async def test_every_other_route_stays_open(path):
    async with _client(TOKEN) as client:
        response = await client.request(OPEN[path], path, json={} if OPEN[path] == "POST" else None)
    assert response.status_code != 401


@pytest.mark.asyncio
@pytest.mark.parametrize("value, seen", [(f"Bearer {TOKEN}", None), ("Bearer a-childs-own", "Bearer a-childs-own")],
                         ids=["the-gateways", "another"])
async def test_no_route_behind_the_gateway_reads_its_token(value, seen):
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], control_token=TOKEN)

    @gw._mcp.custom_route("/echo", methods=["GET"])
    async def echo(request):
        return JSONResponse({"authorization": request.headers.get("authorization")})

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw._asgi_app()), base_url="http://gateway") as client:
        response = await client.get("/echo", headers={"Authorization": value})
    assert response.json() == {"authorization": seen}


@pytest.mark.parametrize("path, kept", [
    ("/svc/desktop/agentenv/ext/computer-exec", True), ("/svc/desktop", True), ("/clock/watch", True),
    ("/svc/mcp-items/agentenv", False), ("/svc/desktop-2/agentenv", False), ("/website/desktop/", False),
    ("/state", False)])
def test_its_token_goes_on_only_to_the_computers_it_fronts_and_to_its_clock_watch(path, kept):
    """A multi env's computers boot with the gateway's token and take it on their /svc routes, and send it to its clock
    watch as their read token; every other request loses it."""
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], control_token=TOKEN,
                 control_token_children=["desktop"])
    assert gw._keeps_control_token(path) is kept


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(CONTROL))
async def test_a_gateway_started_without_a_token_checks_none(path):
    async with _client(None) as client:
        response = await client.request(CONTROL[path], path, json={})
    assert response.status_code != 401


class _Backing:
    """The services behind the gateway's proxy: each answers every request, and the paths reaching them are kept."""

    def __init__(self, monkeypatch):
        self.paths: list[str] = []
        self.authorization: list[str | None] = []
        transport = httpx.MockTransport(self._answer)
        monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.authorization.append(request.headers.get("authorization"))

        async def body():
            yield b'{"jsonrpc": "2.0", "id": 1, "result": {}}'

        return httpx.Response(200, headers={"content-type": "application/json"}, content=body())


async def _proxied(token: str | None, method: str, path: str, headers: dict[str, str], **children) -> int:
    """The status of one request through the gateway's served app, its path as uvicorn hands it: percent-decoded, its
    dot segments and doubled slashes kept; an MCP server and a website behind it unless ``children`` say otherwise."""
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", control_token=token, **{
        "internal_mcp_servers": [], "rest_proxy_urls": {"mcp-svc": "http://svc", "shop": "http://shop-backend"},
        "website_urls": {"shop": "http://shop-backend"}, **children})
    url = urlsplit(path)
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1", "method": method,
             "scheme": "http", "path": unquote(url.path), "raw_path": url.path.encode(), "query_string": url.query.encode(),
             "root_path": "", "server": ("gateway", 80), "client": ("agent", 1),
             "headers": [(b"content-type", b"application/json"),
                         *((name.lower().encode(), value.encode()) for name, value in headers.items())]}
    received, started = False, []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": b'{"jsonrpc": "2.0", "id": 1, "method": "data/reset"}',
                    "more_body": False}
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.start":
            started.append(message["status"])

    await gw._asgi_app()(scope, receive, send)
    return started[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", GUARDED)
@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}], ids=["none", "wrong"])
async def test_the_data_plane_and_a_servers_own_endpoint_behind_the_proxy_refuse_a_call_without_the_token(
        monkeypatch, caplog, method, path, headers):
    backing = _Backing(monkeypatch)
    caplog.set_level(logging.WARNING, logger="agent_env.env.gateway.gateway")
    assert await _proxied(TOKEN, method, path, headers) == 401
    assert backing.paths == []
    assert [r.getMessage() for r in caplog.records] == [
        f"refused {method} {unquote(urlsplit(path).path)}: it takes the env's control token"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", GUARDED)
async def test_the_data_plane_and_a_servers_own_endpoint_behind_the_proxy_answer_a_call_with_the_token(
        monkeypatch, method, path):
    _Backing(monkeypatch)
    assert await _proxied(TOKEN, method, path, {"Authorization": f"Bearer {TOKEN}"}) != 401


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", PROXIED_OPEN)
async def test_the_rest_of_the_proxy_stays_open(monkeypatch, method, path):
    backing = _Backing(monkeypatch)
    assert await _proxied(TOKEN, method, path, {}) != 401
    assert backing.paths


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", GUARDED)
async def test_a_gateway_started_without_a_token_guards_nothing_behind_its_proxy(monkeypatch, method, path):
    _Backing(monkeypatch)
    assert await _proxied(None, method, path, {}) != 401


@pytest.mark.asyncio
@pytest.mark.parametrize("path", SITE_REFUSED)
@pytest.mark.parametrize("headers", [{}, {"Authorization": f"Bearer {TOKEN}"}], ids=["none", "token"])
async def test_a_sites_route_never_reaches_its_reset_or_add_on_a_gateway_with_a_token(monkeypatch, path, headers):
    backing = _Backing(monkeypatch)
    assert await _proxied(TOKEN, "POST", path, headers) == 404
    assert [at for at in backing.paths if at != WELL_KNOWN_PATH] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", SITE_REFUSED)
async def test_a_sites_route_passes_its_reset_and_add_on_a_gateway_without_a_token(monkeypatch, path):
    backing = _Backing(monkeypatch)
    assert await _proxied(None, "POST", path, {}) == 200
    assert [at for at in backing.paths if at != WELL_KNOWN_PATH]


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", [("POST", "/svc/mcp-x/rpc"), ("GET", "/svc/mcp-x/rpc/"), ("POST", "/svc/mcp-x/agentenv")])
async def test_an_mcp_server_whose_endpoint_is_elsewhere_is_guarded_at_its_endpoint(monkeypatch, method, path):
    backing = _Backing(monkeypatch)
    servers = {"internal_mcp_servers": [InternalMCPServer(name="x", mcp_url="http://x:8000/rpc")],
               "rest_proxy_urls": {"mcp-x": "http://x:8000"}, "website_urls": {}}
    assert await _proxied(TOKEN, method, path, {}, **servers) == 401
    assert await _proxied(TOKEN, method, path, {"Authorization": f"Bearer {TOKEN}"}, **servers) != 401
    assert backing.paths == [path.removeprefix("/svc/mcp-x")]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/svc/mcp-x/mcp", "/svc/mcp-x/%6dcp", "/svc/mcp-x/public%2Fx/../mcp"])
async def test_an_mcp_server_whose_endpoint_is_configured_encoded_is_guarded_at_it(monkeypatch, path):
    backing = _Backing(monkeypatch)
    encoded = {"internal_mcp_servers": [InternalMCPServer(name="x", mcp_url="http://x:8000/%6dcp")],
               "rest_proxy_urls": {"mcp-x": "http://x:8000"}, "website_urls": {}}
    assert await _proxied(TOKEN, "POST", path, {}, **encoded) == 401
    assert backing.paths == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/svc/desk/agentenv/ext/browser/cdp", "/svc/desk/mcp"])
async def test_a_child_env_of_its_own_takes_its_own_token_through_the_proxy(monkeypatch, path):
    backing = _Backing(monkeypatch)
    computer = {"internal_mcp_servers": [InternalMCPServer(name="desk", mcp_url="http://desk:18765/mcp")],
                "rest_proxy_urls": {"desk": "http://desk:18765"}, "website_urls": {}}
    assert await _proxied(TOKEN, "POST", path, {"Authorization": "Bearer the-computers"}, **computer) != 401
    forwarded = [(at, sent) for at, sent in zip(backing.paths, backing.authorization) if at != WELL_KNOWN_PATH]
    assert forwarded == [(path.removeprefix("/svc/desk"), "Bearer the-computers")]


@pytest.mark.asyncio
async def test_a_live_websites_data_plane_is_guarded_though_it_has_no_website_route(monkeypatch):
    backing = _Backing(monkeypatch)
    live = {"rest_proxy_urls": {"github": "http://github-website-backend:8000"}, "website_urls": {}}
    assert await _proxied(TOKEN, "POST", "/svc/github/agentenv", {}, **live) == 401
    assert await _proxied(TOKEN, "POST", "/svc/github/agentenv", {"Authorization": f"Bearer {TOKEN}"}, **live) != 401
    assert backing.paths == ["/agentenv"]


@pytest.mark.asyncio
@pytest.mark.parametrize("base", ["http://x:8000/prefix", "http://x:8000/%70refix"], ids=["plain", "encoded"])
@pytest.mark.parametrize("path", ["/svc/mcp-x/../prefix/mcp", "/svc/mcp-x/../prefix/agentenv", "/svc/mcp-x/mcp",
                                  "/svc/mcp-x/public%2Fx/../agentenv"])
async def test_a_server_behind_a_base_path_is_guarded_however_a_path_climbs_into_it(monkeypatch, base, path):
    backing = _Backing(monkeypatch)
    prefixed = {"internal_mcp_servers": [InternalMCPServer(name="x", mcp_url=f"{base}/mcp")],
                "rest_proxy_urls": {"mcp-x": base}, "website_urls": {}}
    assert await _proxied(TOKEN, "POST", path, {}, **prefixed) == 401
    assert backing.paths == []
