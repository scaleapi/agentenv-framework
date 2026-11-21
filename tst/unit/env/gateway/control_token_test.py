"""The gateway's control routes, the data plane and an MCP server's own endpoint behind its proxy take the deploy's
control token when it has one; every other route stays open."""
from __future__ import annotations

import asyncio
import logging
from urllib.parse import unquote, urlsplit

import httpx
import pytest
from starlette.responses import JSONResponse

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.constants import CONTROL_ROUTES, WELL_KNOWN_PATH
from agent_env.env.gateway.gateway import Gateway

TOKEN = "s3cret"
# A multi env's computers' own tokens, by name, as the gateway is started with them (CHILD_CONTROL_TOKENS).
DESK, LAPTOP = "the-desks-own", "the-laptops-own"
COMPUTERS = {"desk": DESK, "laptop": LAPTOP}
CONTROL = {
    "/tools/disable": "POST", "/tools/enable": "POST",
    "/triggers/register": "POST", "/triggers/remove": "POST", "/triggers/clear": "POST",
    "/notifications/config": "POST", "/notifications/take": "POST", "/notifications/clear": "POST",
    "/clock/followers": "PUT",
}
OPEN = {"/state": "GET", "/step": "POST", "/clock/time": "GET", "/clock/state": "GET", "/triggers/state": "GET",
        "/notifications/state": "GET", "/trajectory": "GET", "/changes": "GET", "/.well-known/agent-env.json": "GET",
        "/clock/set-time": "PUT", "/clock/advance": "PUT", "/clock/clear": "POST"}


# Behind the proxy: the data plane, its extensions, each service's included, and an MCP server's own endpoint, however
# the path is spelt on its way to the gateway.
GUARDED = [("POST", "/agentenv"), ("POST", "/svc/mcp-svc/agentenv"), ("POST", "/svc/mcp-svc/agentenv/ext/grant"),
           ("POST", "/svc/mcp-svc/x/../agentenv"), ("POST", "/svc/mcp-svc/%61gentenv"), ("POST", "/svc/shop/agentenv"),
           ("POST", "/svc/mcp-svc/mcp"), ("GET", "/svc/mcp-svc/mcp/"),
           ("DELETE", "/svc/mcp-svc/./mcp"), ("POST", "/svc/mcp-svc//mcp"), ("POST", "/svc/mcp-svc/public%2Fx/../agentenv"),
           ("POST", "/svc/mcp-svc/public%2Fx/../mcp"), ("POST", "/svc/mcp-svc/public%2Fx/../agentenv/ext/set_acting_user"),
           ("POST", "/svc/mcp-svc/AgentEnv"), ("POST", "/svc/mcp-svc/MCP")]
PROXIED_OPEN = [("GET", "/svc/mcp-svc/.well-known/agent-env.json"), ("GET", "/svc/mcp-svc/openapi.yaml"),
                ("GET", "/svc/mcp-svc/export-state"), ("POST", "/svc/shop/api/orders"), ("POST", "/svc/shop/mcp"),
                ("GET", "/website/shop/index.html")]
_RealAsyncClient = httpx.AsyncClient


def _client(token: str | None) -> httpx.AsyncClient:
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], control_token=token)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=gw._asgi_app()), base_url="http://gateway")


def test_the_routes_under_test_are_the_ones_the_gateway_covers():
    assert set(CONTROL) == CONTROL_ROUTES


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


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(OPEN))
async def test_a_gateway_started_without_a_token_lets_a_childs_through_nowhere_but_the_child(path):
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", internal_mcp_servers=[], control_token=None,
                 child_control_tokens={"desk": DESK})

    @gw._mcp.custom_route("/echo", methods=["GET"])
    async def echo(request):
        return JSONResponse({"authorization": request.headers.get("authorization")})

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw._asgi_app()), base_url="http://gateway") as client:
        response = await client.get("/echo", headers={"Authorization": f"Bearer {DESK}"})
    assert response.json() == {"authorization": None}

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
        self.hosts: list[str] = []
        self.authorization: list[str | None] = []
        transport = httpx.MockTransport(self._answer)
        monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.hosts.append(request.url.host)
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
             "scheme": "http", "path": unquote(url.path), "raw_path": url.path.encode(), "query_string": b"",
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
    assert [r.getMessage() for r in caplog.records] == [f"refused {method} {unquote(path)}: it takes the env's control token"]


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


def _computers(**extra) -> dict:
    """A multi env's two computers behind the gateway, each with its own token, beside an MCP server, a website and
    the SQL consoles."""
    return {"internal_mcp_servers": [InternalMCPServer(name="desk", mcp_url="http://desk:18765/mcp"),
                                     InternalMCPServer(name="laptop", mcp_url="http://laptop:18765/mcp")],
            "rest_proxy_urls": {"mcp-svc": "http://svc", "shop": "http://shop-backend", "desk": "http://desk:18765",
                                "laptop": "http://laptop:18765"},
            "website_urls": {"shop": "http://shop-backend"},
            "console_urls": {"pgweb": "http://pgweb:8081", "db-mcp": "http://db-mcp:8000"},
            "child_control_tokens": COMPUTERS, **extra}


def _reached(backing: _Backing) -> list[tuple[str, str | None]]:
    return [(at, sent) for at, sent in zip(backing.paths, backing.authorization) if at != WELL_KNOWN_PATH]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/svc/desk/agentenv", "/svc/desk/agentenv/ext/computer-exec", "/svc/desk/mcp",
                                  "/svc/desk/svc/notes/api"])
async def test_the_gateways_token_reaches_a_computer_as_that_computers_own(monkeypatch, path):
    """A harness caller holding the multi env's token reaches its computer's framework-only routes; the computer
    receives its own token, never the gateway's."""
    backing = _Backing(monkeypatch)
    assert await _proxied(TOKEN, "POST", path, {"Authorization": f"Bearer {TOKEN}"}, **_computers()) != 401
    assert _reached(backing) == [(path.removeprefix("/svc/desk"), f"Bearer {DESK}")]


@pytest.mark.asyncio
@pytest.mark.parametrize("path, reached", [
    ("/svc/desk/agentenv", ("/agentenv", f"Bearer {DESK}")),
    ("/svc/laptop/agentenv", ("/agentenv", None)),
    ("/svc/laptop/agentenv/ext/computer-exec", ("/agentenv/ext/computer-exec", None)),
    ("/svc/mcp-svc/export-state", ("/export-state", None)),
    ("/svc/shop/api/orders", ("/api/orders", None)),
    ("/website/shop/index.html", ("/index.html", None))])
async def test_a_computers_token_goes_on_only_to_that_computer(monkeypatch, path, reached):
    """A computer's token, read from it, carries to no other computer, server or site behind the gateway."""
    backing = _Backing(monkeypatch)
    await _proxied(TOKEN, "POST", path, {"Authorization": f"Bearer {DESK}"}, **_computers())
    assert _reached(backing) == [reached]


@pytest.mark.asyncio
@pytest.mark.parametrize("method, path", [
    ("GET", "/svc/pgweb/"), ("POST", "/svc/pgweb/api/query"), ("POST", "/svc/db-mcp/mcp"),
    *GUARDED])
async def test_a_computers_token_opens_no_route_that_takes_the_gateways(monkeypatch, method, path):
    """A computer's token is no key to the SQL consoles, the data plane or an MCP server's endpoint."""
    backing = _Backing(monkeypatch)
    assert await _proxied(TOKEN, method, path, {"Authorization": f"Bearer {DESK}"}, **_computers()) == 401
    assert backing.paths == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(CONTROL))
async def test_a_computers_token_opens_none_of_the_gateways_control_routes(path):
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", control_token=TOKEN, **_computers())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gw._asgi_app()), base_url="http://gateway") as client:
        response = await client.request(CONTROL[path], path, json={}, headers={"Authorization": f"Bearer {DESK}"})
    assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/svc/desktop/agentenv/ext/computer-exec", "/svc/desktop/mcp"])
async def test_a_computer_an_older_agent_env_booted_with_the_gateways_token_still_takes_it(monkeypatch, path):
    """An older agent-env names its computers in CONTROL_TOKEN_CHILDREN, each booted with the gateway's own token."""
    backing = _Backing(monkeypatch)
    older = {"internal_mcp_servers": [], "rest_proxy_urls": {"desktop": "http://desktop:18765"}, "website_urls": {},
             "control_token_children": ["desktop"]}
    await _proxied(TOKEN, "POST", path, {"Authorization": f"Bearer {TOKEN}"}, **older)
    assert [sent for _, sent in _reached(backing)] == [f"Bearer {TOKEN}"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/svc/desk/../pgweb/api/query", "/svc/desk/%2E%2E/pgweb/api/query",
                                  "/svc/desk/..%2Fpgweb/api/query"])
async def test_a_computers_route_that_climbs_out_of_it_reaches_only_that_computer(monkeypatch, path):
    """Whatever a path under a computer's route spells, its token goes only to that computer, never to a console."""
    backing = _Backing(monkeypatch)
    await _proxied(TOKEN, "GET", path, {"Authorization": f"Bearer {TOKEN}"}, **_computers())
    assert set(backing.hosts) <= {"desk"}
    assert set(backing.authorization) <= {f"Bearer {DESK}"}
