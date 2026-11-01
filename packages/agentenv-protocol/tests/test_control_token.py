"""The control token: with CONTROL_TOKEN_ENV set when an env mounts, everything it serves but its MCP endpoint and its
card refuses a call without that token, or with another, before the route runs, and answers one with it; the MCP
endpoint and the card stay open, and without the variable every route is open. The client sends a call's own headers,
or else the headers a grant gives the address it calls."""
from __future__ import annotations

import contextlib

import anyio
import httpx
import pytest

pytest.importorskip("mcp.server.fastmcp")

from mcp import ClientSession  # noqa: E402
from mcp.client import streamable_http as _streamable_http  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from agentenv_protocol import (  # noqa: E402
    CONTROL_TOKEN_ENV,
    RPC_PATH,
    WELL_KNOWN_PATH,
    AgentEnvEnvironment,
    add_data,
    client,
    environment_card,
    extension,
    tool,
)

pytestmark = pytest.mark.skipif(not hasattr(_streamable_http, "streamable_http_client"),
                                reason="mcp without streamable_http_client")

_BASE = "http://127.0.0.1:8000"
RAN: list[str] = []


@environment_card(name="desk")
class _Desk(AgentEnvEnvironment):
    tool_access = True

    @tool()
    def read(self) -> str:
        return "read"

    @add_data
    async def load(self, parts: list) -> None:
        RAN.append("data/add")

    @extension("urn:agentenv:test-exec/v1", op="run")
    async def run(self, argv: list) -> dict:
        RAN.append("run")
        return {"ran": argv}


@contextlib.asynccontextmanager
async def _served():
    """An ``_Desk`` mounted on mcp's FastMCP as the environment's variables stand, and its REST and MCP clients."""
    app = FastMCP("desk", json_response=True)
    _Desk().mount(app)
    asgi = app.streamable_http_app()
    async with app.session_manager.run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url=_BASE) as rest:
            async with _streamable_http.streamable_http_client(f"{_BASE}/mcp", http_client=rest) as (read, write, _):
                async with ClientSession(read, write) as mcp:
                    await mcp.initialize()
                    yield rest, mcp


async def _calls(rest: httpx.AsyncClient, headers: dict) -> list[int]:
    """The status of a call to each framework-only route with ``headers``: the data plane, an extension, tool access."""
    load = {"jsonrpc": "2.0", "id": 1, "method": "data/add", "params": {"parts": [{"kind": "data", "data": {}}]}}
    return [(await rest.post(RPC_PATH, json=load, headers=headers)).status_code,
            (await rest.post("/agentenv/ext/test-exec/run", json={"argv": ["ls"]}, headers=headers)).status_code,
            (await rest.post("/tools/disable", json={"role": "agent", "tools": ["read"]}, headers=headers)).status_code]


@pytest.mark.asyncio
async def test_with_a_token_every_framework_only_route_refuses_a_call_without_it_before_it_runs(monkeypatch):
    monkeypatch.setenv(CONTROL_TOKEN_ENV, "s3cret")
    RAN.clear()

    with anyio.fail_after(30):
        async with _served() as (rest, mcp):
            without = await _calls(rest, {})
            wrong = await _calls(rest, client.control_headers("guess"))
            refusal = (await rest.post("/agentenv/ext/test-exec/run", json={"argv": ["ls"]})).json()
            ran_before = list(RAN)
            right = await _calls(rest, client.control_headers("s3cret"))
            card = await rest.get(WELL_KNOWN_PATH)
            tools = [t.name for t in (await mcp.list_tools()).tools]

    assert without == wrong == [401, 401, 401] and ran_before == []
    assert refusal["error"]["code"] == "unauthorized" and refusal["error"]["applied"] == "no"
    assert right == [200, 200, 200] and RAN == ["data/add", "run"]
    assert card.status_code == 200 and tools == ["read"]


@pytest.mark.asyncio
async def test_without_a_token_every_route_is_open(monkeypatch):
    monkeypatch.delenv(CONTROL_TOKEN_ENV, raising=False)

    with anyio.fail_after(30):
        async with _served() as (rest, _):
            assert await _calls(rest, {}) == [200, 200, 200]


def test_a_call_sends_its_own_headers_or_else_the_ones_its_grant_gives_an_address_under_the_envs():
    env = "http://desk:18765"

    with client.authorized({env: client.control_headers("s3cret"), "http://other:1": {},
                            None: client.control_headers("orphan")}):
        granted = [client.request_headers(url) for url in (env, f"{env}/svc/mail/agentenv", f"{env}0/x",
                                                            "http://other:1/x")]
        own = client.request_headers(f"{env}/agentenv", {"X-Mine": "1"})
        with client.authorized({env: client.control_headers("rotated")}):
            inner = client.request_headers(f"{env}/agentenv")
        outer = client.request_headers(f"{env}/agentenv")
    after = client.request_headers(f"{env}/agentenv")

    bearer = {"Authorization": "Bearer s3cret"}
    assert granted == [bearer, bearer, {}, {}] and own == {"X-Mine": "1"}
    assert inner == {"Authorization": "Bearer rotated"} and outer == bearer and after == {}
    assert client.control_headers(None) == {} and client.control_headers("") == {}


@pytest.mark.asyncio
async def test_invoke_extension_and_the_data_plane_send_the_token(monkeypatch):
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("authorization")))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"parts": []}})

    real = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    card = {"capabilities": {"extensions": [{"uri": "urn:agentenv:test-exec/v1", "params": {
        "methods": {"run": {"endpoint": "/agentenv/ext/test-exec/run"}}}}]}}

    await client.invoke_extension("http://desk", card, "urn:agentenv:test-exec/v1", {}, method="run",
                                  headers=client.control_headers("own"))
    await client.get_data("http://desk", headers=client.control_headers("own"))
    with client.authorized({"http://desk": client.control_headers("granted")}):
        await client.add_data("http://desk", [])
        await client.reset_data("http://desk", headers={})

    assert seen == [("/agentenv/ext/test-exec/run", "Bearer own"), ("/agentenv", "Bearer own"),
                    ("/agentenv", "Bearer granted"), ("/agentenv", None)]
