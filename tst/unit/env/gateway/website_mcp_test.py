"""A website env that is an MCP server too: the gateway finds its MCP endpoint on its card and serves its tools as it
does an MCP env's, while the site's route keeps that endpoint, like the rest of what its card advertises, out of an
agent's browsing reach."""
from __future__ import annotations

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pytest_socket import enable_socket

from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER, InternalMCPServer
from agent_env.env.gateway import entrypoint as entrypoint_module
from agent_env.env.gateway.entrypoint import website_mcp_servers
from agent_env.env.gateway.gateway import Gateway, site_reserved_paths
from agentenv_protocol import DataPart, ToolContext, environment_card, get_data, tool
from agentenv_protocol.website import WebsiteEnvironment, website
from tst.unit.env.gateway.process_scope_socket_test import _serve

CARD = "/.well-known/agent-env.json"


def test_a_sites_mcp_interface_is_one_of_its_reserved_paths():
    card = {"url": "/agentenv", "additionalInterfaces": [{"url": "/mcp", "transport": "mcp"}, {"transport": "x"}]}
    assert site_reserved_paths(card) == frozenset({"/agentenv", "/mcp"})


@pytest.mark.asyncio
async def test_the_sites_whose_cards_declare_an_mcp_interface_are_internal_servers(monkeypatch):
    calls: dict[str, int] = {}
    cards = {
        "shop": {"additionalInterfaces": [{"url": "/tools/mcp", "transport": "mcp"}]},
        "mail": {"additionalInterfaces": []},
        "other": {"additionalInterfaces": [{"url": "/rpc", "transport": "grpc"}]},
    }

    def site(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls[host] = calls.get(host, 0) + 1
        assert request.url.path == CARD
        if host == "late" and calls[host] == 1:
            raise httpx.ConnectError("not up yet")
        if host == "late":
            return httpx.Response(200, json={"additionalInterfaces": [{"url": "/mcp", "transport": "mcp"}]})
        if host == "nocard":
            return httpx.Response(404)
        if host == "html":
            return httpx.Response(200, text="<html></html>")
        return httpx.Response(200, json=cards[host])

    real = httpx.AsyncClient
    monkeypatch.setattr(entrypoint_module.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(site), **kw))
    monkeypatch.setattr(entrypoint_module, "_INTERNAL_SERVER_POLL_S", 0)
    sites = {name: f"http://{name}:8000/" for name in ("shop", "mail", "other", "late", "nocard", "html", "slack")}

    found = await website_mcp_servers(sites, taken=frozenset({"slack"}))

    assert found == [InternalMCPServer("shop", "http://shop:8000/tools/mcp"), InternalMCPServer("late", "http://late:8000/mcp")]
    assert "slack" not in calls and calls["late"] == 2


@pytest.mark.asyncio
async def test_a_site_whose_server_never_answers_is_left_out_once_the_time_is_up(monkeypatch):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    real = httpx.AsyncClient
    monkeypatch.setattr(entrypoint_module.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(down), **kw))
    monkeypatch.setattr(entrypoint_module, "_INTERNAL_SERVER_POLL_S", 0)
    assert await website_mcp_servers({"shop": "http://shop:8000"}, timeout_s=0) == []


@environment_card(name="shop")
@website(domains=["shop.test"])
class _Shop(WebsiteEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.orders: list[dict] = []

    @tool()
    def place_order(self, item: str, ctx: ToolContext) -> str:
        """Place an order for item, as the caller."""
        self.orders.append({"item": item, "by": ctx.role})
        return f"order {len(self.orders)} placed"

    @get_data
    def export(self) -> list:
        return [DataPart(data={"orders": self.orders})]


@pytest.mark.asyncio
async def test_the_gateway_serves_a_sites_tools_and_its_route_refuses_the_sites_mcp_endpoint():
    enable_socket()
    site = _Shop()
    site_server, site_url = await _serve(site.create_app())
    try:
        servers = await website_mcp_servers({"shop": site_url})
        assert servers == [InternalMCPServer("shop", f"{site_url}/mcp")]
        gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=servers,
                     website_urls={"shop": site_url}, rest_proxy_urls={"shop": site_url})
        gw_server, gw_url = await _serve(gw._asgi_app())
        try:
            async with (
                httpx.AsyncClient(headers={AGENT_ENV_ROLE_HEADER: "alice@example.com"}, timeout=30) as http,
                streamable_http_client(f"{gw_url}/mcp", http_client=http) as (read, write, _),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                assert "place_order" in [t.name for t in (await session.list_tools()).tools]
                result = await session.call_tool("place_order", {"item": "milk"})
                assert not result.isError and result.content[0].text == "order 1 placed"
            async with httpx.AsyncClient(timeout=30) as http:
                browsing = await http.post(f"{gw_url}/website/shop/mcp", json={"jsonrpc": "2.0", "id": 1,
                                                                              "method": "tools/list"})
                data = await http.post(f"{gw_url}/svc/shop/agentenv", json={"jsonrpc": "2.0", "id": 1,
                                                                            "method": "data/get"})
        finally:
            await gw_server.shutdown()
    finally:
        await site_server.shutdown()
    assert site.orders == [{"item": "milk", "by": "alice@example.com"}]
    assert browsing.status_code == 404
    assert data.json()["result"]["parts"][0]["data"]["orders"][0]["item"] == "milk"
