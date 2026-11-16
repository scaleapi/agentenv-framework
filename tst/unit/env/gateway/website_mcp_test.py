"""A website env that is an MCP server too: the gateway finds its MCP endpoint on its card and serves its tools as it
does an MCP env's, while the site's route keeps that endpoint, like the rest of what its card advertises, out of an
agent's browsing reach."""
from __future__ import annotations

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pytest_socket import enable_socket

from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER, AGENT_ENV_ROLE_META_KEY, InternalMCPServer
from agent_env.env.gateway import entrypoint as entrypoint_module
from agent_env.env.gateway.entrypoint import website_mcp_servers
from agent_env.env.gateway.gateway import Gateway, site_mcp_endpoints, site_reserved_paths
from agentenv_protocol import DataPart, ToolContext, environment_card, get_data, tool
from agentenv_protocol.website import WebsiteEnvironment, website
from tst.unit.env.gateway.process_scope_socket_test import _serve

CARD = "/.well-known/agent-env.json"


@pytest.mark.parametrize("declared", ["/mcp", "http://shop:8000/mcp", "/mcp?transport=http", "http://shop:8000/mcp/#x",
                                      "mcp", "./mcp", "mcp?transport=http"])
def test_a_sites_mcp_interface_is_one_of_its_reserved_paths_however_its_card_writes_it(declared):
    card = {"url": "/agentenv", "additionalInterfaces": [{"url": declared, "transport": "mcp"}, {"transport": "x"}]}
    assert site_reserved_paths(card) == frozenset({"/agentenv", "/mcp"})


def test_a_relative_mcp_interface_is_reserved_under_the_sites_root_as_discovery_resolves_it():
    card = {"additionalInterfaces": [{"url": "tools/mcp", "transport": "mcp"}]}
    assert site_reserved_paths(card) == site_reserved_paths(card, "/base") == frozenset({"/tools/mcp"})


def test_a_sites_mcp_endpoint_is_its_internal_servers_path_under_its_svc_base():
    servers = [InternalMCPServer("shop", "http://shop-backend:8000/tools/mcp?transport=http"),
               InternalMCPServer("slack", "http://slack:18765/mcp"),
               InternalMCPServer("odd", "http://elsewhere:9/mcp")]
    sites = {"shop": "http://shop-backend:8000", "odd": "http://odd-backend:8000"}
    assert site_mcp_endpoints(servers, sites, {**sites, "mcp-slack": "http://slack:18765"}) == {"shop": "/tools/mcp"}
    spelled = [InternalMCPServer("tls", "https://TLS.test:443/mcp"), InternalMCPServer("deep", "http://d:80/base/mcp")]
    bases = {"tls": "https://tls.test", "deep": "http://d/base/"}
    assert site_mcp_endpoints(spelled, bases, bases) == {"tls": "/mcp", "deep": "/mcp"}


@pytest.mark.parametrize(("base", "declared", "mcp_url"), [
    ("https://shop.test", "https://shop.test:443/mcp", "https://shop.test/mcp"),
    ("https://shop.test", "https://SHOP.test/mcp?transport=http", "https://shop.test/mcp?transport=http"),
    ("http://shop:8000", "/mcp", "http://shop:8000/mcp"),
    ("http://shop:8000", "http://shop:8001/mcp", None),
    ("http://shop:8000", "https://shop:8000/mcp", None),
    ("http://shop/base", "http://shop/base/mcp", "http://shop/base/mcp"),
    ("http://shop/base", "/mcp", "http://shop/base/mcp"),
])
def test_a_declared_mcp_url_on_the_sites_server_keeps_the_sites_own_spelling(base, declared, mcp_url):
    assert entrypoint_module._site_mcp_url(base, declared) == mcp_url
    if mcp_url is not None:
        servers = [InternalMCPServer("shop", mcp_url)]
        assert site_mcp_endpoints(servers, {"shop": base}, {"shop": base}) == {"shop": "/mcp"}


@pytest.mark.asyncio
async def test_a_site_behind_a_path_prefix_is_found_at_the_full_url_its_card_declares(monkeypatch):
    def site(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/base" + CARD
        return httpx.Response(200, json={"additionalInterfaces": [{"url": "http://shop/base/mcp", "transport": "mcp"}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(entrypoint_module.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(site), **kw))
    assert await website_mcp_servers({"shop": "http://shop/base"}) == [InternalMCPServer("shop", "http://shop/base/mcp")]


@pytest.mark.asyncio
async def test_the_sites_whose_cards_declare_an_mcp_interface_are_internal_servers(monkeypatch):
    calls: dict[str, int] = {}
    cards = {
        "shop": {"additionalInterfaces": [{"url": "/tools/mcp", "transport": "mcp"}]},
        "mail": {"additionalInterfaces": []},
        "other": {"additionalInterfaces": [{"url": "/rpc", "transport": "grpc"}]},
        "full": {"additionalInterfaces": [{"url": "http://full:8000/tools/mcp", "transport": "mcp"}]},
        "away": {"additionalInterfaces": [{"url": "http://elsewhere:8000/mcp", "transport": "mcp"}]},
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
    sites = {name: f"http://{name}:8000/" for name in ("shop", "mail", "other", "late", "nocard", "html", "full", "away",
                                                       "slack")}

    found = await website_mcp_servers(sites, taken=frozenset({"slack"}))

    assert found == [InternalMCPServer("shop", "http://shop:8000/tools/mcp"), InternalMCPServer("late", "http://late:8000/mcp"),
                     InternalMCPServer("full", "http://full:8000/tools/mcp")]
    assert "slack" not in calls and calls["late"] == 2


@pytest.mark.asyncio
async def test_a_site_whose_card_cant_be_read_in_time_stops_the_gateway_starting(monkeypatch):
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    real = httpx.AsyncClient
    monkeypatch.setattr(entrypoint_module.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(down), **kw))
    monkeypatch.setattr(entrypoint_module, "_INTERNAL_SERVER_POLL_S", 0)
    with pytest.raises(RuntimeError, match="not readable after 0s: shop"):
        await website_mcp_servers({"shop": "http://shop:8000"}, timeout_s=0)


@pytest.mark.asyncio
async def test_sites_are_asked_at_once_and_the_one_never_up_is_named(monkeypatch):
    calls: dict[str, int] = {}

    def site(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls[host] = calls.get(host, 0) + 1
        if host == "down" or calls[host] < 3:
            raise httpx.ConnectError("not up yet")
        return httpx.Response(200, json={"additionalInterfaces": [{"url": "/mcp", "transport": "mcp"}]})

    real = httpx.AsyncClient
    monkeypatch.setattr(entrypoint_module.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(site), **kw))
    monkeypatch.setattr(entrypoint_module, "_INTERNAL_SERVER_POLL_S", 0.01)
    with pytest.raises(RuntimeError, match=r"not readable after 1s: down$"):
        await website_mcp_servers({"down": "http://down", "late": "http://late"}, timeout_s=1)
    assert calls["late"] == 3


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
            forged = {AGENT_ENV_ROLE_META_KEY: "mallory@example.com"}
            async with (
                httpx.AsyncClient(headers={AGENT_ENV_ROLE_HEADER: "bob@example.com"}, timeout=30) as http,
                streamable_http_client(f"{gw_url}/svc/shop/mcp", http_client=http) as (read, write, _),
                ClientSession(read, write) as session,
            ):
                await session.initialize()
                assert not (await session.call_tool("place_order", {"item": "eggs"}, meta=forged)).isError
            async with httpx.AsyncClient(timeout=30) as http:
                browsing = await http.post(f"{gw_url}/website/shop/mcp", json={"jsonrpc": "2.0", "id": 1,
                                                                              "method": "tools/list"})
                data = await http.post(f"{gw_url}/svc/shop/agentenv", json={"jsonrpc": "2.0", "id": 1,
                                                                            "method": "data/get"})
        finally:
            await gw_server.shutdown()
    finally:
        await site_server.shutdown()
    assert site.orders == [{"item": "milk", "by": "alice@example.com"}, {"item": "eggs", "by": "bob@example.com"}]
    assert browsing.status_code == 404
    assert data.json()["result"]["parts"][0]["data"]["orders"][0]["item"] == "milk"


def test_each_sites_names_are_read_from_their_variable_and_a_malformed_entry_is_refused():
    assert entrypoint_module.parse_site_names(" shop=shop.test|*.shop.test , docs=docs.test,") == {
        "shop": ("shop.test", "*.shop.test"), "docs": ("docs.test",)}
    assert entrypoint_module.parse_site_names("") == {}
    with pytest.raises(ValueError, match="WEBSITE_NAMES entry: 'shop.test'"):
        entrypoint_module.parse_site_names("shop.test")
