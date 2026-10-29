"""Backing servers that serve one tool name: discovery registers none of their tools, says which names clash and where,
and keeps saying so on /state, on /step and on every later discovery, rather than serving a partial tool set."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from mcp.types import Tool as MCPTool

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway.constants import STATE_TOOL_NAME_CLASHES_KEY
from agent_env.env.gateway.gateway import Gateway, ToolNameClash
from agent_env.env.gateway.get_time import GET_TIME_TOOL_NAME

T = "2019-06-15T12:00:00Z"


class _Session:
    def __init__(self, *names: str):
        self.tools = [MCPTool(name=n, description="", inputSchema={"type": "object", "properties": {}}) for n in names]
        self.lists = 0

    async def list_tools(self):
        self.lists += 1
        return SimpleNamespace(tools=self.tools)


def _gateway(servers: dict[str, _Session], **kw) -> Gateway:
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit", **kw,
                 internal_mcp_servers=[InternalMCPServer(name, f"http://{name}/mcp") for name in servers])
    sessions = {f"http://{name}/mcp": session for name, session in servers.items()}

    async def open_sessions(stack):
        return sessions

    gw._open_persistent_sessions = open_sessions  # type: ignore[method-assign]
    return gw


def _items_stock_ledger() -> dict[str, _Session]:
    return {"items": _Session("items_add_item", "list_items"), "stock": _Session("stock_add_item", "list_items"),
            "ledger": _Session("ledger_add_item")}


def _registered(gw: Gateway) -> set[str]:
    return set(gw._mcp._tool_manager._tools)


async def _request(gw: Gateway, method: str, path: str, **kw) -> httpx.Response:
    transport = httpx.ASGITransport(app=gw._mcp.streamable_http_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        return await client.request(method, path, **kw)


@pytest.mark.asyncio
async def test_servers_sharing_a_tool_name_register_none_of_their_tools():
    gw = _gateway(_items_stock_ledger())

    with pytest.raises(ToolNameClash) as raised:
        await gw._ensure_tools_discovered()

    assert raised.value.clashes == {"list_items": ["items", "stock"]}
    assert str(raised.value) == "Duplicate tool name 'list_items' from 'items' and 'stock'"
    assert _registered(gw) == set() and gw._tool_server_urls == {} and gw._server_tools == {}
    assert gw._tools_discovered is False
    await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_clash_is_named_with_every_source_and_every_name():
    gw = _gateway({"a": _Session("x", "y"), "b": _Session("x", "y"), "c": _Session("x")})

    with pytest.raises(ToolNameClash) as raised:
        await gw._ensure_tools_discovered()

    assert str(raised.value) == "Duplicate tool names 'x' from 'a', 'b' and 'c'; 'y' from 'a' and 'b'"
    await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_clash_found_once_is_raised_again_without_listing_the_servers_again():
    servers = _items_stock_ledger()
    gw = _gateway(servers)
    with pytest.raises(ToolNameClash):
        await gw._ensure_tools_discovered()

    with pytest.raises(ToolNameClash, match="'list_items' from 'items' and 'stock'"):
        await gw._ensure_tools_discovered()

    assert [s.lists for s in servers.values()] == [1, 1, 1]
    await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_state_reports_the_clash_and_step_refuses_naming_it():
    gw = _gateway(_items_stock_ledger())

    state = await _request(gw, "GET", "/state")
    step = await _request(gw, "POST", "/step", json={"action": "list_tools"})

    assert state.status_code == 200
    assert state.json()[STATE_TOOL_NAME_CLASHES_KEY] == {"list_items": ["items", "stock"]}
    assert state.json()["mcp_servers"] == []
    assert step.status_code == 503
    assert step.json()["error"] == ("Duplicate tool name 'list_items' from 'items' and 'stock', so the env serves none "
                                    "of its servers' tools.")
    await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_state_reports_no_clash_for_servers_that_name_their_tools_apart():
    gw = _gateway({"items": _Session("items_add_item"), "stock": _Session("stock_add_item")})

    state = (await _request(gw, "GET", "/state")).json()

    assert state[STATE_TOOL_NAME_CLASHES_KEY] == {}
    assert [s["name"] for s in state["mcp_servers"]] == ["items", "stock"]
    assert _registered(gw) == {"items_add_item", "stock_add_item"}
    await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_backing_tool_named_like_the_gateways_own_clashes_and_leaves_the_gateways_in_place():
    gw = _gateway({"web": _Session("list_website_urls", "web_search")}, website_urls={"web": "http://web:3000"})
    own = gw._mcp._tool_manager._tools["list_website_urls"]

    with pytest.raises(ToolNameClash) as raised:
        await gw._ensure_tools_discovered()

    assert raised.value.clashes == {"list_website_urls": ["gateway", "web"]}
    assert gw._mcp._tool_manager._tools == {"list_website_urls": own}
    await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_backing_get_time_still_wins_over_the_armed_clock_without_a_clash():
    gw = _gateway({"svc": _Session(GET_TIME_TOOL_NAME)})
    gw._clock.set_time(T, 1)
    gw._register_get_time_tool()

    await gw._ensure_tools_discovered()

    assert gw._tool_server_urls == {GET_TIME_TOOL_NAME: "http://svc/mcp"} and gw._tool_name_clashes == {}
    await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_server_that_fails_to_list_leaves_no_tool_of_the_servers_listed_before_it():
    class _Failing(_Session):
        async def list_tools(self):
            raise RuntimeError("listing failed")

    gw = _gateway({"items": _Session("items_add_item"), "stock": _Failing()})

    with pytest.raises(RuntimeError, match="listing failed"):
        await gw._ensure_tools_discovered()

    assert _registered(gw) == set() and gw._tool_server_urls == {} and gw._tool_name_clashes == {}
    await gw._close_child_sessions()


def test_a_clash_at_startup_leaves_the_gateway_serving(caplog):
    """Uvicorn exits on a raised startup; the gateway stays up to report the clash, and logs it once, without a trace."""
    gw = _gateway(_items_stock_ledger())

    async def startup_and_shutdown():
        lifespan = gw._process_lifespan(lambda app: _NoMcpLifespan())
        async with lifespan(None):
            pass

    asyncio.run(startup_and_shutdown())

    errors = [r for r in caplog.records if r.levelname == "ERROR" and "Duplicate tool name 'list_items'" in r.getMessage()]
    assert len(errors) == 1 and errors[0].exc_info is None


class _NoMcpLifespan:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False
