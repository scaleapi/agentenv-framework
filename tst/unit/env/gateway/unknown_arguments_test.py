"""Arguments a tool's schema doesn't declare reach the child on the MCP proxy path exactly as on /step, so the
child's own validation decides: a tool closed with additionalProperties false answers an error naming the key,
and one that allows extras receives them. A tool the gateway serves itself still drops them and answers."""
from __future__ import annotations

import json

import pytest
import pytest_asyncio
from mcp.server.fastmcp.exceptions import ToolError
from mcp.server.lowlevel import Server
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, TextContent, Tool as MCPTool

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway.gateway import Gateway
from agent_env.env.gateway.get_time import GET_TIME_TOOL_NAME, build_get_time_tool
from agent_env.env.gateway.notifications import build_check_tools
from agent_env.env.gateway.user_chat import build_stay_quiet_tool, build_user_chat_tool

URL = "http://backing/mcp"
CLOSED = MCPTool(name="search", description="", inputSchema={
    "type": "object", "properties": {"q": {"type": "string"}, "limit": {"type": "integer"}},
    "additionalProperties": False})
OPEN = MCPTool(name="tag", description="", inputSchema={
    "type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"],
    "additionalProperties": True})


def _child(received: list[tuple[str, dict]]) -> Server:
    """A child built on mcp's low-level server, which validates every call against the tool's inputSchema."""
    server = Server("child")

    @server.list_tools()
    async def list_tools():
        return [CLOSED, OPEN]

    @server.call_tool()
    async def call_tool(name, arguments):
        received.append((name, dict(arguments)))
        return [TextContent(type="text", text=json.dumps(arguments, sort_keys=True))]

    return server


@pytest_asyncio.fixture
async def gateway():
    received: list[tuple[str, dict]] = []
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[InternalMCPServer("svc", URL)])

    async def _open(server, stack, report):
        return await stack.enter_async_context(create_connected_server_and_client_session(_child(received)))

    gw._sessions.open_session = _open  # type: ignore[method-assign]
    for tool in (CLOSED, OPEN):
        gw._create_and_register_proxy_tool(URL, tool)
        gw._tool_server_urls[tool.name] = URL
    try:
        yield gw, received
    finally:
        await gw._close_child_sessions()


async def _via_mcp(gw: Gateway, name: str, arguments: dict) -> tuple[bool, str]:
    try:
        result = await gw._mcp._tool_manager.call_tool(name, arguments)
    except ToolError as e:  # FastMCP's server answers a raised ToolError as isError true
        return True, str(e)
    if isinstance(result, CallToolResult):
        return result.isError, result.content[0].text
    return False, json.dumps(result)


async def _via_step(gw: Gateway, name: str, arguments: dict) -> tuple[bool, str]:
    resp = await gw._step_call_tool({"tool_name": name, "arguments": arguments}, "default")
    assert resp.status_code == 200
    body = json.loads(resp.body)
    result = body.get("result", body)
    return result["isError"], result["content"][0]["text"]


CALLS = [_via_mcp, _via_step]


@pytest.mark.asyncio
@pytest.mark.parametrize("call", CALLS)
async def test_an_undeclared_key_on_a_closed_schema_is_an_error_naming_it(gateway, call):
    gw, received = gateway
    is_error, text = await call(gw, CLOSED.name, {"query": "invoices"})
    assert is_error is True
    assert "query" in text
    assert received == []


@pytest.mark.asyncio
@pytest.mark.parametrize("call", CALLS)
async def test_extras_reach_a_tool_whose_schema_allows_them(gateway, call):
    gw, received = gateway
    is_error, text = await call(gw, OPEN.name, {"name": "urgent", "color": "red"})
    assert is_error is False
    assert received == [(OPEN.name, {"name": "urgent", "color": "red"})]
    assert json.loads(text) == {"name": "urgent", "color": "red"}


@pytest.mark.asyncio
@pytest.mark.parametrize("call", CALLS)
async def test_declared_arguments_are_forwarded_as_sent(gateway, call):
    gw, received = gateway
    is_error, _ = await call(gw, CLOSED.name, {"q": "invoices", "limit": None})
    assert is_error is True  # an explicit null is forwarded, and the child's schema refuses it
    is_error, _ = await call(gw, CLOSED.name, {"q": "invoices"})
    assert is_error is False
    assert received == [(CLOSED.name, {"q": "invoices"})]  # an unset optional stays unset


@pytest.mark.asyncio
@pytest.mark.parametrize("call", CALLS)
async def test_a_gateway_tool_drops_an_undeclared_key_and_answers(gateway, call):
    gw, _ = gateway
    gw._clock.set_time("2019-06-15T12:00:00Z", 0)
    gw._register_get_time_tool()
    is_error, text = await call(gw, GET_TIME_TOOL_NAME, {"timezone": "Asia/Tokyo"})
    assert is_error is False and "2019-06-15T12:00:00Z" in text


@pytest.mark.asyncio
async def test_every_gateway_tool_drops_an_undeclared_key():
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[])
    make = gw._create_arg_model_from_schema
    tools = [build_get_time_tool(gw._clock, make), *build_check_tools(gw._notifications, make),
             build_user_chat_tool(gw._user_chat, make), build_stay_quiet_tool(gw._user_chat, make)]
    for tool in tools:
        args = tool.fn_metadata.arg_model.model_validate({"bogus": 1, "note": "n", "check_id": "c", "text": "t"})
        assert "bogus" not in args.model_dump_one_level(), tool.name
