"""A child tool's arguments reach the child on the MCP proxy path exactly as on /step, unvalidated by the gateway,
so the child's own validation decides: an undeclared key on a tool closed with additionalProperties false, a missing
or misnamed required argument and a wrong-typed one each get the child's error naming the key, and a tool that allows
extras receives them. A tool the gateway serves itself still drops an undeclared key and answers."""
from __future__ import annotations

import json

import pytest
import pytest_asyncio
from jsonschema import Draft202012Validator
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
EVENTS = MCPTool(name="list_events", description="", inputSchema={
    "type": "object", "properties": {"calendarId": {"type": "string"}, "maxResults": {"type": "integer"}},
    "required": ["calendarId"], "additionalProperties": False})
TOOLS = [CLOSED, OPEN, EVENTS]


def _child(received: list[tuple[str, dict]]) -> Server:
    """A child on mcp's low-level server that validates every call against the tool's inputSchema and, like the hub's
    servers, refuses an invalid one with invalid_arguments naming every key at fault."""
    server = Server("child")

    @server.list_tools()
    async def list_tools():
        return TOOLS

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        schema = next(tool.inputSchema for tool in TOOLS if tool.name == name)
        if errors := list(Draft202012Validator(schema).iter_errors(arguments)):
            properties = schema["properties"]
            fields = {key for key in arguments if key not in properties and not schema["additionalProperties"]}
            fields |= {key for key in schema.get("required", []) if key not in arguments}
            fields |= {error.path[0] for error in errors if error.path}
            text = json.dumps({"error": "invalid_arguments", "invalid_fields": sorted(fields),
                               "messages": sorted(error.message for error in errors)})
            return CallToolResult(isError=True, content=[TextContent(type="text", text=text)])
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
    for tool in TOOLS:
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
@pytest.mark.parametrize(("arguments", "invalid_fields", "message"), [
    ({"calendar_id": "primary"}, ["calendarId", "calendar_id"], "'calendar_id' was unexpected"),
    ({"maxResults": 5}, ["calendarId"], "'calendarId' is a required property"),
    ({"calendarId": "primary", "maxResults": "[10]"}, ["maxResults"], "'[10]' is not of type 'integer'"),
], ids=["misnamed_required", "missing_required", "wrong_type"])
async def test_an_invalid_call_gets_the_childs_own_error_on_both_paths(gateway, arguments, invalid_fields, message):
    gw, received = gateway
    via_mcp = await _via_mcp(gw, EVENTS.name, arguments)
    via_step = await _via_step(gw, EVENTS.name, arguments)
    assert via_mcp == via_step
    is_error, text = via_mcp
    assert is_error is True
    error = json.loads(text)
    assert error["error"] == "invalid_arguments" and error["invalid_fields"] == invalid_fields
    assert any(message in m for m in error["messages"])
    assert received == []


@pytest.mark.asyncio
@pytest.mark.parametrize("call", CALLS)
async def test_a_valid_call_reaches_the_child_unchanged(gateway, call):
    gw, received = gateway
    arguments = {"calendarId": '{"id": "primary"}', "maxResults": 5}  # a string that parses as JSON stays a string
    is_error, text = await call(gw, EVENTS.name, arguments)
    assert is_error is False
    assert received == [(EVENTS.name, arguments)]
    assert json.loads(text) == arguments


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


@pytest.mark.asyncio
async def test_the_agent_sees_the_childs_schema_verbatim(gateway):
    gw, _ = gateway
    listed = {tool.name: tool.inputSchema for tool in await gw._mcp.list_tools()}
    assert {name: listed[name] for name in (t.name for t in TOOLS)} == {t.name: t.inputSchema for t in TOOLS}
