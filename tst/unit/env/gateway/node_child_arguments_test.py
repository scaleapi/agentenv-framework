"""A child that is not FastMCP (node_child/server.mjs: plain Node over MCP's streamable HTTP, validating strictly and
never re-parsing a JSON string) gets a tool's arguments from the gateway's MCP path exactly as the caller sent them:
an undeclared key reaches it (and it refuses it by name), declared ones arrive unchanged, and a string that parses as
JSON stays a string. Before the gateway stopped validating child arguments, the undeclared key was dropped and
'["a"]' arrived as the list ["a"], which this child would have accepted."""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest
import pytest_asyncio
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import CallToolResult
from pytest_socket import enable_socket

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway.gateway import Gateway

NODE = shutil.which("node")
SERVER = Path(__file__).parent / "node_child" / "server.mjs"
TOOL = "find_messages"

pytestmark = pytest.mark.skipif(NODE is None, reason="needs the node binary to run the Node child server")


@pytest.fixture(autouse=True)
def _disable_network():
    """Override the suite-wide socket block (tst/unit/conftest.py) for this module: the child listens on loopback."""
    enable_socket()
    yield


@pytest_asyncio.fixture
async def gateway():
    proc = await asyncio.create_subprocess_exec(NODE, str(SERVER), stdout=asyncio.subprocess.PIPE)
    try:
        port = int((await asyncio.wait_for(proc.stdout.readline(), timeout=30)).decode().strip())
        gw = Gateway(host="127.0.0.1", port=0, server_name="t",
                     internal_mcp_servers=[InternalMCPServer("node", f"http://127.0.0.1:{port}/mcp")])
        try:
            await gw._ensure_tools_discovered()
            yield gw
        finally:
            await gw._close_child_sessions()
    finally:
        proc.kill()
        await proc.wait()


async def _via_mcp(gw: Gateway, arguments: dict) -> tuple[bool, dict]:
    try:
        result = await gw._mcp._tool_manager.call_tool(TOOL, arguments)
    except ToolError as e:
        return True, {"gateway_error": str(e)}
    assert isinstance(result, CallToolResult), result
    return result.isError, json.loads(result.content[0].text)


async def _via_step(gw: Gateway, arguments: dict) -> tuple[bool, dict]:
    resp = await gw._step_call_tool({"tool_name": TOOL, "arguments": arguments}, "default")
    body = json.loads(resp.body)
    result = body.get("result", body)
    return result["isError"], json.loads(result["content"][0]["text"])


@pytest.mark.asyncio
async def test_the_gateway_lists_the_node_childs_schema_verbatim(gateway):
    listed = {tool.name: tool for tool in await gateway._mcp.list_tools()}
    assert listed[TOOL].inputSchema["additionalProperties"] is False
    assert set(listed[TOOL].inputSchema["properties"]) == {"q", "labels"}


@pytest.mark.asyncio
async def test_declared_arguments_reach_the_node_child_unchanged(gateway):
    arguments = {"q": "invoice", "labels": ["INBOX", "a b"]}
    assert await _via_mcp(gateway, arguments) == (False, arguments)
    assert await _via_step(gateway, arguments) == (False, arguments)


@pytest.mark.asyncio
async def test_an_undeclared_argument_reaches_the_node_child_which_refuses_it_by_name(gateway):
    arguments = {"q": "invoice", "query": "invoice"}
    expected = (True, {"error": "invalid_arguments", "invalid_fields": ["query"]})
    assert await _via_mcp(gateway, arguments) == expected
    assert await _via_step(gateway, arguments) == expected


@pytest.mark.asyncio
async def test_a_stringified_array_reaches_the_node_child_as_the_string(gateway):
    # the child never re-parses JSON strings, so it sees (and refuses) the string the caller sent
    arguments = {"labels": '["a"]'}
    expected = (True, {"error": "invalid_arguments", "invalid_fields": ["labels"]})
    assert await _via_mcp(gateway, arguments) == expected
    assert await _via_step(gateway, arguments) == expected
