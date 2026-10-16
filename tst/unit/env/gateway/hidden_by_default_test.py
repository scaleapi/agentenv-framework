"""A tool its server lists with `_meta` {"agentenv.io/hiddenByDefault": true} is plumbing a task drives (a trigger's
tool action, a simulated party) that the agent under test must not see: discovery records it, and the MCP tools/list,
/step list_tools and both call paths hide it from every role until a rule names it for that role."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from mcp.server.fastmcp.exceptions import ToolError
from mcp.types import Tool as MCPTool

from agent_env.env.gateway import AGENT_ENV_HIDDEN_BY_DEFAULT_META_KEY, AGENT_ENV_ROLE_HEADER, InternalMCPServer
from agent_env.env.gateway.caller import Caller
from agent_env.env.gateway.gateway import Gateway

HIDDEN = "mail_insert_email"


def _tool(name: str, meta: dict | None = None) -> MCPTool:
    return MCPTool(name=name, description="", inputSchema={"type": "object", "properties": {}}, _meta=meta)


class _Session:
    def __init__(self, *tools: MCPTool):
        self.tools = list(tools)

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)


def _gateway(*tools: MCPTool) -> Gateway:
    gw = Gateway(host="127.0.0.1", port=0, server_name="unit",
                 internal_mcp_servers=[InternalMCPServer("mail", "http://mail/mcp")])

    async def open_sessions(stack):
        return {"http://mail/mcp": _Session(*tools)}

    gw._open_persistent_sessions = open_sessions  # type: ignore[method-assign]
    return gw


def _mail() -> Gateway:
    return _gateway(_tool("mail_send"), _tool(HIDDEN, {AGENT_ENV_HIDDEN_BY_DEFAULT_META_KEY: True}))


async def _request(gw: Gateway, method: str, path: str, **kw) -> httpx.Response:
    transport = httpx.ASGITransport(app=gw._mcp.streamable_http_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        return await client.request(method, path, **kw)


def _mcp_listed(gw: Gateway, role: str) -> set[str]:
    token = Caller(role=role).bind()
    try:
        return {t.name for t in gw._mcp._tool_manager.list_tools()}
    finally:
        Caller.reset(token)


async def _step_listed(gw: Gateway, role: str) -> set[str]:
    r = await _request(gw, "POST", "/step", json={"action": "list_tools"}, headers={AGENT_ENV_ROLE_HEADER: role})
    assert r.status_code == 200, r.text
    return {t["name"] for t in r.json()["tools"]}


@pytest.mark.asyncio
async def test_discovery_records_only_tools_flagged_true():
    gw = _gateway(_tool("a"), _tool("b", {AGENT_ENV_HIDDEN_BY_DEFAULT_META_KEY: True}),
                  _tool("c", {AGENT_ENV_HIDDEN_BY_DEFAULT_META_KEY: False}),
                  _tool("d", {AGENT_ENV_HIDDEN_BY_DEFAULT_META_KEY: "true"}), _tool("e", {"other": True}))
    await gw._ensure_tools_discovered()
    assert gw._hidden_by_default == {"b"}
    await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_hidden_from_both_tool_lists_for_every_role_with_no_rule_at_all():
    gw = _mail()
    await gw._ensure_tools_discovered()
    for role in ("default", "grader", "brand-new-role"):
        assert _mcp_listed(gw, role) == {"mail_send"}
        assert await _step_listed(gw, role) == {"mail_send"}
    await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_a_rule_naming_it_shows_it_to_that_role_only_and_wildcard_enables_do_not():
    gw = _mail()
    await gw._ensure_tools_discovered()
    for body in ({"role": "*", "tools": "*"}, {"role": "default", "tools": "*"}):
        assert (await _request(gw, "POST", "/tools/enable", json=body)).status_code == 200
    assert await _step_listed(gw, "default") == {"mail_send"}
    r = await _request(gw, "POST", "/tools/enable", json={"role": "sim", "tools": [HIDDEN]})
    assert r.status_code == 200 and HIDDEN in r.json()["allowed"]
    assert await _step_listed(gw, "sim") == {"mail_send", HIDDEN}
    assert _mcp_listed(gw, "sim") == {"mail_send", HIDDEN}
    assert await _step_listed(gw, "default") == {"mail_send"}
    await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_calling_it_by_name_is_refused_on_both_paths():
    gw = _mail()
    await gw._ensure_tools_discovered()
    r = await _request(gw, "POST", "/step", json={"action": "call_tool", "tool_name": HIDDEN, "arguments": {}})
    assert r.status_code == 403 and "disabled" in r.text, r.text
    token = Caller(role="default").bind()
    try:
        with pytest.raises(ToolError, match="disabled"):
            await gw._mcp._tool_manager.call_tool(HIDDEN, {})
    finally:
        Caller.reset(token)
    await gw._close_child_sessions()
