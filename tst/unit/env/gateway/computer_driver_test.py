"""A computer's browser requests take the role of whoever drives the computer now, the last to make an input call to its
tools, and a computer's own calls claim none of the writes others make while they run."""
from __future__ import annotations

import asyncio
import contextvars
import json
from types import SimpleNamespace

import httpx
import pytest
from mcp.types import CallToolResult, TextContent, Tool as MCPTool, ToolAnnotations

from agent_env.env.gateway import InternalMCPServer
from agent_env.env.gateway import rest_proxy as rest_proxy_module
from agent_env.env.gateway.caller import Caller
from agent_env.env.gateway.constants import AGENT_ENV_COMPUTER_HEADER, AGENT_ENV_ROLE_HEADER
from agent_env.env.gateway.gateway import GATEWAY_TRAJECTORY_FILE, Gateway
from tst.unit.env.gateway.changes_test import _Changelog, _http, _RealAsyncClient, _site

MAIL, DESKTOP = "http://mail/mcp", "http://desktop/mcp"
_SCHEMA = {"type": "object", "properties": {}}
SEND = MCPTool(name="mail_send_message", inputSchema=_SCHEMA)
CLICK, TYPE, SCREENSHOT = (
    MCPTool(name=f"desktop_computer_{op}", inputSchema=_SCHEMA, annotations=ToolAnnotations(readOnlyHint=read))
    for op, read in (("click", False), ("type", False), ("screenshot", True)))
FROM_DESKTOP = {AGENT_ENV_COMPUTER_HEADER: "desktop"}


class _Child:
    """A child server whose every call runs `effect` with the tool's name first."""

    def __init__(self, tools: list[MCPTool], effect=lambda name: None):
        self.tools, self.effect = tools, effect

    async def list_tools(self):
        return SimpleNamespace(tools=self.tools)

    async def call_tool(self, name, arguments, *args, meta=None, **kwargs):
        self.effect(name)
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)


async def _office(changelog: _Changelog, monkeypatch, desktop_effect=lambda name: None) -> Gateway:
    """A gateway in front of a mail server, its site, and the computer `desktop`, whose site route writes a draft on
    every POST."""
    async def ok():
        yield b'{"ok": true}'

    def website(request: httpx.Request) -> httpx.Response:
        changelog.write("drafts", f"d{len(changelog.rows) + 1}")
        return httpx.Response(201, content=ok())

    transport = _site(website)
    monkeypatch.setattr(rest_proxy_module.httpx, "AsyncClient", lambda **kw: _RealAsyncClient(transport=transport, **kw))
    gw = Gateway(host="127.0.0.1", port=0, server_name="t",
                 internal_mcp_servers=[InternalMCPServer("mail", MAIL), InternalMCPServer("desktop", DESKTOP)],
                 website_urls={"mail": "http://mail-frontend"}, control_token_children=["desktop"])
    children = {MAIL: _Child([SEND], lambda name: changelog.write("messages", f"m{len(changelog.rows) + 1}")),
                DESKTOP: _Child([CLICK, TYPE, SCREENSHOT], desktop_effect)}

    async def _open(server, stack, report):
        return children[server.mcp_url]

    gw._sessions.open_session = _open  # type: ignore[method-assign]
    gw._query_changelog_id = changelog.max_id  # type: ignore[method-assign]
    gw._db_conn = changelog
    await gw._ensure_tools_discovered()
    return gw


async def _step(gw: Gateway, tool: MCPTool, role: str) -> None:
    response = await gw._step_call_tool({"tool_name": tool.name, "arguments": {}}, role)
    assert response.status_code == 200, response.body


async def _post(gw: Gateway, headers: dict) -> None:
    status, _ = await _http(gw, "POST", "/website/mail/api/drafts", headers=headers)
    assert status == 201


async def _actors(gw: Gateway) -> dict[str, list[tuple[str, str]]]:
    _, body = await _http(gw, "GET", "/changes")
    return {c["row_id"]: [(a["kind"], a["role"]) for a in c["actors"]] for c in json.loads(body)["changes"]}


@pytest.mark.asyncio
async def test_a_computers_browser_writes_are_its_drivers_and_its_calls_claim_no_one_elses(monkeypatch):
    changelog = _Changelog()
    gw = await _office(changelog, monkeypatch, desktop_effect=lambda name: changelog.write(
        "messages", f"world{len(changelog.rows) + 1}") if name == CLICK.name else None)
    try:
        await _step(gw, CLICK, "person")
        await _post(gw, FROM_DESKTOP)
        await _step(gw, SCREENSHOT, "helper")
        await _post(gw, FROM_DESKTOP)
        await _step(gw, TYPE, "helper")
        await _post(gw, FROM_DESKTOP)
        ctx = contextvars.copy_context()
        ctx.run(Caller(role="carol").bind)
        await ctx.run(asyncio.create_task, gw._mcp._tool_manager.call_tool(CLICK.name, {}))
        await _post(gw, FROM_DESKTOP)
        await _post(gw, {})
        await _post(gw, {AGENT_ENV_COMPUTER_HEADER: "laptop"})
        await _post(gw, {**FROM_DESKTOP, AGENT_ENV_ROLE_HEADER: "auditor"})
        actors = await _actors(gw)
    finally:
        await gw._close_child_sessions()

    assert actors == {"world1": [], "d2": [("request", "person")], "d3": [("request", "person")],
                      "d4": [("request", "helper")], "world5": [], "d6": [("request", "carol")],
                      "d7": [("request", "default")], "d8": [("request", "default")], "d9": [("request", "auditor")]}
    with open(GATEWAY_TRAJECTORY_FILE) as f:
        results = [json.loads(line) for line in f if '"tool_call_result"' in line]
    assert not any("changelog_id_before" in e for e in results[-4:])


@pytest.mark.asyncio
async def test_a_trigger_actions_input_on_a_computer_drives_it_as_the_role_it_acts_as(monkeypatch):
    changelog = _Changelog()
    gw = await _office(changelog, monkeypatch)
    gw._trigger_engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "nora-types", "when": {"type": "action", "tool": SEND.name},
         "actions": [{"type": "tool", "tool": TYPE.name, "as": "nora", "args": {}}]}]})
    try:
        await _step(gw, SEND, "default")
        for _ in range(100):
            if gw._trigger_engine.state()["triggers"][0]["status"] == "fired":
                break
            await asyncio.sleep(0.02)
        await _post(gw, FROM_DESKTOP)
        actors = await _actors(gw)
    finally:
        await gw._close_child_sessions()

    assert actors["d2"] == [("request", "nora")]
    assert actors["m1"] == [("tool_call", "default")]
