"""A tool its server lists with `_meta` {"agentenv.io/hiddenByDefault": true} is plumbing a task drives (a trigger's
tool action, a simulated party) that the agent under test must not see. Being hidden is a property of the tool:
discovery records it, and the MCP tools/list, /step list_tools and both call paths hide it from every role until a
rule names it. Rules naming it work as for any tool; rules for every tool (`"*"`) never show it, though they still
replace earlier named rules as they do for any tool."""
from __future__ import annotations

import random
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


def _expected_disabled(steps: list[tuple[str, str, bool]], role: str, tool: str, hidden: bool) -> bool:
    """The documented behavior, read from the steps alone (newest first), as a task author would: the newest step
    that covers `role` and `tool` decides. A step for every role and every tool also wipes all earlier steps, and a
    step for every tool of `role` wipes that role's earlier steps that named a tool. A step for every tool (`"*"`)
    never covers a hidden tool, so a hidden tool no step names is disabled; any other tool is enabled."""
    role_wiped = False
    for step_role, step_tool, disabled in reversed(steps):
        if step_role == "*" and step_tool == "*":
            return hidden or disabled
        if step_tool == "*":
            if step_role == role:
                if not hidden:
                    return disabled
                role_wiped = True
            continue
        if step_tool != tool:
            continue
        if step_role == "*" or (step_role == role and not role_wiped):
            return disabled
    return hidden


ROLES = ("default", "grader", "sim")


def test_every_mix_of_steps_matches_the_documented_behavior():
    """Thousands of random step sequences mixing a hidden and an ordinary tool, named and `"*"` tools, one role and
    every role, enables and disables; every role, including one no step mentions, sees what the rule above says."""
    rng = random.Random(20261016)
    for _ in range(5000):
        gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[])
        gw._hidden_by_default = {"h"}
        steps = [(rng.choice(ROLES + ("*",)), rng.choice(("h", "n", "*")), rng.random() < 0.5)
                 for _ in range(rng.randint(0, 8))]
        for step in steps:
            gw._apply_rule(*step)
        for role in ROLES + ("never-mentioned",):
            for tool, hidden in (("h", True), ("n", False)):
                assert gw._is_disabled(role, tool) is _expected_disabled(steps, role, tool, hidden), (steps, role, tool)


@pytest.mark.parametrize("steps,role,shown", [
    ([("*", "*", False)], "default", False),
    ([("*", "*", False), ("sim", "*", False)], "sim", False),
    ([("*", "*", False), ("sim", HIDDEN, False)], "sim", True),
    ([("*", "*", False), ("sim", HIDDEN, False)], "default", False),
    ([("sim", "*", True), ("sim", HIDDEN, False)], "sim", True),
    ([("sim", HIDDEN, False), ("sim", "*", True)], "sim", False),
    ([("sim", HIDDEN, False), ("sim", "*", False)], "sim", False),
    ([("sim", HIDDEN, False), ("*", "*", False)], "sim", False),
    ([("sim", HIDDEN, False), ("sim", HIDDEN, True)], "sim", False),
    ([("*", HIDDEN, False)], "default", True),
    ([("*", HIDDEN, False), ("default", "*", False)], "default", True),
    ([("*", HIDDEN, False), ("default", HIDDEN, True)], "default", False),
    ([("*", HIDDEN, False), ("default", HIDDEN, True)], "grader", True),
])
def test_the_steps_a_task_writes(steps, role, shown):
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[])
    gw._hidden_by_default = {HIDDEN}
    for step in steps:
        gw._apply_rule(*step)
    assert gw._is_disabled(role, HIDDEN) is (not shown)
    assert gw._is_disabled(role, "mail_send") is _expected_disabled(steps, role, "mail_send", False)


@pytest.mark.asyncio
async def test_a_trigger_permission_action_naming_it_shows_it_and_one_for_every_tool_does_not():
    gw = _mail()
    await gw._ensure_tools_discovered()
    run = gw._trigger_engine._run_action
    await run("grant", {}, 0, {"type": "permission", "action": "enable", "role": "sim", "tools": ["*"]}, {}, 0.0)
    assert await _step_listed(gw, "sim") == {"mail_send"}
    await run("grant", {}, 0, {"type": "permission", "action": "enable", "role": "sim", "tools": [HIDDEN]}, {}, 0.0)
    assert await _step_listed(gw, "sim") == {"mail_send", HIDDEN}
    await run("grant", {}, 0, {"type": "permission", "action": "disable", "role": "sim", "tools": [HIDDEN]}, {}, 0.0)
    assert await _step_listed(gw, "sim") == {"mail_send"}
    await gw._close_child_sessions()


@pytest.mark.asyncio
async def test_state_reports_a_named_grant_like_any_other_rule():
    gw = _mail()
    await gw._ensure_tools_discovered()
    await _request(gw, "POST", "/tools/enable", json={"role": "sim", "tools": [HIDDEN]})
    roles = (await _request(gw, "GET", "/state")).json()["roles"]
    assert roles["sim"] == {"disabled": [], "allowed": [HIDDEN]}
    await gw._close_child_sessions()
