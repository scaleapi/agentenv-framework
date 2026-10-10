"""Who is calling rides on every child tools/call as MCP request `_meta`, written by the gateway from what it
attributed the caller itself: the inbound MCP session's role and key on the proxy path, the AgentEnv-Role header on
/step, a tool action's literal `as` in the trigger engine. Nothing a caller wrote in its own `_meta` is copied."""
from __future__ import annotations

import asyncio
import contextvars
import json
from types import SimpleNamespace

import anyio
import pytest
from mcp.shared.exceptions import McpError
from mcp.types import CONNECTION_CLOSED, CallToolResult, ErrorData, TextContent, Tool as MCPTool

from agent_env.env.gateway import AGENT_ENV_ROLE_META_KEY, AGENT_ENV_SESSION_META_KEY, InternalMCPServer
from agent_env.env.gateway import gateway as gateway_module
from agent_env.env.gateway.gateway import GATEWAY_TRAJECTORY_FILE, Gateway, _call_never_reached_child, _is_dead_session
from agent_env.env.gateway.get_time import GET_TIME_TOOL_NAME

URL = "http://backing/mcp"
TOOL = MCPTool(name="svc_thing", description="", inputSchema={"type": "object", "properties": {}})


class _RecordingSession:
    """Stands in for the child's ClientSession: records each call's `_meta`, failing as told first."""

    def __init__(self, failures=()):
        self.calls: list[tuple[str, dict, dict | None]] = []
        self.failures = list(failures)

    async def list_tools(self):
        return SimpleNamespace(tools=[TOOL])

    async def call_tool(self, name, arguments, *args, meta=None, **kwargs):
        self.calls.append((name, dict(arguments or {}), meta))
        if self.failures:
            raise self.failures.pop(0)
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)


def _gateway(session: _RecordingSession) -> Gateway:
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[InternalMCPServer("svc", URL)])

    async def _open(stack):
        return {URL: session}

    gw._open_persistent_sessions = _open  # type: ignore[method-assign]
    return gw


def _metas(session: _RecordingSession) -> list[dict | None]:
    return [meta for _, _, meta in session.calls]


def _trajectory(gw: Gateway) -> list[dict]:
    gw._trajectory_file.flush()
    with open(GATEWAY_TRAJECTORY_FILE) as f:
        return [json.loads(line) for line in f]


def test_the_mcp_proxy_stamps_the_session_s_role_and_key_never_the_request_s():
    session = _RecordingSession()
    gw = _gateway(session)
    gw._create_and_register_proxy_tool(URL, TOOL)
    gw._tool_server_urls[TOOL.name] = URL

    def call(role=None, key=None):
        ctx = contextvars.copy_context()  # what the ASGI wrapper and the per-session lifespan would have set
        if role is not None:
            ctx.run(gateway_module._role_var.set, role)
        if key is not None:
            ctx.run(gateway_module._caller_session_var.set, key)
        ctx.run(asyncio.run, gw._mcp._tool_manager.call_tool(TOOL.name, {}))

    call("alice@example.com", "s1")
    call()
    assert _metas(session) == [
        {AGENT_ENV_ROLE_META_KEY: "alice@example.com", AGENT_ENV_SESSION_META_KEY: "s1"},
        {AGENT_ENV_ROLE_META_KEY: "default"},  # stamped for the default role too; no key outside an MCP session
    ]


@pytest.mark.asyncio
async def test_step_stamps_the_header_role_in_meta_including_on_the_reconnect():
    session = _RecordingSession(failures=[anyio.ClosedResourceError()])
    gw = _gateway(session)
    gw._tool_server_urls[TOOL.name] = URL
    before = len(_trajectory(gw))
    try:
        resp = await gw._step_call_tool({"tool_name": TOOL.name, "arguments": {"a": 1}}, "bob@example.com")
    finally:
        await gw._close_child_sessions()
    assert resp.status_code == 200
    assert _metas(session) == [{AGENT_ENV_ROLE_META_KEY: "bob@example.com"}] * 2  # the retry is stamped too
    calls = [e for e in _trajectory(gw)[before:] if e["event_type"] == "tool_call"]
    assert len(calls) == 1 and calls[0]["tool_call"]["function_name"] == TOOL.name  # logged once, not per attempt


@pytest.mark.asyncio
async def test_trigger_internal_calls_are_stamped_only_when_an_as_role_is_given():
    session = _RecordingSession()
    gw = _gateway(session)
    engine = gw._trigger_engine
    try:
        await engine._internal_call(TOOL.name, {}, role="carol@example.com")
        await engine._internal_call(TOOL.name, {})
        gw._clock.set_time("2019-06-15T12:00:00Z", 1)
        gw._register_get_time_tool()
        native = await engine._internal_call(GET_TIME_TOOL_NAME, {}, role="carol@example.com")
    finally:
        await gw._close_child_sessions()
    assert _metas(session) == [{AGENT_ENV_ROLE_META_KEY: "carol@example.com"}, None]
    assert "current_time" in native.content[0].text  # gateway-native: no child, no identity, no session touched


@pytest.mark.parametrize("exc,dead,retry", [
    (anyio.ClosedResourceError(), True, True),
    (anyio.BrokenResourceError(), True, True),
    (McpError(ErrorData(code=32600, message="Session terminated")), True, True),
    (McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed")), True, False),  # in flight: may have written
    (McpError(ErrorData(code=-32602, message="Invalid params")), False, False),
    (asyncio.TimeoutError(), False, False),
    (ValueError("x"), False, False),
], ids=["closed", "broken", "session_terminated", "connection_closed", "invalid_params", "timeout", "other"])
def test_the_dead_session_predicates(exc, dead, retry):
    assert _is_dead_session(exc) is dead
    assert _call_never_reached_child(exc) is retry
