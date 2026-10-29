"""Two agents on one gateway process, over real loopback sockets and the app `run()` serves: one child-session
generation shared by their MCP sessions, /step and the time-trigger driver; every child call stamped with its own
session's role and key; and one agent leaving taking nothing away from the other."""
from __future__ import annotations

import asyncio
import socket
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ErrorData, TextContent, Tool as MCPTool
from pytest_socket import enable_socket

from agent_env.env.gateway import (
    AGENT_ENV_ROLE_HEADER,
    AGENT_ENV_ROLE_META_KEY,
    AGENT_ENV_SESSION_META_KEY,
    InternalMCPServer,
)
from agent_env.env.gateway.gateway import Gateway

URL = "http://backing/mcp"
T0 = "2026-01-01T00:00:00Z"


@pytest.fixture(autouse=True)
def _disable_network():
    """Override the suite-wide socket block (tst/unit/conftest.py) for this module: loopback is the point here."""
    enable_socket()
    yield


class _RecordingSession:
    def __init__(self):
        self.calls: list[tuple[str, dict, dict | None]] = []
        self.failures: list[BaseException] = []

    async def list_tools(self):
        return SimpleNamespace(tools=[MCPTool(name="svc_thing", description="",
                                              inputSchema={"type": "object", "properties": {"v": {"type": "string"}}})])

    async def call_tool(self, name, arguments, *args, meta=None, **kwargs):
        self.calls.append((name, dict(arguments or {}), meta))
        if self.failures:
            raise self.failures.pop(0)
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)


async def _serve(app) -> tuple[uvicorn.Server, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    # lifespan on (the default): startup runs the gateway's process lifespan, shutdown unwinds it
    config = uvicorn.Config(app, log_level="critical", access_log=False, timeout_graceful_shutdown=3)
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return server, f"http://127.0.0.1:{sock.getsockname()[1]}"


@asynccontextmanager
async def _stack():
    """The real gateway app on a loopback uvicorn server, its one backing server a recording fake session."""
    child = _RecordingSession()
    opened: list[asyncio.Task] = []
    gw = Gateway(host="127.0.0.1", port=0, server_name="t", internal_mcp_servers=[InternalMCPServer("svc", URL)],
                 website_urls={"web": "http://web"})

    async def _open(stack):
        opened.append(asyncio.current_task())
        return {URL: child}

    gw._open_persistent_sessions = _open  # type: ignore[method-assign]
    gw._trigger_engine._driver_interval = 0.01
    server, url = await _serve(gw._asgi_app())
    try:
        yield gw, url, child, opened
    finally:
        await server.shutdown()


@asynccontextmanager
async def _client(url: str, role: str | None = None):
    headers = {AGENT_ENV_ROLE_HEADER: role} if role else {}
    async with (
        httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(10, read=60)) as http,
        streamable_http_client(f"{url}/mcp", http_client=http) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


@pytest.mark.asyncio
async def test_two_mcp_clients_share_one_generation_and_one_leaving_keeps_the_other_working():
    async with _stack() as (gw, url, child, opened):
        owner0 = gw._child_owner
        assert owner0 is not None and not owner0.done() and len(opened) == 1  # opened once, at startup
        forged = {AGENT_ENV_ROLE_META_KEY: "victim@example.com", AGENT_ENV_SESSION_META_KEY: "forged"}
        async with _client(url, "alice@example.com") as alice:
            async with _client(url, "bob@example.com") as bob:
                assert not (await alice.call_tool("svc_thing", {"v": "a1"}, meta=forged)).isError
                assert not (await bob.call_tool("svc_thing", {"v": "b1"})).isError
            # bob's DELETE ended his server session, which used to close alice's child sessions with it
            assert not (await alice.call_tool("svc_thing", {"v": "a2"})).isError
            async with httpx.AsyncClient() as http:
                step = await http.post(f"{url}/step", headers={AGENT_ENV_ROLE_HEADER: "grader"},
                                       json={"action": "call_tool", "tool_name": "svc_thing", "arguments": {"v": "s1"}})
                assert step.status_code == 200, step.text
                for _ in range(3):
                    async with _client(url):
                        pass
                state = (await http.get(f"{url}/state")).json()
        metas = [meta for _, _, meta in child.calls]
        assert [m[AGENT_ENV_ROLE_META_KEY] for m in metas] == ["alice@example.com", "bob@example.com",
                                                               "alice@example.com", "grader"]
        keys = [m.get(AGENT_ENV_SESSION_META_KEY) for m in metas]
        assert keys[0] == keys[2] != keys[1] and keys[3] is None  # one key per inbound MCP session; /step has none
        assert "forged" not in keys and all(len(k) == 32 for k in keys[:3])
        assert gw._child_owner is owner0 and len(opened) == 1
        gateway_entry = next(s for s in state["mcp_servers"] if s["name"] == "gateway")
        assert [t["name"] for t in gateway_entry["tools"]].count("list_website_urls") == 1  # not once per session
    assert gw._child_owner is None and gw._trigger_engine._driver_task is None  # process shutdown closed both


@pytest.mark.asyncio
async def test_an_armed_time_trigger_keeps_firing_after_a_client_leaves():
    async with _stack() as (gw, url, child, opened):
        async with httpx.AsyncClient() as http:
            r = await http.put(f"{url}/clock/set-time", json={"virtual_time": T0, "virtual_seconds_per_real_second": 3600})
            assert r.status_code == 200, r.text
            r = await http.post(f"{url}/triggers/register", json={"watch_roles": ["default"], "triggers": [
                {"id": "rec", "when": {"type": "time", "every": "PT10M"},
                 "actions": [{"type": "tool", "tool": "svc_thing", "args": {"v": "tick"}, "as": "carol@example.com"}]}]})
            assert r.status_code == 200, r.text
            async with _client(url, "bob@example.com") as bob:
                assert not (await bob.call_tool("svc_thing", {"v": "b1"})).isError
            await asyncio.sleep(0.3)  # 18 virtual minutes: at least one PT10M mark, after bob has left
            state = (await http.get(f"{url}/triggers/state")).json()
        rec = next(t for t in state["triggers"] if t["id"] == "rec")
        assert rec["fire_count"] >= 1 and rec["status"] == "armed", state["events"][-5:]
        driver = gw._trigger_engine._driver_task
        assert driver is not None and not driver.done()
        ticks = [meta for _, args, meta in child.calls if args == {"v": "tick"}]
        assert ticks and all(meta == {AGENT_ENV_ROLE_META_KEY: "carol@example.com"} for meta in ticks)
        assert len(opened) == 1
    assert gw._trigger_engine._driver_task is None and gw._child_owner is None and gw._child_sessions == {}


@pytest.mark.asyncio
async def test_a_dead_child_session_is_reopened_once_under_a_live_mcp_client():
    async with _stack() as (gw, url, child, opened):
        async with _client(url, "alice@example.com") as alice:
            assert not (await alice.call_tool("svc_thing", {"v": "1"})).isError
            owner0 = gw._child_owner
            child.failures.append(McpError(ErrorData(code=32600, message="Session terminated")))  # idle expiry / restart
            assert not (await alice.call_tool("svc_thing", {"v": "2"})).isError  # alice's own session was never dropped
        assert len(opened) == 2 and owner0.done() and gw._child_owner is not owner0 and not gw._child_owner.done()
        assert [args for _, args, _ in child.calls] == [{"v": "1"}, {"v": "2"}, {"v": "2"}]
