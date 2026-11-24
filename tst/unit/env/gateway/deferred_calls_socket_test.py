"""A deferred call end to end over real loopback sockets: a FastMCP chat server declares a scheduled message in its
result, the agent's MCP client never sees the declaration, the gateway posts the message when its clock gets there
and its trigger engine reacts as to the agent's own post, and a call still pending ends with the gateway. The same
holds for a server built with agentenv_protocol whose tool declares the call in its decorator."""
from __future__ import annotations

import asyncio
import json
import socket
import time
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from pytest_socket import enable_socket

from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER, InternalMCPServer
from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.gateway import Gateway
from agentenv_protocol import AgentEnvEnvironment, DeferredCall, Defers, environment_card, tool

DEFERRED = "agentenv.io/deferredCalls"
T0 = "2026-07-06T19:00:00Z"


@pytest.fixture(autouse=True)
def _disable_network():
    """Override the suite-wide socket block (tst/unit/conftest.py) for this module: loopback is the point here."""
    enable_socket()
    yield


def _chat(posted: list[dict], notes: list[str]) -> FastMCP:
    app = FastMCP("chat")
    scheduled: dict[str, dict] = {}

    @app.tool()
    async def chat_schedule(channel_id: str, message: str, post_at: str) -> CallToolResult:
        sid = f"Q{len(scheduled) + 1}"
        scheduled[sid] = {"channel": channel_id, "text": message}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps({"ok": True, "scheduled_message_id": sid}))],
            _meta={DEFERRED: [{"key": sid, "at": post_at,
                               "call": {"tool": "chat_deliver", "arguments": {"scheduled_message_id": sid}},
                               "report_as": {"tool": "chat_send",
                                             "arguments": {"channel_id": channel_id, "message": message}}}]})

    @app.tool()
    async def chat_deliver(scheduled_message_id: str) -> str:
        message = scheduled.pop(scheduled_message_id)
        posted.append(message)
        return json.dumps({"ok": True, "channel": message["channel"], "ts": f"{len(posted)}.000100"})

    @app.tool()
    async def chat_send(channel_id: str, message: str) -> str:
        return json.dumps({"ok": True})

    @app.tool()
    async def chat_note(text: str) -> str:
        notes.append(text)
        return "noted"

    return app


async def _serve(app) -> tuple[uvicorn.Server, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(app, log_level="critical", access_log=False, timeout_graceful_shutdown=3)
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return server, f"http://127.0.0.1:{sock.getsockname()[1]}"


@asynccontextmanager
async def _client(url: str, role: str):
    async with (
        httpx.AsyncClient(headers={AGENT_ENV_ROLE_HEADER: role}, timeout=httpx.Timeout(10, read=60)) as http,
        streamable_http_client(f"{url}/mcp", http_client=http) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


async def _until(predicate, seconds: float = 10.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.02)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_scheduled_message_is_posted_at_its_time_and_answered_as_the_sender_s_post(tmp_path, monkeypatch):
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    posted: list[dict] = []
    notes: list[str] = []
    child, child_url = await _serve(_chat(posted, notes).streamable_http_app())
    gw = Gateway(host="127.0.0.1", port=0, server_name="t",
                 internal_mcp_servers=[InternalMCPServer("chat", f"{child_url}/mcp")])
    gateway, url = await _serve(gw._asgi_app())
    try:
        async with httpx.AsyncClient(base_url=url) as http:
            r = await http.put("/clock/set-time", json={"virtual_time": T0, "virtual_seconds_per_real_second": 0})
            assert r.status_code == 200, r.text
            r = await http.post("/triggers/register", json={"watch_roles": ["alice"], "triggers": [
                {"id": "reply", "when": {"type": "action", "tool": "chat_send", "repeat": True},
                 "actions": [{"type": "tool", "tool": "chat_note", "as": "bob",
                              "args": {"text": "bob read '${args.message}' at ${result.ts}"}}]}]})
            assert r.status_code == 200, r.text

            async with _client(url, "alice") as alice:
                result = await alice.call_tool("chat_schedule", {"channel_id": "C1", "message": "see you at 3",
                                                                 "post_at": "2026-07-06T19:03:00Z"})
                assert not result.isError and DEFERRED not in (result.meta or {})
                await alice.call_tool("chat_schedule", {"channel_id": "C1", "message": "tomorrow",
                                                        "post_at": "2026-07-07T09:00:00Z"})
            await asyncio.sleep(0.2)
            assert posted == [] and notes == []

            r = await http.put("/clock/advance", json={"virtual_time": "2026-07-06T19:03:00Z"})
            assert r.status_code == 200, r.text
            await _until(lambda: notes)
            assert posted == [{"channel": "C1", "text": "see you at 3"}]
            assert notes == ["bob read 'see you at 3' at 1.000100"]
            state = (await http.get("/triggers/state")).json()
            assert [(c["key"], c["status"], c["role"]) for c in state["deferred_calls"]] == [
                ("Q1", "done", "alice"), ("Q2", "pending", "alice")]
    finally:
        await gateway.shutdown()
        await child.shutdown()
    assert not gw._deferred._tasks and posted == [{"channel": "C1", "text": "see you at 3"}]


@environment_card(name="chat")
class _ChatEnv(AgentEnvEnvironment):
    """The chat server above built with agentenv_protocol: its scheduling tool declares the delivery in its decorator."""

    def __init__(self, posted: list[dict], notes: list[str]) -> None:
        self.posted, self.notes, self.scheduled = posted, notes, {}

    @tool(defers=Defers(call="chat_deliver", report_as="chat_send", using="_delivery"))
    def chat_schedule(self, channel_id: str, message: str, post_at: str) -> str:
        sid = f"Q{len(self.scheduled) + 1}"
        self.scheduled[sid] = {"channel": channel_id, "text": message}
        return json.dumps({"ok": True, "scheduled_message_id": sid, "post_at": post_at})

    @tool(hidden_by_default=True)
    def chat_deliver(self, scheduled_message_id: str) -> str:
        message = self.scheduled.pop(scheduled_message_id)
        self.posted.append(message)
        return json.dumps({"ok": True, "channel": message["channel"], "ts": f"{len(self.posted)}.000100"})

    @tool()
    def chat_send(self, channel_id: str, message: str) -> str:
        return json.dumps({"ok": True})

    @tool()
    def chat_note(self, text: str) -> str:
        self.notes.append(text)
        return "noted"

    def _delivery(self, arguments: dict, result) -> DeferredCall:
        return DeferredCall(key=result["scheduled_message_id"], at=result["post_at"],
                            arguments={"scheduled_message_id": result["scheduled_message_id"]},
                            report_arguments={"channel_id": arguments["channel_id"], "message": arguments["message"]})


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_a_call_a_protocol_tool_defers_in_its_decorator_is_made_at_its_time_and_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    posted: list[dict] = []
    notes: list[str] = []
    child, child_url = await _serve(_ChatEnv(posted, notes).create_app().streamable_http_app())
    gw = Gateway(host="127.0.0.1", port=0, server_name="t",
                 internal_mcp_servers=[InternalMCPServer("chat", f"{child_url}/mcp")])
    gateway, url = await _serve(gw._asgi_app())
    try:
        async with httpx.AsyncClient(base_url=url) as http:
            r = await http.put("/clock/set-time", json={"virtual_time": T0, "virtual_seconds_per_real_second": 0})
            assert r.status_code == 200, r.text
            r = await http.post("/triggers/register", json={"watch_roles": ["alice"], "triggers": [
                {"id": "reply", "when": {"type": "action", "tool": "chat_send", "repeat": True},
                 "actions": [{"type": "tool", "tool": "chat_note", "as": "bob",
                              "args": {"text": "bob read '${args.message}' at ${result.ts}"}}]}]})
            assert r.status_code == 200, r.text

            async with _client(url, "alice") as alice:
                listed = {t.name for t in (await alice.list_tools()).tools}
                result = await alice.call_tool("chat_schedule", {"channel_id": "C1", "message": "see you at 3",
                                                                 "post_at": "2026-07-06T14:03:00-05:00"})
            assert "chat_deliver" not in listed and "chat_schedule" in listed
            assert not result.isError and DEFERRED not in (result.meta or {})
            assert json.loads(result.content[0].text)["scheduled_message_id"] == "Q1"
            await asyncio.sleep(0.2)
            assert posted == [] and notes == []

            r = await http.put("/clock/advance", json={"virtual_time": "2026-07-06T19:03:00Z"})
            assert r.status_code == 200, r.text
            await _until(lambda: notes)
            assert posted == [{"channel": "C1", "text": "see you at 3"}]
            assert notes == ["bob read 'see you at 3' at 1.000100"]
            state = (await http.get("/triggers/state")).json()
            assert [(c["key"], c["at"], c["tool"], c["status"], c["role"]) for c in state["deferred_calls"]] == [
                ("Q1", "2026-07-06T19:03:00Z", "chat_deliver", "done", "alice")]
    finally:
        await gateway.shutdown()
        await child.shutdown()
    assert not gw._deferred._tasks
