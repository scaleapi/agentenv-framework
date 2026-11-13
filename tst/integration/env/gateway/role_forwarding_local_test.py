"""A real gateway in front of a real protocol child, over loopback: the role an MCP client connects with reaches the
child's tools as the caller the protocol binds; a tool that never asked for the caller is served exactly as before; and
a refusal the child's ``on_tool_call`` returns reaches the client through the gateway untouched."""
from __future__ import annotations

import json
import socket
from contextlib import asynccontextmanager

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER, AGENT_ENV_ROLE_META_KEY, AGENT_ENV_SESSION_META_KEY, InternalMCPServer
from agent_env.env.gateway.gateway import Gateway
from agentenv_protocol import (
    ROLE_HEADER,
    ROLE_META_KEY,
    SESSION_META_KEY,
    AgentEnvEnvironment,
    ToolContext,
    environment_card,
    tool,
)

REFUSED = "nobody@example.com"


@environment_card(name="items")
class _Items(AgentEnvEnvironment):
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.seen: list[ToolContext] = []

    def on_tool_call(self, context: ToolContext) -> CallToolResult | None:
        if context.caller.role == REFUSED:
            payload = {"error": "unknown_identity", "role": context.caller.role, "tool": context.tool}
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(payload))], isError=True)
        return None

    @tool(name="{environment_name}_add")
    def add(self, item: str, times: int = 1) -> dict:
        """Add an item."""
        self.items.append({"item": item, "times": times, "by": ToolContext.current().caller.role})
        return {"count": len(self.items)}

    @tool(name="{environment_name}_whoami")
    def whoami(self, ctx: ToolContext) -> dict:
        self.seen.append(ctx)
        return {"role": ctx.caller.role, "session": ctx.caller.session}


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
async def _stack():
    """The child on one loopback server, the gateway on another with the child as its only backing server."""
    env = _Items()
    child_server, child_url = await _serve(env.create_app().streamable_http_app())
    gw = Gateway(host="127.0.0.1", port=0, server_name="t",
                 internal_mcp_servers=[InternalMCPServer("items", f"{child_url}/mcp")])
    gw_server, gw_url = await _serve(gw._asgi_app())
    try:
        yield env, gw_url, child_url
    finally:
        await gw_server.shutdown()
        await child_server.shutdown()


@asynccontextmanager
async def _client(url: str, role: str | None = None):
    headers = {AGENT_ENV_ROLE_HEADER: role} if role else {}
    async with (
        httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(10, read=60)) as http,
        streamable_http_client(f"{url}/mcp", http_client=http) as (read, write, session_id),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session, session_id


def _body(result: CallToolResult) -> dict:
    return json.loads(result.content[0].text)


def test_gateway_and_protocol_agree_on_the_wire_names():
    assert (AGENT_ENV_ROLE_META_KEY, AGENT_ENV_SESSION_META_KEY, AGENT_ENV_ROLE_HEADER) == (ROLE_META_KEY, SESSION_META_KEY, ROLE_HEADER)


@pytest.mark.asyncio
async def test_roles_reach_the_child_through_the_gateway_and_a_slotless_tool_is_served_as_before():
    async with _stack() as (env, gw_url, child_url):
        async with _client(gw_url, "alice@example.com") as (alice, _), _client(gw_url, "bob@example.com") as (bob, _):
            listed = {t.name: t for t in (await alice.list_tools()).tools}
            assert set(listed["items_add"].inputSchema["properties"]) == {"item", "times"}
            assert listed["items_whoami"].inputSchema.get("properties", {}) == {}  # the injected parameter never reaches the wire

            a, b = _body(await alice.call_tool("items_whoami", {})), _body(await bob.call_tool("items_whoami", {}))
            assert (a["role"], b["role"]) == ("alice@example.com", "bob@example.com")
            assert a["session"] != b["session"] and len(a["session"]) == 32  # the gateway's per-session key, not its connection
            assert a == _body(await alice.call_tool("items_whoami", {}))  # stable across calls on one session

            added = await alice.call_tool("items_add", {"item": "pen", "times": 2})
            assert added.isError is False and _body(added) == {"count": 1}
            assert env.items == [{"item": "pen", "times": 2, "by": "alice@example.com"}]

        async with _client(gw_url) as (anon, _):
            assert _body(await anon.call_tool("items_whoami", {}))["role"] is None
            assert _body(await anon.call_tool("items_add", {"item": "cup"})) == {"count": 2}
        assert env.items[-1]["by"] is None

        async with _client(gw_url, REFUSED) as (nobody, _):
            refused = await nobody.call_tool("items_add", {"item": "x"})
            assert refused.isError is True
            assert _body(refused) == {"error": "unknown_identity", "role": REFUSED, "tool": "items_add"}
        assert len(env.items) == 2

        async with httpx.AsyncClient() as http:
            step = await http.post(f"{gw_url}/step", headers={AGENT_ENV_ROLE_HEADER: "grader"},
                                   json={"action": "call_tool", "tool_name": "items_whoami", "arguments": {}})
            assert step.status_code == 200, step.text
        assert (env.seen[-1].caller.role, env.seen[-1].caller.session) == ("grader", None)

        async with _client(child_url, "dave@example.com") as (direct, session_id):
            d = _body(await direct.call_tool("items_whoami", {}))
            assert d["role"] == "dave@example.com" and d["session"] == session_id()  # no gateway: the header and its own connection
