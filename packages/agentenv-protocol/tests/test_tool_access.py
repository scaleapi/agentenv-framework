"""Per-role tool access with no gateway: tools/list filtered and tools/call refused per caller role on both FastMCPs,
disable-tool/v1 and enable-tool/v1 at the gateway's paths and bodies, and the rules' edge cases."""
from __future__ import annotations

import contextlib

import anyio
import httpx
import pytest

pytest.importorskip("mcp.server.fastmcp")
fastmcp = pytest.importorskip("fastmcp")

from mcp import ClientSession  # noqa: E402
from mcp.client import streamable_http as _streamable_http  # noqa: E402
from mcp.server.fastmcp import FastMCP  # noqa: E402

from agentenv_protocol import (  # noqa: E402
    ROLE_HEADER,
    ROLE_META_KEY,
    AgentEnvEnvironment,
    AgentEnvFastMCPApplication,
    EnvironmentCard,
    environment_card,
    tool,
)
from agentenv_protocol.tool_access import DISABLE_TOOL_EXTENSION_URI, ENABLE_TOOL_EXTENSION_URI, ToolAccess  # noqa: E402

pytestmark = pytest.mark.skipif(not hasattr(_streamable_http, "streamable_http_client"),
                                reason="mcp without streamable_http_client")

_BASE = "http://127.0.0.1:8000"


@environment_card(name="desk")
class _Env(AgentEnvEnvironment):
    tool_access = True

    @tool()
    def read(self) -> str:
        return "read"

    @tool()
    def write(self) -> str:
        return "write"

    @tool(hidden_by_default=True)
    def grade(self) -> str:
        return "grade"


@contextlib.asynccontextmanager
async def _served():
    """The env on mcp's FastMCP over streamable HTTP, and a way to open a session as a role."""
    app = FastMCP("desk", json_response=True)
    _Env().mount(app)
    asgi = app.streamable_http_app()

    @contextlib.asynccontextmanager
    async def session(role: str | None = None):
        headers = {ROLE_HEADER: role} if role else {}
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url=_BASE, headers=headers) as http:
            async with _streamable_http.streamable_http_client(f"{_BASE}/mcp", http_client=http) as (read, write, _):
                async with ClientSession(read, write) as s:
                    await s.initialize()
                    yield s

    async with app.session_manager.run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url=_BASE) as rest:
            yield rest, session


async def _names(session) -> list:
    return sorted(t.name for t in (await session.list_tools()).tools)


def test_the_card_advertises_both_extensions_at_the_gateways_paths():
    card = AgentEnvFastMCPApplication(EnvironmentCard(name="t"), _Env()).environment_card
    extensions = {e.uri: e for e in card.capabilities.extensions}
    assert extensions[DISABLE_TOOL_EXTENSION_URI].params["endpoint"] == "/tools/disable"
    assert extensions[DISABLE_TOOL_EXTENSION_URI].params["methods"]["disable"]["method"] == "POST"
    assert extensions[ENABLE_TOOL_EXTENSION_URI].params["methods"]["enable"]["request"]["required"] == ["role", "tools"]


def test_an_env_without_tool_access_advertises_neither():
    class _Plain(_Env):
        tool_access = False

    card = AgentEnvFastMCPApplication(EnvironmentCard(name="t"), _Plain()).environment_card
    assert not [e for e in (card.capabilities.extensions or []) if "tool/v1" in e.uri]


@pytest.mark.asyncio
async def test_roles_see_and_call_only_what_their_rules_allow():
    with anyio.fail_after(30):
        async with _served() as (rest, session):
            async with session() as anyone:
                assert await _names(anyone) == ["read", "write"]
                hidden = await anyone.call_tool("grade", {})
            changed = await rest.post("/tools/disable", json={"role": "agent", "tools": ["write"]})
            shown = await rest.post("/tools/enable", json={"role": "grader", "tools": ["grade"]})
            async with session("agent") as agent:
                agent_tools = await _names(agent)
                refused = await agent.call_tool("write", {})
                allowed = await agent.call_tool("read", {})
                forwarded = await agent.call_tool("grade", {}, meta={ROLE_META_KEY: "grader"})
            async with session("grader") as grader:
                grader_tools = await _names(grader)
            async with session("default") as default:
                default_tools = await _names(default)
    assert hidden.isError and hidden.structuredContent["code"] == "permission_denied"
    assert changed.json() == {"role": "agent", "disabled": ["write"], "allowed": []}
    assert shown.json() == {"role": "grader", "disabled": [], "allowed": ["grade"]}
    assert agent_tools == ["read"]
    assert refused.isError and refused.content[0].text == "permission_denied: Tool 'write' is disabled for role 'agent'"
    assert refused.structuredContent["applied"] == "no" and not allowed.isError
    assert not forwarded.isError and forwarded.content[0].text == "grade", "a gateway-stamped role wins over the header"
    assert grader_tools == ["grade", "read", "write"] and default_tools == ["read", "write"]


@pytest.mark.asyncio
async def test_everything_off_then_one_tool_on_for_one_role():
    with anyio.fail_after(30):
        async with _served() as (rest, session):
            await rest.post("/tools/disable", json={"role": "*", "tools": "*"})
            await rest.post("/tools/enable", json={"role": "grader", "tools": ["read"]})
            async with session("agent") as agent:
                agent_tools = await _names(agent)
            async with session("grader") as grader:
                grader_tools = await _names(grader)
    assert agent_tools == [] and grader_tools == ["read"]


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {"tools": ["read"]},
    {"role": "", "tools": ["read"]},
    {"role": "agent", "tools": "read"},
    {"role": "agent", "tools": ["read", 3]},
    ["not", "an", "object"],
])
async def test_a_malformed_request_is_refused_and_changes_nothing(body):
    with anyio.fail_after(30):
        async with _served() as (rest, session):
            r = await rest.post("/tools/disable", json=body)
            garbled = await rest.post("/tools/disable", content=b"{not json")
            async with session("agent") as agent:
                tools = await _names(agent)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_params"
    assert garbled.status_code == 400 and tools == ["read", "write"]


@pytest.mark.asyncio
async def test_roles_are_honoured_on_fastmcp_too():
    app = fastmcp.FastMCP("desk")
    _Env().mount(app)
    asgi = app.http_app(path="/mcp", json_response=True)
    with anyio.fail_after(30):
        async with asgi.router.lifespan_context(asgi):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url=_BASE) as rest:
                await rest.post("/tools/disable", json={"role": "agent", "tools": ["write"]})
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url=_BASE,
                                         headers={ROLE_HEADER: "agent"}) as http:
                async with _streamable_http.streamable_http_client(f"{_BASE}/mcp", http_client=http) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await _names(session)
                        refused = await session.call_tool("write", {})
    assert tools == ["read"]
    assert refused.isError and refused.structuredContent["code"] == "permission_denied"


def test_tool_access_needs_a_dispatch_to_read_the_caller_from():
    class _Shapeless:
        def tool(self, name=None, description=None, **kwargs):
            return lambda fn: fn

        def custom_route(self, path, methods):
            return lambda fn: fn

    with pytest.raises(RuntimeError, match="tool_access would read an empty caller"):
        AgentEnvFastMCPApplication(EnvironmentCard(name="t"), _Env()).add_routes_to_app(_Shapeless())


def test_a_rule_for_every_role_reaches_roles_that_already_have_rules():
    access = ToolAccess(hidden_by_default=["grade"])
    access.apply("agent", "read", True)
    access.apply("*", "write", True)
    access.apply("*", "grade", False)
    assert access.rules == {"agent": {"read": True, "write": True, "grade": False}, "*": {"write": True, "grade": False}}
    assert access.is_disabled("agent", "grade") is False and access.is_disabled("other", "write") is True
    access.apply("agent", "*", False)
    assert access.rules["agent"] == {"*": False}, "a rule for every tool of a role replaces that role's named rules"
    assert access.is_disabled("agent", "write") is False
    assert access.is_disabled("agent", "grade") is False, "a hidden tool skips the role's '*' rule; the every-role rule naming it decides"
