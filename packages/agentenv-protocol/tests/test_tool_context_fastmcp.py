"""ToolContext on the standalone ``fastmcp`` package, the way a plugin mounts it: no ``_tool_manager`` seam, so the
protocol joins the app's middleware chain, outermost. The hook, both role sources, the per-call reset and the
one-handler-per-app refusal hold as on mcp's FastMCP; an app with neither seam mounts unbound unless a reader is
declared."""
from __future__ import annotations

import asyncio
import logging

import anyio
import httpx
import pytest

fastmcp = pytest.importorskip("fastmcp")
from fastmcp import Client, FastMCP  # noqa: E402
from fastmcp.exceptions import ToolError  # noqa: E402
from fastmcp.server.middleware import Middleware  # noqa: E402
from mcp import ClientSession  # noqa: E402
from mcp.client import streamable_http as _streamable_http  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402

from agentenv_protocol import (  # noqa: E402
    ROLE_HEADER,
    ROLE_META_KEY,
    AgentEnvEnvironment,
    AgentEnvFastMCPApplication,
    Caller,
    EnvironmentCard,
    ToolContext,
    environment_card,
    tool,
)
from agentenv_protocol.agent_env_environment import _TOOL_CONTEXT_BOUND_ATTR  # noqa: E402
from agentenv_protocol.tool_context import EMPTY  # noqa: E402


class _Budget(Middleware):
    """A handler's own middleware, as a plugin installs one: records the caller it sees, refuses past a budget."""

    def __init__(self, budget: int = 100) -> None:
        self.seen: list = []
        self.budget = budget

    async def on_call_tool(self, context, call_next):
        self.seen.append(ToolContext.current().caller.role)
        if len(self.seen) > self.budget:
            raise ToolError("budget spent")
        return await call_next(context)


@environment_card(name="fm")
class _Env(AgentEnvEnvironment):
    def __init__(self) -> None:
        self.hook_roles: list = []
        self.seen: list = []

    def on_tool_call(self, context: ToolContext):
        self.hook_roles.append(context.caller.role)
        if context.caller.role == "intruder":
            return CallToolResult(isError=True, content=[TextContent(type="text", text="refused: intruder")])
        return None

    @tool()
    def whoami_sync(self, ctx: ToolContext) -> str:
        self.seen.append(ctx)
        return f"{ctx.caller.role or '-'}|{ToolContext.current().caller.role or '-'}"

    @tool()
    async def whoami_async(self) -> str:
        await asyncio.sleep(0)
        self.seen.append(ToolContext.current())
        return ToolContext.current().caller.role or "-"


@environment_card(name="plain")
class _PlainEnv(AgentEnvEnvironment):
    @tool()
    def echo(self, q: str) -> str:
        return q


@environment_card(name="reader")
class _ReaderEnv(AgentEnvEnvironment):
    @tool()
    def whoami(self, ctx: ToolContext) -> str:
        return ctx.caller.role or "-"


@environment_card(name="hooked")
class _HookedEnv(_PlainEnv):
    def on_tool_call(self, context: ToolContext):
        return None


class _NoSeamApp:
    """An app with the generic surfaces only: no tool manager, no middleware list."""

    def __init__(self) -> None:
        self.routes: list = []
        self.tools: dict = {}

    def custom_route(self, path, methods):
        def register(fn):
            self.routes.append((path, tuple(methods), fn))
            return fn
        return register

    def tool(self, name, description):
        def register(fn):
            self.tools[name] = fn
            return fn
        return register


class _ForeignListApp(_NoSeamApp):
    """Not fastmcp: a ``middleware`` list without ``add_middleware`` is somebody else's list."""

    def __init__(self) -> None:
        super().__init__()
        self.middleware: list = []


def _text(result) -> str:
    return result.content[0].text


def _is_error(result) -> bool:
    return bool(getattr(result, "is_error", getattr(result, "isError", False)))


@pytest.mark.asyncio
async def test_mount_joins_the_middleware_chain_outermost_and_binds_every_call():
    budget = _Budget()
    app = FastMCP("fm", middleware=[budget])
    env = _Env()
    env.mount(app)
    assert getattr(app.middleware[0], _TOOL_CONTEXT_BOUND_ATTR, False) is True and app.middleware[1] is budget
    with anyio.fail_after(30):
        async with Client(app) as client:
            alice = await client.call_tool("whoami_sync", {}, meta={ROLE_META_KEY: "alice"})
            bob = await client.call_tool("whoami_async", {}, meta={ROLE_META_KEY: "bob"})
            nobody = await client.call_tool("whoami_sync", {})
            blank = await client.call_tool("whoami_async", {}, meta={ROLE_META_KEY: "default"})
    assert (_text(alice), _text(bob), _text(nobody), _text(blank)) == ("alice|alice", "bob", "-|-", "-")
    assert budget.seen == ["alice", "bob", None, None], "the handler's own middleware runs under the binding"
    assert env.hook_roles == ["alice", "bob", None, None]
    assert [c.tool for c in env.seen] == ["whoami_sync", "whoami_async", "whoami_sync", "whoami_async"]
    assert env.seen[0].arguments == {} and env.seen[0].call_id
    assert ToolContext.current() is EMPTY


@pytest.mark.asyncio
async def test_the_hook_answers_a_call_before_the_tool_runs():
    app = FastMCP("fm")
    env = _Env()
    env.mount(app)
    with anyio.fail_after(30):
        async with Client(app) as client:
            refused = await client.call_tool("whoami_sync", {}, meta={ROLE_META_KEY: "intruder"}, raise_on_error=False)
            served = await client.call_tool("whoami_sync", {}, meta={ROLE_META_KEY: "carol"})
    assert _is_error(refused) and _text(refused) == "refused: intruder"
    assert [c.caller.role for c in env.seen] == ["carol"], "the refused call never reached the tool"
    assert _text(served) == "carol|carol"
    assert env.hook_roles == ["intruder", "carol"]


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(_streamable_http, "streamable_http_client"), reason="mcp without streamable_http_client")
async def test_a_direct_caller_over_http_gets_the_header_role_and_its_own_session():
    app = FastMCP("fm")
    env = _Env()
    env.mount(app)
    asgi = app.http_app(path="/mcp", json_response=True)
    base = "http://127.0.0.1:8000"
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url=base, headers={ROLE_HEADER: "grader"})
    with anyio.fail_after(30):
        async with asgi.router.lifespan_context(asgi), client:
            async with _streamable_http.streamable_http_client(f"{base}/mcp", http_client=client) as (read, write, session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    direct = await session.call_tool("whoami_sync", {})
                    proxied = await session.call_tool("whoami_sync", {}, meta={ROLE_META_KEY: "bob"})
                    sid = session_id()
    assert sid and direct.content[0].text == "grader|grader" and proxied.content[0].text == "bob|bob"
    assert env.seen[-2].caller == Caller("grader", sid)
    assert env.seen[-1].caller == Caller("bob")  # a _meta role entry: the connection's id is not the caller's


@pytest.mark.asyncio
async def test_concurrent_callers_keep_their_own_context():
    app = FastMCP("fm")
    env = _Env()
    env.mount(app)
    roles = [f"agent-{i}" for i in range(24)]
    with anyio.fail_after(60):
        async with Client(app) as client:
            results = await asyncio.gather(
                *(client.call_tool("whoami_async", {}, meta={ROLE_META_KEY: role}) for role in roles))
    assert [_text(r) for r in results] == roles
    assert sorted(env.hook_roles) == sorted(roles)
    assert ToolContext.current() is EMPTY


def test_one_handler_per_fastmcp_app():
    app = FastMCP("fm")
    _Env().mount(app)
    with pytest.raises(RuntimeError, match="already carries"):
        AgentEnvFastMCPApplication(EnvironmentCard(name="second"), _PlainEnv()).add_routes_to_app(app)
    assert sum(1 for entry in app.middleware if getattr(entry, _TOOL_CONTEXT_BOUND_ATTR, False)) == 1


def test_an_app_with_no_seam_mounts_unbound_unless_a_reader_is_declared(caplog):
    app = _NoSeamApp()
    with caplog.at_level(logging.WARNING, logger="agentenv_protocol.agent_env_environment"):
        _PlainEnv().mount(app)
    assert "echo" in app.tools and app.tools["echo"]("hi") == "hi"
    assert "empty caller" in caplog.text
    for env, reader in ((_ReaderEnv(), "whoami"), (_HookedEnv(), "on_tool_call")):
        with pytest.raises(RuntimeError, match=reader):
            env.mount(_NoSeamApp())
    foreign = _ForeignListApp()
    with pytest.raises(RuntimeError, match="whoami"):
        _ReaderEnv().mount(foreign)
    assert foreign.middleware == [] and foreign.tools == {}, "a foreign list is not a seam, and nothing was added"
