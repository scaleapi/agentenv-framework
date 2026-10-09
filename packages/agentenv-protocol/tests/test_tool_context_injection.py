"""ToolContext on a real FastMCP app (mcp installed, in-memory sessions): schema omission on the
wire, injection, per-call isolation, the on_tool_call chokepoint, and the ordering contract with a
pre-installed dispatch guard."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Annotated, Any

import anyio
import httpx
import pytest
from pydantic import Field

pytest.importorskip("mcp.server.fastmcp")

from mcp import ClientSession  # noqa: E402
from mcp.client import streamable_http as _streamable_http  # noqa: E402
from mcp.server.fastmcp import Context, FastMCP  # noqa: E402
from mcp.server.fastmcp.tools.base import Tool  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402

from agentenv_protocol import (  # noqa: E402
    ROLE_HEADER,
    ROLE_META_KEY,
    SESSION_META_KEY,
    AgentEnvEnvironment,
    AgentEnvFastMCPApplication,
    Caller,
    EnvironmentCard,
    ToolContext,
    create_fastmcp_app,
    environment_card,
    injecting,
    tool,
)
from agentenv_protocol.agent_env_environment import _is_fastmcp_context_annotation, _schema_from_signature  # noqa: E402
from agentenv_protocol.tool_context import EMPTY, bound  # noqa: E402


def _text(result: Any) -> str:
    """The text of FastMCP.call_tool's result: a (content, structured) tuple for structured tools, else content."""
    content = result[0] if isinstance(result, tuple) else result
    return content[0].text


def _request_context(meta_extra: dict | None = None, headers: dict | None = None) -> SimpleNamespace:
    meta = SimpleNamespace(model_extra=meta_extra) if meta_extra is not None else None
    request = SimpleNamespace(headers=headers) if headers is not None else None
    return SimpleNamespace(request_context=SimpleNamespace(meta=meta, request=request))


async def _card_tools(app: FastMCP) -> dict:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.streamable_http_app()), base_url="http://t") as client:
        card = (await client.get("/.well-known/agent-env.json")).json()
    return {t["name"]: t for t in card["capabilities"]["tools"]}


@environment_card(name="items")
class _Env(AgentEnvEnvironment):
    def __init__(self) -> None:
        self.seen: list = []
        self.hook_roles: list = []
        self.alice_in = asyncio.Event()
        self.release = asyncio.Event()

    def on_tool_call(self, context: ToolContext) -> None:
        self.hook_roles.append(context.caller.role)

    @tool(name="rendezvous")
    async def rendezvous(self, who: str, tc: ToolContext) -> str:
        """alice parks inside her call until bob has run his, so the two bodies provably overlap."""
        if who == "alice":
            self.alice_in.set()
            await self.release.wait()
        else:
            await self.alice_in.wait()
            self.release.set()
        return f"{tc.role}|{ToolContext.current().role}"

    @tool(name="whoami")
    async def whoami(self, q: str, tc: ToolContext, limit: int = 3) -> dict[str, Any]:
        self.seen.append((tc, ToolContext.current()))
        return {"q": q, "role": tc.role, "limit": limit}

    @tool(name="whoami_sync")
    def whoami_sync(self, tc: ToolContext) -> str:
        self.seen.append((tc, ToolContext.current()))
        return tc.role or "-"

    @tool(name="slow")
    async def slow(self, delay: float, tc: ToolContext) -> str:
        await asyncio.sleep(delay)
        return f"{tc.role}|{ToolContext.current().role}"

    @tool(name="{environment_name}_add_item")
    async def add_item(
        self,
        item: Annotated[str, Field(description="The item to add.")],
        times: Annotated[int, Field(description="How many copies to add.")] = 1,
    ) -> dict:
        """Add an item to the store."""
        self.seen.append(ToolContext.current())
        return {"item": item, "times": times}


@pytest.mark.asyncio
async def test_card_and_fastmcp_tools_list_both_omit_the_slot():
    env = _Env()
    app = env.create_app()
    expected = {
        "type": "object",
        "properties": {"q": {"type": "string"}, "limit": {"type": "integer", "default": 3}},
        "required": ["q"],
    }
    card_tools = await _card_tools(app)
    assert card_tools["whoami"]["inputSchema"] == expected
    listed = {t.name: t for t in await app.list_tools()}
    assert set(listed["whoami"].inputSchema["properties"]) == {"q", "limit"}
    assert listed["whoami"].inputSchema["required"] == ["q"]
    assert "$defs" not in listed["whoami"].inputSchema
    assert listed["whoami_sync"].inputSchema["properties"] == {}
    assert listed["whoami"].outputSchema is not None  # dict[str, Any] stays structured through the twin

    # card and tools/list derive the schema independently from the same reduced signature
    ours = _schema_from_signature(env.whoami)
    theirs = Tool.from_function(injecting(env.whoami)).parameters
    assert set(ours["properties"]) == set(theirs["properties"]) and ours["required"] == theirs["required"]
    for prop, spec in ours["properties"].items():
        assert spec["type"] == theirs["properties"][prop]["type"]


@pytest.mark.asyncio
async def test_injection_sync_and_async_and_ambient_equals_injected():
    env = _Env()
    app = env.create_app()
    assert app._tool_manager.get_tool("whoami").is_async is True
    assert app._tool_manager.get_tool("whoami_sync").is_async is False

    assert json.loads(_text(await app.call_tool("whoami", {"q": "x"}))) == {"q": "x", "role": None, "limit": 3}
    assert _text(await app.call_tool("whoami_sync", {})) == "-"
    assert len(env.seen) == 2
    for name, (injected, ambient) in zip(("whoami", "whoami_sync"), env.seen, strict=True):
        assert isinstance(injected, ToolContext) and injected is ambient
        assert injected.tool == name and injected.transport == "mcp" and injected.call_id
    assert env.seen[0][0].arguments == {"q": "x"}
    assert ToolContext.current() is EMPTY

    twin = app._tool_manager.get_tool("whoami").fn
    with bound(ToolContext.for_test(role="x")) as ctx:
        assert await twin(q="direct") == {"q": "direct", "role": "x", "limit": 3}
    assert env.seen[-1][0] is ctx


@pytest.mark.asyncio
async def test_concurrent_calls_with_different_meta_roles_never_cross():
    env = _Env()
    app = env.create_app()
    results: dict[str, str] = {}

    async def call(role: str) -> None:
        async with create_connected_server_and_client_session(app) as session:
            result = await session.call_tool("rendezvous", {"who": role}, meta={ROLE_META_KEY: role})
            assert result.isError is False
            results[role] = result.content[0].text

    with anyio.fail_after(10):
        await asyncio.gather(call("alice"), call("bob"))
    assert results == {"alice": "alice|alice", "bob": "bob|bob"}
    assert set(env.hook_roles) == {"alice", "bob"} and len(env.hook_roles) == 2
    assert ToolContext.current() is EMPTY


@pytest.mark.asyncio
async def test_header_fallback_and_default_blank_roles():
    env = _Env()
    app = env.create_app()
    call = app._tool_manager.call_tool
    headers = {ROLE_HEADER: "hdr", "mcp-session-id": "sid"}

    await call("whoami_sync", {}, context=_request_context(headers=headers))
    assert env.seen[-1][0].caller == Caller("hdr", "sid", "hdr")
    await call("whoami_sync", {}, context=_request_context({ROLE_META_KEY: "meta", SESSION_META_KEY: "s-meta"}, headers))
    assert env.seen[-1][0].caller == Caller("meta", "s-meta", "meta")
    await call("whoami_sync", {}, context=_request_context({ROLE_META_KEY: "default"}, headers))
    assert env.seen[-1][0].caller == Caller(None, None, "default")
    await call("whoami_sync", {}, context=_request_context({ROLE_META_KEY: "   "}, None))
    assert env.seen[-1][0].caller.role is None

    async with create_connected_server_and_client_session(app) as session:
        result = await session.call_tool("whoami_sync", {}, meta={ROLE_META_KEY: "default"})
    assert result.content[0].text == "-" and env.seen[-1][0].caller.raw_role == "default"


def _refusal() -> CallToolResult:
    return CallToolResult(isError=True, content=[TextContent(type="text", text="refused: intruder")])


@environment_card(name="guarded")
class _GuardedEnv(AgentEnvEnvironment):
    def __init__(self) -> None:
        self.hook_calls: list = []
        self.ran: list = []

    def on_tool_call(self, context: ToolContext) -> CallToolResult | None:
        self.hook_calls.append((context.tool, dict(context.arguments), context.caller.role))
        return _refusal() if context.caller.role == "intruder" else None

    @tool(name="structured")
    def structured(self, x: int, tc: ToolContext) -> dict[str, Any]:
        self.ran.append("structured")
        return {"x": x, "role": tc.role}

    @tool(name="plain")
    def plain(self, x: int, tc: ToolContext) -> str:
        self.ran.append("plain")
        return f"{x}:{tc.role}"

    @tool(name="slotless")
    def slotless(self, x: int) -> str:
        self.ran.append("slotless")
        return str(x)


class _AsyncGuardedEnv(_GuardedEnv):
    async def on_tool_call(self, context: ToolContext) -> CallToolResult | None:
        await asyncio.sleep(0)
        return _GuardedEnv.on_tool_call(self, context)


@pytest.mark.asyncio
@pytest.mark.parametrize("env_cls", [_GuardedEnv, _AsyncGuardedEnv])
async def test_on_tool_call_refusal_reaches_the_wire_as_is_error_for_structured_and_unstructured_tools(env_cls):
    env = env_cls()
    app = env.create_app()
    assert app._tool_manager.get_tool("structured").output_schema is not None
    assert app._tool_manager.get_tool("plain").output_schema is not None

    async with create_connected_server_and_client_session(app) as session:
        for name in ("structured", "plain", "slotless"):
            result = await session.call_tool(name, {"x": 1}, meta={ROLE_META_KEY: "intruder"})
            assert result.isError is True, name
            assert result.content[0].text == "refused: intruder", name
            assert result.structuredContent is None
        assert env.ran == []
        assert env.hook_calls == [(name, {"x": 1}, "intruder") for name in ("structured", "plain", "slotless")]

        result = await session.call_tool("structured", {"x": 1}, meta={ROLE_META_KEY: "alice"})
        assert result.isError is False and result.structuredContent == {"x": 1, "role": "alice"}
        result = await session.call_tool("plain", {"x": 2}, meta={ROLE_META_KEY: "alice"})
        assert result.isError is False and result.content[0].text == "2:alice"
        assert env.ran == ["structured", "plain"]


@pytest.mark.asyncio
async def test_on_tool_call_exceptions_and_bad_returns_surface_as_errors():
    class _Raising(_GuardedEnv):
        def on_tool_call(self, context: ToolContext) -> None:
            raise RuntimeError("boom")

    class _BadReturn(_GuardedEnv):
        def on_tool_call(self, context: ToolContext) -> Any:
            return {"not": "a CallToolResult"}

    async with create_connected_server_and_client_session(_Raising().create_app()) as session:
        result = await session.call_tool("plain", {"x": 1})
        assert result.isError is True and "boom" in result.content[0].text

    bad = _BadReturn()
    app = bad.create_app()
    with pytest.raises(TypeError, match="on_tool_call must return None or mcp.types.CallToolResult, got dict"):
        await app.call_tool("plain", {"x": 1})
    async with create_connected_server_and_client_session(app) as session:
        result = await session.call_tool("plain", {"x": 1})
        assert result.isError is True and "on_tool_call must return None or mcp.types.CallToolResult" in result.content[0].text
    assert bad.ran == []


@environment_card(name="ctx")
class _ContextEnv(AgentEnvEnvironment):
    @tool(name="t")
    def t(self, x: int, ctx: Context) -> str:
        return type(ctx).__name__

    @tool(name="both")
    def both(self, x: int, ctx: Context, tc: ToolContext) -> str:
        return f"{type(ctx).__name__}:{tc.tool}"


@pytest.mark.asyncio
async def test_fastmcp_context_param_is_off_the_card_but_still_injected():
    app = _ContextEnv().create_app()
    card_tools = await _card_tools(app)
    assert set(card_tools["t"]["inputSchema"]["properties"]) == {"x"}
    assert set(card_tools["both"]["inputSchema"]["properties"]) == {"x"}
    listed = {t.name: t for t in await app.list_tools()}
    assert set(listed["t"].inputSchema["properties"]) == {"x"}
    assert set(listed["both"].inputSchema["properties"]) == {"x"}
    assert app._tool_manager.get_tool("t").context_kwarg == "ctx"
    assert app._tool_manager.get_tool("both").context_kwarg == "ctx"
    assert _text(await app.call_tool("t", {"x": 1})) == "Context"
    assert _text(await app.call_tool("both", {"x": 1})) == "Context:both"


def test_fastmcp_context_string_fallback_when_hints_do_not_resolve():
    for text in ("Context", "Optional[Context]", "mcp.server.fastmcp.Context", "Context | None"):
        assert _is_fastmcp_context_annotation(text), text
    assert not _is_fastmcp_context_annotation("ContextVar")

    def handler(x: "Undefined", ctx: Context, tc: ToolContext) -> str: ...  # noqa: F821

    assert set(_schema_from_signature(handler)["properties"]) == {"x"}


@pytest.mark.asyncio
async def test_tool_without_slot_is_registered_verbatim_with_byte_identical_schema():
    env = _Env()
    app = env.create_app()
    registered = app._tool_manager.get_tool("items_add_item")
    assert registered.fn == env.add_item
    assert registered.parameters == Tool.from_function(env.add_item, name="items_add_item").parameters
    card_tools = await _card_tools(app)
    assert json.dumps(card_tools["items_add_item"]["inputSchema"], sort_keys=True) == json.dumps({
        "type": "object",
        "properties": {
            "item": {"type": "string", "description": "The item to add."},
            "times": {"type": "integer", "description": "How many copies to add.", "default": 1},
        },
        "required": ["item"],
    }, sort_keys=True)

    await app.call_tool("items_add_item", {"item": "x", "times": 2})
    ambient = env.seen[-1]
    assert ambient.tool == "items_add_item" and ambient.arguments == {"item": "x", "times": 2}
    assert ToolContext.current() is EMPTY


@pytest.mark.asyncio
async def test_pre_installed_guard_runs_inside_the_binding():
    app = FastMCP("items")
    original = app._tool_manager.call_tool
    guard_saw: list = []

    async def guard(name: str, arguments: dict, context: Any = None, convert_result: bool = False) -> Any:
        guard_saw.append((name, ToolContext.current().caller.role))
        return await original(name, arguments, context=context, convert_result=convert_result)

    app._tool_manager.call_tool = guard
    env = _Env()
    env.mount(app)
    installed = app._tool_manager.call_tool
    assert installed is not guard and installed.__wrapped__ is guard
    assert installed.__agentenv_tool_context__ is True

    async with create_connected_server_and_client_session(app) as session:
        result = await session.call_tool("slow", {"delay": 0}, meta={ROLE_META_KEY: "alice"})
    assert result.isError is False and result.content[0].text == "alice|alice"
    assert guard_saw == [("slow", "alice")]
    assert env.hook_roles == ["alice"]


def test_raw_slotted_function_cannot_be_registered_on_fastmcp():
    env = _Env()
    with pytest.raises(TypeError, match=r"agentenv_protocol\.injecting\(fn\)"):
        Tool.from_function(env.whoami)
    app = FastMCP("raw")
    with pytest.raises(TypeError, match="ToolContext is injected by the SDK and is not a wire parameter"):
        app.tool(name="whoami")(env.whoami)

    def optional_slot(q: str, tc: ToolContext | None = None) -> str:
        return q

    with pytest.raises(TypeError, match="injecting"):
        Tool.from_function(optional_slot)

    app.tool(name="whoami")(injecting(env.whoami))
    assert set(app._tool_manager.get_tool("whoami").parameters["properties"]) == {"q", "limit"}


@pytest.mark.asyncio
async def test_create_fastmcp_app_binds_tool_calls_for_a_plain_handler_without_a_hook():
    class _Plain:
        def __init__(self) -> None:
            self.seen: list = []

        @tool(name="echo")
        def echo(self, q: str, tc: ToolContext) -> str:
            self.seen.append(tc)
            return q

    handler = _Plain()
    app = create_fastmcp_app(handler, card=EnvironmentCard(name="plain"))
    assert app._tool_manager.call_tool.__agentenv_tool_context__ is True
    assert _text(await app.call_tool("echo", {"q": "hi"})) == "hi"
    assert handler.seen[0].tool == "echo" and handler.seen[0].transport == "mcp"


@pytest.mark.asyncio
@pytest.mark.skipif(not hasattr(_streamable_http, "streamable_http_client"), reason="mcp without streamable_http_client")
async def test_direct_caller_over_streamable_http_gets_the_header_role_and_its_own_session():
    env = _Env()
    app = FastMCP("direct", json_response=True)
    env.mount(app)
    base = "http://127.0.0.1:8000"
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app.streamable_http_app()), base_url=base,
                               headers={ROLE_HEADER: "grader"})

    with anyio.fail_after(30):
        async with app.session_manager.run(), client:
            async with _streamable_http.streamable_http_client(f"{base}/mcp", http_client=client) as (read, write, session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool("whoami_sync", {})
                    forwarded = await session.call_tool("whoami_sync", {}, meta={ROLE_META_KEY: "bob"})
                    sid = session_id()
    assert sid and result.content[0].text == "grader" and forwarded.content[0].text == "bob"
    direct, proxied = env.seen[-2][0], env.seen[-1][0]
    assert direct.caller == Caller("grader", sid, "grader") and direct.mcp is not None
    assert proxied.caller == Caller("bob", None, "bob")  # a _meta role entry: the connection's id is not the caller's


def _app_shape(app: FastMCP) -> tuple:
    return (sorted(t.name for t in app._tool_manager.list_tools()), len(app._custom_starlette_routes), app._tool_manager.call_tool)


def test_a_second_handler_on_the_same_app_is_refused_before_the_app_is_touched():
    app = FastMCP("shared")
    _Env().mount(app)
    before = _app_shape(app)

    @environment_card(name="other")
    class _Other(AgentEnvEnvironment):
        @tool(name="other_tool")
        def other_tool(self, q: str) -> str:
            return q

    with pytest.raises(RuntimeError, match="one handler per app"):
        _Other().mount(app)
    assert _app_shape(app) == before


def test_positional_only_slot_and_mismatched_hook_fail_at_mount():
    class _PositionalOnly(AgentEnvEnvironment):
        @tool(name="po")
        def po(self, tc: ToolContext, /, x: int) -> int:
            return x

    with pytest.raises(TypeError, match="must not be positional-only"):
        _PositionalOnly().create_app()

    class _Arity:
        def on_tool_call(self, role, name, arguments, result):
            return None

        @tool(name="echo")
        def echo(self, q: str) -> str:
            return q

    with pytest.raises(TypeError, match="must take the ToolContext as its only argument"):
        create_fastmcp_app(_Arity(), card=EnvironmentCard(name="arity"))

    app = FastMCP("untouched")
    before = _app_shape(app)
    with pytest.raises(TypeError, match="must take the ToolContext as its only argument"):
        AgentEnvFastMCPApplication(EnvironmentCard(name="arity"), _Arity()).add_routes_to_app(app)
    assert _app_shape(app) == before


@pytest.mark.asyncio
async def test_unknown_tool_is_answered_by_the_dispatch_without_the_hook():
    env = _Env()
    app = env.create_app()
    async with create_connected_server_and_client_session(app) as session:
        result = await session.call_tool("nope", {"x": 1}, meta={ROLE_META_KEY: "alice"})
    assert result.isError is True and "Unknown tool" in result.content[0].text
    assert env.hook_roles == [] and env.seen == []
