"""ToolContext without mcp: the value and its binding, how a call builds it, slot detection, the
registration twin, and the extension slot (FastMCP-shaped fake target, in-process)."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import typing
from types import SimpleNamespace
from typing import Annotated, Optional, Union

import httpx
import pytest
from pydantic import Field
from starlette.applications import Starlette
from starlette.routing import Route

from agentenv_protocol import (
    DEFAULT_ROLE,
    ROLE_HEADER,
    ROLE_META_KEY,
    SESSION_META_KEY,
    AgentEnvFastMCPApplication,
    Caller,
    EnvironmentCard,
    ToolContext,
    extension,
    injecting,
    normalize_role,
    tool,
)
from agentenv_protocol import testing
from agentenv_protocol.agent_env_environment import _is_tool_context_annotation, _schema_from_signature, _tool_context_slot
from agentenv_protocol.tool_context import EMPTY, bound, caller_from_request, context_from_http, context_from_mcp


class _FakeMCP:
    """FastMCP-shaped target: records routes and tool registrations; its tool manager is the dispatch the mount binds."""

    def __init__(self) -> None:
        self.routes: list[Route] = []
        self.tools: dict[str, tuple] = {}
        self.settings = SimpleNamespace(streamable_http_path="/mcp")
        self._tool_manager = SimpleNamespace(get_tool=self.tools.get, call_tool=self._call_tool)

    async def _call_tool(self, name: str, arguments: dict, *_: object, **__: object):
        result = self.tools[name][1](**arguments)
        return await result if inspect.isawaitable(result) else result

    def custom_route(self, path: str, methods: list[str]):
        def deco(fn):
            self.routes.append(Route(path, fn, methods=methods))
            return fn
        return deco

    def tool(self, name: str | None = None, description: str | None = None):
        def deco(fn):
            self.tools[name or fn.__name__] = (description, fn)
            return fn
        return deco


def _mounted(handler) -> _FakeMCP:
    mcp = _FakeMCP()
    AgentEnvFastMCPApplication(environment_card=EnvironmentCard(name="items"), handler=handler).add_routes_to_app(mcp)
    return mcp


def _client(handler) -> httpx.AsyncClient:
    app = Starlette(routes=_mounted(handler).routes)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


def test_normalize_role_absent_blank_default_and_non_string_are_none():
    for value in (None, "", "  ", "default", " default ", 5):
        assert normalize_role(value) is None
    assert normalize_role(" alice ") == "alice"
    assert ROLE_META_KEY == "agentenv.io/role"
    assert SESSION_META_KEY == "agentenv.io/session"
    assert ROLE_HEADER == "AgentEnv-Role"
    assert DEFAULT_ROLE == "default"


def test_current_is_empty_outside_a_call_and_bound_restores():
    assert ToolContext.current() is EMPTY
    assert EMPTY.transport == "none" and EMPTY.caller.role is None
    with bound(ToolContext.for_test(role="a")) as outer:
        assert ToolContext.current() is outer and outer.role == "a"
        with bound(ToolContext.for_test(role="b")) as inner:
            assert ToolContext.current() is inner
        assert ToolContext.current() is outer
    assert ToolContext.current() is EMPTY
    with pytest.raises(dataclasses.FrozenInstanceError):
        outer.tool = "x"
    with pytest.raises(dataclasses.FrozenInstanceError):
        outer.caller.role = "x"


def test_caller_from_request_precedence():
    headers = {ROLE_HEADER: "hdr", "mcp-session-id": "sid"}
    assert caller_from_request({ROLE_META_KEY: "meta"}, headers).role == "meta"
    assert caller_from_request({}, headers) == Caller("hdr", "sid", "hdr")
    assert caller_from_request(None, headers) == Caller("hdr", "sid", "hdr")
    assert caller_from_request({SESSION_META_KEY: "s-meta"}, headers).session == "s-meta"
    assert caller_from_request({SESSION_META_KEY: ""}, headers).session == "sid"
    assert caller_from_request({SESSION_META_KEY: 7}, headers).session == "sid"
    assert caller_from_request({ROLE_META_KEY: "default"}, None) == Caller(None, None, "default")
    assert caller_from_request(None, None) == Caller()


def test_context_from_mcp_tolerates_every_missing_link():
    empty = context_from_mcp(None, "t", {})
    assert empty.transport == "mcp" and empty.caller == Caller() and empty.mcp is None

    class _OutsideRequest:
        @property
        def request_context(self):
            raise ValueError("Context is not available outside of a request")

    assert context_from_mcp(_OutsideRequest(), "t", {}).caller == Caller()
    bare = SimpleNamespace(request_context=SimpleNamespace(meta=None, request=None))
    assert context_from_mcp(bare, "t", {}).caller == Caller()

    arguments = {"q": 1}
    live = SimpleNamespace(request_context=SimpleNamespace(
        meta=SimpleNamespace(model_extra={ROLE_META_KEY: "bob"}),
        request=SimpleNamespace(headers={"mcp-session-id": "s1"}),
    ))
    ctx = context_from_mcp(live, "search", arguments)
    assert ctx.caller == Caller("bob", None, "bob") and ctx.tool == "search" and ctx.mcp is live
    assert ctx.call_id and ctx.arguments == {"q": 1}
    arguments["q"] = 2
    assert ctx.arguments == {"q": 1}


def test_context_from_http_and_testing_helper():
    request = SimpleNamespace(headers={ROLE_HEADER: "carol"})
    ctx = context_from_http(request, "echo", {"m": 1})
    assert ctx.transport == "rest" and ctx.role == "carol" and ctx.tool == "echo" and ctx.arguments == {"m": 1}

    with testing.tool_context(role="alice", session="s") as bound_ctx:
        assert ToolContext.current() is bound_ctx
        assert bound_ctx.caller == Caller("alice", "s", "alice")
    assert ToolContext.current() is EMPTY
    with testing.tool_context(role="default") as default_ctx:
        assert default_ctx.role is None and default_ctx.caller.raw_role == "default"


def test_tool_context_slot_detection_by_annotation():
    def plain(tc: ToolContext, q: str) -> str: ...
    def optional(q: str, maybe: Optional[ToolContext] = None) -> str: ...
    def pipe(q: str, u: ToolContext | None = None) -> str: ...
    def union(q: str, un: Union[ToolContext, None] = None) -> str: ...
    def annotated(a: Annotated[ToolContext, "x"], q: str) -> str: ...
    def prefixed(t: typing.Optional[ToolContext], q: str) -> str: ...
    def unresolvable_sibling(x: "Undefined", tc: ToolContext) -> str: ...  # noqa: F821
    def none(q: str, n: int = 0) -> str: ...
    def two(a: ToolContext, b: ToolContext) -> str: ...

    assert _tool_context_slot(plain) == "tc"
    assert _tool_context_slot(optional) == "maybe"
    assert _tool_context_slot(pipe) == "u"
    assert _tool_context_slot(union) == "un"
    assert _tool_context_slot(annotated) == "a"
    assert _tool_context_slot(prefixed) == "t"
    with pytest.raises(NameError):
        typing.get_type_hints(unresolvable_sibling)  # the string fallback is what finds the slot below
    assert _tool_context_slot(unresolvable_sibling) == "tc"
    assert _tool_context_slot(none) is None
    with pytest.raises(TypeError, match="more than one ToolContext parameter"):
        _tool_context_slot(two)


@pytest.mark.parametrize("text", [
    "ToolContext",
    "pkg.ToolContext",
    "Optional[ToolContext]",
    "typing.Optional[ToolContext]",
    "Union[ToolContext, None]",
    "typing.Union[None, ToolContext]",
    "ToolContext | None",
    "None | agentenv_protocol.ToolContext",
    "Annotated[ToolContext, Field(description='a, b')]",
    "Annotated[Optional[ToolContext], 'x']",
    "'ToolContext'",
])
def test_string_annotation_forms_that_name_the_slot(text):
    assert _is_tool_context_annotation(text)


@pytest.mark.parametrize("text", ["ToolContextual", "str", "Optional[str]", "list[ToolContext]", "dict[str, ToolContext]", "None"])
def test_string_annotation_forms_that_do_not_name_the_slot(text):
    assert not _is_tool_context_annotation(text)


class _SlotHandler:
    def __init__(self) -> None:
        self.seen: list = []

    @tool(name="whoami", description="Who is calling.")
    async def whoami(self, q: str, tc: ToolContext, limit: int = 3) -> dict:
        self.seen.append(tc)
        return {"q": q, "role": tc.role, "limit": limit}

    @tool(name="whoami_sync")
    def whoami_sync(self, tc: Optional[ToolContext] = None) -> str:
        self.seen.append(tc)
        return tc.role or ""

    @tool(name="plain")
    async def plain(self, item: Annotated[str, Field(description="The item.")], times: int = 1) -> dict:
        """Add an item."""
        return {"item": item, "times": times}


@pytest.mark.asyncio
async def test_card_schema_omits_the_slot_and_registration_gets_the_twin():
    handler = _SlotHandler()
    async with _client(handler) as client:
        tools = {t["name"]: t for t in (await client.get("/.well-known/agent-env.json")).json()["capabilities"]["tools"]}
    assert tools["whoami"]["inputSchema"] == {
        "type": "object",
        "properties": {"q": {"type": "string"}, "limit": {"type": "integer", "default": 3}},
        "required": ["q"],
    }
    assert tools["whoami"]["description"] == "Who is calling."
    assert tools["whoami_sync"]["inputSchema"] == {"type": "object", "properties": {}}

    mcp = _mounted(handler)
    twin = mcp.tools["whoami"][1]
    assert "tc" not in inspect.signature(twin).parameters
    assert list(inspect.signature(twin).parameters) == ["q", "limit"]
    assert "tc" not in twin.__annotations__ and twin.__annotations__["return"] is dict
    assert twin.__wrapped__ == handler.whoami and twin.__name__ == "whoami"
    assert inspect.iscoroutinefunction(twin) and not inspect.iscoroutinefunction(mcp.tools["whoami_sync"][1])

    with bound(ToolContext.for_test(role="alice")) as ctx:
        assert await twin(q="x") == {"q": "x", "role": "alice", "limit": 3}
        assert mcp.tools["whoami_sync"][1]() == "alice"
    assert handler.seen == [ctx, ctx]
    assert await twin(q="y") == {"q": "y", "role": None, "limit": 3}  # EMPTY outside any binding


def test_slot_less_tool_is_registered_verbatim():
    handler = _SlotHandler()
    plain = handler.plain
    assert _mounted(handler).tools["plain"][1] == plain
    assert injecting(plain) is plain
    assert _schema_from_signature(plain) == {
        "type": "object",
        "properties": {"item": {"type": "string", "description": "The item."}, "times": {"type": "integer", "default": 1}},
        "required": ["item"],
    }


@pytest.mark.asyncio
async def test_twin_takes_positional_arguments_in_its_advertised_order():
    class _Handler:
        async def first(self, tc: ToolContext, q: str, limit: int = 3) -> dict:
            return {"q": q, "limit": limit, "role": tc.role}

        def middle(self, q: str, tc: ToolContext, limit: int = 3) -> dict:
            return {"q": q, "limit": limit, "role": tc.role}

    handler = _Handler()
    first, middle = injecting(handler.first), injecting(handler.middle)
    assert list(inspect.signature(first).parameters) == ["q", "limit"]
    with bound(ToolContext.for_test(role="alice")):
        assert await first("x") == {"q": "x", "limit": 3, "role": "alice"}
        assert await first("x", 5) == await first(q="x", limit=5) == {"q": "x", "limit": 5, "role": "alice"}
        assert middle("y", 2) == middle(limit=2, q="y") == {"q": "y", "limit": 2, "role": "alice"}
    with pytest.raises(TypeError, match="missing a required argument: 'q'"):
        await first()
    with pytest.raises(TypeError, match="too many positional arguments"):
        await first("x", 5, 7)


def test_injecting_raises_when_a_slotted_function_has_unresolvable_annotations():
    def broken(x: "Undefined", tc: ToolContext) -> str: ...  # noqa: F821

    with pytest.raises(TypeError, match=r"broken: cannot resolve the annotations of \['x', 'return'\]"):
        injecting(broken)


@pytest.mark.parametrize("strip", [lambda mcp: delattr(mcp, "_tool_manager"), lambda mcp: delattr(mcp._tool_manager, "call_tool")])
def test_target_without_a_tool_dispatch_is_refused_before_any_route_is_added(strip):
    mcp = _FakeMCP()
    strip(mcp)
    with pytest.raises(RuntimeError, match="no FastMCP tool dispatch"):
        AgentEnvFastMCPApplication(environment_card=EnvironmentCard(name="items"), handler=_SlotHandler()).add_routes_to_app(mcp)
    assert mcp.routes == [] and mcp.tools == {}


@pytest.mark.asyncio
async def test_mount_binds_the_target_dispatch():
    handler = _SlotHandler()
    mcp = _mounted(handler)
    assert mcp._tool_manager.call_tool.__agentenv_tool_context__ is True
    assert await mcp._tool_manager.call_tool("whoami", {"q": "x"}, context=None) == {"q": "x", "role": None, "limit": 3}
    assert handler.seen[-1].transport == "mcp" and handler.seen[-1].tool == "whoami"
    assert ToolContext.current() is EMPTY


class _ExtHandler:
    @extension(uri="urn:agentenv:whoami/v1")
    async def whoami(self, tc: ToolContext, greeting: str = "hi") -> dict:
        assert ToolContext.current() is tc
        return {"role": tc.role, "transport": tc.transport, "tool": tc.tool, "greeting": greeting}

    @extension(uri="urn:agentenv:whoami-get/v1", method="GET")
    def whoami_get(self, greeting: str, tc: Optional[ToolContext] = None) -> dict:
        return {"role": tc.role, "transport": tc.transport, "greeting": greeting}


@pytest.mark.asyncio
async def test_extension_slot_binds_rest_caller_from_headers():
    async with _client(_ExtHandler()) as client:
        card = (await client.get("/.well-known/agent-env.json")).json()
        exts = {e["uri"]: e for e in card["capabilities"]["extensions"]}
        assert exts["urn:agentenv:whoami/v1"]["params"]["methods"]["whoami"]["request"] == {
            "type": "object", "properties": {"greeting": {"type": "string", "default": "hi"}},
        }
        assert set(exts["urn:agentenv:whoami-get/v1"]["params"]["methods"]["whoami_get"]["request"]["properties"]) == {"greeting"}

        r = await client.post("/agentenv/ext/whoami", json={}, headers={ROLE_HEADER: "carol"})
        assert r.status_code == 200
        assert r.json() == {"role": "carol", "transport": "rest", "tool": "whoami", "greeting": "hi"}
        r = await client.post("/agentenv/ext/whoami", json={"greeting": "yo"})
        assert r.status_code == 200 and r.json()["role"] is None and r.json()["greeting"] == "yo"
        r = await client.post("/agentenv/ext/whoami", json={"greeting": "yo"}, headers={ROLE_HEADER: "default"})
        assert r.json()["role"] is None

        r = await client.post("/agentenv/ext/whoami", json={"tc": 1})
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_params"

        r = await client.get("/agentenv/ext/whoami_get", params={"greeting": "hey"}, headers={ROLE_HEADER: "dave"})
        assert r.status_code == 200 and r.json() == {"role": "dave", "transport": "rest", "greeting": "hey"}
    assert ToolContext.current() is EMPTY


def test_caller_from_request_consults_headers_only_when_no_meta_role_entry_exists():
    headers = {ROLE_HEADER: "hdr", "mcp-session-id": "conn"}
    assert caller_from_request(None, headers) == Caller("hdr", "conn", "hdr")
    assert caller_from_request({}, headers) == Caller("hdr", "conn", "hdr")
    assert caller_from_request({SESSION_META_KEY: "s-only"}, headers) == Caller("hdr", "s-only", "hdr")
    # a _meta role entry marks a proxied call: blank and default stay unforwarded, the connection id is not the caller's
    assert caller_from_request({ROLE_META_KEY: ""}, headers) == Caller(None, None, "")
    assert caller_from_request({ROLE_META_KEY: "default"}, headers) == Caller(None, None, "default")
    assert caller_from_request({ROLE_META_KEY: "bob"}, headers) == Caller("bob", None, "bob")
    assert caller_from_request({ROLE_META_KEY: "bob", SESSION_META_KEY: "s1"}, headers) == Caller("bob", "s1", "bob")
    assert caller_from_request({ROLE_META_KEY: 7, SESSION_META_KEY: ""}, headers) == Caller(None, None, None)


def test_tool_context_is_hashable_and_its_arguments_are_read_only():
    ctx = ToolContext.for_test(role="a", arguments={"q": 1})
    assert hash(ctx) == hash(ctx) and {ctx: 1}[ctx] == 1 and ctx in {ctx}
    assert hash(EMPTY) is not None and EMPTY != ctx
    assert ctx.arguments == {"q": 1} and dict(ctx.arguments) == {"q": 1}
    with pytest.raises(TypeError):
        ctx.arguments["q"] = 2  # type: ignore[index]
    with pytest.raises(TypeError):
        EMPTY.arguments["k"] = 1  # type: ignore[index]
    assert ToolContext.current().arguments == {}


def test_tool_context_embeds_in_a_pydantic_model_but_never_renders_as_a_wire_schema():
    from pydantic import BaseModel, TypeAdapter, ValidationError

    class Audit(BaseModel):
        context: ToolContext
        note: str = ""

    ctx = ToolContext.for_test(role="a")
    assert Audit(context=ctx).context is ctx
    with pytest.raises(ValidationError):
        Audit(context={"caller": {}})
    assert TypeAdapter(Optional[ToolContext]).validate_python(None) is None
    assert TypeAdapter(Optional[ToolContext]).validate_python(ctx) is ctx
    with pytest.raises(TypeError, match=r"agentenv_protocol\.injecting\(fn\)"):
        Audit.model_json_schema()


@pytest.mark.asyncio
async def test_binding_follows_the_task_into_to_thread_and_child_tasks():
    ctx = ToolContext.for_test(role="a")
    with bound(ctx):
        assert await asyncio.to_thread(ToolContext.current) is ctx
        assert await asyncio.create_task(_current_later()) is ctx
    assert await asyncio.to_thread(ToolContext.current) is EMPTY


async def _current_later() -> ToolContext:
    await asyncio.sleep(0)
    return ToolContext.current()
