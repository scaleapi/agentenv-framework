"""Unit tests for AgentEnvApplication route wiring (in-process, no Docker)."""

from __future__ import annotations

import datetime
import decimal
import enum
import inspect
import json
import sys
import uuid
from types import SimpleNamespace
from typing import Annotated, Literal, Optional

import httpx
import pytest
from pydantic import Field
from starlette.applications import Starlette
from starlette.routing import Route

from agentenv_protocol import (
    AgentEnvEnvironment,
    AgentEnvFastMCPApplication,
    AgentEnvStarletteApplication,
    DataPart,
    EnvironmentCapabilities,
    EnvironmentCard,
    EnvironmentExtension,
    EnvironmentInterface,
    EnvironmentTool,
    add_data,
    create_fastmcp_app,
    environment_card,
    extension,
    get_data,
    reset_data,
    tool,
)
from agentenv_protocol import agent_env_environment
from agentenv_protocol.agent_env_environment import _schema_from_signature
from agentenv_protocol.client import mcp_path


class _FakeMCP:
    """Captures custom_route registrations into a Starlette route list and tool registrations into a dict (FastMCP-shaped target)."""

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


class _ItemsHandler:
    def __init__(self) -> None:
        self.store: list = []

    @reset_data
    async def _reset(self) -> None:
        self.store.clear()

    @add_data
    async def _add(self, parts: list) -> None:
        for part in parts:
            if part.kind == "data":
                self.store.extend(part.data["items"])

    @get_data
    async def _state(self) -> list:
        return [DataPart(data={"items": self.store})]


class _BadHandler:
    @reset_data
    async def _reset(self) -> None:
        raise RuntimeError("boom")

    @add_data
    async def _add(self, parts: list) -> None:
        pass

    @get_data
    async def _state(self) -> list:
        return []


def _rpc(method: str, params: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}


def _client(handler_cls, card: EnvironmentCard | None = None) -> httpx.AsyncClient:
    mcp = _FakeMCP()
    AgentEnvFastMCPApplication(
        environment_card=card or EnvironmentCard(name="items"),
        handler=handler_cls(),
    ).add_routes_to_app(mcp)
    app = Starlette(routes=mcp.routes)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


@pytest.mark.asyncio
async def test_full_data_plane_roundtrip():
    async with _client(_ItemsHandler) as client:
        r = await client.post("/agentenv", json=_rpc("data/get"))
        assert r.json()["result"]["parts"][0]["data"] == {"items": []}

        r = await client.post("/agentenv", json=_rpc("data/add", {"parts": [{"kind": "data", "data": {"items": ["a", "b"]}}]}))
        assert r.status_code == 200 and r.json()["result"] == {}

        r = await client.post("/agentenv", json=_rpc("data/get"))
        assert r.json()["result"]["parts"][0]["data"] == {"items": ["a", "b"]}

        r = await client.post("/agentenv", json=_rpc("data/reset"))
        assert r.json()["result"] == {}

        r = await client.post("/agentenv", json=_rpc("data/get"))
        assert r.json()["result"]["parts"][0]["data"] == {"items": []}


@pytest.mark.asyncio
async def test_add_invalid_params_returns_error():
    async with _client(_ItemsHandler) as client:
        r = await client.post("/agentenv", json=_rpc("data/add", {"parts": []}))
        assert r.json()["error"]["code"] == -32602

        r = await client.post("/agentenv", json=_rpc("data/add", {"parts": [{"kind": "bogus"}]}))
        assert r.json()["error"]["code"] == -32602


@pytest.mark.asyncio
async def test_unknown_method_returns_error():
    async with _client(_ItemsHandler) as client:
        r = await client.post("/agentenv", json=_rpc("data/bogus"))
        assert r.json()["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_operation_failure_returns_error():
    async with _client(_BadHandler) as client:
        r = await client.post("/agentenv", json=_rpc("data/reset"))
        body = r.json()
        assert body["error"]["code"] == -32000
        assert body["error"]["data"]["code"] == "reset_failed"
        assert "boom" in body["error"]["message"]


@pytest.mark.asyncio
async def test_well_known_card_advertises_transport():
    card = EnvironmentCard(name="Items Store")
    async with _client(_ItemsHandler, card=card) as client:
        r = await client.get("/.well-known/agent-env.json")
        assert r.status_code == 200
        body = r.json()
        assert body["name"] == "Items Store"
        assert body["url"] == "/agentenv"
        assert body["preferredTransport"] == "JSONRPC"


@pytest.mark.asyncio
async def test_well_known_card_serves_null_extensions_and_tools_by_default():
    async with _client(_ItemsHandler, card=EnvironmentCard(name="items")) as client:
        body = (await client.get("/.well-known/agent-env.json")).json()
        assert body["capabilities"] == {"extensions": None, "tools": None, "operations": ["data/reset", "data/add", "data/get"]}


@pytest.mark.asyncio
async def test_well_known_card_serves_advertised_extensions():
    card = EnvironmentCard(
        name="items",
        capabilities=EnvironmentCapabilities(
            extensions=[EnvironmentExtension(uri="urn:agentenv:interfaces/v1", params={"interfaces": ["cli"]})]
        ),
    )
    async with _client(_ItemsHandler, card=card) as client:
        body = (await client.get("/.well-known/agent-env.json")).json()
        assert body["capabilities"]["extensions"] == [{
            "uri": "urn:agentenv:interfaces/v1",
            "description": None,
            "params": {"interfaces": ["cli"]},
            "required": None,
        }]


@pytest.mark.asyncio
async def test_well_known_card_serves_null_children_by_default():
    async with _client(_ItemsHandler, card=EnvironmentCard(name="items")) as client:
        body = (await client.get("/.well-known/agent-env.json")).json()
        assert body["children_environments"] is None


def test_environment_card_nests_children_recursively():
    card = EnvironmentCard(
        name="gw",
        children_environments=[EnvironmentCard(name="slack"), EnvironmentCard(name="linear")],
    )
    dumped = card.model_dump()
    assert [c["name"] for c in dumped["children_environments"]] == ["slack", "linear"]
    assert EnvironmentCard(name="leaf").model_dump()["children_environments"] is None


@pytest.mark.asyncio
async def test_mounts_on_plain_starlette_app():
    app = Starlette()
    AgentEnvStarletteApplication(
        environment_card=EnvironmentCard(name="items"),
        handler=_ItemsHandler(),
    ).add_routes_to_app(app)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        r = await client.post("/agentenv", json=_rpc("data/get"))
        assert r.json()["result"]["parts"][0]["data"] == {"items": []}
        assert (await client.get("/.well-known/agent-env.json")).json()["url"] == "/agentenv"


# (interfaces the card declares, path the app serves MCP on, interfaces the served card lists)
_MCP_INTERFACE_CASES = [
    pytest.param([], "/mcp", [{"url": "/mcp", "transport": "mcp"}], id="default"),
    pytest.param([], "/custom-mcp", [{"url": "/custom-mcp", "transport": "mcp"}], id="configured-path"),
    pytest.param(
        [{"url": "/custom-mcp", "transport": "mcp"}], "/custom-mcp",
        [{"url": "/custom-mcp", "transport": "mcp"}], id="declared-mcp-wins",
    ),
    pytest.param(
        [{"url": "/grpc", "transport": "grpc"}], "/mcp",
        [{"url": "/grpc", "transport": "grpc"}, {"url": "/mcp", "transport": "mcp"}], id="other-transport-kept",
    ),
    pytest.param(
        [{"url": "/custom-mcp", "transport": "streamable-http"}], "/custom-mcp",
        [{"url": "/custom-mcp", "transport": "streamable-http"}, {"url": "/custom-mcp", "transport": "mcp"}],
        id="same-path-other-transport-kept",
    ),
]


@pytest.mark.parametrize("declared,served_path,expected", _MCP_INTERFACE_CASES)
@pytest.mark.asyncio
async def test_card_declares_the_mcp_endpoint_the_app_serves(declared, served_path, expected):
    pytest.importorskip("mcp.server.fastmcp")
    card = EnvironmentCard(name="items", additionalInterfaces=declared)
    http_app = create_fastmcp_app(_ToolHandler(), card=card, streamable_http_path=served_path).streamable_http_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=http_app), base_url="http://t") as client:
        served = (await client.get("/.well-known/agent-env.json")).json()
    assert served["additionalInterfaces"] == expected
    assert mcp_path(served) == served_path
    assert served_path in {getattr(route, "path", None) for route in http_app.routes}


@pytest.mark.asyncio
async def test_mounted_env_card_declares_its_mcp_endpoint():
    body, _ = await _mount_and_fetch_card(_ServeEnv())
    assert body["additionalInterfaces"] == [{"url": "/mcp", "transport": "mcp"}]


def test_starlette_card_declares_no_mcp_endpoint():
    app = AgentEnvStarletteApplication(environment_card=EnvironmentCard(name="items"), handler=_ItemsHandler())
    assert app.environment_card.additionalInterfaces == []


@pytest.mark.asyncio
async def test_target_without_settings_mounts_and_declares_no_mcp_endpoint():
    target = _FakeMCP()
    del target.settings
    AgentEnvFastMCPApplication(environment_card=EnvironmentCard(name="items"), handler=_ItemsHandler()).add_routes_to_app(target)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=Starlette(routes=target.routes)), base_url="http://t") as client:
        card = (await client.get("/.well-known/agent-env.json")).json()
    assert card["additionalInterfaces"] == []
    assert mcp_path(card) == "/mcp"


@pytest.mark.asyncio
async def test_partial_data_ops_construct_and_unregistered_op_answers_method_not_found():
    class _ResetOnly:
        @reset_data
        async def _reset(self) -> None:
            pass

    mcp = _FakeMCP()
    AgentEnvFastMCPApplication(environment_card=EnvironmentCard(name="x"), handler=_ResetOnly()).add_routes_to_app(mcp)
    app = Starlette(routes=mcp.routes)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        assert (await client.post("/agentenv", json=_rpc("data/reset"))).json()["result"] == {}
        assert (await client.post("/agentenv", json=_rpc("data/add", {"parts": [{"kind": "data", "data": {}}]}))).json()["error"]["code"] == -32601
        card = (await client.get("/.well-known/agent-env.json")).json()
        assert card["capabilities"]["operations"] == ["data/reset"]


@pytest.mark.asyncio
async def test_tools_only_env_serves_card_and_tools_without_data_plane():
    class _ToolsOnly:
        @tool(name="ping")
        async def ping(self) -> dict:
            return {"pong": True}

    mcp = _FakeMCP()
    AgentEnvFastMCPApplication(environment_card=EnvironmentCard(name="x"), handler=_ToolsOnly()).add_routes_to_app(mcp)
    assert "ping" in mcp.tools
    app = Starlette(routes=mcp.routes)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        card = (await client.get("/.well-known/agent-env.json")).json()
        assert card["capabilities"]["operations"] == []
        assert (await client.post("/agentenv", json=_rpc("data/get"))).json()["error"]["code"] == -32601
        assert (await client.post("/agentenv", json=_rpc("data/add", {"parts": []}))).json()["error"]["code"] == -32601


class _ExtHandler(_ItemsHandler):
    @extension(uri="urn:agentenv:echo/v1", description="Echo a message back")
    async def echo(self, message: str, shout: bool = False) -> dict:
        return {"echoed": message.upper() if shout else message}

    @extension(uri="urn:agentenv:ping/v1", method="GET")
    async def ping(self) -> dict:
        return {"pong": True}

    @extension(uri="urn:agentenv:repeat/v1", method="GET")
    async def repeat(self, text: str, times: int = 1, upper: bool = False) -> dict:
        return {"out": (text.upper() if upper else text) * times, "times_type": type(times).__name__}

    @extension(uri="urn:agentenv:pick/v1", method="GET")
    async def pick(self, choice: Literal[1, 2, 3]) -> dict:
        # `choice` arrives as a raw query string; the request path must coerce it to the
        # advertised int literal and reject values outside {1, 2, 3}.
        return {"choice": choice, "choice_type": type(choice).__name__}

    @extension(uri="urn:agentenv:boom/v1")
    async def boom(self) -> dict:
        raise RuntimeError("kaboom")


@pytest.mark.asyncio
async def test_decorated_extension_advertises_a2a_shaped_params():
    async with _client(_ExtHandler, card=EnvironmentCard(name="items")) as client:
        body = (await client.get("/.well-known/agent-env.json")).json()
        exts = {e["uri"]: e for e in body["capabilities"]["extensions"]}
        assert set(exts) == {
            "urn:agentenv:echo/v1", "urn:agentenv:ping/v1", "urn:agentenv:repeat/v1",
            "urn:agentenv:pick/v1", "urn:agentenv:boom/v1",
        }
        assert exts["urn:agentenv:echo/v1"]["description"] == "Echo a message back"
        # A2A-shaped: endpoint = <RPC_PATH>/ext/<method-name> (verbatim); request schema from signature
        assert exts["urn:agentenv:echo/v1"]["params"] == {
            "endpoint": "/agentenv/ext/echo",
            "methods": {
                "echo": {
                    "method": "POST",
                    "request": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}, "shout": {"type": "boolean", "default": False}},
                        "required": ["message"],
                    },
                }
            },
        }
        assert exts["urn:agentenv:ping/v1"]["params"] == {
            "endpoint": "/agentenv/ext/ping",
            "methods": {"ping": {"method": "GET", "request": {"type": "object", "properties": {}}}},
        }


@pytest.mark.asyncio
async def test_decorated_extension_invocable_via_rest_route():
    async with _client(_ExtHandler) as client:
        r = await client.post("/agentenv/ext/echo", json={"message": "hi"})
        assert r.status_code == 200 and r.json() == {"echoed": "hi"}
        r = await client.post("/agentenv/ext/echo", json={"message": "hi", "shout": True})
        assert r.json() == {"echoed": "HI"}


@pytest.mark.asyncio
async def test_decorated_extension_get_route():
    async with _client(_ExtHandler) as client:
        r = await client.get("/agentenv/ext/ping")
        assert r.status_code == 200 and r.json() == {"pong": True}


@pytest.mark.asyncio
async def test_get_extension_coerces_query_param_types():
    async with _client(_ExtHandler) as client:
        # "times" coerced str->int (else "hi"*"3" would TypeError); "upper" str->bool
        r = await client.get("/agentenv/ext/repeat", params={"text": "hi", "times": "3", "upper": "true"})
        assert r.status_code == 200 and r.json() == {"out": "HIHIHI", "times_type": "int"}
        # "false" must coerce to False, not a truthy non-empty string
        r2 = await client.get("/agentenv/ext/repeat", params={"text": "hi", "upper": "false"})
        assert r2.json()["out"] == "hi"


@pytest.mark.asyncio
async def test_get_extension_uncoercible_value_returns_400():
    async with _client(_ExtHandler) as client:
        r = await client.get("/agentenv/ext/repeat", params={"text": "hi", "times": "lots"})
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_params"


@pytest.mark.asyncio
async def test_get_extension_coerces_and_validates_int_literal():
    async with _client(_ExtHandler) as client:
        # raw query string "2" is coerced to the advertised int literal
        r = await client.get("/agentenv/ext/pick", params={"choice": "2"})
        assert r.status_code == 200 and r.json() == {"choice": 2, "choice_type": "int"}


@pytest.mark.asyncio
async def test_get_extension_value_outside_literal_returns_400():
    async with _client(_ExtHandler) as client:
        # 9 is not one of the advertised Literal[1, 2, 3] values -> reject, don't run the handler
        r = await client.get("/agentenv/ext/pick", params={"choice": "9"})
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_params"


@pytest.mark.asyncio
async def test_extension_post_value_outside_str_literal_returns_400():
    # mode: Literal["live", "fixed"] must reject "delete" even though it is a string matching the
    # advertised slot's type — the card only allows the enumerated values.
    async with _client(_TypesHandler, card=EnvironmentCard(name="items")) as client:
        r = await client.post("/agentenv/ext/types_demo", json={
            "mode": "delete", "color": "red", "priority": 2,
            "when": "2026-01-02T03:04:05+00:00", "day": "2026-01-02",
            "ident": "12345678-1234-5678-1234-567812345678",
            "amount": "19.99", "tags": [],
        })
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_params"


@pytest.mark.asyncio
async def test_extension_missing_required_param_returns_400():
    async with _client(_ExtHandler) as client:
        r = await client.post("/agentenv/ext/echo", json={})
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_params"


@pytest.mark.asyncio
async def test_extension_handler_error_returns_500():
    async with _client(_ExtHandler) as client:
        r = await client.post("/agentenv/ext/boom", json={})
        assert r.status_code == 500
        body = r.json()
        assert body["error"]["code"] == "extension_failed"
        assert "kaboom" in body["error"]["message"]


@pytest.mark.asyncio
async def test_card_declared_extension_not_duplicated_by_decorator():
    card = EnvironmentCard(
        name="items",
        capabilities=EnvironmentCapabilities(
            extensions=[EnvironmentExtension(uri="urn:agentenv:echo/v1", description="card-declared")]
        ),
    )
    async with _client(_ExtHandler, card=card) as client:
        body = (await client.get("/.well-known/agent-env.json")).json()
        echo = [e for e in body["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:echo/v1"]
        assert len(echo) == 1 and echo[0]["description"] == "card-declared"  # card entry wins
        r = await client.post("/agentenv/ext/echo", json={"message": "hi"})
        assert r.json() == {"echoed": "hi"}  # handler still invocable


def test_duplicate_extension_handler_raises():
    class _Dup(_ItemsHandler):
        @extension(uri="urn:agentenv:dup/v1")
        async def a(self) -> dict:
            return {}

        @extension(uri="urn:agentenv:dup/v1")
        async def b(self) -> dict:
            return {}

    with pytest.raises(ValueError, match="Multiple handlers for extension"):
        AgentEnvFastMCPApplication(environment_card=EnvironmentCard(name="x"), handler=_Dup())


def test_duplicate_extension_path_raises():
    class _DupPath(_ItemsHandler):
        @extension(uri="urn:agentenv:one/v1", path="/agentenv/ext/shared")
        async def one(self) -> dict:
            return {}

        @extension(uri="urn:agentenv:two/v1", path="/agentenv/ext/shared")
        async def two(self) -> dict:
            return {}

    with pytest.raises(ValueError, match="Multiple handlers for extension path"):
        AgentEnvFastMCPApplication(environment_card=EnvironmentCard(name="x"), handler=_DupPath())


class _ToolHandler(_ItemsHandler):
    @tool(description="Count items in the store.")
    async def count_items(self) -> dict:
        return {"count": len(self.store)}

    @tool(name="{environment_name}_add_item")
    async def add_item(
        self,
        item: Annotated[str, Field(description="The item to add.")],
        times: Annotated[int, Field(description="How many copies to add.")] = 1,
    ) -> dict:
        """Add an item to the store."""
        self.store.extend([item] * times)
        return {"count": len(self.store)}


def _mounted(handler, card: EnvironmentCard | None = None) -> _FakeMCP:
    mcp = _FakeMCP()
    AgentEnvFastMCPApplication(environment_card=card or EnvironmentCard(name="items"), handler=handler).add_routes_to_app(mcp)
    return mcp


@pytest.mark.asyncio
async def test_decorated_tool_registered_on_fastmcp_as_bound_method():
    handler = _ToolHandler()
    mcp = _mounted(handler)
    assert set(mcp.tools) == {"count_items", "items_add_item"}
    assert mcp.tools["count_items"][0] == "Count items in the store."
    assert mcp.tools["items_add_item"][0] == "Add an item to the store."  # docstring default
    assert await mcp.tools["items_add_item"][1](item="x", times=2) == {"count": 2}
    assert handler.store == ["x", "x"]  # bound to the same instance


@pytest.mark.asyncio
async def test_decorated_tool_advertised_on_card():
    async with _client(_ToolHandler, card=EnvironmentCard(name="items")) as client:
        body = (await client.get("/.well-known/agent-env.json")).json()
        tools = {t["name"]: t for t in body["capabilities"]["tools"]}
        assert set(tools) == {"count_items", "items_add_item"}
        assert tools["items_add_item"]["description"] == "Add an item to the store."
        assert tools["items_add_item"]["inputSchema"] == {
            "type": "object",
            "properties": {
                "item": {"type": "string", "description": "The item to add."},
                "times": {"type": "integer", "description": "How many copies to add.", "default": 1},
            },
            "required": ["item"],
        }


def test_legacy_service_token_no_longer_resolves():
    class _PingMixin:
        @tool(name="{service}_ping")
        async def ping(self) -> dict:
            return {"pong": True}

    class _WithMixin(_PingMixin, _ItemsHandler):
        pass

    with pytest.raises(ValueError, match="unresolved placeholder"):
        _mounted(_WithMixin(), card=EnvironmentCard(name="slack"))


def test_tool_environment_name_placeholder_resolves():
    class _PingMixin:
        @tool(name="{environment_name}_ping")
        async def ping(self) -> dict:
            return {"pong": True}

    class _WithMixin(_PingMixin, _ItemsHandler):
        pass

    mcp = _mounted(_WithMixin(), card=EnvironmentCard(name="slack"))
    assert "slack_ping" in mcp.tools


def test_tool_unresolved_placeholder_raises():
    class _BadMixin(_ItemsHandler):
        @tool(name="{env_name}_ping")
        async def ping(self) -> dict:
            return {"pong": True}

    with pytest.raises(ValueError, match="unresolved placeholder"):
        _mounted(_BadMixin(), card=EnvironmentCard(name="slack"))


def test_duplicate_tool_handler_raises():
    class _Dup(_ItemsHandler):
        @tool(name="dup")
        async def a(self) -> dict:
            return {}

        @tool(name="dup")
        async def b(self) -> dict:
            return {}

    with pytest.raises(ValueError, match="Multiple handlers for tool"):
        AgentEnvFastMCPApplication(environment_card=EnvironmentCard(name="x"), handler=_Dup())


@pytest.mark.asyncio
async def test_card_declared_tool_not_duplicated_by_decorator():
    card = EnvironmentCard(
        name="items",
        capabilities=EnvironmentCapabilities(tools=[EnvironmentTool(name="count_items", description="card-declared")]),
    )
    handler = _ToolHandler()
    mcp = _mounted(handler, card=card)
    assert "count_items" in mcp.tools  # handler still registered
    assert mcp.tools["count_items"][0] == "card-declared"  # registration matches the advertisement
    async with _client(_ToolHandler, card=card) as client:
        body = (await client.get("/.well-known/agent-env.json")).json()
        entries = [t for t in body["capabilities"]["tools"] if t["name"] == "count_items"]
        assert len(entries) == 1 and entries[0]["description"] == "card-declared"  # card entry wins


def test_starlette_application_with_tools_raises_at_construction():
    with pytest.raises(ValueError, match="FastMCP"):
        AgentEnvStarletteApplication(environment_card=EnvironmentCard(name="items"), handler=_ToolHandler())


def test_bare_tool_decorator_raises():
    with pytest.raises(TypeError, match="parentheses"):
        @tool
        async def nope() -> dict:
            return {}


@pytest.mark.asyncio
async def test_extension_schema_includes_annotated_field_descriptions():
    class _AnnotatedExt(_ItemsHandler):
        @extension(uri="urn:agentenv:describe/v1")
        async def describe(self, product_id: Annotated[str, Field(description="The product ID.")]) -> dict:
            return {"product_id": product_id}

    async with _client(_AnnotatedExt, card=EnvironmentCard(name="items")) as client:
        body = (await client.get("/.well-known/agent-env.json")).json()
        ext = next(e for e in body["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:describe/v1")
        assert ext["params"]["methods"]["describe"]["request"]["properties"]["product_id"] == {
            "type": "string",
            "description": "The product ID.",
        }
        # coercion still binds the underlying annotated type
        r = await client.post("/agentenv/ext/describe", json={"product_id": "p-1"})
        assert r.status_code == 200 and r.json() == {"product_id": "p-1"}


def test_tool_schema_parity_with_fastmcp_derivation():
    # Card and tools/list schemas are derived independently from the same signature — pin agreement.
    fastmcp_base = pytest.importorskip("mcp.server.fastmcp.tools.base")

    async def sample(
        product_id: Annotated[str, Field(description="The product ID.")],
        limit: Annotated[int, Field(description="Max results.")] = 10,
    ) -> str:
        """List products."""
        return "[]"

    ours = _schema_from_signature(sample)
    theirs = fastmcp_base.Tool.from_function(sample).parameters
    assert set(ours["properties"]) == set(theirs["properties"])
    assert set(ours.get("required", [])) == set(theirs.get("required", []))
    for prop, spec in ours["properties"].items():
        assert spec["type"] == theirs["properties"][prop]["type"]
        assert spec.get("description") == theirs["properties"][prop].get("description")


class _Color(enum.Enum):
    RED = "red"
    GREEN = "green"


class _Priority(enum.IntEnum):
    LOW = 1
    HIGH = 2


class _TypesHandler(_ItemsHandler):
    @extension(uri="urn:agentenv:types/v1")
    async def types_demo(
        self,
        mode: Literal["live", "fixed"],
        color: _Color,
        priority: _Priority,
        when: datetime.datetime,
        day: datetime.date,
        ident: uuid.UUID,
        amount: decimal.Decimal,
        tags: tuple,
        maybe: Optional[int] = None,
    ) -> dict:
        # Report the runtime types/derived ops the handler received so a test can prove the
        # request path coerced the advertised types instead of handing over raw strings.
        return {
            "mode": mode,
            "color": type(color).__name__,
            "color_is_enum": isinstance(color, _Color),
            "priority": type(priority).__name__,
            "priority_is_enum": isinstance(priority, _Priority),
            "when": type(when).__name__,
            "day": type(day).__name__,
            "ident": type(ident).__name__,
            "ident_hex": ident.hex,
            "amount": type(amount).__name__,
            "amount_x2": str(amount * 2),
            "maybe": maybe,
        }


@pytest.mark.asyncio
async def test_schema_covers_literal_enum_datetime_uuid_decimal_tuple():
    async with _client(_TypesHandler, card=EnvironmentCard(name="items")) as client:
        body = (await client.get("/.well-known/agent-env.json")).json()
        ext = next(e for e in body["capabilities"]["extensions"] if e["uri"] == "urn:agentenv:types/v1")
        req = ext["params"]["methods"]["types_demo"]["request"]
        props = req["properties"]
        assert props["mode"] == {"type": "string", "enum": ["live", "fixed"]}          # Literal
        assert props["color"] == {"type": "string", "enum": ["red", "green"]}          # str Enum -> values
        assert props["priority"] == {"type": "integer", "enum": [1, 2]}                # IntEnum -> values
        assert props["when"] == {"type": "string", "format": "date-time"}
        assert props["day"] == {"type": "string", "format": "date"}
        assert props["ident"] == {"type": "string", "format": "uuid"}
        assert props["amount"] == {"type": "string"}                                   # Decimal -> string
        assert props["tags"] == {"type": "array"}                                      # tuple -> array
        assert props["maybe"] == {"type": "integer", "default": None}                  # Optional[int] unwrapped
        assert "maybe" not in req["required"]
        assert set(req["required"]) == {"mode", "color", "priority", "when", "day", "ident", "amount", "tags"}


@pytest.mark.asyncio
async def test_extension_post_coerces_advertised_types():
    # A request matching the advertised schema must reach the handler as the annotated runtime
    # types (datetime/date/UUID/Decimal/enum), not raw strings — otherwise ident.hex / Decimal
    # arithmetic / enum checks 500 even though the request matched the card.
    async with _client(_TypesHandler, card=EnvironmentCard(name="items")) as client:
        r = await client.post("/agentenv/ext/types_demo", json={
            "mode": "live", "color": "red", "priority": 2,
            "when": "2026-01-02T03:04:05+00:00", "day": "2026-01-02",
            "ident": "12345678-1234-5678-1234-567812345678",
            "amount": "19.99", "tags": ["a", "b"],
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] == "live"
        assert body["color"] == "_Color" and body["color_is_enum"]
        assert body["priority"] == "_Priority" and body["priority_is_enum"]   # IntEnum from JSON int
        assert body["when"] == "datetime"
        assert body["day"] == "date"
        assert body["ident"] == "UUID" and body["ident_hex"] == "12345678123456781234567812345678"
        assert body["amount"] == "Decimal" and body["amount_x2"] == "39.98"


class _FakeServableMCP(_FakeMCP):
    def __init__(self) -> None:
        super().__init__()
        self.ran_with: str | None = None

    def run(self, transport: str) -> None:
        self.ran_with = transport


@environment_card(name="items")
class _ServeEnv(_ToolHandler, AgentEnvEnvironment):
    pass


def test_serve_zero_config_builds_app_and_runs(monkeypatch):
    built = _FakeServableMCP()
    monkeypatch.setattr(agent_env_environment, "create_fastmcp_app", lambda handler, **kw: built)
    env = _ServeEnv()
    env.serve()
    assert env.mcp is built and built.ran_with == "streamable-http"


def test_serve_mounts_byo_app_then_runs():
    env = _ServeEnv()
    env.mcp = _FakeServableMCP()
    env.serve()
    assert "items_add_item" in env.mcp.tools and env.mcp.ran_with == "streamable-http"


def test_serve_premounted_app_runs_without_remount():
    env = _ServeEnv()
    app = _FakeServableMCP()
    env.mount(app)
    tools_before = dict(app.tools)
    routes_before = len(app.routes)
    env.serve(transport="sse")
    assert app.tools == tools_before and len(app.routes) == routes_before
    assert app.ran_with == "sse"


def test_mount_twice_raises():
    env = _ServeEnv()
    env.mount(_FakeServableMCP())
    with pytest.raises(RuntimeError, match="already mounted"):
        env.mount(_FakeServableMCP())


def test_tools_only_environment_mounts():
    @environment_card(name="pinger")
    class _ToolsOnlyEnv(AgentEnvEnvironment):
        pass

        @tool(name="{environment_name}_ping")
        async def ping(self) -> dict:
            return {"pong": True}

    app = _ToolsOnlyEnv().mount(_FakeServableMCP())
    assert "pinger_ping" in app.tools


def test_create_app_after_mount_raises(monkeypatch):
    monkeypatch.setattr(agent_env_environment, "create_fastmcp_app", lambda handler, **kw: _FakeServableMCP())
    env = _ServeEnv()
    env.create_app()
    with pytest.raises(RuntimeError, match="already mounted"):
        env.create_app()


def test_injected_environment_name_overrides_card_config_name(monkeypatch):
    """ENVIRONMENT_NAME beats the card's declared name, and SERVICE_NAME is still ignored.

    agent-env injects ENVIRONMENT_NAME as the env's registered name, so a registration that
    renames an env must rename its card too, or card-first lookups by that name miss it.
    """
    monkeypatch.setenv("ENVIRONMENT_NAME", "slack_env")
    monkeypatch.setenv("SERVICE_NAME", "slack")
    env = _ServeEnv()
    app = env.mount(_FakeServableMCP())
    assert env._build_card().name == "slack_env"
    assert "slack_env_add_item" in app.tools
    assert "items_add_item" not in app.tools


@pytest.mark.parametrize("injected", [None, ""], ids=["unset", "empty"])
def test_carded_env_falls_back_to_card_name_without_environment_name(monkeypatch, injected):
    """Run outside agent-env (no ENVIRONMENT_NAME, or an empty one), the declared name is used."""
    if injected is None:
        monkeypatch.delenv("ENVIRONMENT_NAME", raising=False)
    else:
        monkeypatch.setenv("ENVIRONMENT_NAME", injected)
    env = _ServeEnv()
    app = env.mount(_FakeServableMCP())
    assert env._build_card().name == "items"
    assert "items_add_item" in app.tools


def test_service_name_env_is_no_longer_consulted(monkeypatch):
    """SERVICE_NAME is dead as an identity source: a cardless env with ONLY SERVICE_NAME set
    resolves to its class name, not to "slack".

    agent-env injects ENVIRONMENT_NAME everywhere it injects SERVICE_NAME, and
    every synthetic server declares @environment_card(name=...), so the legacy variable has
    no remaining reader. It is still injected for now; it is simply ignored here."""
    monkeypatch.delenv("ENVIRONMENT_NAME", raising=False)
    monkeypatch.setenv("SERVICE_NAME", "slack")

    class _Cardless(_ToolHandler, AgentEnvEnvironment):
        pass

    env = _Cardless()
    app = env.mount(_FakeServableMCP())
    assert env._build_card().name == "_Cardless"
    assert "_Cardless_add_item" in app.tools
    assert "slack_add_item" not in app.tools


def test_undecorated_class_defaults_card_name_to_class_name(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT_NAME", raising=False)
    monkeypatch.delenv("SERVICE_NAME", raising=False)

    class NotesEnv(_ToolHandler, AgentEnvEnvironment):
        pass

    env = NotesEnv()
    app = env.mount(_FakeServableMCP())
    assert env._build_card().name == "NotesEnv"
    assert "NotesEnv_add_item" in app.tools


def test_cardless_class_ignores_service_name_when_both_are_injected(monkeypatch):
    """The gateway still injects both names for the same container; only the new one is read.

    A deployed MCP container carries ENVIRONMENT_NAME and SERVICE_NAME side by side until a
    later release drops the legacy injection, so this pins that the stale one cannot win back.
    """
    monkeypatch.setenv("ENVIRONMENT_NAME", "slack")
    monkeypatch.setenv("SERVICE_NAME", "legacy_slack")

    class _Cardless(_ToolHandler, AgentEnvEnvironment):
        pass

    env = _Cardless()
    app = env.mount(_FakeServableMCP())
    assert env._build_card().name == "slack"
    assert "slack_add_item" in app.tools


def test_cardless_class_falls_back_to_environment_name_env(monkeypatch):
    """Post-rename images inject only ENVIRONMENT_NAME, with no SERVICE_NAME to fall back on."""
    monkeypatch.setenv("ENVIRONMENT_NAME", "slack")
    monkeypatch.delenv("SERVICE_NAME", raising=False)

    class _Cardless(_ToolHandler, AgentEnvEnvironment):
        pass

    env = _Cardless()
    app = env.mount(_FakeServableMCP())
    assert env._build_card().name == "slack"
    assert "slack_add_item" in app.tools


def test_blank_environment_name_env_falls_through_to_class_name(monkeypatch):
    """An injected-but-empty ENVIRONMENT_NAME is not a name — resolution keeps falling through.

    A compose file (or ``-e ENVIRONMENT_NAME=``) can set the variable to an empty string;
    that must degrade to the class name rather than naming the environment "". It must NOT
    reach back to SERVICE_NAME, even when SERVICE_NAME holds a usable value.
    """
    monkeypatch.setenv("ENVIRONMENT_NAME", "")
    monkeypatch.setenv("SERVICE_NAME", "slack")

    class _Cardless(_ToolHandler, AgentEnvEnvironment):
        pass

    env = _Cardless()
    assert env._build_card().name == "_Cardless"


class _CardlessEnv(_ToolHandler, AgentEnvEnvironment):
    """No ``@environment_card``: the shape of envs that are not synthetic servers, whose
    identity comes from resolution alone."""


# A value only ever reachable through the deleted ``SERVICE_NAME`` term: if it ever shows up in a
# card name, a tool name, or anywhere in the served card, the fallback has been reintroduced.
_SERVICE_NAME_SENTINEL = "legacy_service_name_sentinel"


async def _mount_and_fetch_card(env: AgentEnvEnvironment) -> tuple[dict, set[str]]:
    """Mount ``env`` and read back the card the wire actually serves, plus the registered tool names."""
    mcp = env.mount(_FakeServableMCP())
    app = Starlette(routes=mcp.routes)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        return (await client.get("/.well-known/agent-env.json")).json(), set(mcp.tools)


@pytest.mark.asyncio
async def test_cardless_env_with_only_environment_name_serves_it_as_identity(monkeypatch):
    """The deployed post-rename shape for a cardless env: the gateway injects
    ENVIRONMENT_NAME, nothing else names the env, and that value has to reach the wire —
    the served card *and* every ``{environment_name}`` tool name.
    """
    monkeypatch.setenv("ENVIRONMENT_NAME", "notesdesk")
    monkeypatch.delenv("SERVICE_NAME", raising=False)

    body, tools = await _mount_and_fetch_card(_CardlessEnv())

    assert body["name"] == "notesdesk"
    assert tools == {"count_items", "notesdesk_add_item"}
    assert {t["name"] for t in body["capabilities"]["tools"]} == tools


@pytest.mark.asyncio
async def test_only_service_name_leaves_no_trace_anywhere_in_the_served_card(monkeypatch):
    """An image that still injects only SERVICE_NAME names itself after its class, and the
    SERVICE_NAME value leaks nowhere — not the card name, not a tool name, not any other field.
    """
    monkeypatch.delenv("ENVIRONMENT_NAME", raising=False)
    monkeypatch.setenv("SERVICE_NAME", _SERVICE_NAME_SENTINEL)

    body, tools = await _mount_and_fetch_card(_CardlessEnv())

    assert body["name"] == "_CardlessEnv"
    assert tools == {"count_items", "_CardlessEnv_add_item"}
    assert _SERVICE_NAME_SENTINEL not in json.dumps(body)


@pytest.mark.asyncio
async def test_carded_env_serves_environment_name_and_ignores_service_name(monkeypatch):
    """The injected ENVIRONMENT_NAME reaches the wire over the declared name; SERVICE_NAME never does."""
    monkeypatch.setenv("ENVIRONMENT_NAME", "injected_environment_name")
    monkeypatch.setenv("SERVICE_NAME", _SERVICE_NAME_SENTINEL)

    body, tools = await _mount_and_fetch_card(_ServeEnv())

    assert body["name"] == "injected_environment_name"
    assert tools == {"count_items", "injected_environment_name_add_item"}
    assert {t["name"] for t in body["capabilities"]["tools"]} == tools
    assert _SERVICE_NAME_SENTINEL not in json.dumps(body)


@pytest.mark.parametrize(
    "injected,expected",
    [("", "_CardlessEnv"), ("   ", "   "), ("\t", "\t"), ("\n", "\n")],
    ids=["empty", "spaces", "tab", "newline"],
)
def test_blankish_environment_name_never_reaches_service_name(monkeypatch, injected, expected):
    """``or`` rejects only the empty string, so that is the whole "blank" contract.

    ``ENVIRONMENT_NAME=`` falls through to the class name; a whitespace-only value is truthy and
    is taken verbatim (a pre-existing wart — under the old chain it shadowed SERVICE_NAME too,
    since ENVIRONMENT_NAME always came first). Either way resolution never reaches back to
    SERVICE_NAME, which is what this pins.
    """
    monkeypatch.setenv("ENVIRONMENT_NAME", injected)
    monkeypatch.setenv("SERVICE_NAME", _SERVICE_NAME_SENTINEL)

    assert _CardlessEnv()._build_card().name == expected


@pytest.mark.asyncio
async def test_environment_name_fallback_leaves_the_rest_of_the_card_config_intact(monkeypatch):
    """A card may declare everything *but* the name and take its name from the deployment.

    Only ``name`` is resolved; every other ``@environment_card`` field is passed through to the
    served card verbatim.
    """
    monkeypatch.setenv("ENVIRONMENT_NAME", "hubspot")
    monkeypatch.setenv("SERVICE_NAME", _SERVICE_NAME_SENTINEL)

    @environment_card(
        protocolVersion="9.9",
        url="/custom-rpc",
        preferredTransport="HTTP",
        additionalInterfaces=[EnvironmentInterface(url="/mcp", transport="streamable-http")],
        capabilities=EnvironmentCapabilities(
            extensions=[EnvironmentExtension(uri="urn:agentenv:pre/v1", params={"k": "v"})]
        ),
    )
    class _NamelessCard(_ToolHandler, AgentEnvEnvironment):
        pass

    body, tools = await _mount_and_fetch_card(_NamelessCard())

    assert body["name"] == "hubspot" and tools == {"count_items", "hubspot_add_item"}
    assert body["protocolVersion"] == "9.9"
    assert body["url"] == "/custom-rpc" and body["preferredTransport"] == "HTTP"
    assert body["additionalInterfaces"] == [{"url": "/mcp", "transport": "streamable-http"}, {"url": "/mcp", "transport": "mcp"}]
    assert body["capabilities"]["extensions"][0] == {
        "uri": "urn:agentenv:pre/v1", "description": None, "params": {"k": "v"}, "required": None,
    }
    assert _SERVICE_NAME_SENTINEL not in json.dumps(body)


@pytest.mark.parametrize(
    "env_cls,environment_name,expected",
    [
        (_ServeEnv, "injected_environment_name", "injected_environment_name"),
        (_ServeEnv, None, "items"),
        (_CardlessEnv, "injected_environment_name", "injected_environment_name"),
        (_CardlessEnv, None, "_CardlessEnv"),
    ],
    ids=["carded_with_environment_name", "carded_only", "cardless_with_environment_name", "cardless_only"],
)
def test_service_name_sentinel_never_wins_any_resolution_branch(monkeypatch, env_cls, environment_name, expected):
    """The regression net for the dropped fallback, across every branch of the ``or`` chain.

    Re-adding a ``SERVICE_NAME`` term anywhere it could win flips one of these cases: ahead of
    ENVIRONMENT_NAME breaks both ``*_with_environment_name`` cases, ahead of the card name breaks
    ``carded_only``, ahead of the class name breaks ``cardless_only``.
    """
    monkeypatch.setenv("SERVICE_NAME", _SERVICE_NAME_SENTINEL)
    if environment_name is None:
        monkeypatch.delenv("ENVIRONMENT_NAME", raising=False)
    else:
        monkeypatch.setenv("ENVIRONMENT_NAME", environment_name)

    env = env_cls()
    tools = set(env.mount(_FakeServableMCP()).tools)

    assert env._build_card().name == expected
    assert tools == {"count_items", f"{expected}_add_item"}


def test_environment_card_unknown_field_raises():
    with pytest.raises(TypeError, match="unknown EnvironmentCard field"):
        environment_card(nme="typo")


def test_environment_card_without_parentheses_raises():
    with pytest.raises(TypeError, match="parentheses"):
        @environment_card
        class _Nope:
            pass


@pytest.mark.asyncio
async def test_environment_card_config_preserved_through_mount(monkeypatch):
    monkeypatch.delenv("SERVICE_NAME", raising=False)

    @environment_card(
        name="items",
        capabilities=EnvironmentCapabilities(
            extensions=[EnvironmentExtension(uri="urn:agentenv:pre/v1", params={"k": "v"})]
        ),
    )
    class _Seeded(_ToolHandler, AgentEnvEnvironment):
        pass

    mcp = _Seeded().mount(_FakeServableMCP())
    app = Starlette(routes=mcp.routes)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        body = (await client.get("/.well-known/agent-env.json")).json()
        assert body["name"] == "items"
        assert body["capabilities"]["extensions"][0]["uri"] == "urn:agentenv:pre/v1"
        assert {t["name"] for t in body["capabilities"]["tools"]} == {"count_items", "items_add_item"}


def test_environment_card_config_inherited_by_subclass(monkeypatch):
    monkeypatch.delenv("SERVICE_NAME", raising=False)

    class _Child(_ServeEnv):
        pass

    app = _Child().mount(_FakeServableMCP())
    assert "items_add_item" in app.tools


def test_create_fastmcp_app_without_mcp_raises_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", None)
    with pytest.raises(ImportError, match="requires the 'mcp' package"):
        create_fastmcp_app(_ToolHandler(), card=EnvironmentCard(name="items"))


@pytest.mark.asyncio
async def test_create_fastmcp_app_encodes_deploy_contract(monkeypatch):
    pytest.importorskip("mcp.server.fastmcp")
    monkeypatch.delenv("SERVICE_NAME", raising=False)
    monkeypatch.setenv("MCP_HOST", "127.0.0.1")
    monkeypatch.setenv("MCP_PORT", "19999")
    handler = _ToolHandler()
    app = create_fastmcp_app(handler, card=EnvironmentCard(name="items"))
    assert app.settings.host == "127.0.0.1" and app.settings.port == 19999
    assert app.settings.transport_security.enable_dns_rebinding_protection is False
    assert {"count_items", "items_add_item"} <= {t.name for t in await app.list_tools()}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.streamable_http_app()), base_url="http://t") as client:
        card = (await client.get("/.well-known/agent-env.json")).json()
        assert card["name"] == "items"
        assert {t["name"] for t in card["capabilities"]["tools"]} == {"count_items", "items_add_item"}
        r = await client.post("/agentenv", json=_rpc("data/get"))
        assert r.json()["result"]["parts"][0]["data"] == {"items": []}
    await app.call_tool("items_add_item", {"item": "x", "times": 2})
    assert handler.store == ["x", "x"]


def test_create_fastmcp_app_uses_card_verbatim(monkeypatch):
    # identity is the caller's card; SERVICE_NAME resolution lives in the env's card config
    pytest.importorskip("mcp.server.fastmcp")
    monkeypatch.setenv("SERVICE_NAME", "fromenv")
    assert create_fastmcp_app(_ToolHandler(), card=EnvironmentCard(name="carded")).name == "carded"


@pytest.mark.asyncio
async def test_create_app_allows_imperative_registration_before_serve(monkeypatch):
    pytest.importorskip("mcp.server.fastmcp")
    monkeypatch.delenv("SERVICE_NAME", raising=False)
    env = _ServeEnv()
    app = env.create_app()
    assert app is env.mcp

    def extra() -> str:
        """Extra imperative tool."""
        return "ok"

    app.tool(name="extra")(extra)
    assert {"count_items", "items_add_item", "extra"} <= {t.name for t in await app.list_tools()}


@pytest.mark.asyncio
async def test_extension_post_invalid_typed_value_returns_400():
    # A value that matches the advertised type's slot but is unparseable (bad ISO datetime) is a
    # client error — surface 400 invalid_params, not a 500 from the handler.
    async with _client(_TypesHandler, card=EnvironmentCard(name="items")) as client:
        r = await client.post("/agentenv/ext/types_demo", json={
            "mode": "live", "color": "red", "priority": 2,
            "when": "not-a-datetime", "day": "2026-01-02",
            "ident": "12345678-1234-5678-1234-567812345678",
            "amount": "19.99", "tags": [],
        })
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "invalid_params"
