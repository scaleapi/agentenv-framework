"""Unit tests for the v1 data-plane protocol client."""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agentenv_protocol import DataPart, client as protocol_v1

_BASE = "http://gw/svc/mcp-items"


def _resp(json_value, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json = MagicMock(return_value=json_value)
    if status >= 400:
        err = httpx.HTTPStatusError("err", request=MagicMock(), response=MagicMock(status_code=status))
        r.raise_for_status = MagicMock(side_effect=err)
    else:
        r.raise_for_status = MagicMock()
    return r


def _client(resp: MagicMock) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    client.get = AsyncMock(return_value=resp)
    client.request = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


def _patch(client: MagicMock):
    return patch("agentenv_protocol.client.httpx.AsyncClient", return_value=client)


@pytest.mark.asyncio
async def test_reset_data_posts_rpc():
    client = _client(_resp({"jsonrpc": "2.0", "id": 1, "result": {}}))
    with _patch(client):
        assert (await protocol_v1.reset_data(_BASE)).model_dump() == {}
    assert client.post.call_args.args[0] == f"{_BASE}/agentenv"
    assert client.post.call_args.kwargs["json"]["method"] == "data/reset"


@pytest.mark.asyncio
async def test_add_data_posts_parts():
    client = _client(_resp({"jsonrpc": "2.0", "id": 1, "result": {}}))
    with _patch(client):
        assert (await protocol_v1.add_data(_BASE, [DataPart(data={"items": ["a"]})])).model_dump() == {}
    sent = client.post.call_args.kwargs["json"]
    assert client.post.call_args.args[0] == f"{_BASE}/agentenv"
    assert sent["method"] == "data/add"
    assert sent["params"] == {"parts": [{"kind": "data", "data": {"items": ["a"]}}]}


@pytest.mark.asyncio
async def test_get_data_posts_rpc():
    client = _client(_resp({"jsonrpc": "2.0", "id": 1, "result": {"parts": [{"kind": "data", "data": {"items": []}}]}}))
    with _patch(client):
        assert (await protocol_v1.get_data(_BASE)).parts[0].data == {"items": []}
    assert client.post.call_args.args[0] == f"{_BASE}/agentenv"
    assert client.post.call_args.kwargs["json"]["method"] == "data/get"


@pytest.mark.asyncio
async def test_get_card_gets_well_known():
    client = _client(_resp({"name": "items"}))
    with _patch(client):
        assert await protocol_v1.get_card(_BASE) == {"name": "items"}
    assert client.get.call_args.args[0] == f"{_BASE}/.well-known/agent-env.json"


@pytest.mark.asyncio
async def test_supports_v1_true_on_card():
    with _patch(_client(_resp({"name": "items"}))):
        assert await protocol_v1.supports_v1(_BASE) is True


@pytest.mark.asyncio
async def test_supports_v1_false_on_404():
    with _patch(_client(_resp(None, status=404))):
        assert await protocol_v1.supports_v1(_BASE) is False


@pytest.mark.asyncio
async def test_supports_v1_false_on_500():
    with _patch(_client(_resp(None, status=500))):
        assert await protocol_v1.supports_v1(_BASE) is False


@pytest.mark.asyncio
async def test_supports_v1_false_on_connect_error():
    client = MagicMock()
    client.get = AsyncMock(side_effect=httpx.ConnectError("boom"))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with _patch(client):
        assert await protocol_v1.supports_v1(_BASE) is False


_CARD_WITH_EXT = {
    "name": "items",
    "capabilities": {
        "extensions": [
            {"uri": "urn:agentenv:clock/v1", "params": {"mode": "fixed"}},
            {"uri": "urn:agentenv:auth/v1"},
        ],
    },
}


def test_find_extension_returns_matching():
    assert protocol_v1.find_extension(_CARD_WITH_EXT, "urn:agentenv:clock/v1") == {
        "uri": "urn:agentenv:clock/v1", "params": {"mode": "fixed"}
    }


def test_find_extension_unknown_uri_returns_none():
    assert protocol_v1.find_extension(_CARD_WITH_EXT, "urn:agentenv:nope/v1") is None


def test_find_extension_tolerates_missing_or_null_extensions():
    assert protocol_v1.find_extension({"name": "items"}, "urn:agentenv:clock/v1") is None
    assert protocol_v1.find_extension({"capabilities": None}, "urn:agentenv:clock/v1") is None
    assert protocol_v1.find_extension({"capabilities": {"extensions": None}}, "urn:agentenv:clock/v1") is None


def test_extension_params_returns_params_when_present():
    assert protocol_v1.extension_params(_CARD_WITH_EXT, "urn:agentenv:clock/v1") == {"mode": "fixed"}


def test_extension_params_returns_empty_when_absent_or_paramless():
    assert protocol_v1.extension_params(_CARD_WITH_EXT, "urn:agentenv:auth/v1") == {}
    assert protocol_v1.extension_params({"name": "items"}, "urn:agentenv:clock/v1") == {}


_CARD_WITH_TOOLS = {
    "name": "slack",
    "capabilities": {
        "tools": [{"name": "slack_ping", "description": "Ping the server.", "inputSchema": {"type": "object", "properties": {}}}],
    },
}


def test_find_tool_returns_matching():
    assert protocol_v1.find_tool(_CARD_WITH_TOOLS, "slack_ping")["description"] == "Ping the server."


def test_find_tool_tolerates_unknown_missing_or_null_tools():
    assert protocol_v1.find_tool(_CARD_WITH_TOOLS, "slack_nope") is None
    assert protocol_v1.find_tool({"name": "items"}, "slack_ping") is None
    assert protocol_v1.find_tool({"capabilities": None}, "slack_ping") is None
    assert protocol_v1.find_tool({"capabilities": {"tools": None}}, "slack_ping") is None


_CARD_WITH_CHILDREN = {
    "name": "gw",
    "capabilities": {"extensions": [{"uri": "urn:agentenv:disable-tool/v1"}]},
    "children_environments": [
        {"name": "slack", "capabilities": {"extensions": [{"uri": "urn:agentenv:set-errors/v1"}]}},
        {"name": "linear", "capabilities": {"extensions": [{"uri": "urn:agentenv:set-errors/v1"}]}},
    ],
}


def test_find_child_returns_matching():
    assert protocol_v1.find_child(_CARD_WITH_CHILDREN, "linear")["name"] == "linear"


def test_find_child_unknown_name_returns_none():
    assert protocol_v1.find_child(_CARD_WITH_CHILDREN, "gmail") is None


def test_find_child_tolerates_missing_children():
    assert protocol_v1.find_child({"name": "leaf"}, "slack") is None
    assert protocol_v1.find_child({"children_environments": None}, "slack") is None


def test_find_extension_ignores_children():
    # set-errors/v1 is advertised only on the children, never at the top level
    assert protocol_v1.find_extension(_CARD_WITH_CHILDREN, "urn:agentenv:set-errors/v1") is None
    # the gateway-level extension stays discoverable at the top level
    assert protocol_v1.find_extension(_CARD_WITH_CHILDREN, "urn:agentenv:disable-tool/v1") is not None


_CARD_SET_ERRORS = {"capabilities": {"extensions": [{
    "uri": "urn:agentenv:set-errors/v1",
    "params": {"endpoint": "/agentenv/ext/set_errors",
               "methods": {"set_errors": {"method": "POST", "request": {}}}},
}]}}


@pytest.mark.asyncio
async def test_invoke_extension_posts_to_advertised_endpoint():
    client = _client(_resp({"tool_name": "list_items", "error_rate": 0.75}))
    with _patch(client):
        result = await protocol_v1.invoke_extension(
            _BASE, _CARD_SET_ERRORS, "urn:agentenv:set-errors/v1",
            {"tool_name": "list_items", "error_rate": 0.75},
        )
    assert result == {"tool_name": "list_items", "error_rate": 0.75}
    args, kwargs = client.request.call_args
    assert args[0] == "POST" and args[1] == f"{_BASE}/agentenv/ext/set_errors"
    assert kwargs["json"] == {"tool_name": "list_items", "error_rate": 0.75}


@pytest.mark.asyncio
async def test_invoke_extension_get_uses_query_params():
    card = {"capabilities": {"extensions": [{
        "uri": "urn:agentenv:ping/v1",
        "params": {"endpoint": "/agentenv/ext/ping", "methods": {"ping": {"method": "GET", "request": {}}}},
    }]}}
    client = _client(_resp({"pong": True}))
    with _patch(client):
        result = await protocol_v1.invoke_extension(_BASE, card, "urn:agentenv:ping/v1")
    assert result == {"pong": True}
    assert client.get.call_args.args[0] == f"{_BASE}/agentenv/ext/ping"


@pytest.mark.asyncio
async def test_invoke_extension_unknown_uri_raises():
    with pytest.raises(ValueError, match="not advertised"):
        await protocol_v1.invoke_extension(_BASE, {"capabilities": {"extensions": []}}, "urn:agentenv:nope/v1")


def test_mcp_path_reads_the_declared_interface():
    card = {"additionalInterfaces": [{"url": "/grpc", "transport": "grpc"}, {"url": "/rpc/mcp", "transport": "mcp"}]}
    assert protocol_v1.mcp_path(card) == "/rpc/mcp"


def test_mcp_path_falls_back_to_convention():
    assert protocol_v1.mcp_path({"name": "items"}) == "/mcp"
    assert protocol_v1.mcp_path({"additionalInterfaces": None}) == "/mcp"
    assert protocol_v1.mcp_path({"additionalInterfaces": [{"url": "/stream", "transport": "streamable-http"}]}) == "/mcp"


# Shaped like the gateway's clock/v1, where each method declares its own endpoint.
_CARD_CLOCK = {"capabilities": {"extensions": [{
    "uri": "urn:agentenv:clock/v1",
    "params": {"endpoint": "/clock/time", "methods": {
        "get_time": {"method": "GET", "endpoint": "/clock/time"},
        "set_time": {"method": "PUT", "endpoint": "/clock/set-time", "request": {}},
        "state": {"method": "GET", "endpoint": "/clock/state"},
    }},
}]}}


def test_find_extension_method_returns_the_methods_own_endpoint():
    assert protocol_v1.find_extension_method(_CARD_CLOCK, "urn:agentenv:clock/v1", "set_time") == {
        "method": "PUT", "endpoint": "/clock/set-time", "request": {},
    }


def test_find_extension_method_falls_back_to_the_extension_endpoint():
    method = protocol_v1.find_extension_method(_CARD_SET_ERRORS, "urn:agentenv:set-errors/v1", "set_errors")
    assert method == {"method": "POST", "request": {}, "endpoint": "/agentenv/ext/set_errors"}


def test_find_extension_method_unknown_method_or_uri_returns_none():
    assert protocol_v1.find_extension_method(_CARD_CLOCK, "urn:agentenv:clock/v1", "sync_time") is None
    assert protocol_v1.find_extension_method(_CARD_CLOCK, "urn:agentenv:nope/v1", "set_time") is None
    assert protocol_v1.find_extension_method(_CARD_WITH_EXT, "urn:agentenv:auth/v1", "set_time") is None


@pytest.mark.asyncio
async def test_invoke_extension_calls_the_named_method():
    client = _client(_resp({"virtual_time": "2026-01-01T00:00:00Z"}))
    with _patch(client):
        await protocol_v1.invoke_extension(
            _BASE, _CARD_CLOCK, "urn:agentenv:clock/v1", {"time": "2026-01-01T00:00:00Z"}, method="set_time",
        )
    args, kwargs = client.request.call_args
    assert args[0] == "PUT" and args[1] == f"{_BASE}/clock/set-time"
    assert kwargs["json"] == {"time": "2026-01-01T00:00:00Z"}


@pytest.mark.asyncio
async def test_invoke_extension_named_get_method_uses_query_params():
    client = _client(_resp({"frozen": False}))
    with _patch(client):
        assert await protocol_v1.invoke_extension(_BASE, _CARD_CLOCK, "urn:agentenv:clock/v1", method="state") == {"frozen": False}
    assert client.get.call_args.args[0] == f"{_BASE}/clock/state"


@pytest.mark.asyncio
async def test_invoke_extension_without_a_method_calls_the_first_listed():
    client = _client(_resp({}))
    with _patch(client):
        await protocol_v1.invoke_extension(_BASE, _CARD_CLOCK, "urn:agentenv:clock/v1")
    assert client.get.call_args.args[0] == f"{_BASE}/clock/time"
    client.request.assert_not_called()


@pytest.mark.asyncio
async def test_invoke_extension_unknown_method_raises_without_calling():
    client = _client(_resp({}))
    offered = "does not advertise method 'sync_time' (advertises: get_time, set_time, state)"
    with _patch(client), pytest.raises(ValueError, match=re.escape(offered)):
        await protocol_v1.invoke_extension(_BASE, _CARD_CLOCK, "urn:agentenv:clock/v1", method="sync_time")
    client.get.assert_not_called()
    client.request.assert_not_called()


@pytest.mark.asyncio
async def test_invoke_extension_named_method_on_a_methodless_extension_raises():
    with pytest.raises(ValueError, match=re.escape("does not advertise method 'set_time' (advertises: none)")):
        await protocol_v1.invoke_extension(_BASE, _CARD_WITH_EXT, "urn:agentenv:clock/v1", method="set_time")
