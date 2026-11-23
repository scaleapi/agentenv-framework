"""Unit tests for the v1 data-plane protocol client."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from unittest.mock import patch

import httpx
import pytest

from agentenv_protocol import DataPart, ExtensionCallError, client as protocol_v1

_BASE = "http://gw/svc/mcp-items"


@contextmanager
def _serve(response: httpx.Response | httpx.ConnectError):
    requests = []

    def handle(request):
        requests.append(request)
        if isinstance(response, httpx.ConnectError):
            raise response
        return response

    transport = httpx.MockTransport(handle)
    client = httpx.AsyncClient(transport=transport)
    with patch("agentenv_protocol.client.httpx.AsyncClient", return_value=client):
        yield requests


@pytest.mark.asyncio
async def test_reset_data_posts_rpc():
    with _serve(httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})) as requests:
        assert (await protocol_v1.reset_data(_BASE)).model_dump() == {}
    assert [(r.method, str(r.url)) for r in requests] == [("POST", f"{_BASE}/agentenv")]
    assert json.loads(requests[0].content)["method"] == "data/reset"


@pytest.mark.asyncio
async def test_add_data_posts_parts():
    with _serve(httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})) as requests:
        assert (await protocol_v1.add_data(_BASE, [DataPart(data={"items": ["a"]})])).model_dump() == {}
    sent = json.loads(requests[0].content)
    assert [(r.method, str(r.url)) for r in requests] == [("POST", f"{_BASE}/agentenv")]
    assert sent["method"] == "data/add"
    assert sent["params"] == {"parts": [{"kind": "data", "data": {"items": ["a"]}}]}


@pytest.mark.asyncio
async def test_get_data_posts_rpc():
    with _serve(httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"parts": [{"kind": "data", "data": {"items": []}}]}})) as requests:
        assert (await protocol_v1.get_data(_BASE)).parts[0].data == {"items": []}
    assert [(r.method, str(r.url)) for r in requests] == [("POST", f"{_BASE}/agentenv")]
    assert json.loads(requests[0].content)["method"] == "data/get"


@pytest.mark.asyncio
async def test_get_card_gets_well_known():
    with _serve(httpx.Response(200, json={"name": "items"})) as requests:
        assert await protocol_v1.get_card(_BASE) == {"name": "items"}
    assert [(r.method, str(r.url)) for r in requests] == [("GET", f"{_BASE}/.well-known/agent-env.json")]


@pytest.mark.asyncio
async def test_rpc_and_card_strip_trailing_slashes_from_base_url():
    base_with_slash = f"{_BASE}/"
    with _serve(httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})) as requests:
        await protocol_v1.reset_data(base_with_slash)
    assert [(r.method, str(r.url)) for r in requests] == [("POST", f"{_BASE}/agentenv")]

    with _serve(httpx.Response(200, json={"name": "items"})) as requests:
        await protocol_v1.get_card(base_with_slash)
    assert [(r.method, str(r.url)) for r in requests] == [("GET", f"{_BASE}/.well-known/agent-env.json")]


@pytest.mark.asyncio
async def test_supports_v1_true_on_card():
    with _serve(httpx.Response(200, json={"name": "items"})):
        assert await protocol_v1.supports_v1(_BASE) is True


@pytest.mark.asyncio
async def test_supports_v1_false_on_404():
    with _serve(httpx.Response(404)):
        assert await protocol_v1.supports_v1(_BASE) is False


@pytest.mark.asyncio
async def test_supports_v1_false_on_500():
    with _serve(httpx.Response(500)):
        assert await protocol_v1.supports_v1(_BASE) is False


@pytest.mark.asyncio
async def test_supports_v1_false_on_connect_error():
    with _serve(httpx.ConnectError("boom")):
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


def test_operations_are_what_a_card_declares_and_none_when_it_declares_none():
    assert protocol_v1.operations({"capabilities": {"operations": ["data/get"]}}) == ["data/get"]
    assert protocol_v1.operations({"capabilities": {"operations": []}}) == []
    assert protocol_v1.operations({"capabilities": {}}) is None
    assert protocol_v1.operations({"capabilities": None}) is None
    assert protocol_v1.operations({}) is None


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


_SHARED_NAME = {"name": "gw", "children_environments": [
    {"name": "mail", "url": "/svc/mcp-mail/agentenv", "capabilities": {"operations": ["data/get"], "extensions": [
        {"uri": "urn:agentenv:clock/v1"}]}},
    {"name": "mail", "url": "/svc/mail/agentenv", "capabilities": {"operations": []}},
    {"name": "docs", "capabilities": {}},
    "not a card",
]}


def test_children_are_each_nested_card_by_its_name_url_and_card():
    found = protocol_v1.children(_SHARED_NAME)

    assert [(c.name, c.url) for c in found] == [("mail", "/svc/mcp-mail/agentenv"), ("mail", "/svc/mail/agentenv"), ("docs", None)]
    assert found[0].card is _SHARED_NAME["children_environments"][0]
    assert [c.extensions for c in found] == [[{"uri": "urn:agentenv:clock/v1"}], [], []]
    assert [c.operations for c in found] == [["data/get"], [], None]
    assert [c.serves("data/get") for c in found] == [True, False, None]
    assert found[0].find_extension("urn:agentenv:clock/v1") is not None and found[1].find_extension("urn:agentenv:clock/v1") is None
    assert protocol_v1.children({"name": "leaf"}) == [] and protocol_v1.children({"children_environments": None}) == []


def test_a_childs_base_url_is_its_url_without_the_rpc_path_against_the_composed_cards_address():
    mcp, website, docs = protocol_v1.children(_SHARED_NAME)

    assert mcp.base_url("https://env.example/") == "https://env.example/svc/mcp-mail"
    assert website.base_url("https://env.example") == "https://env.example/svc/mail"
    assert docs.base_url("https://env.example") == "https://env.example"
    assert protocol_v1.ChildEnv("x", "https://x.example/agentenv", {}).base_url("https://env.example") == "https://x.example"


def test_child_at_finds_a_child_by_its_url_or_data_plane_route_where_two_share_a_name():
    assert protocol_v1.child_at(_SHARED_NAME, "/svc/mail").url == "/svc/mail/agentenv"
    assert protocol_v1.child_at(_SHARED_NAME, "/svc/mcp-mail/").url == "/svc/mcp-mail/agentenv"
    assert protocol_v1.child_at(_SHARED_NAME, "/svc/mail/agentenv").url == "/svc/mail/agentenv"
    assert protocol_v1.child_at(_SHARED_NAME, "/svc/docs") is None
    assert protocol_v1.child_at({"name": "leaf"}, "/svc/mail") is None


def test_child_named_refuses_a_name_two_children_share():
    assert protocol_v1.child_named(_SHARED_NAME, "docs").name == "docs"
    assert protocol_v1.child_named(_SHARED_NAME, "gmail") is None
    with pytest.raises(protocol_v1.AmbiguousChildError, match="2 child envs are named 'mail'"):
        protocol_v1.child_named(_SHARED_NAME, "mail")
    assert protocol_v1.find_child(_SHARED_NAME, "mail")["url"] == "/svc/mcp-mail/agentenv"


_STATED = {"name": "gw", "children_environments": [
    {"name": "Notes", "url": "/svc/notes/agentenv", "environment_name": "notes"},
    {"name": "slack", "url": "/svc/mcp-slack/agentenv"},
]}


def test_a_child_is_known_by_the_environment_name_its_composer_states_else_by_its_cards_name():
    notes, slack = protocol_v1.children(_STATED)
    assert (notes.name, notes.environment_name, notes.known_as) == ("Notes", "notes", "notes")
    assert (slack.environment_name, slack.known_as) == (None, "slack")  # a composer that states none
    assert protocol_v1.child_named(_STATED, "notes").url == "/svc/notes/agentenv"
    assert protocol_v1.child_named(_STATED, "Notes") is None
    assert protocol_v1.find_child(_STATED, "notes")["name"] == "Notes"
    assert protocol_v1.find_child(_STATED, "slack")["url"] == "/svc/mcp-slack/agentenv"


def test_serves_is_what_a_card_declares_of_a_method_and_none_when_it_declares_no_operations():
    assert protocol_v1.serves({"capabilities": {"operations": ["data/get"]}}, "data/get") is True
    assert protocol_v1.serves({"capabilities": {"operations": ["data/add"]}}, "data/get") is False
    assert protocol_v1.serves({"capabilities": {}}, "data/get") is None


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
    with _serve(httpx.Response(200, json={"tool_name": "list_items", "error_rate": 0.75})) as requests:
        result = await protocol_v1.invoke_extension(
            _BASE, _CARD_SET_ERRORS, "urn:agentenv:set-errors/v1",
            {"tool_name": "list_items", "error_rate": 0.75},
        )
    assert result == {"tool_name": "list_items", "error_rate": 0.75}
    request, = requests
    assert (request.method, str(request.url)) == ("POST", f"{_BASE}/agentenv/ext/set_errors")
    assert json.loads(request.content) == {"tool_name": "list_items", "error_rate": 0.75}


@pytest.mark.asyncio
async def test_invoke_extension_strips_trailing_slashes_from_base_url():
    base_with_slash = f"{_BASE}/"
    with _serve(httpx.Response(200, json={"tool_name": "list_items", "error_rate": 0.75})) as requests:
        await protocol_v1.invoke_extension(
            base_with_slash, _CARD_SET_ERRORS, "urn:agentenv:set-errors/v1",
            {"tool_name": "list_items", "error_rate": 0.75},
        )
    assert [str(r.url) for r in requests] == [f"{_BASE}/agentenv/ext/set_errors"]


@pytest.mark.asyncio
async def test_invoke_extension_get_uses_query_params():
    card = {"capabilities": {"extensions": [{
        "uri": "urn:agentenv:ping/v1",
        "params": {"endpoint": "/agentenv/ext/ping", "methods": {"ping": {"method": "GET", "request": {}}}},
    }]}}
    with _serve(httpx.Response(200, json={"pong": True})) as requests:
        result = await protocol_v1.invoke_extension(_BASE, card, "urn:agentenv:ping/v1")
    assert result == {"pong": True}
    assert [(r.method, str(r.url)) for r in requests] == [("GET", f"{_BASE}/agentenv/ext/ping")]


def _error_response(status: int, **kwargs) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", f"{_BASE}/agentenv/ext/set_errors"), **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("response, fields", [
    (_error_response(429, json={"ok": False, "error": {
        "code": "limit_exceeded", "message": "3 leases held", "retryable": True, "applied": "no", "hint": "release one"}}),
     ("limit_exceeded", "3 leases held", True, "no", "release one")),
    # A server from before the structured fields: its code, and nothing claimed about the effect.
    (_error_response(500, json={"ok": False, "error": {"code": "extension_failed", "message": "kaboom"}}),
     ("extension_failed", "kaboom", False, "unknown", None)),
    # A proxy's own error page.
    (_error_response(502, text="<html>Bad Gateway</html>"), (None, "Bad Gateway", False, "unknown", None)),
    (_error_response(404, json=["not", "an", "object"]), (None, "Not Found", False, "unknown", None)),
])
async def test_invoke_extension_raises_the_error_fields_it_was_sent(response, fields):
    with _serve(response), pytest.raises(ExtensionCallError) as raised:
        await protocol_v1.invoke_extension(_BASE, _CARD_SET_ERRORS, "urn:agentenv:set-errors/v1", {})
    error = raised.value
    assert (error.code, error.detail, error.retryable, error.applied, error.hint) == fields
    assert isinstance(error, httpx.HTTPStatusError) and error.response is response
    assert str(error).startswith(f"{response.status_code} {fields[0] or 'error'}: ")


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
    with _serve(httpx.Response(200, json={"virtual_time": "2026-01-01T00:00:00Z"})) as requests:
        await protocol_v1.invoke_extension(
            _BASE, _CARD_CLOCK, "urn:agentenv:clock/v1", {"time": "2026-01-01T00:00:00Z"}, method="set_time",
        )
    request, = requests
    assert (request.method, str(request.url)) == ("PUT", f"{_BASE}/clock/set-time")
    assert json.loads(request.content) == {"time": "2026-01-01T00:00:00Z"}


@pytest.mark.asyncio
async def test_invoke_extension_named_get_method_uses_query_params():
    with _serve(httpx.Response(200, json={"frozen": False})) as requests:
        assert await protocol_v1.invoke_extension(_BASE, _CARD_CLOCK, "urn:agentenv:clock/v1", method="state") == {"frozen": False}
    assert [(r.method, str(r.url)) for r in requests] == [("GET", f"{_BASE}/clock/state")]


@pytest.mark.asyncio
async def test_invoke_extension_without_a_method_calls_the_first_listed():
    with _serve(httpx.Response(200, json={})) as requests:
        await protocol_v1.invoke_extension(_BASE, _CARD_CLOCK, "urn:agentenv:clock/v1")
    assert [(r.method, str(r.url)) for r in requests] == [("GET", f"{_BASE}/clock/time")]


@pytest.mark.asyncio
async def test_invoke_extension_unknown_method_raises_without_calling():
    offered = "does not advertise method 'sync_time' (advertises: get_time, set_time, state)"
    with _serve(httpx.Response(200, json={})) as requests, pytest.raises(ValueError, match=re.escape(offered)):
        await protocol_v1.invoke_extension(_BASE, _CARD_CLOCK, "urn:agentenv:clock/v1", method="sync_time")
    assert not requests


@pytest.mark.asyncio
async def test_invoke_extension_named_method_on_a_methodless_extension_raises():
    with pytest.raises(ValueError, match=re.escape("does not advertise method 'set_time' (advertises: none)")):
        await protocol_v1.invoke_extension(_BASE, _CARD_WITH_EXT, "urn:agentenv:clock/v1", method="set_time")


@pytest.mark.asyncio
async def test_invoke_extension_missing_endpoint_raises_value_error():
    card = {"capabilities": {"extensions": [{"uri": "urn:agentenv:test/v1", "params": {}}]}}
    with pytest.raises(ValueError, match="has no endpoint in params"):
        await protocol_v1.invoke_extension(_BASE, card, "urn:agentenv:test/v1")
