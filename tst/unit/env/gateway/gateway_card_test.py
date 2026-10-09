"""Unit tests for the gateway composed environment card + env-level data plane."""
from __future__ import annotations

import json

import agentenv_protocol
import jsonschema
import pytest

from agentenv_protocol import EnvironmentCard, client as protocol_client
from agent_env.env.gateway import constants as gateway_constants
from agent_env.env.gateway.gateway import Gateway


def _gateway(rest_proxy_urls: dict[str, str]) -> Gateway:
    return Gateway(
        host="127.0.0.1",
        port=0,
        server_name="AgentEnvGateway",
        internal_mcp_servers=[],
        rest_proxy_urls=rest_proxy_urls,
    )


_SLACK_CARD = {
    "name": "slack",
    "protocolVersion": "1.0",
    "url": "/agentenv",
    "preferredTransport": "JSONRPC",
    "additionalInterfaces": [],
    "capabilities": {
        "extensions": [
            {
                "uri": "urn:agentenv:set-errors/v1",
                "description": None,
                "params": {"endpoint": "/agentenv/ext/set_errors",
                           "methods": {"set_errors": {"method": "POST", "request": {}}}},
                "required": None,
            },
            {
                "uri": "urn:agentenv:disable-tool/v1",
                "description": None,
                "params": {"endpoint": "/tools/disable",
                           "methods": {"disable": {"method": "POST", "request": {}}}},
                "required": None,
            },
        ],
        "tools": [
            {
                "name": "slack_send_message",
                "description": "Send a message to a channel.",
                "inputSchema": {"type": "object", "properties": {"channel": {"type": "string"}}, "required": ["channel"]},
            },
        ],
    },
}


class _FakeReq:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def json(self) -> dict:
        return self._payload


def test_rewrite_child_card_rewrites_only_server_paths():
    gw = _gateway({"mcp-slack": "http://slack:18765"})
    rewritten = gw._rewrite_child_card("mcp-slack", _SLACK_CARD)
    assert rewritten["url"] == "/svc/mcp-slack/agentenv"
    endpoints = {e["uri"]: e["params"]["endpoint"] for e in rewritten["capabilities"]["extensions"]}
    assert endpoints["urn:agentenv:set-errors/v1"] == "/svc/mcp-slack/agentenv/ext/set_errors"
    assert endpoints["urn:agentenv:disable-tool/v1"] == "/tools/disable"
    assert rewritten["capabilities"]["tools"] == _SLACK_CARD["capabilities"]["tools"]  # tools carry no endpoints; untouched
    assert _SLACK_CARD["url"] == "/agentenv"  # original untouched


def test_rewrite_child_card_drops_nested_children():
    gw = _gateway({"mcp-slack": "http://slack:18765"})
    composite = {"name": "slack", "url": "/agentenv", "capabilities": {"extensions": []},
                 "children_environments": [{"name": "inner", "url": "/agentenv"}]}
    assert gw._rewrite_child_card("mcp-slack", composite)["children_environments"] is None


_CLOCK_SYNC = {"uri": "urn:agentenv:clock/v1",
               "params": {"endpoint": "/ext/clock/sync-time", "methods": {"sync_time": {"method": "POST"}}}}


def test_rewrite_child_card_prefixes_every_child_path_but_the_gateways_routes():
    """A path outside `/agentenv`, or a method's own, is the child's too; a path stays on the gateway only for
    an operation the gateway's extension with the same uri offers."""
    child = {"name": "slack", "url": "/agentenv", "capabilities": {"extensions": [
        _CLOCK_SYNC,
        {"uri": "urn:example:export/v1",
         "params": {"endpoint": "/agentenv/ext/export", "methods": {"start": {"method": "POST", "endpoint": "/export/start"}}}},
        {"uri": "urn:example:state/v1", "params": {"endpoint": "/state"}},
        {"uri": "urn:agentenv:enable-tool/v1", "params": {"endpoint": "/tools/enable"}},
        {"uri": "urn:example:remote/v1", "params": {"endpoint": "https://example.com/hook"}},
    ]}}

    rewritten = _gateway({"mcp-slack": "http://slack:18765"})._rewrite_child_card("mcp-slack", child)

    clock, export, state, enable, remote = rewritten["capabilities"]["extensions"]
    assert clock["params"]["endpoint"] == "/svc/mcp-slack/ext/clock/sync-time"
    assert export["params"]["endpoint"] == "/svc/mcp-slack/agentenv/ext/export"
    assert export["params"]["methods"]["start"]["endpoint"] == "/svc/mcp-slack/export/start"
    assert state["params"]["endpoint"] == "/svc/mcp-slack/state"  # the gateway's /state belongs to state/v1
    assert enable["params"]["endpoint"] == "/tools/enable"
    assert remote["params"]["endpoint"] == "https://example.com/hook"


def test_a_child_operation_on_a_gateway_path_with_another_verb_is_the_childs():
    """The gateway's `/clock/time` is a GET, so a child's POST there is the child's. When a gateway operation
    falls back to the same extension endpoint, the endpoint stays and the child's method gets its own."""
    child = {"name": "slack", "url": "/agentenv", "capabilities": {"extensions": [
        {"uri": "urn:agentenv:clock/v1", "params": {"endpoint": "/clock/time", "methods": {"sync_time": {"method": "POST"}}}},
        {"uri": "urn:agentenv:clock/v1", "params": {"endpoint": "/clock/time", "methods": {
            "get_time": {"method": "GET"}, "sync_time": {"method": "POST"}}}},
    ]}}

    rewritten = _gateway({"mcp-slack": "http://slack:18765"})._rewrite_child_card("mcp-slack", child)

    child_only, shared = rewritten["capabilities"]["extensions"]
    assert child_only["params"]["endpoint"] == "/svc/mcp-slack/clock/time"
    assert shared["params"]["endpoint"] == "/clock/time"
    assert "endpoint" not in shared["params"]["methods"]["get_time"]
    assert shared["params"]["methods"]["sync_time"]["endpoint"] == "/svc/mcp-slack/clock/time"


def test_a_child_method_named_otherwise_on_a_gateway_path_is_the_childs():
    """Same uri, verb and path as the gateway's `disable` but another name: a client picks it by name, so the
    child serves it."""
    child = {"name": "slack", "url": "/agentenv", "capabilities": {"extensions": [
        {"uri": "urn:agentenv:disable-tool/v1",
         "params": {"endpoint": "/tools/disable", "methods": {"mute": {"method": "POST"}}}},
    ]}}

    (ext,) = _gateway({"mcp-slack": "http://slack:18765"})._rewrite_child_card("mcp-slack", child)["capabilities"]["extensions"]

    assert ext["params"]["endpoint"] == "/svc/mcp-slack/tools/disable"


def test_a_slash_ended_child_url_comes_back_without_its_slash():
    """`v1_base_url` strips `/agentenv` off the child's url to reach its data plane, so `/agentenv/` must not
    become `/svc/<key>/agentenv/`."""
    child = {**_SLACK_CARD, "url": "/agentenv/"}
    assert _gateway({"mcp-slack": "http://slack:18765"})._rewrite_child_card("mcp-slack", child)["url"] == "/svc/mcp-slack/agentenv"


@pytest.mark.asyncio
async def test_a_childs_clock_sync_resolves_to_the_child_through_the_protocol_client(monkeypatch):
    """`find_child` promises child endpoints reachable from the env card's address, as `sync_env_clock` reads them."""
    gw = _gateway({"mcp-slack": "http://slack:18765"})

    async def fake_fetch(client, key, base_url):
        return gw._rewrite_child_card(key, {**_SLACK_CARD, "capabilities": {"extensions": [_CLOCK_SYNC]}})

    monkeypatch.setattr(gw, "_fetch_child_card", fake_fetch)
    card = json.loads((await gw._serve_env_card()).body)

    child = protocol_client.find_child(card, "slack")
    sync = protocol_client.find_extension_method(child, "urn:agentenv:clock/v1", "sync_time")
    assert (sync["method"], sync["endpoint"]) == ("POST", "/svc/mcp-slack/ext/clock/sync-time")


@pytest.mark.asyncio
async def test_serve_env_card_composes_children_and_omits_legacy(monkeypatch):
    gw = _gateway({"mcp-slack": "http://slack:18765", "mcp-legacy": "http://legacy:18765"})

    async def fake_fetch(client, key, base_url):
        return gw._rewrite_child_card(key, _SLACK_CARD) if key == "mcp-slack" else None

    monkeypatch.setattr(gw, "_fetch_child_card", fake_fetch)
    card = json.loads((await gw._serve_env_card()).body)

    assert card["name"] == "AgentEnvGateway"
    assert card["url"] == "/agentenv"
    uris = {e["uri"] for e in card["capabilities"]["extensions"]}
    assert {"urn:agentenv:disable-tool/v1", "urn:agentenv:enable-tool/v1"} <= uris
    assert [c["name"] for c in card["children_environments"]] == ["slack"]  # legacy omitted
    child_ext = card["children_environments"][0]["capabilities"]["extensions"][0]
    assert child_ext["params"]["endpoint"] == "/svc/mcp-slack/agentenv/ext/set_errors"
    child_tools = card["children_environments"][0]["capabilities"]["tools"]
    assert [t["name"] for t in child_tools] == ["slack_send_message"]  # child tools survive composition

    # drift guard: the hand-rolled shape validates against the canonical schema and keeps children
    validated = EnvironmentCard.model_validate(card)
    assert validated.children_environments[0].name == "slack"
    assert validated.children_environments[0].capabilities.tools[0].name == "slack_send_message"


@pytest.mark.asyncio
async def test_serve_env_card_declares_the_gateways_mcp_endpoint():
    card = json.loads((await _gateway({})._serve_env_card()).body)
    assert card["additionalInterfaces"] == [{"url": "/mcp", "transport": "mcp"}]
    assert protocol_client.mcp_path(card) == "/mcp"


def test_rewrite_child_card_drops_the_childs_interfaces():
    child = {**_SLACK_CARD, "additionalInterfaces": [{"url": "/mcp", "transport": "mcp"}]}
    assert _gateway({"mcp-slack": "http://slack:18765"})._rewrite_child_card("mcp-slack", child)["additionalInterfaces"] == []


def test_gateway_card_lists_its_seven_extensions():
    assert _gateway({})._gateway_extensions() is gateway_constants.GATEWAY_EXTENSIONS
    assert [e["uri"] for e in _gateway({})._gateway_extensions()] == [
        "urn:agentenv:disable-tool/v1", "urn:agentenv:enable-tool/v1", "urn:agentenv:triggers/v1",
        "urn:agentenv:clock/v1", "urn:agentenv:trajectory/v1", "urn:agentenv:step/v1", "urn:agentenv:state/v1",
    ]


def test_step_and_state_resolve_through_the_protocol_client():
    card = {"capabilities": {"extensions": _gateway({})._gateway_extensions()}}
    step = protocol_client.find_extension_method(card, "urn:agentenv:step/v1", "step")
    state = protocol_client.find_extension_method(card, "urn:agentenv:state/v1", "get")
    assert (step["method"], step["endpoint"]) == ("POST", "/step")
    assert step["request"]["properties"]["action"]["enum"] == ["list_tools", "call_tool"]
    assert (state["method"], state["endpoint"]) == ("GET", "/state")


@pytest.mark.parametrize("body,valid", [
    ({"action": "list_tools"}, True),
    ({"action": "call_tool", "tool_name": "slack_send_message", "arguments": {"channel": "general"}}, True),
    ({"action": "call_tool", "tool_name": "slack_send_message"}, True),
    ({"action": "call_tool"}, False),
    ({"action": "call_tool", "tool_name": ""}, False),
    ({"action": "call_tool", "tool_name": "slack_send_message", "arguments": ["general"]}, False),
    ({"action": "delete"}, False),
    ({}, False),
])
def test_step_request_schema_matches_what_step_accepts(body, valid):
    """The advertised /step schema admits exactly the bodies _handle_step accepts (it 400s the rest)."""
    card = {"capabilities": {"extensions": _gateway({})._gateway_extensions()}}
    schema = protocol_client.find_extension_method(card, "urn:agentenv:step/v1", "step")["request"]
    assert jsonschema.Draft202012Validator(schema).is_valid(body) is valid


def test_every_advertised_endpoint_is_a_served_route():
    """Each (endpoint, verb) the card advertises is a route the gateway serves, resolved the way consumers resolve it."""
    gw = _gateway({})
    served = {(route.path, verb) for route in gw._mcp.streamable_http_app().routes for verb in getattr(route, "methods", None) or ()}
    card = {"capabilities": {"extensions": gw._gateway_extensions()}}
    advertised = set()
    for ext in card["capabilities"]["extensions"]:
        for name in ext["params"]["methods"]:
            method = protocol_client.find_extension_method(card, ext["uri"], name)
            advertised.add((method["endpoint"], method["method"]))
    assert advertised <= served, advertised - served


@pytest.mark.parametrize("name", ["WELL_KNOWN_PATH", "RPC_PATH", "PROTOCOL_VERSION", "METHOD_RESET", "METHOD_ADD", "METHOD_GET", "MCP_TRANSPORT"])
def test_gateway_constants_mirror_the_protocol(name):
    """constants.py copies these by hand because the gateway image has no agentenv_protocol dependency."""
    assert getattr(gateway_constants, name) == getattr(agentenv_protocol, name)


@pytest.mark.asyncio
async def test_serve_env_card_composes_website_backend(monkeypatch):
    gw = _gateway({"mcp-items": "http://items:18765", "webitems": "http://webitems-backend:8000"})

    async def fake_fetch(client, key, base_url):
        return gw._rewrite_child_card(key, {"name": key, "url": "/agentenv", "capabilities": {"extensions": []}})

    monkeypatch.setattr(gw, "_fetch_child_card", fake_fetch)
    card = json.loads((await gw._serve_env_card()).body)
    assert {c["url"] for c in card["children_environments"]} == {"/svc/mcp-items/agentenv", "/svc/webitems/agentenv"}


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["data/reset", "data/add", "data/get"])
async def test_env_data_plane_requires_single_child(method):
    gw = _gateway({"mcp-slack": "http://slack:18765", "mcp-linear": "http://linear:18765"})
    body = json.loads((await gw._serve_env_data_plane(
        _FakeReq({"jsonrpc": "2.0", "id": 1, "method": method}))).body)
    assert body["error"]["code"] == -32000
    assert "exactly one" in body["error"]["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["data/reset", "data/add", "data/get"])
async def test_env_data_plane_forwards_to_child(monkeypatch, method):
    gw = _gateway({"mcp-slack": "http://slack:18765"})
    captured: dict = {}

    class _Resp:
        status_code = 200
        content = b'{"jsonrpc":"2.0","id":1,"result":{}}'
        headers = {"content-type": "application/json"}

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            return _Resp()

    monkeypatch.setattr("agent_env.env.gateway.gateway.httpx.AsyncClient", _Client)
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": {}}
    resp = await gw._serve_env_data_plane(_FakeReq(payload))
    assert resp.status_code == 200
    assert captured["url"] == "http://slack:18765/agentenv"
    assert captured["json"] == payload


@pytest.mark.asyncio
async def test_env_data_plane_unknown_method():
    gw = _gateway({"mcp-slack": "http://slack:18765"})
    body = json.loads((await gw._serve_env_data_plane(
        _FakeReq({"jsonrpc": "2.0", "id": 1, "method": "data/bogus"}))).body)
    assert body["error"]["code"] == -32601


def test_triggers_advertised_as_one_extension():
    """Triggers ride a single urn:agentenv:triggers/v1 (register/remove/clear/state) — no separate get URI."""
    exts = _gateway({})._gateway_extensions()
    trigger = [e for e in exts if e["uri"] == "urn:agentenv:triggers/v1"]
    assert len(trigger) == 1
    assert set(trigger[0]["params"]["methods"]) == {"register", "remove", "clear", "state"}
    assert not any(e["uri"] == "urn:agentenv:get-triggers/v1" for e in exts)
