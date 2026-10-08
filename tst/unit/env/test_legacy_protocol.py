"""Unit tests for the legacy MCP data-plane reset protocol."""

from __future__ import annotations

import copy
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from agentenv_protocol import RPC_PATH, WELL_KNOWN_PATH, DataPart, uploaded_file_part
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.client import GetDataResponse

from agent_env.env import legacy_protocol
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.env.gateway.gateway import Gateway


@pytest.mark.asyncio
async def test_reset_via_rest_posts_and_returns_json():
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json = MagicMock(return_value={"ok": True})
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)

    with patch("agent_env.env.legacy_protocol.httpx.AsyncClient", return_value=client):
        result = await legacy_protocol.reset_via_rest("http://gw/svc/mcp-email", "/data/seed.json")

    assert result == {"ok": True}
    resp.raise_for_status.assert_called_once()
    url, kwargs = client.post.call_args.args[0], client.post.call_args.kwargs
    assert url == "http://gw/svc/mcp-email/api/reset"
    assert kwargs["json"] == {"mock_data_path": "/data/seed.json"}
    assert kwargs["timeout"] == 30


@pytest.mark.asyncio
async def test_reset_uses_rest_on_success(monkeypatch):
    rest = AsyncMock(return_value={"ok": True})
    mcp = AsyncMock()
    monkeypatch.setattr(legacy_protocol, "reset_via_rest", rest)
    monkeypatch.setattr(legacy_protocol, "reset_via_mcp_tool", mcp)

    result = await legacy_protocol.reset("http://gw", "email", "/data/seed.json")

    assert result == {"ok": True}
    rest.assert_awaited_once_with("http://gw/svc/mcp-email", "/data/seed.json")
    mcp.assert_not_called()


@pytest.mark.asyncio
async def test_reset_falls_back_to_mcp_tool(monkeypatch):
    rest = AsyncMock(side_effect=RuntimeError("rest down"))
    mcp = AsyncMock()
    monkeypatch.setattr(legacy_protocol, "reset_via_rest", rest)
    monkeypatch.setattr(legacy_protocol, "reset_via_mcp_tool", mcp)

    result = await legacy_protocol.reset("http://gw", "email", "/data/seed.json", max_retries=2)

    assert result is None
    mcp.assert_awaited_once_with("http://gw/mcp", "email", "/data/seed.json", max_retries=2)


def test_service_base_url_mcp_vs_website():
    assert legacy_protocol.environment_base_url("http://gw", "email", mcp=True) == "http://gw/svc/mcp-email"
    assert legacy_protocol.environment_base_url("http://gw", "slack", mcp=False) == "http://gw/svc/slack"


def _fake_client(resp):
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.mark.asyncio
async def test_reset_via_rest_no_body_when_no_seed():
    resp = MagicMock(); resp.raise_for_status = MagicMock(); resp.json = MagicMock(return_value={"ok": True})
    client = _fake_client(resp)
    with patch("agent_env.env.legacy_protocol.httpx.AsyncClient", return_value=client) as ctor:
        await legacy_protocol.reset_via_rest("http://gw/svc/slack", timeout=60)

    ctor.assert_called_once_with(verify=True)
    assert client.post.call_args.args[0] == "http://gw/svc/slack/api/reset"
    assert client.post.call_args.kwargs["json"] is None
    assert client.post.call_args.kwargs["timeout"] == 60


@pytest.mark.asyncio
async def test_add_via_rest_posts_file_path():
    resp = MagicMock(); resp.raise_for_status = MagicMock(); resp.json = MagicMock(return_value={"ok": True})
    client = _fake_client(resp)
    with patch("agent_env.env.legacy_protocol.httpx.AsyncClient", return_value=client) as ctor:
        r = await legacy_protocol.add_via_rest("http://gw/svc/slack", "/tmp/data/seed.json")

    assert r == {"ok": True}
    ctor.assert_called_once_with(verify=True)
    assert client.post.call_args.args[0] == "http://gw/svc/slack/api/add"
    assert client.post.call_args.kwargs["json"] == {"file_path": "/tmp/data/seed.json"}


@pytest.mark.asyncio
async def test_export_state_gets_mcp_endpoint_and_returns_json():
    resp = MagicMock(); resp.raise_for_status = MagicMock(); resp.json = MagicMock(return_value={"emails": []})
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch("agent_env.env.legacy_protocol.httpx.AsyncClient", return_value=client):
        r = await legacy_protocol.export_state("http://gw", "email")

    assert r == {"emails": []}
    assert client.get.call_args.args[0] == "http://gw/svc/mcp-email/export-state"
    assert client.get.call_args.kwargs["timeout"] == 60


@pytest.mark.asyncio
async def test_a_gateway_path_without_a_gateway_fails_readably():
    from agent_env.env.env import DeployedSandboxEnv, EnvNeedsGateway

    record = DeployedSandboxEnv(env_id="e", env_version=1, sandbox_id="srv")
    message = "Reaching 'slack' here needs a gateway; this env was deployed without one"
    with pytest.raises(EnvNeedsGateway, match=message):
        await legacy_protocol.v1_base_url(record, None, "slack")
    with pytest.raises(EnvNeedsGateway, match=message):
        await legacy_protocol.child_env_card(record, None, "slack")
    with pytest.raises(EnvNeedsGateway, match=message):
        await legacy_protocol.export_state(None, "slack")


@pytest.mark.asyncio
async def test_a_stored_child_card_gets_the_childs_path_on_each_endpoint_the_child_serves_and_leaves_the_record_as_stored():
    record = _composed_record({"name": "slack", "url": RPC_PATH, "capabilities": {"extensions": [
        {"uri": "urn:agentenv:set-errors/v1", "params": {"endpoint": "/agentenv/ext/set_errors"}},
        {"uri": "urn:agentenv:clock/v1", "params": {"endpoint": "/ext/clock/sync-time", "methods": {
            "sync_time": {"method": "POST", "endpoint": "/agentenv/ext/sync_time"}, "state": {"method": "GET"}}}},
        {"uri": "urn:agentenv:disable-tool/v1", "params": {"endpoint": "/tools/disable"}},
        {"uri": "urn:agentenv:triggers/v1", "params": {"methods": {"remove": {"method": "POST", "endpoint": "/triggers/remove"}}}},
        {"uri": "urn:agentenv:set-acting-user/v1", "params": {"endpoint": "/svc/mcp-slackbot/ext/set_acting_user"}},
        {"uri": "urn:agentenv:export-as-file/v1", "params": {"endpoint": "/agentenv/ext/set_export_as_file"}},
    ]}})
    stored = copy.deepcopy(record.environment_card)

    base, card = await legacy_protocol.child_env_card(record, "http://gw", "slack")

    assert base == "http://gw"
    assert [e["params"] for e in card["capabilities"]["extensions"]] == [
        {"endpoint": "/svc/mcp-slack/agentenv/ext/set_errors"},
        {"endpoint": "/svc/mcp-slack/ext/clock/sync-time", "methods": {
            "sync_time": {"method": "POST", "endpoint": "/svc/mcp-slack/agentenv/ext/sync_time"}, "state": {"method": "GET"}}},
        {"endpoint": "/tools/disable"},
        {"methods": {"remove": {"method": "POST", "endpoint": "/triggers/remove"}}},
        {"endpoint": "/svc/mcp-slack/svc/mcp-slackbot/ext/set_acting_user"},
        {"endpoint": "/svc/mcp-slack/agentenv/ext/set_export_as_file"},
    ]
    assert record.environment_card == stored


@pytest.mark.asyncio
async def test_a_child_endpoint_its_composer_already_put_under_the_childs_path_is_left_as_it_is():
    child = {"name": "x", "url": "/envs/x/agentenv", "capabilities": {"extensions": [
        {"uri": "urn:agentenv:clock/v1", "params": {"endpoint": "/envs/x/ext/clock", "methods": {
            "sync_time": {"method": "POST", "endpoint": "/envs/x/ext/clock/sync"}}}},
    ]}}
    record = DeployedEnv(env_id="e", env_version=1, environment_card_url=f"http://plugin.example{WELL_KNOWN_PATH}",
                         environment_card={"name": "suite", "children_environments": [child]})

    assert await legacy_protocol.child_env_card(record, None, "x") == ("http://plugin.example", child)


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [RPC_PATH, "/prefix/agentenv"], ids=["rpc-path", "rpc-path-under-a-prefix"])
async def test_a_stored_leaf_card_resolves_against_the_envs_address_unchanged(url):
    card = {"name": "slack", "url": url, "capabilities": {"extensions": [
        {"uri": "urn:agentenv:set-errors/v1", "params": {"endpoint": "/agentenv/ext/set_errors"}},
        {"uri": "urn:agentenv:clock/v1", "params": {"endpoint": "/ext/clock/sync-time"}},
    ]}}
    record = DeployedEnv(env_id="e", env_version=1, environment_card_url=f"https://sandbox.example/sb-1{WELL_KNOWN_PATH}",
                         environment_card=card)

    assert await legacy_protocol.child_env_card(record, None, "slack") == ("https://sandbox.example/sb-1", card)


@pytest.mark.asyncio
@pytest.mark.parametrize("params, served_at", [
    ({"endpoint": "/agentenv/ext/set_errors"}, "/agentenv/ext/set_errors"),
    ({"endpoint": "/ext/clock/sync-time"}, "/ext/clock/sync-time"),
    ({"endpoint": "/agentenv/ext/clock", "methods": {"sync_time": {"endpoint": "/agentenv/ext/sync_time"}}}, "/agentenv/ext/sync_time"),
], ids=["under-the-rpc-path", "outside-the-rpc-path", "a-methods-own"])
async def test_a_child_env_endpoint_resolves_to_the_child_from_the_stored_card_as_from_a_live_read(monkeypatch, params, served_at):
    own_card = {"name": "slack", "url": RPC_PATH, "capabilities": {"extensions": [{"uri": "urn:agentenv:clock/v1", "params": params}]}}
    called = _child_behind_gateway(monkeypatch, own_card)

    for record in (_composed_record(own_card), None):
        base_url, card = await legacy_protocol.child_env_card(record, "http://gw", "slack")
        await protocol_v1.invoke_extension(base_url, card, "urn:agentenv:clock/v1")

    assert called == [f"http://gw/svc/mcp-slack{served_at}"] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("fields", [
    {},
    {"capabilities": None},
    {"capabilities": {"extensions": None}},
    {"capabilities": {"extensions": [{"uri": "urn:agentenv:clock/v1"}]}},
    {"capabilities": {"extensions": [{"uri": "urn:agentenv:clock/v1", "params": None}]}},
    {"capabilities": {"extensions": [{"uri": "urn:agentenv:clock/v1", "params": "/ext/clock/sync-time"}]}},
], ids=["no-capabilities", "null-capabilities", "null-extensions", "no-params", "null-params", "params-not-an-object"])
async def test_a_stored_child_card_without_endpoints_comes_back_as_stored(fields):
    record = _composed_record({"name": "slack", "url": RPC_PATH, **fields})
    (stored,) = record.environment_card["children_environments"]

    assert await legacy_protocol.child_env_card(record, "http://gw", "slack") == ("http://gw", stored)


def _composed_record(own_card: dict) -> DeployedGatewayEnv:
    """The record of a deploy at http://gw whose stored card composes `own_card` as the gateway does, under mcp-<name>,
    beside the gateway's own extensions."""
    gateway = Gateway(host="127.0.0.1", port=0, server_name="gw", internal_mcp_servers=[], rest_proxy_urls={})
    child = gateway._rewrite_child_card(f"mcp-{own_card['name']}", own_card)
    card = {"name": "gw", "url": RPC_PATH, "capabilities": {"extensions": gateway._gateway_extensions()}, "children_environments": [child]}
    return DeployedGatewayEnv(env_id="e", env_version=1, gateway_url="http://gw", sandbox_id="sb-1",
                              environment_card_url=f"http://gw{WELL_KNOWN_PATH}", environment_card=card)


def _child_behind_gateway(monkeypatch, own_card: dict) -> list[str]:
    """http://gw proxying /svc/mcp-slack to a child that serves `own_card` and answers anything else with {};
    returns the URL of each request other than a card read."""
    called, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/svc/mcp-slack{WELL_KNOWN_PATH}":
            return httpx.Response(200, json=own_card)
        called.append(str(request.url))
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return called


def _v1_service(monkeypatch, answer: list, export_state: dict) -> list[str]:
    """A v1 service at http://gw/svc/mcp-slack that answers data/get with ``answer`` and serves ``export_state``;
    returns each request it got, as ``METHOD path``."""
    asked, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        asked.append(f"{request.method} {request.url.path}")
        if request.url.path.endswith(WELL_KNOWN_PATH):
            return httpx.Response(200, json={"name": "slack", "url": RPC_PATH})
        if request.url.path.endswith("/export-state"):
            return httpx.Response(200, json=export_state)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": GetDataResponse(parts=answer).model_dump(mode="json")})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return asked


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, state, reads_export_state", [
    ([DataPart(data={"messages": 2})], {"messages": 2}, False),
    ([uploaded_file_part("slack.zip", name="slack.zip", mime_type="application/zip")], {"messages": 3}, True),
    ([], {}, False),
], ids=["data", "file-bundle", "nothing"])
async def test_a_v1_services_state_is_the_data_it_answers_with_else_its_export_state(monkeypatch, answer, state, reads_export_state):
    asked = _v1_service(monkeypatch, answer, export_state={"messages": 3})

    assert await legacy_protocol.service_state(None, "http://gw", "slack") == state
    assert ("GET /svc/mcp-slack/export-state" in asked) is reads_export_state
