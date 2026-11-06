"""Unit tests for the legacy MCP data-plane reset protocol."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from agentenv_protocol import RPC_PATH, WELL_KNOWN_PATH, DataPart, uploaded_file_part
from agentenv_protocol.client import GetDataResponse

from agent_env.env import legacy_protocol
from agent_env.env.env import DeployedSandboxEnv


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
async def test_a_childs_state_is_read_with_its_deployments_control_token(monkeypatch):
    seen, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("authorization")))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"parts": [
            {"kind": "data", "data": {"alarms": []}}]}})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    deployed = DeployedSandboxEnv(
        env_id="desk", env_version=1, sandbox_id="sb-1", environment_card_url=f"http://desk:18765{WELL_KNOWN_PATH}",
        environment_card={"name": "desk", "capabilities": {"operations": ["data/get"]},
                          "children_environments": [{"name": "clock", "url": "/svc/clock/agentenv"}]},
        control_token="t0ken")

    assert await legacy_protocol.service_state(deployed, None, "clock") == {"alarms": []}
    assert seen == [("/svc/clock/agentenv", "Bearer t0ken")]


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


def test_definition_servers_are_an_envs_mcp_servers_then_its_websites():
    from types import SimpleNamespace

    from agent_env.env.envs.website import WebsiteEnv
    from agent_env.env.legacy_protocol import BackingServer

    named = lambda *names: [SimpleNamespace(environment_name=n) for n in names]
    multi = SimpleNamespace(mcp_server_envs=named("slack", "email"), website_envs=named("shop", "slack"))

    assert legacy_protocol.definition_servers(multi) == [BackingServer("slack"), BackingServer("email"),
                                                         BackingServer("shop", mcp=False), BackingServer("slack", mcp=False)]
    assert legacy_protocol.definition_servers(WebsiteEnv(id="w", version=1, environment_name="shop")) == [BackingServer("shop", mcp=False)]
    assert legacy_protocol.definition_servers(SimpleNamespace(environment_name="notes")) == [BackingServer("notes")]
    assert legacy_protocol.definition_servers(SimpleNamespace()) == []


_COMPOSED = {"name": "gw", "url": RPC_PATH, "capabilities": {"extensions": []}, "children_environments": [
    {"name": "slack", "url": "/svc/mcp-slack/agentenv", "capabilities": {"extensions": []}},
    {"name": "slack", "url": "/svc/slack/agentenv", "capabilities": {"extensions": []}},
    {"name": "docs", "url": "/svc/mcp-docs/agentenv", "capabilities": {"extensions": [{"uri": "urn:agentenv:website/v1"}]}},
]}


def test_card_servers_are_the_cards_children_a_website_told_apart_by_its_route_or_its_website_extension():
    servers = legacy_protocol.card_servers(_COMPOSED, "https://env.example")

    assert [(s.environment_name, s.mcp, s.address) for s in servers] == [
        ("slack", True, "https://env.example"), ("slack", False, "https://env.example"), ("docs", False, "https://env.example")]
    assert [s.child.base_url(s.address) for s in servers] == [
        "https://env.example/svc/mcp-slack", "https://env.example/svc/slack", "https://env.example/svc/mcp-docs"]


def test_card_servers_of_a_leaf_card_is_the_env_itself_and_a_composed_card_serving_data_get_comes_first():
    leaf = {"name": "notes", "url": RPC_PATH}
    assert [(s.environment_name, s.card) for s in legacy_protocol.card_servers(leaf, "https://n")] == [("notes", leaf)]
    driver = {**_COMPOSED, "name": "desktop", "capabilities": {"operations": ["data/get"]}}
    assert [s.environment_name for s in legacy_protocol.card_servers(driver, "https://d")] == ["desktop", "slack", "slack", "docs"]


def _record(card: dict | None, card_url: str | None = "https://env.example/.well-known/agent-env.json"):
    from agent_env.env.env import DeployedGatewayEnv

    return DeployedGatewayEnv(env_id="e", env_version=1, gateway_url="http://gw", mcp_url="http://gw/mcp", db_web_url=None,
                              sandbox_id="sb", environment_card_url=card_url, environment_card=card)


@pytest.mark.asyncio
async def test_backing_servers_are_the_stored_cards_then_those_the_definition_names_serving_no_card():
    from types import SimpleNamespace

    env = SimpleNamespace(mcp_server_envs=[SimpleNamespace(environment_name=n) for n in ("slack", "legacy")],
                          website_envs=[SimpleNamespace(environment_name="slack")])

    servers = await legacy_protocol.backing_servers(_record(_COMPOSED), "http://gw", env)

    assert [(s.environment_name, s.mcp, s.card is not None, s.address) for s in servers] == [
        ("slack", True, True, "https://env.example"), ("slack", False, True, "https://env.example"),
        ("docs", False, True, "https://env.example"), ("legacy", True, False, "https://env.example")]
    assert await legacy_protocol.server_card(servers[-1], "http://gw") == ("https://env.example", None)
    assert await legacy_protocol.server_v1_base_url(servers[-1], "http://gw") is None


@pytest.mark.asyncio
async def test_backing_servers_read_the_card_live_without_a_stored_one_or_for_another_gateway(monkeypatch):
    from types import SimpleNamespace

    read = []

    async def get_card(base_url, **kw):
        read.append(base_url)
        return _COMPOSED

    monkeypatch.setattr(legacy_protocol.protocol_v1, "get_card", get_card)
    env = SimpleNamespace(environment_name="slack")
    for record, gateway in ((_record(None), "http://gw/"), (_record(_COMPOSED), "http://elsewhere")):
        servers = await legacy_protocol.backing_servers(record, gateway, env)
        assert [(s.environment_name, s.address) for s in servers] == [
            ("slack", gateway.rstrip("/")), ("slack", gateway.rstrip("/")), ("docs", gateway.rstrip("/"))]
    assert read == ["http://gw", "http://elsewhere"]


@pytest.mark.asyncio
async def test_a_record_whose_card_cant_be_read_falls_back_to_its_definition_logged_once(monkeypatch, caplog):
    from types import SimpleNamespace

    async def get_card(base_url, **kw):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(legacy_protocol.protocol_v1, "get_card", get_card)
    env = SimpleNamespace(id="e", mcp_server_envs=[SimpleNamespace(environment_name="slack")],
                          website_envs=[SimpleNamespace(environment_name="slack")])
    with caplog.at_level("INFO", logger="agent_env.env.legacy_protocol"):
        for record in (_record(None, card_url=None), _record(None), None):
            servers = await legacy_protocol.backing_servers(record, "http://gw", env)
            assert [(s.environment_name, s.mcp, s.address) for s in servers] == [("slack", True, None), ("slack", False, None)]

    assert [r.getMessage().split(",")[0] for r in caplog.records] == [
        "env 'e': its record has no card URL", "env 'e': its card at http://gw could not be read (ConnectError)",
        "env 'e': it has no record"]


@pytest.mark.asyncio
async def test_a_stored_cards_website_and_mcp_server_sharing_a_name_are_each_reached_by_kind():
    site = {"name": "mail", "url": "/svc/mail/agentenv",
            "capabilities": {"extensions": [{"uri": "urn:agentenv:website/v1", "params": {}}]}}
    server = {"name": "mail", "url": "/svc/mcp-mail/agentenv"}
    deployed = DeployedSandboxEnv(
        env_id="office", env_version=1, sandbox_id="sb-1", environment_card_url=f"http://gw:18765{WELL_KNOWN_PATH}",
        environment_card={"name": "office", "children_environments": [site, server]})

    assert await legacy_protocol.v1_base_url(deployed, None, "mail", mcp=False) == "http://gw:18765/svc/mail"
    assert await legacy_protocol.v1_base_url(deployed, None, "mail", mcp=True) == "http://gw:18765/svc/mcp-mail"
    assert (await legacy_protocol.child_env_card(deployed, None, "mail", mcp=False))[1] is site
    assert (await legacy_protocol.child_env_card(deployed, None, "mail", mcp=True))[1] is server
