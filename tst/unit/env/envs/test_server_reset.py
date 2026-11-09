"""An MCP server's or website's reset empties what loads put in, as each load does before it adds: through data/reset, or
an older server's /api/reset, its changelog started afresh after; one serving no reset takes no load and is left as it
is. Alone, each attaches to its record as a load does."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

from agent_env.env.env import DeployedEnv
from agent_env.env.envs import _deployment, mcp_server, website
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.gateway.constants import DATA_PLANE_LOAD_TIMEOUT_S
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider

GATEWAY = "https://gw.example"


@pytest.fixture
def server(monkeypatch):
    """Each request a reset sends, in ``requests``, answered ``status``; and the changelog triggers installed after it."""
    seen = SimpleNamespace(requests=[], status=200, triggers=AsyncMock())
    real = httpx.AsyncClient

    def handle(request):
        seen.requests.append(request)
        return httpx.Response(seen.status, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    monkeypatch.setattr(EnvironmentGatewayProvider, "install_changelog_triggers", seen.triggers)
    return seen


def _record(card: dict | None) -> DeployedEnv:
    return DeployedEnv(env_id="mcp-email", env_version=1, env_provider_type="gateway", instance_id="inst-1",
                       environment_card_url=f"{GATEWAY}/.well-known/agent-env.json", environment_card=card)


def _server(card: dict | None, env_provider_type: str = "gateway") -> MCPServerEnv:
    """The email server behind a gateway of ours, attached to a record keeping ``card``."""
    env = MCPServerEnv(id="mcp-email", version=1, docker_image_artifact=MagicMock(), environment_name="email",
                       env_provider_type=env_provider_type)
    env._deployed, env._gateway_url = _record(card), GATEWAY
    return env


def _card(*operations: str) -> dict:
    return {"name": "email", "url": "/agentenv", "capabilities": {"operations": list(operations)}}


@pytest.mark.asyncio
@pytest.mark.parametrize("card", [_card("data/reset", "data/add", "data/get"), {"name": "email", "url": "/agentenv"}],
                         ids=["declared", "an-older-card-declaring-none"])
async def test_a_server_serving_data_reset_is_emptied_through_it_and_its_changelog_starts_afresh(server, card):
    assert await _deployment.reset_server_data(_server(card), mcp=True) is True

    [request] = server.requests
    assert (str(request.url), json.loads(request.content)["method"]) == (f"{GATEWAY}/agentenv", "data/reset")
    assert request.extensions["timeout"]["read"] == DATA_PLANE_LOAD_TIMEOUT_S  # a load's budget, not the client's 30 s
    server.triggers.assert_awaited_once_with("email")


@pytest.mark.asyncio
async def test_a_server_whose_card_declares_no_data_reset_takes_no_load_and_is_left_as_it_is(server):
    assert await _deployment.reset_server_data(_server(_card()), mcp=True) is False
    assert server.requests == [] and not server.triggers.await_count


@pytest.mark.asyncio
@pytest.mark.parametrize("status, reset", [(200, True), (404, False)], ids=["serves-it", "serves-none"])
async def test_an_older_server_is_emptied_through_its_api_reset_and_one_answering_404_has_none(server, status, reset):
    server.status = status
    with patch("agent_env.env.legacy_protocol.v1_base_url", AsyncMock(return_value=None)), \
         patch("agent_env.env.legacy_protocol.offers_tool", AsyncMock(return_value=False)) as offers:
        assert await _deployment.reset_server_data(_server(None), mcp=True) is reset

    [request] = server.requests
    assert (request.method, str(request.url)) == ("POST", f"{GATEWAY}/svc/mcp-email/api/reset")
    assert server.triggers.await_count == reset
    assert [call.args for call in offers.await_args_list] == ([] if reset else [(f"{GATEWAY}/mcp", "email_reset")])


@pytest.mark.asyncio
async def test_an_older_server_answering_404_resets_through_its_tool_with_no_seed_to_the_data_it_boots_with(server):
    server.status = 404
    tool = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="Data reset for email.")]))
    with patch("agent_env.env.legacy_protocol.v1_base_url", AsyncMock(return_value=None)), \
         patch("agent_env.env.legacy_protocol.offers_tool", AsyncMock(return_value=True)), \
         patch("agent_env.env.legacy_protocol.reset_via_mcp_tool", tool):
        assert await _deployment.reset_server_data(_server(None), mcp=True) is True

    tool.assert_awaited_once_with(f"{GATEWAY}/mcp", "email")  # no seed
    server.triggers.assert_awaited_once_with("email")


@pytest.mark.asyncio
async def test_an_older_server_whose_reset_tool_fails_fails_its_reset_named(server):
    server.status = 404
    failed = CallToolResult(content=[TextContent(type="text", text="database is locked")], isError=True)
    with patch("agent_env.env.legacy_protocol.v1_base_url", AsyncMock(return_value=None)), \
         patch("agent_env.env.legacy_protocol.offers_tool", AsyncMock(return_value=True)), \
         patch("agent_env.env.legacy_protocol.reset_via_mcp_tool", AsyncMock(return_value=failed)), \
         pytest.raises(RuntimeError, match="^MCP server 'email': its email_reset tool failed: database is locked$"):
        await _deployment.reset_server_data(_server(None), mcp=True)
    assert not server.triggers.await_count


@pytest.mark.asyncio
async def test_an_older_server_failing_its_reset_fails_it(server):
    server.status = 500
    with patch("agent_env.env.legacy_protocol.v1_base_url", AsyncMock(return_value=None)), \
         pytest.raises(httpx.HTTPStatusError):
        await _deployment.reset_server_data(_server(None), mcp=True)
    assert not server.triggers.await_count


@pytest.mark.asyncio
async def test_a_server_a_plugin_deployed_with_no_data_plane_is_left_as_it_is(server):
    env = _server(None, env_provider_type="gateway")
    env._env_provider = MagicMock()  # a plugin's provider, no built-in
    with patch("agent_env.env.legacy_protocol.v1_base_url", AsyncMock(return_value=None)):
        assert await _deployment.reset_server_data(env, mcp=True) is False
    assert server.requests == []


@pytest.mark.asyncio
async def test_a_website_is_emptied_at_its_own_path(server):
    env = WebsiteEnv(id="shop-site", version=1, server_docker_image_artifact=MagicMock(), environment_name="shop")
    env._deployed, env._gateway_url = None, GATEWAY  # a name an MCP server shares: the live probe, no record
    with patch("agent_env.env.legacy_protocol.v1_base_url", AsyncMock(return_value=f"{GATEWAY}/svc/shop")) as base:
        assert await _deployment.reset_server_data(env, mcp=False) is True

    assert base.await_args.kwargs == {"mcp": False}
    assert [str(request.url) for request in server.requests] == [f"{GATEWAY}/svc/shop/agentenv"]
    server.triggers.assert_awaited_once_with("shop")


@pytest.mark.asyncio
async def test_a_server_not_deployed_cannot_reset(server):
    env = _server(None)
    env._deployed = env._gateway_url = None
    with pytest.raises(RuntimeError, match="not deployed"):
        await _deployment.reset_server_data(env, mcp=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls, module, mcp, what", [
    (MCPServerEnv, mcp_server, True, "MCP server 'email'"),
    (WebsiteEnv, website, False, "website 'email'"),
], ids=["mcp-server", "website"])
async def test_alone_each_attaches_to_its_record_as_a_load_does_and_one_serving_no_reset_has_none(
        monkeypatch, cls, module, mcp, what):
    record, attached, calls, served = _record(None), SimpleNamespace(environment_name="email"), [], [True, False]

    async def attach(_cls, deployed):
        calls.append(("attach", deployed))
        return attached

    async def reset(env, *, mcp):
        calls.append(("reset", env, mcp))
        return served.pop(0)

    monkeypatch.setattr(cls, "from_deployed_env", classmethod(attach))
    monkeypatch.setattr(module, "reset_server_data", reset)
    env = cls("email-env", 1, MagicMock(), environment_name="email")

    await env.reset(record)
    with pytest.raises(NotImplementedError, match=f"^{what} serves no reset$"):
        await env.reset(record)

    assert calls == [("attach", record), ("reset", attached, mcp)] * 2
