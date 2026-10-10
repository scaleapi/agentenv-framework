"""deploy_agent relays the env card's name as the MCP server alias to agents that accept one."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.env.env import DeployedEnv, DeployedGatewayEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep, _mcp_add_body

_LEGACY_CARD_EXT = {"params": {"endpoint": "/ext/mcp-config", "methods": {"add": {"method": "POST", "request": {"required": ["url"]}}}}}
_NAMING_CARD_EXT = {"params": {"endpoint": "/ext/mcp-config", "methods": {"add": {"method": "POST", "request": {"required": ["url"], "optional": ["name"]}}}}}


def test_agents_that_do_not_advertise_name_get_the_legacy_body():
    assert _mcp_add_body(_LEGACY_CARD_EXT, "https://gw/mcp", None, "env") == {"url": "https://gw/mcp"}
    assert _mcp_add_body(_LEGACY_CARD_EXT, "https://gw/mcp", {"Authorization": "Bearer t"}, "env") == {
        "url": "https://gw/mcp", "headers": {"Authorization": "Bearer t"},
    }


def test_the_card_name_is_relayed_verbatim_whatever_it_is():
    for name in ("env", "crm", "AgentEnvGateway"):  # default, declared, and an env on an older gateway image
        assert _mcp_add_body(_NAMING_CARD_EXT, "https://gw/mcp", None, name) == {"url": "https://gw/mcp", "name": name}


def test_a_hand_written_supported_list_also_takes_the_name():
    ext = {"params": {"methods": {"add": {"request": {"supported": ["url", "headers", "name"]}}}}}
    assert _mcp_add_body(ext, "https://gw/mcp", None, "crm") == {"url": "https://gw/mcp", "name": "crm"}


def test_no_card_name_means_the_agent_mints_its_own():
    assert _mcp_add_body(_NAMING_CARD_EXT, "https://gw/mcp", None, None) == {"url": "https://gw/mcp"}
    assert "name" not in _mcp_add_body({"params": {"methods": {"add": {"request": {"optional": None}}}}}, "https://gw/mcp", None, "env")


@pytest.mark.asyncio
async def test_the_stored_card_name_is_relayed_without_reading_the_card(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        requests = await _deploy_agent_against(monkeypatch, _deployed_env({"name": "env5647", "protocolVersion": "1"}))

    assert [(r.method, r.url.path) for r in requests] == [("POST", "/ext/mcp-config")]
    assert json.loads(requests[0].content) == {"url": _MCP_URL, "name": "env5647"}
    assert not any("no env card name" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_a_record_without_a_stored_card_relays_no_name_and_warns(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        requests = await _deploy_agent_against(monkeypatch, _deployed_env(None))

    assert [(r.method, r.url.path) for r in requests] == [("POST", "/ext/mcp-config")]
    assert json.loads(requests[0].content) == {"url": _MCP_URL}
    assert "Env crm has no env card name; the agent will pick its own MCP alias" in caplog.messages


_MCP_URL = "https://gw.example/mcp"
_LOGGER = "agent_env.task_step.task_steps.deploy_agent"


def _deployed_env(card: dict | None) -> DeployedEnv:
    return DeployedGatewayEnv(
        env_id="crm", env_version=1, gateway_url="https://gw.example", mcp_url=_MCP_URL, db_web_url=None,
        sandbox_id="sb-1", environment_card_url="https://gw.example/.well-known/agent-env.json", environment_card=card,
    )


async def _deploy_agent_against(monkeypatch, deployed_env: DeployedEnv) -> list[httpx.Request]:
    """Every request deploy_agent sends while wiring an agent that accepts `name` to `deployed_env`."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    agent_card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_MCP_CONFIG, **_NAMING_CARD_EXT}]}}
    agent = MagicMock(metadata={})
    agent.deploy = AsyncMock(return_value=MagicMock(a2a_url="https://agent.example", agent_card=agent_card, sandbox_type=None))
    context = TaskStepContext(deployed_envs=[deployed_env])
    with patch.object(A2AAgent, "get", return_value=agent), patch("agent_env.task_step.task_steps.deploy_agent.get_config") as get_config:
        get_config.return_value.get_model_for_role.return_value = None
        await DeployAgentTaskStep(id="ag", version=None, env_ids=["crm"]).execute(context)
    return requests
