"""Unit tests for DeployHumanAgentTaskStep (registers an external human A2A peer)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.config.errors import ConfigError
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.deploy_human_agent import DeployHumanAgentTaskStep


def _patch_httpx(*, card=None, exc=None):
    """Patch the step's httpx.AsyncClient so the card GET returns `card` (200) or raises `exc`."""
    client = MagicMock()
    if exc is not None:
        client.get = AsyncMock(side_effect=exc)
    else:
        client.get = AsyncMock(return_value=MagicMock(status_code=200, json=MagicMock(return_value=card)))
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return patch("agent_env.task_step.task_steps.deploy_human_agent.httpx.AsyncClient", MagicMock(return_value=ctx))


@pytest.mark.asyncio
async def test_execute_uses_fetched_card():
    step = DeployHumanAgentTaskStep(id="t.dh", version=None, agent_name="human", a2a_url="http://h:8765")
    with _patch_httpx(card={"name": "human_agent", "description": "real card"}):
        result = await step.execute(TaskStepContext())

    assert len(result.deployed_agents) == 1
    human = result.deployed_agents[0]
    assert human.agent_name == "human"
    assert human.api_url == human.a2a_url == "http://h:8765"
    assert human.a2a_card == {"name": "human_agent", "description": "real card"}


@pytest.mark.asyncio
async def test_execute_synthesizes_card_when_fetch_fails():
    step = DeployHumanAgentTaskStep(id="t.dh", version=None, agent_name="human", a2a_url="http://h:8765")
    with _patch_httpx(exc=RuntimeError("unreachable")):
        result = await step.execute(TaskStepContext())

    assert result.deployed_agents[0].a2a_card == {
        "name": "human",
        "description": "A human operator available to answer questions",
        "capabilities": {},
        "default_input_modes": ["text"],
        "default_output_modes": ["text"],
    }


@pytest.mark.asyncio
async def test_execute_default_constructs_instance_url():
    # No a2a_url -> build {default_human_a2a_url}/instance/<task_instance_id>.
    step = DeployHumanAgentTaskStep(id="t.dh", version=None, agent_name="human")
    cfg = MagicMock()
    cfg.get_default_human_a2a_url.return_value = "http://human"
    with patch("agent_env.config.get_config", return_value=cfg):
        with _patch_httpx(exc=RuntimeError("unreachable")):
            result = await step.execute(TaskStepContext(instance_id="ti-123"))

    human = result.deployed_agents[0]
    assert human.api_url == human.a2a_url == "http://human/instance/ti-123"
    assert result.metadata["human_agents"] == [
        {"step_id": "t.dh", "agent_name": "human", "a2a_url": "http://human/instance/ti-123"}
    ]


@pytest.mark.asyncio
async def test_execute_surfaces_the_unconfigured_error():
    step = DeployHumanAgentTaskStep(id="t.dh", version=None, agent_name="human")
    cfg = MagicMock()
    cfg.get_default_human_a2a_url.side_effect = ConfigError("No human-A2A URL configured")
    with patch("agent_env.config.get_config", return_value=cfg):
        with pytest.raises(ConfigError, match="No human-A2A URL configured"):
            await step.execute(TaskStepContext(instance_id="ti-123"))


@pytest.mark.asyncio
async def test_execute_explicit_a2a_url_is_verbatim():
    # An explicit a2a_url is used as-is (no instance path appended).
    step = DeployHumanAgentTaskStep(id="t.dh", version=None, agent_name="human", a2a_url="http://h:8765")
    with _patch_httpx(exc=RuntimeError("unreachable")):
        result = await step.execute(TaskStepContext(instance_id="ti-123"))

    assert result.deployed_agents[0].a2a_url == "http://h:8765"


@pytest.mark.asyncio
async def test_execute_rejects_duplicate_agent_name():
    context = TaskStepContext(deployed_agents=[DeployedAgent(agent_name="human", api_url="http://x")])
    step = DeployHumanAgentTaskStep(id="t.dh", version=None, agent_name="human")
    with pytest.raises(RuntimeError, match="already deployed"):
        await step.execute(context)


@pytest.mark.asyncio
async def test_execute_rejects_second_human_agent():
    context = TaskStepContext()
    with _patch_httpx(exc=RuntimeError("unreachable")):
        await DeployHumanAgentTaskStep(
            id="t.h1", version=None, agent_name="human1", a2a_url="http://h:8765",
        ).execute(context)
        with pytest.raises(RuntimeError, match="only one human agent"):
            await DeployHumanAgentTaskStep(
                id="t.h2", version=None, agent_name="human2", a2a_url="http://h:8765",
            ).execute(context)


def test_roundtrip_and_registry():
    step = DeployHumanAgentTaskStep(id="t.dh", version=3, agent_name="human", a2a_url="http://h:8765")
    restored = DeployHumanAgentTaskStep.from_dict(step.to_dict())
    assert (restored.id, restored.version, restored.agent_name, restored.a2a_url) == (
        "t.dh", 3, "human", "http://h:8765",
    )
    assert get_task_step_registry()["deploy_human_agent"] is DeployHumanAgentTaskStep
