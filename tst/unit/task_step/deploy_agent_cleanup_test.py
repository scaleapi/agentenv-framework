"""An agent that fails to be configured after its sandbox exists is closed, since a run's teardown only finds the
agents its context records, and deploy_agent records one only once it is configured."""

from types import SimpleNamespace

import pytest

import agent_env.a2a_agent as a2a_agent_package
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep


class _Agent:
    metadata: dict = {}

    def __init__(self) -> None:
        self.closed = 0

    async def deploy(self, **_):
        return SimpleNamespace(
            a2a_url="https://agent.example.test", agent_card={}, sandbox_id="sb-1", sandbox_type="fake",
            instance_id="i-1", network_policy=None,
        )

    async def close(self) -> None:
        self.closed += 1


@pytest.mark.asyncio
async def test_an_agent_whose_snapshot_load_fails_is_closed_and_not_recorded(monkeypatch):
    agent = _Agent()
    monkeypatch.setattr(a2a_agent_package.A2AAgent, "get", classmethod(lambda cls, *a, **k: agent))

    async def unloadable(self, *args, **kwargs):
        raise RuntimeError("staging an object on the agent failed: the agent could not be reached")

    monkeypatch.setattr(DeployAgentTaskStep, "_load_snapshot", unloadable)
    step = DeployAgentTaskStep(
        id="deploy", version=None, a2a_agent_id="a", agent_name="solver", agent_snapshot_files_artifact_id="u"
    )
    context = TaskStepContext()

    with pytest.raises(RuntimeError, match="could not be reached"):
        await step.execute(context)
    assert agent.closed == 1
    assert context.deployed_agents == []


@pytest.mark.asyncio
async def test_a_configured_agent_is_recorded_and_left_running(monkeypatch):
    agent = _Agent()
    monkeypatch.setattr(a2a_agent_package.A2AAgent, "get", classmethod(lambda cls, *a, **k: agent))
    step = DeployAgentTaskStep(id="deploy", version=None, a2a_agent_id="a", agent_name="solver")
    context = await step.execute(TaskStepContext())

    assert agent.closed == 0
    assert [a.sandbox_id for a in context.deployed_agents] == ["sb-1"]
