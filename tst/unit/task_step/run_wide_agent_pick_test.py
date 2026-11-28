"""A run-wide agent pick (``user_overrides.a2a_agent_id``: ``task run --a2a-agent-id``, a hub run's agent pick) reaches
every ``deploy_agent`` and ``install_agent`` step, a ``prompt_agent`` task's user agent included. A step's pinned
version belongs to the agent the step names, so a pick of another agent never takes it. Each case runs the step and
reads what it asks the agent store for. A ``deploy_user_sim_agent`` step keeps its own agent through its own
``_resolve_agent_ref``, not anything these steps read."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.config import ConfigError, get_config, reset_config
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep

SOLVER = {"id": "deploy-solver", "type": "deploy_agent", "env_ids": [], "agent_name": "solver",
          "a2a_agent_id": "theirs", "a2a_agent_version": 4}
HUMAN = {"id": "deploy-human", "type": "deploy_agent", "env_ids": [], "agent_name": "human_agent",
         "a2a_agent_id": "sim", "a2a_agent_version": 2}
INSTALL = {"id": "install-human", "type": "install_agent", "agent_name": "human_agent", "a2a_agent_id": "sim",
           "a2a_agent_version": 2, "sandbox_name": "world"}
UNNAMED = {"id": "deploy-default", "type": "deploy_agent", "env_ids": [], "agent_name": "solver",
           "a2a_agent_version": 6}
UNNAMED_INSTALL = {"id": "install-default", "type": "install_agent", "agent_name": "solver", "a2a_agent_version": 6,
                   "sandbox_name": "world"}


class _Asked(Exception):
    pass


@pytest.fixture
def asked(monkeypatch, local_stores) -> list:
    """Each (id, version) a step asks the agent store for; the step stops there."""
    asked: list = []

    def get(agent_id, version=None):
        asked.append((agent_id, version))
        raise _Asked

    monkeypatch.setattr(A2AAgent, "get", staticmethod(get))
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: SimpleNamespace(
        get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=SimpleNamespace())))
    return asked


async def _run(step_dict: dict, asked: list, **overrides) -> tuple:
    step_class = InstallAgentTaskStep if step_dict["type"] == "install_agent" else DeployAgentTaskStep
    context = TaskStepContext(instance_id="run-1", metadata={"user_overrides": overrides}, deployed_sandboxes=[
        DeployedSandbox(sandbox_name="world", sandbox_id="sb-1", sandbox_mode="vm", sandbox_type="local",
                        tunnel_urls={"8000": "http://127.0.0.1:8000"})])
    with pytest.raises(_Asked):
        await step_class.from_dict(step_dict).execute(context)
    return asked[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [SOLVER, HUMAN, INSTALL], ids=lambda step: step["id"])
async def test_a_step_with_no_pick_keeps_its_agent_and_pin(asked, step):
    assert await _run(step, asked) == (step["a2a_agent_id"], step["a2a_agent_version"])


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [SOLVER, HUMAN, INSTALL], ids=lambda step: step["id"])
async def test_a_pick_of_another_agent_does_not_take_the_steps_pinned_version(asked, step):
    assert await _run(step, asked, a2a_agent_id="mine") == ("mine", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [SOLVER, HUMAN, INSTALL], ids=lambda step: step["id"])
async def test_a_pick_of_another_agent_with_a_version_deploys_that_version(asked, step):
    assert await _run(step, asked, a2a_agent_id="mine", a2a_agent_version=9) == ("mine", 9)


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [SOLVER, HUMAN, INSTALL], ids=lambda step: step["id"])
async def test_a_pick_of_the_steps_own_agent_keeps_its_pin(asked, step):
    assert await _run(step, asked, a2a_agent_id=step["a2a_agent_id"]) == (step["a2a_agent_id"], step["a2a_agent_version"])


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [UNNAMED, UNNAMED_INSTALL], ids=lambda step: step["id"])
async def test_a_step_naming_no_agent_keeps_its_pin_only_for_the_default_agent(asked, local_stores, step):
    default = local_stores.get_default_a2a_agent_id()
    assert await _run(step, asked) == (default, 6)
    assert await _run(step, asked, a2a_agent_id=default) == (default, 6)
    assert await _run(step, asked, a2a_agent_id="mine") == ("mine", None)


@pytest.mark.asyncio
@pytest.mark.parametrize("step", [UNNAMED, UNNAMED_INSTALL], ids=lambda step: step["id"])
async def test_a_pick_runs_where_the_default_agent_cannot_be_read(asked, tmp_path, monkeypatch, step):
    config = tmp_path / "config.toml"
    config.write_text('[agents]\ndefault_a2a_agent_id = "env:AGENT_ENV_TEST_UNSET_DEFAULT_AGENT"\n')
    monkeypatch.delenv("AGENT_ENV_TEST_UNSET_DEFAULT_AGENT", raising=False)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    reset_config()
    with pytest.raises(ConfigError):
        get_config().get_default_a2a_agent_id()

    assert await _run(step, asked, a2a_agent_id="mine") == ("mine", None)
    assert await _run(step, asked, a2a_agent_id="mine", a2a_agent_version=9) == ("mine", 9)


@pytest.mark.asyncio
async def test_a_picked_version_alone_applies_to_every_deploy_and_no_install(asked):
    assert await _run(SOLVER, asked, a2a_agent_version=9) == ("theirs", 9)
    assert await _run(HUMAN, asked, a2a_agent_version=9) == ("sim", 9)
    assert await _run(INSTALL, asked, a2a_agent_version=9) == ("sim", 2)


@pytest.mark.asyncio
async def test_a_steps_own_version_override_wins_over_the_pick(asked):
    step_params = {"deploy-solver": {"a2a_agent_version": 5}}
    assert await _run(SOLVER, asked, a2a_agent_id="mine", step_params=step_params) == ("mine", 5)
    assert await _run(SOLVER, asked, a2a_agent_id="mine", a2a_agent_version=9, step_params=step_params) == ("mine", 5)
