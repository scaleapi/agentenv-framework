"""A run-wide agent pick (``user_overrides.a2a_agent_id``: ``task run --a2a-agent-id``, a hub run's agent pick) reaches
every ``deploy_agent`` and ``install_agent`` step, a ``prompt_agent`` task's user agent included, exactly as on main.
A ``deploy_user_sim_agent`` step keeps its own agent through its own ``_resolve_agent_ref``, not anything these steps
read."""

from __future__ import annotations

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep

SOLVER = {"id": "deploy-solver", "type": "deploy_agent", "env_ids": [], "agent_name": "solver",
          "a2a_agent_id": "theirs", "a2a_agent_version": 4}
HUMAN = {"id": "deploy-human", "type": "deploy_agent", "env_ids": [], "agent_name": "human_agent",
         "a2a_agent_id": "sim", "a2a_agent_version": 2}


def _picked(**overrides) -> TaskStepContext:
    return TaskStepContext(instance_id="run-1", metadata={"user_overrides": {"a2a_agent_id": "mine", **overrides}})


def test_the_pick_reaches_a_prompt_agent_tasks_user_agent_too_at_each_steps_stored_version():
    ctx = _picked()
    assert DeployAgentTaskStep.from_dict(SOLVER)._resolve_agent_ref(ctx) == ("mine", 4)
    assert DeployAgentTaskStep.from_dict(HUMAN)._resolve_agent_ref(ctx) == ("mine", 2)


def test_a_picked_version_applies_to_every_deploy():
    ctx = _picked(a2a_agent_version=9)
    assert DeployAgentTaskStep.from_dict(SOLVER)._resolve_agent_ref(ctx) == ("mine", 9)
    assert DeployAgentTaskStep.from_dict(HUMAN)._resolve_agent_ref(ctx) == ("mine", 9)


def test_install_agent_takes_the_pick_at_its_stored_version():
    installed = InstallAgentTaskStep.from_dict({"id": "install-human", "type": "install_agent",
                                                "agent_name": "human_agent", "a2a_agent_id": "sim",
                                                "a2a_agent_version": 2, "sandbox_name": "world"})
    assert installed._agent_ref(_picked()) == ("mine", 2)
