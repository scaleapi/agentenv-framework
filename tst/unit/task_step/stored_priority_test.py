"""Stored step documents written before priority was removed still load, and drop the key."""

import pytest

from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep


@pytest.mark.parametrize("step, key", [
    (DeployEnvTaskStep(id="d", version=None, env_id="e"), "priority"),
    (DeployAgentTaskStep(id="d", version=None), "priority"),
    (DeploySandboxTaskStep(id="d", version=None, sandbox_name="sb", sandbox_mode="vm"), "priority"),
    (RubricsVerifierTaskStep(id="v", version=None, criteria=[], prompt_id="p"), "judge_priority"),
], ids=["deploy_env", "deploy_agent", "deploy_sandbox", "rubrics_verifier"])
def test_a_stored_step_with_an_unread_priority_still_loads(step, key):
    document = step.to_dict()
    assert type(step).from_dict({**document, key: 1}).to_dict() == document
