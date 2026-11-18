"""`task run` shows a deploy step its own agent, even when a parallel deploy finished after it."""

from __future__ import annotations

from agent_env.cli.task.run import _step_output_lines
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep


def _context() -> TaskStepContext:
    ctx = TaskStepContext()
    ctx.deployed_agents += [DeployedAgent(agent_name="solver", api_url="http://solver.test"),
                            DeployedAgent(agent_name="hannah", api_url="http://hannah.test")]
    return ctx


def test_each_deploy_step_shows_the_agent_it_deployed():
    for name in ("solver", "hannah"):  # solver's deploy ended first, so hannah's agent is the last one in the run
        [line] = _step_output_lines(DeployAgentTaskStep(id=f"deploy-{name}", version=None, agent_name=name), _context())
        assert f"agent_name: {name}\n" in line


def test_a_deploy_step_whose_agent_isnt_deployed_shows_nothing():
    assert _step_output_lines(DeployAgentTaskStep(id="deploy", version=None, agent_name="other"), _context()) == []
