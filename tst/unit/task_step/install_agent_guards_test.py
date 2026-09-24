"""Same-sandbox tunnel-collision guard for multi-agent host installs (no I/O in the guard)."""
import pytest
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep


def _step(agent_name):
    return InstallAgentTaskStep(id="s", version=None, sandbox_name="vm", a2a_agent_id="agent-x", agent_name=agent_name)


def _ctx(agent_name="solver-a", api_url="https://t-8000.gw", sandbox_id="vm-1"):
    ctx = TaskStepContext()
    ctx.deployed_agents.append(DeployedAgent(agent_name=agent_name, api_url=api_url, sandbox_id=sandbox_id))
    return ctx


def test_tunnel_url_collision_rejected():
    with pytest.raises(RuntimeError, match="already serves"):
        _step("solver-b")._check_tunnel_collision(_ctx(), "vm-1", "https://t-8000.gw", 8000)


def test_distinct_port_passes():
    _step("solver-b")._check_tunnel_collision(_ctx(), "vm-1", "https://t-8010.gw", 8010)


def test_other_sandbox_ignored():
    _step("solver-b")._check_tunnel_collision(_ctx(sandbox_id="vm-2"), "vm-1", "https://t-8000.gw", 8000)
