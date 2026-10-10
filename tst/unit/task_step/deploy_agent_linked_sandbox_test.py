"""deploy_agent places an agent only on a VM sandbox that exposes its A2A port, and refuses any other before deploying
anything, in the words a bundle run's preflight refuses it with too."""

import re

import pytest

from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep, linked_sandbox_problem


@pytest.mark.parametrize("mode, tunnel_urls, problem", [
    ("container", {"8000": "https://t-8000"}, "Cannot link agent to sandbox 'box' (mode='container'); only VM-mode "
                                              "sandboxes can host an additional agent container"),
    ("vm", {"9000": "https://t-9000"}, "Sandbox 'box' does not expose port 8000; re-deploy DeploySandboxTaskStep with "
                                       "exposed_ports=[8000]"),
    ("vm", None, "Sandbox 'box' does not expose port 8000; re-deploy DeploySandboxTaskStep with exposed_ports=[8000]"),
], ids=["container", "other-port", "no-ports"])
@pytest.mark.asyncio
async def test_an_agent_is_refused_a_sandbox_it_cant_be_placed_on(mode, tunnel_urls, problem):
    step = DeployAgentTaskStep(id="agent", version=1, env_ids=[], a2a_agent_id="solver", sandbox_name="box")
    context = TaskStepContext()
    context.deployed_sandboxes.append(DeployedSandbox(sandbox_name="box", sandbox_id="sb-1", sandbox_mode=mode,
                                                      tunnel_urls=tunnel_urls))

    with pytest.raises(RuntimeError, match=f"^{re.escape(problem)}$"):
        await step.execute(context)


def test_a_vm_sandbox_exposing_the_a2a_port_takes_the_agent():
    assert linked_sandbox_problem("box", "vm", {"8000": "https://t-8000"}) is None
    assert linked_sandbox_problem("box", "vm", [8000]) is None  # as a deploy_sandbox step's exposed_ports names it
