"""A deploy_sandbox step that sets no cpu reserves one vCPU, whether built in code or loaded from a stored dict, and
a stored cpu is kept as written."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep


def _provider():
    sandbox = AsyncMock()
    sandbox.sandbox_id, sandbox.mode, sandbox.type = "sb-1", "vm", "modal_vm"
    sandbox.tunnel_urls, sandbox.vnc_url, sandbox.network_policy = {}, None, NetworkPolicy()
    provider = AsyncMock(ON_THIS_MACHINE=False)
    provider.links = (provider,)
    provider.create_vm.return_value = sandbox
    provider.create_container.return_value = sandbox
    return provider


def _stored(**fields) -> dict:
    return {"id": "s", "version": 1, "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", **fields}


@pytest.mark.parametrize("step", [
    DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="vm"),
    DeploySandboxTaskStep.from_dict(_stored()),
], ids=["constructed", "from-dict"])
def test_an_unsized_step_reserves_one_cpu(step):
    assert step.cpu == 1.0


def test_a_stored_cpu_is_kept():
    assert DeploySandboxTaskStep.from_dict(_stored(cpu=2.0)).cpu == 2.0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, create, extra", [
    ("vm", "create_vm", {}),
    ("container", "create_container", {"image": "example.com/box:v1", "port": 80}),
], ids=["vm", "container"])
@pytest.mark.parametrize("step_cpu, requested", [(None, 1.0), (4.0, 4.0)], ids=["unsized", "sized"])
async def test_the_provider_is_asked_for_the_steps_cpu(mode, create, extra, step_cpu, requested):
    sized = {} if step_cpu is None else {"cpu": step_cpu}
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode=mode, sandbox_type="any",
                                 **sized, **extra)
    provider = _provider()

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider", return_value=provider):
        await step.execute(TaskStepContext())

    assert getattr(provider, create).call_args.kwargs["cpu"] == requested
