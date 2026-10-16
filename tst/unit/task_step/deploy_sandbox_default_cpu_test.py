"""A deploy_sandbox step that sets no cpu reserves its provider's DEPLOY_SANDBOX_CPU: 1 vCPU on Modal, whose cpu is
a reservation the sandbox bursts above, and 2 on every other provider. A step that sets cpu keeps it."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep


def _provider(default_cpu: float):
    sandbox = AsyncMock()
    sandbox.sandbox_id, sandbox.mode, sandbox.type = "sb-1", "vm", "vm"
    sandbox.tunnel_urls, sandbox.vnc_url, sandbox.network_policy = {}, None, NetworkPolicy()
    provider = AsyncMock(DEPLOY_SANDBOX_CPU=default_cpu, ON_THIS_MACHINE=False)
    provider.links = (provider,)
    provider.create_vm.return_value = sandbox
    provider.create_container.return_value = sandbox
    return provider


def _stored(**fields) -> dict:
    return {"id": "s", "version": 1, "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", **fields}


def test_modal_reserves_one_cpu_and_every_other_provider_two():
    assert (ModalSandboxProvider.DEPLOY_SANDBOX_CPU, ModalVmSandboxProvider.DEPLOY_SANDBOX_CPU) == (1.0, 1.0)
    assert SandboxProvider.DEPLOY_SANDBOX_CPU == 2.0


def test_a_chain_reserves_its_providers_largest():
    assert ChainedSandboxProvider([_provider(1.0), _provider(2.0)]).DEPLOY_SANDBOX_CPU == 2.0


@pytest.mark.parametrize("step", [
    DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="vm"),
    DeploySandboxTaskStep.from_dict(_stored()),
], ids=["constructed", "from-dict"])
def test_an_unsized_step_leaves_cpu_to_the_provider(step):
    assert step.cpu is None


def test_a_stored_cpu_is_kept():
    assert DeploySandboxTaskStep.from_dict(_stored(cpu=2.0)).cpu == 2.0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, create, extra", [
    ("vm", "create_vm", {}),
    ("container", "create_container", {"image": "example.com/box:v1", "port": 80}),
], ids=["vm", "container"])
@pytest.mark.parametrize("step_cpu, default_cpu, requested", [
    (None, 1.0, 1.0),
    (None, 2.0, 2.0),
    (4.0, 1.0, 4.0),
], ids=["unsized-on-modal", "unsized-elsewhere", "sized"])
async def test_the_provider_is_asked_for_the_resolved_cpu(mode, create, extra, step_cpu, default_cpu, requested):
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode=mode, sandbox_type="any",
                                 cpu=step_cpu, **extra)
    provider = _provider(default_cpu)

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider", return_value=provider):
        await step.execute(TaskStepContext())

    assert getattr(provider, create).call_args.kwargs["cpu"] == requested
