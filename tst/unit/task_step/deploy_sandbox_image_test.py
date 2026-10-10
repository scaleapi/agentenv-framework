"""A container-mode deploy_sandbox runs its image by name, so a provider whose sandboxes are elsewhere can't pull one from
this machine's registry; one on this machine can."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep


def _provider(on_this_machine: bool):
    sandbox = AsyncMock()
    sandbox.sandbox_id, sandbox.mode, sandbox.type = "sb-1", "container", "local"
    sandbox.tunnel_urls, sandbox.vnc_url, sandbox.network_policy = {}, None, NetworkPolicy()
    provider = AsyncMock(ON_THIS_MACHINE=on_this_machine)
    provider.links = (provider,)
    provider.create_sandbox.return_value = sandbox
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("on_this_machine", [False, True], ids=["elsewhere", "on-this-machine"])
async def test_a_container_sandbox_pulls_an_image_from_this_machines_registry_only_on_this_machine(on_this_machine):
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="container",
                                 image="localhost:5000/box:v1", port=80)
    provider = _provider(on_this_machine)

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider", return_value=provider):
        if on_this_machine:
            await step.execute(TaskStepContext())
            provider.create_sandbox.assert_awaited_once()
        else:
            with pytest.raises(ValueError, match="Can't deploy sandbox 'box' with AsyncMock: localhost:5000/box:v1 is in a "
                                                 "registry on this machine, which a sandbox elsewhere can't pull from"):
                await step.execute(TaskStepContext())
            provider.create_sandbox.assert_not_awaited()
