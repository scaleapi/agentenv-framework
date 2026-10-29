"""A container-mode deploy_sandbox runs its image by name, so a provider whose sandboxes are elsewhere can't pull one from
this machine's registry; one on this machine can, and a chain skips the rest for it."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep


def _provider(on_this_machine: bool):
    sandbox = AsyncMock()
    sandbox.sandbox_id, sandbox.mode, sandbox.type = "sb-1", "container", "local"
    sandbox.tunnel_urls, sandbox.vnc_url, sandbox.network_policy = {}, None, NetworkPolicy()
    provider = AsyncMock(ON_THIS_MACHINE=on_this_machine)
    provider.links = (provider,)
    provider.create_container.return_value = sandbox
    return provider


class _CustomContainerProvider(SandboxProvider):
    ON_THIS_MACHINE = True

    def __init__(self):
        self.calls = []

    async def create_sandbox(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            sandbox_id="custom-1", mode="container", type="custom", tunnel_urls={},
            vnc_url=None, network_policy=NetworkPolicy(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("chained", [False, True], ids=["direct", "fallback-chain"])
async def test_custom_container_only_provider_still_deploys(chained):
    step = DeploySandboxTaskStep(
        id="s", version=1, sandbox_name="box", sandbox_mode="container",
        image="example.com/box:v1", port=80, env_vars={"MODE": "test"},
    )
    custom = _CustomContainerProvider()
    if chained:
        first = _provider(on_this_machine=True)
        first.create_container.side_effect = RuntimeError("unavailable")
        provider = ChainedSandboxProvider([first, custom])
    else:
        provider = custom

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider", return_value=provider):
        context = await step.execute(TaskStepContext())

    assert custom.calls[0]["env"] == {"MODE": "test"}
    assert context.deployed_sandboxes[0].sandbox_mode == "container"


@pytest.mark.asyncio
@pytest.mark.parametrize("on_this_machine", [False, True], ids=["elsewhere", "on-this-machine"])
async def test_a_container_sandbox_pulls_an_image_from_this_machines_registry_only_on_this_machine(on_this_machine):
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="container",
                                 image="localhost:5000/box:v1", port=80)
    provider = _provider(on_this_machine)

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider", return_value=provider):
        if on_this_machine:
            await step.execute(TaskStepContext())
            provider.create_container.assert_awaited_once()
        else:
            with pytest.raises(ValueError, match="Can't deploy sandbox 'box' with AsyncMock, whose sandboxes pull it by "
                                                 "name: localhost:5000/box:v1 is in a registry on this machine, which a "
                                                 "sandbox elsewhere can't pull from"):
                await step.execute(TaskStepContext())
            provider.create_container.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_chain_pulls_an_image_from_this_machines_registry_on_its_provider_here_skipping_one_elsewhere():
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="container",
                                 image="localhost:5000/box:v1", port=80)
    elsewhere, here = _provider(on_this_machine=False), _provider(on_this_machine=True)

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider",
               return_value=ChainedSandboxProvider([elsewhere, here])):
        await step.execute(TaskStepContext())

    elsewhere.create_container.assert_not_awaited()
    here.create_container.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_container_deploy_falls_back_when_the_first_provider_fails():
    step = DeploySandboxTaskStep(
        id="s", version=1, sandbox_name="box", sandbox_mode="container",
        image="localhost:5000/box:v1", port=80, env_vars={"MODE": "test"},
    )
    first, second = _provider(on_this_machine=True), _provider(on_this_machine=True)
    first.create_container.side_effect = RuntimeError("unavailable")
    context = TaskStepContext()

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.get_sandbox_provider",
               return_value=ChainedSandboxProvider([first, second])):
        await step.execute(context)

    first.create_container.assert_awaited_once()
    second.create_container.assert_awaited_once()
    assert second.create_container.await_args.kwargs["env"] == {"MODE": "test"}
    assert context.deployed_sandboxes[0].sandbox_mode == "container"
