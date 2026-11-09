"""A deploy_sandbox step that sets no cpu or memory reserves one vCPU and 8 GiB, whether built in code or loaded from a
stored dict, while keeping them unset on the step and in its stored dict; a stored size is kept as written."""

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
def test_an_unsized_step_stays_unsized(step):
    assert (step.cpu, step.memory_mb) == (None, None)
    stored = step.to_dict()
    assert "cpu" not in stored and "memory_mb" not in stored  # left out, so no older reader meets a null
    assert DeploySandboxTaskStep.from_dict(stored).cpu is None


@pytest.mark.parametrize("size", [{"cpu": 2.0, "memory_mb": 8192}, {"cpu": 1.0}, {"memory_mb": 8192},
                                  {"cpu": 1.0, "memory_mb": 4096}], ids=["both", "cpu", "memory", "cpu-at-default"])
def test_a_stored_size_is_kept(size):
    step = DeploySandboxTaskStep.from_dict(_stored(**size))
    assert (step.cpu, step.memory_mb) == (size.get("cpu"), size.get("memory_mb"))


def test_a_step_that_sets_the_default_pair_keeps_it_through_a_save():
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode="vm", cpu=1.0, memory_mb=8192)
    reloaded = DeploySandboxTaskStep.from_dict(step.to_dict())
    assert (reloaded.cpu, reloaded.memory_mb) == (1.0, 8192)


def test_the_old_stored_default_reads_as_unsized():
    """Every unsized step was stored as 1.0 / 8192 before a step could leave its size unset."""
    step = DeploySandboxTaskStep.from_dict(_stored(cpu=1.0, memory_mb=8192))
    assert (step.cpu, step.memory_mb) == (None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, create, extra", [
    ("vm", "create_vm", {}),
    ("container", "create_container", {"image": "example.com/box:v1", "port": 80}),
], ids=["vm", "container"])
@pytest.mark.parametrize("size, requested", [
    ({}, (1.0, 8192)),
    ({"cpu": 4.0, "memory_mb": 2048}, (4.0, 2048)),
    ({"memory_mb": 2048}, (1.0, 2048)),
], ids=["unsized", "sized", "memory-only"])
async def test_the_provider_is_asked_for_the_steps_size(mode, create, extra, size, requested):
    step = DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", sandbox_mode=mode, sandbox_type="any",
                                 **size, **extra)
    provider = _provider()

    with patch("agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider", return_value=provider):
        await step.execute(TaskStepContext())

    kwargs = getattr(provider, create).call_args.kwargs
    assert (kwargs["cpu"], kwargs["memory"]) == requested
