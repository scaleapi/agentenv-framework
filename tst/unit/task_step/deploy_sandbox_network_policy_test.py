"""network_policy on DeploySandboxTaskStep: wire round-trip, override, and both create paths."""

from unittest.mock import AsyncMock, patch

import pytest

from agent_env.providers.sandbox import NetworkMode, NetworkPolicy
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep

ALLOWLIST = {"mode": "allowlist", "allow_hosts": ["llm-proxy.example.com"]}


def _step(**kw):
    kw.setdefault("sandbox_mode", "vm")
    return DeploySandboxTaskStep(id="s", version=1, sandbox_name="box", **kw)


def _fake_sandbox(policy=None):
    sandbox = AsyncMock()
    sandbox.sandbox_id, sandbox.mode, sandbox.type = "sb-1", "vm", "modal_vm"
    sandbox.tunnel_urls, sandbox.vnc_url = {}, None
    sandbox.network_policy = policy or NetworkPolicy()
    return sandbox


def test_absent_policy_stays_none():
    step = _step()
    assert step.network_policy is None
    assert step.to_dict()["network_policy"] is None


def test_round_trip_through_the_wire():
    step = _step(network_policy=ALLOWLIST)
    assert step.network_policy == {
        "mode": "allowlist", "allow_hosts": ["llm-proxy.example.com"], "allow_cidrs": []
    }
    assert DeploySandboxTaskStep.from_dict(step.to_dict()).network_policy == step.network_policy


def test_an_unknown_mode_is_rejected_when_the_task_is_built():
    with pytest.raises(ValueError, match="Unknown network policy mode"):
        _step(network_policy={"mode": "no-network"})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, create_call, extra", [
    ("vm", "create_vm", {}),
    ("container", "create_sandbox", {"port": 8000}),  # container mode requires image + port
])
async def test_the_policy_reaches_both_create_paths(mode, create_call, extra):
    step = _step(sandbox_mode=mode, image="img", network_policy=ALLOWLIST, **extra)
    provider = AsyncMock()
    getattr(provider, create_call).return_value = _fake_sandbox()
    with patch("agent_env.providers.sandbox_provider.get_sandbox_provider", return_value=provider):
        await step.execute(TaskStepContext())
    passed = getattr(provider, create_call).await_args.kwargs["network_policy"]
    assert passed == NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("llm-proxy.example.com",))


@pytest.mark.asyncio
async def test_a_per_run_override_wins_over_the_step():
    step = _step(image="img", network_policy=ALLOWLIST)
    provider = AsyncMock()
    provider.create_vm.return_value = _fake_sandbox()
    context = TaskStepContext(metadata={"user_overrides": {"network_policy": {"mode": "allowlist", "allow_hosts": ["override.example.com"]}}})
    with patch("agent_env.providers.sandbox_provider.get_sandbox_provider", return_value=provider):
        await step.execute(context)
    assert provider.create_vm.await_args.kwargs["network_policy"].allow_hosts == ("override.example.com",)


@pytest.mark.asyncio
async def test_the_effective_policy_is_recorded_not_the_requested_one():
    """Under a chain the caller cannot tell which backend ran, so record what was applied."""
    step = _step(image="img", network_policy=ALLOWLIST)
    applied = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("llm-proxy.example.com", "*.modal.host"))
    provider = AsyncMock()
    provider.create_vm.return_value = _fake_sandbox(applied)
    context = TaskStepContext()
    with patch("agent_env.providers.sandbox_provider.get_sandbox_provider", return_value=provider):
        await step.execute(context)
    assert context.deployed_sandboxes[0].network_policy == applied.to_dict()
