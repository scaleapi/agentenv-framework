"""Sandbox TTL resolution: the run budget in `user_overrides['ttl_seconds']` wins over the
step's own value, else falls back to it."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep


async def _env_deploy_ttl(step_ttl, overrides) -> int:
    env = MagicMock()
    env.deploy = AsyncMock(return_value=DeployedEnv(
        env_id="art-1", env_version=2, gateway_url="http://gw", mcp_url="http://mcp",
        db_web_url=None, sandbox_id="s1", metadata=None,
    ))
    step = DeployEnvTaskStep(id="d", version=None, env_id="art-1", env_version=2, ttl_seconds=step_ttl)
    ctx = TaskStepContext(metadata={"user_overrides": overrides})
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(ctx)
    return env.deploy.await_args.kwargs["ttl_seconds"]


@pytest.mark.asyncio
async def test_deploy_env_uses_budget_override():
    assert await _env_deploy_ttl(7200, {"ttl_seconds": 30600}) == 30600


@pytest.mark.asyncio
async def test_deploy_env_keeps_step_ttl_without_override():
    assert await _env_deploy_ttl(7200, {}) == 7200


@pytest.mark.asyncio
async def test_deploy_sandbox_uses_budget_override():
    sb = MagicMock(sandbox_id="s1", mode="vm", type="remote", tunnel_urls={}, vnc_url=None)
    provider = MagicMock()
    provider.create_vm = AsyncMock(return_value=sb)
    step = DeploySandboxTaskStep(
        id="d", version=None, sandbox_name="sbx", sandbox_mode="vm", image="img", ttl_seconds=1800,
    )
    ctx = TaskStepContext(metadata={"user_overrides": {"ttl_seconds": 30600}})
    with patch("agent_env.providers.sandbox_provider.get_sandbox_provider", return_value=provider):
        await step.execute(ctx)
    assert provider.create_vm.await_args.kwargs["timeout"] == 30600


@pytest.mark.asyncio
async def test_deploy_sandbox_container_mode_uses_budget_override():
    sb = MagicMock(sandbox_id="s1", mode="container", type="remote", tunnel_urls={}, vnc_url=None)
    provider = MagicMock()
    provider.create_sandbox = AsyncMock(return_value=sb)
    step = DeploySandboxTaskStep(
        id="d", version=None, sandbox_name="sbx", sandbox_mode="container",
        image="img", port=8080, ttl_seconds=1800,
    )
    ctx = TaskStepContext(metadata={"user_overrides": {"ttl_seconds": 30600}})
    with patch("agent_env.providers.sandbox_provider.get_sandbox_provider", return_value=provider):
        await step.execute(ctx)
    assert provider.create_sandbox.await_args.kwargs["timeout"] == 30600


@pytest.mark.asyncio
async def test_deploy_sandbox_keeps_step_ttl_without_override():
    sb = MagicMock(sandbox_id="s1", mode="vm", type="remote", tunnel_urls={}, vnc_url=None)
    provider = MagicMock()
    provider.create_vm = AsyncMock(return_value=sb)
    step = DeploySandboxTaskStep(
        id="d", version=None, sandbox_name="sbx", sandbox_mode="vm", image="img", ttl_seconds=1800,
    )
    ctx = TaskStepContext(metadata={"user_overrides": {}})
    with patch("agent_env.providers.sandbox_provider.get_sandbox_provider", return_value=provider):
        await step.execute(ctx)
    assert provider.create_vm.await_args.kwargs["timeout"] == 1800
