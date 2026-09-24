"""Which deployment of an env a `load_artifact` step loads into.

Matching on `env_id` alone sends every branch's load to one deployment, leaving the
others deployed but empty — their agents then bind correctly (#808) to an env with
no data.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep

ENV_ID = "openclaw-damien_coleman-multi-0gr8479d"


def _fake_env(sandbox_id: str):
    async def _deploy(**_kwargs):
        await asyncio.sleep(0)
        return DeployedEnv(
            env_id=ENV_ID, env_version=1, gateway_url=f"https://gw-{sandbox_id}",
            mcp_url=f"https://{sandbox_id}/mcp", db_web_url=None, sandbox_id=sandbox_id,
        )

    env = MagicMock()
    env.deploy = AsyncMock(side_effect=_deploy)
    return env


def _loader(env_step_id: str | None) -> LoadArtifactTaskStep:
    return LoadArtifactTaskStep(
        id=f"load-{env_step_id or 'unbound'}", version=None, env_id=ENV_ID,
        env_step_id=env_step_id, artifact_id="universe", artifact_version=1,
    )


async def _deploy_two_in_parallel() -> TaskStepContext:
    context = TaskStepContext()
    envs = {"env-a": _fake_env("sb-AAA"), "env-b": _fake_env("sb-BBB")}
    order = iter(["env-a", "env-b"])
    with patch("agent_env.env.env.Env") as Env:
        Env.get.side_effect = lambda *a, **k: envs[next(order)]
        await asyncio.gather(
            DeployEnvTaskStep(id="env-a", version=None, env_id=ENV_ID).execute(context),
            DeployEnvTaskStep(id="env-b", version=None, env_id=ENV_ID).execute(context),
        )
    return context


async def _deploy_one() -> TaskStepContext:
    context = TaskStepContext()
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = _fake_env("sb-ONLY")
        await DeployEnvTaskStep(id="env-only", version=None, env_id=ENV_ID).execute(context)
    return context


@pytest.mark.asyncio
async def test_each_load_targets_its_own_branch_env():
    context = await _deploy_two_in_parallel()

    assert _loader("env-a")._deployed_env(context).sandbox_id == "sb-AAA"
    assert _loader("env-b")._deployed_env(context).sandbox_id == "sb-BBB"


@pytest.mark.asyncio
async def test_ambiguous_env_without_env_step_id_raises():
    context = await _deploy_two_in_parallel()

    with pytest.raises(RuntimeError, match="2 deployments of env"):
        _loader(None)._deployed_env(context)


@pytest.mark.asyncio
async def test_stale_env_step_id_raises_even_with_one_deployment():
    context = await _deploy_one()

    with pytest.raises(RuntimeError, match="names no deployment"):
        _loader("env-typo")._deployed_env(context)


@pytest.mark.asyncio
async def test_single_deployment_resolves_without_env_step_id():
    """The unchanged path every existing task takes."""
    context = await _deploy_one()

    assert _loader(None)._deployed_env(context).sandbox_id == "sb-ONLY"


@pytest.mark.asyncio
async def test_env_not_deployed_resolves_to_none():
    assert _loader(None)._deployed_env(TaskStepContext()) is None


@pytest.mark.asyncio
async def test_legacy_lookup_would_have_sent_both_loads_to_one_env():
    """Pins the defect: verbatim the expression load_artifact used before this change."""
    context = await _deploy_two_in_parallel()

    def legacy_lookup(env_id):
        return next((d for d in context.deployed_envs if d.env_id == env_id), None)

    for_load_a = legacy_lookup(ENV_ID)
    for_load_b = legacy_lookup(ENV_ID)

    # Both branches resolve to the first append, so env-b is never loaded.
    assert for_load_a.sandbox_id == for_load_b.sandbox_id == "sb-AAA"


def test_env_step_id_round_trips():
    step = _loader("env-a")
    assert LoadArtifactTaskStep.from_dict(step.to_dict()).env_step_id == "env-a"
