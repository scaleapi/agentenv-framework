"""The `haltbench-mvp-whichenv` DAG in-process: two `deploy_env` of one `env_id`
running concurrently, then an agent per env.

Instance haltbench-mvp-whichenv-7c9fb4o9: both envs deployed, both agents wired
to the first.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep

ENV_ID = "openclaw-damien_coleman-multi-0gr8479d"


def _fake_env_deploying_to(mcp_url: str, sandbox_id: str):
    """An Env whose deploy() yields a distinct sandbox, after a real await point.

    The await is what lets the two steps interleave.
    """
    async def _deploy(**_kwargs):
        await asyncio.sleep(0)
        return DeployedEnv(
            env_id=ENV_ID,
            env_version=1,
            gateway_url=f"https://gw-{sandbox_id}",
            mcp_url=mcp_url,
            db_web_url=None,
            sandbox_id=sandbox_id,
        )

    env = MagicMock()
    env.deploy = AsyncMock(side_effect=_deploy)
    return env


def _agent_bound_to(step_id: str | None) -> DeployAgentTaskStep:
    return DeployAgentTaskStep(
        id=f"ag-{step_id or 'unbound'}",
        version=None,
        env_ids=[ENV_ID],
        env_step_id=step_id,
    )


async def _deploy_two_in_parallel() -> TaskStepContext:
    context = TaskStepContext()
    step_a = DeployEnvTaskStep(id="env-a", version=None, env_id=ENV_ID)
    step_b = DeployEnvTaskStep(id="env-b", version=None, env_id=ENV_ID)

    envs = {
        "env-a": _fake_env_deploying_to("https://a.modal.host/mcp", "sb-AAA"),
        "env-b": _fake_env_deploying_to("https://b.modal.host/mcp", "sb-BBB"),
    }
    # Env.get() is shared, so hand each step its own env by call order.
    order = iter(["env-a", "env-b"])

    with patch("agent_env.env.env.Env") as Env:
        Env.get.side_effect = lambda *a, **k: envs[next(order)]
        await asyncio.gather(step_a.execute(context), step_b.execute(context))
    return context


@pytest.mark.asyncio
async def test_parallel_deploy_of_one_env_id_is_allowed_and_stamped():
    context = await _deploy_two_in_parallel()

    assert len(context.deployed_envs) == 2, "both rollouts must deploy"
    assert {d.sandbox_id for d in context.deployed_envs} == {"sb-AAA", "sb-BBB"}
    assert {(d.metadata or {}).get("deploy_step_id") for d in context.deployed_envs} == {"env-a", "env-b"}


@pytest.mark.asyncio
async def test_each_agent_binds_to_the_env_it_depends_on():
    context = await _deploy_two_in_parallel()
    by_step = {(d.metadata or {}).get("deploy_step_id"): d for d in context.deployed_envs}

    bound_a = _agent_bound_to("env-a")._deployed_env(context, ENV_ID)
    bound_b = _agent_bound_to("env-b")._deployed_env(context, ENV_ID)

    assert bound_a.mcp_url == by_step["env-a"].mcp_url
    assert bound_b.mcp_url == by_step["env-b"].mcp_url
    assert bound_a.mcp_url != bound_b.mcp_url, (
        "the recorded bug: both agents got the same mcp_url"
    )


@pytest.mark.asyncio
async def test_agent_without_env_step_id_refuses_to_guess():
    """The pre-fix behaviour (silently bind to deployed_envs[0]) is now an error."""
    context = await _deploy_two_in_parallel()

    with pytest.raises(RuntimeError, match="2 deployments of env"):
        _agent_bound_to(None)._deployed_env(context, ENV_ID)


@pytest.mark.asyncio
async def test_single_deployment_still_resolves_without_env_step_ids():
    """The unchanged path: one deployment, no disambiguator needed."""
    context = TaskStepContext()
    step = DeployEnvTaskStep(id="env-only", version=None, env_id=ENV_ID)
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = _fake_env_deploying_to("https://only.modal.host/mcp", "sb-ONLY")
        await step.execute(context)

    bound = _agent_bound_to(None)._deployed_env(context, ENV_ID)
    assert bound.sandbox_id == "sb-ONLY"


@pytest.mark.asyncio
async def test_legacy_env_id_lookup_would_have_bound_both_agents_to_one_env():
    """Pins the defect: verbatim the expression deploy_agent used on main."""
    context = await _deploy_two_in_parallel()

    def legacy_lookup(env_id):
        return next((d for d in context.deployed_envs if d.env_id == env_id), None)

    from_branch_a = legacy_lookup(ENV_ID)
    from_branch_b = legacy_lookup(ENV_ID)

    assert from_branch_a is from_branch_b
    assert (from_branch_a.metadata or {}).get("deploy_step_id") == "env-a", "always the first append"
    assert from_branch_b.sandbox_id != "sb-BBB"


@pytest.mark.asyncio
async def test_env_step_id_naming_no_deployment_is_an_error():
    """A stale or typo'd step id must not fall back to guessing."""
    context = await _deploy_two_in_parallel()

    with pytest.raises(RuntimeError, match="names no deployment"):
        _agent_bound_to("env-typo")._deployed_env(context, ENV_ID)


@pytest.mark.asyncio
async def test_stale_env_step_id_errors_even_with_a_single_deployment():
    """An unmatched id is an error even when there is only one candidate.

    Otherwise a typo resolves fine until the task fans out.
    """
    context = TaskStepContext()
    step = DeployEnvTaskStep(id="env-only", version=None, env_id=ENV_ID)
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = _fake_env_deploying_to("https://only.modal.host/mcp", "sb-ONLY")
        await step.execute(context)

    with pytest.raises(RuntimeError, match="names no deployment"):
        _agent_bound_to("env-typo")._deployed_env(context, ENV_ID)


@pytest.mark.asyncio
async def test_env_never_deployed_here_still_resolves_to_none():
    """The `oc_halt_*` shape: a live-endpoint env (never deployed) with no deploy_env step.

    Filtering must leave it alone so the caller can use its live URL.
    """
    context = await _deploy_two_in_parallel()

    assert _agent_bound_to("env-a")._deployed_env(context, "oc-halt-ask-human") is None


def test_deploy_agent_env_step_id_round_trips():
    step = DeployAgentTaskStep(id="ag-a", version=None, env_ids=[ENV_ID], env_step_id="env-a")
    revived = DeployAgentTaskStep.from_dict(step.to_dict())
    assert revived.env_step_id == "env-a"
