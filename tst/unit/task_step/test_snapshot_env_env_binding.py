"""Which deployment of an env a `snapshot_env` step exports.

Matching on `env_id` alone sends every branch's snapshot to one deployment: the
first-appended env gets snapshotted k times while the other rollouts' state is
never captured. Same defect #808 fixed for `deploy_agent` and #817 for
`load_artifact`, on the step that reads the state back out.

`snapshot_env` has three resolution modes rather than one, so the filter is
exercised against each: `env_instance_id` (exact, untouched), `env_id`, and the
bare no-selector path.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.snapshot_env import EnvSnapshotResult, SnapshotEnvTaskStep

ENV_ID = "haltbench-rollout-env-0gr8479d"


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


def _snapshotter(env_step_id: str | None, **kwargs) -> SnapshotEnvTaskStep:
    params = {"env_id": ENV_ID, "env_step_id": env_step_id}
    params.update(kwargs)
    return SnapshotEnvTaskStep(id=f"snap-{env_step_id or 'unbound'}", version=None, **params)


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
async def test_each_snapshot_targets_its_own_branch_env():
    context = await _deploy_two_in_parallel()

    assert _snapshotter("env-a")._resolve_deployed_env(context).sandbox_id == "sb-AAA"
    assert _snapshotter("env-b")._resolve_deployed_env(context).sandbox_id == "sb-BBB"


@pytest.mark.asyncio
async def test_ambiguous_env_without_env_step_id_raises():
    context = await _deploy_two_in_parallel()

    with pytest.raises(RuntimeError, match="2 deployments of env"):
        _snapshotter(None)._resolve_deployed_env(context)


@pytest.mark.asyncio
async def test_ambiguity_raises_even_with_a_gateway_url_override():
    """`gateway_url` excuses *no* deployment, never an unresolved choice between k.

    Silently exporting whichever env appended first while an explicit override is
    also set would snapshot one sandbox's state under another's identity.
    """
    context = await _deploy_two_in_parallel()

    with pytest.raises(RuntimeError, match="2 deployments of env"):
        _snapshotter(None, gateway_url="https://gw-override")._resolve_deployed_env(context)


@pytest.mark.asyncio
async def test_stale_env_step_id_raises_even_with_one_deployment():
    context = await _deploy_one()

    with pytest.raises(RuntimeError, match="names no deployment"):
        _snapshotter("env-typo")._resolve_deployed_env(context)


@pytest.mark.asyncio
async def test_single_deployment_resolves_without_env_step_id():
    """The unchanged path every existing task takes."""
    context = await _deploy_one()

    assert _snapshotter(None)._resolve_deployed_env(context).sandbox_id == "sb-ONLY"


@pytest.mark.asyncio
async def test_pure_override_mode_still_resolves_to_none():
    """env_id + gateway_url with nothing deployed: the filter leaves it alone."""
    assert _snapshotter(None, gateway_url="https://gw-x")._resolve_deployed_env(TaskStepContext()) is None
    assert _snapshotter("env-a", gateway_url="https://gw-x")._resolve_deployed_env(TaskStepContext()) is None


@pytest.mark.asyncio
async def test_missing_env_without_override_still_raises():
    with pytest.raises(RuntimeError, match="no deployed env with env_id|No deployed env with env_id"):
        _snapshotter(None)._resolve_deployed_env(TaskStepContext())


@pytest.mark.asyncio
async def test_env_step_id_alone_selects_one_of_k():
    """No env_id: env_step_id is a third answer to what was an ambiguity error."""
    context = await _deploy_two_in_parallel()

    bare = SnapshotEnvTaskStep(id="snap-bare", version=None, env_step_id="env-b")
    assert bare._resolve_deployed_env(context).sandbox_id == "sb-BBB"

    unbound = SnapshotEnvTaskStep(id="snap-bare", version=None)
    with pytest.raises(RuntimeError, match="needs env_id, env_instance_id or env_step_id"):
        unbound._resolve_deployed_env(context)


@pytest.mark.asyncio
async def test_env_instance_id_takes_precedence_and_is_unfiltered():
    """The exact-instance path predates env_step_id and is not narrowed by it."""
    context = await _deploy_two_in_parallel()
    context.deployed_envs[1].instance_id = "inst-BBB"

    step = SnapshotEnvTaskStep(
        id="snap-inst", version=None, env_id=ENV_ID,
        env_instance_id="inst-BBB", env_step_id="env-a",
    )
    assert step._resolve_deployed_env(context).sandbox_id == "sb-BBB"


@pytest.mark.asyncio
async def test_legacy_lookup_would_have_snapshotted_one_env_twice():
    """Pins the defect: verbatim the expression snapshot_env used before this change."""
    context = await _deploy_two_in_parallel()

    def legacy_lookup(env_id):
        return next((d for d in context.deployed_envs if d.env_id == env_id), None)

    for_snapshot_a = legacy_lookup(ENV_ID)
    for_snapshot_b = legacy_lookup(ENV_ID)

    # Both branches resolve to the first append: sb-AAA is exported twice and
    # sb-BBB's state — which is deployed and running — is never captured.
    assert for_snapshot_a.sandbox_id == for_snapshot_b.sandbox_id == "sb-AAA"
    assert {d.sandbox_id for d in context.deployed_envs} == {"sb-AAA", "sb-BBB"}


def test_env_step_id_round_trips():
    step = _snapshotter("env-a")
    assert SnapshotEnvTaskStep.from_dict(step.to_dict()).env_step_id == "env-a"


# ── through execute(): the bound deployment is the one actually exported ──────
#
# The tests above pin resolution. These pin that the resolved deployment is what
# execute() exports FROM — resolution is only worth anything if its result
# reaches the gateway call.


async def _execute_and_capture(step: SnapshotEnvTaskStep, context: TaskStepContext) -> dict:
    """Run execute() with the export itself mocked; return its call kwargs."""
    seen: dict = {}

    async def _fake_snapshot(**kwargs):
        seen.update(kwargs)
        return EnvSnapshotResult(
            environment_universe_artifact_id=f"universe-from-{kwargs['gateway_url']}",
            environment_universe_artifact_version=1,
            environments_snapshotted=["svc"],
            total=1,
        )

    with patch("agent_env.env.env.Env") as Env, patch.object(
        SnapshotEnvTaskStep, "snapshot_env_state", side_effect=_fake_snapshot
    ):
        Env.get.return_value = MagicMock()
        await step.execute(context)
    return seen


@pytest.mark.asyncio
async def test_execute_exports_from_the_bound_branchs_gateway():
    context = await _deploy_two_in_parallel()
    context.instance_id = "ti-01ABCDEF"

    seen_a = await _execute_and_capture(_snapshotter("env-a"), context)
    seen_b = await _execute_and_capture(_snapshotter("env-b"), context)

    assert seen_a["gateway_url"] == "https://gw-sb-AAA"
    assert seen_b["gateway_url"] == "https://gw-sb-BBB"
    assert seen_a["deployed"].sandbox_id == "sb-AAA"
    assert seen_b["deployed"].sandbox_id == "sb-BBB"

    # Each step records its own result; before this change both named gw-sb-AAA.
    recorded = context.metadata["env_snapshotted_universes"]
    assert recorded["snap-env-a"]["id"] == "universe-from-https://gw-sb-AAA"
    assert recorded["snap-env-b"]["id"] == "universe-from-https://gw-sb-BBB"


@pytest.mark.asyncio
async def test_execute_on_an_ambiguous_env_fails_before_exporting_anything():
    """The error has to land before the gateway call, not export the wrong sandbox."""
    context = await _deploy_two_in_parallel()
    context.instance_id = "ti-01ABCDEF"

    with patch.object(SnapshotEnvTaskStep, "snapshot_env_state") as export:
        with pytest.raises(RuntimeError, match="2 deployments of env"):
            await _snapshotter(None).execute(context)
    export.assert_not_called()
    assert "env_snapshotted_universes" not in context.metadata
