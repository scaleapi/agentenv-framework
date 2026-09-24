"""Unit tests for DeployEnvTaskStep agent registration on deploy()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep


def _deployed_env(metadata):
    return DeployedEnv(
        env_id="art-1",
        env_version=2,
        gateway_url="http://gw",
        mcp_url="http://mcp",
        db_web_url=None,
        sandbox_id="s1",
        metadata=metadata,
    )


def _fake_env(deployed_env):
    env = MagicMock()
    env.deploy = AsyncMock(return_value=deployed_env)
    return env


@pytest.mark.asyncio
async def test_execute_registers_deployed_agent_from_metadata():
    deployed_env = _deployed_env(
        {
            "deployed_agent": {
                "agent_name": "harbor_agent",
                "api_url": "http://x",
                "a2a_url": "http://x",
                "sandbox_id": "s1",
                "sandbox_type": None,
                "a2a_card": {},
            }
        }
    )
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)
    context = TaskStepContext()

    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = _fake_env(deployed_env)
        result = await step.execute(context)

    assert len(result.deployed_envs) == 1
    assert len(result.deployed_agents) == 1
    agent = result.deployed_agents[0]
    assert agent.agent_name == "harbor_agent"
    assert agent.a2a_url == "http://x"


@pytest.mark.asyncio
async def test_execute_no_deployed_agent_leaves_agents_empty():
    deployed_env = _deployed_env(None)
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)
    context = TaskStepContext()

    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = _fake_env(deployed_env)
        result = await step.execute(context)

    assert len(result.deployed_envs) == 1
    assert result.deployed_agents == []


def test_env_state_type_roundtrips_through_serialization():
    step = DeployEnvTaskStep(
        id="t-1.deploy_env", version=None, env_id="art-1", env_version=2,
        env_state_type="remote_postgres",
    )
    assert step.to_dict()["env_state_type"] == "remote_postgres"
    assert DeployEnvTaskStep.from_dict(step.to_dict()).env_state_type == "remote_postgres"


@pytest.mark.asyncio
async def test_execute_threads_env_state_type_to_deploy():
    """The step's env_state_type reaches env.deploy — so Temporal/Task.run deploys can target
    remote_postgres, not just the CLI."""
    env = _fake_env(_deployed_env(None))
    step = DeployEnvTaskStep(
        id="t-1.deploy_env", version=None, env_id="art-1", env_version=2,
        env_state_type="remote_postgres",
    )
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(TaskStepContext())
    assert env.deploy.await_args.kwargs["env_state_type"] == "remote_postgres"


@pytest.mark.asyncio
async def test_execute_user_override_env_state_type_wins():
    env = _fake_env(_deployed_env(None))
    step = DeployEnvTaskStep(
        id="t-1.deploy_env", version=None, env_id="art-1", env_version=2,
        env_state_type="local_postgres",
    )
    ctx = TaskStepContext(metadata={"user_overrides": {"env_state_type": "remote_postgres"}})
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(ctx)
    assert env.deploy.await_args.kwargs["env_state_type"] == "remote_postgres"


@pytest.mark.asyncio
async def test_execute_omits_env_state_type_when_unset():
    """Unset => not passed, so env types that don't accept env_state_type aren't broken."""
    env = _fake_env(_deployed_env(None))
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(TaskStepContext())
    assert "env_state_type" not in env.deploy.await_args.kwargs


def test_env_state_instance_id_roundtrips_through_serialization():
    step = DeployEnvTaskStep(
        id="t-1.deploy_env", version=None, env_id="art-1", env_version=2,
        env_state_type="persistent_remote_postgres", env_state_instance_id="esi-base1",
    )
    assert step.to_dict()["env_state_instance_id"] == "esi-base1"
    assert DeployEnvTaskStep.from_dict(step.to_dict()).env_state_instance_id == "esi-base1"


@pytest.mark.asyncio
async def test_execute_threads_env_state_instance_id_to_deploy():
    """The step's env_state_instance_id reaches env.deploy — so Temporal/Task.run can deploy the
    persistent backend (which requires it), at parity with `env deploy --env-state-instance-id`."""
    env = _fake_env(_deployed_env(None))
    step = DeployEnvTaskStep(
        id="t-1.deploy_env", version=None, env_id="art-1", env_version=2,
        env_state_type="persistent_remote_postgres", env_state_instance_id="esi-base1",
    )
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(TaskStepContext())
    assert env.deploy.await_args.kwargs["env_state_instance_id"] == "esi-base1"


@pytest.mark.asyncio
async def test_execute_user_override_env_state_instance_id_wins():
    env = _fake_env(_deployed_env(None))
    step = DeployEnvTaskStep(
        id="t-1.deploy_env", version=None, env_id="art-1", env_version=2,
        env_state_type="persistent_remote_postgres", env_state_instance_id="esi-step",
    )
    ctx = TaskStepContext(metadata={"user_overrides": {"env_state_instance_id": "esi-override"}})
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(ctx)
    assert env.deploy.await_args.kwargs["env_state_instance_id"] == "esi-override"


@pytest.mark.asyncio
async def test_execute_omits_env_state_instance_id_when_unset():
    """Unset => not passed, so env types that don't accept env_state_instance_id aren't broken."""
    env = _fake_env(_deployed_env(None))
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)
    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(TaskStepContext())
    assert "env_state_instance_id" not in env.deploy.await_args.kwargs


# --- Instance-record annotations ---------------------------------------------
# create_instance runs INSIDE env.deploy(), so assigning to deployed_env.metadata
# after deploy() returns lands on the in-memory object only and never reaches
# Mongo — which is why instance records showed metadata=null even though
# deploy_step_id was always set. These pin that annotations are persisted, and
# that the deploying identity is among them: env-scoped destructive operations
# (a universe load) authorize against it, so it must be on the record itself
# rather than reconstructed later from run documents.

_UPDATE_TARGET = "agent_env.task_step.task_steps.deploy_env.update_env_instance_metadata"


@pytest.mark.asyncio
async def test_execute_persists_annotations_to_the_instance_record():
    deployed_env = _deployed_env(None)
    deployed_env.instance_id = "art-1-abcd1234"
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)
    context = TaskStepContext(
        metadata={"agent_env_hub": {"caller": "copilot", "oauth_subject": "sub-42"}}
    )

    with patch("agent_env.env.env.Env") as Env, patch(_UPDATE_TARGET) as update:
        Env.get.return_value = _fake_env(deployed_env)
        await step.execute(context)

    update.assert_called_once()
    instance_id, values = update.call_args.args
    assert instance_id == "art-1-abcd1234"
    assert values["deployed_by_oauth_subject"] == "sub-42"
    assert values["deploy_step_id"] == "t-1.deploy_env"
    # Still on the in-memory object too, for same-run consumers.
    assert deployed_env.metadata["deployed_by_oauth_subject"] == "sub-42"


@pytest.mark.asyncio
async def test_execute_omits_the_owner_when_no_subject_was_carried():
    # A run started without a verified subject (an S2S caller, or a non-hub entry
    # point) must not stamp a bogus owner: a wrong owner is worse than none, since
    # it would refuse the person who actually deployed the env.
    deployed_env = _deployed_env(None)
    deployed_env.instance_id = "art-1-abcd1234"
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)
    context = TaskStepContext(metadata={"agent_env_hub": {"caller": "copilot"}})

    with patch("agent_env.env.env.Env") as Env, patch(_UPDATE_TARGET) as update:
        Env.get.return_value = _fake_env(deployed_env)
        await step.execute(context)

    _, values = update.call_args.args
    assert "deployed_by_oauth_subject" not in values
    assert values["deploy_step_id"] == "t-1.deploy_env"


@pytest.mark.asyncio
async def test_execute_skips_the_update_when_no_instance_was_recorded():
    # register_env_instance is best-effort, so instance_id can be absent. There is
    # no document to annotate and the deploy must still succeed.
    deployed_env = _deployed_env(None)
    deployed_env.instance_id = None
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)

    with patch("agent_env.env.env.Env") as Env, patch(_UPDATE_TARGET) as update:
        Env.get.return_value = _fake_env(deployed_env)
        result = await step.execute(TaskStepContext())

    update.assert_not_called()
    assert len(result.deployed_envs) == 1


@pytest.mark.asyncio
async def test_execute_merges_annotations_into_metadata_deploy_set():
    # Annotations merge into whatever deploy() already put there (e.g.
    # deployed_agent) rather than replacing it.
    deployed_env = _deployed_env({"deployed_agent_hint": "keep-me"})
    deployed_env.instance_id = "art-1-abcd1234"
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)
    context = TaskStepContext(metadata={"agent_env_hub": {"oauth_subject": "sub-42"}})

    with patch("agent_env.env.env.Env") as Env, patch(_UPDATE_TARGET):
        Env.get.return_value = _fake_env(deployed_env)
        await step.execute(context)

    assert deployed_env.metadata["deployed_agent_hint"] == "keep-me"
    assert deployed_env.metadata["deployed_by_oauth_subject"] == "sub-42"


