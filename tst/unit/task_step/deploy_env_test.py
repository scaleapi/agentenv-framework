"""Unit tests for DeployEnvTaskStep agent registration on deploy()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.env import DeployedGatewayEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep


def _deployed_env(metadata):
    return DeployedGatewayEnv(
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
    """The step's env_state_type reaches env.deploy — so Task.run deploys can target
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
    """The step's env_state_instance_id reaches env.deploy — so Task.run can deploy the
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
# create_instance runs inside env.deploy(), so an annotation assigned to
# deployed_env.metadata afterwards reaches the record only if it is persisted.

_UPDATE_TARGET = "agent_env.task_step.task_steps.deploy_env.update_env_instance_metadata"


@pytest.mark.asyncio
async def test_execute_persists_annotations_to_the_instance_record():
    deployed_env = _deployed_env(None)
    deployed_env.instance_id = "art-1-abcd1234"
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)

    with patch("agent_env.env.env.Env") as Env, patch(_UPDATE_TARGET) as update:
        Env.get.return_value = _fake_env(deployed_env)
        await step.execute(TaskStepContext())

    update.assert_called_once_with("art-1-abcd1234", {"deploy_step_id": "t-1.deploy_env"})
    assert deployed_env.metadata["deploy_step_id"] == "t-1.deploy_env"


@pytest.mark.asyncio
async def test_execute_copies_no_run_metadata_onto_the_instance_record():
    deployed_env = _deployed_env(None)
    deployed_env.instance_id = "art-1-abcd1234"
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)
    context = TaskStepContext(metadata={"platform": {"caller": "c", "subject": "sub-42"}})

    with patch("agent_env.env.env.Env") as Env, patch(_UPDATE_TARGET) as update:
        Env.get.return_value = _fake_env(deployed_env)
        await step.execute(context)

    _, values = update.call_args.args
    assert values == {"deploy_step_id": "t-1.deploy_env"}
    assert deployed_env.metadata == {"deploy_step_id": "t-1.deploy_env"}


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

    with patch("agent_env.env.env.Env") as Env, patch(_UPDATE_TARGET):
        Env.get.return_value = _fake_env(deployed_env)
        await step.execute(TaskStepContext())

    assert deployed_env.metadata == {
        "deployed_agent_hint": "keep-me",
        "deploy_step_id": "t-1.deploy_env",
    }




# --- preflight: an option the step's env would refuse at deploy is a save-time problem ---


@pytest.mark.parametrize("options, refused", [
    ({"gateway_mode": "consistent"}, "gateway_mode"),
    ({"env_state_type": "remote_postgres"}, "env_state_type"),
    ({"env_state_instance_id": "st-1"}, "env_state_instance_id"),
    ({"gateway_mode": "consistent", "env_state_type": "remote_postgres"}, "gateway_mode, env_state_type"),
], ids=["consistent", "state-type", "state-instance", "both"])
def test_preflight_reports_an_option_a_server_env_would_refuse(options, refused):
    # The same words deploy() raises, so a save-time problem reads like the deploy-time one.
    assert _preflight(_mcp_env("server"), **options) == [
        f"deploy_env 'd': env 'mcp-email' has env_provider_type 'server', which doesn't take {refused}"]


@pytest.mark.parametrize("kind, options", [
    ("server", {"sandbox_type": "modal", "ttl_seconds": 60, "cpu": 2.0, "memory_mb": 4096, "disk_size_gb": 40}),
    ("gateway", {"gateway_mode": "consistent", "env_state_type": "remote_postgres", "env_state_instance_id": "st-1"}),
    ("other", {"gateway_mode": "consistent", "env_state_type": "remote_postgres"}),
], ids=["server-sizing", "gateway-env", "other-env-type"])
def test_preflight_passes_what_the_env_takes(kind, options):
    env = MagicMock() if kind == "other" else _mcp_env(kind)
    assert _preflight(env, **options) == []


@pytest.mark.parametrize("kind", ["server", "gateway", "other"])
@pytest.mark.parametrize("mode", ["bogus", "Performance", None])
def test_preflight_reports_an_invalid_gateway_mode_for_every_env(kind, mode):
    env = MagicMock() if kind == "other" else _mcp_env(kind)
    assert _preflight(env, gateway_mode=mode) == [f"deploy_env 'd': {mode!r} is not a valid GatewayMode"]


def test_preflight_reports_a_missing_env():
    from agent_env.store.base import NotFoundError

    with patch("agent_env.env.env.Env.get", side_effect=NotFoundError("Env gone not found")):
        assert DeployEnvTaskStep(id="d", version=None, env_id="gone").preflight() == ["deploy_env 'd': env 'gone' can't be loaded: Env gone not found"]


@pytest.mark.parametrize("doc", [
    {"id": "e", "version": 1, "type": "coding_task_harbor", "metadata": {}},
    {"id": "e", "version": 1, "type": "mcp_server", "docker_image_artifact": {"id": "img", "version": 1}, "environment_name": "email",
     "env_provider_type": "sidecar", "metadata": {}},
], ids=["unregistered-type", "unknown-env-provider-type"])
def test_preflight_reports_an_env_this_process_cannot_load(doc):
    from agent_env.env.store import get_env_store

    # The store's own loader, so the error is the one a real save meets.
    with patch("agent_env.env.env.Env.get", side_effect=lambda *_: get_env_store()._deserialize(dict(doc))), \
            patch("agent_env.artifact.Artifact.get", return_value=MagicMock()):
        [problem] = DeployEnvTaskStep(id="d", version=None, env_id="e").preflight()
    assert problem.startswith("deploy_env 'd': env 'e' can't be loaded: ")


def test_preflight_lets_infrastructure_errors_propagate():
    with patch("agent_env.env.env.Env.get", side_effect=RuntimeError("store down")), pytest.raises(RuntimeError, match="store down"):
        DeployEnvTaskStep(id="d", version=None, env_id="mcp-email").preflight()


def test_a_task_reports_only_its_refused_deploy_step():
    from agent_env.task import Task

    task = Task(id="t", version=None, steps=[
        DeployEnvTaskStep(id="ok", version=None, env_id="mcp-email"),
        DeployEnvTaskStep(id="bad", version=None, env_id="mcp-email", gateway_mode="consistent"),
    ])
    with patch("agent_env.env.env.Env.get", return_value=_mcp_env("server")):
        assert task.preflight() == ["deploy_env 'bad': env 'mcp-email' has env_provider_type 'server', which doesn't take gateway_mode"]


def _mcp_env(env_provider_type: str):
    from agent_env.env.envs.mcp_server import MCPServerEnv

    return MCPServerEnv(id="mcp-email", version=1, docker_image_artifact=MagicMock(), environment_name="email",
                        env_provider_type=env_provider_type)


def _preflight(env, **options) -> list[str]:
    with patch("agent_env.env.env.Env.get", return_value=env):
        return DeployEnvTaskStep(id="d", version=None, env_id="mcp-email", **options).preflight()
