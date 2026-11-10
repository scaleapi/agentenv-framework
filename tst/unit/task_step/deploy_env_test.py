"""Unit tests for DeployEnvTaskStep agent registration on deploy()."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.env import DeployedGatewayEnv, Env
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


def _fake_env(deployed_env, live_websites=()):
    env = MagicMock()
    env.deploy = AsyncMock(return_value=deployed_env)
    env.live_websites.return_value = list(live_websites)
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
async def test_a_run_that_deploys_a_live_website_is_marked_live_and_not_reproducible():
    deployed_env = _deployed_env(None)
    deployed_env.instance_id = "art-1-abcd1234"
    live = [{"env_id": "github-site", "environment_name": "github", "domains": ["github.com"], "allow_hosts": []}]
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)

    with patch("agent_env.env.env.Env") as Env, patch(_UPDATE_TARGET) as update:
        Env.get.return_value = _fake_env(deployed_env, live)
        context = await step.execute(TaskStepContext())

    update.assert_called_once_with("art-1-abcd1234", {"deploy_step_id": "t-1.deploy_env", "live": True})
    assert (context.metadata["live_websites"], context.metadata["reproducible"]) == ({"art-1": live}, False)


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
    env = _OtherEnv("other", 1) if kind == "other" else _mcp_env(kind)
    assert _preflight(env, **options) == []


@pytest.mark.parametrize("kind", ["server", "gateway", "other"])
def test_preflight_refuses_env_params_an_env_doesnt_take(kind):
    env = _OtherEnv("other", 1) if kind == "other" else _mcp_env(kind)
    assert _preflight(env, env_params={"form": "container"}) == [f"deploy_env 'd': {env.type} envs take no env_params"]


class _TakesEverything:
    """A plugin provider whose deploy() takes every option."""

    async def deploy(self, env, sandbox_provider, **options): ...


@pytest.mark.parametrize("option, refusal", [
    ({"env_params": {"form": "container"}}, "mcp_server envs take no env_params"),
    ({"child_envs": {"desktop": {"cpu": 4}}}, "mcp_server envs have no child envs"),
], ids=["env-params", "child-envs"])
@pytest.mark.asyncio
async def test_what_a_run_refuses_behind_a_provider_that_takes_every_option_is_refused_at_save_too(option, refusal):
    step = DeployEnvTaskStep(id="d", version=None, env_id="mcp-email", **option)
    with patch("agent_env.env.envs.mcp_server.provider_or_class", return_value=_TakesEverything), \
            patch("agent_env.env.env.Env.get", return_value=_mcp_env("plugin_kw")):
        assert step.preflight() == [f"deploy_env 'd': {refusal}"]
        with pytest.raises(ValueError, match=refusal):
            await step.execute(TaskStepContext())


@pytest.mark.parametrize("kind", ["server", "gateway", "other"])
@pytest.mark.parametrize("mode", ["bogus", "Performance", None])
def test_preflight_reports_an_invalid_gateway_mode_for_every_env(kind, mode):
    env = _OtherEnv("other", 1) if kind == "other" else _mcp_env(kind)
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


class _OtherEnv(Env):
    """An env type that deploys itself, through no provider, and checks none of its options."""

    type = "other"


def _mcp_env(env_provider_type: str):
    from agent_env.env.envs.mcp_server import MCPServerEnv

    return MCPServerEnv(id="mcp-email", version=1, docker_image_artifact=MagicMock(), environment_name="email",
                        env_provider_type=env_provider_type)


def _preflight(env, **options) -> list[str]:
    with patch("agent_env.env.env.Env.get", return_value=env):
        return DeployEnvTaskStep(id="d", version=None, env_id="mcp-email", **options).preflight()


def _env_with_fixed_deploy(deployed_env):
    """An env whose deploy() takes a fixed param list, like every built-in env."""
    env = MagicMock()
    env.live_websites.return_value = []

    async def deploy(ttl_seconds=10800, disk_size_gb=10, gateway_mode=None, cpu=None,
                     memory_mb=None, sandbox_type=None, priority=None, env_state_type=None,
                     env_state_instance_id=None, *, attribution=None):
        return deployed_env

    env.deploy = deploy
    return env


@pytest.mark.asyncio
async def test_artifact_is_forwarded_to_an_env_that_accepts_it():
    deployed_env = _deployed_env(None)
    env = _fake_env(deployed_env)  # AsyncMock: accepts **kwargs
    step = DeployEnvTaskStep(
        id="t-1.deploy_env", version=None, env_id="art-1", env_version=2,
        artifact_id="hg4_real", artifact_version=5,
    )

    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(TaskStepContext())

    kwargs = env.deploy.await_args.kwargs
    assert kwargs["artifact_id"] == "hg4_real"
    assert kwargs["artifact_version"] == 5


@pytest.mark.asyncio
async def test_artifact_is_omitted_for_an_env_that_does_not_accept_it():
    """Built-in envs take a fixed param list and would TypeError on an unexpected kwarg."""
    deployed_env = _deployed_env(None)
    env = _env_with_fixed_deploy(deployed_env)
    step = DeployEnvTaskStep(
        id="t-1.deploy_env", version=None, env_id="art-1", env_version=2,
        artifact_id="hg4_real", artifact_version=5,
    )

    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        result = await step.execute(TaskStepContext())  # must not raise

    assert len(result.deployed_envs) == 1


@pytest.mark.asyncio
async def test_artifact_defaults_to_absent():
    """A step that names no artifact passes none, so existing taxonomies are unaffected."""
    deployed_env = _deployed_env(None)
    env = _fake_env(deployed_env)
    step = DeployEnvTaskStep(id="t-1.deploy_env", version=None, env_id="art-1", env_version=2)

    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        await step.execute(TaskStepContext())

    assert "artifact_id" not in env.deploy.await_args.kwargs


def test_artifact_survives_a_dict_round_trip():
    step = DeployEnvTaskStep(
        id="t-1.deploy_env", version=None, env_id="art-1", env_version=2,
        artifact_id="hg4_real", artifact_version=5,
    )
    restored = DeployEnvTaskStep.from_dict(step.to_dict())
    assert restored.artifact_id == "hg4_real"
    assert restored.artifact_version == 5


@pytest.mark.asyncio
async def test_an_env_taking_only_the_id_is_not_sent_the_version():
    """Each field is gated on its own name: an env that declares one and not the other
    would otherwise get an unexpected keyword."""
    deployed_env = _deployed_env(None)
    env = MagicMock()
    seen = {}

    async def deploy(artifact_id=None, **kw):  # declares the id, not the version
        seen["artifact_id"] = artifact_id
        return deployed_env

    # a fixed list that names artifact_id only, so accepts_kwargs is False
    async def fixed_deploy(ttl_seconds=10800, disk_size_gb=10, gateway_mode=None, cpu=None,
                           memory_mb=None, sandbox_type=None, env_state_type=None,
                           env_state_instance_id=None, artifact_id=None, *, attribution=None):
        seen["artifact_id"] = artifact_id
        return deployed_env

    env.deploy = fixed_deploy
    step = DeployEnvTaskStep(
        id="t-1.deploy_env", version=None, env_id="art-1", env_version=2,
        artifact_id="hg4_real", artifact_version=5,
    )

    with patch("agent_env.env.env.Env") as Env:
        Env.get.return_value = env
        result = await step.execute(TaskStepContext())  # must not raise

    assert seen["artifact_id"] == "hg4_real" and len(result.deployed_envs) == 1


def test_deployment_settings_keep_their_positions():
    """The artifact params are appended, so a positional caller's values still mean what
    they did: the fifth positional is still ttl_seconds."""
    step = DeployEnvTaskStep("t-1.deploy_env", None, "art-1", 2, 60)
    assert (step.ttl_seconds, step.artifact_id, step.artifact_version) == (60, None, None)


# --- child_envs: each child env's options by its name, which only an env with child envs takes ---


def _multi_env():
    from agent_env.env.envs.multi_env import MultiEnv
    from agent_env.env.envs.website import WebsiteEnv

    shop = WebsiteEnv(id="web-shop", version=1, backend_docker_image_artifact=MagicMock(),
                      frontend_docker_image_artifact=MagicMock(), environment_name="shop")
    return MultiEnv(id="office", version=1, mcp_server_envs=[_mcp_env("gateway")], website_envs=[shop])


def test_child_envs_round_trip_and_are_written_only_when_set():
    entries = {"desktop": {"cpu": 4, "env_params": {"form": "container"}}}
    step = DeployEnvTaskStep(id="d", version=None, env_id="office", child_envs=entries)
    assert DeployEnvTaskStep.from_dict(step.to_dict()).child_envs == entries
    assert "child_envs" not in DeployEnvTaskStep(id="d", version=None, env_id="office").to_dict()


def test_writing_a_step_out_for_elsewhere_drops_each_child_envs_sandbox_type_as_it_drops_the_steps():
    from agent_env.entity.scopes import FieldScope, prune

    data = DeployEnvTaskStep(id="d", version=None, env_id="office", sandbox_type="modal",
                             child_envs={"desktop": {"sandbox_type": "modal_vm", "cpu": 4}}).to_dict()
    left = prune(data, DeployEnvTaskStep.field_scopes, frozenset({FieldScope.TASK}))
    assert "sandbox_type" not in data and data["child_envs"] == {"desktop": {"cpu": 4}}
    assert {where for where, _ in left} == {"sandbox_type", "child_envs.desktop.sandbox_type"}
    assert prune({"child_envs": ["desktop"]}, DeployEnvTaskStep.field_scopes, frozenset({FieldScope.TASK})) == []


@pytest.mark.parametrize("kind", ["server", "gateway", "other"])
def test_preflight_refuses_child_envs_for_an_env_with_no_child_envs(kind):
    env = _OtherEnv("other", 1) if kind == "other" else _mcp_env(kind)
    assert _preflight(env, child_envs={"desktop": {"cpu": 4}}) == [f"deploy_env 'd': {env.type} envs have no child envs"]


def test_exporting_a_step_leaves_the_steps_own_child_envs_as_they_were():
    from agent_env.entity.scopes import FieldScope, prune

    step = DeployEnvTaskStep(id="d", version=None, env_id="office", child_envs={"desktop": {"sandbox_type": "modal_vm"}})
    prune(step.to_dict(), DeployEnvTaskStep.field_scopes, frozenset({FieldScope.TASK}))
    assert step.child_envs == {"desktop": {"sandbox_type": "modal_vm"}}


_ENTRY = "an entry is a table of sandbox_type, cpu, memory_mb, disk_size_gb, env_params, not"
_SHARED = "runs on the multi env's own sandbox, which the step's own fields set, so it takes no entry"


@pytest.mark.parametrize("child_envs, problem", [
    ({"desktop": {"cpu": 4}}, "child_envs.desktop: env 'office' has no child env named 'desktop'; its child envs are email, shop"),
    ({"shop": {"cpu": 4}}, f"child_envs.shop: shop {_SHARED}"),
    ({"email": {"env_params": {"form": "container"}}}, f"child_envs.email: email {_SHARED}"),
    ({"email": {"ttl_seconds": 60, "gateway_mode": "consistent"}}, f"child_envs.email: {_ENTRY} gateway_mode, ttl_seconds"),
    ({"email": "big"}, f"child_envs.email: {_ENTRY} a str"),
    ({"email": {1: "a", "b": 2}}, f"child_envs.email: {_ENTRY} 1, b"),
    ({"email": {}, "shop": {}}, f"child_envs.email: email {_SHARED}; child_envs.shop: shop {_SHARED}"),
    (["email"], "child_envs is a table of child env names, not list"),
], ids=["unknown-name", "website", "mcp-server-params", "whole-env-fields", "not-a-table", "keys-of-two-types", "each-named",
     "not-a-table-of-names"])
def test_preflight_names_the_child_env_each_problem_is_with(child_envs, problem):
    assert _preflight(_multi_env(), child_envs=child_envs) == [f"deploy_env 'd': {problem}"]


@pytest.mark.asyncio
async def test_a_multi_env_deploy_refuses_child_envs_before_it_deploys_anything():
    with patch("agent_env.env.envs.multi_env.deploy_through_provider") as deploy, \
            pytest.raises(ValueError, match=f"child_envs.shop: shop {_SHARED}"):
        await _multi_env().deploy(child_envs={"shop": {"cpu": 4}})
    deploy.assert_not_called()


@pytest.mark.asyncio
async def test_execute_passes_child_envs_to_a_deploy_that_takes_them():
    entries = {"desktop": {"cpu": 4, "env_params": {"form": "container"}}}
    step = DeployEnvTaskStep(id="d", version=None, env_id="art-1", env_version=2, child_envs=entries)
    with patch("agent_env.env.env.Env") as Env, patch(_UPDATE_TARGET):
        Env.get.return_value = _fake_env(_deployed_env(None))
        await step.execute(TaskStepContext())
    assert Env.get.return_value.deploy.call_args.kwargs["child_envs"] == entries


@pytest.mark.asyncio
async def test_execute_refuses_child_envs_for_an_env_whose_deploy_takes_none():
    step = DeployEnvTaskStep(id="d", version=None, env_id="art-1", child_envs={"desktop": {"cpu": 4}})
    env = _env_with_fixed_deploy(_deployed_env(None))
    with patch("agent_env.env.env.Env") as Env, pytest.raises(ValueError, match="have no child envs"):
        Env.get.return_value = env
        await step.execute(TaskStepContext())
