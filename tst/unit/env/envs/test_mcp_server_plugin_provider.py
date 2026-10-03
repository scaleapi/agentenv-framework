"""A stock MCPServerEnv deploys through a plugin's environment provider: the provider gets every option, its record is
registered as it returns it, and the env holds no sandbox of ours. Loads hand the server a signed URL; reattaching a
plugin's record touches no sandbox and looks up no provider; close() closes the provider that deployed. A MultiEnv
child is still the gateway's, whatever type it declares."""

from __future__ import annotations

import dataclasses
import json
from importlib.metadata import EntryPoint
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_env.config import get_config, reset_config
from agent_env.env.env import DeployedEnv, DeployedSandboxEnv
from agent_env.env.envs._deployment import LOAD_OPERATIONS
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.gateway import GatewayMode
from agent_env.env.gateway.constants import data_plane_load_timeout_s
from agent_env.plugins import _discovery
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.env_providers.env_provider import EnvironmentProvider
from agent_env.providers.env_providers.env_server_provider import EnvironmentServerProvider

_URL = "https://plug.example"
_CARD = {"name": "email", "url": "/agentenv"}
_SIZE = 3_000_000_000  # big enough that the load timeout is above its floor
SEEN: dict = {}


class _KwargsProvider(EnvironmentProvider):
    type = "plugin_kw"

    async def deploy(self, env, sandbox_provider, **options):
        SEEN.setdefault("deploys", []).append((sandbox_provider, options))
        return DeployedEnv(env_id=env.id, env_version=env.version, env_provider_type=self.type,
                           environment_card_url=f"{_URL}/.well-known/agent-env.json", environment_card=_CARD)

    async def close(self):
        SEEN["closed"] = SEEN.get("closed", 0) + 1


class _ServerSubclass(EnvironmentServerProvider):
    type = "plugin_srvsub"


class _EP:
    def __init__(self, name, attr):
        self.name, self.value = name, f"{__name__}:{attr}"
        self.dist = MagicMock(version="1.0", requires=[])
        self.dist.name = "demo"

    def load(self):
        return EntryPoint(self.name, self.value, "unused").load()


@pytest.fixture(autouse=True)
def plugin(monkeypatch):
    by_group = {"agent_env.env_providers": [_EP("plugin_kw", "_KwargsProvider"), _EP("plugin_srvsub", "_ServerSubclass"),
                                            _EP("plugin_named", "_NamedOptionsProvider")]}
    monkeypatch.setattr(_discovery, "entry_points", lambda *, group: by_group.get(group, []))
    reset_config()
    SEEN.clear()
    yield
    reset_config()


@pytest.fixture
def sent(monkeypatch) -> list[httpx.Request]:
    requests, real = [], httpx.AsyncClient

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return requests


def _env(t="plugin_kw") -> MCPServerEnv:
    return MCPServerEnv(id="mcp-email", version=1, docker_image_artifact=MagicMock(), environment_name="email", env_provider_type=t)


async def _deploy(env, **kwargs):
    with patch("agent_env.providers.get_env_sandbox_provider", MagicMock(return_value="SANDBOXES")), \
         patch("agent_env.env.envs._deployment.register_env_instance", side_effect=lambda d, ttl: dataclasses.replace(d, instance_id="inst-1")):
        return await env.deploy(**kwargs)


def _artifact():
    return SimpleNamespace(environment_name="email", get_file_artifact=lambda: SimpleNamespace(
        filename="email.json", object_url="s3://bucket/email.json", content_type="application/json"))


def test_reading_a_stored_env_looks_up_no_environment_provider(monkeypatch):
    monkeypatch.setattr("agent_env.artifact.Artifact.get", classmethod(lambda cls, id, version=None: MagicMock(id=id, version=version)))
    doc = {"id": "mcp-email", "version": 1, "type": "mcp_server", "metadata": {}, "environment_name": "email", "service_version": 1,
           "docker_image_artifact": {"id": "img", "version": 1, "type": "docker_image"}, "env_provider_type": "not_installed_here"}
    with patch.object(type(get_config()), "env_provider_registry", side_effect=AssertionError("looked up")):
        env = MCPServerEnv.from_dict(doc)
    assert (env.env_provider_type, env._env_provider) == ("not_installed_here", None)


@pytest.mark.asyncio
async def test_a_plugin_provider_gets_every_option_and_its_record_is_registered():
    env = _env()
    record = await _deploy(env, ttl_seconds=60, cpu=2.0, priority=1, gateway_mode=GatewayMode.CONSISTENT, attribution={"team": "t"})

    [(sandbox_provider, options)] = SEEN["deploys"]
    assert sandbox_provider == "SANDBOXES"
    assert options == {"ttl_seconds": 60, "disk_size_gb": 10, "gateway_mode": GatewayMode.CONSISTENT, "cpu": 2.0, "memory_mb": None,
                       "priority": 1, "env_state_type": None, "env_state_instance_id": None, "attribution": {"team": "t"}}
    assert (record.instance_id, env._instance_id, env._deployed, env._sandbox, env._gateway_url) == ("inst-1", "inst-1", record, None, None)


@pytest.mark.asyncio
async def test_a_failed_plugin_deploy_closes_the_plugin():
    env = _env()
    with patch.object(_KwargsProvider, "deploy", AsyncMock(side_effect=RuntimeError("boom"))), pytest.raises(RuntimeError, match="boom"):
        await _deploy(env)
    assert SEEN["closed"] == 1


def _store(signed: str | None, size: int | None = _SIZE):
    return MagicMock(signed_get_url=MagicMock(return_value=signed),
                     get_object_metadata_at=MagicMock(return_value=SimpleNamespace(size=size) if size is not None else None))


def _stores(store):
    return patch("agent_env.config.get_config", MagicMock(return_value=MagicMock(get_object_store_at=MagicMock(return_value=store))))


@pytest.mark.asyncio
async def test_a_plugin_env_loads_by_a_signed_url_with_no_staging(sent):
    env = _env()
    await _deploy(env)
    store = _store("https://signed.example/email.json?sig=1")
    with _stores(store):
        await env.load_environment_artifact(_artifact())

    timeout = data_plane_load_timeout_s(_SIZE)
    store.signed_get_url.assert_called_once_with("s3://bucket/email.json", len(LOAD_OPERATIONS) * timeout)  # outlives the reset and the add
    assert [json.loads(r.content)["method"] for r in sent] == ["data/reset", "data/add"]
    assert [str(r.url) for r in sent] == [f"{_URL}/agentenv"] * 2
    assert json.loads(sent[1].content)["params"]["parts"][0]["file"] == {"uri": "https://signed.example/email.json?sig=1", "mimeType": "application/json", "name": "email.json"}


@pytest.mark.asyncio
async def test_a_plugin_env_refuses_a_load_the_store_cannot_sign_before_touching_its_data(sent):
    env = _env()
    await _deploy(env)
    with _stores(_store(None)), pytest.raises(RuntimeError, match="can't sign a URL for it; loading into such an env needs an object store that signs"):
        await env.load_environment_artifact(_artifact())
    assert sent == []


@pytest.mark.asyncio
async def test_a_plugin_env_whose_record_has_no_card_cannot_load(sent):
    env = _env()
    with patch.object(_KwargsProvider, "deploy", AsyncMock(return_value=DeployedEnv(
            env_id="mcp-email", env_version=1, env_provider_type="plugin_kw", mcp_url=f"{_URL}/mcp"))):
        await _deploy(env)
    with _stores(_store("https://signed.example/x")), pytest.raises(RuntimeError, match="carries no env card"):
        await env.load_environment_artifact(_artifact())
    assert sent == []


@pytest.mark.asyncio
async def test_a_plugin_env_refuses_to_stage_files_on_a_host_it_has_not():
    env = _env()
    await _deploy(env)
    with pytest.raises(RuntimeError, match="needs a built-in env provider; env 'mcp-email' was deployed by env_provider_type 'plugin_kw'"):
        await env.load_file_artifact_universe(MagicMock())


@pytest.mark.asyncio
@pytest.mark.parametrize("record", [
    DeployedEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_kw", environment_card_url=f"{_URL}/.well-known/agent-env.json",
                environment_card=_CARD),
    DeployedSandboxEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_kw", sandbox_id="theirs", sandbox_type="not-ours",
                       sandbox_ids={"mcp_server": {"email": "theirs"}}),
], ids=["no-sandbox", "foreign-sandbox"])
@pytest.mark.parametrize("env_provider_type", ["plugin_kw", "not_installed_here"])
async def test_a_plugin_record_reattaches_no_sandbox_and_builds_no_provider(record, env_provider_type):
    with patch("agent_env.env.env.Env.get", return_value=_env(env_provider_type)), \
         patch("agent_env.providers.build_sandbox_provider", side_effect=AssertionError("reattached")):
        env = await MCPServerEnv.from_deployed_env(record)
        await env.close()
    assert (env._sandbox, env._env_provider, env._deployed, env._gateway_url) == (None, None, record, None)
    assert "closed" not in SEEN


@pytest.mark.asyncio
async def test_close_closes_the_plugin_that_deployed():
    env = _env()
    await _deploy(env)
    await env.close()
    assert SEEN["closed"] == 1


@pytest.mark.asyncio
async def test_a_multi_env_child_of_a_plugin_type_is_still_the_gateways():
    child = _env()
    multi = MultiEnv(id="multi", version=1, mcp_server_envs=[child])
    record = SimpleNamespace(env_id="multi", env_version=1, sandbox_id="gw", sandbox_type="modal", sandbox_ids={},
                             env_state_instance_ids=[], instance_id="i", mcp_server_name="m", environment_card=None,
                             environment_card_url=None, gateway_url="https://gw.example")
    gw = MagicMock(sandbox_id="gw", mode="vm", type="modal")
    with patch("agent_env.env.env.Env.get", return_value=multi), \
         patch("agent_env.providers.build_sandbox_provider", return_value=MagicMock(get_sandbox=AsyncMock(return_value=gw))), \
         patch("agent_env.providers.env_state.build_state_provider", MagicMock()):
        await MultiEnv.from_deployed_env(record)
    assert isinstance(child._env_provider, EnvironmentGatewayProvider) and child._sandbox is gw
    child._copy_artifact_into_container = AsyncMock(return_value="/data/email.json")
    child._staged_artifact_size = AsyncMock(return_value=None)
    with patch("agent_env.env.legacy_protocol.v1_base_url", AsyncMock(return_value=None)), \
         patch("agent_env.env.legacy_protocol.reset", AsyncMock()) as legacy_reset, \
         patch.object(EnvironmentGatewayProvider, "install_changelog_triggers", AsyncMock()) as triggers:
        await child.load_environment_artifact(_artifact())
    child._copy_artifact_into_container.assert_awaited_once()
    legacy_reset.assert_awaited_once()
    triggers.assert_awaited_once_with("email")


@pytest.mark.asyncio
async def test_a_plugin_record_with_no_sandbox_passes_the_round_trip_check_and_is_registered():
    env = _env()
    store = MagicMock(create_instance=MagicMock(side_effect=lambda d, ttl_seconds: dataclasses.replace(d, instance_id="inst-9")))
    with patch("agent_env.providers.get_env_sandbox_provider", MagicMock()), \
         patch("agent_env.env.store.get_env_instance_store", return_value=store):
        record = await env.deploy()
    assert type(record) is DeployedEnv and record.instance_id == "inst-9"
    assert type(DeployedEnv.from_dict(dataclasses.asdict(record))) is DeployedEnv


class _NamedOptionsProvider(EnvironmentProvider):
    type = "plugin_named"

    async def deploy(self, env, sandbox_provider, *, ttl_seconds: int, attribution=None):
        SEEN.setdefault("deploys", []).append((sandbox_provider, {"ttl_seconds": ttl_seconds, "attribution": attribution}))
        return DeployedEnv(env_id=env.id, env_version=env.version, env_provider_type=self.type, mcp_url=f"{_URL}/mcp")

    async def close(self):
        SEEN["closed"] = SEEN.get("closed", 0) + 1


@pytest.mark.asyncio
async def test_a_plugin_that_names_its_options_gets_those_and_the_rest_left_at_their_defaults():
    env = _env("plugin_named")
    env._env_provider = _NamedOptionsProvider()
    await _deploy(env, ttl_seconds=60, attribution={"team": "t"})
    assert SEEN["deploys"] == [("SANDBOXES", {"ttl_seconds": 60, "attribution": {"team": "t"}})]


@pytest.mark.asyncio
async def test_a_plugin_that_names_its_options_refuses_one_set_that_it_does_not_name_before_any_spend():
    env = _env("plugin_named")
    env._env_provider = _NamedOptionsProvider()
    with patch("agent_env.providers.get_env_sandbox_provider") as sandboxes, pytest.raises(ValueError, match="doesn't take cpu"):
        await env.deploy(cpu=2.0)
    assert sandboxes.called is False and "deploys" not in SEEN


def _preflight(env, **options) -> list[str]:
    from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep

    with patch("agent_env.env.env.Env.get", return_value=env):
        return DeployEnvTaskStep(id="d", version=None, env_id="mcp-email", **options).preflight()


def test_saving_a_task_passes_every_option_to_a_plugin_that_takes_options_and_builds_no_provider():
    env = _env()
    assert _preflight(env, gateway_mode="consistent", env_state_type="remote_postgres", cpu=2.0) == []
    assert env._env_provider is None


def test_saving_a_task_reports_an_option_a_plugin_that_names_its_options_does_not_take():
    assert _preflight(_env("plugin_named"), cpu=2.0) == ["deploy_env 'd': env 'mcp-email' has env_provider_type 'plugin_named', which doesn't "
                                                         "take cpu; name it in the provider's deploy() or take **options"]


@pytest.mark.asyncio
async def test_a_plugin_record_with_no_mcp_url_fails_the_deploy_and_closes_the_plugin():
    env = _env()
    with patch.object(_KwargsProvider, "deploy", AsyncMock(return_value=DeployedEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_kw"))), \
         pytest.raises(RuntimeError, match="returned a record with no MCP URL"):
        await _deploy(env)
    assert SEEN["closed"] == 1 and env._deployed is None


@pytest.mark.asyncio
async def test_deploy_env_deploys_a_stock_env_through_the_plugin_and_the_record_reattaches():
    from agent_env.task_step.context import TaskStepContext
    from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep

    registered = {}

    def register(record, ttl_seconds):
        registered["record"] = dataclasses.replace(record, instance_id="inst-7")
        return registered["record"]

    context = TaskStepContext()
    with patch("agent_env.env.env.Env.get", return_value=_env()), \
         patch("agent_env.providers.get_env_sandbox_provider", MagicMock(return_value="SANDBOXES")), \
         patch("agent_env.env.envs._deployment.register_env_instance", side_effect=register):
        await DeployEnvTaskStep(id="deploy", version=None, env_id="mcp-email").execute(context)

    [deployed] = context.deployed_envs
    assert deployed.instance_id == "inst-7" and deployed.mcp_url == f"{_URL}/mcp"
    [(_, options)] = SEEN["deploys"]
    assert options["attribution"] is not None  # deploy_env always attributes its deploys

    store = MagicMock(get=MagicMock(return_value=registered["record"]))
    with patch("agent_env.env.store.get_env_instance_store", return_value=store), \
         patch("agent_env.env.env.Env.get", return_value=_env()), \
         patch("agent_env.providers.build_sandbox_provider", side_effect=AssertionError("reattached a sandbox")):
        env = await MCPServerEnv.from_instance_id("inst-7")
    assert (env._deployed, env._env_provider, env._sandbox, env._instance_id) == (registered["record"], None, None, "inst-7")


@pytest.mark.asyncio
async def test_validate_cleanup_survives_a_record_without_a_sandbox():
    record = DeployedEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_kw", mcp_url=f"{_URL}/mcp", instance_id="inst-1")
    context = SimpleNamespace(deployed_envs=[record], deployed_agents=[], instance_id="task-inst")
    with patch("agent_env.task.Task.put", return_value=MagicMock(id="t", version=1, run=AsyncMock(return_value=context))), \
         patch.object(MCPServerEnv, "from_deployed_env", AsyncMock(side_effect=RuntimeError("gone"))):
        assert await _env().validate() == "task-inst"


def _put(tmp_path, *flags):
    from click.testing import CliRunner
    from agent_env.cli import cli

    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM scratch\n")
    put = MagicMock(return_value=MagicMock(id="x", version=1, environment_name="email"))
    with patch("agent_env.cli.env.mcp_server.build_image"), \
         patch("agent_env.cli.env.mcp_server.DockerImageArtifact.put", return_value=MagicMock(id="a", version=1)), \
         patch("agent_env.cli.env.mcp_server.detect_env_metadata", return_value={}), \
         patch("agent_env.cli.env.mcp_server.MCPServerEnv.put", put):
        result = CliRunner().invoke(cli, ["env", "mcp-server", "put", "--id", "x", "--dockerfile", str(dockerfile),
                                          "--environment-name", "email", *flags])
    return result, put


def test_put_stores_the_env_provider_type_it_is_given(tmp_path):
    result, put = _put(tmp_path, "--env-provider-type", "plugin_kw")
    assert result.exit_code == 0, result.output
    assert put.call_args.kwargs["env_provider_type"] == "plugin_kw"


def test_put_defaults_to_the_gateway(tmp_path):
    result, put = _put(tmp_path)
    assert result.exit_code == 0, result.output
    assert put.call_args.kwargs["env_provider_type"] == "gateway"


def test_put_refuses_an_env_provider_type_nothing_registers_before_building(tmp_path):
    result, put = _put(tmp_path, "--env-provider-type", "not_installed")
    assert result.exit_code == 2 and "Unknown env_provider_type: 'not_installed'" in result.output
    assert put.called is False


@pytest.mark.parametrize(("env_provider_type", "provider_type"), [("gateway", "gateway"), ("server", "server"), ("plugin_kw", None)])
def test_a_built_ins_provider_is_built_with_the_env_and_a_plugins_is_not(env_provider_type, provider_type):
    with patch.object(type(get_config()), "env_provider_registry", side_effect=AssertionError("looked up")):
        env = _env(env_provider_type)
    assert getattr(env._env_provider, "type", None) == provider_type


@pytest.mark.asyncio
async def test_an_env_wired_up_by_hand_loads_through_its_built_in_provider_as_before(sent):
    """As callers outside agent-env do: Env.get, then _sandbox and _gateway_url set, then a load."""
    env = _env("gateway")
    env._sandbox, env._gateway_url = MagicMock(mode="vm"), "https://gw.example"
    env._copy_artifact_into_container = AsyncMock(return_value="/data/email.json")
    env._staged_artifact_size = AsyncMock(return_value=None)
    with patch("agent_env.env.legacy_protocol.v1_base_url", AsyncMock(return_value="https://gw.example/svc/email")), \
         patch.object(EnvironmentGatewayProvider, "install_changelog_triggers", AsyncMock()) as triggers:
        await env.load_environment_artifact(_artifact())
    assert json.loads(sent[1].content)["params"]["parts"][0]["file"]["uri"] == "file:///data/email.json"
    triggers.assert_awaited_once_with("email")


@pytest.mark.asyncio
async def test_a_plugin_that_subclasses_a_built_in_reattaches_like_one():
    record = DeployedSandboxEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_srvsub", sandbox_id="sb", sandbox_type="local",
                                sandbox_ids={"mcp_server": {"email": "sb"}})
    with patch("agent_env.env.env.Env.get", return_value=_env("plugin_srvsub")), \
         patch.object(_ServerSubclass, "_reattach", AsyncMock(return_value="THE-CONTAINER")) as reattach:
        env = await MCPServerEnv.from_deployed_env(record)
    assert isinstance(env._env_provider, _ServerSubclass) and env._sandbox == "THE-CONTAINER"
    reattach.assert_awaited_once_with(env, record)


@pytest.mark.asyncio
@pytest.mark.parametrize(("record", "message"), [
    (DeployedEnv(env_id="mcp-email", env_version=1, environment_card_url=f"{_URL}/.well-known/agent-env.json", environment_card=_CARD),
     "returned a record with env_provider_type None; set it to 'plugin_kw'"),
    (DeployedEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_kw", environment_card_url=f"{_URL}/.well-known/agent-env.json",
                 environment_card={"name": "other", "url": "/agentenv"}),
     "whose env card is named 'other'; it must be the env's environment_name 'email'"),
], ids=["no-type", "card-of-another-env"])
async def test_a_plugin_record_the_env_could_not_use_fails_the_deploy_and_closes_the_plugin(record, message):
    env = _env()
    with patch.object(_KwargsProvider, "deploy", AsyncMock(return_value=record)), pytest.raises(RuntimeError, match=message):
        await _deploy(env)
    assert SEEN["closed"] == 1 and env._deployed is None


@pytest.mark.asyncio
async def test_a_load_of_an_object_that_is_gone_leaves_the_envs_data_alone(sent):
    from agent_env.store.base import ObjectNotFoundError

    env = _env()
    await _deploy(env)
    store = _store("https://signed.example/x", size=None)
    with _stores(store), pytest.raises(ObjectNotFoundError, match="No object at s3://bucket/email.json"):
        await env.load_environment_artifact(_artifact())
    assert sent == [] and store.signed_get_url.called is False


@pytest.mark.asyncio
async def test_a_load_into_a_server_whose_card_lists_the_protocol_sdks_operations_goes_through(sent):
    """The protocol SDK lists a server's operations by their wire methods."""
    env = _env()
    card = {"name": "email", "url": "/agentenv", "capabilities": {"operations": ["data/reset", "data/add", "data/get"]}}
    with patch.object(_KwargsProvider, "deploy", AsyncMock(return_value=DeployedEnv(
            env_id="mcp-email", env_version=1, env_provider_type="plugin_kw",
            environment_card_url=f"{_URL}/.well-known/agent-env.json", environment_card=card))):
        await _deploy(env)
    with _stores(_store("https://signed.example/email.json?sig=1")):
        await env.load_environment_artifact(_artifact())
    assert [json.loads(r.content)["method"] for r in sent] == ["data/reset", "data/add"]


@pytest.mark.asyncio
async def test_a_load_into_a_server_whose_card_lists_no_data_plane_is_refused_before_signing(sent):
    env = _env()
    card = {"name": "email", "url": "/agentenv", "capabilities": {"operations": ["data/get"]}}
    with patch.object(_KwargsProvider, "deploy", AsyncMock(return_value=DeployedEnv(
            env_id="mcp-email", env_version=1, env_provider_type="plugin_kw",
            environment_card_url=f"{_URL}/.well-known/agent-env.json", environment_card=card))):
        await _deploy(env)
    store = _store("https://signed.example/x")
    with _stores(store), pytest.raises(RuntimeError, match="offers 'email' no data/reset, data/add operation to load with"):
        await env.load_environment_artifact(_artifact())
    assert sent == [] and store.signed_get_url.called is False


@pytest.mark.asyncio
async def test_a_card_that_leaves_out_its_default_url_still_gives_its_data_plane():
    from agent_env.env import legacy_protocol

    record = DeployedEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_kw",
                         environment_card_url=f"{_URL}/.well-known/agent-env.json", environment_card={"name": "email"})
    assert await legacy_protocol.v1_base_url(record, None, "email") == _URL


@pytest.mark.asyncio
async def test_validate_cleanup_terminates_the_sandboxes_a_plugins_record_names():
    record = DeployedSandboxEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_kw", mcp_url=f"{_URL}/mcp",
                                sandbox_id="sb-1", sandbox_type="local")
    context = SimpleNamespace(deployed_envs=[record], deployed_agents=[], instance_id="task-inst")
    with patch("agent_env.task.Task.put", return_value=MagicMock(id="t", version=1, run=AsyncMock(return_value=context))), \
         patch("agent_env.env.env.Env.get", return_value=_env()), \
         patch("agent_env.task_step.task_steps.teardown_sandboxes._terminate", AsyncMock()) as terminate:
        assert await _env().validate() == "task-inst"
    terminate.assert_awaited_once_with("sb-1", "local", "env")


@pytest.mark.asyncio
async def test_validate_cleanup_says_it_cannot_close_a_plugin_deployment_outside_our_sandboxes(caplog):
    record = DeployedEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_kw", mcp_url=f"{_URL}/mcp")
    context = SimpleNamespace(deployed_envs=[record], deployed_agents=[], instance_id="task-inst")
    with patch("agent_env.task.Task.put", return_value=MagicMock(id="t", version=1, run=AsyncMock(return_value=context))), \
         patch("agent_env.env.env.Env.get", return_value=_env()), caplog.at_level("WARNING"):
        await _env().validate()
    assert "outside agent-env's sandboxes, so it isn't closed here" in caplog.text


def test_the_cli_load_prints_a_refusal_as_an_error_rather_than_a_traceback():
    from click.testing import CliRunner
    from agent_env.cli import cli

    env = MagicMock(id="mcp-email", version=1, environment_name="email",
                    load_environment_artifact=AsyncMock(side_effect=RuntimeError("can't sign a URL for it")))
    with patch("agent_env.cli.env.mcp_server.deployed_env_from_instance", return_value=env), \
         patch("agent_env.cli.env.mcp_server.EnvironmentArtifact.get", return_value=MagicMock(id="ea", version=1, environment_name="email")):
        result = CliRunner().invoke(cli, ["env", "mcp-server", "load-environment-artifact", "--environment-artifact-id", "ea",
                                          "--instance-id", "inst-1"])
    assert result.exit_code == 1 and "Error: can't sign a URL for it" in result.output and "Traceback" not in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize(("record", "message"), [
    (DeployedEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_kw", mcp_url=f"{_URL}/mcp", environment_card=_CARD),
     "with no environment_card_url; an env card and its environment_card_url come together"),
    (DeployedEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_kw", mcp_url=f"{_URL}/mcp",
                 environment_card_url=f"{_URL}/.well-known/agent-env.json"),
     "with no environment_card; an env card and its environment_card_url come together"),
], ids=["card-without-url", "url-without-card"])
async def test_a_plugin_record_with_half_an_env_card_fails_the_deploy(record, message):
    env = _env()
    with patch.object(_KwargsProvider, "deploy", AsyncMock(return_value=record)), pytest.raises(RuntimeError, match=message):
        await _deploy(env)
    assert SEEN["closed"] == 1


@pytest.mark.asyncio
async def test_validate_cleanup_terminates_every_sandbox_even_when_one_is_already_gone(caplog):
    record = DeployedSandboxEnv(env_id="mcp-email", env_version=1, env_provider_type="plugin_kw", mcp_url=f"{_URL}/mcp",
                                sandbox_id="sb-gone", sandbox_type="local", sandbox_ids={"mcp_server": {"email": "sb-2"}})
    context = SimpleNamespace(deployed_envs=[record], deployed_agents=[], instance_id="task-inst")
    terminate = AsyncMock(side_effect=lambda sandbox_id, *_: (_ for _ in ()).throw(RuntimeError("gone")) if sandbox_id == "sb-gone" else None)
    with patch("agent_env.task.Task.put", return_value=MagicMock(id="t", version=1, run=AsyncMock(return_value=context))), \
         patch("agent_env.env.env.Env.get", return_value=_env()), \
         patch("agent_env.task_step.task_steps.teardown_sandboxes._terminate", terminate), caplog.at_level("WARNING"):
        await _env().validate()
    assert sorted(c.args[0] for c in terminate.await_args_list) == ["sb-2", "sb-gone"]
    assert "terminate sb-gone failed" in caplog.text
