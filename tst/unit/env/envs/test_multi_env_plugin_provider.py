"""A stock WebsiteEnv or MultiEnv deploys through a plugin's environment provider, as an MCPServerEnv does: the provider gets every
option and the whole deployment, the record is checked and registered, its children load through the card by signed URL, and a
reattach touches no sandbox. What needs our gateway or its host is refused with the reason before anything is built or loaded."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from importlib.metadata import EntryPoint
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from agent_env.cli.env.multi import multi
from agent_env.cli.env.website import website
from agent_env.config import reset_config
from agent_env.env.env import DeployedEnv, DeployedSandboxEnv
from agent_env.env.envs._deployment import close_replaced, host_staging_refusal
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.snapshot_store import EnvSnapshot
from agent_env.plugins import _discovery
from agent_env.providers.env_providers import env_gateway_provider
from agent_env.providers.env_providers.env_gateway_provider import DeployedGateway, EnvironmentGatewayProvider
from agent_env.providers.env_providers.env_provider import EnvironmentProvider
from agent_env.providers.env_providers.env_server_provider import EnvironmentServerProvider
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep

_URL = "https://pods.example"
SEEN: dict = {}


def _card_for(env) -> dict:
    """A composed card with a child per server and website, or a website's own card."""
    if isinstance(env, MultiEnv):
        children = [*env.mcp_server_envs, *env.website_envs]
        return {"name": env.name or "suite", "children_environments": [
            {"name": c.environment_name, "url": f"/svc/{c.environment_name}/agentenv"} for c in children]}
    return {"name": env.environment_name, "url": "/agentenv"}


class _PodProvider(EnvironmentProvider):
    type = "plugin_pods"

    async def deploy(self, env, sandbox_provider, **options):
        SEEN.setdefault("deploys", []).append((env, options))
        return DeployedEnv(env_id=env.id, env_version=env.version, env_provider_type=self.type,
                           environment_card_url=f"{_URL}/.well-known/agent-env.json", environment_card=_card_for(env))

    async def close(self):
        SEEN["closed"] = SEEN.get("closed", 0) + 1


class _NamedOptionsProvider(_PodProvider):
    type = "plugin_named"

    async def deploy(self, env, sandbox_provider, *, ttl_seconds: int, attribution=None):
        return await super().deploy(env, sandbox_provider, ttl_seconds=ttl_seconds, attribution=attribution)


class _TTLOnlyProvider(_PodProvider):
    type = "plugin_ttl"

    async def deploy(self, env, sandbox_provider, *, ttl_seconds: int):
        return await super().deploy(env, sandbox_provider, ttl_seconds=ttl_seconds)


class _GatewaySubclass(EnvironmentGatewayProvider):
    type = "plugin_gwsub"

    async def deploy(self, env, sandbox_provider, **options):
        return DeployedEnv(env_id=env.id, env_version=env.version, env_provider_type=self.type, mcp_url=f"{_URL}/mcp")


class _EP:
    def __init__(self, name, attr):
        self.name, self.value = name, f"{__name__}:{attr}"
        self.dist = MagicMock(version="1.0", requires=[])
        self.dist.name = "demo"

    def load(self):
        return EntryPoint(self.name, self.value, "unused").load()


@pytest.fixture(autouse=True)
def plugin(monkeypatch):
    by_group = {"agent_env.env_providers": [_EP("plugin_pods", "_PodProvider"), _EP("plugin_named", "_NamedOptionsProvider"),
                                            _EP("plugin_ttl", "_TTLOnlyProvider"), _EP("plugin_gwsub", "_GatewaySubclass")]}
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


def _mcp(name: str) -> MCPServerEnv:
    return MCPServerEnv(id=f"mcp-{name}", version=1, docker_image_artifact=MagicMock(image_name=f"mcp-{name}"), environment_name=name)


def _site(name: str, t: str = "gateway") -> WebsiteEnv:
    return WebsiteEnv(id=f"web-{name}", version=1, backend_docker_image_artifact=MagicMock(image_name=f"{name}-be"),
                      frontend_docker_image_artifact=MagicMock(image_name=f"{name}-fe"), environment_name=name, env_provider_type=t)


def _multi(t: str = "plugin_pods", servers=("slack",), websites=("shop",)) -> MultiEnv:
    return MultiEnv(id="crm-suite", version=2, mcp_server_envs=[_mcp(n) for n in servers], website_envs=[_site(n) for n in websites],
                    name="crm", env_provider_type=t)


async def _deploy(env, **kwargs):
    with patch("agent_env.providers.get_env_sandbox_provider", MagicMock(return_value="SANDBOXES")), \
         patch("agent_env.env.envs._deployment.register_env_instance", side_effect=lambda d, ttl: dataclasses.replace(d, instance_id="inst-1")):
        return await env.deploy(**kwargs)


def _artifact(name: str):
    return SimpleNamespace(environment_name=name, get_file_artifact=lambda: SimpleNamespace(
        filename=f"{name}.json", object_url=f"s3://bucket/{name}.json", content_type="application/json"))


def _universe(*names: str, metadata=None):
    return SimpleNamespace(id="u", version=1, get_environment_artifacts=lambda: [_artifact(n) for n in names],
                           get_metadata=lambda: metadata or {})


def _signing_store():
    store = MagicMock(signed_get_url=MagicMock(side_effect=lambda url, ttl: f"https://signed.example/{url.rsplit('/', 1)[1]}"),
                      get_object_metadata_at=MagicMock(return_value=SimpleNamespace(size=1000)))
    return patch("agent_env.config.get_config", MagicMock(return_value=MagicMock(get_object_store_at=MagicMock(return_value=store))))


@pytest.mark.asyncio
async def test_deploy_env_deploys_a_stock_multi_env_through_the_plugin_and_the_record_reattaches():
    registered = {}

    def register(record, ttl_seconds):
        registered["record"] = dataclasses.replace(record, instance_id="inst-7")
        return registered["record"]

    env = _multi()
    context = TaskStepContext()
    with patch("agent_env.env.env.Env.get", return_value=env), \
         patch("agent_env.providers.get_env_sandbox_provider", MagicMock(return_value="SANDBOXES")), \
         patch("agent_env.env.envs._deployment.register_env_instance", side_effect=register):
        await DeployEnvTaskStep(id="deploy", version=None, env_id="crm-suite").execute(context)

    [deployed] = context.deployed_envs
    [(deployed_env, options)] = SEEN["deploys"]
    assert deployed_env is env and set(options) == {"ttl_seconds", "disk_size_gb", "gateway_mode", "cpu", "memory_mb",
                                                     "env_state_type", "env_state_instance_id", "attribution",
                                                     "artifact_id", "artifact_version"}
    assert (deployed.instance_id, deployed.mcp_url, env._sandbox, env._deployed) == ("inst-7", f"{_URL}/mcp", None, deployed)
    for child in [*env.mcp_server_envs, *env.website_envs]:
        assert (child._env_provider, child._deployed, child._sandbox) == (env._env_provider, deployed, None)

    store = MagicMock(get=MagicMock(return_value=registered["record"]))
    with patch("agent_env.env.store.get_env_instance_store", return_value=store), \
         patch("agent_env.env.env.Env.get", return_value=_multi()), \
         patch("agent_env.providers.build_sandbox_provider", side_effect=AssertionError("reattached a sandbox")):
        again = await MultiEnv.from_instance_id("inst-7")
    assert (again._deployed, again._env_provider, again._sandbox, again._instance_id) == (registered["record"], None, None, "inst-7")
    assert [child._deployed for child in [*again.mcp_server_envs, *again.website_envs]] == [registered["record"]] * 2


@pytest.mark.asyncio
async def test_a_plugin_multi_env_loads_a_universe_service_by_service_by_signed_url(sent, caplog):
    env = _multi()
    await _deploy(env)
    with _signing_store(), patch("agent_env.env.store.get_env_instance_store"), \
         patch("agent_env.env.store.update_env_instance_environment_universe"), \
         patch("agent_env.env.snapshot_store.get_env_snapshot_store", side_effect=AssertionError("looked for a snapshot")):
        result = await env.load_environment_universe_artifact(_universe("slack", "shop"))

    calls = sorted((str(r.url), json.loads(r.content)["method"]) for r in sent)
    assert calls == sorted([(f"{_URL}/svc/{name}/agentenv", method) for name in ("slack", "shop") for method in ("data/reset", "data/add")])
    added = {json.loads(r.content)["params"]["parts"][0]["file"]["uri"] for r in sent if json.loads(r.content)["method"] == "data/add"}
    assert added == {"https://signed.example/slack.json", "https://signed.example/shop.json"}
    assert (result.restored_from_snapshot, result.metadata_filepaths) == (False, {})
    assert "Bake a snapshot" not in caplog.text  # no snapshot can restore a plugin's deployment


@pytest.mark.asyncio
async def test_a_plugin_multi_env_refuses_a_universe_with_metadata_files_before_loading_any_service(sent):
    env = _multi()
    await _deploy(env)
    with pytest.raises(RuntimeError, match="Staging a universe's metadata files onto the env's host needs a built-in env provider; "
                                           "env 'crm-suite' was deployed by env_provider_type 'plugin_pods'"):
        await env.load_environment_universe_artifact(_universe("slack", metadata={"config": MagicMock()}))
    assert sent == []


@pytest.mark.asyncio
async def test_a_plugin_website_env_deploys_loads_by_signed_url_and_reattaches(sent):
    env = _site("shop", "plugin_pods")
    assert env._env_provider is None  # a plugin's provider is built when the env deploys
    record = await _deploy(env)
    with _signing_store():
        await env.load_environment_artifact(_artifact("shop"))
    assert [(str(r.url), json.loads(r.content)["method"]) for r in sent] == [(f"{_URL}/agentenv", "data/reset"), (f"{_URL}/agentenv", "data/add")]

    with patch("agent_env.env.env.Env.get", return_value=_site("shop", "plugin_pods")), \
         patch("agent_env.providers.build_sandbox_provider", side_effect=AssertionError("reattached a sandbox")):
        again = await WebsiteEnv.from_deployed_env(record)
    assert (again._deployed, again._env_provider, again._sandbox) == (record, None, None)


@pytest.mark.asyncio
async def test_a_plugin_multi_env_card_without_one_of_its_children_fails_the_deploy_and_closes_the_plugin():
    env = _multi(servers=("slack", "gmail"))
    record = DeployedEnv(env_id="crm-suite", env_version=2, env_provider_type="plugin_pods", environment_card_url=f"{_URL}/.well-known/agent-env.json",
                         environment_card={"name": "crm", "children_environments": [{"name": "slack", "url": "/svc/slack/agentenv"}]})
    with patch.object(_PodProvider, "deploy", AsyncMock(return_value=record)), \
         pytest.raises(RuntimeError, match="whose env card lists no child env named 'gmail', 'shop'; a MultiEnv's card lists each"):
        await _deploy(env)
    assert SEEN["closed"] == 1 and env._deployed is None


def _preflight(env, **options) -> list[str]:
    with patch("agent_env.env.env.Env.get", return_value=env):
        return DeployEnvTaskStep(id="d", version=None, env_id=env.id, **options).preflight()


@pytest.mark.asyncio
@pytest.mark.parametrize("make, refusal", [
    (lambda: _multi(websites=("slack",)),
     "which gives a multi one env card, so it can't tell an MCP server and a website apart by name, and both are named 'slack'"),
    (lambda: _multi("server"), "which deploys one MCP server, not a multi env"),
    (lambda: _site("shop", "server"), "which deploys one MCP server, not a website env"),
    (lambda: _multi("plugin_named"), "which doesn't take cpu; name it in the provider's deploy() or take **options"),
], ids=["shared-name", "server-multi", "server-website", "named-options"])
async def test_a_deploy_the_env_cannot_take_is_refused_at_save_and_before_anything_is_built(make, refusal):
    [problem] = _preflight(make(), cpu=2.0)
    assert refusal in problem
    with patch("agent_env.providers.get_env_sandbox_provider") as sandboxes, pytest.raises(ValueError, match=re.escape(refusal)):
        await make().deploy(cpu=2.0)
    assert (sandboxes.called, "deploys" in SEEN) == (False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("make", [lambda: _multi("gateway"), lambda: _site("shop")], ids=["multi", "website"])
async def test_the_server_provider_refuses_an_env_that_is_not_one_mcp_server(make):
    with pytest.raises(TypeError, match="deploys one MCP server"):
        await EnvironmentServerProvider().deploy(make(), MagicMock(create_container=AsyncMock(side_effect=AssertionError("built one"))))


@pytest.mark.asyncio
@pytest.mark.parametrize("universe", [{"id": "u", "version": 1}, None], ids=["loaded", "none-loaded"])
async def test_snapshot_capture_is_refused_for_a_plugin_multi_env(universe):
    record = DeployedEnv(env_id="crm-suite", env_version=2, env_provider_type="plugin_pods", environment_card_url=f"{_URL}/.well-known/agent-env.json",
                         environment_card=_card_for(_multi()), instance_id="inst-1")
    store = MagicMock(get=MagicMock(return_value=record), get_environment_universe=MagicMock(return_value=universe))
    with patch("agent_env.env.store.get_env_instance_store", return_value=store), patch("agent_env.env.env.Env.get", return_value=_multi()), \
         pytest.raises(NotImplementedError, match="Snapshot capture needs our gateway and its local Postgres store; env 'crm-suite' was "
                                                  "deployed by env_provider_type 'plugin_pods'"):
        await EnvSnapshot.create("inst-1")


@pytest.mark.asyncio
@pytest.mark.parametrize("make", [lambda: _multi(), lambda: _site("shop", "plugin_pods")], ids=["multi", "website"])
async def test_staging_files_onto_a_plugin_deployment_is_refused(make):
    env = make()
    await _deploy(env)
    with pytest.raises(RuntimeError, match="Staging files onto the env's host needs a built-in env provider"):
        await env.load_file_artifact_universe(MagicMock())


@pytest.mark.asyncio
async def test_validate_terminates_the_sandboxes_a_plugin_multi_env_record_names():
    record = DeployedSandboxEnv(env_id="crm-suite", env_version=2, env_provider_type="plugin_pods", mcp_url=f"{_URL}/mcp",
                                sandbox_id="sb-1", sandbox_type="modal", instance_id="inst-1")
    context = SimpleNamespace(deployed_envs=[record], deployed_agents=[], instance_id="task-inst")
    with patch("agent_env.task.Task.put", return_value=MagicMock(id="t", version=1, run=AsyncMock(return_value=context))), \
         patch("agent_env.env.env.Env.get", return_value=_multi()), \
         patch("agent_env.task_step.task_steps.teardown_sandboxes._terminate", AsyncMock()) as terminate:
        assert await _multi().validate() == "task-inst"
    terminate.assert_awaited_once_with("sb-1", "modal", "env")


@pytest.mark.parametrize("env_class, make", [(MultiEnv, _multi), (WebsiteEnv, lambda t: _site("shop", t))], ids=["multi", "website"])
def test_env_provider_type_round_trips_and_defaults_to_gateway(env_class, make):
    doc = make("plugin_pods").to_dict()
    assert doc["env_provider_type"] == "plugin_pods"
    with patch("agent_env.env.env.Env.get", side_effect=lambda id, version=None: _mcp("slack") if id.startswith("mcp-") else _site("shop")), \
         patch("agent_env.artifact.Artifact.get", return_value=MagicMock()):
        assert env_class.from_dict(doc).env_provider_type == "plugin_pods"
        assert env_class.from_dict({k: v for k, v in doc.items() if k != "env_provider_type"}).env_provider_type == "gateway"
    with pytest.raises(ValueError, match="env_provider_type cannot be empty"):
        make("")


def test_multi_env_put_takes_an_installed_env_provider_type_and_refuses_another():
    server = MagicMock(type="mcp_server", id="slack", version=3)
    with patch("agent_env.cli.env.multi.Env.get", return_value=server), \
         patch("agent_env.cli.env.multi.detect_base_metadata", return_value={}), \
         patch("agent_env.cli.env.multi.MultiEnv.put", return_value=MagicMock(id="crm", version=1, env_provider_type="plugin_pods")) as put:
        ok = CliRunner().invoke(multi, ["put", "--id", "crm", "--mcp-server", "slack", "--env-provider-type", "plugin_pods"])
        refused = CliRunner().invoke(multi, ["put", "--id", "crm", "--mcp-server", "slack", "--env-provider-type", "not_installed"])
    assert ok.exit_code == 0, ok.output
    assert put.call_count == 1 and put.call_args.kwargs["env_provider_type"] == "plugin_pods"
    assert refused.exit_code == 2 and "Unknown env_provider_type: 'not_installed'" in refused.output


@pytest.mark.asyncio
@pytest.mark.parametrize("env_class, make", [(MultiEnv, _multi), (WebsiteEnv, lambda: _site("shop", "plugin_pods"))], ids=["multi", "website"])
async def test_validate_cleanup_survives_a_failure_on_a_record_without_a_sandbox(env_class, make):
    record = DeployedEnv(env_id="crm-suite", env_version=2, env_provider_type="plugin_pods", mcp_url=f"{_URL}/mcp", instance_id="inst-1")
    context = SimpleNamespace(deployed_envs=[record], deployed_agents=[], instance_id="task-inst")
    with patch("agent_env.task.Task.put", return_value=MagicMock(id="t", version=1, run=AsyncMock(return_value=context))), \
         patch.object(env_class, "from_deployed_env", AsyncMock(side_effect=RuntimeError("gone"))):
        assert await make().validate() == "task-inst"


@pytest.mark.asyncio
async def test_a_plugin_multi_env_record_with_no_card_fails_the_deploy_and_closes_the_plugin():
    record = DeployedEnv(env_id="crm-suite", env_version=2, env_provider_type="plugin_pods", mcp_url=f"{_URL}/mcp")
    env = _multi()
    with patch.object(_PodProvider, "deploy", AsyncMock(return_value=record)), \
         pytest.raises(RuntimeError, match="returned a record with no env card; a MultiEnv's children are reached through the card"):
        await _deploy(env)
    assert SEEN["closed"] == 1 and env._deployed is None


@pytest.mark.asyncio
async def test_a_failed_second_deploy_of_a_plugin_multi_env_leaves_the_first_deployment_as_it_was():
    env = _multi()
    first = await _deploy(env)
    first_provider = env._env_provider
    with patch.object(_PodProvider, "deploy", AsyncMock(side_effect=RuntimeError("boom"))), pytest.raises(RuntimeError, match="boom"):
        await _deploy(env)
    assert (env._deployed, env._env_provider, env._instance_id) == (first, first_provider, "inst-1")
    assert SEEN["closed"] == 1  # the second deploy's own provider, not the first's


@pytest.mark.asyncio
async def test_a_failed_second_deploy_of_a_gateway_multi_env_does_not_terminate_the_first_deployment():
    vm = MagicMock(sandbox_id="vm-1", type="modal_vm", terminate=AsyncMock())
    calls = []

    async def deploy_gateway(self, sandbox_provider, **kwargs):
        calls.append(self)
        if len(calls) > 1:
            raise RuntimeError("second boom")
        self._sandbox = vm
        return DeployedGateway(gateway_url="https://gw", mcp_url="https://gw/mcp", db_web_url=None)

    env = _multi("gateway", websites=())
    with patch.object(EnvironmentGatewayProvider, "_deploy_gateway", deploy_gateway), \
         patch.object(env_gateway_provider, "_probe_tools", AsyncMock()), patch("agent_env.env.env.Env.get"), \
         patch("agent_env.providers.env_state.acquire_state_for_deploy", AsyncMock(return_value=None)):
        first = await _deploy(env)
        with pytest.raises(RuntimeError, match="second boom"):
            await _deploy(env)
    assert calls[0] is not calls[1]  # the second deploy got its own provider
    assert (env._deployed, env._sandbox, env._env_provider) == (first, vm, calls[0]) and vm.terminate.await_count == 0


@pytest.mark.asyncio
async def test_a_failed_second_deploy_whose_provider_fails_to_close_still_raises_its_error_and_leaves_the_first_as_it_was():
    env = _multi()
    first = await _deploy(env)
    first_provider = env._env_provider
    with patch.object(_PodProvider, "deploy", AsyncMock(side_effect=RuntimeError("boom"))), \
         patch.object(_PodProvider, "close", AsyncMock(side_effect=RuntimeError("close failed"))), \
         pytest.raises(RuntimeError, match="boom"):
        await _deploy(env)
    assert (env._deployed, env._env_provider) == (first, first_provider)


@pytest.mark.asyncio
@pytest.mark.parametrize("make", [
    lambda: MCPServerEnv(id="mcp-slack", version=1, docker_image_artifact=MagicMock(image_name="mcp-slack"), environment_name="slack",
                         env_provider_type="plugin_pods"),
    lambda: _site("shop", "plugin_pods"),
], ids=["mcp", "website"])
async def test_a_second_deploy_of_a_plugin_env_gets_a_provider_of_its_own_and_close_ends_both(make):
    env = make()
    await _deploy(env)
    first_provider = env._env_provider
    await _deploy(env)
    assert isinstance(env._env_provider, _PodProvider) and env._env_provider is not first_provider
    await env.close()
    assert SEEN["closed"] == 2


@pytest.mark.asyncio
async def test_close_after_two_successful_deploys_tears_both_down_and_a_failed_deploy_between_leaves_them_alone():
    env = _multi()
    await _deploy(env)
    second = await _deploy(env)
    with patch.object(_PodProvider, "deploy", AsyncMock(side_effect=RuntimeError("boom"))), pytest.raises(RuntimeError, match="boom"):
        await _deploy(env)
    assert SEEN["closed"] == 1 and env._deployed is second  # only the failed attempt's own provider
    await env.close()
    assert SEEN["closed"] == 3 and env._replaced == []


@pytest.mark.asyncio
async def test_close_after_two_successful_gateway_deploys_terminates_both_vms():
    vms = [MagicMock(sandbox_id=f"vm-{i}", type="modal_vm", terminate=AsyncMock()) for i in (1, 2)]
    created = iter(vms)

    async def deploy_gateway(self, sandbox_provider, **kwargs):
        self._sandbox = next(created)
        return DeployedGateway(gateway_url="https://gw", mcp_url="https://gw/mcp", db_web_url=None)

    env = _multi("gateway", websites=())
    with patch.object(EnvironmentGatewayProvider, "_deploy_gateway", deploy_gateway), \
         patch.object(env_gateway_provider, "_probe_tools", AsyncMock()), patch("agent_env.env.env.Env.get"), \
         patch("agent_env.providers.env_state.acquire_state_for_deploy", AsyncMock(return_value=None)):
        await _deploy(env)
        await _deploy(env)
    assert env._sandbox is vms[1] and all(vm.terminate.await_count == 0 for vm in vms)
    await env.close()
    assert [vm.terminate.await_count for vm in vms] == [1, 1]


@pytest.mark.asyncio
async def test_a_cancelled_close_leaves_what_it_did_not_finish_for_the_next_close():
    first = (MagicMock(close=AsyncMock()), MagicMock(sandbox_id="vm-1", terminate=AsyncMock(side_effect=[asyncio.CancelledError(), None])))
    second = (MagicMock(close=AsyncMock()), MagicMock(sandbox_id="vm-2", terminate=AsyncMock()))
    env = _multi()
    env._replaced = [first, second]
    with pytest.raises(asyncio.CancelledError):
        await close_replaced(env)
    assert env._replaced == [(None, first[1]), second]
    await close_replaced(env)
    assert env._replaced == [] and [first[0].close.await_count, first[1].terminate.await_count] == [1, 2]
    assert [second[0].close.await_count, second[1].terminate.await_count] == [1, 1]


@pytest.mark.asyncio
async def test_two_closes_at_once_tear_down_each_replaced_deployment_once():
    async def slow_close():
        await asyncio.sleep(0)

    entries = [(MagicMock(close=AsyncMock(side_effect=slow_close)), MagicMock(sandbox_id=f"vm-{i}", terminate=AsyncMock())) for i in range(3)]
    env = _multi()
    env._replaced = list(entries)
    await asyncio.gather(close_replaced(env), close_replaced(env))
    assert env._replaced == [] and all([p.close.await_count, sb.terminate.await_count] == [1, 1] for p, sb in entries)


@pytest.mark.asyncio
@pytest.mark.parametrize("make, card", [
    (_multi, {"name": "crm", "children_environments": [{"name": "slack", "url": "http://127.0.0.1:5001/agentenv"},
                                                       {"name": "shop", "url": "/svc/shop/agentenv"}]}),
    (lambda: _site("shop", "plugin_pods"), {"name": "shop", "url": "http://127.0.0.1:5001/agentenv"}),
], ids=["multi", "website"])
async def test_a_plugin_card_whose_child_url_is_not_a_path_fails_the_deploy(make, card):
    record = DeployedEnv(env_id="x", env_version=1, env_provider_type="plugin_pods", environment_card_url=f"{_URL}/.well-known/agent-env.json",
                         environment_card=card)
    with patch.object(_PodProvider, "deploy", AsyncMock(return_value=record)), \
         pytest.raises(RuntimeError, match="a url that isn't a path; a child env's url is a path under the record's address"):
        await _deploy(make())
    assert SEEN["closed"] == 1


@pytest.mark.asyncio
async def test_a_plugin_multi_env_record_names_its_mcp_server_after_the_env():
    record = DeployedEnv(env_id="crm-suite", env_version=2, env_provider_type="plugin_pods", environment_card_url=f"{_URL}/.well-known/agent-env.json",
                         environment_card={**_card_for(_multi()), "name": "other"})
    with patch.object(_PodProvider, "deploy", AsyncMock(return_value=record)), \
         pytest.raises(RuntimeError, match="whose MCP server is named 'other'; agents see it under the record's mcp_server_name"):
        await _deploy(_multi())


@pytest.mark.asyncio
async def test_a_provider_that_subclasses_a_builtin_must_return_its_kind_of_record_and_its_vm_is_terminated():
    vm = MagicMock(sandbox_id="vm-1", terminate=AsyncMock())

    async def deploy(self, env, sandbox_provider, **options):
        self._sandbox = vm
        return DeployedEnv(env_id=env.id, env_version=env.version, env_provider_type=self.type, mcp_url=f"{_URL}/mcp")

    env = _multi("plugin_gwsub", websites=())
    with patch.object(_GatewaySubclass, "deploy", deploy), patch.object(EnvironmentGatewayProvider, "close", AsyncMock()) as close, \
         pytest.raises(TypeError, match=re.escape("handled as one: its deploy() must return the built-in's kind of record, not a DeployedEnv")):
        await _deploy(env)
    assert close.await_count == 1 and vm.terminate.await_count == 1 and env._deployed is None


@pytest.mark.asyncio
async def test_a_multi_env_keeps_a_provider_whose_close_failed_for_the_next_close():
    env = _multi()
    await _deploy(env)
    provider = env._env_provider
    with patch.object(_PodProvider, "close", AsyncMock(side_effect=[RuntimeError("busy"), None])) as close:
        await env.close()
        assert env._env_provider is provider
        await env.close()
    assert close.await_count == 2 and env._env_provider is None


def test_a_provider_that_does_not_take_attribution_is_refused_at_save():
    [problem] = _preflight(_multi("plugin_ttl"))
    assert "which doesn't take attribution" in problem


@pytest.mark.asyncio
async def test_a_child_deployed_on_its_own_does_not_take_over_its_parent_deployment():
    parent = _multi(websites=())
    await _deploy(parent)
    child = parent.mcp_server_envs[0]
    child.env_provider_type = "plugin_pods"
    await _deploy(child)
    await child.close()
    assert child._replaced == [] and SEEN["closed"] == 1  # the child's own provider, not the parent's


@pytest.mark.asyncio
async def test_a_hand_wired_sandbox_is_not_the_env_s_to_close():
    env = _multi()
    env._sandbox = hand_wired = MagicMock(terminate=AsyncMock())
    await _deploy(env)
    await env.close()
    assert env._replaced == [] and hand_wired.terminate.await_count == 0


def test_multi_load_universe_takes_a_plugin_record_with_no_sandbox():
    record = DeployedEnv(env_id="crm-suite", env_version=2, env_provider_type="plugin_pods", mcp_url=f"{_URL}/mcp", instance_id="inst-1")
    env = _multi()
    with patch("agent_env.env.store.get_env_instance_store", return_value=MagicMock(get=MagicMock(return_value=record))), \
         patch("agent_env.cli.env.multi.Env.get", return_value=env), \
         patch.object(MultiEnv, "from_deployed_env", AsyncMock(return_value=env)), \
         patch("agent_env.cli.env.multi.EnvironmentUniverseArtifact.get", return_value=MagicMock(id="u", version=1)), \
         patch.object(MultiEnv, "load_environment_universe_artifact", AsyncMock(return_value=MagicMock(restored_from_snapshot=False))):
        result = CliRunner().invoke(multi, ["load-environment-universe-artifact", "--env-instance-id", "inst-1",
                                            "--environment-universe-artifact-id", "u"])
    assert result.exit_code == 0, result.output
    assert "Sandbox backend" not in result.output


@pytest.mark.parametrize("group, args", [
    (lambda: website, ["put", "--id", "shop", "--env-provider-type", "server"]),
    (lambda: multi, ["put", "--id", "crm", "--mcp-server", "slack", "--env-provider-type", "server"]),
], ids=["website", "multi"])
def test_put_refuses_the_server_provider_for_an_env_that_is_not_one_mcp_server(group, args):
    with patch("agent_env.cli.env.multi.MultiEnv.put") as put:
        result = CliRunner().invoke(group(), args)
    assert result.exit_code == 2 and "'server' deploys one MCP server" in result.output and not put.called


def test_multi_put_refuses_a_plugin_env_whose_server_and_website_share_a_name():
    envs = {"slack": MagicMock(type="mcp_server", id="slack", version=1, environment_name="shop"),
            "shop": MagicMock(type="website", id="shop", version=1, environment_name="shop")}
    with patch("agent_env.cli.env.multi.Env.get", side_effect=lambda id, version=None: envs[id]), \
         patch("agent_env.cli.env.multi.detect_base_metadata", return_value={}), \
         patch("agent_env.cli.env.multi.MultiEnv.put") as put:
        result = CliRunner().invoke(multi, ["put", "--id", "crm", "--mcp-server", "slack", "--website", "shop", "--env-provider-type", "plugin_pods"])
    assert result.exit_code == 1 and "both are named 'shop'" in result.output and not put.called


def test_a_closed_builtin_multi_env_is_not_taken_for_a_plugin_deployment():
    env = _multi("gateway")
    env._deployed = DeployedEnv(env_id="crm-suite", env_version=2, env_provider_type="gateway", mcp_url=f"{_URL}/mcp")
    env._env_provider = None  # as close() leaves it
    assert host_staging_refusal(env, "Staging") is None


@pytest.mark.asyncio
async def test_a_deploy_onto_the_state_instance_the_env_s_deployment_holds_is_refused_before_anything_is_built():
    env = _multi()
    env._deployed = SimpleNamespace(env_id="crm-suite", instance_id="inst-1", env_state_instance_ids=["esi-1"])
    with patch("agent_env.providers.get_env_sandbox_provider") as sandboxes, \
         pytest.raises(ValueError, match="already has a deployment, instance 'inst-1', on env state instance 'esi-1'"):
        await env.deploy(env_state_instance_id="esi-1")
    assert (sandboxes.called, "deploys" in SEEN) == (False, False)


@pytest.mark.asyncio
async def test_a_built_in_env_forwards_the_universe_artifact_to_its_provider():
    """A warm pool keys on env *and* universe, so the provider has to be told which universe
    the run will load. Absent, both arrive as None — the pool then treats the env as unpooled
    rather than guessing, and every existing caller is unaffected."""
    SEEN.clear()
    SEEN["deploys"] = []
    await _deploy(_multi(), artifact_id="hg4_real", artifact_version=5)
    [(_, options)] = SEEN["deploys"]
    assert (options["artifact_id"], options["artifact_version"]) == ("hg4_real", 5)

    SEEN["deploys"] = []
    await _deploy(_multi())
    [(_, options)] = SEEN["deploys"]
    assert (options["artifact_id"], options["artifact_version"]) == (None, None)
