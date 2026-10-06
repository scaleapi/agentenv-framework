"""A standalone MCPServerEnv on Modal containers stages into its server's own container and records every container it
created, so a later process can reattach them: loads stage into the server again, and close() and the reapers reach
them all. A dead servicedb or sidecar is skipped on restore; a dead server fails it. A VM deploy's record is unchanged.
A load is timed by the size of the payload it staged, read in the server's container, or in the env's docker container on a VM.
Deployed without a gateway, the server is the env: its own card and container are the record's, and loads go straight to it.
Only the sandboxes, the HTTP boundary and each topology's deploy path are faked."""

from __future__ import annotations

import dataclasses
import inspect
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from agentenv_protocol import WELL_KNOWN_PATH

from agent_env.artifact import Artifact
from agent_env.env.env import DeployedEnv, DeployedSandboxEnv
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.gateway import GatewayMode
from agent_env.env.gateway.constants import data_plane_load_timeout_s
from agent_env.providers.env_providers.env_gateway_provider import DeployedGateway
from agent_env.providers.env_providers.env_provider import _builtin_env_providers, build_env_provider
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER, SANDBOX_MODE_VM
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.teardown_sandboxes import _env_sandbox_ids

_GW = "https://gw.example"
_CARD = {"name": "env1234", "url": "/agentenv", "children_environments": [{"name": "email", "url": "/svc/mcp-email/agentenv"}]}
_RECORDED = {"gateway_server": "gw", "mcp_server": {"email": "srv"}, "service_db": {"servicedb": "db", "pgweb": "pg", "db-mcp": "dm"}}
_SRV = "https://srv.example"
_LEAF_CARD = {"name": "email", "url": "/agentenv"}


@pytest.fixture
def sent(monkeypatch) -> list[httpx.Request]:
    """Every data-plane request a load sends, each answered with an empty JSON-RPC result."""
    requests, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return requests


@pytest.mark.asyncio
@pytest.mark.usefixtures("sent")
async def test_a_container_deploy_stages_into_the_server_and_records_every_container():
    env, sandboxes = _env(), _containers()

    deployed = await _deploy(env, sandboxes)
    await env.load_environment_artifact(_artifact())

    assert env._sandbox is sandboxes["srv"]
    assert (deployed.sandbox_id, deployed.sandbox_type) == ("gw", "modal")
    assert deployed.sandbox_ids == _RECORDED
    assert {i for i, _ in _env_sandbox_ids(deployed)} == {sb.sandbox_id for sb in env._env_provider._container_sandboxes}
    sandboxes["srv"].write_file_from_s3.assert_awaited_once_with("s3://bucket/email.json", "/data/email.json")
    sandboxes["gw"].write_file_from_s3.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_container_load_is_timed_by_the_payload_staged_in_the_server(sent):
    env, sandboxes, size = _env(), _containers(), 3_500 * 1024 * 1024
    sandboxes["srv"].exec_with_output = AsyncMock(return_value=(0, f"{size}\n", ""))

    await _deploy(env, sandboxes)
    await env.load_environment_artifact(_artifact())

    sandboxes["srv"].exec_with_output.assert_awaited_once_with("stat", "-c", "%s", "/data/email.json")
    assert sent and {r.extensions["timeout"]["read"] for r in sent} == {data_plane_load_timeout_s(size)}
    assert data_plane_load_timeout_s(size) > data_plane_load_timeout_s(None)


@pytest.mark.asyncio
async def test_a_container_stat_that_fails_leaves_the_load_its_floor_timeout():
    env, server = _env(), _sandbox("srv")
    server.exec_with_output = AsyncMock(return_value=(1, "", "stat: cannot statx '/data/email.json'"))
    env._sandbox = server

    assert await env._staged_artifact_size("/data/email.json") is None


@pytest.mark.asyncio
async def test_a_container_on_a_docker_host_is_measured_through_docker_exec():
    """A VM-backed sandbox in container mode (the local one) execs on the container's host, not in it."""
    env = _env()
    env._sandbox = MagicMock(spec=VmSandbox, mode=SANDBOX_MODE_CONTAINER, container_name="agent-env-srv")
    env._sandbox.exec_with_output = AsyncMock(return_value=(0, "1234\n", ""))

    assert await env._staged_artifact_size("/data/email.json") == 1234
    env._sandbox.exec_with_output.assert_awaited_once_with(
        "sudo", "docker", "exec", "agent-env-srv", "stat", "-c", "%s", "/data/email.json")


@pytest.mark.asyncio
async def test_a_vm_payload_is_measured_in_the_envs_docker_container():
    env, vm = _env(), _sandbox("vm-1", mode=SANDBOX_MODE_VM, sandbox_type="modal_vm")
    vm.exec_script = AsyncMock(return_value="1234\n")
    env._sandbox, env._env_provider._get_container_id = vm, AsyncMock(return_value="ctr-1")

    assert await env._staged_artifact_size("/data/email.json") == 1234
    vm.exec_script.assert_awaited_once_with("docker exec ctr-1 stat -c %s /data/email.json")


@pytest.mark.asyncio
async def test_a_vm_deploy_record_is_unchanged():
    env, vm = _env(), _sandbox("vm-1", mode="vm", sandbox_type="modal_vm")

    deployed = await _deploy(env, {"gw": vm})

    assert env._sandbox is vm
    assert (deployed.sandbox_id, deployed.sandbox_type, deployed.sandbox_ids) == ("vm-1", "modal_vm", {})


@pytest.mark.asyncio
async def test_a_container_deploy_on_remote_state_records_and_closes_the_gateway_and_server():
    record = await _deploy(_env(), {"gw": _sandbox("gw"), "srv": _sandbox("srv")})
    reattached = {"gw": _sandbox("gw"), "srv": _sandbox("srv")}

    env = await _restore(record, reattached)
    await env.close()

    assert record.sandbox_ids == {"gateway_server": "gw", "mcp_server": {"email": "srv"}}
    assert [name for name, sb in reattached.items() if sb.terminate.await_count == 0] == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("sent")
async def test_a_restored_container_env_stages_into_the_server_and_closes_every_container():
    record = await _deploy(_env(), _containers())
    reattached = _containers()

    env = await _restore(record, reattached)
    await env.load_environment_artifact(_artifact())
    await env.close()

    assert env._sandbox is None
    reattached["srv"].write_file_from_s3.assert_awaited_once_with("s3://bucket/email.json", "/data/email.json")
    reattached["gw"].write_file_from_s3.assert_not_awaited()
    assert [name for name, sb in reattached.items() if sb.terminate.await_count == 0] == []


@pytest.mark.asyncio
async def test_a_dead_sidecar_is_skipped_and_the_rest_still_close(caplog):
    record = await _deploy(_env(), _containers())
    reattached = _containers()

    env = await _restore(record, reattached, dead=("pg",))
    await env.close()

    assert env._sandbox is None
    assert "skipping recorded container pg" in caplog.text
    assert [name for name, sb in reattached.items() if sb.terminate.await_count == 0] == ["pg"]


@pytest.mark.asyncio
async def test_a_dead_server_container_fails_the_restore():
    record = await _deploy(_env(), _containers())

    with pytest.raises(RuntimeError, match="can't reattach the server's container srv"):
        await _restore(record, _containers(), dead=("srv",))


@pytest.mark.asyncio
async def test_a_record_without_sandbox_ids_restores_to_its_primary_sandbox():
    gateway = _sandbox("gw")
    record = dataclasses.replace(await _deploy(_env(), _containers()), sandbox_ids={})

    env = await _restore(record, {"gw": gateway})

    assert env._sandbox is gateway
    assert env._env_provider._container_sandboxes == []



@pytest.mark.asyncio
async def test_a_record_without_a_gateway_restores_with_no_gateway_url():
    record = DeployedSandboxEnv(env_id="mcp-email", env_version=1, sandbox_id="srv", sandbox_type="modal",
                                sandbox_ids={"mcp_server": {"email": "srv"}})
    env = await _restore(record, {"srv": _sandbox("srv")})
    assert (env._gateway_url, env._sandbox.sandbox_id) == (None, "srv")

@pytest.mark.asyncio
async def test_a_deploy_hands_the_gateway_its_sizing_state_and_attribution_and_registers_the_record():
    env, state, seen = _env(), SimpleNamespace(state_type="remote", instance_id="st-ext"), {}

    record = await _deploy(env, {"gw": _sandbox("gw")}, seen, state, gateway_mode=GatewayMode.CONSISTENT, env_state_type="remote",
                           ttl_seconds=60, disk_size_gb=20, cpu=2.0, memory_mb=4096, attribution={"team": "t1"})

    assert ([c.environment_name for c in seen["mcp_servers"]], seen["mcp_server_images"]) == (["email"], [env.docker_image_artifact])
    assert {k: seen[k] for k in ("ttl_seconds", "disk_size_gb", "cpu", "memory_mb", "attribution", "gateway_mode")} == {
        "ttl_seconds": 60, "disk_size_gb": 20, "cpu": 2.0, "memory_mb": 4096, "attribution": {"team": "t1"},
        "gateway_mode": GatewayMode.CONSISTENT}
    assert (seen["gateway_port"], seen["website_configs"], seen["website_images"], seen["existing_sandbox"], seen["sidecars"]) == (
        18765, None, None, None, None)
    assert seen["state_instance"] is state and (seen["acquired_for"], seen["acquired_type"]) == ("mcp-email", "remote")
    assert re.fullmatch(r"env\d{4}", seen["mcp_server_name"])
    assert (record.mcp_server_name, record.gateway_mode, record.environment_card_url) == ("env1234", "consistent", f"{_GW}{WELL_KNOWN_PATH}")
    assert (record.instance_id, env._instance_id, env._deployed) == ("inst-1", "inst-1", record)


def test_the_env_provider_type_round_trips_and_defaults_to_the_gateway(monkeypatch):
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: MagicMock(id=id, version=version, type="docker_image")))
    bare, fronted = _env("server").to_dict(), _env().to_dict()
    assert (bare["env_provider_type"], fronted["env_provider_type"]) == ("server", "gateway")
    assert (MCPServerEnv.from_dict(bare).env_provider_type, MCPServerEnv.from_dict(fronted).env_provider_type) == ("server", "gateway")
    del fronted["env_provider_type"]  # a doc written before the field
    assert MCPServerEnv.from_dict(fronted).env_provider_type == "gateway"
    assert type(MCPServerEnv.from_dict(bare)._env_provider).type == "server"


@pytest.mark.asyncio
async def test_an_env_provider_type_nothing_registers_reads_and_is_refused_at_deploy_before_any_spend():
    env = _env("vm")
    with patch("agent_env.providers.get_env_sandbox_provider") as sandbox_provider, pytest.raises(ValueError, match="Unknown env_provider_type: 'vm'"):
        await env.deploy()
    assert sandbox_provider.called is False


def test_every_option_the_env_maps_is_one_a_declared_provider_takes():
    mapped = set(inspect.signature(MCPServerEnv.deploy).parameters) - {"self", "sandbox_type"}  # it consumes sandbox_type itself
    taken = set().union(*(inspect.signature(build_env_provider(t).deploy).parameters for t in _builtin_env_providers()))
    assert mapped <= taken


@pytest.mark.asyncio
async def test_a_deploy_without_a_gateway_records_the_servers_own_card():
    deployed = await _deploy(_env("server"), {"srv": _sandbox("srv")})

    assert type(deployed) is DeployedSandboxEnv and not hasattr(deployed, "gateway_url")
    assert (deployed.env_provider_type, deployed.sandbox_id, deployed.sandbox_type, deployed.sandbox_ids) == (
        "server", "srv", "modal", {"mcp_server": {"email": "srv"}})
    assert (deployed.environment_card_url, deployed.environment_card, deployed.mcp_url, deployed.mcp_server_name) == (
        f"{_SRV}{WELL_KNOWN_PATH}", _LEAF_CARD, f"{_SRV}/mcp", "email")
    assert {i for i, _ in _env_sandbox_ids(deployed)} == {"srv"}


@pytest.mark.asyncio
async def test_a_deploy_without_a_gateway_hands_the_server_its_sizing_and_attribution():
    seen = {}
    await _deploy(_env("server"), {"srv": _sandbox("srv")}, seen, gateway_mode=GatewayMode.PERFORMANCE,  # the deploy_env step's default
                  ttl_seconds=60, disk_size_gb=20, cpu=2.0, memory_mb=4096, attribution={"team": "t1"})

    assert seen == {"ttl_seconds": 60, "disk_size_gb": 20, "cpu": 2.0, "memory_mb": 4096, "attribution": {"team": "t1"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("options, refused", [
    ({"gateway_mode": GatewayMode.CONSISTENT}, "gateway_mode"),
    ({"env_state_type": "remote_postgres"}, "env_state_type"),
    ({"env_state_instance_id": "st-1"}, "env_state_instance_id"),
    ({"gateway_mode": "consistent", "env_state_type": "local_postgres"}, "gateway_mode, env_state_type"),
], ids=["consistent", "state-type", "state-instance", "both"])
async def test_a_deploy_without_a_gateway_refuses_what_only_a_gateway_takes_before_any_spend(options, refused):
    env = _env("server")
    with patch("agent_env.config.get_config") as config, patch("agent_env.providers.get_env_sandbox_provider") as sandbox_provider, \
         patch.object(env._env_provider, "_deploy_server") as server, pytest.raises(ValueError) as e:
        await env.deploy(**options)
    assert str(e.value) == f"env 'mcp-email' has env_provider_type 'server', which doesn't take {refused}"
    assert (config.called, sandbox_provider.called, server.called) == (False, False, False)


@pytest.mark.asyncio
async def test_a_deploy_without_a_gateway_loads_straight_into_the_server(sent):
    env, sandboxes = _env("server"), {"srv": _sandbox("srv")}

    await _deploy(env, sandboxes)
    await env.load_environment_artifact(_artifact())

    sandboxes["srv"].write_file_from_s3.assert_awaited_once_with("s3://bucket/email.json", "/data/email.json")
    assert {str(r.url) for r in sent} == {f"{_SRV}/agentenv"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, sandbox_type", [(SANDBOX_MODE_CONTAINER, "modal"), (SANDBOX_MODE_VM, "beta_scale")],
                         ids=["container-handle", "vm-handle"])  # get_sandbox off Modal rebuilds a VM-mode handle
async def test_a_restored_env_without_a_gateway_loads_into_the_server_and_closes_it(sent, mode, sandbox_type):
    record = await _deploy(_env("server"), {"srv": _sandbox("srv")})
    reattached = {"srv": _sandbox("srv", mode=mode, sandbox_type=sandbox_type)}

    env = await _restore(DeployedEnv.from_dict(dataclasses.asdict(record)), reattached, env_provider_type="server")
    await env.load_environment_artifact(_artifact())
    await env.close()

    reattached["srv"].write_file_from_s3.assert_awaited_once_with("s3://bucket/email.json", "/data/email.json")
    assert {str(r.url) for r in sent} == {f"{_SRV}/agentenv"} and reattached["srv"].mode == SANDBOX_MODE_CONTAINER
    assert reattached["srv"].terminate.await_count > 0


@pytest.mark.asyncio
async def test_validation_deploys_a_server_env_as_declared_on_the_default_sandbox_provider():
    env, seen, default = _env("server"), {}, MagicMock()
    ran = MagicMock(run=AsyncMock(return_value=MagicMock(deployed_envs=[], deployed_agents=[], instance_id="i")))
    with patch("agent_env.task.Task.put", side_effect=lambda **kwargs: seen.update(kwargs) or ran):
        await env.validate()
    deploy_step = seen["steps"][0]

    async def server_path(env, sandbox_provider, **kwargs):
        seen.update(sandbox_provider=sandbox_provider, options=kwargs)
        env._env_provider._environment_sandboxes = {"email": _sandbox("srv")}
        return DeployedSandboxEnv(env_id=env.id, env_version=env.version, env_provider_type="server", environment_card_url=f"{_SRV}{WELL_KNOWN_PATH}",
                                  environment_card=_LEAF_CARD, sandbox_id="srv", sandbox_type="modal", sandbox_ids={"mcp_server": {"email": "srv"}})

    with patch("agent_env.env.env.Env.get", return_value=env), patch.object(env._env_provider, "_deploy_server", side_effect=server_path), \
         patch("agent_env.providers.get_env_sandbox_provider", return_value=default), \
         patch("agent_env.env.envs._deployment.register_env_instance", side_effect=lambda deployed, ttl: dataclasses.replace(deployed, instance_id="inst-1")), \
         patch("agent_env.task_step.task_steps.deploy_env.update_env_instance_metadata"):
        context = await deploy_step.execute(TaskStepContext())

    [record] = context.deployed_envs
    assert (record.env_provider_type, record.environment_card_url) == ("server", f"{_SRV}{WELL_KNOWN_PATH}")  # the card the gate records
    assert seen["sandbox_provider"] is default
    assert set(seen["options"]) == {"ttl_seconds", "disk_size_gb", "cpu", "memory_mb", "attribution"}


def _env(env_provider_type: str = "gateway") -> MCPServerEnv:
    return MCPServerEnv(id="mcp-email", version=1, docker_image_artifact=MagicMock(), environment_name="email",
                        env_provider_type=env_provider_type)


def _sandbox(sandbox_id: str, mode: str = SANDBOX_MODE_CONTAINER, sandbox_type: str = "modal") -> MagicMock:
    return MagicMock(sandbox_id=sandbox_id, mode=mode, type=sandbox_type, terminate=AsyncMock(), write_file_from_s3=AsyncMock())


def _containers() -> dict[str, MagicMock]:
    """What a local-Postgres container deploy creates: gateway, server, servicedb, pgweb and db-mcp."""
    return {i: _sandbox(i) for i in ("gw", "srv", "db", "pg", "dm")}


async def _deploy(env: MCPServerEnv, sandboxes: dict[str, MagicMock], seen: dict | None = None, state=None, **deploy_kwargs) -> DeployedEnv:
    """env.deploy() through the real provider's deploy(), with its deploy path faked to fill the provider the way the real one
    does: the gateway's _deploy_via_vm, or the server's _deploy_server serving _LEAF_CARD; `seen` gets the deploy path's arguments."""
    gp, seen, gateway = env._env_provider, seen if seen is not None else {}, env.env_provider_type == "gateway"

    async def acquire(**kwargs):
        seen["acquired_for"], seen["acquired_type"] = kwargs["name_hint"], kwargs["env_state_type"]
        return state

    async def deploy_path(sandbox_provider, mcp_servers, mcp_server_images, **kwargs):
        seen.update(kwargs, mcp_servers=mcp_servers, mcp_server_images=mcp_server_images, state_instance=gp._state_instance)
        gp._sandbox = sandboxes["gw"]
        if "srv" in sandboxes:
            gp._environment_sandboxes = {"email": sandboxes["srv"]}
            gp._db_sandbox, gp._pgweb_sandbox, gp._db_mcp_sandbox = sandboxes.get("db"), sandboxes.get("pg"), sandboxes.get("dm")
            gp._container_sandboxes = [sandboxes[i] for i in ("db", "srv", "pg", "dm", "gw") if i in sandboxes]
        return DeployedGateway(gateway_url=_GW, mcp_url=f"{_GW}/mcp", db_web_url=None, db_mcp_url=None, mcp_server_name=kwargs["mcp_server_name"],
                               env_state_instance_ids=["st-1"], environment_card=_CARD, environment_card_read_at_utc="2026-09-26T00:00:00+00:00")

    async def server_path(env, sandbox_provider, **kwargs):
        seen.update(kwargs)
        gp._environment_sandboxes, gp._container_sandboxes = {"email": sandboxes["srv"]}, [sandboxes["srv"]]
        return DeployedSandboxEnv(env_id=env.id, env_version=env.version, env_provider_type="server", environment_card_url=f"{_SRV}{WELL_KNOWN_PATH}",
                                  environment_card=_LEAF_CARD, environment_card_read_at_utc="2026-09-26T00:00:00+00:00",
                                  sandbox_id="srv", sandbox_type="modal", sandbox_ids={"mcp_server": {"email": "srv"}})

    with patch.object(gp, "_deploy_via_vm" if gateway else "_deploy_server", side_effect=deploy_path if gateway else server_path), \
         patch("agent_env.providers.env_providers.env_gateway_provider._tool_names", AsyncMock(return_value={"email_send"})), \
         patch("agent_env.providers.env_state.build_state_provider", MagicMock(return_value=MagicMock(teardown=AsyncMock()))), \
         patch("agent_env.config.get_config", MagicMock()), patch("agent_env.env.env.Env.get", MagicMock()), \
         patch("agent_env.providers.get_env_sandbox_provider", MagicMock(return_value=MagicMock())), \
         patch("agent_env.providers.env_state.acquire_state_for_deploy", side_effect=acquire), \
         patch("agent_env.env.envs._deployment.register_env_instance",
               side_effect=lambda deployed, ttl: dataclasses.replace(deployed, instance_id="inst-1")):
        return await env.deploy(**deploy_kwargs)


async def _restore(record: DeployedEnv, sandboxes: dict[str, MagicMock], dead: tuple[str, ...] = (), env_provider_type: str = "gateway") -> MCPServerEnv:
    """from_deployed_env() with get_sandbox answering from `sandboxes`, and raising as Modal does for a finished one in `dead`."""
    def get_sandbox(sandbox_id):
        if sandbox_id in dead:
            raise RuntimeError("Sandbox has already finished with status terminated")
        return sandboxes[sandbox_id]

    provider = MagicMock(get_sandbox=AsyncMock(side_effect=get_sandbox))
    with patch("agent_env.env.env.Env.get", return_value=_env(env_provider_type)), \
         patch("agent_env.providers.build_sandbox_provider", return_value=provider):
        return await MCPServerEnv.from_deployed_env(record)


def _artifact():
    return MagicMock(environment_name="email", **{"get_file_artifact.return_value": SimpleNamespace(
        filename="email.json", object_url="s3://bucket/email.json", content_type="application/json")})
