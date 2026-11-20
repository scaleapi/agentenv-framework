"""A gateway stack's layout in its sandbox: the default is the one stack a sandbox has always held, a slot is one of several."""

from __future__ import annotations

import shlex
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env import snapshot_store
from agent_env.env.envs.service_db import DB_MCP_PORT, DB_WEB_PORT, ServiceDBConfig
from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT
from agent_env.providers.env_providers.env_gateway_provider import (
    TEARDOWN_RETRIES,
    EnvironmentGatewayProvider,
    MCPServerConfig,
    SidecarConfig,
    tear_down_slot,
)
from agent_env.providers.env_providers.stack_layout import DEFAULT_STACK_LAYOUT, StackLayout

SIDECAR_HOST_PORT = 9100


class _RecordingVm:
    """A VM sandbox that records every command and answers the ones a gateway deploy waits on.

    ``running`` holds the compose projects whose gateway is up, keyed ``None`` for the default (unnamed) project; ``docker ps``
    lists a gateway container for each of them, as it would on a VM shared by several stacks.
    """

    type = "modal_vm"
    mode = "vm"
    sandbox_id = "vm-1"
    host_ips: tuple[str, ...] = ()
    extra_hosts: tuple[str, ...] = ()
    ON_THIS_MACHINE = False

    def __init__(self, exposed_ports: list[int], running: set[str | None], walls_verified: bool = False):
        self.exposed_ports = exposed_ports
        self.walls_verified = walls_verified
        self.tunnel_urls = {port: f"https://tunnel-{port}" for port in exposed_ports}
        self.running = running
        self.calls: list[tuple] = []

    def host_port(self, port: int) -> int:
        return port

    async def load_docker_images(self, images) -> None:
        self.calls.append(("load", tuple(i.image_name for i in images)))

    async def exec_script(self, script: str, *, max_retries: int = 0) -> str:
        self.calls.append(("script", script))
        return "cid-servicedb" if " ps -a -q " in script else ""

    async def exec_with_output(self, *args: str) -> tuple[int, str, str]:
        self.calls.append(("exec", args))
        if args[0] == "curl":
            return 0, "200", ""
        if args == ("sudo", "docker", "ps"):
            return 0, "\n".join(f"{p or 'app'}-gateway-1" for p in self.running), ""
        if "--status" in args:
            project = args[args.index("-p") + 1] if "-p" in args else None
            return 0, "cid-gateway" if project in self.running else "", ""
        if args[-2:-1] == ("-q",):
            return 0, f"cid-{args[-1]}", ""
        return 0, "", ""


def _image(name: str) -> MagicMock:
    image = MagicMock()
    image.image_name = name
    image.load_problem.return_value = None
    return image


async def _deploy(running: set[str | None] = frozenset({None}), **layout) -> tuple[_RecordingVm, object, EnvironmentGatewayProvider]:
    """One VM gateway deploy (an MCP server, local Postgres with pgweb and db-mcp, one sidecar) with every I/O faked."""
    vms: list[_RecordingVm] = []

    async def create_vm(*, exposed_ports, **_):
        vms.append(_RecordingVm(exposed_ports, set(running)))
        return vms[-1]

    sandbox_provider = MagicMock(ON_THIS_MACHINE=False, SUPPORTED_SANDBOX_MODES=("vm",))
    sandbox_provider.create_vm = create_vm
    gateway_env = MagicMock()
    gateway_env.docker_image_artifact = _image("agent-gateway")
    db_env = MagicMock()
    db_env.to_config.return_value = ServiceDBConfig(db_web_image="pgweb", db_mcp_image="db-mcp")
    db_env.db_docker_image_artifact = _image("postgres")
    db_env.db_web_docker_image_artifact = _image("pgweb")
    db_env.db_mcp_docker_image_artifact = _image("db-mcp")
    config = MagicMock(default_gateway_env_id="gw-id", default_service_db_env_id="db-id")

    gp = EnvironmentGatewayProvider()
    gp._wait_for_tunnel = AsyncMock(return_value={"name": "crm"})
    gp._refuse_tool_name_clashes = AsyncMock()
    with patch("agent_env.env.env.Env.get", side_effect=lambda env_id, *a, **k: gateway_env if env_id == "gw-id" else db_env), \
         patch("agent_env.config.get_config", return_value=config), \
         patch("agent_env.providers.env_providers.env_gateway_provider.asyncio.sleep", AsyncMock()):
        result = await gp._deploy_gateway(
            sandbox_provider,
            state_instance=None,
            mcp_servers=[MCPServerConfig(image="mcp-slack", environment_name="slack")],
            mcp_server_images=[_image("mcp-slack")],
            sidecars=[SidecarConfig(environment_name="probe", image_artifact=_image("probe"), container_port=80, host_port=SIDECAR_HOST_PORT)],
            mcp_server_name="crm",
            **layout,
        )
    return (vms[0] if vms else layout["existing_sandbox"]), result, gp


def _scripts(vm: _RecordingVm) -> list[str]:
    return [c[1] for c in vm.calls if c[0] == "script"]


def _written(script: str) -> tuple[str, str] | None:
    """The path and content of a ``_write_file`` script (``printf '%s\\n' <content> > <path>``), else None."""
    argv = shlex.split(script)
    if argv[:2] == ["printf", "%s\\n"] and argv[3:4] == [">"]:
        return argv[4], argv[2]
    return None


def _execs(vm: _RecordingVm) -> list[tuple]:
    return [c[1] for c in vm.calls if c[0] == "exec" and c[1][0] != "curl"]


@pytest.mark.asyncio
async def test_default_deploy_runs_todays_commands_on_todays_paths_and_ports():
    vm, result, _ = await _deploy()

    assert vm.exposed_ports == [AGENT_ENV_GATEWAY_MCP_PORT, DB_WEB_PORT, DB_MCP_PORT, SIDECAR_HOST_PORT]
    scripts = _scripts(vm)
    assert scripts[0] == "mkdir -p /app"
    assert _written(scripts[1])[0] == "/app/init-schemas.sql"
    path, compose = _written(scripts[2])
    assert path == "/app/docker-compose.yml"
    assert f'"{AGENT_ENV_GATEWAY_MCP_PORT}:{AGENT_ENV_GATEWAY_MCP_PORT}"' in compose
    assert f'"{SIDECAR_HOST_PORT}:80"' in compose
    assert scripts[3:] == ["cd /app && docker compose up -d"]
    execs = _execs(vm)
    assert ("sudo", "docker", "ps") in execs
    for service in ("gateway", "pgweb", "db-mcp"):
        assert ("sudo", "docker", "compose", "-f", "/app/docker-compose.yml", "ps", "-a", "-q", service) in execs
    assert not any("-p" in e or "--status" in e for e in execs)
    assert result.gateway_url == f"https://tunnel-{AGENT_ENV_GATEWAY_MCP_PORT}"
    assert result.db_web_url == f"https://tunnel-{DB_WEB_PORT}/"
    assert result.db_mcp_url == f"https://tunnel-{DB_MCP_PORT}/mcp"


SLOT_1_PORTS = [p + 2000 for p in (AGENT_ENV_GATEWAY_MCP_PORT, DB_WEB_PORT, DB_MCP_PORT, SIDECAR_HOST_PORT)]


def _walled_vm(running: set[str | None], ports: list[int] = SLOT_1_PORTS) -> _RecordingVm:
    return _RecordingVm(ports, running, walls_verified=True)


@pytest.mark.asyncio
async def test_slot_deploy_uses_its_own_dir_project_and_host_ports():
    slot = StackLayout.for_slot(1)
    vm = _walled_vm({None, "s1"})
    _, result, gp = await _deploy(stack_layout=slot, existing_sandbox=vm)

    scripts = _scripts(vm)
    assert scripts[0] == "mkdir -p /app/s1"
    assert _written(scripts[1])[0] == "/app/s1/init-schemas.sql"
    path, compose = _written(scripts[2])
    assert path == "/app/s1/docker-compose.yml"
    assert f'"{AGENT_ENV_GATEWAY_MCP_PORT + 2000}:{AGENT_ENV_GATEWAY_MCP_PORT}"' in compose
    assert f'"{DB_WEB_PORT + 2000}:{DB_WEB_PORT}"' in compose
    assert f'"{SIDECAR_HOST_PORT + 2000}:80"' in compose
    assert f"MCP_PORT={AGENT_ENV_GATEWAY_MCP_PORT}" in compose
    assert scripts[3:] == ["cd /app/s1 && docker compose -p s1 up -d"]
    compose_ps = ("sudo", "docker", "compose", "-f", "/app/s1/docker-compose.yml", "-p", "s1", "ps")
    assert (*compose_ps, "--status", "running", "-q", "gateway") in _execs(vm)
    assert (*compose_ps, "-a", "-q", "gateway") in _execs(vm)
    assert ("sudo", "docker", "ps") not in _execs(vm)
    assert result.gateway_url == f"https://tunnel-{AGENT_ENV_GATEWAY_MCP_PORT + 2000}"
    assert result.db_web_url == f"https://tunnel-{DB_WEB_PORT + 2000}/"
    assert result.db_mcp_url == f"https://tunnel-{DB_MCP_PORT + 2000}/mcp"
    assert gp.stack_layout == slot


@pytest.mark.asyncio
async def test_a_slot_is_never_deployed_on_a_new_vm():
    with pytest.raises(ValueError, match="walls between slots are verified"):
        await _deploy(stack_layout=StackLayout.for_slot(1))


@pytest.mark.asyncio
async def test_a_slot_is_never_deployed_on_a_vm_whose_walls_are_not_verified():
    vm = _RecordingVm(SLOT_1_PORTS, {"s1"})
    with pytest.raises(ValueError, match="walls between slots are verified"):
        await _deploy(stack_layout=StackLayout.for_slot(1), existing_sandbox=vm)
    assert vm.calls == []


def test_only_a_modal_vm_sandbox_can_have_verified_walls():
    import importlib
    import pkgutil

    import agent_env.providers.sandbox_providers as package
    from agent_env.providers.sandbox_providers.sandbox import Sandbox

    for module in pkgutil.walk_packages(package.__path__, f"{package.__name__}.", onerror=lambda _: None):
        try:
            importlib.import_module(module.name)
        except ImportError:
            continue
    subclasses, pending = [], [Sandbox]
    while pending:
        subclasses += (children := pending.pop().__subclasses__())
        pending += children
    assert subclasses
    ours = [c for c in subclasses if c.__module__.startswith(package.__name__)]
    assert [c.__name__ for c in ours if "walls_verified" in vars(c)] == ["ModalVmSandbox"]


def test_a_provider_subclass_that_skips_init_keeps_the_default_layout():
    class Skips(EnvironmentGatewayProvider):
        def __init__(self):
            pass

    assert Skips().stack_layout == DEFAULT_STACK_LAYOUT


@pytest.mark.asyncio
async def test_slot_deploy_is_not_satisfied_by_another_stacks_gateway():
    with pytest.raises(RuntimeError, match="Gateway container not running"):
        await _deploy(stack_layout=StackLayout.for_slot(1), existing_sandbox=_walled_vm({None, "s0"}))


@pytest.mark.asyncio
async def test_slot_deploy_on_a_vm_not_exposing_its_ports_names_the_missing_port():
    vm = _walled_vm({"s1"}, [AGENT_ENV_GATEWAY_MCP_PORT, DB_WEB_PORT, DB_MCP_PORT])
    with pytest.raises(RuntimeError, match=f"no tunnel for slot port {AGENT_ENV_GATEWAY_MCP_PORT + 2000}"):
        await _deploy(stack_layout=StackLayout.for_slot(1), existing_sandbox=vm)


@pytest.mark.asyncio
async def test_slot_commands_after_deploy_stay_in_the_slot():
    vm = _walled_vm({"s1"})
    _, _, gp = await _deploy(stack_layout=StackLayout.for_slot(1), existing_sandbox=vm)
    gp._sandbox = vm
    vm.calls.clear()

    await gp.read_trajectory(vm)
    await gp.install_changelog_triggers("slack")
    await gp._down_services(vm)

    slot_compose = ("sudo", "docker", "compose", "-f", "/app/s1/docker-compose.yml", "-p", "s1")
    assert _execs(vm)[0] == (*slot_compose, "ps", "-a", "-q", "gateway")
    assert _execs(vm)[-2][: len(slot_compose) + 3] == (*slot_compose, "exec", "-T", "servicedb")
    assert _execs(vm)[-1] == (*slot_compose, "ps", "-a", "--format", "json")


@pytest.mark.asyncio
async def test_slot_snapshot_uses_its_own_compose_and_scratch_files():
    vm = _RecordingVm([], set())
    object_store = MagicMock()
    object_store.signed_put_url.return_value = "https://put"
    config = MagicMock(get_object_store_for=lambda _env_id: object_store, get_artifact_key_prefix=lambda: "")
    with patch.object(snapshot_store, "get_config", return_value=config):
        await snapshot_store._snapshot_servicedb(vm, "env1", "uni1", lambda *a: None, layout=StackLayout.for_slot(0))

    scripts = _scripts(vm)
    assert scripts[0] == "docker compose -f /app/s0/docker-compose.yml -p s0 ps -a -q servicedb"
    assert all("/tmp/pgdata" not in s and "/tmp/snapshot-" not in s for s in scripts)
    assert "/tmp/s0-pgdata" in scripts[1] and "/tmp/s0-snapshot-build" in scripts[2]
    assert scripts[-1].startswith("rm -rf /tmp/s0-snapshot-build /tmp/s0-snapshot-image.tar.gz")


async def _snapshot_image_commands(layout: StackLayout) -> list[str]:
    vm = _RecordingVm([], set())
    object_store = MagicMock()
    object_store.signed_put_url.return_value = "https://put"
    config = MagicMock(get_object_store_for=lambda _env_id: object_store, get_artifact_key_prefix=lambda: "")
    with patch.object(snapshot_store, "get_config", return_value=config):
        await snapshot_store._snapshot_servicedb(vm, "env1", "uni1", lambda *a: None, layout=layout)
    return [s for s in _scripts(vm) if s.startswith(("docker build", "docker save")) or "docker rmi" in s]


@pytest.mark.asyncio
async def test_default_snapshot_image_tag_is_unchanged():
    build, save, rmi = await _snapshot_image_commands(DEFAULT_STACK_LAYOUT)
    assert build == "docker build --platform linux/amd64 -t env-snapshot-env1-uni1 /tmp/snapshot-build"
    assert save.startswith("docker save env-snapshot-env1-uni1 |")
    assert rmi.endswith("docker rmi env-snapshot-env1-uni1")


@pytest.mark.asyncio
async def test_two_slots_snapshotting_one_env_and_universe_build_different_images():
    s0 = await _snapshot_image_commands(StackLayout.for_slot(0))
    s1 = await _snapshot_image_commands(StackLayout.for_slot(1))
    assert all("env-snapshot-env1-uni1-s0" in c for c in s0)
    assert all("env-snapshot-env1-uni1-s1" in c for c in s1)


def test_default_layout_is_the_single_stack_paths_and_ports():
    assert DEFAULT_STACK_LAYOUT.compose_command == "docker compose -f /app/docker-compose.yml"
    assert DEFAULT_STACK_LAYOUT.compose_command_from_app_dir == "docker compose"
    assert DEFAULT_STACK_LAYOUT.artifact_staging_dir == "/app/_artifact_staging"
    assert DEFAULT_STACK_LAYOUT.scratch_path("pgdata") == "/tmp/pgdata"
    assert DEFAULT_STACK_LAYOUT.host_port(AGENT_ENV_GATEWAY_MCP_PORT) == AGENT_ENV_GATEWAY_MCP_PORT


def test_slots_get_disjoint_host_port_blocks_from_the_default_and_each_other():
    ports = (AGENT_ENV_GATEWAY_MCP_PORT, DB_WEB_PORT, DB_MCP_PORT)
    layouts = [DEFAULT_STACK_LAYOUT, *(StackLayout.for_slot(i) for i in range(8))]
    published = [layout.host_port(p) for layout in layouts for p in ports]
    assert len(set(published)) == len(published)


def test_for_slot_derives_its_project_dir_scratch_and_ports_from_the_index():
    slot = StackLayout.for_slot(3)
    assert slot.compose_project == "s3"
    assert slot.compose_command == "docker compose -f /app/s3/docker-compose.yml -p s3"
    assert slot.artifact_staging_dir == "/app/s3/_artifact_staging"
    assert slot.scratch_path("pgdata") == "/tmp/s3-pgdata"
    assert slot.host_port(AGENT_ENV_GATEWAY_MCP_PORT) == AGENT_ENV_GATEWAY_MCP_PORT + 4000


def test_for_slot_rejects_a_negative_index():
    with pytest.raises(ValueError):
        StackLayout.for_slot(-1)


def test_slot_45_is_the_highest_whose_gateway_port_fits():
    assert StackLayout.for_slot(45).host_port(AGENT_ENV_GATEWAY_MCP_PORT) <= 65535
    with pytest.raises(ValueError):
        StackLayout.for_slot(46).host_port(AGENT_ENV_GATEWAY_MCP_PORT)


@pytest.mark.parametrize("project", ["", "S0", "-s0", "s 0", "s/0", "app"])
def test_layout_rejects_a_bad_project_or_the_default_stacks_implicit_one(project):
    with pytest.raises(ValueError):
        StackLayout(app_dir="/app/x", compose_project=project)


@pytest.mark.parametrize("app_dir", ["app", "/app/../etc", "/app/x;rm -rf /", "/app/$(id)", "/app/a b", "/app/'x'"])
def test_layout_rejects_an_app_dir_unsafe_in_a_shell_script(app_dir):
    with pytest.raises(ValueError):
        StackLayout(app_dir=app_dir)


@pytest.mark.parametrize("scratch_prefix", ["tmp/", "/tmp/$x-", "/tmp/a b-", "/tmp/x;", "/tmp/../"])
def test_layout_rejects_a_scratch_prefix_unsafe_in_a_shell_script(scratch_prefix):
    with pytest.raises(ValueError):
        StackLayout(scratch_prefix=scratch_prefix)


@pytest.mark.parametrize("offset", [-1000, 1, 999])
def test_layout_rejects_an_offset_off_the_slot_stride(offset):
    with pytest.raises(ValueError):
        StackLayout(host_port_offset=offset)


def test_host_port_rejects_a_shift_past_the_port_range():
    with pytest.raises(ValueError):
        StackLayout.for_slot(60).host_port(AGENT_ENV_GATEWAY_MCP_PORT)


@pytest.mark.parametrize("changes", [
    {"app_dir": "/app/s0"}, {"host_port_offset": 1000}, {"scratch_prefix": "/tmp/s0-"},
])
def test_a_layout_that_differs_from_the_default_must_name_its_project(changes):
    with pytest.raises(ValueError, match="must name its compose project"):
        StackLayout(**changes)


def test_only_a_named_project_makes_a_slot():
    assert not DEFAULT_STACK_LAYOUT.is_slot
    assert StackLayout.for_slot(0).is_slot


@pytest.mark.parametrize("index", [0, 3, 45])
def test_a_slot_layout_comes_back_the_same_from_its_record(index):
    layout = StackLayout.for_slot(index)
    assert StackLayout.from_record(layout.to_record()) == layout


def test_the_default_stack_records_nothing_and_a_record_without_a_layout_is_the_default():
    assert DEFAULT_STACK_LAYOUT.to_record() is None
    assert StackLayout.from_record(None) is DEFAULT_STACK_LAYOUT


@pytest.mark.parametrize("data", [
    "s1",
    {"app_dir": "/app/s1", "compose_project": "s1", "host_port_offset": 2000},
    {**StackLayout.for_slot(1).to_record(), "extra": 1},
    {**StackLayout.for_slot(1).to_record(), "host_port_offset": True},
    {**StackLayout.for_slot(1).to_record(), "host_port_offset": "2000"},
    {**StackLayout.for_slot(1).to_record(), "app_dir": "/app/../etc"},
    {"app_dir": "/app", "compose_project": None, "host_port_offset": 0, "scratch_prefix": "/tmp/"},
])
def test_a_damaged_recorded_layout_is_refused_rather_than_read_as_the_default_stack(data):
    with pytest.raises((ValueError, TypeError)):
        StackLayout.from_record(data)


def test_the_default_stack_is_never_torn_down_on_its_own():
    with pytest.raises(ValueError, match="default stack"):
        DEFAULT_STACK_LAYOUT.teardown_script


@pytest.mark.parametrize("layout", [
    StackLayout(compose_project="s1"),
    StackLayout(app_dir="/app/s2", compose_project="s1", scratch_prefix="/tmp/s1-"),
    StackLayout(app_dir="/app/s1", compose_project="s1", scratch_prefix="/tmp/"),
    StackLayout(app_dir="/app/s1", compose_project="s1", scratch_prefix="/tmp/s1"),
    StackLayout(app_dir="/app/s1-x", compose_project="s1-x", scratch_prefix="/tmp/s1-x-", host_port_offset=2000),
    StackLayout(app_dir="/app/s1", compose_project="s1", scratch_prefix="/tmp/s1-", host_port_offset=5000),
    StackLayout(app_dir="/app/web", compose_project="web", scratch_prefix="/tmp/web-", host_port_offset=1000),
    StackLayout(app_dir="/app/s01", compose_project="s01", scratch_prefix="/tmp/s01-", host_port_offset=2000),
])
def test_a_slot_layout_not_made_by_for_slot_is_never_torn_down(layout):
    with pytest.raises(ValueError, match="for_slot"):
        layout.teardown_script


def test_a_slots_teardown_names_only_its_own_project_dir_and_scratch_files():
    script = StackLayout.for_slot(7).teardown_script
    assert script.count("label=com.docker.compose.project=s7") == 3
    assert script.splitlines()[-1] == "rm -rf /app/s7 /tmp/s7-*"


@pytest.mark.asyncio
async def test_tearing_down_a_slot_runs_its_script_on_the_vm_and_may_retry():
    vm = _RecordingVm([], set())
    vm.exec_script = AsyncMock(return_value="")
    await tear_down_slot(vm, StackLayout.for_slot(2))
    vm.exec_script.assert_awaited_once_with(StackLayout.for_slot(2).teardown_script, max_retries=TEARDOWN_RETRIES)


@pytest.mark.asyncio
async def test_a_failed_slot_teardown_raises_so_the_slot_is_never_counted_gone():
    vm = _RecordingVm([], set())
    vm.exec_script = AsyncMock(side_effect=RuntimeError("Script failed (exit 1)"))
    with pytest.raises(RuntimeError, match="exit 1"):
        await tear_down_slot(vm, StackLayout.for_slot(2))


@pytest.mark.asyncio
async def test_only_a_slot_on_a_vm_can_be_torn_down():
    vm = _RecordingVm([], set())
    with pytest.raises(ValueError, match="default stack"):
        await tear_down_slot(vm, DEFAULT_STACK_LAYOUT)
    vm.mode = "container"
    with pytest.raises(ValueError, match="only be on a VM"):
        await tear_down_slot(vm, StackLayout.for_slot(2))
    assert vm.calls == []
