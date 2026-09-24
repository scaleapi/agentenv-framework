"""Unit tests for GatewayProvider container-mode deploy."""

from __future__ import annotations

import asyncio
import logging
import re
import socket
from datetime import datetime

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP
from pytest_socket import enable_socket

from agent_env.env.envs.service_db import SERVICE_DB_PORT
from agent_env.providers import gateway_provider
from agent_env.providers.gateway_provider import (
    DeployedGateway,
    GatewayProvider,
    MCPServerConfig,
    SidecarConfig,
    WebsiteConfig,
    _mcp_url,
    _redact_compose_secrets,
)
from agent_env.providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox import Sandbox
from agent_env.providers.state import LocalPostgresStateProvider
from agent_env.config import set_document_store
from tst.unit.providers.state.fakes import (  # noqa: F401  (fixture)
    EXTERNAL_READINESS_SERVICE,
    EXTERNAL_STATE_TYPE,
    registered_external_provider,
)


@pytest.fixture(autouse=True)
def _fake_env_state_store():
    """Keep acquire()'s persist step off a real backend — real store over an in-memory DocumentStore."""
    from agent_env.providers.state import (
        EnvStateInstanceStore,
        reset_env_state_instance_store,
        set_env_state_instance_store,
    )
    from tst.unit.store.fakes import FakeDocumentStore

    store = EnvStateInstanceStore()
    set_document_store(FakeDocumentStore())
    set_env_state_instance_store(store)
    yield store
    reset_env_state_instance_store()


_CARD = {"name": "env1234", "additionalInterfaces": [{"url": "/mcp", "transport": "mcp"}]}

class _FakeSandbox(Sandbox):
    type = "modal"

    _i6pn_counter = 0

    def __init__(self, sandbox_id: str, port: int, endpoint: str):
        self.sandbox_id = sandbox_id
        self.tunnel_urls = {port: endpoint}
        self.vnc_url = None
        self.mode = "container"
        _FakeSandbox._i6pn_counter += 1
        self.i6pn_address = f"fdaa::fake:{_FakeSandbox._i6pn_counter:x}"
        self.terminated = False
        self.exec_calls: list[tuple] = []
        self.write_calls: list[tuple] = []

    async def terminate(self) -> None:
        self.terminated = True

    async def exec_with_output(self, *args: str):
        self.exec_calls.append(args)
        # pg_isready returns 0; psql -f returns 0
        return (0, "", "")

    async def write_file_from_text(self, content: str, destination_path: str) -> None:
        self.write_calls.append((destination_path, content))


@pytest.mark.asyncio
async def test_container_mode_rejects_websites():
    provider = ModalSandboxProvider()
    gp = GatewayProvider()
    with pytest.raises(NotImplementedError, match="websites"):
        await gp._deploy_via_containers(
            sandbox_provider=provider,
            mcp_servers=[],
            mcp_server_images=[],
            gateway_port=18765,
            website_configs=[WebsiteConfig(backend_image="b", frontend_image="f", environment_name="x")],
            gateway_mode=MagicMock(value="performance"),
            ttl_seconds=60,
            disk_size_gb=10,
        )


@pytest.mark.asyncio
async def test_container_mode_constructs_internal_mcp_servers_url_format():
    """Container mode must pass INTERNAL_MCP_SERVERS as 'name=url,name=url' to the gateway."""
    provider = ModalSandboxProvider()
    gp = GatewayProvider()
    gp._state_provider = LocalPostgresStateProvider()  # normally set by create_gateway

    # Track every create_container call so we can inspect the gateway env var
    create_calls: list[dict] = []
    fake_sandboxes = []

    async def fake_create_container(**kwargs):
        create_calls.append(kwargs)
        port = kwargs["port"]
        if kwargs.get("is_tcp"):
            sb = _FakeSandbox(f"db-{port}", port, f"tcp://db.modal.host:{port}")
        else:
            # MCP servers and gateway both use HTTPS; differentiate by image
            image = kwargs["image_name"]
            sb = _FakeSandbox(f"sb-{image}-{port}", port, f"https://{image}.modal.host")
        fake_sandboxes.append(sb)
        return sb

    # Spy on the provider's acquire() (wrapping the real impl) to assert wiring.
    acquire_calls: list = []
    _real_acquire = LocalPostgresStateProvider.acquire

    async def _spy_acquire(self, ctx):
        acquire_calls.append(ctx)
        return await _real_acquire(self, ctx)

    # Patch create_container on the provider
    with patch.object(provider, "create_container", side_effect=fake_create_container):
        # Patch _init_service_db_via_exec to skip real psql
        gp._init_service_db_via_exec = AsyncMock()
        # Patch _wait_for_tunnel to skip the real HTTP poll; it returns the card it read
        gp._wait_for_tunnel = AsyncMock(return_value=_CARD)
        # Patch Env.get and config to return mock envs
        with patch("agent_env.env.env.Env.get") as env_get, \
             patch("agent_env.config.get_config") as get_cfg, \
             patch.object(LocalPostgresStateProvider, "acquire", _spy_acquire):
            gateway_env = MagicMock()
            gateway_env.docker_image_artifact.image_name = "agent-gateway"
            db_env = MagicMock()
            db_env.to_config.return_value.db_image = "postgres:16-alpine"
            cfg = MagicMock()
            cfg.default_gateway_env_id = "gw-id"
            cfg.default_service_db_env_id = "db-id"
            get_cfg.return_value = cfg
            env_get.side_effect = lambda env_id, *a, **k: gateway_env if env_id == "gw-id" else db_env

            mcp_servers = [
                MCPServerConfig(image="mcp-slack", environment_name="slack"),
                MCPServerConfig(image="mcp-email", environment_name="email"),
            ]
            mcp_images = [MagicMock(image_name="mcp-slack"), MagicMock(image_name="mcp-email")]

            from agent_env.env.gateway import GatewayMode
            result = await gp._deploy_via_containers(
                sandbox_provider=provider,
                mcp_servers=mcp_servers,
                mcp_server_images=mcp_images,
                gateway_port=18765,
                website_configs=None,
                gateway_mode=GatewayMode.PERFORMANCE,
                ttl_seconds=60,
                disk_size_gb=10,
                mcp_server_name="crm",
            )

    # Last create_container call is the gateway. Inspect its env var.
    gateway_call = create_calls[-1]
    env = gateway_call["env"]
    assert "INTERNAL_MCP_SERVERS" in env
    spec = env["INTERNAL_MCP_SERVERS"]
    # With i6pn enabled (the default), gateway-to-MCP URLs use i6pn addresses.
    import re
    assert re.search(r"slack=http://\[fdaa::fake:[0-9a-f]+\]:\d+/mcp", spec), spec
    assert re.search(r"email=http://\[fdaa::fake:[0-9a-f]+\]:\d+/mcp", spec), spec
    # REST_PROXY_URLS routes /svc/mcp-<name>/<path> to each MCP's base URL (no /mcp suffix).
    rest = env["REST_PROXY_URLS"]
    assert re.search(r"mcp-slack=http://\[fdaa::fake:[0-9a-f]+\]:\d+(?!/mcp)", rest), rest
    assert re.search(r"mcp-email=http://\[fdaa::fake:[0-9a-f]+\]:\d+(?!/mcp)", rest), rest
    assert "/mcp," not in rest and not rest.endswith("/mcp"), rest
    assert env["GATEWAY_MODE"] == GatewayMode.PERFORMANCE.value
    assert re.search(r"^postgresql://agentenv:agentenv@\[fdaa::fake:[0-9a-f]+\]:5432/agentenv", env["SERVICE_DB_URL"]), env["SERVICE_DB_URL"]
    assert env["MCP_SERVER_NAME"] == "crm"
    assert result.mcp_server_name == "crm"

    # Each MCP got DATABASE_URL with its scoped search_path
    mcp_calls = [c for c in create_calls if c["image_name"].startswith("mcp-")]
    assert len(mcp_calls) == 2
    assert "search_path%3D%22slack%22" in mcp_calls[0]["env"]["DATABASE_URL"]
    assert "search_path%3D%22email%22" in mcp_calls[1]["env"]["DATABASE_URL"]

    assert len(gp._container_sandboxes) == 4

    # PR 2: the deploy routes DB state through the provider.
    # servicedb container env comes from the provider's store-spec
    db_call = next(c for c in create_calls if c["port"] == SERVICE_DB_PORT)
    assert db_call["env"] == LocalPostgresStateProvider().store_spec(["slack", "email"]).env
    # acquire() runs once, after the db container exists; its context carries the
    # universe's services and the i6pn host the compute layer stood up.
    assert len(acquire_calls) == 1
    assert acquire_calls[0].environment_names == ["slack", "email"]
    assert re.search(r"^\[fdaa::fake:[0-9a-f]+\]$", acquire_calls[0].host), acquire_calls[0].host
    # The deploy records the EnvStateInstance id it acquired on the result (forward pointer
    # the env deploy path copies onto DeployedEnv.env_state_instance_ids).
    assert len(result.env_state_instance_ids) == 1
    assert result.env_state_instance_ids[0].startswith("esi-")
    # db sandbox slot populated (MultiEnv bookkeeping / from_deployed_env restore)
    assert gp._db_sandbox is not None

    # Result has the gateway URL but no db_web/db_mcp
    assert result.gateway_url == "https://agent-gateway.modal.host"
    assert result.mcp_url == "https://agent-gateway.modal.host/mcp"
    # The card the readiness probe read is the card the deploy records
    gp._wait_for_tunnel.assert_awaited_once_with("https://agent-gateway.modal.host", timeout=120)
    assert result.environment_card == _CARD
    assert datetime.fromisoformat(result.environment_card_read_at_utc).utcoffset().total_seconds() == 0
    assert result.db_web_url is None
    assert result.db_mcp_url is None


@pytest.mark.asyncio
async def test_container_mode_remote_skips_servicedb_and_points_at_remote():
    """Modal/container path with a pre-acquired EXTERNAL instance (remote Postgres): no servicedb
    container is stood up, and the MCP + gateway containers connect out to the remote run-db handle."""
    from agent_env.env.gateway import GatewayMode
    from agent_env.providers.state import EnvStateInstance
    from tst.unit.providers.state.fakes import ExternalDbStateProvider

    provider = ModalSandboxProvider()
    gp = GatewayProvider()
    # Simulate create_gateway having consumed a pre-acquired external instance.
    db_url_base = "postgresql://runrole_x:pw@rds.example.internal:5432/run_x?sslmode=require"
    gp._state_instance = EnvStateInstance(state_type=EXTERNAL_STATE_TYPE, _db_url_base=db_url_base)
    gp._state_provider = ExternalDbStateProvider()

    create_calls: list[dict] = []

    async def fake_create_container(**kwargs):
        create_calls.append(kwargs)
        image = kwargs["image_name"]
        return _FakeSandbox(f"sb-{image}-{kwargs['port']}", kwargs["port"], f"https://{image}.modal.host")

    with patch.object(provider, "create_container", side_effect=fake_create_container):
        gp._wait_for_tunnel = AsyncMock(return_value=_CARD)
        with patch("agent_env.env.env.Env.get") as env_get, \
             patch("agent_env.config.get_config") as get_cfg:
            gateway_env = MagicMock()
            gateway_env.docker_image_artifact.image_name = "agent-gateway"
            db_env = MagicMock()
            db_env.to_config.return_value.db_image = "postgres:16-alpine"
            cfg = MagicMock()
            cfg.default_gateway_env_id = "gw-id"
            cfg.default_service_db_env_id = "db-id"
            get_cfg.return_value = cfg
            env_get.side_effect = lambda env_id, *a, **k: gateway_env if env_id == "gw-id" else db_env

            await gp._deploy_via_containers(
                sandbox_provider=provider,
                mcp_servers=[MCPServerConfig(image="mcp-slack", environment_name="slack")],
                mcp_server_images=[MagicMock(image_name="mcp-slack")],
                gateway_port=18765,
                website_configs=None,
                gateway_mode=GatewayMode.PERFORMANCE,
                ttl_seconds=60,
                disk_size_gb=10,
            )

    # No servicedb container was stood up (no container on the servicedb port), and no db sandbox.
    assert all(c["port"] != SERVICE_DB_PORT for c in create_calls)
    assert gp._db_sandbox is None
    # Only the MCP server + gateway containers exist (no servicedb, no sidecars).
    assert len(gp._container_sandboxes) == 2
    # The MCP server's DATABASE_URL is the remote base scoped to its schema.
    mcp_call = next(c for c in create_calls if c["image_name"] == "mcp-slack")
    assert mcp_call["env"]["DATABASE_URL"] == gp._state_provider.url_for_environment("slack", instance=gp._state_instance)
    assert "search_path%3D%22slack%22" in mcp_call["env"]["DATABASE_URL"]
    # The gateway points at the remote base (not an i6pn servicedb address).
    gateway_call = create_calls[-1]
    assert gateway_call["env"]["SERVICE_DB_URL"] == db_url_base



@pytest.mark.asyncio
async def test_container_mode_mcp_url_follows_the_card():
    """The recorded MCP URL is the one the gateway's card declares, joined onto the gateway URL."""
    from agent_env.env.gateway import GatewayMode
    from agent_env.providers.state import EnvStateInstance
    from tst.unit.providers.state.fakes import ExternalDbStateProvider

    provider = ModalSandboxProvider()
    gp = GatewayProvider()
    gp._state_instance = EnvStateInstance(state_type=EXTERNAL_STATE_TYPE, _db_url_base="postgresql://db.example.internal:5432/run")
    gp._state_provider = ExternalDbStateProvider()
    card = {"name": "env1234", "additionalInterfaces": [{"url": "/custom-mcp", "transport": "mcp"}]}

    async def fake_create_container(**kwargs):
        image = kwargs["image_name"]
        return _FakeSandbox(f"sb-{image}", kwargs["port"], f"https://{image}.modal.host")

    with patch.object(provider, "create_container", side_effect=fake_create_container):
        gp._wait_for_tunnel = AsyncMock(return_value=card)
        with patch("agent_env.env.env.Env.get") as env_get, patch("agent_env.config.get_config") as get_cfg:
            env_get.return_value.docker_image_artifact.image_name = "agent-gateway"
            get_cfg.return_value = MagicMock(default_gateway_env_id="gw-id")
            result = await gp._deploy_via_containers(
                sandbox_provider=provider,
                mcp_servers=[MCPServerConfig(image="mcp-slack", environment_name="slack")],
                mcp_server_images=[MagicMock(image_name="mcp-slack")],
                gateway_port=18765, website_configs=None, gateway_mode=GatewayMode.PERFORMANCE,
                ttl_seconds=60, disk_size_gb=10,
            )
    assert result.mcp_url == "https://agent-gateway.modal.host/custom-mcp"
    assert result.environment_card == card



@pytest.mark.parametrize("gateway_url,declared,expected", [
    ("https://gw.example", None, "https://gw.example/mcp"),
    ("https://gw.example", "/custom-mcp", "https://gw.example/custom-mcp"),
    ("https://gw.example", "custom-mcp", "https://gw.example/custom-mcp"),
    ("https://gw.example/", "/mcp", "https://gw.example/mcp"),
    # a sandbox proxy's path prefix survives, which URL resolution would drop
    ("https://sandbox.example/sandbox/sb-01abc-18765", "/mcp", "https://sandbox.example/sandbox/sb-01abc-18765/mcp"),
], ids=["undeclared", "declared", "relative", "trailing-slash", "path-prefix"])
def test_mcp_url_joins_the_declared_path_onto_the_gateway_url(gateway_url, declared, expected):
    card = {"name": "env1234", "additionalInterfaces": [{"url": declared, "transport": "mcp"}] if declared else []}
    assert _mcp_url(gateway_url, card) == expected

async def _probe(monkeypatch, *replies, timeout: int = 3) -> tuple[dict | None, list[str]]:
    """Run _wait_for_tunnel against scripted replies (a Response factory or an exception per attempt; the last repeats)."""
    real_client, seen, queue = httpx.AsyncClient, [], list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return reply()

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real_client(transport=httpx.MockTransport(handler)))
    monkeypatch.setattr("agent_env.providers.gateway_provider.asyncio.sleep", AsyncMock())
    return await GatewayProvider()._wait_for_tunnel("https://gw.example", timeout=timeout), seen


@pytest.mark.asyncio
async def test_wait_for_tunnel_returns_the_card_it_reads(monkeypatch):
    card, seen = await _probe(monkeypatch, lambda: httpx.Response(200, json=_CARD))
    assert card == _CARD
    assert seen == ["https://gw.example/.well-known/agent-env.json"]


@pytest.mark.parametrize("reply", [
    lambda: httpx.Response(404, text="Not Found"),
    lambda: httpx.Response(200, text="<html>sign in</html>", headers={"content-type": "text/html"}),
    lambda: httpx.Response(200, json=[_CARD]),
    lambda: httpx.Response(200, json={"protocolVersion": "1.0"}),
    lambda: httpx.Response(302, headers={"location": "https://login.example"}),
], ids=["404", "html", "json-list", "no-name", "redirect"])
@pytest.mark.asyncio
async def test_wait_for_tunnel_accepts_only_a_card(monkeypatch, reply):
    card, seen = await _probe(monkeypatch, reply)
    assert card is None and len(seen) == 3


@pytest.mark.asyncio
async def test_wait_for_tunnel_polls_through_connection_errors(monkeypatch):
    card, seen = await _probe(monkeypatch, httpx.ConnectError("refused"), lambda: httpx.Response(200, json=_CARD))
    assert card == _CARD and len(seen) == 2

async def _run_container_deploy(*, cpu=None, memory_mb=None) -> list[dict]:
    """Run _deploy_via_containers with all I/O mocked, returning the list of
    create_container kwargs (one per provisioned component: db, mcp, pgweb,
    db-mcp, gateway). ECR-style db images are used so the pgweb/db-mcp aux
    services are provisioned too."""
    from agent_env.env.gateway import GatewayMode

    provider = ModalSandboxProvider()
    gp = GatewayProvider()
    gp._state_provider = LocalPostgresStateProvider()  # normally set by create_gateway
    create_calls: list[dict] = []

    async def fake_create_container(**kwargs):
        create_calls.append(kwargs)
        port = kwargs["port"]
        image = kwargs["image_name"]
        if kwargs.get("is_tcp"):
            return _FakeSandbox(f"db-{port}", port, f"tcp://db.modal.host:{port}")
        return _FakeSandbox(f"sb-{image}-{port}", port, f"https://{image}.modal.host")

    with patch.object(provider, "create_container", side_effect=fake_create_container):
        gp._init_service_db_via_exec = AsyncMock()
        gp._wait_for_tunnel = AsyncMock(return_value=_CARD)
        with patch("agent_env.env.env.Env.get") as env_get, \
             patch("agent_env.config.get_config") as get_cfg:
            gateway_env = MagicMock()
            gateway_env.docker_image_artifact.image_name = "agent-gateway"
            db_env = MagicMock()
            db_cfg = db_env.to_config.return_value
            db_cfg.db_image = "123.dkr.ecr.us-west-2.amazonaws.com/postgres"
            db_cfg.db_web_image = "123.dkr.ecr.us-west-2.amazonaws.com/pgweb"
            db_cfg.db_mcp_image = "123.dkr.ecr.us-west-2.amazonaws.com/db-mcp"
            cfg = MagicMock()
            cfg.default_gateway_env_id = "gw-id"
            cfg.default_service_db_env_id = "db-id"
            get_cfg.return_value = cfg
            env_get.side_effect = lambda env_id, *a, **k: gateway_env if env_id == "gw-id" else db_env

            await gp._deploy_via_containers(
                sandbox_provider=provider,
                mcp_servers=[MCPServerConfig(image="mcp-slack", environment_name="slack")],
                mcp_server_images=[MagicMock(image_name="mcp-slack")],
                gateway_port=18765,
                website_configs=None,
                gateway_mode=GatewayMode.PERFORMANCE,
                ttl_seconds=60,
                disk_size_gb=10,
                cpu=cpu,
                memory_mb=memory_mb,
            )
    return create_calls


@pytest.mark.asyncio
async def test_deploy_level_size_tunes_the_gateway_only():
    """cpu/memory_mb are the gateway's; the per-service sandboxes have their own
    parameters, and with none set they reach the provider unsized."""
    calls = await _run_container_deploy(cpu=1.0, memory_mb=2048)
    gateway = next(c for c in calls if c["image_name"] == "agent-gateway")
    mcp = next(c for c in calls if c["image_name"] == "mcp-slack")
    db = next(c for c in calls if c["image_name"].endswith("postgres"))
    aux = [c for c in calls if c["image_name"].endswith(("pgweb", "db-mcp"))]
    assert (gateway["cpu"], gateway["memory"]) == (1.0, 2048)
    for call in (mcp, db, *aux):
        assert call["cpu"] == 1.0 and "memory" not in call


@pytest.mark.asyncio
async def test_the_deploys_cpu_reaches_every_container():
    """A caller sizes the deploy through the existing cpu/memory_mb; nothing is baked in."""
    calls = await _run_container_deploy(cpu=2.0, memory_mb=2048)
    gateway = next(c for c in calls if c["image_name"] == "agent-gateway")
    mcp = next(c for c in calls if c["image_name"] == "mcp-slack")
    db = next(c for c in calls if c["image_name"].endswith("postgres"))
    aux = [c for c in calls if c["image_name"].endswith(("pgweb", "db-mcp"))]
    assert (gateway["cpu"], gateway["memory"]) == (2.0, 2048)
    # One cpu for the deploy: every container in it takes that size, memory only the gateway.
    assert mcp["cpu"] == 2.0 and "memory" not in mcp
    assert db["cpu"] == 2.0 and "memory" not in db
    assert aux and all(c["cpu"] == 2.0 and "memory" not in c for c in aux)


@pytest.mark.asyncio
async def test_nothing_set_means_nothing_sent():
    """An unsized deploy must reach the provider with no cpu at all, so its per-backend
    floor applies. Passing None would instead mean "no reservation" to the backend."""
    calls = await _run_container_deploy()
    gateway = next(c for c in calls if c["image_name"] == "agent-gateway")
    mcp = next(c for c in calls if c["image_name"] == "mcp-slack")
    for call in (gateway, mcp):
        assert "cpu" not in call and "memory" not in call


@pytest.mark.asyncio
async def test_close_terminates_all_container_sandboxes():
    gp = GatewayProvider()
    fake1 = _FakeSandbox("sb1", 8000, "https://a.modal.host")
    fake2 = _FakeSandbox("sb2", 5432, "tcp://b.modal.host:5432")
    gp._container_sandboxes = [fake1, fake2]
    await gp.close()
    assert fake1.terminated
    assert fake2.terminated
    assert gp._container_sandboxes == []


@pytest.mark.asyncio
async def test_modal_vm_provider_routes_to_vm_path_not_containers():
    """A ModalVmSandboxProvider must dispatch to _deploy_via_vm (one VM + docker-compose),
    NOT _deploy_via_containers. This is the whole point of the standalone class: it is not a
    ModalSandboxProvider, so create_gateway's isinstance check falls through to the VM path
    and the i6pn (container-only) gate never fires."""
    from agent_env.env.gateway import GatewayMode
    from agent_env.providers.modal_vm_sandbox import ModalVmSandboxProvider

    provider = ModalVmSandboxProvider()
    gp = GatewayProvider()
    gp._deploy_via_vm = AsyncMock(return_value="VM_RESULT")
    gp._deploy_via_containers = AsyncMock(return_value="CONTAINER_RESULT")

    result = await gp.create_gateway(
        sandbox_provider=provider,
        mcp_servers=[MCPServerConfig(image="mcp-a", environment_name="a")],
        mcp_server_images=[MagicMock(image_name="mcp-a")],
        gateway_mode=GatewayMode.PERFORMANCE,
        ttl_seconds=60,
        disk_size_gb=10,
    )
    assert result == "VM_RESULT"
    gp._deploy_via_vm.assert_awaited_once()
    gp._deploy_via_containers.assert_not_awaited()


@pytest.mark.asyncio
async def test_build_local_store_no_services():
    """No services at all (and website_configs left unset, as the container path calls it) must
    self-build local Postgres without raising — guards the empty service list + website_configs=None
    default. The instance/provider land on self; the spec is returned."""
    from agent_env.providers.state import EnvStateInstance, LOCAL_POSTGRES_STATE_TYPE

    gp = GatewayProvider()
    gp._state_provider = LocalPostgresStateProvider()  # normally set by create_gateway

    async def _stand_up(spec):
        return "servicedb"

    spec = await gp._build_local_store([], _stand_up)
    assert isinstance(gp._state_instance, EnvStateInstance)
    assert gp._state_instance.state_type == LOCAL_POSTGRES_STATE_TYPE
    assert gp._state_instance.metadata == {"host": "servicedb"}


@pytest.mark.asyncio
async def test_build_local_store_dedups_service_names_from_mcp_and_websites():
    """environment_names are derived inside _build_local_store from BOTH MCP servers and websites
    (deduplicated by environment_name) and passed to store_spec — one schema per distinct name."""
    gp = GatewayProvider()
    gp._state_provider = LocalPostgresStateProvider()  # normally set by create_gateway
    seen_names: list = []
    _real_store_spec = LocalPostgresStateProvider.store_spec

    def _spy_store_spec(self, names):
        seen_names.append(names)
        return _real_store_spec(self, names)

    async def _stand_up(spec):
        return "servicedb"

    with patch.object(LocalPostgresStateProvider, "store_spec", _spy_store_spec):
        await gp._build_local_store(
            [MCPServerConfig(image="mcp-slack", environment_name="slack")],
            _stand_up,
            website_configs=[
                WebsiteConfig(backend_image="shop-be", frontend_image="shop-fe", environment_name="shop"),
                # shares a environment_name with the MCP server -> deduped
                WebsiteConfig(backend_image="slack-be", frontend_image="slack-fe", environment_name="slack"),
            ],
        )
    assert seen_names == [["slack", "shop"]]


@pytest.mark.asyncio
async def test_install_changelog_triggers_delegates_to_provider():
    """The gateway shim delegates changelog install to the state provider, supplying a
    store_exec that runs the provider's psql argv against the db sandbox (container mode)."""
    from agent_env.env.envs.service_db import DB_NAME, DB_USER

    gp = GatewayProvider()
    gp._state_provider = LocalPostgresStateProvider()
    gp._state_instance = LocalPostgresStateProvider.default_instance()
    db_sb = _FakeSandbox("db-5432", SERVICE_DB_PORT, "tcp://db.modal.host:5432")
    gp._db_sandbox = db_sb

    await gp.install_changelog_triggers("slack")

    # The provider built the psql argv; the gateway's store_exec ran it on the db sandbox.
    assert db_sb.exec_calls == [
        ("psql", "-U", DB_USER, "-d", DB_NAME, "-c", "SELECT _install_changelog_triggers('slack')"),
    ]


@pytest.mark.asyncio
async def test_install_changelog_triggers_noop_without_state():
    """No implicit local default: if no state provider/instance is set (e.g. a standalone env
    reattached without acquiring a store), install is a no-op — it skips rather than guessing a
    local backend or raising (a store-less deploy has no changelog to install)."""
    # Must not raise and must not touch a store (no sandbox is wired).
    await GatewayProvider().install_changelog_triggers("slack")


def test_create_docker_compose_requires_explicit_provider():
    """provider is a required keyword-only arg — no implicit Local default — so every caller
    (live deploy, export, snapshot) declares its backend at the call site."""
    with pytest.raises(TypeError):
        GatewayProvider().create_docker_compose(mcp_servers=[])


# --- remote_postgres backend wiring -------------------------------------------

@pytest.mark.asyncio
async def test_install_changelog_triggers_noop_when_no_state_provider_but_sandbox_wired():
    """A db sandbox is wired but no state provider => still a no-op, NOT a silent local-Postgres
    assumption (which would be wrong for a remote backend). Nothing runs against the store."""
    gp = GatewayProvider()  # _state_provider stays None
    db_sb = _FakeSandbox("db-5432", SERVICE_DB_PORT, "tcp://db.modal.host:5432")
    gp._db_sandbox = db_sb

    await gp.install_changelog_triggers("slack")  # must not raise

    assert db_sb.exec_calls == []  # nothing ran — no fallback to local psql


def test_compose_remote_renders_probe_and_points_at_remote():
    """External backend: no servicedb/pgweb/db-mcp store containers, but the backend's OWN readiness
    probe is rendered, clients depend_on it, and per-service DATABASE_URL + gateway SERVICE_DB_URL
    point at the remote run-db handle."""
    from agent_env.env.envs.service_db import ServiceDBConfig
    from agent_env.providers.state import EnvStateInstance
    from tst.unit.providers.state.fakes import ExternalDbStateProvider

    provider = ExternalDbStateProvider()
    db_url_base = "postgresql://runrole_x:pw@rds.example.internal:5432/run_x?sslmode=require"
    instance = EnvStateInstance(state_type=EXTERNAL_STATE_TYPE, _db_url_base=db_url_base)
    gp = GatewayProvider()
    compose = gp.create_docker_compose(
        mcp_servers=[MCPServerConfig(image="mcp-slack", environment_name="slack")],
        gateway_image="agent-gateway",
        gateway_port=18765,
        state_provider=provider,  # remote: renders no servicedb/sidecars regardless of config
        state_instance=instance,
    )
    # No local store or its direct-SQL sidecars — but the readiness probe IS rendered.
    assert "  servicedb:" not in compose
    assert "pgweb" not in compose and "db-mcp" not in compose
    assert f"  {EXTERNAL_READINESS_SERVICE}:" in compose
    # Clients gate on the probe via the same service_healthy mechanism as local.
    assert f"      {EXTERNAL_READINESS_SERVICE}:\n        condition: service_healthy" in compose
    # The MCP server's DATABASE_URL is the remote base scoped to its schema.
    assert f"DATABASE_URL={provider.url_for_environment('slack', instance=instance)}" in compose
    # Gateway points at the remote base.
    assert f"SERVICE_DB_URL={db_url_base}" in compose


def test_compose_local_still_renders_servicedb():
    """Sanity: a LocalPostgres provider still renders the local servicedb service + healthcheck."""
    from agent_env.env.envs.service_db import ServiceDBConfig

    gp = GatewayProvider()
    compose = gp.create_docker_compose(
        mcp_servers=[MCPServerConfig(image="mcp-slack", environment_name="slack")],
        gateway_image="agent-gateway",
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
        state_instance=LocalPostgresStateProvider.default_instance(),
    )
    assert "  servicedb:" in compose
    assert "condition: service_healthy" in compose


# --- sidecar rendering (e.g. the CUA controller on macOS) ---------------------

class _FakeImageArtifact:
    """Stand-in DockerImageArtifact: create_docker_compose only reads `.image_name`
    (the tag load_docker_images restores on the VM)."""
    def __init__(self, image_name):
        self.image_name = image_name


def _controller_sidecar():
    return SidecarConfig(
        environment_name="cua-controller",
        image_artifact=_FakeImageArtifact("ecr/cua/controller:1.0.4"),
        container_port=8000,
        host_port=18768,
        env={
            "CUA_MODE": "remote",
            "REMOTE_VM_IP": "https://mac.sandbox.example.com/abc",
            "REMOTE_SERVER_PORT": "5000",
            "CF_ACCESS_CLIENT_SECRET": "s3cr3t-token-value",
        },
    )


def _compose_with_controller_sidecar():
    """Render a compose exactly as the macOS CuaEnv deploy does: one MCP server plus the
    CUA controller sidecar carrying a secret CF Access token in its env."""
    from agent_env.env.envs.service_db import ServiceDBConfig

    gp = GatewayProvider()
    return gp.create_docker_compose(
        mcp_servers=[MCPServerConfig(image="mcp-cua", environment_name="cua")],
        gateway_image="agent-gateway",
        gateway_port=18765,
        sidecars=[_controller_sidecar()],
        # Explicit ServiceDBConfig so rendering never resolves the default-db env
        # from the store (which would hit the network — blocked in unit tests).
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
        state_instance=LocalPostgresStateProvider.default_instance(),
    )


def test_sidecar_image_is_the_artifact_image_name():
    """The compose image tag is the side-loaded artifact's image_name (so it matches what
    load_docker_images restores on the VM) — no external registry ref is invented."""
    assert _controller_sidecar().image == "ecr/cua/controller:1.0.4"


def test_sidecar_rendered_as_compose_service():
    """The sidecar becomes its own compose service: image, env (incl. the secret), a published
    host->container port, and the shared env-network. Its env rides `environment:`, not a cmdline."""
    compose = _compose_with_controller_sidecar()
    assert "  cua-controller:" in compose
    assert "    image: ecr/cua/controller:1.0.4" in compose
    assert "      - CUA_MODE=remote" in compose
    # Full https URL passed verbatim (the controller's RemoteProvider is URL-aware).
    assert "      - REMOTE_VM_IP=https://mac.sandbox.example.com/abc" in compose
    assert '      - "18768:8000"' in compose
    assert "restart: unless-stopped" in compose
    # The secret reaches the container via the compose environment block.
    assert "      - CF_ACCESS_CLIENT_SECRET=s3cr3t-token-value" in compose


def test_no_sidecars_renders_no_extra_service():
    """Linux CUA (and every non-mac env) passes no sidecars => no controller service, so the
    deploy is byte-identical to before this change."""
    from agent_env.env.envs.service_db import ServiceDBConfig

    gp = GatewayProvider()
    compose = gp.create_docker_compose(
        mcp_servers=[MCPServerConfig(image="mcp-cua", environment_name="cua")],
        gateway_image="agent-gateway",
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
        state_instance=LocalPostgresStateProvider.default_instance(),
    )
    assert "cua-controller" not in compose


def test_redact_compose_secrets_masks_only_secret_env_lines():
    """The logged compose masks *SECRET/PASSWORD/TOKEN env values but leaves everything else
    (image refs, ports, non-secret env, DATABASE_URL) intact."""
    compose = _compose_with_controller_sidecar()
    redacted = _redact_compose_secrets(compose)
    # secret value gone, key preserved
    assert "s3cr3t-token-value" not in redacted
    assert "      - CF_ACCESS_CLIENT_SECRET=***" in redacted
    # non-secret lines untouched
    assert "      - REMOTE_VM_IP=https://mac.sandbox.example.com/abc" in redacted
    assert "    image: ecr/cua/controller:1.0.4" in redacted
    assert '      - "18768:8000"' in redacted
    # the un-redacted compose still carries the real secret (only the log copy is masked)
    assert "s3cr3t-token-value" in compose


@pytest.mark.asyncio
async def test_acquire_state_for_deploy_local_returns_none():
    """Local Postgres cannot be pre-provisioned (its container's host doesn't exist yet), so the
    helper returns None and the gateway self-provisions it. Covers the default (env_state_type=None)."""
    from agent_env.providers.state import acquire_state_for_deploy

    assert await acquire_state_for_deploy() is None
    assert await acquire_state_for_deploy(env_state_type="local_postgres") is None


@pytest.mark.asyncio
async def test_create_gateway_consumes_external_instance(registered_external_provider):
    """create_gateway records a pre-acquired external instance up front — deriving the consume-only
    provider from its state_type — so the deploy paths + close() read it off self. It does NOT
    re-acquire, and it no longer threads the instance as a deploy-path kwarg."""
    from agent_env.env.gateway import GatewayMode
    from agent_env.providers.modal_vm_sandbox import ModalVmSandboxProvider
    from agent_env.providers.state import EnvStateInstance
    from tst.unit.providers.state.fakes import ExternalDbStateProvider

    gp = GatewayProvider()
    gp._deploy_via_vm = AsyncMock(return_value="VM_RESULT")
    external = EnvStateInstance(
        state_type=EXTERNAL_STATE_TYPE,
        _db_url_base="postgresql://r:pw@rds.example.internal:5432/run_x?sslmode=require",
    )

    await gp.create_gateway(
        sandbox_provider=ModalVmSandboxProvider(),
        mcp_servers=[MCPServerConfig(image="mcp-a", environment_name="a")],
        mcp_server_images=[MagicMock(image_name="mcp-a")],
        gateway_mode=GatewayMode.PERFORMANCE,
        state_instance=external,
    )
    # Consumed up front: instance recorded as-is + consume-only provider derived from its tag.
    assert gp._state_instance is external
    assert isinstance(gp._state_provider, ExternalDbStateProvider)
    # The deploy path reads it off self — not passed as a kwarg anymore.
    _, kwargs = gp._deploy_via_vm.call_args
    assert "state_instance" not in kwargs


@pytest.mark.asyncio
async def test_create_gateway_local_sets_provider_but_defers_instance():
    """With no external instance (local Postgres), create_gateway sets the local state PROVIDER
    immediately (so the store-image + compose wiring can go through it) but leaves _state_instance
    unset — the deploy path self-builds the instance (its container's host doesn't exist yet)."""
    from agent_env.env.gateway import GatewayMode
    from agent_env.providers.modal_vm_sandbox import ModalVmSandboxProvider

    gp = GatewayProvider()
    gp._deploy_via_vm = AsyncMock(return_value="VM_RESULT")

    await gp.create_gateway(
        sandbox_provider=ModalVmSandboxProvider(),
        mcp_servers=[MCPServerConfig(image="mcp-a", environment_name="a")],
        mcp_server_images=[MagicMock(image_name="mcp-a")],
        gateway_mode=GatewayMode.PERFORMANCE,
    )
    assert gp._state_instance is None  # instance self-built later in the deploy path
    assert isinstance(gp._state_provider, LocalPostgresStateProvider)


@pytest.mark.asyncio
async def test_close_tears_down_env_state_then_sandboxes():
    """close() releases the acquired env state (drops the per-run DB for remote) and still
    terminates sandboxes even if teardown fails."""
    class _RecordingProvider:
        def __init__(self):
            self.torn = "not-called"

        async def teardown(self, instance):
            self.torn = instance

    gp = GatewayProvider()
    provider = _RecordingProvider()
    gp._state_provider = provider
    gp._state_instance = "INSTANCE"
    await gp.close()
    assert provider.torn == "INSTANCE"
    assert gp._state_provider is None and gp._state_instance is None


@pytest.mark.asyncio
async def test_close_teardown_failure_does_not_block_sandbox_teardown():
    class _BoomProvider:
        async def teardown(self, instance):
            raise RuntimeError("teardown failed")

    gp = GatewayProvider()
    gp._state_provider = _BoomProvider()
    gp._state_instance = "INSTANCE"
    fake = _FakeSandbox("sb1", 8000, "https://a.modal.host")
    gp._container_sandboxes = [fake]
    await gp.close()  # must not raise
    assert fake.terminated


def test_needs_local_postgres_reflects_provider_type():
    """The single local-vs-external check driving image loading / store standup / init: True only
    for a LocalPostgres provider (servicedb images + containers are pulled/rendered iff local)."""
    from agent_env.providers.state import LocalPostgresStateProvider
    from tst.unit.providers.state.fakes import ExternalDbStateProvider

    gp = GatewayProvider()
    assert gp._needs_local_postgres is False  # no provider set yet
    gp._state_provider = LocalPostgresStateProvider()
    assert gp._needs_local_postgres is True
    gp._state_provider = ExternalDbStateProvider()
    assert gp._needs_local_postgres is False


@pytest.mark.asyncio
async def test_deploy_via_vm_prepares_schemas_including_website_browser():
    """The gateway auto-adds the browser MCP when websites are present, then drives stage 2
    (prepare) with the full post-append service list — so a remote store provisions the browser's
    schema. This replaces the earlier band-aid that made Env.deploy compute that list."""
    from agent_env.env.gateway import GatewayMode
    from agent_env.providers.state import DatabaseStateProvider

    gp = GatewayProvider()
    # An external (consume-only) DB provider whose prepare we spy on; not local, so the VM path
    # skips the local store-build/image-load entirely.
    state_provider = MagicMock(spec=DatabaseStateProvider)
    state_provider.prepare = AsyncMock()
    gp._state_provider = state_provider
    gp._state_instance = MagicMock()

    sandbox = AsyncMock()
    sandbox_provider = MagicMock()
    sandbox_provider.create_vm = AsyncMock(return_value=sandbox)

    class _Stop(Exception):
        """Abort right after prepare — the rest of the VM deploy needs a real VM."""

    def _fake_env_get(env_id, *a, **k):
        env = MagicMock()
        env.environment_name = "website-browser" if env_id == "wb-id" else env_id
        return env

    with patch("agent_env.env.env.Env.get", side_effect=_fake_env_get), \
         patch("agent_env.config.get_config", return_value=MagicMock(
             default_gateway_env_id="gw-id", default_website_browser_env_id="wb-id")), \
         patch.object(gp, "create_docker_compose", side_effect=_Stop) as compose:
        with pytest.raises(_Stop):
            await gp._deploy_via_vm(
                sandbox_provider=sandbox_provider,
                mcp_servers=[MCPServerConfig(image="mcp-slack", environment_name="slack")],
                mcp_server_images=[MagicMock(image_name="mcp-slack")],
                gateway_port=18765,
                website_configs=[WebsiteConfig(backend_image="b", frontend_image="f", environment_name="shop")],
                website_images=[MagicMock(), MagicMock()],
                gateway_mode=GatewayMode.PERFORMANCE,
                ttl_seconds=60,
                disk_size_gb=10,
                mcp_server_name="crm",
            )

    state_provider.prepare.assert_awaited_once()
    names = state_provider.prepare.call_args.args[0]
    assert set(names) == {"slack", "shop", "website-browser"}
    assert compose.call_args.kwargs["mcp_server_name"] == "crm"


@pytest.mark.asyncio
async def test_deploy_via_vm_side_loads_sidecar_image():
    """A sidecar's image_artifact is side-loaded onto the VM via load_docker_images alongside
    the gateway/MCP images — so `docker compose up` never pulls it from a registry."""
    from agent_env.env.gateway import GatewayMode
    from agent_env.providers.state import DatabaseStateProvider

    gp = GatewayProvider()
    state_provider = MagicMock(spec=DatabaseStateProvider)
    state_provider.prepare = AsyncMock()
    gp._state_provider = state_provider
    gp._state_instance = MagicMock()

    sandbox = AsyncMock()
    sandbox_provider = MagicMock()
    sandbox_provider.create_vm = AsyncMock(return_value=sandbox)

    controller_art = MagicMock(image_name="ecr/cua/controller:1.0.4")

    class _Stop(Exception):
        pass

    def _fake_env_get(env_id, *a, **k):
        env = MagicMock(); env.environment_name = env_id; return env

    with patch("agent_env.env.env.Env.get", side_effect=_fake_env_get), \
         patch("agent_env.config.get_config", return_value=MagicMock(default_gateway_env_id="gw-id")), \
         patch.object(gp, "create_docker_compose", side_effect=_Stop):
        with pytest.raises(_Stop):
            await gp._deploy_via_vm(
                sandbox_provider=sandbox_provider,
                mcp_servers=[MCPServerConfig(image="mcp-cua", environment_name="cua")],
                mcp_server_images=[MagicMock(image_name="mcp-cua")],
                gateway_port=18765,
                website_configs=None,
                website_images=None,
                gateway_mode=GatewayMode.PERFORMANCE,
                ttl_seconds=60,
                disk_size_gb=10,
                sidecars=[SidecarConfig(
                    environment_name="cua-controller", image_artifact=controller_art,
                    container_port=8000, host_port=18768,
                )],
            )

    sandbox.load_docker_images.assert_awaited_once()
    loaded = sandbox.load_docker_images.call_args.args[0]
    assert controller_art in loaded, "sidecar image_artifact was not side-loaded onto the VM"


# --- name injection: SERVICE_NAME + ENVIRONMENT_NAME --------------------------
#
# The compose emitter injects the env's own name into every environment container so the
# server can identify itself (agentenv-protocol resolves its EnvironmentCard name from it).
# During the service_name -> environment_name rename BOTH vars are emitted with the same
# value; a later pass drops SERVICE_NAME once the fleet reads ENVIRONMENT_NAME.

# Deliberately un-guessable values: the name shares no substring with the image, so a wrong
# substitution (image tag, compose service name, schema) is visible rather than accidentally right.
_ENV_NAME = "quebec-crm"
_ENV_IMAGE = "ecr/mcp-zulu:9.9.9"
_SITE_NAME = "sierra-shop"


def _compose_for(mcp_servers, website_configs=None):
    """Render a compose the way a live deploy does. Explicit ServiceDBConfig so rendering never
    resolves the default-db env from the store (which would hit the network — blocked here)."""
    from agent_env.env.envs.service_db import ServiceDBConfig

    return GatewayProvider().create_docker_compose(
        mcp_servers=list(mcp_servers),
        gateway_image="agent-gateway",
        gateway_port=18765,
        website_configs=list(website_configs) if website_configs else None,
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
        state_instance=LocalPostgresStateProvider.default_instance(),
    )


def _service_block(compose: str, service_name: str) -> list[str]:
    """The lines of one compose service — from `  <name>:` to the next 2-space-indented key — so
    an assertion can never be satisfied by an env line belonging to a neighbouring container."""
    lines = compose.splitlines()
    start = lines.index(f"  {service_name}:")
    for i, line in enumerate(lines[start + 1:], start + 1):
        if line.startswith("  ") and not line.startswith("   "):
            return lines[start:i]
    return lines[start:]


def _env_values(block: list[str], key: str) -> list[str]:
    """Every value bound to `key` by a `      - KEY=VALUE` line, in emission order (compose
    applies last-wins for repeated keys, so the order is behaviour, not cosmetics)."""
    prefix = f"      - {key}="
    return [line[len(prefix):] for line in block if line.startswith(prefix)]


def test_mcp_server_gets_both_service_name_and_environment_name():
    """Dual-inject: one MCP container's own block carries BOTH names, bound once each to
    the same value, so a server reading either var during the migration resolves the same identity."""
    compose = _compose_for([MCPServerConfig(image=_ENV_IMAGE, environment_name=_ENV_NAME)])
    block = _service_block(compose, _ENV_NAME)

    assert _env_values(block, "SERVICE_NAME") == [_ENV_NAME]
    assert _env_values(block, "ENVIRONMENT_NAME") == [_ENV_NAME]
    # Both land inside the fixed block, ahead of DATABASE_URL — not appended after the container's
    # other env (which is where a caller override goes, see the shadowing test).
    assert block.index(f"      - ENVIRONMENT_NAME={_ENV_NAME}") < block.index("    healthcheck:")


def test_website_backend_gets_both_service_name_and_environment_name():
    """Website backends are environments too (their own schema), so they get the same pair. The
    frontend is a static container with no identity and keeps carrying neither."""
    compose = _compose_for(
        [],
        website_configs=[WebsiteConfig(
            backend_image="ecr/shop-be:1", frontend_image="ecr/shop-fe:1", environment_name=_SITE_NAME,
        )],
    )
    backend = _service_block(compose, f"{_SITE_NAME}-website-backend")

    assert _env_values(backend, "SERVICE_NAME") == [_SITE_NAME]
    assert _env_values(backend, "ENVIRONMENT_NAME") == [_SITE_NAME]

    frontend = _service_block(compose, f"{_SITE_NAME}-website-frontend")
    assert _env_values(frontend, "SERVICE_NAME") == []
    assert _env_values(frontend, "ENVIRONMENT_NAME") == []


def test_injected_names_are_the_environment_name_not_the_image_or_container_name():
    """Both vars carry environment_name verbatim. The image tags here differ from the names, and
    the website backend's container name is `<name>-website-backend`, so substituting the wrong
    field (image, compose service name) cannot coincidentally produce the expected value."""
    compose = _compose_for(
        [MCPServerConfig(image=_ENV_IMAGE, environment_name=_ENV_NAME)],
        website_configs=[WebsiteConfig(
            backend_image="ecr/shop-be:1", frontend_image="ecr/shop-fe:1", environment_name=_SITE_NAME,
        )],
    )
    mcp = _service_block(compose, _ENV_NAME)
    backend = _service_block(compose, f"{_SITE_NAME}-website-backend")

    assert _env_values(mcp, "SERVICE_NAME") == _env_values(mcp, "ENVIRONMENT_NAME") == [_ENV_NAME]
    assert _env_values(backend, "SERVICE_NAME") == _env_values(backend, "ENVIRONMENT_NAME") == [_SITE_NAME]
    # ...and neither var picked up a neighbour's value or the image tag.
    assert _ENV_IMAGE not in _env_values(mcp, "SERVICE_NAME") + _env_values(mcp, "ENVIRONMENT_NAME")
    assert _SITE_NAME not in _env_values(mcp, "ENVIRONMENT_NAME")
    assert f"{_SITE_NAME}-website-backend" not in _env_values(backend, "ENVIRONMENT_NAME")


def test_extra_env_vars_shadow_both_names_symmetrically():
    """extra_env_vars are emitted AFTER the fixed block, so compose's last-wins makes a caller
    override effective — and identically for both keys: each override sits the same distance
    behind its fixed line, so overriding both keeps SERVICE_NAME and ENVIRONMENT_NAME in agreement."""
    compose = _compose_for([MCPServerConfig(
        image=_ENV_IMAGE,
        environment_name=_ENV_NAME,
        extra_env_vars={"SERVICE_NAME": "override-name", "ENVIRONMENT_NAME": "override-name"},
    )])
    block = _service_block(compose, _ENV_NAME)

    assert _env_values(block, "SERVICE_NAME") == [_ENV_NAME, "override-name"]
    assert _env_values(block, "ENVIRONMENT_NAME") == [_ENV_NAME, "override-name"]
    svc = [i for i, line in enumerate(block) if line.startswith("      - SERVICE_NAME=")]
    envn = [i for i, line in enumerate(block) if line.startswith("      - ENVIRONMENT_NAME=")]
    assert svc[1] - svc[0] == envn[1] - envn[0]


def test_extra_env_vars_overriding_only_service_name_diverges_the_pair():
    """Pinning what the code actually does: the shadowing is per-key, so a caller that overrides
    only the legacy SERVICE_NAME leaves ENVIRONMENT_NAME on the env's own name and the two
    disagree — the dual-inject does not keep an overridden pair in sync."""
    compose = _compose_for([MCPServerConfig(
        image=_ENV_IMAGE,
        environment_name=_ENV_NAME,
        extra_env_vars={"SERVICE_NAME": "override-name"},
    )])
    block = _service_block(compose, _ENV_NAME)

    assert _env_values(block, "SERVICE_NAME") == [_ENV_NAME, "override-name"]
    assert _env_values(block, "ENVIRONMENT_NAME") == [_ENV_NAME]


def test_service_name_still_emitted_until_fd1730_removes_it():
    """REGRESSION GUARD — the dual-inject is additive-only: SERVICE_NAME must keep being emitted
    for MCP servers AND website backends, because the deployed fleet still reads it. Dropping that
    line belongs to a later pass (after the fleet moves to ENVIRONMENT_NAME); until then it must
    fail here."""
    compose = _compose_for(
        [MCPServerConfig(image=_ENV_IMAGE, environment_name=_ENV_NAME)],
        website_configs=[WebsiteConfig(
            backend_image="ecr/shop-be:1", frontend_image="ecr/shop-fe:1", environment_name=_SITE_NAME,
        )],
    )
    assert f"      - SERVICE_NAME={_ENV_NAME}" in _service_block(compose, _ENV_NAME)
    assert f"      - SERVICE_NAME={_SITE_NAME}" in _service_block(compose, f"{_SITE_NAME}-website-backend")


def test_each_mcp_server_gets_its_own_scoped_name_pair():
    """Multi-server universes (the common case) scope the pair per container: every block binds
    both vars to its own env name and to no other server's — no leakage from the loop variable."""
    names = ["quebec-crm", "romeo-mail", "tango-docs"]
    compose = _compose_for([
        MCPServerConfig(image=f"ecr/mcp-{i}:1", environment_name=n) for i, n in enumerate(names)
    ])
    for name in names:
        block = _service_block(compose, name)
        assert _env_values(block, "SERVICE_NAME") == [name]
        assert _env_values(block, "ENVIRONMENT_NAME") == [name]
        assert not [n for n in names if n != name and n in "".join(block)]


def test_gateway_service_carries_neither_name_var():
    """The gateway is not an environment: it must not gain either var (its SERVICE_DB_URL is the
    only SERVICE_*-prefixed key it owns), so the dual-inject didn't widen who gets an identity."""
    compose = _compose_for([MCPServerConfig(image=_ENV_IMAGE, environment_name=_ENV_NAME)])
    gateway = _service_block(compose, "gateway")

    assert _env_values(gateway, "SERVICE_NAME") == []
    assert _env_values(gateway, "ENVIRONMENT_NAME") == []
    assert any(line.startswith("      - SERVICE_DB_URL=") for line in gateway)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, deploy_method", [(ModalSandboxProvider(), "_deploy_via_containers"), (MagicMock(), "_deploy_via_vm")])
async def test_create_gateway_passes_the_env_name_to_both_deploy_paths(provider, deploy_method):
    """A declared name reaches the gateway on either provider."""
    gp = GatewayProvider()
    with patch.object(gp, deploy_method, new=AsyncMock(return_value=MagicMock())) as deploy:
        await gp.create_gateway(sandbox_provider=provider, mcp_servers=[], mcp_server_images=[], env_id="crm-suite", mcp_server_name="crm")
    assert deploy.call_args.kwargs["mcp_server_name"] == "crm"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, deploy_method", [(ModalSandboxProvider(), "_deploy_via_containers"), (MagicMock(), "_deploy_via_vm")])
async def test_create_gateway_draws_a_random_env_name_when_none_is_declared(provider, deploy_method):
    gp = GatewayProvider()
    with patch.object(gp, deploy_method, new=AsyncMock(return_value=MagicMock())) as deploy:
        await gp.create_gateway(sandbox_provider=provider, mcp_servers=[], mcp_server_images=[], env_id="crm-suite")
    assert re.fullmatch(r"env\d{4}", deploy.call_args.kwargs["mcp_server_name"])


@pytest.mark.parametrize("name, rendered", [(None, "env"), ("crm", "crm"), ("a$b", "a$$b")])
def test_compose_writes_the_env_name_literally(name, rendered):
    """Default `env`, a declared name verbatim, and `$` escaped so compose does not interpolate it."""
    from agent_env.env.envs.service_db import ServiceDBConfig

    compose = GatewayProvider().create_docker_compose(
        mcp_servers=[MCPServerConfig(image="mcp-slack", environment_name="slack")],
        gateway_image="agent-gateway",
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
        state_instance=LocalPostgresStateProvider.default_instance(),
        mcp_server_name=name,
    )
    assert f"      - MCP_SERVER_NAME={rendered}\n" in compose


_PROBE_CARD = {"name": "env1234", "children_environments": [
    {"name": "a", "capabilities": {"tools": [{"name": "a_list"}, {"name": "a_get"}]}},
    {"name": "b", "capabilities": {"tools": [{"name": "b_send"}]}},
]}


def _probed(mcp_url: str = "https://gw.example/mcp") -> DeployedGateway:
    return DeployedGateway(gateway_url="https://gw.example", mcp_url=mcp_url, db_web_url=None, environment_card=_PROBE_CARD)


def _probe_records(caplog) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.getMessage().startswith("env_deploy_tools_probe ")]


@pytest.mark.asyncio
@pytest.mark.parametrize("tools, status, missing", [
    ({"a_list", "a_get", "b_send", "get_time"}, "ok", ""),
    ({"a_list", "a_get"}, "missing", "b"),
    (set(), "missing", "a,b"),
], ids=["covered", "one-child-short", "zero-tools"])
async def test_probe_compares_the_gateway_tools_with_the_stored_card(monkeypatch, caplog, tools, status, missing):
    """Gateway-native tools (get_time) count toward the total but no child has to cover them."""
    monkeypatch.setattr(gateway_provider, "_tool_names", AsyncMock(return_value=tools))
    caplog.set_level(logging.INFO, logger=gateway_provider.logger.name)
    await gateway_provider._probe_tools("crm-suite", _probed())
    [record] = _probe_records(caplog)
    assert record.levelno == (logging.INFO if status == "ok" else logging.WARNING)
    assert f"env_id=crm-suite status={status} tools={len(tools)} " in record.getMessage()
    assert record.getMessage().endswith(f" children=a,b missing={missing}")
    gateway_provider._tool_names.assert_awaited_once_with("https://gw.example/mcp")


async def _never_answers(_mcp_url: str) -> set[str]:
    await asyncio.sleep(60)
    return set()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, deploy_method", [(ModalSandboxProvider(), "_deploy_via_containers"), (MagicMock(), "_deploy_via_vm")])
@pytest.mark.parametrize("tool_names, error_type", [
    (lambda: AsyncMock(side_effect=RuntimeError("boom")), "RuntimeError"),
    (lambda: _never_answers, "TimeoutError"),
], ids=["raises", "times-out"])
async def test_a_failed_probe_never_fails_the_deploy(monkeypatch, caplog, provider, deploy_method, tool_names, error_type):
    """The probe runs once, after the success timing line, and a probe failure only logs."""
    monkeypatch.setattr(gateway_provider, "_TOOLS_PROBE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(gateway_provider, "_tool_names", tool_names())
    caplog.set_level(logging.INFO, logger=gateway_provider.logger.name)
    deployed, gp = _probed(), GatewayProvider()
    with patch.object(gp, deploy_method, new=AsyncMock(return_value=deployed)):
        assert await gp.create_gateway(sandbox_provider=provider, mcp_servers=[], mcp_server_images=[], env_id="crm-suite") is deployed
    messages = [r.getMessage() for r in caplog.records]
    timing = [i for i, m in enumerate(messages) if m.startswith("env_deploy_provider_attempt ") and "status=success" in m]
    probe = [i for i, m in enumerate(messages) if m.startswith("env_deploy_tools_probe ")]
    assert len(timing) == len(probe) == 1 and timing[0] < probe[0]
    assert "status=failed" in messages[probe[0]] and f"error_type={error_type} " in messages[probe[0]]


async def _serve_loopback(app) -> tuple[uvicorn.Server, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(app, log_level="critical", access_log=False)
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return server, f"http://127.0.0.1:{sock.getsockname()[1]}"


@pytest.mark.asyncio
async def test_tool_names_lists_a_live_mcp_server():
    enable_socket()  # loopback only; re-disabled by the autouse fixture's next setup
    app = FastMCP("probe")

    @app.tool()
    def a_list() -> str:
        return "ok"

    server, url = await _serve_loopback(app.streamable_http_app())
    try:
        assert await gateway_provider._tool_names(f"{url}/mcp") == {"a_list"}
    finally:
        await server.shutdown()


@pytest.mark.asyncio
async def test_probe_against_a_server_that_never_answers_logs_a_timeout(monkeypatch, caplog):
    """The real MCP client under the cap: the timeout surfaces as a plain Exception the probe catches."""
    enable_socket()
    release = asyncio.Event()

    async def hold(reader, writer):
        await release.wait()
        writer.close()

    server = await asyncio.start_server(hold, "127.0.0.1", 0)
    monkeypatch.setattr(gateway_provider, "_TOOLS_PROBE_TIMEOUT_S", 0.5)
    caplog.set_level(logging.INFO, logger=gateway_provider.logger.name)
    try:
        await gateway_provider._probe_tools("crm-suite", _probed(f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/mcp"))
    finally:
        release.set()
        server.close()
        await server.wait_closed()
    [record] = _probe_records(caplog)
    assert "status=failed" in record.getMessage() and "error_type=TimeoutError " in record.getMessage()

