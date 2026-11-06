"""A gateway deploy with ``capture_traffic``: its compose runs tcpdump beside each website container, in that
container's network namespace and holding the only capture capabilities, rotating within a size cap onto a volume the
gateway alone mounts, read-only; the deploy draws the control token the gateway's traffic routes take and keeps it on
the record; and a deploy that can't capture (no website, containers, a capture that isn't running) is refused. Off,
the compose is the one it always was."""

from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from agentenv_protocol.website import SESSION_SECRET_ENV

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.env.envs.service_db import ServiceDBConfig
from agent_env.env.envs.website import WebsiteEnv
from agent_env.providers.env_providers import env_gateway_provider
from agent_env.providers.env_providers.constants import GATEWAY_SERVICE_NAME
from agent_env.providers.env_providers.env_gateway_provider import (
    TRAFFIC_CAPTURE_FILE_MB,
    TRAFFIC_CAPTURE_FILES,
    DeployedGateway,
    EnvironmentGatewayProvider,
    MCPServerConfig,
    WebsiteConfig,
    _redact_compose_secrets,
    traffic_capture_services,
)
from agent_env.providers.env_state import LocalPostgresStateProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import Accepts
from tst.util.sandbox_providers import VMS_AND_CONTAINERS

_WEBSITES = [WebsiteConfig(backend_image="notes-be", frontend_image=None, environment_name="notes"),
             WebsiteConfig(backend_image="mail-be", frontend_image="mail-fe", environment_name="mail")]


def _compose(**kw) -> str:
    return EnvironmentGatewayProvider().create_docker_compose(
        mcp_servers=[MCPServerConfig(image="mcp-slack", environment_name="slack")], gateway_image="agent-gateway",
        website_configs=_WEBSITES, state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
        state_instance=LocalPostgresStateProvider.default_instance(), **kw)


def _without_session_secrets(compose: str) -> str:
    return re.sub(rf"(- {SESSION_SECRET_ENV}=)\S+", r"\1", compose)


def _vm_provider(**attrs) -> MagicMock:
    provider = MagicMock(ON_THIS_MACHINE=False, SUPPORTED_SANDBOX_MODES=VMS_AND_CONTAINERS,
                         SANDBOX_ACCEPTS=Accepts.LOADABLE, **attrs)
    provider.links = (provider,)
    return provider


def test_each_website_container_gets_a_capture_in_its_own_network_namespace():
    services = yaml.safe_load(_compose(capture_traffic=True, control_token="ctl"))["services"]

    captures = {name: service for name, service in services.items() if name.endswith("-traffic")}
    assert traffic_capture_services(_WEBSITES) == {
        "notes-website-backend-traffic": "notes-website-backend", "mail-website-backend-traffic": "mail-website-backend",
        "mail-website-frontend-traffic": "mail-website-frontend"}
    assert set(captures) == set(traffic_capture_services(_WEBSITES))
    for name, container in traffic_capture_services(_WEBSITES).items():
        capture = captures[name]
        assert capture["image"] == "agent-gateway"
        assert capture["network_mode"] == f"service:{container}" and "networks" not in capture and "ports" not in capture
        assert capture["depends_on"] == [container]
        assert capture["cap_add"] == ["NET_RAW", "NET_ADMIN"]
        assert capture["entrypoint"] == ["tcpdump"]
        assert capture["command"] == ["-i", "any", "-n", "-s", "0", "-U", "-Z", "root", "-C", str(TRAFFIC_CAPTURE_FILE_MB),
                                      "-W", str(TRAFFIC_CAPTURE_FILES), "-w", f"/captures/{container}.pcap"]
        assert capture["volumes"] == ["traffic-captures:/captures"]
    assert TRAFFIC_CAPTURE_FILE_MB * TRAFFIC_CAPTURE_FILES <= 500  # a long run can't fill the VM's disk


def test_only_the_captures_hold_capture_capabilities_and_only_the_gateway_reads_their_volume():
    compose = yaml.safe_load(_compose(capture_traffic=True, control_token="ctl"))
    services = compose["services"]

    assert [name for name, s in services.items() if s.get("cap_add")] == sorted(
        traffic_capture_services(_WEBSITES), key=list(services).index)
    mounting = {name: s["volumes"] for name, s in services.items()
                if any(str(v).startswith("traffic-captures:") for v in s.get("volumes") or [])}
    assert mounting.pop(GATEWAY_SERVICE_NAME) == ["traffic-captures:/var/lib/agentenv/traffic:ro"]
    assert set(mounting) == set(traffic_capture_services(_WEBSITES))
    assert compose["volumes"] == {"traffic-captures": {}}
    gateway = services[GATEWAY_SERVICE_NAME]
    assert {"AGENTENV_CONTROL_TOKEN=ctl", "TRAFFIC_CAPTURE_DIR=/var/lib/agentenv/traffic"} <= set(gateway["environment"])
    assert {name: gateway["depends_on"][name] for name in traffic_capture_services(_WEBSITES)} == dict.fromkeys(
        traffic_capture_services(_WEBSITES), {"condition": "service_started"})


def test_the_logged_compose_masks_the_control_token():
    compose = _compose(capture_traffic=True, control_token="ctl-secret-value")
    assert "ctl-secret-value" in compose
    assert "ctl-secret-value" not in _redact_compose_secrets(compose)
    assert "      - AGENTENV_CONTROL_TOKEN=***" in _redact_compose_secrets(compose)


def test_without_capture_traffic_the_compose_is_unchanged():
    plain = _compose()
    # Each render draws its websites' session secrets afresh, so they're left out of the comparison.
    assert _without_session_secrets(plain) == _without_session_secrets(_compose(capture_traffic=False, control_token=None))
    assert "traffic" not in plain and "tcpdump" not in plain and "cap_add" not in plain and "AGENTENV_CONTROL_TOKEN" not in plain
    assert yaml.safe_load(_compose(capture_traffic=True, control_token="t"))["services"].keys() - yaml.safe_load(
        plain)["services"].keys() == set(traffic_capture_services(_WEBSITES))


def test_a_name_a_capture_would_take_is_refused():
    with pytest.raises(ValueError, match="'notes-website-backend-traffic'"):
        EnvironmentGatewayProvider().create_docker_compose(
            mcp_servers=[MCPServerConfig(image="mcp", environment_name="notes-website-backend-traffic")],
            website_configs=_WEBSITES, state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
            state_instance=LocalPostgresStateProvider.default_instance(), capture_traffic=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("capture_traffic", [True, False], ids=["capturing", "plain"])
async def test_create_gateway_draws_a_control_token_only_for_a_capturing_deploy(capture_traffic):
    gp = EnvironmentGatewayProvider()
    with patch.object(gp, "_deploy_via_vm", AsyncMock(return_value=DeployedGateway(
            gateway_url="https://gw", mcp_url="https://gw/mcp", db_web_url=None))) as vm, \
         patch.object(env_gateway_provider, "_tool_names", AsyncMock(return_value=set())):
        await gp.create_gateway(sandbox_provider=_vm_provider(), mcp_servers=[], mcp_server_images=[],
                                website_configs=_WEBSITES, env_id="notes", capture_traffic=capture_traffic)

    kwargs = vm.await_args.kwargs
    if capture_traffic:
        assert kwargs["capture_traffic"] is True and isinstance(kwargs["control_token"], str)
        assert len(kwargs["control_token"]) >= 32
    else:
        assert "capture_traffic" not in kwargs and "control_token" not in kwargs


@pytest.mark.asyncio
async def test_a_capturing_deploy_keeps_its_control_token_on_the_record():
    gp = EnvironmentGatewayProvider()
    env = MagicMock(spec=WebsiteEnv, id="notes-env", version=1, environment_name="notes", health_path="/api/health",
                    settings={}, frontend_docker_image_artifact=None,
                    server_docker_image_artifact=DockerImageArtifact(id="be", description="d", image_name="notes-be"))
    gp._sandbox = MagicMock(sandbox_id="vm-1", type="modal_vm")
    launched = DeployedGateway(gateway_url="https://gw", mcp_url="https://gw/mcp", db_web_url=None,
                               environment_card={"name": "env1"}, control_token="drawn")
    with patch("agent_env.providers.env_state.acquire_state_for_deploy", AsyncMock(return_value=MagicMock(
            state_type="remote", instance_id="st"))), \
         patch.object(gp, "_launch", AsyncMock(return_value=launched)) as launch:
        record = await gp.deploy(env, _vm_provider(), capture_traffic=True)

    assert launch.await_args.kwargs["capture_traffic"] is True
    assert record.control_token == "drawn" and record.control_headers == {"Authorization": "Bearer drawn"}


@pytest.mark.asyncio
async def test_a_deploy_that_serves_no_website_is_refused_capture_before_anything_starts():
    gp = EnvironmentGatewayProvider()
    env = MagicMock(id="slack-env", environment_name="slack", docker_image_artifact=MagicMock(image_name="mcp-slack"))
    acquire = AsyncMock()
    with patch("agent_env.providers.env_state.acquire_state_for_deploy", acquire), \
         pytest.raises(ValueError, match="serves no website"):
        await gp.deploy(env, _vm_provider(), capture_traffic=True)
    acquire.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_container_mode_deploy_is_refused_capture():
    gp = EnvironmentGatewayProvider()
    with pytest.raises(ValueError, match="Can't capture traffic on ModalSandboxProvider"):
        await gp._deploy_gateway(ModalSandboxProvider(), state_instance=None, mcp_servers=[], mcp_server_images=[],
                                 website_configs=_WEBSITES, capture_traffic=True, control_token="t")


def _ps(*rows: dict) -> str:
    return "\n".join(json.dumps(row) for row in rows)


@pytest.mark.asyncio
async def test_a_capture_that_isnt_running_refuses_the_deploy_with_its_output():
    gp, sandbox = EnvironmentGatewayProvider(), MagicMock()
    rows = _ps({"Service": "notes-website-backend-traffic", "State": "running", "ID": "a"},
               {"Service": "mail-website-backend-traffic", "State": "restarting", "ExitCode": 127, "ID": "b"})

    async def exec_with_output(*argv):
        if "ps" in argv:
            return 0, rows, ""
        return 0, 'exec: "tcpdump": executable file not found in $PATH', ""

    sandbox.exec_with_output = exec_with_output
    with pytest.raises(RuntimeError) as raised:
        await gp._refuse_stopped_captures(sandbox, ["notes-website-backend-traffic", "mail-website-backend-traffic",
                                                    "mail-website-frontend-traffic"])
    message = str(raised.value)
    assert "mail-website-backend-traffic keeps restarting (last exit 127)" in message and "tcpdump" in message
    assert "mail-website-frontend-traffic has no container" in message
    assert "notes-website-backend-traffic" not in message


@pytest.mark.asyncio
async def test_running_captures_let_the_deploy_through():
    gp, sandbox = EnvironmentGatewayProvider(), MagicMock()
    sandbox.exec_with_output = AsyncMock(return_value=(0, _ps(
        {"Service": "notes-website-backend-traffic", "State": "running"}), ""))
    await gp._refuse_stopped_captures(sandbox, ["notes-website-backend-traffic"])
