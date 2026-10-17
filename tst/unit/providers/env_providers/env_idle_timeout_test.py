"""A gateway deploy's ``idle_timeout_seconds`` reaches the VM it creates, and an unset one isn't asked for; a
container-mode gateway deploys without one, with a warning."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.env_providers.env_gateway_provider import (
    EnvironmentGatewayProvider,
    MCPServerConfig,
)
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import Accepts
from tst.util.sandbox_providers import VMS_AND_CONTAINERS

_GATEWAY = DockerImageArtifact(id="gw", description="d", image_name="public.ecr.aws/x/gateway:1",
                               tar_gz_s3_url="s3://bucket/gateway.tar.gz")
_SERVER = DockerImageArtifact(id="srv", description="d", image_name="public.ecr.aws/x/email:1",
                              tar_gz_s3_url="s3://bucket/email.tar.gz")


class _Stop(Exception):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("idle", [900, None], ids=["set", "unset"])
async def test_a_vm_gateway_creates_its_vm_with_the_idle_timeout(idle):
    provider = MagicMock(ON_THIS_MACHINE=False, SUPPORTED_SANDBOX_MODES=VMS_AND_CONTAINERS, SANDBOX_ACCEPTS=Accepts.LOADABLE)
    provider.create_vm = AsyncMock(side_effect=_Stop)
    gp = EnvironmentGatewayProvider()
    gp._state_provider = MagicMock()
    with patch("agent_env.env.env.Env.get", return_value=MagicMock(docker_image_artifact=_GATEWAY)), pytest.raises(_Stop):
        await gp._deploy_via_vm(
            provider, [MCPServerConfig(image=_SERVER.image_name, environment_name="email")], [_SERVER],
            gateway_port=18765, website_configs=None, website_images=None, gateway_mode=MagicMock(value="performance"),
            ttl_seconds=60, disk_size_gb=10, idle_timeout_seconds=idle,
        )
    assert provider.create_vm.await_args.kwargs.get("idle_timeout") == idle


@pytest.mark.asyncio
async def test_a_container_gateway_deploys_without_an_idle_timeout_and_warns(caplog):
    gp = EnvironmentGatewayProvider()
    with patch.object(gp, "_deploy_via_containers", AsyncMock(side_effect=_Stop)) as containers, pytest.raises(_Stop):
        await gp._deploy_gateway(
            ModalSandboxProvider(), state_instance=None,
            mcp_servers=[MCPServerConfig(image=_SERVER.image_name, environment_name="email")], mcp_server_images=[_SERVER],
            idle_timeout_seconds=900,
        )
    assert "idle_timeout_seconds" not in containers.await_args.kwargs
    assert "deploys without an idle timeout" in caplog.text
