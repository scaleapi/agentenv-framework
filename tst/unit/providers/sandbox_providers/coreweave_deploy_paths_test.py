"""Which CoreWeave sandbox each deploy gets: an agent and a server without a gateway run their image as a container of
their own, and a gateway runs on a Docker host. The Sandbox API is faked; the deploys run for real."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT
from agent_env.providers.env_providers import env_server_provider
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.env_providers.env_server_provider import EnvironmentServerProvider
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.sandbox_providers.coreweave.provider import CoreWeaveSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER, Accepts
from tst.unit.providers.sandbox_providers.coreweave_fakes import fake_sandbox_cls

AGENT_IMAGE = DockerImageArtifact(id="solver-image", version=2, description="d", image_name="registry.example/solver:v2")
PUBLISHED_IMAGE = AGENT_IMAGE.model_copy(update={
    "image_digest": "sha256:" + "a" * 64,
    "build_context_object_url": "s3://bucket/solver-context.tar.gz",
    "dockerfile_path": "Dockerfile",
})
TARBALL_ONLY = DockerImageArtifact(id="solver-image", version=3, description="d", image_name="local/solver:v3",
                                   tar_gz_object_url="s3://bucket/solver.tar.gz")


@pytest.fixture(autouse=True)
def _no_endpoint_parents(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(CoreWeaveSandboxProvider, "EGRESS_HOSTS", ())


@pytest.fixture
def litellm(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.example/v1")


@pytest.fixture
def coreweave(monkeypatch):
    sandbox_cls = fake_sandbox_cls()
    provider = CoreWeaveSandboxProvider(api_key="cw_test", sandbox_cls=sandbox_cls)
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)
    return SimpleNamespace(provider=provider, server=sandbox_cls.server)


@pytest.mark.asyncio
@pytest.mark.parametrize("image", [AGENT_IMAGE, PUBLISHED_IMAGE], ids=["tag", "published-with-context"])
async def test_an_agent_runs_its_image_as_a_container_with_no_docker(monkeypatch, local_stores, litellm, coreweave,
                                                                  image):
    monkeypatch.setattr(A2AAgent, "_wait_for_agent_card", AsyncMock(return_value={"name": "solver"}))

    deployed = await A2AAgent(id="solver", version=1, docker_image_artifact=image).deploy(sandbox_type="coreweave")

    [raw] = coreweave.server.created
    [container] = raw.kwargs["containers"]
    assert (container.image, container.command, container.args) == (image.pinned_name, None, None)
    assert container.environment_variables["A2A_PORT"] == "8000"
    assert container.environment_variables["LITELLM_BASE_URL"] == "http://litellm.example/v1"
    assert [service.port for service in raw.kwargs["services"]] == [8000]
    assert "security_context" not in raw.kwargs
    assert raw.exec_calls == []  # nothing loaded or started under Docker
    assert (deployed.sandbox_type, deployed.sandbox_id, deployed.a2a_url) == ("coreweave", "sb-1", raw.service_endpoints[0].url)
    assert deployed.network_policy == {"mode": "allow_all", "allow_hosts": [], "allow_cidrs": []}


@pytest.mark.asyncio
async def test_an_agent_whose_image_is_only_a_tarball_is_refused_before_any_sandbox(local_stores, litellm, coreweave):
    with pytest.raises(ValueError, match="which runs an agent's image by name"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=TARBALL_ONLY).deploy(sandbox_type="coreweave")
    assert coreweave.server.created == []


@pytest.mark.asyncio
async def test_an_agent_that_wants_docker_on_its_own_sandbox_is_refused_and_its_sandbox_deleted(
        local_stores, litellm, coreweave):
    """Docker for an agent needs a VM: deploy one with a deploy_sandbox step and place the agent on it."""
    with pytest.raises(ValueError, match="enable_docker=True requires a VM sandbox"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=AGENT_IMAGE).deploy(
            sandbox_type="coreweave", enable_docker=True)
    assert coreweave.server.deleted == ["sb-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("digest", [None, "sha256:" + "b" * 64], ids=["tag", "digest"])
async def test_a_server_without_a_gateway_runs_its_image_as_a_container(coreweave, digest):
    env = SimpleNamespace(id="slack-bare", version=2, environment_name="slack",
                          docker_image_artifact=DockerImageArtifact(id="slack-image", description="d",
                                                                    image_name="registry.example/mcp-slack:1",
                                                                    image_digest=digest))
    server = EnvironmentServerProvider()
    server._wait_for_tunnel = AsyncMock(return_value={"name": "slack"})
    with patch.object(env_server_provider, "_tool_names", new=AsyncMock(return_value=["slack_list"])):
        record = await server.deploy(env, coreweave.provider, ttl_seconds=600)

    [raw] = coreweave.server.created
    [container] = raw.kwargs["containers"]
    assert (container.image, container.command) == (env.docker_image_artifact.pinned_name, None)
    assert container.environment_variables["ENVIRONMENT_NAME"] == "slack"
    assert [service.port for service in raw.kwargs["services"]] == [AGENT_ENV_GATEWAY_MCP_PORT]
    assert raw.kwargs["max_lifetime_seconds"] == 600 and raw.exec_calls == []
    assert (record.sandbox_type, record.sandbox_id) == ("coreweave", "sb-1")
    assert server.environment_sandbox("slack").mode == SANDBOX_MODE_CONTAINER


@pytest.mark.asyncio
async def test_preparing_another_version_of_a_tag_replaces_or_removes_its_digest(coreweave):
    for digest in ("sha256:" + "a" * 64, "sha256:" + "b" * 64, None):
        image = AGENT_IMAGE.model_copy(update={"image_digest": digest})
        await coreweave.provider.prepare_image(image)
        await coreweave.provider.create_container(image_name=image.image_name, port=8000, env={})
        assert coreweave.server.created[-1].kwargs["containers"][0].image == image.pinned_name


def test_a_gateway_runs_on_a_docker_host_and_loads_any_image_it_can():
    provider = CoreWeaveSandboxProvider(api_key="cw_test", sandbox_cls=fake_sandbox_cls())
    assert EnvironmentGatewayProvider.in_vm(provider)
    assert EnvironmentGatewayProvider.accepts(provider) is Accepts.LOADABLE
    assert EnvironmentGatewayProvider.sandbox_problem(provider, websites=True) is None
    assert EnvironmentServerProvider.accepts(provider) is Accepts.NAME
    assert A2AAgent.accepts(provider) is Accepts.NAME
    assert A2AAgent.accepts(provider, sandbox_mode="vm") is Accepts.LOADABLE
