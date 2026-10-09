"""An agent whose image is only a build context deploys only where a VM builds it: a provider that runs an agent in a
container by its image's name refuses it before creating anything."""

import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider


class _Created(Exception):
    """Raised by a fake provider once asked for a sandbox, which is as far as a test needs to go."""


class _LocalProvider(LocalSandboxProvider):
    async def create_sandbox(self, **_):
        raise AssertionError("an image that is only a build context was run by name")


class _VmProvider(SandboxProvider):
    async def create_sandbox(self, **_):
        raise _Created


IMAGE = DockerImageArtifact(id="solver-image", version=2, description="d", image_name="local/solver-0123456789ab:v2",
                            build_context_object_url="s3://bucket/ctx.tar.gz")


@pytest.fixture
def litellm(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.example/v1")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", [_LocalProvider(), ChainedSandboxProvider([_VmProvider(), _LocalProvider()])],
                         ids=["local", "chain-with-local"])
async def test_a_provider_that_runs_the_agent_by_name_refuses_it(monkeypatch, local_stores, litellm, provider):
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)

    with pytest.raises(ValueError, match="Can't deploy agent 'solver' with _LocalProvider, which runs an agent's image by "
                                         "name: 'solver-image' v2 is only a build context, which only a VM sandbox builds"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox_type="any")


@pytest.mark.asyncio
async def test_a_vm_provider_is_asked_for_a_sandbox(monkeypatch, local_stores, litellm):
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: _VmProvider())

    with pytest.raises(_Created):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox_type="any")
