"""An agent whose image is only a build context deploys only where it's built, in a VM or by Modal: a provider that runs an
agent in a container by its image's name refuses it before creating anything, and Modal is told of it first."""

import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import Accepts, SandboxProvider


class _Created(Exception):
    """Raised by a fake provider once asked for a sandbox, which is as far as a test needs to go."""


class _LocalProvider(LocalSandboxProvider):
    async def create_sandbox(self, **_):
        raise AssertionError("an image that is only a build context was run by name")


class _VmProvider(SandboxProvider):
    CREATES_VMS = True
    SANDBOX_ACCEPTS = Accepts.LOADABLE

    async def create_sandbox(self, **_):
        raise _Created

    async def create_vm(self, **_):
        raise _Created


class _ModalProvider(ModalSandboxProvider):
    """Modal with its build and its container faked, recording the order they're asked for."""

    def __init__(self):
        super().__init__()
        self.steps, self.attributions = [], []

    async def prepare_image(self, image, *, attribution=None):
        self.steps.append(("prepare", image.image_name))
        self.attributions.append(attribution)

    async def create_container(self, **kwargs):
        self.steps.append(("create", kwargs["image_name"]))
        self.attributions.append(kwargs["attribution"])
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
                                         "name: 'solver-image' v2 is only a build context, which has to be built, not "
                                         "pulled"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox_type="any")


@pytest.mark.asyncio
async def test_a_vm_provider_is_asked_for_a_sandbox(monkeypatch, local_stores, litellm):
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: _VmProvider())

    with pytest.raises(_Created):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox_type="any")


@pytest.mark.asyncio
async def test_modal_is_told_of_the_image_before_it_is_asked_for_a_sandbox(monkeypatch, local_stores, litellm):
    modal = _ModalProvider()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: modal)

    with pytest.raises(_Created):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(
            sandbox_type="any", attribution={"project_id": "0123456789abcdef01234567"})
    assert modal.steps == [("prepare", IMAGE.image_name), ("create", IMAGE.image_name)]
    prepared, created = modal.attributions
    assert prepared == created and prepared["project_id"] == "0123456789abcdef01234567"


@pytest.mark.asyncio
async def test_a_chain_tells_modal_of_the_image_once_it_tries_modal(monkeypatch, local_stores, litellm):
    modal = _ModalProvider()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda name: ChainedSandboxProvider([_VmProvider(), modal]))

    with pytest.raises(RuntimeError, match="All 2 providers failed"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox_type="any")
    assert modal.steps == [("prepare", IMAGE.image_name), ("create", IMAGE.image_name)]


@pytest.mark.asyncio
async def test_an_image_that_is_not_only_a_build_context_is_not_prepared(monkeypatch, local_stores, litellm):
    modal = _ModalProvider()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: modal)
    pulled = DockerImageArtifact(id="solver-image", version=2, description="d", image_name="registry.example/solver:v2")

    with pytest.raises(_Created):
        await A2AAgent(id="solver", version=1, docker_image_artifact=pulled).deploy(sandbox_type="any")
    assert modal.steps == [("create", "registry.example/solver:v2")]


class _ContainerSandbox:
    sandbox_id = "sb-container"
    mode = "container"
    tunnel_urls = {}

    def url_from_sandbox(self, url):
        return url


@pytest.mark.asyncio
async def test_an_existing_container_sandbox_refuses_it_since_it_never_runs_the_agents_image(local_stores, litellm):
    with pytest.raises(ValueError, match="Can't deploy agent 'solver' on the container sandbox 'sb-container', which runs its "
                                         "own image, never the agent's: 'solver-image' v2 is only a build context"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox=_ContainerSandbox())
