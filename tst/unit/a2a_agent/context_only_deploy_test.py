"""An agent whose image is only a build context deploys only where it's built, in a VM or by Modal: a provider that runs an
agent in a container by its image's name refuses it before creating anything, and Modal is told of it first. An image a
provider can't get otherwise, such as one a sandbox elsewhere would pull from this machine's registry, is refused before
anything is created too."""

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
                            build_context_object_url="s3://bucket/ctx.tar.gz", dockerfile_path="Dockerfile")


@pytest.fixture
def litellm(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "k")
    monkeypatch.setenv("LITELLM_BASE_URL", "http://litellm.example/v1")


BY_NAME = ("with _LocalProvider, which runs an agent's image by name: 'solver-image' v2 is only a build context, which has "
           "to be built, not pulled")


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, refusal", [
    (_LocalProvider(), BY_NAME),
    (ChainedSandboxProvider([_LocalProvider(), _LocalProvider()]), f"{BY_NAME}; nor {BY_NAME}"),
], ids=["local", "chain-of-locals"])
async def test_a_provider_that_runs_the_agent_by_name_refuses_it(monkeypatch, local_stores, litellm, provider, refusal):
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)

    with pytest.raises(ValueError, match=f"Can't deploy agent 'solver' {refusal}"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox_type="any")


@pytest.mark.asyncio
async def test_a_chain_skips_a_provider_that_runs_the_agent_by_name_for_one_that_builds_it(monkeypatch, local_stores,
                                                                                           litellm, caplog):
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda name: ChainedSandboxProvider([_LocalProvider(), _VmProvider()]))

    with pytest.raises(RuntimeError, match=r"All 1 providers failed: _VmProvider: _Created\(\)"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox_type="any")
    assert ("sandbox_provider_skipped provider=_LocalProvider: which runs an agent's image by name: 'solver-image' v2 is "
            "only a build context") in caplog.text


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
async def test_every_image_is_prepared_before_its_sandbox_is_created(monkeypatch, local_stores, litellm):
    """Preparing an image that isn't only a build context is the provider's to skip, as Modal does."""
    modal = _ModalProvider()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: modal)
    pulled = DockerImageArtifact(id="solver-image", version=2, description="d", image_name="registry.example/solver:v2")

    with pytest.raises(_Created):
        await A2AAgent(id="solver", version=1, docker_image_artifact=pulled).deploy(sandbox_type="any")
    assert modal.steps == [("prepare", "registry.example/solver:v2"), ("create", "registry.example/solver:v2")]


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


PULLED_HERE = DockerImageArtifact(id="solver-image", version=2, description="d",
                                  image_name="localhost:5000/local/solver:v2")
HERE = "localhost:5000/local/solver:v2 is in a registry on this machine, which a sandbox elsewhere can't pull from"
NO_REGISTRY = DockerImageArtifact(id="solver-image", version=2, description="d", image_name="solver:v2")


@pytest.mark.asyncio
async def test_an_existing_container_sandbox_elsewhere_takes_an_agent_whose_image_is_in_this_machines_registry(
        local_stores, litellm):
    """It pulls nothing, so where the agent's image is doesn't matter: the deploy goes on to the container's card."""
    with pytest.raises(RuntimeError, match="No tunnel URL for port"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=PULLED_HERE).deploy(sandbox=_ContainerSandbox())


class _LocalPulls(LocalSandboxProvider):
    async def create_sandbox(self, **_):
        raise _Created


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, image, refusal", [
    (_VmProvider(), PULLED_HERE, f"with _VmProvider, whose VMs can't run it: {HERE}"),
    (_ModalProvider(), PULLED_HERE, f"with _ModalProvider, which runs an agent's image by name: {HERE}"),
    (_VmProvider(), NO_REGISTRY, "with _VmProvider, whose VMs can't run it: 'solver-image' v2 has no tar.gz, and its "
                                 "image name 'solver:v2' doesn't name a registry to pull it from"),
], ids=["vm-elsewhere", "modal", "vm-nothing-to-pull"])
async def test_an_image_a_provider_cant_get_is_refused_before_anything_is_created(monkeypatch, local_stores, litellm,
                                                                                 provider, image, refusal):
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)

    with pytest.raises(ValueError, match=f"Can't deploy agent 'solver' {refusal}"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=image).deploy(sandbox_type="any")
    assert getattr(provider, "steps", []) == []


@pytest.mark.asyncio
async def test_the_local_provider_is_asked_for_a_sandbox_for_an_image_in_its_own_registry(monkeypatch, local_stores,
                                                                                         litellm):
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: _LocalPulls())

    with pytest.raises(_Created):
        await A2AAgent(id="solver", version=1, docker_image_artifact=PULLED_HERE).deploy(sandbox_type="any")


class _Container:
    """A container sandbox, recording whether it was terminated."""

    sandbox_id, mode, type, tunnel_urls, vnc_url, network_policy = "sb-agent", "container", "fake", {}, None, None
    ON_THIS_MACHINE = False

    def __init__(self):
        self.terminated = False

    async def terminate(self):
        self.terminated = True


class _ContainerAgentsUndeclared(SandboxProvider):
    """One that creates VMs, so is taken to load an agent's image on one, but gives the agent a container."""

    def __init__(self):
        self.sandbox = _Container()

    async def create_vm(self, **_):
        raise AssertionError("an agent's sandbox comes from create_sandbox")

    async def create_sandbox(self, **_):
        return self.sandbox


class _BuildingContainerAgentsUndeclared(_ContainerAgentsUndeclared):
    """The same, but its containers run a build context it builds first (prepare_image), as Modal's do."""

    CONTAINER_ACCEPTS = Accepts.NAME_OR_CONTEXT


class _ContainerAgentsDeclared(_ContainerAgentsUndeclared):
    """The same, but declaring that its agents' containers run a build context it builds, unlike create_container's."""

    SANDBOX_ACCEPTS = Accepts.NAME_OR_CONTEXT


class _ModalContainer(_ModalProvider):
    """Modal with its build faked, giving a container."""

    async def create_container(self, **kwargs):
        self.steps.append(("create", kwargs["image_name"]))
        return _Container()


PULLED = DockerImageArtifact(id="solver-image", version=2, description="d", image_name="registry.example/solver:v2")


MISDECLARED = (r"_ContainerAgentsUndeclared gave a container, which can't run the image: 'solver-image' v2 is only a "
               r"build context, which has to be built, not pulled; its SANDBOX_ACCEPTS is LOADABLE, the default for a "
               r"provider that implements create_vm, so declare the images its containers run instead: "
               r"SANDBOX_ACCEPTS = Accepts.NAME")


@pytest.mark.asyncio
async def test_a_provider_taken_to_load_the_agents_image_that_gives_a_container_names_the_declaration_it_lacks(
        monkeypatch, local_stores, litellm):
    provider = _ContainerAgentsUndeclared()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)

    with pytest.raises(ValueError, match=f"^{MISDECLARED}$"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox_type="any")
    assert provider.sandbox.terminated


@pytest.mark.asyncio
async def test_a_chain_closes_a_container_that_cant_run_the_agents_image_and_tries_the_next_provider(
        monkeypatch, local_stores, litellm, caplog):
    misdeclared, modal = _ContainerAgentsUndeclared(), _ModalContainer()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda name: ChainedSandboxProvider([misdeclared, modal]))

    with pytest.raises(RuntimeError, match="No tunnel URL for port"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox_type="any")
    assert misdeclared.sandbox.terminated
    assert modal.steps == [("prepare", IMAGE.image_name), ("create", IMAGE.image_name)]
    assert f"{MISDECLARED.replace(chr(92), '')}; trying next provider" in caplog.text


@pytest.mark.asyncio
async def test_a_chain_none_of_whose_sandboxes_runs_the_agents_image_names_each_refusal(monkeypatch, local_stores,
                                                                                       litellm):
    """The provider after one that gives a container it can't run is tried, a VM provider here."""
    misdeclared = _ContainerAgentsUndeclared()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda name: ChainedSandboxProvider([misdeclared, _VmProvider()]))

    with pytest.raises(RuntimeError, match=r"All 2 providers failed: _ContainerAgentsUndeclared: ValueError\(.*declare "
                                           r"the images its containers run instead: "
                                           r"SANDBOX_ACCEPTS = Accepts.NAME.*\); _VmProvider: _Created\(\)$"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox_type="any")
    assert misdeclared.sandbox.terminated


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_class, image", [
    (_ContainerAgentsUndeclared, PULLED),
    (_BuildingContainerAgentsUndeclared, IMAGE),
    (_ContainerAgentsDeclared, IMAGE),
], ids=["pulled-by-name", "built-first", "declared"])
async def test_a_container_that_can_run_the_agents_image_goes_on_to_serve_it(monkeypatch, local_stores, litellm,
                                                                            provider_class, image):
    """Its provider should say so, unless it declared it, but the deploy it gave works, so it isn't refused."""
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider_class())

    with pytest.raises(RuntimeError, match="No tunnel URL for port"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=image).deploy(sandbox_type="any")


@pytest.mark.asyncio
async def test_a_chain_that_falls_back_from_a_vm_provider_to_modal_serves_the_agent_from_modals_container(
        monkeypatch, local_stores, litellm):
    modal = _ModalContainer()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda name: ChainedSandboxProvider([_VmProvider(), modal]))

    with pytest.raises(RuntimeError, match="No tunnel URL for port"):
        await A2AAgent(id="solver", version=1, docker_image_artifact=IMAGE).deploy(sandbox_type="any")
    assert modal.steps == [("prepare", IMAGE.image_name), ("create", IMAGE.image_name)]
