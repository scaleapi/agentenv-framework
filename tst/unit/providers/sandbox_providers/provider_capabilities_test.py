"""What each sandbox provider declares it runs, and that nothing outside the providers asks which provider it has: every
choice that depends on a provider reads what it declares (``ON_THIS_MACHINE``, ``SUPPORTED_SANDBOX_MODES``,
``SANDBOX_ACCEPTS``, ``CONTAINER_ACCEPTS``, ``PRIVATE_NETWORK``, ``links`` and ``url_from_sandbox``) and its sandboxes'
``private_host`` and ``platform``."""

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import modal
import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.e2b.provider import E2BSandboxProvider
from agent_env.providers.sandbox_providers.e2b.sandbox import E2BSandbox
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandbox, ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox, ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.sail_vm.provider import SailVmSandboxProvider
from agent_env.providers.sandbox_providers.sail_vm.sandbox import SailVmSandbox
from agent_env.providers.sandbox_providers.sandbox import Sandbox
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SANDBOX_MODE_CONTAINER,
    SANDBOX_MODE_VM,
    Accepts,
    SandboxProvider,
)
from agent_env.providers.sandbox_providers.tensorlake.provider import TensorlakeSandboxProvider
from agent_env.providers.sandbox_providers.tensorlake.sandbox import TensorlakeSandbox

SRC = Path(__file__).resolve().parents[4] / "src" / "agent_env"
PROVIDERS = SRC / "providers" / "sandbox_providers"

LOADABLE, NAME, NAME_OR_CONTEXT = Accepts.LOADABLE, Accepts.NAME, Accepts.NAME_OR_CONTEXT
PROVIDER_CLASSES = [LocalSandboxProvider, ModalSandboxProvider, ModalVmSandboxProvider, E2BSandboxProvider,
                    SailVmSandboxProvider, TensorlakeSandboxProvider, ChainedSandboxProvider]


@pytest.mark.parametrize("cls, on_this_machine, creates_vms, sandbox, container, private_network", [
    (LocalSandboxProvider, True, True, NAME, NAME, False),
    (ModalSandboxProvider, False, False, NAME_OR_CONTEXT, NAME_OR_CONTEXT, True),
    (ModalVmSandboxProvider, False, True, LOADABLE, NAME, False),
    (E2BSandboxProvider, False, True, LOADABLE, NAME, False),
    (SailVmSandboxProvider, False, True, LOADABLE, NAME, False),
    (TensorlakeSandboxProvider, False, True, LOADABLE, NAME, False),
], ids=["local", "modal", "modal_vm", "e2b", "sail_vm", "tensorlake"])
def test_each_provider_declares_where_it_runs_and_the_images_it_runs(cls, on_this_machine, creates_vms, sandbox,
                                                                    container, private_network):
    assert ((cls.ON_THIS_MACHINE, SANDBOX_MODE_VM in cls.SUPPORTED_SANDBOX_MODES, cls.PRIVATE_NETWORK)
            == (on_this_machine, creates_vms, private_network))
    assert (cls.SANDBOX_ACCEPTS, cls.CONTAINER_ACCEPTS) == (sandbox, container)


@pytest.mark.parametrize("cls", PROVIDER_CLASSES, ids=lambda cls: cls.__name__)
def test_a_provider_supports_the_modes_it_implements_and_declares_a_private_network_only_with_containers(cls):
    assert cls.SUPPORTED_SANDBOX_MODES == (
        {SANDBOX_MODE_CONTAINER} | ({SANDBOX_MODE_VM} if cls.create_vm is not SandboxProvider.create_vm else set()))
    assert not cls.PRIVATE_NETWORK or cls.create_container is not SandboxProvider.create_container


class _Vms(SandboxProvider):
    async def create_sandbox(self, **_):
        raise NotImplementedError

    async def create_vm(self, **_):
        raise NotImplementedError


def test_a_provider_that_declares_nothing_has_what_it_implements_and_loads_an_agents_image_on_a_vm():
    assert (_Vms.ON_THIS_MACHINE, _Vms.PRIVATE_NETWORK) == (False, False)
    assert _Vms.SUPPORTED_SANDBOX_MODES == {SANDBOX_MODE_CONTAINER, SANDBOX_MODE_VM}
    assert (_Vms.SANDBOX_ACCEPTS, _Vms.CONTAINER_ACCEPTS) == (LOADABLE, NAME)
    assert _Vms().url_from_sandbox("http://localhost:4000") == "http://localhost:4000"


class _VmsOnly(SandboxProvider):
    async def create_vm(self, **_):
        raise NotImplementedError


class _ContainersOnly(SandboxProvider):
    async def create_sandbox(self, **_):
        raise NotImplementedError


class _Declared(_VmsOnly):
    """One that declares its modes, which are derived regardless: what it implements says what it supports."""

    SUPPORTED_SANDBOX_MODES = frozenset({SANDBOX_MODE_CONTAINER})


def test_a_provider_supports_only_the_modes_it_implements():
    assert (_VmsOnly.SUPPORTED_SANDBOX_MODES, _ContainersOnly.SUPPORTED_SANDBOX_MODES) == (
        {SANDBOX_MODE_VM}, {SANDBOX_MODE_CONTAINER})
    assert SandboxProvider.SUPPORTED_SANDBOX_MODES == set()
    assert _Declared.SUPPORTED_SANDBOX_MODES == {SANDBOX_MODE_VM}


class _BuildingContainers(_ContainersOnly):
    CONTAINER_ACCEPTS = NAME_OR_CONTEXT


class _ContainerAgents(_Vms):
    """One whose agents run as containers by name although it creates VMs, as the local provider's do."""

    SANDBOX_ACCEPTS = NAME


class _ContainerAgentsAgain(_ContainerAgents):
    pass


class _VmsAgain(_Vms):
    CONTAINER_ACCEPTS = NAME_OR_CONTEXT


@pytest.mark.parametrize("cls, accepts", [
    (_Vms, LOADABLE), (_VmsOnly, LOADABLE),  # create_sandbox gives a bare VM the deploy loads the image into
    (_ContainersOnly, NAME), (_BuildingContainers, NAME_OR_CONTEXT),  # create_sandbox is one of its containers
    (_ContainerAgents, NAME), (_ContainerAgentsAgain, NAME),  # declared, and kept by a subclass
    (_VmsAgain, LOADABLE),  # derived again for a subclass that declares nothing
    (LocalSandboxProvider, NAME),
], ids=lambda value: getattr(value, "__name__", getattr(value, "name", value)))
def test_an_agents_sandbox_takes_what_create_sandbox_gives_unless_the_provider_declares_otherwise(cls, accepts):
    assert cls.SANDBOX_ACCEPTS is accepts


def test_a_chain_is_its_providers_in_order_and_one_provider_is_itself():
    first, second = _Vms(), ModalSandboxProvider()

    assert ChainedSandboxProvider([first, second]).links == (first, second)
    assert first.links == (first,)


@pytest.mark.asyncio
async def test_a_container_on_modals_private_network_is_on_i6pn_in_the_configured_region(monkeypatch):
    provider = ModalSandboxProvider()
    provider._get_client = AsyncMock(return_value=MagicMock())
    provider._get_app = AsyncMock(return_value="app")
    monkeypatch.setattr(provider, "_registry_image", AsyncMock(return_value="image"))
    create = MagicMock()
    create.aio = AsyncMock(side_effect=RuntimeError("stop"))
    with patch.object(modal.Sandbox, "_experimental_create", create), pytest.raises(RuntimeError, match="stop"):
        await provider.create_container(image_name="registry.example/srv:1", port=8000, env={}, private_network=True)

    assert create.aio.call_args.kwargs["i6pn"] is True and "region" in create.aio.call_args.kwargs


def test_a_modal_sandbox_on_i6pn_is_reached_at_its_bracketed_address():
    sandbox = ModalSandbox(MagicMock(object_id="sb-1"), {}, i6pn_address="fdaa::1")
    assert (sandbox.private_host, ModalSandbox(MagicMock(object_id="sb-2"), {}).private_host) == ("[fdaa::1]", None)


@pytest.mark.asyncio
async def test_a_provider_whose_containers_are_vms_of_their_own_puts_none_on_a_private_network():
    with pytest.raises(ValueError, match="can't put containers on a private network"):
        await _Vms().create_container(image_name="img:1", port=8000, env={}, private_network=True)


CONTEXT_ONLY = DockerImageArtifact(id="img", version=2, description="d", image_name="local/img-0123456789ab:v2",
                                   build_context_object_url="s3://bucket/ctx.tar.gz")
NO_REGISTRY = DockerImageArtifact(id="img", version=2, description="d", image_name="img:v2")


IN_THIS_MACHINES_REGISTRY = DockerImageArtifact(id="img", version=2, description="d",
                                                image_name="localhost:5000/local/img:v2")
WITH_TAR_GZ = DockerImageArtifact(id="img", version=2, description="d", image_name="localhost:5000/local/img:v2",
                                  tar_gz_object_url="file:///state/img.tar.gz")
WITH_TAR_GZ_AND_CONTEXT = WITH_TAR_GZ.model_copy(update={"build_context_object_url": "file:///state/ctx.tar.gz"})
HERE = "is in a registry on this machine, which a sandbox elsewhere can't pull from"


@pytest.mark.parametrize("image, accepts, problem", [
    (CONTEXT_ONLY, LOADABLE, None), (CONTEXT_ONLY, NAME_OR_CONTEXT, None), (CONTEXT_ONLY, NAME, "only a build context"),
    (NO_REGISTRY, LOADABLE, "doesn't name a registry"), (NO_REGISTRY, NAME, None), (NO_REGISTRY, NAME_OR_CONTEXT, None),
])
@pytest.mark.parametrize("on_this_machine", [True, False])
def test_an_image_is_refused_by_the_form_its_provider_accepts(image, accepts, problem, on_this_machine):
    found = accepts.problem(image, on_this_machine=on_this_machine)
    assert (found is None) if problem is None else (problem in found)


@pytest.mark.parametrize("image, accepts, problem", [
    (IN_THIS_MACHINES_REGISTRY, LOADABLE, HERE), (IN_THIS_MACHINES_REGISTRY, NAME, HERE),
    (IN_THIS_MACHINES_REGISTRY, NAME_OR_CONTEXT, HERE),
    (WITH_TAR_GZ, LOADABLE, None), (WITH_TAR_GZ, NAME, HERE), (WITH_TAR_GZ, NAME_OR_CONTEXT, HERE),
    (WITH_TAR_GZ_AND_CONTEXT, LOADABLE, None), (WITH_TAR_GZ_AND_CONTEXT, NAME, HERE),
    (WITH_TAR_GZ_AND_CONTEXT, NAME_OR_CONTEXT, None),
])
def test_a_sandbox_elsewhere_cant_pull_from_this_machines_registry_and_one_on_it_can(image, accepts, problem):
    """A VM loads an image's tar.gz, which agent-env pushes to it, and pulls one only when it has none; a provider that
    builds images builds one it can't pull from its build context."""
    found = accepts.problem(image, on_this_machine=False)
    assert (found is None) if problem is None else (problem in found)
    assert accepts.problem(image, on_this_machine=True) is None


ARM64_TAR_GZ = DockerImageArtifact(id="img", version=2, description="d", image_name="img:v2", platform="linux/arm64",
                                   tar_gz_object_url="file:///state/img.tar.gz")
ARM64_PULLED = DockerImageArtifact(id="img", version=2, description="d", platform="linux/arm64",
                                   image_name="123456789012.dkr.ecr.us-west-2.amazonaws.com/img:v2")
ARM64_CONTEXT = DockerImageArtifact(id="img", version=2, description="d", image_name="local/img-0123456789ab:v2",
                                    build_context_object_url="s3://bucket/ctx.tar.gz", platform="linux/arm64")
ARM64 = "'img' v2 is a linux/arm64 image, and the sandbox runs linux/amd64; rebuild it for linux/amd64"


@pytest.mark.parametrize("image", [ARM64_TAR_GZ, ARM64_PULLED, ARM64_CONTEXT], ids=["tar.gz", "pulled", "context"])
def test_a_vm_runs_only_images_built_for_its_platform(image):
    """A VM builds a context for the platform it records, so a context is checked too."""
    assert image.load_problem(on_this_machine=False, platform="linux/amd64").startswith(ARM64)
    assert image.load_problem(on_this_machine=False, platform="linux/arm64") is None
    assert image.load_problem(on_this_machine=False, platform=None) is None  # a sandbox that runs any, as a local one does


@pytest.mark.parametrize("accepts", [LOADABLE, NAME, NAME_OR_CONTEXT])
def test_a_provider_checks_no_platform_before_its_sandbox_exists(accepts):
    assert accepts.problem(ARM64_PULLED, on_this_machine=False) is None


@pytest.mark.parametrize("recorded, runs", [
    ("linux/amd64/v2", "linux/amd64"), ("linux/arm64/v8", "linux/arm64"), ("linux/amd64", "linux/x86_64"),
])
def test_a_platform_is_the_same_however_the_image_or_the_sandbox_spells_it(recorded, runs):
    image = DockerImageArtifact(id="img", version=2, description="d", image_name="img:v2", platform=recorded,
                                tar_gz_object_url="s3://bucket/img.tar.gz")
    assert image.load_problem(on_this_machine=False, platform=runs) is None


def test_an_image_that_records_no_platform_isnt_checked():
    image = DockerImageArtifact(id="img", version=2, description="d", image_name="img:v2",
                                tar_gz_object_url="s3://bucket/img.tar.gz")
    assert image.platform is None and image.load_problem(on_this_machine=False, platform="linux/arm64") is None


@pytest.mark.parametrize("sandbox", [Sandbox, ModalSandbox, ModalVmSandbox, E2BSandbox, SailVmSandbox, TensorlakeSandbox])
def test_a_sandbox_runs_linux_amd64_unless_its_provider_sets_another(sandbox):
    assert sandbox.platform == "linux/amd64"


def test_a_local_sandbox_runs_any_platform_unchecked_since_its_docker_emulates_others():
    assert LocalSandbox.platform is None


def _concrete_classes() -> set[str]:
    """The sandbox and sandbox provider classes the providers define, but the base classes everything else may name."""
    names = set()
    for path in PROVIDERS.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and node.name.endswith(("Sandbox", "SandboxProvider")):
                names.add(node.name)
    return names - {"Sandbox", "VmSandbox", "SandboxProvider"}


def test_nothing_outside_the_providers_checks_which_provider_or_sandbox_it_has():
    concrete, found = _concrete_classes(), []
    for path in SRC.rglob("*.py"):
        if PROVIDERS in path.parents:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("isinstance", "issubclass"):
                kinds = node.args[1].elts if isinstance(node.args[1], ast.Tuple) else [node.args[1]]
                named = {kind.id if isinstance(kind, ast.Name) else getattr(kind, "attr", "") for kind in kinds}
                found += [f"{path.relative_to(SRC)}:{node.lineno} {name}" for name in sorted(named & concrete)]
    assert found == [], "ask the provider what it declares instead"
