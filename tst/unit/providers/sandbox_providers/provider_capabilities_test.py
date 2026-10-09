"""What each sandbox provider declares it runs, and that nothing outside the providers asks which provider it has: every
choice that depends on a provider reads what it declares (``ON_THIS_MACHINE``, ``CREATES_VMS``, ``SANDBOX_ACCEPTS``,
``CONTAINER_ACCEPTS``, ``PRIVATE_NETWORK``, ``links`` and ``url_from_sandbox``) and its sandboxes' ``private_host``."""

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import modal
import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.e2b.provider import E2BSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandbox, ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.sail_vm.provider import SailVmSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import Accepts, SandboxProvider

SRC = Path(__file__).resolve().parents[4] / "src" / "agent_env"
PROVIDERS = SRC / "providers" / "sandbox_providers"

LOADABLE, NAME, NAME_OR_CONTEXT = Accepts.LOADABLE, Accepts.NAME, Accepts.NAME_OR_CONTEXT
PROVIDER_CLASSES = [LocalSandboxProvider, ModalSandboxProvider, ModalVmSandboxProvider, E2BSandboxProvider,
                    SailVmSandboxProvider, ChainedSandboxProvider]


@pytest.mark.parametrize("cls, on_this_machine, creates_vms, sandbox, container, private_network", [
    (LocalSandboxProvider, True, True, NAME, NAME, False),
    (ModalSandboxProvider, False, False, NAME_OR_CONTEXT, NAME_OR_CONTEXT, True),
    (ModalVmSandboxProvider, False, True, LOADABLE, NAME, False),
    (E2BSandboxProvider, False, True, LOADABLE, NAME, False),
    (SailVmSandboxProvider, False, True, LOADABLE, NAME, False),
], ids=["local", "modal", "modal_vm", "e2b", "sail_vm"])
def test_each_provider_declares_where_it_runs_and_the_images_it_runs(cls, on_this_machine, creates_vms, sandbox,
                                                                    container, private_network):
    assert (cls.ON_THIS_MACHINE, cls.CREATES_VMS, cls.PRIVATE_NETWORK) == (on_this_machine, creates_vms, private_network)
    assert (cls.SANDBOX_ACCEPTS, cls.CONTAINER_ACCEPTS) == (sandbox, container)


@pytest.mark.parametrize("cls", PROVIDER_CLASSES, ids=lambda cls: cls.__name__)
def test_a_provider_declares_vms_and_a_private_network_only_where_it_implements_them(cls):
    assert cls.CREATES_VMS == (cls.create_vm is not SandboxProvider.create_vm)
    assert not cls.PRIVATE_NETWORK or cls.create_container is not SandboxProvider.create_container


class _Vms(SandboxProvider):
    async def create_sandbox(self, **_):
        raise NotImplementedError

    async def create_vm(self, **_):
        raise NotImplementedError


def test_a_provider_that_declares_nothing_runs_images_by_name_and_creates_no_vm():
    """Implementing create_vm declares nothing: a provider says it creates VMs, as it says everything else."""
    assert (_Vms.ON_THIS_MACHINE, _Vms.CREATES_VMS, _Vms.PRIVATE_NETWORK) == (False, False, False)
    assert (_Vms.SANDBOX_ACCEPTS, _Vms.CONTAINER_ACCEPTS) == (NAME, NAME)
    assert _Vms().url_from_sandbox("http://localhost:4000") == "http://localhost:4000"


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


@pytest.mark.asyncio
async def test_a_gpu_container_is_refused_the_private_network_before_any_modal_call():
    provider = ModalSandboxProvider(gpu="H100")
    provider._get_app = AsyncMock()
    with pytest.raises(ValueError, match="unavailable on GPU sandboxes"):
        await provider.create_container(image_name="registry.example/srv:1", port=8000, env={}, private_network=True)

    provider._get_app.assert_not_called()


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


@pytest.mark.parametrize("image, accepts, problem", [
    (CONTEXT_ONLY, LOADABLE, None), (CONTEXT_ONLY, NAME_OR_CONTEXT, None), (CONTEXT_ONLY, NAME, "only a build context"),
    (NO_REGISTRY, LOADABLE, "doesn't name a registry"), (NO_REGISTRY, NAME, None), (NO_REGISTRY, NAME_OR_CONTEXT, None),
])
def test_an_image_is_refused_by_the_form_its_provider_accepts(image, accepts, problem):
    found = accepts.problem(image)
    assert (found is None) if problem is None else (problem in found)


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
