"""What each sandbox provider declares it runs, and that nothing outside the providers asks which provider it has: every
choice that depends on a provider reads what it declares (``ON_THIS_MACHINE``, ``runs``, ``links``, ``creates_vms``,
``url_from_sandbox``, ``gateway_container_options``)."""

import ast
from pathlib import Path

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.e2b.provider import E2BSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.sail_vm.provider import SailVmSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import ImageUse, Runs, SandboxProvider, image_problem

SRC = Path(__file__).resolve().parents[4] / "src" / "agent_env"
PROVIDERS = SRC / "providers" / "sandbox_providers"

IN_VM, BY_NAME, BUILDS = Runs.IN_VM, Runs.BY_NAME, Runs.BUILDS


@pytest.mark.parametrize("cls, on_this_machine, creates_vms, agent, server, gateway", [
    (LocalSandboxProvider, True, True, BY_NAME, BY_NAME, IN_VM),
    (ModalSandboxProvider, False, False, BUILDS, BUILDS, BUILDS),
    (ModalVmSandboxProvider, False, True, IN_VM, BY_NAME, IN_VM),
    (E2BSandboxProvider, False, True, IN_VM, BY_NAME, IN_VM),
    (SailVmSandboxProvider, False, True, IN_VM, BY_NAME, IN_VM),
], ids=["local", "modal", "modal_vm", "e2b", "sail_vm"])
def test_each_provider_declares_where_it_runs_and_how_it_runs_each_image(cls, on_this_machine, creates_vms, agent, server,
                                                                         gateway):
    assert (cls.ON_THIS_MACHINE, cls.creates_vms()) == (on_this_machine, creates_vms)
    assert [cls.runs(use) for use in (ImageUse.AGENT, ImageUse.SERVER, ImageUse.GATEWAY)] == [agent, server, gateway]


class _Vms(SandboxProvider):
    async def create_sandbox(self, **_):
        raise NotImplementedError

    async def create_vm(self, **_):
        raise NotImplementedError


class _Containers(SandboxProvider):
    async def create_sandbox(self, **_):
        raise NotImplementedError


def test_a_provider_that_declares_nothing_runs_images_as_the_base_class_does():
    """In a VM it creates, which loads the image, but a lone server from its image's name; with no VM, by name."""
    assert [_Vms.runs(use) for use in ImageUse] == [IN_VM, BY_NAME, IN_VM]
    assert [_Containers.runs(use) for use in ImageUse] == [BY_NAME, BY_NAME, BY_NAME]
    assert (_Vms.ON_THIS_MACHINE, _Vms().url_from_sandbox("http://localhost:4000"), _Vms().gateway_container_options()) == (
        False, "http://localhost:4000", {})


def test_a_chain_is_its_providers_in_order_and_one_provider_is_itself():
    first, second = _Vms(), ModalSandboxProvider()

    assert ChainedSandboxProvider([first, second]).links == (first, second)
    assert first.links == (first,)
    assert ChainedSandboxProvider.creates_vms() is False


def test_modals_gateway_containers_reach_one_another_over_its_private_network():
    options = ModalSandboxProvider().gateway_container_options()
    assert options["i6pn"] is True and set(options) == {"i6pn", "region"}


CONTEXT_ONLY = DockerImageArtifact(id="img", version=2, description="d", image_name="local/img-0123456789ab:v2",
                                   build_context_object_url="s3://bucket/ctx.tar.gz")
NO_REGISTRY = DockerImageArtifact(id="img", version=2, description="d", image_name="img:v2")


@pytest.mark.parametrize("image, runs, problem", [
    (CONTEXT_ONLY, IN_VM, None), (CONTEXT_ONLY, BUILDS, None), (CONTEXT_ONLY, BY_NAME, "only a build context"),
    (NO_REGISTRY, IN_VM, "doesn't name a registry"), (NO_REGISTRY, BY_NAME, None), (NO_REGISTRY, BUILDS, None),
])
def test_an_image_is_refused_by_how_its_provider_runs_it(image, runs, problem):
    found = image_problem(image, runs)
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
