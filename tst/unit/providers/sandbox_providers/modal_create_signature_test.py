"""Every argument core passes to Modal's create calls exists in the installed Modal.

The other create tests patch Modal out, so they still pass when the installed release lacks a
parameter, and the real create then fails with a TypeError. Binding each recorded call to the
real signature catches that for whichever Modal the lock resolves.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import modal
import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.attribution import PIPELINE_STEP_KEY
from agent_env.config import get_config, reset_config
from agent_env.providers.sandbox_providers import modal_sandbox
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from tst.unit.store.fakes import FakeImageStore

_ALLOWLIST = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("only-this.example.com",))
_ATTRIBUTION = {PIPELINE_STEP_KEY: "t_s"}


class _Stop(Exception):
    pass


@pytest.fixture(autouse=True)
def _image_store():
    get_config().set_image_store(FakeImageStore())
    yield
    reset_config()


def _provider(cls, **kwargs):
    provider = cls(app_name="agent-env-test", **kwargs)
    provider._get_app = AsyncMock(return_value=MagicMock())
    provider._get_client = AsyncMock(return_value=MagicMock())
    return provider


_CREATES = {
    "container": (ModalSandboxProvider, {}, "_experimental_create", lambda p: p.create_container(
        image_name="img:latest", port=8000, env={"A": "1"}, network_policy=_ALLOWLIST, attribution=_ATTRIBUTION,
        region="us-east-1")),
    "gpu container": (ModalSandboxProvider, {"gpu": "H100"}, "create", lambda p: p.create_container(
        image_name="img:latest", port=8000, env={"A": "1"}, network_policy=_ALLOWLIST, attribution=_ATTRIBUTION,
        region="us-east-1")),
    "vm": (ModalVmSandboxProvider, {}, "_experimental_create", lambda p: p.create_vm(
        exposed_ports=[8000], network_policy=_ALLOWLIST, attribution=_ATTRIBUTION)),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _CREATES)
async def test_the_installed_modal_accepts_every_create_argument(name):
    cls, kwargs, method, create = _CREATES[name]
    signature = inspect.signature(getattr(modal.Sandbox, method).aio)
    fake = MagicMock()
    fake.aio = AsyncMock(side_effect=_Stop)
    with patch.object(modal.Sandbox, method, fake), pytest.raises(Exception):
        await create(_provider(cls, **kwargs))

    call = fake.aio.call_args
    assert call is not None, f"{name} never reached modal.Sandbox.{method}"
    signature.bind(*call.args, **call.kwargs)


@pytest.mark.asyncio
async def test_the_installed_modal_accepts_every_argument_of_a_build_from_a_context(local_stores, tmp_path, monkeypatch):
    (tmp_path / "Dockerfile").write_text("FROM python:3.12\n")
    image = DockerImageArtifact.put_context("solver-image", description="d", context_path=str(tmp_path),
                                            dockerfile_path="Dockerfile")
    monkeypatch.setattr(modal_sandbox, "_built_image_ids", {})
    from_dockerfile, from_id = inspect.signature(modal.Image.from_dockerfile), inspect.signature(modal.Image.from_id)
    built = MagicMock(object_id="im-1")
    built.build.aio = AsyncMock()
    fake_from_dockerfile, fake_from_id = MagicMock(return_value=built), MagicMock()
    provider = _provider(ModalSandboxProvider)

    with patch.object(modal.Image, "from_dockerfile", fake_from_dockerfile), patch.object(modal.Image, "from_id", fake_from_id):
        await provider.prepare_image(image)

    from_dockerfile.bind(*fake_from_dockerfile.call_args.args, **fake_from_dockerfile.call_args.kwargs)
    inspect.signature(modal.Image.build.aio).bind(built, *built.build.aio.call_args.args, **built.build.aio.call_args.kwargs)
    from_id.bind(modal.Image, *fake_from_id.call_args.args, **fake_from_id.call_args.kwargs)  # its cls, then the call
