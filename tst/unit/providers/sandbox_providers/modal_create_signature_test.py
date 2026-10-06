"""Every argument core passes to Modal's create calls exists in the installed Modal.

The other create tests patch Modal out, so they still pass when the installed release lacks a
parameter, and the real create then fails with a TypeError. Binding each recorded call to the
real signature catches that for whichever Modal the lock resolves.
"""

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import modal
import pytest

from agent_env.attribution import PIPELINE_STEP_KEY
from agent_env.config import get_config, reset_config
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
