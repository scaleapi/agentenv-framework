"""``idle_timeout`` reaches Modal on every create path, and an unset one sends nothing, so Modal keeps no idle limit; a
provider written before the parameter is called as before."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

_IMAGE = "public.ecr.aws/x/y:1"


class _Stop(Exception):
    pass


def _provider(cls, **kwargs):
    provider = cls(**kwargs)
    provider._get_app = AsyncMock(return_value=MagicMock())
    provider._get_client = AsyncMock(return_value=MagicMock())
    return provider


_CREATES = {
    "container": (ModalSandboxProvider, {}, "modal.Sandbox._experimental_create",
                  lambda p, **kw: p.create_container(image_name=_IMAGE, port=8000, env={}, **kw)),
    "gpu container": (ModalSandboxProvider, {"gpu": "H100"}, "modal.Sandbox.create",
                      lambda p, **kw: p.create_container(image_name=_IMAGE, port=8000, env={}, **kw)),
    "agent sandbox": (ModalSandboxProvider, {}, "modal.Sandbox._experimental_create",
                      lambda p, **kw: p.create_sandbox(image_name=_IMAGE, port=8000, env={}, **kw)),
    "vm": (ModalVmSandboxProvider, {}, "modal.Sandbox._experimental_create",
           lambda p, **kw: p.create_vm(exposed_ports=[8000], **kw)),
    "vm agent sandbox": (ModalVmSandboxProvider, {}, "modal.Sandbox._experimental_create",
                         lambda p, **kw: p.create_sandbox(image_name=_IMAGE, port=8000, env={}, **kw)),
}


async def _create_kwargs(name: str, **kwargs) -> dict:
    cls, provider_kwargs, target, create = _CREATES[name]
    captured: dict = {}

    async def fake(*args, **create_kwargs):
        captured.update(create_kwargs)
        raise _Stop

    factory = MagicMock()
    factory.aio = fake
    with patch(target, factory), pytest.raises(Exception):
        await create(_provider(cls, **provider_kwargs), **kwargs)
    assert captured, f"{name} never reached {target}"
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _CREATES)
async def test_an_idle_timeout_reaches_modal(name):
    assert (await _create_kwargs(name, idle_timeout=900))["idle_timeout"] == 900


@pytest.mark.asyncio
@pytest.mark.parametrize("name", _CREATES)
async def test_no_idle_timeout_sends_none_to_modal(name):
    assert "idle_timeout" not in await _create_kwargs(name)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["container", "vm"])
async def test_the_create_log_names_the_idle_timeout(name, caplog):
    with caplog.at_level("INFO"):
        await _create_kwargs(name, idle_timeout=120)
    assert any("Creating Modal" in r.getMessage() and "idle_timeout=120" in r.getMessage() for r in caplog.records)


class _Vms(SandboxProvider):
    def __init__(self):
        self.created: list[dict] = []

    async def create_vm(self, **kwargs) -> VmSandbox:
        self.created.append(kwargs)
        raise _Stop


class _OlderVms(SandboxProvider):
    """A provider written before idle_timeout existed."""

    async def create_vm(self, *, cpu: float = 1.0, memory: int = 8192, disk_size_gb: float = 10, timeout: int = 7200,
                        exposed_ports=None, attribution=None, network_policy=None) -> VmSandbox:
        raise _Stop


@pytest.mark.asyncio
async def test_the_default_create_container_hands_a_set_idle_timeout_to_its_vm():
    provider = _Vms()
    with pytest.raises(_Stop):
        await provider.create_container(image_name=_IMAGE, port=8000, env={}, idle_timeout=900)
    assert provider.created[0]["idle_timeout"] == 900


@pytest.mark.asyncio
async def test_the_default_create_container_calls_an_older_provider_as_before_without_an_idle_timeout():
    with pytest.raises(_Stop):
        await _OlderVms().create_container(image_name=_IMAGE, port=8000, env={})


class _Containers(SandboxProvider):
    def __init__(self):
        self.created: list[dict] = []

    async def create_sandbox(self, **kwargs):
        self.created.append(kwargs)
        raise _Stop


@pytest.mark.asyncio
async def test_the_default_create_container_hands_a_set_idle_timeout_to_a_container_backends_create_sandbox():
    provider = _Containers()
    with pytest.raises(_Stop):
        await provider.create_container(image_name=_IMAGE, port=8000, env={}, idle_timeout=900)
    assert provider.created[0]["idle_timeout"] == 900
