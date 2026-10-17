"""Every successful create must stamp the effective policy on the sandbox it returns, and the idle timeout it applied.

`Sandbox.network_policy = None` means "the backend cannot tell us", which is only a useful
signal if the *only* way to get it is a reconnect via get_sandbox. A create path that
forgets to stamp makes None ambiguous and lands a null on the deploy record.

The factory table is keyed off _BUILTIN_SANDBOX_PROVIDERS, so a new backend fails
test_every_builtin_backend_is_covered until someone drives it here.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.providers.sandbox_providers.e2b.provider import E2BSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.sail_vm import _sdk as sail_sdk
from agent_env.providers.sandbox_providers.sail_vm.provider import SailVmSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import Sandbox
from agent_env.providers.sandbox_providers.sandbox_provider import _BUILTIN_SANDBOX_PROVIDERS
from agent_env.providers.sandbox_providers.tenki.provider import TenkiSandboxProvider
from agent_env.providers.sandbox_providers.tensorlake.provider import TensorlakeSandboxProvider
from agent_env.providers.sandbox_providers.vercel.provider import VercelSandboxProvider


def _modal_provider(cls):
    provider = cls()
    provider._get_app = AsyncMock(return_value=MagicMock())
    provider._get_client = AsyncMock(return_value=MagicMock())
    return provider


def _modal_sb():
    sb = MagicMock()
    sb.object_id = "sb-test"
    sb.tunnels.aio = AsyncMock(return_value={})
    sb.wait_until_ready.aio = AsyncMock()
    return sb


def _patched_create(sb):
    create = MagicMock()
    create.aio = AsyncMock(return_value=sb)
    return patch("modal.Sandbox._experimental_create", create)


async def _modal_vm(**create) -> Sandbox:
    with _patched_create(_modal_sb()):
        return await _modal_provider(ModalVmSandboxProvider).create_vm(
            exposed_ports=[], setup_for_gateway=False, **create
        )


async def _modal(**create) -> Sandbox:
    with _patched_create(_modal_sb()):
        return await _modal_provider(ModalSandboxProvider).create_container(
            image_name="public.ecr.aws/x/y:1", port=8000, env={}, expose_externally=False, **create
        )


async def _local(**create) -> Sandbox:
    return await LocalSandboxProvider().create_vm(**create)


async def _e2b(**create) -> Sandbox:
    resolver = MagicMock()
    resolver.resolve = AsyncMock(return_value="agent-env-docker-base-v1-1c-8192m")
    sdk = MagicMock()
    sdk.create = AsyncMock(return_value=MagicMock(sandbox_id="e2b-test"))
    provider = E2BSandboxProvider(
        api_key="test-key",
        base_template="agent-env-docker-base-v1",
        template_resolver=resolver,
        sandbox_cls=sdk,
    )
    return await provider.create_vm(exposed_ports=[], setup_for_gateway=False, **create)


async def _sail(**create) -> Sandbox:
    sailbox = MagicMock(sailbox_id="sb_test", status="running")
    sailbox.listeners.aio = AsyncMock(return_value=[])
    sdk = MagicMock()
    sdk.Sailbox.create.aio = AsyncMock(return_value=sailbox)
    with patch.object(sail_sdk, "connect", return_value=(sdk, SimpleNamespace(id="app_test"))):
        provider = SailVmSandboxProvider(api_key="test-key", sdk=sdk)
        return await provider.create_vm(exposed_ports=[], setup_for_gateway=False, **create)


async def _vercel(**create) -> Sandbox:
    raw = MagicMock(name="vercel-test")
    raw.name = "vercel-test"
    raw.routes = ()
    raw.network_policy = SimpleNamespace(mode="allow-all", allow={})
    client = MagicMock()
    client.create_sandbox = AsyncMock(return_value=raw)
    provider = VercelSandboxProvider(client_factory=lambda: client)
    sandbox = await provider.create_vm(exposed_ports=[], setup_for_gateway=False, **create)
    assert sandbox._client is client
    return sandbox


async def _tensorlake(**create) -> Sandbox:
    sdk = MagicMock()
    pending = MagicMock(sandbox_id="tl-test")
    pending.ready = AsyncMock(return_value=MagicMock(sandbox_id="tl-test"))
    sdk.create = AsyncMock(return_value=pending)
    provider = TensorlakeSandboxProvider(
        api_key="test-key", sandbox_cls=sdk, client_cls=MagicMock(),
    )
    return await provider.create_vm(exposed_ports=[], setup_for_gateway=False, **create)


async def _tenki(**create) -> Sandbox:
    session = MagicMock(id="tenki-test")
    session.wait_ready = AsyncMock()
    session.info.timeout_at = None
    client = MagicMock()
    client.create = AsyncMock(return_value=session)
    provider = TenkiSandboxProvider(api_key="test-key", client_cls=MagicMock(return_value=client))
    return await provider.create_vm(exposed_ports=[], setup_for_gateway=False, **create)
async def _smol_vm(**create) -> Sandbox:
    pytest.importorskip("smol")
    from agent_env.providers.sandbox_providers.smol_vm import provider as smol_module
    from agent_env.providers.sandbox_providers.smol_vm.provider import SmolVmSandboxProvider
    machine = MagicMock(id="agentenv-test")
    machine.exec.return_value = SimpleNamespace(exit_code=0, stdout="x86_64\n", stderr="")
    with (
        patch.object(smol_module, "Machine", SimpleNamespace(create=lambda config, conn: machine)),
        patch.object(smol_module, "_host_port", return_value=44112),
        patch.object(SmolVmSandboxProvider, "_setup_docker", new_callable=AsyncMock),
        patch.object(SmolVmSandboxProvider, "_configure_host_access", new_callable=AsyncMock),
    ):
        return await SmolVmSandboxProvider().create_vm(exposed_ports=[], **create)


FACTORIES = {
    "modal": _modal,
    "modal_vm": _modal_vm,
    "e2b": _e2b,
    "sail_vm": _sail,
    "tensorlake": _tensorlake,
    "vercel": _vercel,
    "tenki": _tenki,
    "smol_vm": _smol_vm,
    "local": _local,
}


def test_every_builtin_backend_is_covered():
    assert set(_BUILTIN_SANDBOX_PROVIDERS) == set(FACTORIES), (
        "a new sandbox backend must be driven here so the stamping invariant covers it"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(FACTORIES))
async def test_a_successful_create_stamps_the_effective_policy(name):
    sandbox = await FACTORIES[name]()
    assert sandbox.network_policy is not None, (
        f"{name} returned a sandbox with no policy; None must mean 'reconnect', not 'forgot'"
    )
    assert sandbox.network_policy.to_dict()["mode"] == "allow_all"


_ENFORCE_IDLE_TIMEOUT = {"modal", "modal_vm"}


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(FACTORIES))
async def test_a_successful_create_stamps_the_idle_timeout_it_applied(name):
    sandbox = await FACTORIES[name](idle_timeout=900)
    assert sandbox.idle_timeout == (900 if name in _ENFORCE_IDLE_TIMEOUT else None)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(_ENFORCE_IDLE_TIMEOUT))
async def test_a_create_without_an_idle_timeout_stamps_none(name):
    assert (await FACTORIES[name]()).idle_timeout is None
