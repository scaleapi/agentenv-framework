"""The allowlist must actually reach Modal's create call.

modal_network_policy_test.py covers the lowering as a pure function; that passes even if
neither backend spreads the result into _experimental_create. Modal is the only backend
that enforces anything, so losing that spread would provision open while the deployment
record still claimed the allowlist.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.modal_vm_sandbox import ModalVmSandboxProvider
from agent_env.providers.sandbox import NetworkMode, NetworkPolicy

ALLOWLIST = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("only-this.example.com",))


class _Stop(Exception):
    """Raised from the patched create so the test stops once the kwargs are captured."""


def _capture():
    captured: dict = {}

    async def fake_create(*args, **kwargs):
        captured.update(kwargs)
        raise _Stop
    return captured, fake_create


def _patched_create(fake):
    create = MagicMock()
    create.aio = fake
    return patch("modal.Sandbox._experimental_create", create)


@pytest.mark.asyncio
async def test_modal_vm_sends_the_allowlist_to_modal():
    captured, fake = _capture()
    provider = ModalVmSandboxProvider()
    provider._get_app = AsyncMock(return_value=MagicMock())
    provider._get_client = AsyncMock(return_value=MagicMock())
    with _patched_create(fake), pytest.raises(Exception):
        await provider.create_vm(exposed_ports=[8000], network_policy=ALLOWLIST)
    assert captured["outbound_domain_allowlist"][0] == "only-this.example.com"
    assert "outbound_cidr_allowlist" in captured


@pytest.mark.asyncio
async def test_modal_vm_sends_nothing_when_unrestricted():
    """The default must stay indistinguishable from not passing the parameter: sending
    outbound_domain_allowlist at all flips Modal to ALLOWLIST and drops raw-IP egress."""
    captured, fake = _capture()
    provider = ModalVmSandboxProvider()
    provider._get_app = AsyncMock(return_value=MagicMock())
    provider._get_client = AsyncMock(return_value=MagicMock())
    with _patched_create(fake), pytest.raises(Exception):
        await provider.create_vm(exposed_ports=[8000], network_policy=None)
    assert "outbound_domain_allowlist" not in captured
    assert "outbound_cidr_allowlist" not in captured
    assert "block_network" not in captured


@pytest.mark.asyncio
async def test_modal_container_sends_the_allowlist_to_modal():
    captured, fake = _capture()
    provider = ModalSandboxProvider()
    provider._get_app = AsyncMock(return_value=MagicMock())
    provider._get_client = AsyncMock(return_value=MagicMock())
    with _patched_create(fake), pytest.raises(Exception):
        await provider.create_container(
            image_name="public.ecr.aws/x/y:1", port=8000, env={}, network_policy=ALLOWLIST
        )
    assert captured["outbound_domain_allowlist"][0] == "only-this.example.com"


@pytest.mark.asyncio
async def test_modal_container_sends_nothing_when_unrestricted():
    captured, fake = _capture()
    provider = ModalSandboxProvider()
    provider._get_app = AsyncMock(return_value=MagicMock())
    provider._get_client = AsyncMock(return_value=MagicMock())
    with _patched_create(fake), pytest.raises(Exception):
        await provider.create_container(
            image_name="public.ecr.aws/x/y:1", port=8000, env={}, network_policy=None
        )
    assert "outbound_domain_allowlist" not in captured
