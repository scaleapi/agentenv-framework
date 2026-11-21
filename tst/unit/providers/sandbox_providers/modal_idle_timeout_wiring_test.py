"""The configured ``idle_timeout`` must reach Modal's create call for a container sandbox.

A sandbox created without one is ended by Scale's Modal budget enforcement, which ends sandboxes that have no idle
timeout, so losing the pass-through would kill agent containers mid-run.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider


class _Stop(Exception):
    """Raised from the patched create so the test stops once the kwargs are captured."""


async def _create_kwargs(provider: ModalSandboxProvider) -> dict:
    captured: dict = {}

    async def fake_create(*args, **kwargs):
        captured.update(kwargs)
        raise _Stop

    create = MagicMock()
    create.aio = fake_create
    provider._get_app = AsyncMock(return_value=MagicMock())
    provider._get_client = AsyncMock(return_value=MagicMock())
    with patch("modal.Sandbox.create", create), pytest.raises(Exception):
        await provider.create_container(image_name="public.ecr.aws/x/y:1", port=8000, env={})
    return captured


@pytest.mark.asyncio
async def test_modal_container_sends_the_configured_idle_timeout():
    assert (await _create_kwargs(ModalSandboxProvider(idle_timeout=1800)))["idle_timeout"] == 1800


@pytest.mark.asyncio
async def test_modal_container_sends_no_idle_timeout_when_unset():
    """Unset keeps Modal's default: the sandbox lasts until its timeout."""
    assert (await _create_kwargs(ModalSandboxProvider()))["idle_timeout"] is None
