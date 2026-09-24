"""GPU support on ModalSandboxProvider.

A ``gpu`` spec must route create_container through the V1 ``modal.Sandbox.create`` factory
(the only one that accepts ``gpu=``), with docker-in-gVisor enabled; the default path must
stay on V2 ``_experimental_create`` with ``i6pn`` unchanged. ``gpu`` + ``i6pn`` is
unsatisfiable (V1 has no i6pn) and must fail before a sandbox is created.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.providers.modal_sandbox import ModalSandboxProvider

# Public registry → the else-branch (modal.Image.from_registry), so no auth/network.
_IMAGE = "public.ecr.aws/x/y:1"


class _Stop(Exception):
    """Raised from the patched create so the test stops once its kwargs are captured."""


def _capture():
    captured: dict = {}

    async def fake_create(*args, **kwargs):
        captured.update(kwargs)
        raise _Stop

    return captured, fake_create


def _patched(target, fake):
    create = MagicMock()
    create.aio = fake
    return patch(target, create)


def _provider(**kwargs) -> ModalSandboxProvider:
    provider = ModalSandboxProvider(**kwargs)
    provider._get_app = AsyncMock(return_value=MagicMock())
    provider._get_client = AsyncMock(return_value=MagicMock())
    return provider


@pytest.mark.asyncio
async def test_gpu_uses_v1_create_with_gpu_and_docker_in_gvisor():
    captured, fake = _capture()
    provider = _provider(gpu="H100")
    with _patched("modal.Sandbox.create", fake), pytest.raises(Exception):
        await provider.create_container(image_name=_IMAGE, port=8000, env={})
    assert captured["gpu"] == "H100"
    assert captured["experimental_options"] == {"enable_docker_in_gvisor": True}
    assert "i6pn" not in captured  # the V1 factory has no i6pn parameter


@pytest.mark.asyncio
async def test_default_stays_on_v2_experimental_create_with_i6pn():
    captured, fake = _capture()
    provider = _provider()  # no gpu
    with _patched("modal.Sandbox._experimental_create", fake), pytest.raises(Exception):
        await provider.create_container(image_name=_IMAGE, port=8000, env={}, i6pn=True)
    assert captured["i6pn"] is True
    assert "gpu" not in captured


@pytest.mark.asyncio
async def test_gpu_with_i6pn_raises_before_creating_a_sandbox():
    provider = _provider(gpu="H100")
    never = MagicMock()
    never.aio = AsyncMock(side_effect=AssertionError("create must not be called"))
    with patch("modal.Sandbox.create", never):
        with pytest.raises(ValueError, match="i6pn"):
            await provider.create_container(image_name=_IMAGE, port=8000, env={}, i6pn=True)
    never.aio.assert_not_called()
