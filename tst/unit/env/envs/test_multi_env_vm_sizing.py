"""A MultiEnv does not size its own box; cpu and memory both defer to the provider.

The providers already floor both per backend (0.125 vCPU on Modal, where cpu is a
burstable reservation, 0.5 on the VM backends; 8GB memory on either), and a number invented
here could only override a better-informed default. In container mode it was actively
wrong: the derived size landed on the gateway, a router, as if it hosted every service.

Nothing depended on the derived cpu except that gateway,
and the two callers that leaned on the derived memory were getting it from `cpu x 4096` --
an 11-service env on 2GB. An explicit value from the caller still wins on both axes.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider


def _env(n_mcp: int = 0, n_web: int = 0) -> MultiEnv:
    env = MultiEnv(id="e", version=1, mcp_server_envs=[MagicMock() for _ in range(n_mcp)])
    env.website_envs = [MagicMock() for _ in range(n_web)]
    return env


async def _deployed_kwargs(env: MultiEnv, **deploy_kwargs) -> dict:
    """What the gateway's deploy path was handed.

    deploy() does more after that call than these tests care about, so any downstream
    explosion is swallowed.
    """
    with patch.object(EnvironmentGatewayProvider, "_deploy_gateway", AsyncMock()) as deploy_gateway, \
         patch("agent_env.env.env.Env.get"), \
         patch("agent_env.providers.get_env_sandbox_provider", MagicMock()), \
         patch("agent_env.providers.env_state.acquire_state_for_deploy", AsyncMock(return_value=None)):
        try:
            await env.deploy(**deploy_kwargs)
        except Exception:
            pass
    deploy_gateway.assert_awaited()
    return deploy_gateway.await_args.kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("services", [1, 6, 13, 50])
async def test_no_size_is_invented_whatever_the_service_count(services):
    kwargs = await _deployed_kwargs(_env(services))
    assert kwargs["cpu"] is None
    assert kwargs["memory_mb"] is None


@pytest.mark.asyncio
async def test_an_explicit_size_is_forwarded_untouched():
    """The escape hatch: a caller that sizes deliberately keeps both numbers."""
    kwargs = await _deployed_kwargs(_env(13), cpu=1.0, memory_mb=2048)
    assert kwargs["cpu"] == 1.0
    assert kwargs["memory_mb"] == 2048


@pytest.mark.asyncio
async def test_either_axis_can_be_set_alone():
    kwargs = await _deployed_kwargs(_env(13), memory_mb=32768)
    assert kwargs["cpu"] is None
    assert kwargs["memory_mb"] == 32768
