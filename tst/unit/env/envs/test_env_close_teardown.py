"""Unit tests for env close() routing teardown through the gateway provider."""

from unittest.mock import AsyncMock

import pytest

from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.website import WebsiteEnv


def _mcp_env() -> MCPServerEnv:
    return MCPServerEnv(id="mcp-email", version=1, docker_image_artifact=None, environment_name="email", service_version=1)


def _website_env() -> WebsiteEnv:
    return WebsiteEnv(
        id="web-email",
        version=1,
        backend_docker_image_artifact=None,
        frontend_docker_image_artifact=None,
        environment_name="email",
        service_version=1,
    )


@pytest.mark.parametrize("make_env", [_mcp_env, _website_env])
@pytest.mark.asyncio
async def test_close_closes_gateway_provider_and_terminates_sandbox(make_env):
    env = make_env()
    env._gateway_provider = AsyncMock()
    sandbox = AsyncMock()
    sandbox.sandbox_id = "sb-1"
    env._sandbox = sandbox

    await env.close()

    env._gateway_provider.close.assert_awaited_once()
    sandbox.terminate.assert_awaited_once()
    assert env._sandbox is None


@pytest.mark.parametrize("make_env", [_mcp_env, _website_env])
@pytest.mark.asyncio
async def test_gateway_close_error_does_not_block_sandbox_termination(make_env):
    env = make_env()
    env._gateway_provider = AsyncMock()
    env._gateway_provider.close.side_effect = RuntimeError("drop database failed")
    sandbox = AsyncMock()
    sandbox.sandbox_id = "sb-1"
    env._sandbox = sandbox

    # Must not raise, and the sandbox must still be terminated.
    await env.close()

    env._gateway_provider.close.assert_awaited_once()
    sandbox.terminate.assert_awaited_once()
    assert env._sandbox is None


@pytest.mark.parametrize("make_env", [_mcp_env, _website_env])
@pytest.mark.asyncio
async def test_close_with_no_sandbox_still_closes_gateway_provider(make_env):
    env = make_env()
    env._gateway_provider = AsyncMock()
    env._sandbox = None

    await env.close()

    env._gateway_provider.close.assert_awaited_once()
