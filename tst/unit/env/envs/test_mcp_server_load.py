"""Unit tests for MCPServerEnv universe-load selection and load_artifact dispatch."""

import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact
from agent_env.env.envs.mcp_server import MCPServerEnv


def _env(environment_name: str = "email") -> MCPServerEnv:
    return MCPServerEnv(id="mcp-email", version=1, docker_image_artifact=None, environment_name=environment_name, service_version=1)


def _universe(service_names: list[str], universe_id: str = "uni-1"):
    services = [types.SimpleNamespace(environment_name=name) for name in service_names]
    return types.SimpleNamespace(id=universe_id, get_environment_artifacts=lambda: services)


@pytest.mark.asyncio
async def test_rejects_universe_without_matching_service():
    with pytest.raises(RuntimeError) as exc:
        await _env("email").load_environment_universe_artifact(_universe(["slack", "calendar"], "uni-nomatch"))
    msg = str(exc.value)
    assert "mcp-email" in msg and "email" in msg and "uni-nomatch" in msg


@pytest.mark.asyncio
async def test_loads_matching_service_from_multi_service_universe():
    # Selection passes, so it reaches the (undeployed) load rather than rejecting as unhostable.
    with pytest.raises(RuntimeError) as exc:
        await _env("email").load_environment_universe_artifact(_universe(["email", "slack"], "uni-multi"))
    msg = str(exc.value)
    assert "not deployed" in msg
    assert "no EnvironmentArtifact" not in msg


@pytest.mark.asyncio
async def test_load_artifact_dispatches_universe():
    env = _env()
    env.load_environment_universe_artifact = AsyncMock(return_value="UNI")
    env.load_environment_artifact = AsyncMock(return_value="SVC")
    artifact = MagicMock(spec=EnvironmentUniverseArtifact)
    assert await env.load_artifact(artifact) == "UNI"
    env.load_environment_universe_artifact.assert_awaited_once_with(artifact)
    env.load_environment_artifact.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_artifact_dispatches_service():
    env = _env()
    env.load_environment_universe_artifact = AsyncMock(return_value="UNI")
    env.load_environment_artifact = AsyncMock(return_value="SVC")
    artifact = MagicMock(spec=EnvironmentArtifact)
    await env.load_artifact(artifact)
    env.load_environment_artifact.assert_awaited_once_with(artifact)
    env.load_environment_universe_artifact.assert_not_awaited()


@pytest.mark.asyncio
async def test_load_artifact_rejects_unsupported_type():
    with pytest.raises(ValueError):
        await _env().load_artifact(types.SimpleNamespace(id="x", type="file"))
