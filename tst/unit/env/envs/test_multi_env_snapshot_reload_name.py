"""The snapshot reload regenerates the compose; it must reuse the name and host IPs the running gateway already has."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.artifact import DockerImageArtifact
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.env_providers import EnvironmentGatewayProvider


class _Stop(Exception):
    """Abort right after the compose is rendered; the rest needs a real sandbox."""


@pytest.mark.asyncio
@pytest.mark.parametrize("declared, deployed_as, expected", [("crm", None, "crm"), (None, "env4821", "env4821"), ("crm", "crm", "crm")])
async def test_snapshot_reload_keeps_the_env_name(declared, deployed_as, expected):
    env = MultiEnv(id="crm-suite", version=1, mcp_server_envs=[], name=declared)
    env._mcp_server_name = deployed_as
    env._sandbox = MagicMock(mode="vm", load_docker_images=AsyncMock(), host_ips=("127.0.0.1",))
    db_image = MagicMock(spec=DockerImageArtifact, image_name="snap:1")
    snapshot = MagicMock(db_image_artifact_id="snap", db_image_artifact_version=1)
    gateway_env, service_db = MagicMock(), MagicMock()

    with patch("agent_env.artifact.Artifact.get", return_value=db_image), \
         patch("agent_env.env.env.Env.get", side_effect=lambda env_id, *a, **k: gateway_env if env_id == "gw-id" else service_db), \
         patch("agent_env.config.get_config", return_value=MagicMock(default_gateway_env_id="gw-id", default_service_db_env_id="db-id")), \
         patch("agent_env.providers.env_state.LocalPostgresStateProvider"), \
         patch.object(EnvironmentGatewayProvider, "create_docker_compose", side_effect=_Stop) as compose:
        with pytest.raises(_Stop):
            await env._load_from_snapshot(snapshot)

    assert compose.call_args.kwargs["mcp_server_name"] == expected
    assert compose.call_args.kwargs["host_ips"] == ("127.0.0.1",)
    assert compose.call_args.kwargs["host_port"] == env._sandbox.host_port
