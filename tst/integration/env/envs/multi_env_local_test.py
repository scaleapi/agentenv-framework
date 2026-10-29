"""A MultiEnv on the local backend: two MCP servers behind one gateway and the local Postgres store, on docker compose.

The env deploys, a fresh object restores its record as a later process would, loads a universe into one of its servers,
and close() takes the whole compose stack down. Requires a docker daemon; spins up a throwaway ``registry:2`` and skips
if it can't start.
"""

import json
import uuid
from pathlib import Path

import pytest
from agentenv_protocol import client as protocol_v1

from agent_env.artifact import DockerImageArtifact, EnvironmentArtifact, EnvironmentUniverseArtifact, FileArtifact
from agent_env.env import legacy_protocol
from agent_env.env.env import DeployedGatewayEnv, Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from tst.integration.env.envs.server_env_local_test import _put_items_image
from tst.integration.store.local_parallel_deploy_test import _docker, _seed_bootstrap_envs, local_stack as local_stack

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

_TST_DATA = Path(__file__).resolve().parents[3] / "data"


@pytest.mark.asyncio
async def test_a_multi_env_deploys_restores_loads_and_tears_down_on_local_compose(local_stack):
    _seed_bootstrap_envs()
    uid = uuid.uuid4().hex[:8]
    slack_tag = f"multi-slack-{uid}"
    build = _docker("build", "-t", slack_tag, "-f", str(_TST_DATA / "slack_mcp/Dockerfile"), str(_TST_DATA))
    assert build.returncode == 0, f"slack_mcp build failed: {build.stderr[-2000:]}"
    slack = MCPServerEnv.put(id=f"multi-slack-{uid}", environment_name="slack",
                             docker_image_artifact=DockerImageArtifact.put(id=slack_tag, description="multi env e2e", image_name=slack_tag))
    items = MCPServerEnv.put(id=f"multi-items-{uid}", environment_name="items",
                             docker_image_artifact=_put_items_image(f"multi-items-{uid}"))
    env = MultiEnv.put(id=f"multi-{uid}", mcp_server_envs=[slack, items], name="suite")

    deployed = await env.deploy(sandbox_type="local", ttl_seconds=900)
    work_dir = env._sandbox.work_dir
    try:
        assert type(deployed) is DeployedGatewayEnv and deployed.sandbox_ids == {"gateway_server": deployed.sandbox_id}
        assert deployed.mcp_server_name == "suite" and len(deployed.env_state_instance_ids) == 1
        assert [c["name"] for c in deployed.environment_card["children_environments"]] == ["items"]  # slack_mcp serves no card

        restored = await Env.from_instance_id(deployed.instance_id)
        assert isinstance(restored, MultiEnv) and restored._sandbox.work_dir == work_dir
        await restored.load_environment_universe_artifact(_items_universe(uid))
        base_url = await legacy_protocol.v1_base_url(deployed, deployed.gateway_url, "items")
        assert (await protocol_v1.get_data(base_url)).model_dump()["parts"][0]["data"] == {"items": ["snap-x", "snap-y"]}

        await restored.close()
        assert _stack(work_dir) == ""
    finally:
        _docker("compose", "--project-directory", str(work_dir), "down", "-v", "--remove-orphans")


def _stack(work_dir: Path) -> str:
    """The ids of the containers still running from the compose stack in work_dir."""
    return _docker("ps", "-aq", "--filter", f"label=com.docker.compose.project.working_dir={work_dir}").stdout.strip()


def _items_universe(uid: str) -> EnvironmentUniverseArtifact:
    items = EnvironmentArtifact.put(
        id=f"multi-items-data-{uid}",
        environment_name="items",
        file_artifact=FileArtifact.put_bytes(id=f"multi-items-file-{uid}", description="multi env e2e", filename="items.json",
                                             content=json.dumps({"items": ["snap-x", "snap-y"]}).encode(), content_type="application/json"),
    )
    return EnvironmentUniverseArtifact.put(id=f"multi-universe-{uid}", environment_artifacts=[items])
