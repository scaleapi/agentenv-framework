"""Run explicitly with ``pytest tst/manual/smol_vm_gateway_check.py`` on a KVM host with Docker and Smol installed."""

import uuid

import pytest
from agentenv_protocol import client as protocol_v1

from agent_env.env import legacy_protocol
from agent_env.env.env import DeployedGatewayEnv, Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from tst.integration.env.envs.multi_env_local_test import _items_universe
from tst.integration.env.envs.server_env_local_test import _put_items_image
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs, local_stack  # noqa: F401


@pytest.mark.asyncio
async def test_gateway_restores_and_serves_data_on_smol_vm(local_stack):
    _seed_bootstrap_envs()
    uid = uuid.uuid4().hex[:8]
    items = MCPServerEnv.put(
        id=f"smol-items-{uid}",
        environment_name="items",
        docker_image_artifact=_put_items_image(f"smol-items-img-{uid}"),
    )
    env = MultiEnv.put(id=f"smol-multi-{uid}", mcp_server_envs=[items], name="smol-suite")

    try:
        deployed = await env.deploy(sandbox_type="smol_vm", ttl_seconds=900, disk_size_gb=16, cpu=2, memory_mb=4096)
        assert isinstance(deployed, DeployedGatewayEnv)
        assert deployed.sandbox_type == "smol_vm"
        assert deployed.gateway_url.startswith("http://127.0.0.1:")

        restored = await Env.from_instance_id(deployed.instance_id)
        await restored.load_environment_universe_artifact(_items_universe(uid))
        base_url = await legacy_protocol.v1_base_url(deployed, deployed.gateway_url, "items")
        payload = (await protocol_v1.get_data(base_url)).model_dump()["parts"][0]["data"]
        assert payload == {"items": ["snap-x", "snap-y"]}
    finally:
        if env._sandbox is not None:
            await env.close()
