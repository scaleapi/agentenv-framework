"""Run explicitly with ``pytest tst/manual/smol_vm_gateway_check.py`` on a KVM host with Docker and Smol installed."""

import json
import uuid

import pytest
from agentenv_protocol import client as protocol_v1

from agent_env.a2a_agent.a2a_agent import DEFAULT_A2A_PORT
from agent_env.env import legacy_protocol
from agent_env.env.env import DeployedGatewayEnv, Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.sandbox_providers.smol_vm.provider import SmolVmSandboxProvider
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from tst.integration.env.envs.multi_env_local_test import _items_universe
from tst.integration.env.envs.server_env_local_test import _put_items_image
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs, local_stack  # noqa: F401
from tst.util.a2a_test_agent import put_test_agent


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

    context = TaskStepContext(instance_id=f"smol-env-agent-run-{uid}")

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

        context.deployed_envs.append(deployed)
        agent = put_test_agent(f"smol-agent-{uid}")
        await DeploySandboxTaskStep(
            id="box", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="smol_vm",
            exposed_ports=[DEFAULT_A2A_PORT], cpu=2, memory_mb=4096, disk_size_gb=16,
        ).execute(context)
        await DeployAgentTaskStep(
            id="agent", version=None, env_ids=[env.id], a2a_agent_id=agent.id,
            agent_name="solver", sandbox_name="box",
        ).execute(context)

        agent_vm = await SmolVmSandboxProvider().get_sandbox(context.deployed_sandboxes[0].sandbox_id)
        code, names, stderr = await agent_vm.exec_with_output("docker", "ps", "--format", "{{.Names}}")
        assert code == 0, stderr
        assert len(names.splitlines()) == 1
        reachable = SmolVmSandboxProvider.get_external_url(deployed.gateway_url)
        code, status, stderr = await agent_vm.exec_with_output(
            "docker", "exec", names.strip(), "python", "-c",
            "import sys; from urllib.request import urlopen; print(urlopen(sys.argv[1], timeout=20).status)",
            f"{reachable}/.well-known/agent-env.json",
        )
        assert code == 0, stderr
        assert status.strip() == "200"

        code, data, stderr = await agent_vm.exec_with_output(
            "docker", "exec", names.strip(), "python", "-c",
            "import asyncio, sys; from agentenv_protocol import client; "
            "print(asyncio.run(client.get_data(sys.argv[1])).model_dump_json())",
            SmolVmSandboxProvider.get_external_url(base_url),
        )
        assert code == 0, stderr
        assert json.loads(data)["parts"][0]["data"] == {"items": ["snap-x", "snap-y"]}
    finally:
        report = await teardown_run(context)
        if env._sandbox is not None:
            await env.close()
        assert not report.still_up
