"""A multi env's deploy_env with child_envs on the local backend: an entry the multi env refuses is refused at save and at
the run before any sandbox is made, and the same env deployed without child_envs serves its MCP server through the
gateway. Requires a docker daemon; spins up a throwaway ``registry:2`` and skips if it can't start.
"""

import re
import uuid

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent_env.env.env import Env
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from tst.integration.env.envs.server_env_local_test import _put_items_image
from tst.integration.store.local_parallel_deploy_test import _docker, _seed_bootstrap_envs, local_stack as local_stack

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

_SHARED = "runs on the multi env's own sandbox, which the step's own fields set, so it takes no entry"


@pytest.fixture
def made(monkeypatch) -> list[str]:
    """The local provider's sandbox-making calls, by name, each still made."""
    calls: list[str] = []
    for name in ("create_sandbox", "create_vm", "create_container"):
        real = getattr(LocalSandboxProvider, name)

        async def spy(self, *args, _real=real, _name=name, **kwargs):
            calls.append(_name)
            return await _real(self, *args, **kwargs)

        monkeypatch.setattr(LocalSandboxProvider, name, spy)
    return calls


@pytest.mark.asyncio
async def test_child_envs_the_multi_env_refuses_make_no_sandbox_and_without_them_it_deploys(local_stack, made):
    _seed_bootstrap_envs()
    uid = uuid.uuid4().hex[:8]
    items = MCPServerEnv.put(id=f"child-items-{uid}", environment_name="items",
                             docker_image_artifact=_put_items_image(f"child-items-{uid}"))
    env = MultiEnv.put(id=f"child-multi-{uid}", mcp_server_envs=[items], name="suite")

    refused = [
        ({"child_envs": {"items": {"cpu": 2}}}, f"child_envs.items: items {_SHARED}"),
        ({"child_envs": {"desktop": {"cpu": 2}}},
         f"child_envs.desktop: env '{env.id}' has no child env named 'desktop'; its child envs are items"),
        ({"child_envs": {"items": {"ttl_seconds": 60}}},
         "child_envs.items: an entry is a table of sandbox_type, cpu, memory_mb, disk_size_gb, env_params, not ttl_seconds"),
        ({"env_params": {"form": "container"}}, "multi envs take no env_params"),
    ]
    for options, problem in refused:
        step = DeployEnvTaskStep(id="env", version=None, env_id=env.id, sandbox_type="local", ttl_seconds=900, **options)
        assert step.preflight() == [f"deploy_env 'env': {problem}"]
        context = TaskStepContext()
        with pytest.raises(ValueError, match=re.escape(problem)):
            await step.execute(context)
        assert context.deployed_envs == []
    assert made == []

    step = DeployEnvTaskStep(id="env", version=None, env_id=env.id, sandbox_type="local", ttl_seconds=900)
    assert step.preflight() == []
    [deployed] = (await step.execute(TaskStepContext())).deployed_envs
    deployed_env = await Env.from_instance_id(deployed.instance_id)
    work_dir = deployed_env._sandbox.work_dir
    try:
        assert made
        async with streamable_http_client(deployed.mcp_url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                assert "list_items" in [tool.name for tool in (await session.list_tools()).tools]
                assert not (await session.call_tool("list_items", {})).isError
        await deployed_env.close()
    finally:
        _docker("compose", "--project-directory", str(work_dir), "down", "-v", "--remove-orphans")
