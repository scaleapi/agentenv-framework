"""The desktop example's computer as a child env of a multi env beside an MCP server, on the local backend: the deploy_env
step deploys the computer on a sandbox of its own and the gateway in front of both. Through the gateway's one /mcp an agent
sees the server's tools and the computer's under its name, takes a screenshot by that name, and a role rule narrows the
computer's tools as it would a server's; /svc/desktop/ reaches the computer. A later process reattaches the computer from the
multi env's record, and the run's teardown leaves nothing up. Needs Docker and the local registry on localhost:5000."""

import json
import re
import uuid

import httpx
import pytest
from agentenv_protocol.tool_access import (DISABLE, DISABLE_TOOL_EXTENSION_URI, ENABLE, ENABLE_TOOL_EXTENSION_URI,
                                           WILDCARD)
from agentenv_protocol.tool_context import DEFAULT_ROLE
from agentenv_protocol.types import WELL_KNOWN_PATH
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agent_env.env import Env
from agent_env.env.envs.computer.env import COMPUTER_TOOLS
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from tst.integration.env.envs.server_env_local_test import _put_items_image
from tst.integration.store.local_parallel_deploy_test import _docker, _seed_bootstrap_envs

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

UNPREFIXED = re.compile(r"^computer_")


async def _tools(deployed) -> tuple[set[str], bool]:
    """The tools the gateway lists to the default role, and whether a screenshot by the computer's name answered."""
    async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            shot = await session.call_tool("desktop_computer_screenshot", {}) if "desktop_computer_screenshot" in names else None
    return names, shot is not None and not shot.isError and any(part.type == "image" for part in shot.content)


def _containers(sandbox_id: str) -> list[str]:
    return _docker("ps", "-aq", "--filter", f"name=agent-{sandbox_id}").stdout.split()


@pytest.mark.asyncio
async def test_a_computer_in_a_multi_env_is_served_through_its_gateway_under_its_name_and_torn_down_with_it(desktop):
    uid = uuid.uuid4().hex[:8]
    with namespace_routing():
        _seed_bootstrap_envs()
        # Local ids, since the computer's is one and a store's id can't reference a local one.
        items = MCPServerEnv.put(id=f"@local/~/office-{uid}/items", environment_name="items",
                                 docker_image_artifact=_put_items_image(f"office-items-{uid}"))
        office = MultiEnv.put(id=f"@local/~/office-{uid}/office", mcp_server_envs=[items],
                              computer_envs=[Env.get(desktop.id)], name="office")
        step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                                 child_envs={"desktop": {"sandbox_type": "local", "env_params": {"form": "container"}}})
        assert step.preflight() == []
        context = TaskStepContext()
        try:
            [deployed] = (await step.execute(context)).deployed_envs
            computer = deployed.child_envs["desktop"]
            listed, shot = await _tools(deployed)
            async with httpx.AsyncClient(timeout=30) as client:
                card = (await client.get(f"{deployed.gateway_url}/svc/desktop{WELL_KNOWN_PATH}")).json()
            await deployed.invoke(DISABLE_TOOL_EXTENSION_URI, DISABLE, {"role": DEFAULT_ROLE, "tools": [WILDCARD]})
            await deployed.invoke(ENABLE_TOOL_EXTENSION_URI, ENABLE, {"role": DEFAULT_ROLE,
                                                                      "tools": ["desktop_computer_screenshot"]})
            narrowed, narrowed_shot = await _tools(deployed)
            reattached = await Env.from_instance_id(deployed.instance_id)
            up = _containers(computer["sandbox_id"])
        finally:
            report = await teardown_run(context)

    assert not report.failed
    assert deployed.get_child_env_card("desktop") is not None and card["name"] == "desktop"
    assert "list_items" in listed and "desktop_computer_screenshot" in listed and shot
    assert {name for name in listed if "computer_" in name} <= {f"desktop_{tool}" for tool in COMPUTER_TOOLS}
    assert not any(UNPREFIXED.match(name) for name in listed)
    assert narrowed == {"desktop_computer_screenshot"} and narrowed_shot
    assert deployed.sandbox_ids["local"] == {"desktop": computer["sandbox_id"]} and up
    assert [env._deployed.sandbox_id for env in reattached.computer_envs] == [computer["sandbox_id"]]
    assert _containers(computer["sandbox_id"]) == []
    assert json.dumps(deployed.child_envs)  # the record stores as JSON
