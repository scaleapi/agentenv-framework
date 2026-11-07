"""The desktop example's computer deployed twice side by side with tool prefixes, as a multi env's computers will be:
each serves every tool under its own prefix and takes screenshots by those names, its skill names its tools that way,
and one whose ``tools`` is ``standard`` gives its agents only its prefixed ``computer_*`` tools, at deploy and again
after a reset. Needs Docker and the local registry on localhost:5000."""

import re

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agentenv_protocol.types import SKILL_EXTENSION_URI

from agent_env.env import Env
from agent_env.env.envs.computer.env import COMPUTER_TOOLS, ComputerEnv
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

UNPREFIXED = re.compile(r"(?<![\w-])computer_")


async def _served(deployed, prefix: str) -> tuple[set[str], bool]:
    """The tools the computer lists to its agents, and whether a screenshot by its prefixed name answered."""
    async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            shot = await session.call_tool(f"{prefix}computer_screenshot", {})
    return names, not shot.isError and any(part.type == "image" for part in shot.content)


@pytest.mark.asyncio
async def test_two_computers_serve_their_tools_under_their_own_prefixes_side_by_side_and_through_a_reset(desktop):
    with namespace_routing():
        standard = ComputerEnv.put(id=f"{desktop.id}-standard", computer=desktop.computer, driver=desktop.driver,
                                   environment_name="laptop", tools="standard")
        context = TaskStepContext()
        try:
            context.deployed_envs.append(await Env.get(desktop.id).deploy(tool_prefix="desktop_"))
            context.deployed_envs.append(await Env.get(standard.id).deploy(tool_prefix="laptop_"))
            left, right = context.deployed_envs
            left_names, left_shot = await _served(left, "desktop_")
            right_names, right_shot = await _served(right, "laptop_")
            skill = await left.invoke(SKILL_EXTENSION_URI, "get", {"name": "desktop"})
            await ResetEnvTaskStep(id="reset", version=None, env_id=standard.id, fail_task_on_error=True).execute(context)
            after_reset, after_shot = await _served(right, "laptop_")
        finally:
            report = await teardown_run(context)

    assert not report.failed
    assert left_shot and right_shot and after_shot
    assert left_names and all(name.startswith("desktop_") for name in left_names)
    assert "desktop_computer_screenshot" in left_names and "computer_screenshot" not in left_names
    assert right_names == after_reset and right_names <= {f"laptop_{tool}" for tool in COMPUTER_TOOLS}
    assert "laptop_computer_click" in right_names
    assert "`desktop_computer_screenshot`" in skill["skill_md"] and not UNPREFIXED.search(skill["skill_md"])
