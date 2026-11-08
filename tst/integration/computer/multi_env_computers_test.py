"""The desktop example's computer as a child env of a multi env beside an MCP server, on the local backend.

The deploy_env step deploys each computer on a sandbox of its own and the gateway in front of them all. Through the
gateway's one /mcp an agent sees the server's tools and each computer's under its name, takes screenshots by those names,
and a role rule narrows them as it would a server's; /svc/<computer>/ reaches the computer. A later process reattaches the
computers from the multi env's record, and the run's teardown leaves nothing up. Two computers keep serving through a
computer's restart and the gateway's, a failed redeploy leaves the earlier deployment serving, and a computer that fails
mid-deploy, or an entry the multi env refuses, leaves nothing up. Needs Docker and the local registry on localhost:5000."""

import asyncio
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
from mcp.client.streamable_http import streamable_http_client

from agent_env.env import Env
from agent_env.env.envs.computer.env import COMPUTER_TOOLS, ComputerEnv
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.env_providers.env_gateway_provider import EnvironmentGatewayProvider
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from tst.integration.env.envs.server_env_local_test import _put_items_image
from tst.integration.store.local_parallel_deploy_test import _docker, _seed_bootstrap_envs

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

UNPREFIXED = re.compile(r"^computer_")
CONTAINER = {"sandbox_type": "local", "env_params": {"form": "container"}}


@pytest.fixture(scope="module")
def items(desktop):
    """The gateway's and service-db's envs, and the items MCP server under a local id, since a multi env naming the
    desktop's local id must be local too and a store's id can't reference a local one."""
    with namespace_routing():
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        return MCPServerEnv.put(id=f"@local/~/office-{uid}/items", environment_name="items",
                                docker_image_artifact=_put_items_image(f"office-items-{uid}"))


def _multi(items, *computers) -> MultiEnv:
    return MultiEnv.put(id=f"@local/~/office-{uuid.uuid4().hex[:8]}/office", mcp_server_envs=[items],
                        computer_envs=list(computers), name="office")


async def _tools(deployed) -> tuple[set[str], bool]:
    """The tools the gateway lists to the default role, and whether a screenshot by the desktop's name answered."""
    async with streamable_http_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            shot = await session.call_tool("desktop_computer_screenshot", {}) if "desktop_computer_screenshot" in names else None
    return names, shot is not None and not shot.isError and any(part.type == "image" for part in shot.content)


async def _answers(deployed, tools: list[str], *, until: bool = True, tries: int = 40) -> dict[str, bool]:
    """Whether each of ``tools`` answers through the gateway, called again every 3 s until all answer, or with
    ``until=False``, once."""
    answered: dict[str, bool] = {}
    for _ in range(tries if until else 1):
        try:
            async with streamable_http_client(deployed.mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    answered = {tool: not (await session.call_tool(tool, {})).isError for tool in tools}
        except Exception:  # noqa: BLE001 - the gateway restarting
            answered = dict.fromkeys(tools, False)
        if all(answered.values()) or not until:
            return answered
        await asyncio.sleep(3)
    return answered


def _containers(sandbox_id: str) -> list[str]:
    return _docker("ps", "-aq", "--filter", f"name=agent-{sandbox_id}").stdout.split()


def _ours() -> set[str]:
    """The containers local sandboxes run: computers' and gateway stacks'."""
    return {name for name in _docker("ps", "-a", "--format", "{{.Names}}").stdout.split() if name.startswith("agent-")}


def _gateway(deployed) -> str:
    return next(name for name in _ours() if name.endswith("-gateway-1") and deployed.sandbox_id.removeprefix("local-") in name)


@pytest.mark.asyncio
async def test_a_computer_in_a_multi_env_is_served_through_its_gateway_under_its_name_and_torn_down_with_it(desktop, items):
    with namespace_routing():
        office = _multi(items, Env.get(desktop.id))
        step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                                 child_envs={"desktop": CONTAINER})
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


@pytest.mark.asyncio
async def test_two_computers_keep_serving_through_restarts_and_a_failed_redeploy_and_close_ends_them(desktop, items,
                                                                                                     monkeypatch):
    """One deployment, to keep the tier's time: two computers side by side, the desktop's container killed and started
    again, the gateway restarted, then a redeploy of the same env object whose gateway fails, which leaves this
    deployment as it was; close() ends it all."""
    every = ["list_items", "desktop_computer_screenshot", "laptop_computer_screenshot"]
    before = _ours()
    with namespace_routing():
        laptop = ComputerEnv.put(id=f"{desktop.id}-laptop", computer=desktop.computer, driver=desktop.driver,
                                 environment_name="laptop")
        env = Env.get(_multi(items, Env.get(desktop.id), laptop).id)
        try:
            deployed = await env.deploy(sandbox_type="local", ttl_seconds=900,
                                        child_envs={"desktop": CONTAINER, "laptop": CONTAINER})
            desktop_box = f"agent-{deployed.child_envs['desktop']['sandbox_id']}"
            both = await _answers(deployed, every)

            _docker("kill", desktop_box)
            killed = await _answers(deployed, every, until=False)
            _docker("start", desktop_box)
            started = await _answers(deployed, every)
            _docker("restart", _gateway(deployed))
            restarted = await _answers(deployed, every)

            deployed_before = _ours()
            real = EnvironmentGatewayProvider._compose_up

            async def no_gateway(self, sandbox, args=""):
                if not args:
                    raise RuntimeError("the gateway didn't start")
                return await real(self, sandbox, args)

            with monkeypatch.context() as patch:
                patch.setattr(EnvironmentGatewayProvider, "_compose_up", no_gateway)
                with pytest.raises(RuntimeError, match="the gateway didn't start"):
                    await env.deploy(sandbox_type="local", ttl_seconds=900,
                                     child_envs={"desktop": CONTAINER, "laptop": CONTAINER})
            after_failure = _ours()
            still = await _answers(deployed, every)
            kept = [computer._deployed.sandbox_id for computer in env.computer_envs]
        finally:
            await env.close()

    assert all(both.values()), both
    assert killed == {"list_items": True, "desktop_computer_screenshot": False, "laptop_computer_screenshot": True}
    assert all(started.values()) and all(restarted.values()), (started, restarted)
    assert after_failure == deployed_before and all(still.values()), (after_failure - deployed_before, still)
    assert kept == [deployed.child_envs[name]["sandbox_id"] for name in ("desktop", "laptop")]
    assert _ours() == before


@pytest.mark.asyncio
async def test_a_computer_that_fails_mid_deploy_fails_the_deploy_and_leaves_nothing_up(desktop, items, monkeypatch):
    async def refuse(self, deployed):
        raise RuntimeError("the computer's tools couldn't be applied")

    before = _ours()
    monkeypatch.setattr(ComputerEnv, "_give_tools", refuse)
    with namespace_routing():
        office = _multi(items, Env.get(desktop.id))
        context = TaskStepContext()
        with pytest.raises(RuntimeError, match="the computer's tools couldn't be applied"):
            await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=600,
                                    child_envs={"desktop": CONTAINER}).execute(context)

    assert context.deployed_envs == [] and _ours() == before


@pytest.mark.asyncio
async def test_a_computer_entry_the_multi_env_refuses_makes_no_sandbox(desktop, items, monkeypatch):
    made = []
    for name in ("create_sandbox", "create_vm", "create_container"):
        monkeypatch.setattr(LocalSandboxProvider, name, lambda *args, _name=name, **kwargs: made.append(_name))
    with namespace_routing():
        office = _multi(items, Env.get(desktop.id))
        refused = [
            ({"desktop": {**CONTAINER, "env_params": {"site_routing": True}}},
             "child_envs.desktop: env_params.site_routing: a computer in a multi env doesn't route sites yet; deploy it "
             "beside the multi env to route them"),
            ({"desktop": {"tools": "standard"}},
             "child_envs.desktop: an entry is a table of sandbox_type, cpu, memory_mb, disk_size_gb, env_params, not tools"),
        ]
        for child_envs, problem in refused:
            step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=600,
                                     child_envs=child_envs)
            assert step.preflight() == [f"deploy_env 'office': {problem}"]
            with pytest.raises(ValueError, match=re.escape(problem)):
                await step.execute(TaskStepContext())

        monkeypatch.setattr(LocalSandboxProvider, "REQUEST_HEADERS", {"*.example": {"X-Token": "secret"}})
        [problem] = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                      ttl_seconds=600).preflight()

    assert "takes credentials on each request to a sandbox's URL" in problem
    assert "set child_envs.desktop.sandbox_type to a sandbox type whose URLs take none" in problem
    assert made == []
