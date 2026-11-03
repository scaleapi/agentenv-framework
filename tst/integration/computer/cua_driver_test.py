"""The cua-driver driver on the ubuntu-24.04 base, as the cua-driver test bundle's desktop names it: its put composes
an image for each arch this machine's Docker builds, gate 3 makes this machine's ready at full with cua-driver's own tools
beside the computer_* tools; deployed, an agent's session gets cua-driver's instructions at connect and reads its skill
resources, drives Thunar by element token through cua-driver's tools, in a cua-driver session another agent's doesn't
see, computer-ready reports the pinned cua-driver, and the skill leads with cua-driver's own pack. Needs Docker and the local
registry on localhost:5000."""

import asyncio
import json
import platform
import re
from pathlib import Path

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import AnyUrl

import agent_env
from agentenv_protocol import SKILL_EXTENSION_URI
from agentenv_protocol.computer import COMPUTER_READY

from agent_env.env import Env
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
MANIFEST = Path(agent_env.__file__).parent / "drivers" / "envs" / "cua-driver" / "mcp.json"
COMPUTER_TOOLS = {"computer_screenshot", "computer_click", "computer_drag", "computer_scroll", "computer_type",
                  "computer_key", "computer_wait"}


def test_gate_3_makes_this_machines_arch_ready_with_cua_drivers_tools_beside_the_computers(cua_desktop):
    results = cua_desktop.metadata["composed"]["container"]
    pinned = {tool["name"] for tool in json.loads(MANIFEST.read_text())["tools"]}

    assert results[ARCH]["status"] == "ready" and results[ARCH]["emulated"] is False
    assert results[ARCH]["probe"]["passed"] and results[ARCH]["probe"]["level"] == "full"
    assert results[ARCH]["installed"]["cua-driver"] == "0.38.0"
    assert {tool["name"] for tool in results[ARCH]["card"]["capabilities"]["tools"]} == pinned | COMPUTER_TOOLS
    for arch, result in results.items():
        if arch != ARCH:
            assert result["status"] == "unverified" or "its Docker has no emulation for it" in result["reason"], result


@pytest.mark.asyncio
async def test_an_agent_drives_thunar_by_element_token_in_its_own_session_and_its_skill_leads_with_cua_drivers_pack(
        cua_desktop):
    with namespace_routing():
        deployed = await Env.get(cua_desktop.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            async with streamablehttp_client(f"{deployed.environment_url}/mcp") as (read, write, _):
                async with ClientSession(read, write) as session:
                    connected = await session.initialize()
                    linux = await session.read_resource(AnyUrl("skill://cua-driver/LINUX.md"))
                    started = await session.call_tool("start_session", {"session": "agent"})
                    mine = await session.call_tool("list_sessions", {})
                    launched = await session.call_tool("launch_app", {"name": "thunar"})
                    window = await _window(session, "user - Thunar")
                    state = await session.call_tool("get_window_state", {"pid": window["pid"],
                                                                         "window_id": window["window_id"]})
                    tree, snapshot = state.structuredContent["tree_markdown"], state.structuredContent["snapshot_id"]
                    row = re.search(r'\[(\d+)\] push button "Open Parent"', tree).group(1)
                    clicked = await session.call_tool("click", {"element_token": f"{snapshot}:{row}"})
                    parent = await _window(session, "home - Thunar")
            async with streamablehttp_client(f"{deployed.environment_url}/mcp") as (read, write, _):
                async with ClientSession(read, write) as other:
                    await other.initialize()
                    others = await other.call_tool("list_sessions", {})
            ready = await deployed.invoke(COMPUTER_READY, "status", None)
            skill = await deployed.invoke(SKILL_EXTENSION_URI, "get", {"name": "desktop"})
        finally:
            report = await teardown_run(context)

    assert not launched.isError and not clicked.isError and parent["window_id"] == window["window_id"]
    assert not started.isError and "agent" in _named(mine) and "agent" not in _named(others)
    assert connected.instructions == json.loads(MANIFEST.read_text())["instructions"] and "AT-SPI" in linux.contents[0].text
    checks = {check["name"]: check for check in ready["checks"]}
    assert ready["ready"] and checks["cua-driver"]["ok"] and checks["cua-driver tools"]["detail"] == "66 tools, as pinned"
    assert {"LINUX.md", "WORKFLOW.md", "RUNTIME.md"} <= set(skill["files"])
    assert "# This computer" in skill["skill_md"] and skill["description"].startswith("Drive a native GUI app")
    assert not report.failed


def _named(listed) -> set[str]:
    return {session["session"] for session in listed.structuredContent["sessions"]}


async def _window(session: ClientSession, title: str) -> dict:
    """Thunar's window once its title is ``title``."""
    for _ in range(40):
        listed = await session.call_tool("list_windows", {"app": "thunar"})
        if found := [w for w in listed.structuredContent["windows"] if w["title"] == title]:
            return found[0]
        await asyncio.sleep(0.5)
    raise TimeoutError(f"no Thunar window titled {title!r}: {listed.structuredContent['windows']}")
