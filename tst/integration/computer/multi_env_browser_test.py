"""agentenv's chromium as a multi env's website browser, on the local backend: ``browser_envs = [chromium]`` deploys it
on a sandbox of its own as one more computer child, beside a synthetic site, with no playwright-mcp. It routes the site,
so its start tab loads it by name through the gateway; an agent drives it through the gateway's one /mcp with its tools
under ``chromium_``; its CDP address, asked through ``/svc/chromium/``, names the gateway and reaches the site by name;
reset_env brings it back after its container is killed and after the gateway restarts; and local Docker, which can't
hold its network to the gateway, runs it allowing every host and says so. ``browser_envs = []`` deploys no browser.
Needs Docker and the local registry on localhost:5000."""

import asyncio
import json
import logging
import subprocess
import uuid

import httpx
import pytest
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.website_browser import BROWSER
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent_env.env import Env
from agent_env.env.env import DeployedEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.playwright_mcp import PLAYWRIGHT_MCP_ENVIRONMENT_NAME
from agent_env.env.envs.website import WebsiteEnv
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from tst.integration.computer.browser_computer_test import _cdp_agent_loads
from tst.integration.computer.multi_env_computers_test import _containers, _gateway, _image
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

SITE_URL = "https://shop.test/api/health"
CHROMIUM = {"sandbox_type": "local",
            "env_params": {"form": "container", "app_settings": {"chromium": {"start_urls": [SITE_URL]}}}}
#: How long the browser may take to answer again after a restart or a reset; how often it's asked meanwhile.
WAIT_S, POLL_S = 120.0, 2.0


@pytest.fixture(scope="module")
def shop(browser_computer):
    """The gateway's and service-db's envs, and a website whose backend is the small agentenv website, named
    shop.test, under a local id, since a multi env naming chromium's local id must be local too."""
    with namespace_routing():
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        return WebsiteEnv.put(id=f"@local/~/browsing-{uid}/shop", environment_name="shop", domains=["shop.test"],
                              backend_docker_image_artifact=_image("agentenv_website", f"browsing-shop-{uid}"))


def _office(shop, browser_envs) -> MultiEnv:
    return MultiEnv.put(id=f"@local/~/browsing-{uuid.uuid4().hex[:8]}/office", mcp_server_envs=[],
                        website_envs=[shop], browser_envs=browser_envs, name="office")


async def _drive(deployed) -> dict:
    """What an agent sees through the gateway's /mcp: the tools it lists, a screenshot by chromium's name, and the
    tabs after it opens the site by name."""
    async with streamable_http_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {tool.name for tool in (await session.list_tools()).tools}
            shot = await session.call_tool("chromium_computer_screenshot", {})
            opened = await session.call_tool("chromium_computer_navigate", {"url": SITE_URL})
            tabs = await session.call_tool("chromium_computer_tabs", {"action": "list"})
    return {"tools": tools, "shot": not shot.isError and any(part.type == "image" for part in shot.content),
            "opened": not opened.isError,
            "tabs": " ".join(part.text for part in tabs.content if part.type == "text")}


async def _answers_again(deployed) -> bool:
    """Whether chromium's screenshot answers through the gateway, asked again until it does or ``WAIT_S`` passes."""
    deadline = asyncio.get_running_loop().time() + WAIT_S
    while asyncio.get_running_loop().time() < deadline:
        try:
            async with streamable_http_client(deployed.mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    if not (await session.call_tool("chromium_computer_screenshot", {})).isError:
                        return True
        except Exception:  # noqa: BLE001 - the gateway or the browser restarting
            pass
        await asyncio.sleep(POLL_S)
    return False


async def _cdp(deployed, browser: DeployedEnv) -> dict:
    """The CDP address the browser gives, asked through the gateway's ``/svc/chromium/`` route with the browser's own
    control token, as agentenv asks a computer child."""
    app = next(child for child in browser.environment_card["children_environments"] if child["name"] == "chromium")
    path = protocol_v1.find_extension_method(app, BROWSER, "cdp")["endpoint"]
    async with httpx.AsyncClient(timeout=180) as http:
        answer = await http.post(f"{deployed.gateway_url}/svc/chromium{path}", json={"user": "ada"},
                                 headers=browser.control_headers)
    assert answer.status_code == 200, answer.text
    return answer.json()


async def _discovery(http_url: str) -> dict:
    async with httpx.AsyncClient(timeout=60) as http:
        answer = await http.get(f"{http_url.rstrip('/')}/json/version")
    assert answer.status_code == 200, answer.text
    return answer.json()


@pytest.mark.asyncio
async def test_chromium_as_a_multi_envs_browser_runs_on_its_own_sandbox_reached_through_the_gateway(
        browser_computer, shop, caplog):
    caplog.set_level(logging.WARNING, logger="agent_env.providers.env_providers.env_computer_provider")
    with namespace_routing():
        office = _office(shop, [Env.get(browser_computer.id)])
        step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                                 child_envs={"chromium": CHROMIUM})
        assert step.preflight() == []
        context = TaskStepContext()
        try:
            [deployed] = (await step.execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            stated = {child.environment_name for child in protocol_v1.children(deployed.environment_card)}
            driven = await _drive(deployed)
            address = await _cdp(deployed, browser)
            discovered = await _discovery(address["http_url"])
            seen = await _cdp_agent_loads(address["ws_url"], [SITE_URL])

            subprocess.run(["docker", "kill", *_containers(browser.sandbox_id)], capture_output=True)
            await ResetEnvTaskStep(id="reset", version=None, env_id=office.id, fail_task_on_error=True).execute(context)
            after_kill = await _answers_again(deployed)
            after_kill_tabs = (await _drive(deployed))["tabs"]

            subprocess.run(["docker", "restart", _gateway(deployed)], capture_output=True, check=True)
            through_restart = await _answers_again(deployed)
            await ResetEnvTaskStep(id="reset-2", version=None, env_id=office.id,
                                   fail_task_on_error=True).execute(context)
            after_restart = await _answers_again(deployed)
            up = _containers(browser.sandbox_id)
        finally:
            report = await teardown_run(context)

    assert not report.failed and up and _containers(browser.sandbox_id) == []
    # chromium is a computer child on a sandbox of its own, and no playwright-mcp runs.
    assert set(deployed.child_envs) == {"chromium"}
    assert deployed.sandbox_ids["local"] == {"chromium": browser.sandbox_id}
    assert "chromium" in stated and PLAYWRIGHT_MCP_ENVIRONMENT_NAME not in stated, stated
    computer = deployed.child_envs["chromium"]["metadata"]["computer"]
    assert computer["sites"]["names"]["routes"] == {"shop": ["shop.test"]}
    [start_tab] = computer["start_tabs"]["chromium"]
    assert (start_tab["url"], start_tab["site"], start_tab["outcome"]) == (SITE_URL, "shop", "loaded"), start_tab
    # An agent drives it through the gateway, every tool under chromium_.
    assert {"chromium_computer_screenshot", "chromium_computer_navigate"} <= driven["tools"], driven
    assert not any(name.startswith("computer_") for name in driven["tools"]), driven
    assert driven["shot"] and driven["opened"] and SITE_URL in driven["tabs"], driven
    # Its CDP address names the gateway, its discovery answer too, and an agent there reaches the site by name.
    token = address["http_url"].split("/cdp/")[1].strip("/")
    assert address["http_url"] == f"{deployed.gateway_url}/browser/chromium/cdp/{token}/", address
    assert browser.environment_url.rstrip("/") not in json.dumps([address, discovered]), (address, discovered)
    assert discovered["webSocketDebuggerUrl"].startswith(
        f"ws{deployed.gateway_url.removeprefix('http')}/browser/chromium/cdp/"), discovered
    assert seen[SITE_URL]["error"] is None and seen[SITE_URL].get("ready") == "complete", seen
    # Through a kill and a gateway restart, a reset brings it back.
    assert after_kill and SITE_URL in after_kill_tabs, after_kill_tabs
    assert after_restart, through_restart
    # Local Docker can't hold its network to the gateway, so it says it allows every host.
    assert "can't hold a network to the hosts" in caplog.text, caplog.text


@pytest.mark.asyncio
async def test_a_multi_env_naming_no_browser_deploys_none(browser_computer, shop):
    with namespace_routing():
        office = _office(shop, [])
        context = TaskStepContext()
        try:
            [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                  ttl_seconds=900).execute(context)).deployed_envs
            stated = {child.environment_name for child in protocol_v1.children(deployed.environment_card)}
        finally:
            report = await teardown_run(context)

    assert not report.failed and not deployed.child_envs
    assert "shop" in stated and PLAYWRIGHT_MCP_ENVIRONMENT_NAME not in stated and "chromium" not in stated, stated
