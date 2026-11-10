"""agentenv's chromium, deployed alone as a browser computer, behind the gateway: the gateway's image built from its
folder and run with the computer as its /svc child ``web``. Playwright and Puppeteer reach the browser through the
gateway's ``/browser/web/cdp/<token>/`` route by the ``http_url`` and the ``ws_url`` the browser's ``cdp`` answer
gives through the gateway, an address the browser didn't mint is refused, and each connection is a trajectory event
that, like the gateway's log, never shows its token. A gateway restart closes a live connection with a code, and a new
one then works; a data/reset closes it with the relaunch code and ends its address, and one minted after works; so does the browser's app server or the computer's driver dying behind the
gateway, and a browser whose app server was down when the gateway first looked is found once it's back. Needs Docker,
and Node 20 or later with npm for the CDP clients."""

import asyncio
import json
import platform
import socket
import subprocess
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.website_browser import BROWSER
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

import agent_env
from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT
from agent_env.providers.sandbox_providers.host_images import host_platform
from agent_env.providers.sandbox_providers.local_docker.provider import host_url_for
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from tst.integration.computer.browser_computer_test import (DATA_PAGE, SERVE_SITE, SERVE_START_S, SITE, _clients,
                                                            _kill, _ready)
from tst.integration.computer.browser_computer_test import _cdp as _driver_cdp
from tst.util.published_image import pull_published_else_build

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

GATEWAY = Path(agent_env.__file__).parent / "env" / "gateway"
EXTRA_HOSTS = ["--add-host", "host.docker.internal:host-gateway"] if platform.system() == "Linux" else []
#: What the browser's relay closes its connections with when it relaunches.
RELAUNCHED = 1012
#: The supervisord program the browser app's own server runs as.
APP_SERVER = "app-chromium"


@pytest.fixture(scope="module")
def gateway_image():
    tag = f"agentenv-test-gateway:{uuid.uuid4().hex[:8]}"
    pull_published_else_build("gateway", GATEWAY / "Dockerfile", GATEWAY, tag, platform=host_platform())
    yield tag
    subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)


@pytest.mark.asyncio
async def test_playwright_and_puppeteer_reach_the_browser_through_the_gateway_and_no_trace_shows_a_token(
        browser_computer, cdp_clients, gateway_image):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [SITE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_SITE, "900"], timeout_s=960))
            await asyncio.sleep(SERVE_START_S)
            async with _gateway(gateway_image, deployed) as (gateway, container):
                address = await _cdp(gateway, deployed, "ada")
                together = await _clients(cdp_clients, "together", address)
                refused = await _clients(cdp_clients, "refused", address)
                events = await _connections(gateway, at_least=4)
                logs = subprocess.run(["docker", "logs", container], capture_output=True, text=True)
            serving.cancel()
        finally:
            report = await teardown_run(context)

    token = address["http_url"].split("/cdp/")[1].strip("/")
    assert address == {"http_url": f"{gateway}/browser/web/cdp/{token}/",
                       "ws_url": f"ws{gateway.removeprefix('http')}/browser/web/cdp/{token}/devtools/browser"}
    for client in ("playwright_http", "puppeteer_ws", "playwright_ws"):
        assert together[client].get("ok", {}).get("title") == "Home", together
    assert "error" in refused["playwright"] and "error" in refused["puppeteer"], refused
    opened = [event for event in events if event["opened_at"]]
    assert len(opened) == 3 and {event["closed_by"] for event in opened} == {"client"}, events
    assert {(event["closed_by"], event["status"]) for event in events if not event["opened_at"]} == {
        ("browser", 403)}, events
    assert token not in json.dumps(events), events
    assert "/cdp/<redacted>/" in logs.stderr + logs.stdout and token not in logs.stderr + logs.stdout
    assert not report.failed


@pytest.mark.asyncio
async def test_a_gateway_restart_or_a_data_reset_closes_a_live_connection_with_a_code_and_a_new_one_works(
        browser_computer, gateway_image):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [DATA_PAGE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            async with _gateway(gateway_image, deployed) as (gateway, container):
                address = await _cdp(gateway, deployed, "ada")
                held, first = await _version(address["ws_url"])
                subprocess.run(["docker", "restart", "-t", "5", container], check=True, capture_output=True)
                at_restart = await _closed(held)
                await _until_up(gateway)
                held, second = await _version(address["ws_url"])
                await protocol_v1.reset_data(f"{deployed.environment_url}/svc/chromium",
                                             headers=deployed.control_headers)
                at_reset = await _closed(held)
                try:
                    await _version(address["ws_url"])
                    ended = None
                except InvalidStatus as e:
                    ended = e.response.status_code
                held, third = await _version((await _cdp(gateway, deployed, "ada"))["ws_url"])
                await held.close()
        finally:
            report = await teardown_run(context)

    assert first.startswith("Chrome/") and first == second == third
    assert at_restart is not None and at_reset == RELAUNCHED and ended == 403
    assert not report.failed


@pytest.mark.asyncio
async def test_behind_the_gateway_a_browser_down_at_first_is_found_once_up_and_a_dying_server_or_driver_closes_a_connection(
        browser_computer, gateway_image):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [DATA_PAGE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            token = (await _driver_cdp(deployed, "ada"))["http_url"].split("/cdp/")[1].strip("/")
            await inside.run(deployed, ["supervisorctl", "stop", APP_SERVER], timeout_s=60)
            async with _gateway(gateway_image, deployed) as (gateway, _):
                discovery = f"{gateway}/browser/web/cdp/{token}/json/version"
                async with httpx.AsyncClient(timeout=60) as http:
                    while_down = (await http.get(discovery)).status_code
                    await inside.run(deployed, ["supervisorctl", "start", APP_SERVER], timeout_s=120)
                    await _ready(deployed)
                    ws_url = (await http.get(discovery)).json()["webSocketDebuggerUrl"]
                held, first = await _version(ws_url)
                await _kill(deployed, "[s]erve_app")
                at_server_death = await _closed(held)
                await _ready(deployed)
                held, after_server = await _version(ws_url)
                await _kill(deployed, "[c]omputer.serve driver.py:PageDriver")
                at_driver_death = await _closed(held)
                await _ready(deployed)
                held, after_driver = await _version(ws_url)
                await held.close()
                events = await _connections(gateway, at_least=3)
        finally:
            report = await teardown_run(context)

    assert while_down == 404
    assert ws_url == f"ws{gateway.removeprefix('http')}/browser/web/cdp/{token}/devtools/browser"
    assert at_server_death is not None and at_driver_death is not None
    assert first.startswith("Chrome/") and first == after_server == after_driver
    assert [event["closed_by"] for event in events if event["opened_at"]] == ["browser", "browser", "client"], events
    assert not report.failed


@asynccontextmanager
async def _gateway(image: str, deployed):
    """The gateway ``image`` run with ``deployed`` as its /svc child ``web``, at a fixed port a restart keeps; yields
    its address and its container."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    name = f"agentenv-test-gateway-{uuid.uuid4().hex[:8]}"
    browser = host_url_for(deployed.environment_url.rstrip("/"), "local")
    subprocess.run(["docker", "run", "-d", "--name", name, *EXTRA_HOSTS, "-e", f"REST_PROXY_URLS=web={browser}",
                    "-p", f"127.0.0.1:{port}:{AGENT_ENV_GATEWAY_MCP_PORT}", image], check=True, capture_output=True)
    try:
        url = f"http://127.0.0.1:{port}"
        await _until_up(url)
        yield url, name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


async def _until_up(url: str, timeout_s: float = 90) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    async with httpx.AsyncClient(timeout=5) as http:
        while asyncio.get_running_loop().time() < deadline:
            try:
                if (await http.get(f"{url}/.well-known/agent-env.json")).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(1)
    raise TimeoutError(f"the gateway at {url} never came up")


async def _cdp(gateway: str, deployed, user: str) -> dict:
    """``user``'s CDP address, asked of the browser through the gateway's /svc route, with the computer's control
    token, as agentenv asks it when the browser is a gateway's child."""
    app = next(child for child in deployed.environment_card["children_environments"] if child["name"] == "chromium")
    path = protocol_v1.find_extension_method(app, BROWSER, "cdp")["endpoint"]
    async with httpx.AsyncClient(timeout=180) as http:
        answer = await http.post(f"{gateway}/svc/web{path}", json={"user": user}, headers=deployed.control_headers)
    assert answer.status_code == 200, answer.text
    return answer.json()


async def _connections(gateway: str, at_least: int, timeout_s: float = 30) -> list[dict]:
    """The gateway's browser_connection events, once there are ``at_least``: each is written as its connection ends."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    async with httpx.AsyncClient(timeout=10) as http:
        while True:
            lines = (await http.get(f"{gateway}/trajectory")).text.splitlines()
            events = [event for event in map(json.loads, lines) if event.get("event_type") == "browser_connection"]
            if len(events) >= at_least or asyncio.get_running_loop().time() > deadline:
                return events
            await asyncio.sleep(1)


async def _version(ws_url: str):
    """A connection to ``ws_url`` and the browser's product, which it answered ``Browser.getVersion`` with."""
    held = await connect(ws_url, max_size=None, open_timeout=30)
    await held.send(json.dumps({"id": 1, "method": "Browser.getVersion"}))
    reply = json.loads(await asyncio.wait_for(held.recv(), 30))
    return held, reply["result"]["product"]


async def _closed(held) -> int | None:
    """The code a connection was closed with, once it is."""
    await asyncio.wait_for(held.wait_closed(), 60)
    return held.close_code
