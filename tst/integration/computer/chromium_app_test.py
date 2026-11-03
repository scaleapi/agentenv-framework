"""agentenv's chromium app on the ubuntu-24.04 base, as the chromium test bundle's desktops list it: gate 3 makes this
machine's arch ready with it; deployed on two start_urls, it shows one Chromium window, on the first, with no first-run,
sign-in or permission window beside it, and the second a tab away, which an agent reaches with Control+Tab; its managed
policies are where Chrome for Testing reads them; and a reset of its data clears its profile and opens it again. As a
website browser it runs Chrome itself with agentenv's flags on its debugging pipe, keeps its shared memory in files on
Docker's 64 MiB /dev/shm, answers location prompts and leaves notification ones to ask as the deploy says, and its menu
launcher, run while it runs, opens another window of it with no second copy, its cache kept and it still ready; and the
desktop that gives it French defaults deploys with them, a deploy's time zone over them, each recorded. Needs Docker and
the local registry on localhost:5000."""

import asyncio
import json
import platform
import re

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agentenv_protocol import client as protocol_v1

from agent_env.env import Env
from agentenv_protocol.types import WELL_KNOWN_PATH
from tst.integration.computer.website_browser_test import _app, _browser_process, _keys, _ready
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
ALPHA = "data:text/html,<title>Alpha page</title><h1>Alpha</h1>"
BRAVO = "data:text/html,<title>Bravo page</title><h1>Bravo</h1>"
LOOPBACK = "http://127.0.0.1:8766/"
# What chromium's menu launcher runs.
LAUNCHER = ("/opt/agentenv/apps/chromium/venv/bin/python -m agentenv_protocol.computer.launch "
            "/opt/agentenv/apps/chromium")
# Serves, on loopback for the seconds it's given, a page whose title is its location and notification permissions.
SERVE_PERMISSIONS = """import http.server, sys, threading, time
class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = (b"<title>asking</title><script>Promise.all(['geolocation', 'notifications'].map(name => "
                b"navigator.permissions.query({name}))).then(states => "
                b"document.title = states.map(s => s.state).join(' '))</script>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 8766), Page)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[1]))
"""
# Each window the window manager manages, its class and its title.
WINDOWS = ('for w in $(xprop -root _NET_CLIENT_LIST | sed "s/.*# //; s/,//g"); do '
           'xprop -id "$w" WM_CLASS _NET_WM_NAME; done')


def test_gate_3_makes_this_machines_arch_ready_with_chromium(chromium_desktop):
    results = chromium_desktop.metadata["composed"]["container"]

    assert results[ARCH]["status"] == "ready" and results[ARCH]["probe"]["passed"], results[ARCH]
    assert [app.environment_name for app in chromium_desktop.composed_apps()] == ["chromium"]


@pytest.mark.asyncio
async def test_a_deploy_opens_its_start_urls_unprompted_and_a_reset_clears_its_profile(chromium_desktop):
    with namespace_routing():
        deployed = await Env.get(chromium_desktop.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [ALPHA, BRAVO]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            opened = await _chromium_windows(deployed, "Alpha page")
            keyed, tabbed = await _tab_to(deployed, "Bravo page")
            policies = await inside.run(
                deployed, ["cat", "/etc/opt/chrome_for_testing/policies/managed/agentenv.json"], timeout_s=30)
            await inside.run(deployed, ["sh", "-c", "touch ~/.config/chromium/marker"], timeout_s=30)
            await protocol_v1.reset_data(f"{deployed.environment_url}/svc/chromium", headers=deployed.control_headers)
            marker = await inside.run(deployed, ["sh", "-c", "test -e ~/.config/chromium/marker"], timeout_s=30)
            reopened = await _chromium_windows(deployed, "Alpha page")
        finally:
            report = await teardown_run(context)

    assert opened == ["Alpha page - Google Chrome for Testing"]
    assert not keyed.isError and tabbed == ["Bravo page - Google Chrome for Testing"]
    assert {"BrowserSignin": 0, "CommandLineFlagSecurityWarningsEnabled": False}.items() <= json.loads(
        policies["stdout"]).items()
    assert marker["exit_code"] != 0 and reopened == ["Alpha page - Google Chrome for Testing"]
    assert not report.failed


@pytest.mark.asyncio
async def test_it_runs_chrome_itself_on_its_pipe_answers_prompts_as_the_deploy_says_and_a_launch_opens_a_window(
        chromium_desktop):
    settings = {"start_urls": [LOOPBACK], "notifications": "ask"}
    with namespace_routing():
        deployed = await Env.get(chromium_desktop.id).deploy(env_params={"app_settings": {"chromium": settings}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            browser = await _browser_process(deployed)
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_PERMISSIONS, "60"],
                                                       timeout_s=90))
            await asyncio.sleep(3)
            await _keys(deployed, ["F5"])
            permissions = await _chromium_windows(deployed, "denied")
            await serving
            await inside.run(deployed, ["sh", "-c", "mkdir -p ~/.cache/chromium && touch ~/.cache/chromium/kept"],
                             timeout_s=30)
            launched = await inside.run(deployed, ["sh", "-c", LAUNCHER], timeout_s=60)
            windows = await _chromium_windows(deployed, "denied", count=2)
            after = await _browser_process(deployed)
            kept = await inside.run(deployed, ["sh", "-c", "test -e ~/.cache/chromium/kept"], timeout_s=30)
            readiness = await _ready(deployed, await _app(deployed, WELL_KNOWN_PATH, "chromium"), "chromium")
        finally:
            report = await teardown_run(context)

    argv = browser["argv"]
    assert argv[0].endswith("chromium/chrome/chrome") and argv[1:3] == ["--no-sandbox", "--class=Chromium"]
    assert argv[3].endswith("/.config/chromium") and argv[4].endswith("/.cache/chromium")
    assert argv[5:10] == ["--remote-debugging-pipe", "--no-first-run", "--no-default-browser-check", "--lang=en-US",
                          "--disable-dev-shm-usage"] and argv[10:] == [LOOPBACK]
    assert browser["listening"] == 0
    assert permissions == ["denied prompt - Google Chrome for Testing"]
    assert launched["exit_code"] == 0 and len(windows) == 2 and after["pid"] == browser["pid"]
    assert kept["exit_code"] == 0 and readiness["ready"] is True, readiness
    recorded = deployed.metadata["computer"]["app_settings"]["chromium"]
    assert (recorded["geolocation"], recorded["notifications"], recorded["start_urls"]) == ("deny", "ask", [LOOPBACK])
    assert not report.failed


@pytest.mark.asyncio
async def test_a_desktop_that_gives_it_defaults_deploys_with_them_and_a_deploys_own_over_them(chromium_desktop):
    with namespace_routing():
        french = Env.get(chromium_desktop.id.rsplit("/", 1)[0] + "/desktop-fr")
        deployed = await french.deploy(env_params={"app_settings": {"chromium": {"time_zone": "Asia/Tokyo"}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            browser = await _browser_process(deployed)
        finally:
            report = await teardown_run(context)

    assert "--lang=fr-FR" in browser["argv"]
    recorded = deployed.metadata["computer"]["app_settings"]["chromium"]
    assert (recorded["locale"], recorded["time_zone"]) == ("fr-FR", "Asia/Tokyo")
    assert deployed.metadata["computer"]["env_params"] == {"app_settings": {"chromium": {"time_zone": "Asia/Tokyo"}}}
    assert not report.failed


async def _tab_to(deployed, title: str):
    """Control+Tab pressed as an agent presses it, again while the window still shows the tab before, which a press
    can reach before the next tab opens; the last press's result, and the window titles once one starts with
    ``title``."""
    async with streamablehttp_client(f"{deployed.environment_url}/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for attempt in range(3):
                keyed = await session.call_tool("computer_key", {"keys": ["Control", "Tab"]})
                try:
                    return keyed, await _chromium_windows(deployed, title, seconds=20)
                except TimeoutError:
                    if attempt == 2:
                        raise


async def _chromium_windows(deployed, title: str, seconds: int = 60, count: int = 1) -> list[str]:
    """The titles of the computer's Chromium windows once there are ``count`` and one starts with ``title``."""
    for _ in range(seconds):
        listed = (await inside.run(deployed, ["sh", "-c", WINDOWS], timeout_s=30))["stdout"]
        classes = re.findall(r'WM_CLASS\(STRING\) = .*"([^"]*)"\n_NET_WM_NAME\(UTF8_STRING\) = "(.*)"', listed)
        titles = [name for cls, name in classes if cls == "Chromium"]
        if any(name.startswith(title) for name in titles) and len(titles) >= count:
            return titles
        await asyncio.sleep(1)
    raise TimeoutError(f"no Chromium window titled {title!r}: {listed}")
