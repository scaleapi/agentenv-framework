"""WebsiteBrowserEnvironment in a computer, as the website-browser test bundle's desktop composes its fixture browser on
the ubuntu-24.04 base: gate 3 makes this machine's arch ready with it, which its debugging-address check is part of;
deployed, Chromium runs with agentenv's flags after the fixture's command, on the deploy's locale and time zone, through
no proxy, with agentenv's policies where its build reads them, and opens a page on loopback; its card declares
browser/v1; and a reset relaunches it on its profile template, at a debugging address its ready check finds. Needs
Docker and the local registry on localhost:5000."""

import asyncio
import json
import platform
import shlex

import httpx
import pytest

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import apps
from agentenv_protocol.types import WELL_KNOWN_PATH
from agentenv_protocol.website_browser import BROWSER

from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
LOOPBACK = "http://127.0.0.1:8765/"
SERVE_LOOPBACK = ("mkdir -p /tmp/loopback && printf '<title>Loopback page</title>' > /tmp/loopback/index.html && "
                  "cd /tmp/loopback && setsid nohup python3 -m http.server 8765 --bind 127.0.0.1 "
                  ">/dev/null 2>&1 < /dev/null &")
# Opens its URL in a new tab through Chromium's HTTP debugging endpoints, then prints that tab's title once it has one.
OPEN_TAB = """import json, sys, time, urllib.request as u
port, url = sys.argv[1:]
direct = u.build_opener(u.ProxyHandler({}))
for _ in range(30):
    try:
        direct.open(url, timeout=5)
        break
    except OSError:
        time.sleep(1)
direct.open(u.Request(f"http://127.0.0.1:{port}/json/new?{url}", method="PUT"), timeout=10)
for _ in range(30):
    titles = [t["title"] for t in json.load(direct.open(f"http://127.0.0.1:{port}/json/list", timeout=10))
              if t.get("url") == url and t["title"] != url]
    if titles:
        print(titles[0])
        break
    time.sleep(1)
"""


def test_gate_3_makes_this_machines_arch_ready_with_the_fixture_browser(website_browser_desktop):
    results = website_browser_desktop.metadata["composed"]["container"]

    assert results[ARCH]["status"] == "ready" and results[ARCH]["probe"]["passed"], results[ARCH]
    assert [app.environment_name for app in website_browser_desktop.composed_apps()] == ["browser"]


@pytest.mark.asyncio
async def test_chromium_runs_with_agentenvs_flags_and_a_reset_relaunches_it_on_its_template(website_browser_desktop):
    with namespace_routing():
        deployed = await Env.get(website_browser_desktop.id).deploy(
            env_params={"app_settings": {"browser": {"locale": "fr-FR", "time_zone": "Europe/Paris"}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            argv, environ = await _browser_process(deployed)
            port = await _devtools_port(deployed)
            policies = await inside.run(
                deployed, ["cat", "/etc/opt/chrome_for_testing/policies/managed/agentenv.json"], timeout_s=30)
            await inside.run(deployed, ["sh", "-c", SERVE_LOOPBACK], timeout_s=30)
            title = await inside.run(deployed, ["python3", "-c", OPEN_TAB, port, LOOPBACK], timeout_s=90)
            card = await _app(deployed, WELL_KNOWN_PATH)
            await protocol_v1.reset_data(f"{deployed.environment_url}/svc/browser", headers=deployed.control_headers)
            relaunched = await _devtools_port(deployed, other_than=port)
            readiness = await _ready(deployed, card)
            template = await inside.run(deployed, ["sh", "-c", "cat ~/.config/chromium/agentenv-template.txt"],
                                        timeout_s=30)
        finally:
            report = await teardown_run(context)

    assert argv[0].endswith("chromium/chrome/chrome") and argv[1:4] == ["--no-sandbox", "--disable-dev-shm-usage",
                                                                       "--class=Chromium"]
    assert argv[4].startswith("--user-data-dir=") and argv[4].endswith("/.config/chromium")
    assert argv[5:9] == ["--remote-debugging-port=0", "--no-first-run", "--no-default-browser-check", "--lang=fr-FR"]
    assert argv[9:] == ["about:blank"] and "TZ=Europe/Paris" in environ
    assert {"BrowserSignin": 0, "DownloadDirectory": "${user_home}/Downloads"}.items() <= json.loads(
        policies["stdout"]).items()
    assert title["stdout"].strip() == "Loopback page", title
    assert protocol_v1.extension_params(card, BROWSER) == {"chromium": "156.0.8078.4", "clock": "real"}
    assert relaunched != port and readiness["ready"] is True, readiness
    assert template["stdout"] == "from the profile template\n"
    assert not report.failed


async def _browser_process(deployed) -> tuple[list[str], list[str]]:
    """The browser process's argv and environment, once it runs: the oldest process with agentenv's debugging flag."""
    find = "pid=$(pgrep -o -f -- --remote-debugging-port=0) && tr '\\0' '\\n' < /proc/$pid/{}"
    for _ in range(60):
        argv = await inside.run(deployed, ["sh", "-c", find.format("cmdline")], timeout_s=30)
        if argv["exit_code"] == 0 and argv["stdout"]:
            environ = await inside.run(deployed, ["sh", "-c", find.format("environ")], timeout_s=30)
            return argv["stdout"].splitlines(), environ["stdout"].splitlines()
        await asyncio.sleep(1)
    raise TimeoutError("no Chromium process with --remote-debugging-port=0")


async def _devtools_port(deployed, other_than: str | None = None) -> str:
    """The port in the profile's DevToolsActivePort, once Chromium has written one other than ``other_than``."""
    for _ in range(60):
        read = await inside.run(deployed, ["sh", "-c", "head -1 ~/.config/chromium/DevToolsActivePort"], timeout_s=30)
        port = read["stdout"].strip()
        if read["exit_code"] == 0 and port and port != other_than:
            return port
        await asyncio.sleep(1)
    raise TimeoutError(f"no DevToolsActivePort other than {shlex.quote(str(other_than))}")


async def _app(deployed, path: str) -> dict:
    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.get(f"{deployed.environment_url}/svc/browser{path}", headers=deployed.control_headers)
    response.raise_for_status()
    return response.json()


async def _ready(deployed, card: dict) -> dict:
    """The app's readiness once it says it's ready, or as it last said within a minute."""
    for _ in range(60):
        readiness = await _app(deployed, apps.endpoint(card, "ready"))
        if readiness["ready"]:
            return readiness
        await asyncio.sleep(1)
    return readiness
