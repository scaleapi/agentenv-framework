"""WebsiteBrowserEnvironment in a computer, as the website-browser test bundle's desktop composes its fixture browser on
the ubuntu-24.04 base: gate 3 makes this machine's arch ready with it, which its check over the debugging pipe is part
of; deployed, Chromium runs with agentenv's flags after the fixture's command, on the deploy's locale, through no proxy,
with agentenv's policies where its build reads them and with no debugging port anything in the computer could find; its
first start tab sees the deploy's time zone, and a page on loopback loads; its card declares browser/v1; a data/reset
relaunches it on its profile template, on a new pipe its ready check answers over; and reset_env, a reboot, brings the
template back too. Its computer's /dev/shm is Docker's
64 MiB, so it keeps its shared memory in files. In a multi env beside a site that offers website/v1's session,
browser_sign_in signs it in, and again after reset_env starts it afresh. Beside a live site, github.com, which the
computer reaches over its own network and leaves out of its site table, the browser loads the real site signed in to it
as well as the synthetic one, and the run is marked live. Needs Docker, the local registry on localhost:5000 and, for
the live site, github.com."""

import asyncio
import json
import platform
import re
import time
import uuid
from typing import Optional

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import apps
from agentenv_protocol.types import WELL_KNOWN_PATH
from agentenv_protocol.website_browser import BROWSER

from agent_env.config import get_config, website_browser_env_id
from agent_env.env import Env
from agent_env.env.env import DeployedEnv
from agent_env.env.envs.computer.utils import inside
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.playwright_mcp import PLAYWRIGHT_MCP_ENVIRONMENT_NAME
from agent_env.env.envs.website import WebsiteEnv
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.browser.base import find_browser
from agent_env.task_step.task_steps.browser.sign_in import BrowserSignInTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from tst.integration.computer.conftest import CONTAINER
from tst.integration.computer.multi_env_computers_test import _image
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
TIME_ZONE_PAGE = ("data:text/html,<title>loading</title>"
                  "<script>document.title=Intl.DateTimeFormat().resolvedOptions().timeZone</script>")
LOOPBACK = "http://127.0.0.1:8765/"
# Serves a page on loopback for the seconds it's given.
SERVE_LOOPBACK = """import http.server, sys, threading, time
class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<title>Loopback page</title>"
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 8765), Page)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[1]))
"""
# The browser's pid and argv, and how many TCP sockets anything in its session listens on: its oldest process with
# agentenv's pipe flag, whose title Chromium rewrites as one line.
BROWSER_PROCESS = """import glob, json, os, subprocess
pid = int(subprocess.run(["pgrep", "-o", "-f", "--", "[-]-remote-debugging-pipe"], capture_output=True, text=True,
                         check=True).stdout)
listening = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    for line in open(table).read().splitlines()[1:]:
        fields = line.split()
        if fields[3] == "0A":
            listening.add(fields[9])
sockets = set()
for proc in glob.glob("/proc/[0-9]*"):
    try:
        if os.getsid(int(proc.rsplit("/", 1)[1])) != os.getsid(pid):
            continue
        for fd in os.listdir(f"{proc}/fd"):
            link = os.readlink(f"{proc}/fd/{fd}")
            if link.startswith("socket:["):
                sockets.add(link[8:-1])
    except OSError:
        continue
argv = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\\0", b" ").decode().split()
print(json.dumps({"pid": pid, "argv": argv, "listening": len(listening & sockets)}))
"""
#: How long apart the key chords a test presses into Chromium are.
KEY_GAP_S = 2.0
# Each window the window manager manages, its class and its title.
WINDOWS = ('for w in $(xprop -root _NET_CLIENT_LIST | sed "s/.*# //; s/,//g"); do '
           'xprop -id "$w" WM_CLASS _NET_WM_NAME; done')


def test_gate_3_makes_this_machines_arch_ready_with_the_fixture_browser(website_browser_desktop):
    results = website_browser_desktop.metadata["composed"]["container"]

    assert results[ARCH]["status"] == "ready" and results[ARCH]["probe"]["passed"], results[ARCH]
    assert [app.environment_name for app in website_browser_desktop.composed_apps()] == ["browser"]


@pytest.mark.asyncio
async def test_chromium_runs_with_agentenvs_flags_on_its_pipe_and_a_reset_relaunches_it_on_its_template(
        website_browser_desktop):
    settings = {"locale": "fr-FR", "time_zone": "Europe/Paris", "start_urls": [TIME_ZONE_PAGE, LOOPBACK]}
    with namespace_routing():
        deployed = await Env.get(website_browser_desktop.id).deploy(env_params={"app_settings": {"browser": settings}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            zone = await _window_titled(deployed, "Europe/Paris")
            browser = await _browser_process(deployed)
            port_file = await inside.run(deployed, ["sh", "-c", "test -e ~/.config/chromium/DevToolsActivePort"],
                                         timeout_s=30)
            policies = await inside.run(
                deployed, ["cat", "/etc/opt/chrome_for_testing/policies/managed/agentenv.json"], timeout_s=30)
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_LOOPBACK, "60"],
                                                       timeout_s=90))
            await asyncio.sleep(3)
            await _keys(deployed, ["Control", "Tab"], ["F5"])
            loopback = await _window_titled(deployed, "Loopback page")
            await serving
            card = await _app(deployed, WELL_KNOWN_PATH)
            readiness = await _ready(deployed, card)
            await protocol_v1.reset_data(f"{deployed.environment_url}/svc/browser", headers=deployed.control_headers)
            relaunched = await _browser_process(deployed, other_than=browser["pid"])
            relaunched_readiness = await _ready(deployed, card)
            template = await inside.run(deployed, ["sh", "-c", "cat ~/.config/chromium/agentenv-template.txt"],
                                        timeout_s=30)
            context = await ResetEnvTaskStep(id="reset", version=None, env_id=website_browser_desktop.id,
                                             fail_task_on_error=True).execute(context)
            rebooted_template = await inside.run(deployed, ["sh", "-c", "cat ~/.config/chromium/agentenv-template.txt"],
                                                 timeout_s=30)
        finally:
            report = await teardown_run(context)

    argv = browser["argv"]
    assert argv[0].endswith("chromium/chrome/chrome") and argv[1:3] == ["--no-sandbox", "--class=Chromium"]
    assert argv[3].startswith("--user-data-dir=") and argv[3].endswith("/.config/chromium")
    assert argv[4].startswith("--disk-cache-dir=") and argv[4].endswith("/.cache/chromium")
    assert argv[5:10] == ["--remote-debugging-pipe", "--no-first-run", "--no-default-browser-check", "--lang=fr-FR",
                          "--disable-dev-shm-usage"]
    assert argv[10:] == ["about:blank"]
    assert browser["listening"] == 0 and port_file["exit_code"] != 0
    assert zone.startswith("Europe/Paris") and loopback.startswith("Loopback page")
    assert {"BrowserSignin": 0, "DownloadDirectory": "${user_home}/Downloads"}.items() <= json.loads(
        policies["stdout"]).items()
    params = protocol_v1.extension_params(card, BROWSER)
    assert (params["chromium"], params["clock"], sorted(params["methods"])) == (
        "156.0.8078.4", "real", ["calls/begin", "calls/end", "cdp", "contexts/grant", "contexts/lease",
                                 "contexts/list", "contexts/release", "cookies/delete", "cookies/get", "cookies/set",
                                 "dialog/answer", "downloads/get", "downloads/list", "files/choose", "history",
                                 "local_storage/clear", "local_storage/get", "local_storage/set", "log", "navigate",
                                 "network", "page/capture", "page/input", "page/state", "page/viewport", "record", "sites/set",
                                 "tabs/activate", "tabs/close", "tabs/list", "tabs/open"])
    assert readiness["ready"] is True, readiness
    assert relaunched["pid"] != browser["pid"] and relaunched["listening"] == 0
    assert relaunched_readiness["ready"] is True, relaunched_readiness
    assert template["stdout"] == rebooted_template["stdout"] == "from the profile template\n"
    assert not report.failed


async def _window_titled(deployed, title: str) -> str:
    """The title of the computer's Chromium window once one starts with ``title``."""
    for _ in range(60):
        listed = (await inside.run(deployed, ["sh", "-c", WINDOWS], timeout_s=30))["stdout"]
        classes = re.findall(r'WM_CLASS\(STRING\) = .*"([^"]*)"\n_NET_WM_NAME\(UTF8_STRING\) = "(.*)"', listed)
        if found := [name for cls, name in classes if cls == "Chromium" and name.startswith(title)]:
            return found[0]
        await asyncio.sleep(1)
    raise TimeoutError(f"no Chromium window titled {title!r}: {listed}")


async def _keys(deployed, *chords: list[str]) -> None:
    """Each chord pressed through the computer's own tools, as an agent presses it, ``KEY_GAP_S`` apart, after a click
    into the middle of Chromium's window, which a window opened with no input may need before it has the keyboard."""
    x, y = await _window_center(deployed)
    async with streamablehttp_client(f"{deployed.environment_url}/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            await session.call_tool("computer_screenshot", {})
            clicked = await session.call_tool("computer_click", {"x": x, "y": y})
            assert not clicked.isError, clicked
            for keys in chords:
                pressed = await session.call_tool("computer_key", {"keys": keys})
                assert not pressed.isError, pressed
                await asyncio.sleep(KEY_GAP_S)


async def _window_center(deployed) -> tuple[int, int]:
    """The middle of Chromium's largest window, in screen pixels."""
    tree = (await inside.run(deployed, ["xwininfo", "-root", "-tree"], timeout_s=30))["stdout"]
    windows = [(int(w) * int(h), int(x) + int(w) // 2, int(y) + int(h) // 2) for w, h, x, y in re.findall(
        r'\("[^"]*" "Chromium"\)\s+(\d+)x(\d+)\+-?\d+\+-?\d+\s+\+(-?\d+)\+(-?\d+)', tree)]
    assert windows, tree
    _, x, y = max(windows)
    return x, y


async def _browser_process(deployed, other_than: int | None = None) -> dict:
    """The browser process's pid, argv and listening sockets, once one runs other than ``other_than``."""
    for _ in range(60):
        found = await inside.run(deployed, ["python3", "-c", BROWSER_PROCESS], timeout_s=30)
        if found["exit_code"] == 0 and (process := json.loads(found["stdout"]))["pid"] != other_than:
            return process
        await asyncio.sleep(1)
    raise TimeoutError(f"no Chromium on --remote-debugging-pipe other than {other_than}: {found}")


async def _app(deployed, path: str, app: str = "browser") -> dict:
    async with httpx.AsyncClient(timeout=30) as http:
        response = await http.get(f"{deployed.environment_url}/svc/{app}{path}", headers=deployed.control_headers)
    response.raise_for_status()
    return response.json()


async def _ready(deployed, card: dict, app: str = "browser") -> dict:
    """The app's readiness once it says it's ready, or as it last said within a minute."""
    for _ in range(60):
        readiness = await _app(deployed, apps.endpoint(card, "ready"), app)
        if readiness["ready"]:
            return readiness
        await asyncio.sleep(1)
    return readiness


# Who a request carrying the cookie signs in, as the site says, asked inside the computer through its site proxy.
WHO_IS_SIGNED_IN = """import json, sys, urllib.request
request = urllib.request.Request("https://shop.test/api/session", headers={"Cookie": sys.argv[1]})
print(json.load(urllib.request.urlopen(request, timeout=20))["user"])
"""


@pytest.fixture(scope="module")
def session_site(website_browser_desktop):
    """The gateway's and service-db's envs, a small stand-in for the website browser's MCP server a gateway in front of
    websites runs, and a WebsiteEnvironment site named shop.test that offers session, under a local id as the
    desktop's."""
    uid = uuid.uuid4().hex[:8]
    with namespace_routing():
        _seed_bootstrap_envs()
        MCPServerEnv.put(id=website_browser_env_id(get_config()), environment_name=PLAYWRIGHT_MCP_ENVIRONMENT_NAME,
                         docker_image_artifact=_image("cardless_mcp", f"signin-browser-{uid}"))
        return WebsiteEnv.put(id=f"@local/~/signin-{uid}/shop", environment_name="shop", domains=["shop.test"],
                              server_docker_image_artifact=_image("session_website", f"signin-shop-{uid}"))


async def _signed_in(computer: DeployedEnv) -> Optional[str]:
    """Who the browser's cookies for shop.test sign in, as the site says inside the computer; None with none."""
    owner, _, card = find_browser(computer, None)
    cookies = (await protocol_v1.invoke_extension(owner.environment_url, card, BROWSER, {"urls": ["https://shop.test/"]},
                                                  30, method="cookies/get", headers=owner.control_headers))["cookies"]
    if not cookies:
        return None
    header = "; ".join(f"{cookie['name']}={cookie['value']}" for cookie in cookies)
    answer = await inside.run(computer, ["python3", "-c", WHO_IS_SIGNED_IN, header], timeout_s=40)
    return answer["stdout"].strip() or answer.get("stderr", "")[-300:]


@pytest.mark.asyncio
async def test_browser_sign_in_signs_a_multi_envs_browser_in_to_its_site_and_again_after_a_reset(
        website_browser_desktop, session_site):
    with namespace_routing():
        office = MultiEnv.put(id=f"{session_site.id.rsplit('/', 1)[0]}/office", mcp_server_envs=[],
                              website_envs=[session_site], computer_envs=[Env.get(website_browser_desktop.id)],
                              name="office")
        sign_in = BrowserSignInTaskStep(id="in", version=None, env_id=office.id, site=session_site.id, user="alex")
        context = TaskStepContext()
        try:
            [deployed] = (await DeployEnvTaskStep(
                id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                child_envs={"desktop": {**CONTAINER, "env_params": {"form": "container", "site_routing": True}}},
            ).execute(context)).deployed_envs
            computer = DeployedEnv.from_dict(deployed.child_envs["desktop"])
            before = await _signed_in(computer)
            await sign_in.execute(context)
            signed = await _signed_in(computer)
            await ResetEnvTaskStep(id="reset", version=None, env_id=office.id, fail_task_on_error=True).execute(context)
            after_reset = await _signed_in(computer)
            await sign_in.execute(context)
            again = await _signed_in(computer)
        finally:
            report = await teardown_run(context)

    assert not report.failed
    assert (before, signed, after_reset, again) == (None, "alex", None, "alex")
    assert context.metadata["browser_steps"]["in"]["names"] == ["shop.test"]


@pytest.fixture(scope="module")
def live_site(session_site):
    """A live website env at github.com, under session_site's local id: its server reads a public repository through
    api.github.com."""
    uid = uuid.uuid4().hex[:8]
    with namespace_routing():
        return WebsiteEnv.put(id=f"{session_site.id.rsplit('/', 1)[0]}/github", environment_name="github",
                              domains=["github.com"], live=True,
                              allow_hosts=["*.github.com", "*.githubassets.com", "*.githubusercontent.com"],
                              server_docker_image_artifact=_image("live_github_website", f"live-github-{uid}"))


async def _browser(computer: DeployedEnv, op: str, params: Optional[dict] = None) -> dict:
    owner, _, card = find_browser(computer, None)
    return await protocol_v1.invoke_extension(owner.environment_url, card, BROWSER, params or {}, 60, method=op,
                                              headers=owner.control_headers)


async def _loads(computer: DeployedEnv, url: str, title: str, timeout_s: float = 90) -> object:
    """True once the tab ``tabs/open`` opens at ``url`` shows ``title`` in its own, else the tabs it last listed."""
    await _browser(computer, "tabs/open", {"url": url})
    deadline, tabs = time.monotonic() + timeout_s, []
    while time.monotonic() < deadline:
        tabs = (await _browser(computer, "tabs/list"))["tabs"]
        if any(tab["url"].startswith(url) and title in (tab.get("title") or "") for tab in tabs):
            return True
        await asyncio.sleep(2)
    return tabs


@pytest.mark.asyncio
async def test_a_live_site_beside_a_synthetic_one_loads_at_its_real_names_signed_in_and_marks_the_run_live(
        website_browser_desktop, session_site, live_site):
    with namespace_routing():
        office = MultiEnv.put(id=f"{session_site.id.rsplit('/', 1)[0]}/office-live", mcp_server_envs=[],
                              website_envs=[session_site, live_site],
                              computer_envs=[Env.get(website_browser_desktop.id)], name="office")
        context = TaskStepContext()
        try:
            [deployed] = (await DeployEnvTaskStep(
                id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                child_envs={"desktop": {**CONTAINER, "env_params": {"form": "container", "site_routing": True}}},
            ).execute(context)).deployed_envs
            computer = DeployedEnv.from_dict(deployed.child_envs["desktop"])
            for site, user in ((live_site, "alex"), (session_site, "bea")):
                await BrowserSignInTaskStep(id=f"in-{site.environment_name}", version=None, env_id=office.id,
                                            site=site.id, user=user).execute(context)
            github = await _loads(computer, "https://github.com/", "GitHub")
            cookies = (await _browser(computer, "cookies/get", {"urls": ["https://github.com/"]}))["cookies"]
            shop_user = await _signed_in(computer)
            child = deployed.child_env("github", mcp=False)
            exported = await protocol_v1.get_data(child.base_url(deployed.environment_url),
                                                  headers=deployed.control_headers)
        finally:
            report = await teardown_run(context)

    assert not report.failed
    assert computer.metadata["computer"]["sites"]["names"]["routes"] == {"shop": ["shop.test"]}
    assert github is True, github
    assert {"name": "agentenv_test_session", "value": "alex"} in [
        {key: cookie[key] for key in ("name", "value")} for cookie in cookies]
    assert shop_user == "bea"
    assert "octocat/Hello-World" in json.dumps(exported.model_dump(mode="json"))
    assert context.metadata["browser_steps"]["in-github"]["names"] == ["github.com"]
    assert (context.metadata["reproducible"], [site["environment_name"] for site in
                                               context.metadata["live_websites"][office.id]]) == (False, ["github"])
