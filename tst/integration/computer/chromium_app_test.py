"""agentenv's chromium app on the ubuntu-24.04 base, as the chromium test bundle's desktops list it: gate 3 makes this
machine's arch ready with it; deployed on two start_urls, it shows one Chromium window, on the first, with no first-run,
sign-in or permission window beside it, and the second a tab away, which an agent reaches with Control+Tab; its managed
policies are where Chrome for Testing reads them; and a reset of its data clears its profile and opens it again. As a
website browser it runs Chrome itself with agentenv's flags on its debugging pipe, its computer's /dev/shm the 1 GiB it
asks for, as a container or a disk, so Chrome keeps its shared memory there, a reset_env too, answers location prompts
and leaves notification ones to ask as the deploy says, and its menu launcher, run while it runs, opens another window
of it with no second copy, its cache kept and it still ready; and the
desktop that gives it French defaults deploys with them, a deploy's time zone over them, each recorded; and the record
its control connection keeps lists a tab an agent opened and closed, a download with its size and sha256, a confirm()
accepted for it, and each launch, exit and stop with its cause, across a kill and a restart of its server; and browser/v1 reads and sets its
cookies, local storage and downloads; and its data plane signs it in, snapshots it, resets it and loads a
snapshot back; and the browser task steps set its cookies and storage and open, activate and close its tabs by URL.
Its start tabs open in order, the first active, a deploy recording each one's outcome and going on past a live page
that never answers; and with the run's sites routed to it, a start tab on one must load, judged by the page it
lands on when it sends itself elsewhere before it loads, so a site answering 503 fails the deploy, naming it. Needs Docker and the local registry on localhost:5000."""

import asyncio
import hashlib
import json
import platform
import re

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agentenv_protocol import client as protocol_v1

from agent_env.env import Env
from agentenv_protocol.types import WELL_KNOWN_PATH
from agentenv_protocol.website_browser import BROWSER, SHM_MIN_BYTES
from tst.integration.computer.browser_data import SITE as DATA_SITE, check_browser_data
from tst.integration.computer.browser_steps import check_browser_steps
from tst.integration.computer.browser_state import check_browser_state
from tst.integration.computer.start_tabs import host_pages, tabs_of
from tst.integration.computer.website_browser_test import _app, _browser_process, _keys, _ready
from tst.util.capabilities import skip_without_kvm
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

#: A page server's time to listen, the record's wait and poll, and the Control+Tab presses for one tab change.
SERVE_START_S, RECORD_WAIT_S, RECORD_POLL_S, TAB_PRESSES = 3, 60, 1, 3
ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
ALPHA = "data:text/html,<title>Alpha page</title><h1>Alpha</h1>"
BRAVO = "data:text/html,<title>Bravo page</title><h1>Bravo</h1>"
LOOPBACK = "http://127.0.0.1:8766/"
#: The size of the computer's /dev/shm, in bytes, as the browser base measures it.
SHM_BYTES = "import os; s = os.statvfs('/dev/shm'); print(s.f_frsize * s.f_blocks)"
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
POPPING = "http://127.0.0.1:8768/"
# Serves, on loopback for the seconds it's given, a page that opens /popped in a popup once, when Enter is pressed.
SERVE_POPPING = """import http.server, sys, threading, time
class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/popped":
            body = b"<title>Popped page</title><h1>Popped</h1>"
        else:
            body = (b"<title>Opener page</title><script>onkeydown = e => { if (e.key === 'Enter' && !window.opened) "
                    b"window.opened = window.open('/popped'); };</script>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 8768), Page)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[1]))
"""
RECORDED = "http://127.0.0.1:8767/"
# Serves, on loopback for the seconds it's given, a page that asks to confirm and then downloads report.bin, whose
# content differs at each request: the nth is _report(n).
SERVE_RECORDED = """import http.server, sys, threading, time
served = []
class Page(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/report.bin":
            served.append(1)
            body, kind = b"agentenv report %d\\n" % len(served) * 64, "application/octet-stream"
        else:
            body = (b"<title>asking</title><script>setTimeout(() => { document.title = confirm('Proceed?') ? "
                    b"'accepted' : 'dismissed'; setTimeout(() => location.href = '/report.bin', 500); }, 500)</script>")
            kind = "text/html"
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        if self.path == "/report.bin":
            self.send_header("Content-Disposition", "attachment; filename=report.bin")
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 8767), Page)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[1]))
"""
# A page that asks to confirm as soon as it loads, and says in its title what it was told.
AGENT_DIALOG = ("data:text/html,<title>asking</title><script>setTimeout(() => document.title = "
                "confirm('Proceed?') ? 'accepted' : 'dismissed', 1000)</script>")
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
            tabbed = await _tab_to(deployed, "Bravo page")
            policies = await inside.run(
                deployed, ["cat", "/etc/opt/chrome_for_testing/policies/managed/agentenv.json"], timeout_s=30)
            await inside.run(deployed, ["sh", "-c", "touch ~/.config/chromium/marker"], timeout_s=30)
            await protocol_v1.reset_data(f"{deployed.environment_url}/svc/chromium", headers=deployed.control_headers)
            marker = await inside.run(deployed, ["sh", "-c", "test -e ~/.config/chromium/marker"], timeout_s=30)
            reopened = await _chromium_windows(deployed, "Alpha page")
        finally:
            report = await teardown_run(context)

    assert opened == ["Alpha page - Google Chrome for Testing"]
    assert tabbed == ["Bravo page - Google Chrome for Testing"]
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
            await asyncio.sleep(SERVE_START_S)
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
    assert argv[5:9] == ["--remote-debugging-pipe", "--no-first-run", "--no-default-browser-check", "--lang=en-US"]
    assert argv[9:] == ["about:blank"]
    assert browser["listening"] == 0
    assert permissions == ["denied prompt - Google Chrome for Testing"]
    assert launched["exit_code"] == 0 and len(windows) == 2 and after["pid"] == browser["pid"]
    assert kept["exit_code"] == 0 and readiness["ready"] is True, readiness
    recorded = deployed.metadata["computer"]["app_settings"]["chromium"]
    assert (recorded["geolocation"], recorded["notifications"], recorded["start_urls"]) == ("deny", "ask", [LOOPBACK])
    assert not report.failed



@pytest.mark.asyncio
@pytest.mark.parametrize("form", ["container", pytest.param("disk", marks=skip_without_kvm())])
async def test_its_computer_gets_the_shared_memory_it_asks_for_so_chrome_keeps_it_there_and_a_reset_env_keeps_it(
        chromium_desktop, form):
    with namespace_routing():
        deployed = await Env.get(chromium_desktop.id).deploy(env_params={"form": form})
        context = TaskStepContext(deployed_envs=[deployed])
        sizes, in_files = [], []
        try:
            for reset in (False, True):
                if reset:
                    context = await ResetEnvTaskStep(id="reset", version=None, env_id=chromium_desktop.id,
                                                     fail_task_on_error=True).execute(context)
                sizes.append(int((await inside.run(deployed, ["python3", "-c", SHM_BYTES], timeout_s=30))["stdout"]))
                in_files.append("--disable-dev-shm-usage" in (await _browser_process(deployed))["argv"])
        finally:
            report = await teardown_run(context)

    recorded = deployed.metadata["computer"]
    assert (recorded["form"], recorded["shm_mb"]) == (form, SHM_MIN_BYTES >> 20)
    assert sizes == [SHM_MIN_BYTES, SHM_MIN_BYTES] and in_files == [False, False]
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


@pytest.mark.asyncio
async def test_a_popup_a_page_opens_loads_with_its_control_attached(chromium_desktop):
    with namespace_routing():
        deployed = await Env.get(chromium_desktop.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [POPPING]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            card = await _app(deployed, WELL_KNOWN_PATH, "chromium")
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_POPPING, "60"],
                                                       timeout_s=90))
            await asyncio.sleep(SERVE_START_S)
            await _keys(deployed, ["F5"])
            await _chromium_windows(deployed, "Opener page")
            await _keys(deployed, ["Enter"])
            popped = await _record(deployed, card, lambda r: any(tab["title"] == "Popped page" for tab in r["tabs"]))
            windows = await _chromium_windows(deployed, "Popped page")
            serving.cancel()
        finally:
            report = await teardown_run(context)

    assert [(tab["url"], tab["title"]) for tab in popped["tabs"]] == [
        (POPPING, "Opener page"), (f"{POPPING}popped", "Popped page")], popped
    assert any(title.startswith("Popped page") for title in windows), windows
    assert not report.failed


@pytest.mark.asyncio
async def test_its_control_records_tabs_a_download_a_dialog_and_each_launch_with_its_cause(chromium_desktop):
    settings = {"start_urls": [RECORDED], "dialogs": "accept"}
    with namespace_routing():
        deployed = await Env.get(chromium_desktop.id).deploy(env_params={"app_settings": {"chromium": settings}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            card = await _app(deployed, WELL_KNOWN_PATH, "chromium")
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_RECORDED, "90"],
                                                       timeout_s=120))
            await asyncio.sleep(SERVE_START_S)
            await _keys(deployed, ["F5"])
            accepted = await _chromium_windows(deployed, "accepted")
            await _keys(deployed, ["Control", "t"], ["Control", "w"])
            first = await _record(deployed, card, lambda r: _hashed(r) and r["closed_tabs"])
            await _keys(deployed, ["F5"])
            recorded = await _record(deployed, card, lambda r: len(_hashed(r)) > len(_hashed(first)))
            await serving
            browser = await _browser_process(deployed)
            await inside.run(deployed, ["kill", "-9", str(browser["pid"])], timeout_s=30)
            exited = await _record(deployed, card, lambda r: r["launches"][-1]["event"] == "exit")
            await inside.run(deployed, ["sh", "-c", LAUNCHER], timeout_s=60)
            relaunched = await _ready(deployed, card, "chromium")
            await inside.run(deployed, ["pkill", "-f", "computer.serve_app chromium.py"], timeout_s=30)
            restarted = await _record(deployed, card, lambda r: r["launches"][-1].get("cause") == "server_restart")
        finally:
            report = await teardown_run(context)

    assert accepted[0].startswith("accepted")
    # Chrome reloads its error page for the start tab once the page's server is up, and F5 reloads it too, so the page
    # may run more than once; each run's dialog and download must be recorded alike.
    assert recorded["dialogs"] and {(d["type"], d["message"], d["answer"], d["by"]) for d in recorded["dialogs"]} == {
        ("confirm", "Proceed?", "accepted", "agentenv")}, recorded
    # Every download is named report.bin, so Chrome saves the later ones as report (1).bin and on: each must be hashed
    # at its own path, as the content served for it.
    downloads = recorded["downloads"]
    assert len(downloads) >= 2 and {(d["filename"], d["state"]) for d in downloads} == {("report.bin", "completed")}
    assert len({d["path"] for d in downloads}) == len(downloads), recorded
    assert sorted(d["sha256"] for d in downloads) == sorted(
        hashlib.sha256(_report(n)).hexdigest() for n in range(1, len(downloads) + 1)), recorded
    assert [tab["closed_by"] for tab in recorded["closed_tabs"]] == ["unknown"], recorded
    assert [(tab["url"], tab["active"]) for tab in recorded["tabs"]] == [(RECORDED, True)], recorded
    assert [entry["event"] for entry in exited["launches"]] == ["launch", "exit"], exited
    assert exited["launches"][1]["signal"] == "SIGKILL" and recorded["since"], exited
    assert exited["launches"][0]["cause"] == "boot" and relaunched["ready"] is True, relaunched
    events = [(entry["event"], entry.get("cause")) for entry in restarted["launches"]]
    assert events[:3] == [("launch", "boot"), ("exit", None), ("launch", "launch")]
    assert events[3:] in ([("stop", None), ("launch", "server_restart")], [("launch", "server_restart")]), events
    assert not report.failed


@pytest.mark.asyncio
async def test_with_dialogs_left_to_the_agent_chromium_shows_one_and_records_the_agents_answer(chromium_desktop):
    with namespace_routing():
        deployed = await Env.get(chromium_desktop.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [AGENT_DIALOG]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            card = await _app(deployed, WELL_KNOWN_PATH, "chromium")
            pending = await _record(deployed, card, lambda r: r["dialogs"])
            await _keys(deployed, ["Enter"])
            answered_title = await _chromium_windows(deployed, "accepted")
            answered = await _record(deployed, card, lambda r: r["dialogs"] and r["dialogs"][0]["answer"])
        finally:
            report = await teardown_run(context)

    assert [(d["type"], d["answer"], d["by"]) for d in pending["dialogs"]] == [("confirm", None, None)], pending
    assert answered_title[0].startswith("accepted")
    assert [(d["type"], d["message"], d["answer"], d["by"]) for d in answered["dialogs"]] == [
        ("confirm", "Proceed?", "accepted", "agent")], answered
    assert not report.failed


@pytest.mark.asyncio
async def test_its_cookies_local_storage_and_downloads_are_read_and_set_through_browser_v1(chromium_desktop):
    with namespace_routing():
        deployed = await Env.get(chromium_desktop.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            card = await _app(deployed, WELL_KNOWN_PATH, "chromium")
            await _ready(deployed, card, "chromium")
            await check_browser_state(deployed, lambda op: f"{deployed.environment_url}/svc/chromium"
                                      f"{protocol_v1.find_extension_method(card, BROWSER, op)['endpoint']}")
        finally:
            report = await teardown_run(context)

    assert not report.failed


@pytest.mark.asyncio
async def test_its_data_plane_signs_it_in_snapshots_it_resets_it_and_loads_a_snapshot_back_whole(chromium_desktop):
    with namespace_routing():
        deployed = await Env.get(chromium_desktop.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [DATA_SITE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            card = await _app(deployed, WELL_KNOWN_PATH, "chromium")
            await _ready(deployed, card, "chromium")

            async def act():
                async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        assert not (await session.call_tool("computer_screenshot", {})).isError

            await check_browser_data(deployed, chromium_desktop.id, lambda op: f"{deployed.environment_url}/svc/chromium"
                                     f"{protocol_v1.find_extension_method(card, BROWSER, op)['endpoint']}", act)
        finally:
            report = await teardown_run(context)

    assert not report.failed


@pytest.mark.asyncio
async def test_its_start_tabs_open_in_order_the_first_active_and_a_live_one_that_never_answers_is_recorded_not_fatal(
        chromium_desktop):
    with host_pages() as port, namespace_routing():
        urls = [f"http://host.docker.internal:{port}/{page}" for page in ("alpha", "bravo", "hang")]
        deployed = await Env.get(chromium_desktop.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": urls}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            tabs = await tabs_of(deployed)
        finally:
            report = await teardown_run(context)

    start_tabs = deployed.metadata["computer"]["start_tabs"]["chromium"]
    assert [(tab["url"], tab["status"], tab["site"], tab["outcome"]) for tab in start_tabs] == [
        (urls[0], 200, None, "loaded"), (urls[1], 200, None, "loaded"), (urls[2], None, None, "timed_out")]
    assert start_tabs[2]["load_ms"] >= 30_000 and [tab["id"] for tab in tabs] == [tab["tab"] for tab in start_tabs]
    assert [(tab["url"], tab["title"], tab["active"]) for tab in tabs[:2]] == [(urls[0], "alpha", True),
                                                                               (urls[1], "bravo", False)]
    assert not tabs[2]["active"] and not report.failed


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 503], ids=["loads", "answers-503"])
async def test_a_start_tab_on_a_routed_site_must_load_where_it_lands_so_a_site_answering_503_fails_the_deploy(
        chromium_desktop, status):
    with host_pages(site_status=status) as port, namespace_routing():
        params = {"site_routing": True, "sites": [{"name": "shop", "url": f"http://127.0.0.1:{port}/website/shop/",
                                                   "domains": ["shop.test"]}],
                  "app_settings": {"chromium": {"start_urls": ["https://shop.test/"]}}}
        if status != 200:
            with pytest.raises(RuntimeError, match=r"chromium's start tab https://shop.test/ on the env's site shop "
                                                   r"answered 503"):
                await Env.get(chromium_desktop.id).deploy(env_params=params)
            return
        deployed = await Env.get(chromium_desktop.id).deploy(env_params=params)
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            tabs = await tabs_of(deployed)
        finally:
            report = await teardown_run(context)

    (tab,) = deployed.metadata["computer"]["start_tabs"]["chromium"]
    assert (tab["url"], tab["final_url"], tab["status"], tab["site"], tab["outcome"]) == (
        "https://shop.test/", "https://shop.test/landed", 200, "shop", "loaded")
    assert [(open_tab["url"], open_tab["title"]) for open_tab in tabs] == [("https://shop.test/landed", "landed")]
    assert not report.failed


@pytest.mark.asyncio
async def test_its_task_steps_set_cookies_and_storage_open_a_tab_behind_and_activate_and_close_it_by_its_url(
        chromium_desktop):
    with namespace_routing():
        deployed = await Env.get(chromium_desktop.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [DATA_SITE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            card = await _app(deployed, WELL_KNOWN_PATH, "chromium")
            await _ready(deployed, card, "chromium")
            await check_browser_steps(deployed, chromium_desktop.id, lambda op: f"{deployed.environment_url}/svc/chromium"
                                      f"{protocol_v1.find_extension_method(card, BROWSER, op)['endpoint']}")
        finally:
            report = await teardown_run(context)

    assert not report.failed


def _report(n: int) -> bytes:
    """The content SERVE_RECORDED serves at its nth request for report.bin."""
    return b"agentenv report %d\n" % n * 64


def _hashed(record: dict) -> list[dict]:
    return [download for download in record["downloads"] if download["sha256"]]


async def _record(deployed, card: dict, until) -> dict:
    """The browser's record, once ``until`` says it has what's waited for, or as it last was within ``RECORD_WAIT_S``."""
    path = protocol_v1.find_extension_method(card, BROWSER, "record")["endpoint"]
    for _ in range(RECORD_WAIT_S // RECORD_POLL_S):
        try:
            record = await _app(deployed, path, "chromium")
            if until(record):
                return record
        except httpx.HTTPError:
            pass
        await asyncio.sleep(RECORD_POLL_S)
    return record


async def _tab_to(deployed, title: str) -> list[str]:
    """Control+Tab pressed as an agent presses it, and again while the window still shows the tab before, which a
    press can reach before the next tab opens; the window titles once one starts with ``title``."""
    for attempt in range(TAB_PRESSES):
        await _keys(deployed, ["Control", "Tab"])
        try:
            return await _chromium_windows(deployed, title, seconds=20)
        except TimeoutError:
            if attempt == TAB_PRESSES - 1:
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
