"""The browser as a computer: agentenv's chromium deployed alone, on the ubuntu-24.04-headless base with no display
and the page driver, whose screen is chromium's active tab, headless. Gate 3 makes this machine's arch ready with it,
its input checks and page tools verified on the HTML desk; deployed through the computer provider like any computer,
its computer_* tools see and drive the page, and its page tools navigate, move in history, answer a dialog, use tabs
and upload; nothing runs X and nothing listens but the driver and the site proxy; a killed app server or driver is
started again by supervisord and the computer is ready again; and chromium killed fails it for good, computer-ready/v1
saying why and the tools answering with it. Through its cdp address, Playwright and Puppeteer drive it two at once,
each on its own session, by http_url and by ws_url; a wrong token is refused, and so is code in the computer without
one; one client's close leaves the browser up for the other, which can't open file: URLs; the address outlives a
relaunch, and the computer's reset ends it. browser_connect_cdp hands the echo agent its address, with the computer's
tools hidden from its role; after reset_env that address fails until the task runs browser_connect_cdp again, whose
fresh one its next turn opens the site through. Through an app server and a driver restart and a relaunch, the agent's
address keeps working, and two agents connected at once each drive the browser through their own. browser/v1 reads
and sets its cookies, local storage and downloads. Needs Docker and the local registry on
localhost:5000, and Node 20 or later and npm for the CDP clients."""

import asyncio
import json
import platform
import re
import shlex
import time
import urllib.parse
import uuid
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from PIL import Image

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import COMPUTER_PAGE, COMPUTER_READY
from agentenv_protocol.computer.conformance.client import image
from agentenv_protocol.website_browser import BROWSER

from agent_env.a2a_agent import conversation_store
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.browser_connect_cdp import BrowserConnectCdpTaskStep
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from tst.integration.computer.browser_state import check_browser_state, check_browser_state_under_chaos
from tst.util.a2a_test_agent import put_test_agent

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
#: How long an effect may take to reach the page, or the computer to be ready again; how often it's asked meanwhile.
WAIT_S, POLL_S = 120.0, 0.5
#: A WebSocket handshake for the browser's relay sent straight to its app server's socket, as code in the computer's
#: session can, with the token it's given; prints the status line it gets.
#: The lines of the computer's logs that hold either of two tokens, each shown masked, then one path the driver
#: logged past ``/cdp/``.
LOGGED_TOKENS = ('grep -rhF -e "$1" -e "$2" /var/log/agentenv | sed -e "s/$1/<1>/g" -e "s/$2/<2>/g" | cut -c1-300; '
                 'grep -rhoF "/cdp/<redacted>" /var/log/agentenv | head -1')
SOCKET_HANDSHAKE = """import socket, sys
s = socket.socket(socket.AF_UNIX)
s.connect("/opt/agentenv/apps/chromium/.agentenv/server.sock")
s.sendall(("GET /cdp/" + sys.argv[1] + "/devtools/browser HTTP/1.1\\r\\nHost: app\\r\\n"
           "Upgrade: websocket\\r\\nConnection: Upgrade\\r\\nSec-WebSocket-Version: 13\\r\\n"
           "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\\r\\n\\r\\n").encode())
print(s.recv(200).split(b"\\r\\n")[0].decode())
"""


# A page whose title says, as JSON, what reached it: clicks and double clicks on its red box, its input's value, how far
# it has scrolled and the last keys pressed.
PAGE = """<title>loading</title><style>body{margin:0;height:4000px;background:rgb(255,255,255)}
#box{position:absolute;left:100px;top:100px;width:200px;height:100px;background:rgb(255,0,0)}
#field{position:absolute;left:100px;top:300px;width:400px;height:40px;font-size:20px}</style>
<div id="box"></div><input id="field">
<script>
const s = {c: 0, d: 0, k: []};
const show = () => document.title = JSON.stringify({c: s.c, d: s.d, v: field.value, y: Math.round(scrollY),
                                                    k: s.k.slice(-4)});
box.onclick = () => { s.c++; show(); };
box.ondblclick = () => { s.d++; show(); };
field.oninput = show;
onscroll = show;
onkeydown = e => { s.k.push((e.ctrlKey ? "C-" : "") + e.key); show(); };
show();
</script>"""
PAGE_URL = "data:text/html," + urllib.parse.quote(PAGE)
SITE = "http://127.0.0.1:8771/"
# Serves, on loopback for the seconds it's given, a home page with a button that asks to confirm and a file input, each
# saying in the title what came of it, and two more pages.
SERVE_SITE = """import http.server, sys, threading, time
PAGES = {
    "/": (b"<title>Home</title><style>body{margin:0} button,input{position:absolute;left:0;width:300px;"
          b"height:100px}</style><button style='top:0' "
          b"onclick=\\"document.title = 'confirmed ' + confirm('Delete this message?')\\">ask</button>"
          b"<input type=file style='top:200px' onchange=\\"document.title = 'chose ' + this.files[0].name\\">"),
    "/two": b"<title>Two</title><h1>Two</h1>",
    "/three": b"<title>Three</title><h1>Three</h1>",
}
class Site(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGES.get(self.path, b"<title>Missing</title>")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass
server = http.server.ThreadingHTTPServer(("127.0.0.1", 8771), Site)
threading.Thread(target=server.serve_forever, daemon=True).start()
time.sleep(float(sys.argv[1]))
"""
SERVE_START_S = 3
#: A page that needs no server, so killing the driver, whose child a served site is, leaves it to open.
DATA_PAGE = "data:text/html,<title>Home</title><h1>Home</h1>"
# Every TCP port something in the computer listens on, and whether any process is an X server.
LISTENING = """import glob
ports = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    for line in open(table).read().splitlines()[1:]:
        fields = line.split()
        if fields[3] == "0A":
            ports.add(int(fields[1].rsplit(":", 1)[1], 16))
x = [p for p in glob.glob("/proc/[0-9]*/comm") if open(p).read().strip() in ("Xvfb", "Xorg", "xfce4-session")]
print(sorted(ports), bool(x))
"""


def test_gate_3_makes_this_machines_arch_ready_with_the_page_driver_its_input_and_page_tools_verified(browser_computer):
    results = browser_computer.metadata["composed"]["container"]
    probe = results[ARCH]["probe"]

    with namespace_routing():
        report = json.loads(FileArtifact.get(probe["report"]["id"], probe["report"]["version"]).load())
    print(json.dumps({"counts": probe["checks"], "extensions": probe["extensions"],
                      "not passed": [(c["check"], c["name"], c["status"], c["detail"][:160]) for c in report["checks"]
                                     if c["status"] != "passed"]}, indent=1))
    assert results[ARCH]["status"] == "ready" and probe["passed"], results[ARCH]
    assert probe["checks"]["unverified"] == 0 and probe["extensions"][COMPUTER_PAGE] == "verified", probe
    assert [app.environment_name for app in browser_computer.composed_apps()] == ["chromium"]


@pytest.mark.asyncio
async def test_its_tools_see_and_drive_the_page_that_is_its_screen(browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [PAGE_URL]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            await _state(deployed, lambda s: True)
            async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    shot = image(await session.call_tool("computer_screenshot", {"space": "frame"}))
                    await _tool(session, "computer_click", {"x": 200, "y": 150, "space": "frame"})
                    clicked = await _state(deployed, lambda s: s["c"] == 1)
                    await _tool(session, "computer_click", {"x": 200, "y": 150, "count": 2, "space": "frame"})
                    doubled = await _state(deployed, lambda s: s["d"] == 1)
                    await _tool(session, "computer_click", {"x": 300, "y": 320, "space": "frame"})
                    await _tool(session, "computer_type", {"text": "Hello, World!"})
                    typed = await _state(deployed, lambda s: s["v"] == "Hello, World!")
                    await _tool(session, "computer_key", {"keys": ["Control", "a"]})
                    await _tool(session, "computer_type", {"text": "x"})
                    replaced = await _state(deployed, lambda s: s["v"] == "x")
                    await _tool(session, "computer_key", {"keys": ["Enter"]})
                    entered = await _state(deployed, lambda s: "Enter" in s["k"])
                    await _tool(session, "computer_scroll", {"x": 640, "y": 600, "direction": "down", "amount": 3,
                                                             "space": "frame"})
                    scrolled = await _state(deployed, lambda s: s["y"] >= 300)
                    after = image(await session.call_tool("computer_screenshot", {"space": "frame"}))
            listening, x_server = (await inside.run(deployed, ["python3", "-c", LISTENING], timeout_s=30))[
                "stdout"].rsplit(" ", 1)
            command = (await inside.run(deployed, ["sh", "-c", "tr '\\0' ' ' < /proc/$(pgrep -o -f -- "
                                                   "'[-]-remote-debugging-pipe')/cmdline"], timeout_s=30))["stdout"]
        finally:
            report = await teardown_run(context)

    assert shot.size == (1280, 800) and shot.getpixel((200, 150)) == (255, 0, 0)
    assert shot.getpixel((640, 700)) == (255, 255, 255)
    assert clicked["c"] == 1 and doubled["d"] == 1 and typed["v"] == "Hello, World!" and replaced["v"] == "x"
    assert "C-a" in replaced["k"] and "Enter" in entered["k"] and scrolled["y"] >= 300
    assert after.getpixel((200, 150)) != (255, 0, 0)
    assert json.loads(listening) == [3128, 18765] and x_server.strip() == "False"
    assert "--headless=new" in command and "--window-size=1280,800" in command
    assert not report.failed


@pytest.mark.asyncio
async def test_a_killed_app_server_or_driver_comes_back_and_a_killed_browser_fails_it_for_good(browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [PAGE_URL]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            await _kill(deployed, "[s]erve_app")
            server_back = await _ready(deployed)
            launches = (await _record(deployed))["launches"]
            await _kill(deployed, "[c]omputer.serve driver.py:PageDriver")
            driver_back = await _ready(deployed)
            shot = await _screenshot(deployed)
            await _kill(deployed, "[-]-remote-debugging-pipe", oldest=True)
            failed = await _until(deployed, lambda status: "failed" in status)
            async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    refused = await session.call_tool("computer_screenshot", {"space": "frame"})
        finally:
            report = await teardown_run(context)

    assert server_back["ready"] and driver_back["ready"] and shot.size == (1280, 800)
    assert [entry.get("cause") for entry in launches if entry["event"] == "launch"] == ["boot", "server_restart"]
    browser = next(check for check in failed["checks"] if check["name"] == "browser")
    assert failed["ready"] is False and browser["final"] is True and failed["failed"] == f"browser: {browser['detail']}"
    assert "by signal SIGKILL" in browser["detail"] and "infrastructure failure" in browser["detail"]
    assert refused.isError and "has failed" in refused.content[0].text and "SIGKILL" in refused.content[0].text
    assert not report.failed


@pytest.mark.asyncio
async def test_an_agent_navigates_moves_in_history_answers_a_dialog_uses_tabs_and_uploads_with_its_page_tools(
        browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [SITE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_SITE, "900"], timeout_s=960))
            await inside.run(deployed, ["sh", "-c", "printf cv > ~/Uploads/cv.txt"], timeout_s=30)
            await asyncio.sleep(SERVE_START_S)
            async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await _tool(session, "computer_history", {"direction": "reload"})
                    home = await _shown(session, lambda page: page["title"] == "Home")
                    await _tool(session, "computer_navigate", {"url": SITE + "two"})
                    two = await _shown(session, lambda page: page["title"] == "Two")
                    await _tool(session, "computer_history", {"direction": "back"})
                    back = await _shown(session, lambda page: page["title"] == "Home")
                    await _tool(session, "computer_history", {"direction": "forward"})
                    forward = await _shown(session, lambda page: page["title"] == "Two")
                    await _tool(session, "computer_history", {"direction": "back"})
                    await _shown(session, lambda page: page["title"] == "Home")
                    asked = await session.call_tool("computer_click", {"x": 150, "y": 50, "screenshot": True})
                    open_dialog = await _shown(session, lambda page: page.get("dialog") is not None)
                    blocked = await session.call_tool("computer_click", {"x": 150, "y": 50})
                    await _tool(session, "computer_dialog", {"accept": True})
                    confirmed = await _shown(session, lambda page: page["title"] == "confirmed true")
                    opened = await session.call_tool("computer_tabs", {"action": "open", "url": SITE + "three"})
                    three = await _shown(session, lambda page: page["title"] == "Three")
                    first = next(tab["tab"] for tab in opened.structuredContent["tabs"] if not tab["active"])
                    second = next(tab["tab"] for tab in opened.structuredContent["tabs"] if tab["active"])
                    await _tool(session, "computer_tabs", {"action": "switch", "tab": first})
                    switched = await _shown(session, lambda page: page["tab"] == first)
                    closed = await session.call_tool("computer_tabs", {"action": "close", "tab": second})
                    await _tool(session, "computer_click", {"x": 150, "y": 250})
                    chooser = await _shown(session, lambda page: page.get("file_chooser") is not None)
                    await _tool(session, "computer_upload", {"paths": ["cv.txt"]})
                    chose = await _shown(session, lambda page: page["title"] == "chose cv.txt")
            record = await _record(deployed)
            serving.cancel()
        finally:
            report = await teardown_run(context)

    assert home["page"]["url"] == SITE and two["page"]["url"] == SITE + "two"
    assert back["page"]["title"] == "Home" and forward["page"]["title"] == "Two"
    assert not asked.isError and asked.structuredContent["applied"] == "yes"
    assert open_dialog["page"]["dialog"] == {"type": "confirm", "message": "Delete this message?", "default_text": ""}
    assert 'a confirm dialog is open: "Delete this message?"' in open_dialog["summary"]
    assert blocked.isError and blocked.structuredContent["code"] == "dialog_open"
    assert confirmed["page"]["title"] == "confirmed true" and "dialog" not in confirmed["page"]
    assert [tab["url"] for tab in opened.structuredContent["tabs"]] == [SITE, SITE + "three"]
    assert three["page"]["tab"] == second and switched["page"]["title"] == "confirmed true"
    assert [tab["tab"] for tab in closed.structuredContent["tabs"]] == [first]
    assert chooser["page"]["file_chooser"] == {"multiple": False} and chose["page"]["title"] == "chose cv.txt"
    assert [(d["message"], d["answer"], d["by"]) for d in record["dialogs"]] == [
        ("Delete this message?", "accepted", "agent")]
    assert any(tab["id"] == second and tab["closed_by"] == "agentenv" for tab in record["closed_tabs"])
    assert not report.failed


@pytest.mark.asyncio
async def test_its_cookies_local_storage_and_downloads_are_read_and_set_through_browser_v1(browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            await _ready(deployed)
            card = (await _get(deployed, "/.well-known/agent-env.json")).json()
            app = next(child for child in card["children_environments"] if child["name"] == "chromium")
            await check_browser_state(deployed, lambda op: deployed.environment_url
                                      + protocol_v1.find_extension_method(app, BROWSER, op)["endpoint"])
        finally:
            report = await teardown_run(context)

    assert not report.failed


@pytest.mark.asyncio
async def test_its_state_operations_hold_up_at_once_through_a_server_restart_and_fail_at_once_with_the_browser_gone(
        browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            await _ready(deployed)
            card = (await _get(deployed, "/.well-known/agent-env.json")).json()
            app = next(child for child in card["children_environments"] if child["name"] == "chromium")
            kept = await check_browser_state_under_chaos(
                deployed, lambda op: deployed.environment_url + protocol_v1.find_extension_method(app, BROWSER, op)[
                    "endpoint"], lambda pattern: _kill(deployed, pattern, oldest=pattern.startswith("[-]")),
                lambda: _ready(deployed))
        finally:
            report = await teardown_run(context)

    print("browser-state-kept", json.dumps(kept))
    assert not report.failed


async def _shown(session: ClientSession, holds) -> dict:
    """A screenshot's result once the page it shows is what ``holds`` waits for, within ``WAIT_S``."""
    last, deadline = None, time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        shot = await session.call_tool("computer_screenshot", {"space": "frame"})
        last = shot.structuredContent
        if not shot.isError and last.get("page") and holds(last["page"]):
            return last
        await asyncio.sleep(POLL_S)
    raise AssertionError(f"the page didn't get there within {WAIT_S:g} s; it last showed {last!r}")


async def _tool(session: ClientSession, name: str, args: dict) -> None:
    result = await session.call_tool(name, args)
    assert not result.isError, f"{name} {args}: {result.content[0].text}"


async def _record(deployed) -> dict:
    card = (await _get(deployed, "/.well-known/agent-env.json")).json()
    app = next(child for child in card["children_environments"] if child["name"] == "chromium")
    path = protocol_v1.find_extension_method(app, BROWSER, "record")["endpoint"]
    return (await _get(deployed, path, headers=deployed.control_headers)).json()


async def _state(deployed, holds) -> dict:
    """The page's state, from its title, once ``holds`` says it has what's waited for, within ``WAIT_S``."""
    last, deadline = None, time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        try:
            last = json.loads((await _record(deployed))["tabs"][0]["title"])
            if holds(last):
                return last
        except (httpx.HTTPError, ValueError, KeyError, IndexError):
            pass
        await asyncio.sleep(POLL_S)
    raise AssertionError(f"the page didn't get there within {WAIT_S:g} s; it last said {last!r}")


async def _status(deployed) -> dict:
    card = (await _get(deployed, "/.well-known/agent-env.json")).json()
    path = protocol_v1.find_extension_method(card, COMPUTER_READY, "status")["endpoint"]
    return (await _get(deployed, path, headers=deployed.control_headers)).json()


async def _until(deployed, holds) -> dict:
    last, deadline = None, time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        try:
            last = await _status(deployed)
            if holds(last):
                return last
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        await asyncio.sleep(POLL_S)
    raise AssertionError(f"computer-ready didn't get there within {WAIT_S:g} s; it last said {last!r}")


async def _ready(deployed) -> dict:
    await asyncio.sleep(2)
    return await _until(deployed, lambda status: status["ready"])


async def _kill(deployed, pattern: str, oldest: bool = False) -> None:
    """SIGKILL the newest, or oldest, process matching ``pattern``, a second after computer-exec has answered, since the
    driver that runs the command may be the one killed."""
    pick = "-o" if oldest else "-n"
    killer = f"sleep 1; kill -9 $(pgrep {pick} -f -- '{pattern}')"
    await inside.run(deployed, ["sh", "-c", f"setsid sh -c {shlex.quote(killer)} >/dev/null 2>&1 < /dev/null &"],
                     timeout_s=30)
    await asyncio.sleep(2)


async def _screenshot(deployed) -> Image.Image:
    async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return image(await session.call_tool("computer_screenshot", {"space": "frame"}))


@pytest.mark.asyncio
async def test_playwright_and_puppeteer_drive_it_through_its_cdp_address_which_a_relaunch_keeps_and_a_reset_ends(
        browser_computer, cdp_clients):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [SITE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_SITE, "900"], timeout_s=960))
            await asyncio.sleep(SERVE_START_S)
            ada, again, bob = [await _cdp(deployed, user) for user in ("ada", "ada", "bob")]
            together = await _clients(cdp_clients, "together", ada)
            refused = await _clients(cdp_clients, "refused", ada)
            from_inside = (await inside.run(deployed, ["python3", "-c", SOCKET_HANDSHAKE, "not-a-token"],
                                            timeout_s=30))["stdout"].strip()
            tokens = [address["http_url"].split("/cdp/")[1].strip("/") for address in (ada, bob)]
            logged = (await inside.run(deployed, ["sh", "-c", LOGGED_TOKENS, "sh", *tokens],
                                       timeout_s=30))["stdout"].splitlines()
            record = await _record(deployed)
            guarded = await _clients(cdp_clients, "guard", ada)
            guarded_record = await _record(deployed)
            await protocol_v1.reset_data(f"{deployed.environment_url}/svc/chromium", headers=deployed.control_headers)
            relaunched_address = await _cdp(deployed, "ada")
            relaunched = await _clients(cdp_clients, "together", ada)
            await ResetEnvTaskStep(id="reset", version=None, env_id=browser_computer.id,
                                   fail_task_on_error=True).execute(context)
            after_reset = await _clients(cdp_clients, "refused", ada, wrong=False)
            fresh = await _cdp(deployed, "ada")
            serving.cancel()
        finally:
            report = await teardown_run(context)

    assert again == ada and bob != ada and ada["http_url"].startswith(f"{deployed.environment_url}/svc/chromium/cdp/")
    for client in ("playwright_http", "puppeteer_ws", "playwright_ws"):
        assert together[client].get("ok", {}).get("title") == "Home", together
        assert relaunched[client].get("ok", {}).get("title") == "Home", relaunched
    assert "error" in refused["playwright"] and "error" in refused["puppeteer"], refused
    assert from_inside == "HTTP/1.1 403 Forbidden"
    assert logged == ["/cdp/<redacted>"], logged
    assert [c["user"] for c in record["cdp_connections"]] == ["ada", "ada", "ada"]
    assert guarded["closed"] == {"ok": True} and guarded["after"] == {"ok": "Home"}, guarded
    assert "agentenv refuses Page.navigate: file:" in guarded["file"].get("error", ""), guarded
    assert {("Browser.close", "closed"), ("Page.navigate", "refused"), ("Browser.setDownloadBehavior", "changed")} <= {
        (entry["method"], entry["did"]) for c in guarded_record["cdp_connections"] for entry in c["guarded"]}
    assert relaunched_address == ada
    assert "error" in after_reset["playwright"] and "error" in after_reset["puppeteer"], after_reset
    assert fresh != ada and not report.failed


@pytest.mark.asyncio
async def test_browser_connect_cdp_hands_an_agent_its_address_with_the_tools_hidden_which_a_reset_ends_till_it_reruns(
        browser_computer):
    with namespace_routing():
        agent = put_test_agent(f"cdp-agent-{uuid.uuid4().hex[:8]}")
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [SITE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        serving = None
        try:
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_SITE, "900"], timeout_s=960))
            await asyncio.sleep(SERVE_START_S)
            await DeployAgentTaskStep(id="deploy-solver", version=None, env_ids=[], a2a_agent_id=agent.id,
                                      a2a_agent_version=agent.version, agent_name="solver",
                                      sandbox_type="local").execute(context)
            await BrowserConnectCdpTaskStep(id="connect", version=None, env_id=browser_computer.id,
                                            agent_name="solver").execute(context)
            first, tools = await _opened_over_cdp(context, "first"), await _tool_names(deployed)
            serving.cancel()
            await ResetEnvTaskStep(id="reset", version=None, env_id=browser_computer.id,
                                   fail_task_on_error=True).execute(context)
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_SITE, "900"], timeout_s=960))
            await asyncio.sleep(SERVE_START_S)
            stale = await _opened_over_cdp(context, "stale")
            await BrowserConnectCdpTaskStep(id="reconnect", version=None, env_id=browser_computer.id,
                                            agent_name="solver").execute(context)
            second, tools_after = await _opened_over_cdp(context, "second"), await _tool_names(deployed)
        finally:
            if serving is not None:
                serving.cancel()
            report = await teardown_run(context)

    (first_address, first_title), (second_address, second_title) = first, second
    assert first_title == second_title == "Home" and first_address != second_address, (first, second)
    assert stale[0] == first_address and stale[1].startswith("failed: "), stale
    assert tools == tools_after == []
    assert context.metadata["browser_connections"] == [{
        "step_id": "reconnect", "env_id": browser_computer.id, "agent_name": "solver", "browser": "chromium",
        "name": "chromium", "keep_tools": False, "role": "default", "tools_hidden": "*"}]
    assert not report.failed


@pytest.mark.asyncio
async def test_a_cdp_agent_keeps_its_address_through_an_app_server_and_a_driver_restart_and_a_browser_relaunch(
        browser_computer):
    with namespace_routing():
        agent = put_test_agent(f"cdp-chaos-{uuid.uuid4().hex[:8]}")
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [DATA_PAGE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            await _connected(context, agent, browser_computer.id, "solver")
            first = await _opened_over_cdp(context, "first", url=DATA_PAGE)
            await _kill(deployed, "[s]erve_app")
            await _ready(deployed)
            after_server = await _opened_over_cdp(context, "after-server", url=DATA_PAGE)
            await _kill(deployed, "[c]omputer.serve driver.py:PageDriver")
            await _ready(deployed)
            after_driver = await _opened_over_cdp(context, "after-driver", url=DATA_PAGE)
            await protocol_v1.reset_data(f"{deployed.environment_url}/svc/chromium", headers=deployed.control_headers)
            after_relaunch = await _opened_over_cdp(context, "after-relaunch", url=DATA_PAGE)
        finally:
            report = await teardown_run(context)

    assert [title for _, title in (first, after_server, after_driver, after_relaunch)] == ["Home"] * 4
    assert first[0] == after_server[0] == after_driver[0] == after_relaunch[0]
    assert not report.failed


@pytest.mark.asyncio
async def test_two_cdp_agents_drive_one_browser_at_once_each_through_an_address_of_its_own(browser_computer):
    with namespace_routing():
        agent = put_test_agent(f"cdp-pair-{uuid.uuid4().hex[:8]}")
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [DATA_PAGE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            for name in ("solver", "buyer"):
                await _connected(context, agent, browser_computer.id, name)
            turns = await asyncio.gather(*(_opened_over_cdp(context, f"{name}-turn-{index}", name, DATA_PAGE)
                                           for name in ("solver", "buyer") for index in range(2)))
            record = await _record(deployed)
        finally:
            report = await teardown_run(context)

    solver, buyer = turns[:2], turns[2:]
    assert [title for _, title in turns] == ["Home"] * 4
    assert len({address for address, _ in solver}) == len({address for address, _ in buyer}) == 1
    assert solver[0][0] != buyer[0][0]
    assert {connection["user"] for connection in record["cdp_connections"]} == {"solver", "buyer"}
    assert [(c["agent_name"], c["name"]) for c in context.metadata["browser_connections"]] == [
        ("solver", "chromium"), ("buyer", "chromium")]
    assert not report.failed


async def _connected(context: TaskStepContext, agent, env_id: str, name: str) -> None:
    """The echo agent deployed as ``name`` on the local provider, with a CDP address to the browser."""
    await DeployAgentTaskStep(id=f"deploy-{name}", version=None, env_ids=[], a2a_agent_id=agent.id,
                              a2a_agent_version=agent.version, agent_name=name, sandbox_type="local").execute(context)
    await BrowserConnectCdpTaskStep(id=f"connect-{name}", version=None, env_id=env_id, agent_name=name).execute(context)


async def _opened_over_cdp(context: TaskStepContext, prompt_id: str, agent_name: str = "solver",
                           url: str = SITE) -> tuple[str, str]:
    """``agent_name``'s turn opening ``url`` over its ``chromium`` CDP address: that address's digest, and the title
    it saw or, when it couldn't open it, ``failed: <why>``."""
    await PromptAgentTaskStep(id=prompt_id, version=None, prompt_id=prompt_id, agent_name=agent_name,
                              prompt=f"cdp chromium {url}", poll_interval_seconds=1).execute(context)
    reply = conversation_store.get_conversation(context.metadata["a2a_conversations"][prompt_id])["messages"][1]
    line = "\n".join(part["text"] for part in reply["parts"] if part["kind"] == "text").splitlines()[-1]
    match = re.fullmatch(r"cdp chromium (\w+)( failed)?: (.*)", line)
    assert match, line
    return match.group(1), f"failed: {match.group(3)}" if match.group(2) else match.group(3)


async def _tool_names(deployed) -> list[str]:
    """The tools the env's MCP lists to a caller with no role."""
    async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return [tool.name for tool in (await session.list_tools()).tools]


async def _cdp(deployed, user: str) -> dict:
    """``user``'s CDP address, asked of the browser as agentenv asks: through the driver, with the control token."""
    card = (await _get(deployed, "/.well-known/agent-env.json")).json()
    app = next(child for child in card["children_environments"] if child["name"] == "chromium")
    path = protocol_v1.find_extension_method(app, BROWSER, "cdp")["endpoint"]
    async with httpx.AsyncClient(timeout=120) as http:
        answer = await http.post(f"{deployed.environment_url}{path}", json={"user": user},
                                 headers=deployed.control_headers)
    assert answer.status_code == 200, answer.text
    return answer.json()


async def _clients(folder: Path, check: str, address: dict, wrong: bool = True) -> dict:
    """What the CDP clients saw doing ``check`` at ``address``; for ``refused``, with its token changed unless not
    ``wrong``."""
    argv = ["node", "connect.mjs", check, address["http_url"], address["ws_url"], SITE] + ([] if wrong else ["exact"])
    process = await asyncio.create_subprocess_exec(*argv, cwd=folder, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(process.communicate(), 300)
    assert process.returncode == 0, err.decode()[-2000:]
    return json.loads(out.decode().strip().splitlines()[-1])


async def _get(deployed, path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30) as http:
        return await http.get(f"{deployed.environment_url}{path}", **kwargs)
