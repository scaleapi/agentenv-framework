"""The browser as a computer: agentenv's chromium deployed alone, on the ubuntu-24.04-headless base with no display
and the page driver, whose screen is chromium's active tab, headless. Gate 3 makes this machine's arch ready with it;
deployed through the computer provider like any computer, its computer_* tools see and drive the page, and its page
tools navigate, move in history, answer a dialog, use tabs and upload; nothing runs X and nothing listens but the driver
and the site proxy; a killed app server or driver is started again by supervisord and the computer is ready again; and
chromium killed fails it for good, computer-ready/v1 saying why and the tools answering with it. Needs Docker and the
local registry on localhost:5000."""

import asyncio
import json
import platform
import shlex
import time
import urllib.parse

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from PIL import Image

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer import COMPUTER_READY
from agentenv_protocol.computer.conformance.client import image
from agentenv_protocol.website_browser import BROWSER

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
#: How long an effect may take to reach the page, or the computer to be ready again; how often it's asked meanwhile.
WAIT_S, POLL_S = 120.0, 0.5
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


def test_gate_3_makes_this_machines_arch_ready_with_the_page_driver(browser_computer):
    results = browser_computer.metadata["composed"]["container"]
    probe = results[ARCH]["probe"]

    with namespace_routing():
        report = json.loads(FileArtifact.get(probe["report"]["id"], probe["report"]["version"]).load())
    print(json.dumps({"counts": probe["checks"], "extensions": probe["extensions"],
                      "not passed": [(c["check"], c["name"], c["status"], c["detail"][:160]) for c in report["checks"]
                                     if c["status"] != "passed"]}, indent=1))
    assert results[ARCH]["status"] == "ready" and probe["passed"], results[ARCH]
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


async def _get(deployed, path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30) as http:
        return await http.get(f"{deployed.environment_url}{path}", **kwargs)
