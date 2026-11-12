"""The browser as a computer: agentenv's chromium deployed alone, on the ubuntu-24.04-browser base with no desktop
and the page driver, whose screen is chromium's active tab, headful on the X server only the browser uses. Gate 3 makes this machine's arch ready with it,
its input checks and page tools verified on the HTML desk; deployed through the computer provider like any computer,
its computer_* tools see and drive the page, Playwright MCP's browser_* tools beside them type into it and answer its
dialog, each call an entry in the browser's log that its CDP commands and the navigation they cause point at, the two
that run code hidden until tool access names one, as the installed Playwright MCP lists them, and back after Playwright
MCP or the driver is killed, and its page tools navigate, move in history, answer a dialog, use tabs
and upload; the browser's X server runs but no desktop, and nothing listens but the driver and the site proxy; a
killed app server or driver is started again by supervisord and the computer is ready again; chromium killed fails it
for good, computer-ready/v1 saying why and the tools answering with it, and so does its X server killed, which takes
chromium with it though supervisord starts X again. Through its cdp address, Playwright and Puppeteer drive it two at once,
each on its own session, by http_url and by ws_url; a wrong token is refused, and so is code in the computer without
one; one client's close leaves the browser up for the other, which can't open file: URLs; the address outlives a
relaunch by its app server's restart, and the browser's reset and the computer's end it. Behind a front that ends TLS, as a remote provider's tunnel does,
an address asked for under the origin named is https and wss, as discovery says too, and the clients connect through it. browser_connect_cdp hands the echo agent its address, with the computer's
tools hidden from its role; after reset_env that address fails until the task runs browser_connect_cdp again, whose
fresh one its next turn opens the site through. Through an app server's restart, which relaunches the browser, and a
driver's, the agent's address keeps working, and two agents connected at once each drive the browser through their own. browser/v1 reads
and sets its cookies, local storage and downloads, and its data plane signs it in, snapshots it, resets it, loads a
snapshot back and holds up under chaos; the browser task steps set its cookies and storage and open, activate and
close its tabs by URL; a task whose deploy_env pins a version of the browser, its latest one with none, passes
preflight on the version pinned and runs its browser steps through task.run; and each start tab's page sees the
window's size from its first script. Alone, it makes no request of its own, its proxy seeing only the page an agent
loads for well over a minute after it's ready, but for one Google account check at launch. Routing sites, it goes out only through the shared site proxy's forward port:
start tabs on a site over HTTPS and on a .localhost one load through it, an event stream comes through as it's sent,
any other host and loopback are refused and logged, the session user can't stop the port, killed it comes back, and
after a reset it serves the sites and refuses the rest again; its network log has each exchange whole, so that
WebArena-Verified's own loader, reading the network.har its data/get exports, sees an agent's sign-in form, redirect,
cookie, JSON and multipart as its grader reads them, and the live hosts' tunnels and plain requests beside them; alone,
it has each connection the base's stand-in made, and each that failed. As a multi env's website browser, ``browser_envs =
[chromium]``, it deploys on a sandbox of its own as one more computer child beside a synthetic site, with no
playwright-mcp: its start tab loads the site by name through the gateway, an agent drives it through the gateway's /mcp
under ``chromium_``, its CDP address asked through ``/svc/chromium/`` names the gateway and reaches the site by name, a
later process reattaches it from the instance store, a universe loaded into the multi env sets its cookies, and
reset_env brings it back after its container is killed and after the gateway restarts; a multi env naming none takes
the run's browser, chromium too, the same way, and ``browser_envs = []`` deploys none; and browser_connect_cdp
hands an agent an address to it at the multi env's gateway, whose relay takes the agent's page to the site by name,
with the multi env's tools hidden from the agent's role. Its log has an entry for every action a CDP agent and an MCP agent take, over CDP, page
tools and pixels, with who, the tab and its URL and no secret either typed or set, and each tab event names the
command that caused it.
Needs Docker and the local registry on
localhost:5000, and Node 20 or later and npm for the CDP clients."""

import asyncio
import base64
import calendar
import contextlib
import gzip
import http.server
import itertools
import json
import os
import platform
import re
import shlex
import shutil
import ssl
import subprocess
import threading
import time
import urllib.parse
import uuid
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client, streamablehttp_client
from PIL import Image
from websockets.asyncio.client import connect

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.client import find_extension_method, invoke_extension
from agentenv_protocol.computer import COMPUTER_PAGE, COMPUTER_READY
from agentenv_protocol.computer.conformance.client import image
from agentenv_protocol.tool_access import ENABLE, ENABLE_TOOL_EXTENSION_URI
from agentenv_protocol.tool_context import ROLE_HEADER
from agentenv_protocol.types import uploaded_object_path
from agentenv_protocol.website_browser import BROWSER, HAR_FILE

import agent_env
from agent_env.a2a_agent import conversation_store
from agent_env.artifact import Artifact, EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.config import get_config
from agent_env.config.runtime import WEBSITE_BROWSER_OVERRIDE, run_env_ids
from agent_env.env import Env
from agent_env.env.env import DeployedEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.playwright_mcp import PLAYWRIGHT_MCP_ENVIRONMENT_NAME
from agent_env.env.envs.website import WebsiteEnv
from agent_env.env.envs.computer.utils import inside
from agent_env.env.store import get_env_store
from agent_env.providers.env_providers.constants import AGENT_ENV_WEBSITE_BACKEND_PORT, AGENT_ENV_WEBSITE_BACKEND_SUFFIX
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.store.object_store.local.tls import LocalCA, local_ca
from agent_env.store.routing import namespace_routing
from agent_env.task import Task
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.browser.connect_cdp import BrowserConnectCdpTaskStep
from agent_env.task_step.task_steps.browser.recording import (RECORDINGS_KEY as BROWSER_RECORDINGS,
                                                             EndBrowserRecordingTaskStep, StartBrowserRecordingTaskStep)
from agent_env.task_step.task_steps.browser.sign_in import BrowserSignInTaskStep
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from tst.integration.computer.browser_contexts import check_browser_contexts
from tst.integration.computer.browser_data import SITE as DATA_SITE, check_browser_data, check_browser_data_under_chaos
from tst.integration.computer.browser_steps import check_browser_steps
from tst.integration.computer.browser_state import check_browser_state, check_browser_state_under_chaos
from tst.integration.computer.multi_env_computers_test import _containers, _gateway, _image
from tst.integration.computer.website_browser_test import _signed_in
from tst.integration.computer.start_tabs import host_pages, tabs_of
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs
from tst.util import local_sandboxes
from tst.util.a2a_test_agent import put_test_agent
from tst.util.tls_front import tls_front

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
#: Playwright MCP's own tools/list in the computer, from the node and cli.js it's given, as JSON.
LISTED_TOOLS = """import json, subprocess, sys
messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18",
             "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"}, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
out = subprocess.run(sys.argv[1:] + ["--cdp-endpoint", "http://127.0.0.1:9/", "--caps", "vision"], capture_output=True,
                     input="\\n".join(map(json.dumps, messages)) + "\\n", text=True, timeout=60).stdout
print(json.dumps(next(r["result"]["tools"] for r in map(json.loads, out.splitlines()) if r.get("id") == 2)))
"""
PLAYWRIGHT_MCP_TOOLS = Path(agent_env.__file__).parent / "drivers" / "envs" / "page" / "playwright-mcp-tools.json"
#: A page that needs no server, so killing the driver, whose child a served site is, leaves it to open.
DATA_PAGE = "data:text/html,<title>Home</title><h1>Home</h1>"
#: A page a task's tab steps open, bring to the front and close by its URL.
PINNED_PAGE = "data:text/html,pinned"
#: A page whose title is the window's size as its first script sees it, told apart by ``{}``.
SIZED = "data:text/html,<title>{}</title><script>document.title = innerWidth + 'x' + innerHeight</script>"
# Every TCP port something in the computer listens on, and whether any process is an X server.
LISTENING = """import glob
ports = set()
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    for line in open(table).read().splitlines()[1:]:
        fields = line.split()
        if fields[3] == "0A":
            ports.add(int(fields[1].rsplit(":", 1)[1], 16))
x = sorted({open(p).read().strip() for p in glob.glob("/proc/[0-9]*/comm")} & {"Xvfb", "Xorg", "xfce4-session"})
print(sorted(ports), ",".join(x))
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
    assert json.loads(listening) == [3128, 18765] and x_server.strip() == "Xvfb"
    assert "--headless" not in command and "--window-size=1280,800" in command
    assert "--enable-unsafe-swiftshader" in command
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
async def test_a_killed_x_server_comes_back_but_takes_the_browser_with_it_which_fails_the_computer_for_good(
        browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [PAGE_URL]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            before = await _screenshot(deployed)
            await _kill(deployed, "^Xvfb :99", oldest=True)
            failed = await _until(deployed, lambda status: "failed" in status)
            x_back = await _until_x(deployed)
        finally:
            report = await teardown_run(context)

    assert before.size == (1280, 800)
    browser = next(check for check in failed["checks"] if check["name"] == "browser")
    assert failed["ready"] is False and browser["final"] is True and "infrastructure failure" in browser["detail"]
    assert x_back
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


@pytest.mark.asyncio
async def test_its_data_plane_signs_it_in_snapshots_it_resets_it_and_loads_a_snapshot_back_whole(browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [DATA_SITE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            await _ready(deployed)
            card = (await _get(deployed, "/.well-known/agent-env.json")).json()
            app = next(child for child in card["children_environments"] if child["name"] == "chromium")

            async def act():
                async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        assert not (await session.call_tool("computer_screenshot", {"space": "frame"})).isError

            await check_browser_data(deployed, browser_computer.id, lambda op: deployed.environment_url
                                     + protocol_v1.find_extension_method(app, BROWSER, op)["endpoint"], act)
        finally:
            report = await teardown_run(context)

    assert not report.failed


@pytest.mark.asyncio
async def test_its_task_steps_set_cookies_and_storage_open_a_tab_behind_and_activate_and_close_it_by_its_url(
        browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [DATA_SITE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            await _ready(deployed)
            card = (await _get(deployed, "/.well-known/agent-env.json")).json()
            app = next(child for child in card["children_environments"] if child["name"] == "chromium")
            await check_browser_steps(deployed, browser_computer.id, lambda op: deployed.environment_url
                                      + protocol_v1.find_extension_method(app, BROWSER, op)["endpoint"])
        finally:
            report = await teardown_run(context)

    assert not report.failed


@pytest.mark.asyncio
async def test_its_contexts_are_leased_by_steps_and_by_agents_each_with_its_own_state_and_end_with_a_reset(
        browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [DATA_SITE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            await _ready(deployed)
            card = (await _get(deployed, "/.well-known/agent-env.json")).json()
            app = next(child for child in card["children_environments"] if child["name"] == "chromium")
            await check_browser_contexts(
                deployed, browser_computer.id,
                lambda op: deployed.environment_url + protocol_v1.find_extension_method(app, BROWSER, op)["endpoint"],
                f"{deployed.environment_url}/svc/chromium", lambda user: _cdp(deployed, user))
        finally:
            report = await teardown_run(context)

    assert not report.failed


@pytest.mark.asyncio
async def test_a_task_pinning_a_version_of_the_browser_passes_preflight_on_it_and_runs_its_steps_through_task_run(
        browser_computer):
    suffix = uuid.uuid4().hex[:8]
    with namespace_routing():
        env = Env.get(browser_computer.id)
        stored, env_id = env.to_dict(), f"{browser_computer.id}-pinned-{suffix}"
        pinned = get_env_store().put_document(type(env).from_dict({**stored, "id": env_id, "version": None}))
        no_browser = {key: value for key, value in stored["declaration"].items() if key != "browser"}
        get_env_store().put_document(type(env).from_dict({**stored, "id": env_id, "version": None,
                                                          "declaration": no_browser}))
        steps = [{"id": "env", "type": "deploy_env", "env_id": env_id, "env_version": pinned.version},
                 {"id": "sign-in", "type": "browser_set_cookies", "env_id": env_id,
                  "cookies": [{"name": "sid", "value": "ada", "url": DATA_SITE, "max_age": 3600}]},
                 {"id": "open", "type": "browser_open_tab", "env_id": env_id, "url": PINNED_PAGE, "background": True},
                 {"id": "front", "type": "browser_activate_tab", "env_id": env_id, "url": PINNED_PAGE},
                 {"id": "close", "type": "browser_close_tab", "env_id": env_id, "url": PINNED_PAGE},
                 {"id": "sign-out", "type": "browser_delete_cookies", "env_id": env_id, "name": "sid"}]
        task_id = f"@local/agentenv-framework/browsers/pinned-steps-{suffix}"
        task = Task.put(id=task_id, steps=Task.from_dict({"id": task_id, "steps": steps}).steps)
        unpinned = Task.from_dict({"id": task_id, "steps": [{**steps[0], "env_version": None}, *steps[1:]]})
        problems, unpinned_problems = task.preflight(), unpinned.preflight()
        context = TaskStepContext()
        try:
            await task.run(context=context)
        finally:
            report = await teardown_run(context)

    assert problems == [], problems
    assert [problem for problem in unpinned_problems if "has no website browser" in problem] == [
        f"step {step!r}: env {env_id!r} has no website browser: browser steps act on a computer env's browser app, "
        "or a browser deployed alone" for step in ("sign-in", "open", "front", "close", "sign-out")], unpinned_problems
    assert not context.metadata.get("failed_steps"), context.metadata.get("failed_steps")
    assert [(deployed.env_id, deployed.env_version) for deployed in context.deployed_envs] == [(env_id, pinned.version)]
    records = context.metadata["browser_steps"]
    assert records["open"]["tab"] == records["front"]["tab"] == records["close"]["tab"], records
    assert records["sign-out"]["deleted"] == 1 and not report.failed, (records, report)


@pytest.mark.asyncio
async def test_each_start_tabs_page_sees_the_windows_size_from_its_first_script_in_order_the_first_active(
        browser_computer):
    urls = [SIZED.format(index) for index in range(3)]
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": urls}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            tabs = await tabs_of(deployed)
        finally:
            report = await teardown_run(context)

    start_tabs = deployed.metadata["computer"]["start_tabs"]["chromium"]
    assert [tab["outcome"] for tab in start_tabs] == ["loaded"] * 3
    assert [tab["id"] for tab in tabs] == [tab["tab"] for tab in start_tabs]
    assert [(tab["title"], tab["active"]) for tab in tabs] == [("1280x800", True), ("1280x800", False),
                                                               ("1280x800", False)]
    assert not report.failed


@pytest.mark.asyncio
async def test_its_data_calls_take_turns_a_server_restart_takes_a_load_and_with_the_browser_gone_they_fail_at_once(
        browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            await _ready(deployed)
            await check_browser_data_under_chaos(
                deployed, lambda pattern: _kill(deployed, pattern, oldest=pattern.startswith("[-]")),
                lambda: _ready(deployed))
        finally:
            report = await teardown_run(context)

    assert not report.failed


#: A form whose password field and text field fill the page's top, for typing into either by a click.
FORM = "data:text/html," + urllib.parse.quote(
    "<title>form</title><body style='margin:0'><input id=p type=password style='position:absolute;left:0;top:0;"
    "width:600px;height:80px'><input id=t style='position:absolute;left:0;top:120px;width:600px;height:80px'>")
#: The secrets the agents send, which the log must never hold, and the text they type that it must.
SECRETS, TYPED = ("hunter2", "hunter3", "s3cret"), ("standing desk", "oak desk")


@pytest.mark.asyncio
async def test_its_log_has_an_entry_for_every_action_over_cdp_page_tools_and_pixels_with_no_secrets(browser_computer):
    """M8.19: one task driven over CDP, page tools and pixels has an entry for every action, with its time, who, tab,
    URL and method or tool, and no secrets; each tab event names the command that caused it."""
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            sent = await _cdp_agent_types((await _cdp(deployed, "ada"))["ws_url"])
            async with streamablehttp_client(deployed.mcp_url, headers={ROLE_HEADER: "bo"}) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    await _tool(session, "computer_navigate", {"url": FORM})
                    await _shown(session, lambda page: page.get("title") == "form")
                    await _tool(session, "computer_click", {"x": 300, "y": 40, "space": "frame"})
                    await _tool(session, "computer_type", {"text": "hunter3"})
                    await _tool(session, "computer_click", {"x": 300, "y": 160, "space": "frame"})
                    await _tool(session, "computer_type", {"text": "oak desk"})
                    await _tool(session, "computer_tabs", {"action": "list"})
                    shot = await session.call_tool("computer_screenshot", {"space": "frame"})
            await asyncio.sleep(1)
            logged = await _browser_log(deployed)
        finally:
            report = await teardown_run(context)

    lines = logged["lines"]
    actions = [line["action"] | {"seq": line["seq"], "at": line["at"]} for line in lines if "action" in line]
    tab_events = [line["tab"] for line in lines if "tab" in line]
    by_ada = [action for action in actions if action["via"] == "cdp" and action["who"] == "ada"]
    by_bo = [action for action in actions if action["via"] == "mcp"]
    text = json.dumps(logged)
    assert not shot.isError and [line["seq"] for line in lines] == sorted(line["seq"] for line in lines)
    assert [action["method"] for action in by_ada] == [method for method, _ in sent], by_ada
    assert all(action["at"] and action["tab"] == sent[0][1] and isinstance(action["url"], str) for action in by_ada)
    assert {action["method"]: action["acts"] for action in by_ada} == {
        "Target.createTarget": True, "Target.attachToTarget": False, "Runtime.evaluate": True, "Input.insertText": True,
        "Network.setCookie": True, "DOM.getDocument": False, "Page.navigate": True, "Target.closeTarget": True}
    typed = [action["params"]["text"] for action in by_ada if action["method"] == "Input.insertText"]
    assert typed == [{"length": 7}, "standing desk"]
    tools = {action["tool"] for action in by_bo}
    assert tools == {"computer_navigate", "computer_click", "computer_type", "computer_tabs", "computer_screenshot"}
    assert all(action["who"] == "bo" and action["at"] and action["op"] and action["call"] for action in by_bo)
    acting = [action for action in by_bo if action["acts"]]
    assert all(action["tab"] and isinstance(action["url"], str) for action in acting), acting
    assert [action["args"]["text"] for action in acting if action["tool"] == "computer_type"] == \
        [{"length": 7}, "oak desk"]
    assert any(action["tool"] == "computer_click" and action["op"] == "page/input" and action["params"]["events"]
               for action in acting)
    assert not any(secret in text for secret in SECRETS) and all(words in text for words in TYPED)
    created = next(action for action in by_ada if action["method"] == "Target.createTarget")
    navigated = next(action for action in by_ada if action["method"] == "Page.navigate")
    closed = next(action for action in by_ada if action["method"] == "Target.closeTarget")
    opened_by_bo = next(action for action in by_bo if action["op"] == "navigate")
    ada_tab = [event for event in tab_events if event["id"] == created["tab"]]
    assert {"event": "opened", "by": "command", "entry": created["seq"], "via": "cdp", "who": "ada"}.items() <= \
        ada_tab[0].items(), ada_tab
    assert any(event["event"] == "navigated" and event.get("entry") == navigated["seq"] for event in ada_tab), ada_tab
    assert {"event": "closed", "by": "command", "entry": closed["seq"]}.items() <= ada_tab[-1].items(), ada_tab
    assert any(event["event"] == "navigated" and event.get("entry") == opened_by_bo["seq"] and event["who"] == "bo"
               for event in tab_events), tab_events
    assert not report.failed


@contextlib.asynccontextmanager
async def _cdp_calls(ws_url: str, tls: ssl.SSLContext | None = None, reply_s: float = 60):
    """A CDP connection to ``ws_url``, trusting ``tls`` for a wss address, as its ``call``: a command's whole reply,
    within ``reply_s``, the events before it skipped."""
    async with connect(ws_url, max_size=None, open_timeout=60, **({"ssl": tls} if tls else {})) as ws:
        ids = itertools.count(1)

        async def call(method: str, params: dict, session: str | None = None) -> dict:
            message_id = next(ids)
            await ws.send(json.dumps({"id": message_id, "method": method, "params": params,
                                      **({"sessionId": session} if session else {})}))
            while (reply := json.loads(await asyncio.wait_for(ws.recv(), reply_s))).get("id") != message_id:
                pass
            return reply

        yield call


async def _cdp_agent_types(ws_url: str) -> list[tuple[str, str]]:
    """A CDP agent, ada, that opens the form in a tab of its own, types a password and a search into it, sets a cookie,
    reads the page, navigates and closes the tab: each command it sent, in order, and the tab, the same for all."""
    sent: list[str] = []
    async with _cdp_calls(ws_url) as cdp:

        async def call(method: str, params: dict, session: str | None = None) -> dict:
            sent.append(method)
            return await cdp(method, params, session)

        target = (await call("Target.createTarget", {"url": FORM}))["result"]["targetId"]
        session = (await call("Target.attachToTarget", {"targetId": target, "flatten": True}))["result"]["sessionId"]
        await asyncio.sleep(1)
        await call("Runtime.evaluate", {"expression": "document.getElementById('p').focus()"}, session)
        await call("Input.insertText", {"text": "hunter2"}, session)
        await call("Runtime.evaluate", {"expression": "document.getElementById('t').focus()"}, session)
        await call("Input.insertText", {"text": "standing desk"}, session)
        await call("Network.setCookie", {"name": "sid", "value": "s3cret", "url": "https://shop.test/"}, session)
        await call("DOM.getDocument", {"depth": 1}, session)
        await call("Page.navigate", {"url": FORM + "%3C!--again--%3E"}, session)
        await asyncio.sleep(1)
        await call("Target.closeTarget", {"targetId": target})
    return [(method, target) for method in sent]


async def _network_log(deployed, after: int = 0) -> dict:
    card = (await _get(deployed, "/.well-known/agent-env.json")).json()
    app = next(child for child in card["children_environments"] if child["name"] == "chromium")
    path = protocol_v1.find_extension_method(app, BROWSER, "network")["endpoint"]
    return (await _get(deployed, path, headers=deployed.control_headers, params={"after": after})).json()


async def _har(deployed) -> bytes:
    """The browser's network log as its data/get exports it, ``network.har``."""
    answer = await protocol_v1.get_data(f"{deployed.environment_url}/svc/chromium", timeout=120,
                                        headers=deployed.control_headers)
    return next(base64.b64decode(part.file.bytes) for part in answer.parts if uploaded_object_path(part) == HAR_FILE)


async def _har_entries(deployed) -> list[dict]:
    return json.loads(await _har(deployed))["log"]["entries"]


async def _browser_log(deployed, after: int = 0) -> dict:
    card = (await _get(deployed, "/.well-known/agent-env.json")).json()
    app = next(child for child in card["children_environments"] if child["name"] == "chromium")
    path = protocol_v1.find_extension_method(app, BROWSER, "log")["endpoint"]
    return (await _get(deployed, path, headers=deployed.control_headers, params={"after": after})).json()


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


async def _until_x(deployed) -> bool:
    """Whether the browser's X server, which supervisord starts again, answers within the wait."""
    deadline = time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        if (await inside.run(deployed, ["sh", "-c", "pgrep -x Xvfb >/dev/null && test -S /tmp/.X11-unix/X99 && "
                                                    "echo up || echo down"], timeout_s=30))["stdout"].strip() == "up":
            return True
        await asyncio.sleep(POLL_S)
    return False


#: Loads its URL through the proxy it's given, as a browser alone does, printing the status, the seconds it took and the
#: first line of the body, or the error.
THROUGH_PROXY = """import sys, time, urllib.request
proxy = urllib.request.ProxyHandler({"http": sys.argv[2], "https": sys.argv[2]})
began = time.monotonic()
try:
    with urllib.request.build_opener(proxy).open(sys.argv[1], timeout=20) as answer:
        print(answer.status, round(time.monotonic() - began), answer.readline().decode().strip())
except Exception as e:
    print("error", getattr(e, "code", ""), e)
"""


async def _through_port(deployed, url: str, proxy: str = "http://127.0.0.1:3129") -> str:
    done = await inside.run(deployed, ["python3", "-c", THROUGH_PROXY, url, proxy], timeout_s=60)
    return done["stdout"].strip()


#: What a CDP agent sees of the tab it navigates: its title, its first words, and whether it finished loading.
PAGE_SEEN = ("JSON.stringify({ready: document.readyState, title: document.title, "
             "text: document.body ? document.body.innerText.slice(0, 200) : ''})")


async def _cdp_agent_loads(ws_url: str, urls: list[str], tls: ssl.SSLContext | None = None) -> dict:
    """What a CDP agent at ``ws_url`` sees navigating a tab of its own to each of ``urls``, one after another:
    ``Page.navigate``'s error, if any, and the page once it has loaded; trusting ``tls`` for a wss address."""
    async with _cdp_calls(ws_url, tls=tls, reply_s=90) as call:
        target = (await call("Target.createTarget", {"url": "about:blank"}))["result"]["targetId"]
        session = (await call("Target.attachToTarget", {"targetId": target, "flatten": True}))["result"]["sessionId"]
        seen = {}
        for url in urls:
            navigated = await call("Page.navigate", {"url": url}, session)
            page, deadline = {}, time.monotonic() + 30
            while page.get("ready") != "complete" and time.monotonic() < deadline:
                await asyncio.sleep(1)
                evaluated = await call("Runtime.evaluate", {"expression": PAGE_SEEN, "returnByValue": True}, session)
                page = json.loads(evaluated.get("result", {}).get("result", {}).get("value") or "{}")
            seen[url] = {"error": navigated.get("result", {}).get("errorText") or navigated.get("error", {}).get(
                "message"), **page}
        return seen


async def _container_of(deployed) -> str:
    """The Docker container ``deployed``'s computer runs in, on this machine's local provider."""
    return (await LocalSandboxProvider().get_sandbox(deployed.sandbox_id)).container_name


async def _serve_pid(deployed) -> str:
    return (await inside.run(deployed, ["pgrep", "-f", "[s]ite_proxy.py serve"], timeout_s=30))["stdout"].strip()


def _as_root(container: str, script: str) -> str:
    return subprocess.run(["docker", "exec", "-u", "0", container, "sh", "-c", script], capture_output=True,
                          text=True).stdout


@pytest.mark.asyncio
async def test_routing_sites_it_goes_out_only_through_the_forward_port_which_the_agent_cant_stop_and_comes_back(
        browser_computer):
    with host_pages() as port, namespace_routing():
        params = {"site_routing": True,
                  "sites": [{"name": "shop", "url": f"http://127.0.0.1:{port}/website/shop/", "domains": ["shop.test"]},
                            {"name": "notes", "url": f"http://127.0.0.1:{port}/website/notes/",
                             "domains": ["notes.localhost"]}],
                  "app_settings": {"chromium": {"start_urls": ["https://shop.test/", "http://notes.localhost/"]}}}
        deployed = await Env.get(browser_computer.id).deploy(env_params=params)
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            tabs = await tabs_of(deployed)
            named = (await inside.run(deployed, ["sh", "-c", "echo $AGENTENV_SITE_PROXY"], timeout_s=30))["stdout"]
            pages = {url: await _through_port(deployed, url) for url in (
                "https://shop.test/alpha", "http://notes.localhost/bravo", "http://shop.test/charlie?x=1",
                "https://shop.test/events", "https://example.com/", "http://127.0.0.1:18765/",
                "https://127.0.0.1:18765/")}
            container = await _container_of(deployed)
            logged = _as_root(container, "cat /etc/agentenv/sites/site-proxy.log")
            serving = await _serve_pid(deployed)
            stopped = await inside.run(deployed, ["pkill", "-f", "[s]ite_proxy.py serve"], timeout_s=30)
            after_stop = await _through_port(deployed, "https://shop.test/alpha")
            still = await _serve_pid(deployed)
            _as_root(container, "pkill -9 -f '[s]ite_proxy.py serve'")
            deadline, back = time.monotonic() + 30, ""
            while time.monotonic() < deadline and not back.startswith("200 "):
                await asyncio.sleep(1)
                back = await _through_port(deployed, "https://shop.test/alpha")
            restarted = await _serve_pid(deployed)
            await ResetEnvTaskStep(id="reset", version=None, env_id=browser_computer.id,
                                   fail_task_on_error=True).execute(context)
            after_reset = await _through_port(deployed, "https://shop.test/alpha")
            refused_after_reset = await _through_port(deployed, "https://example.com/")
            named_after_reset = (await inside.run(deployed, ["sh", "-c", "echo $AGENTENV_SITE_PROXY"],
                                                  timeout_s=30))["stdout"]
            tabs_after_reset = await tabs_of(deployed)
        finally:
            report = await teardown_run(context)

    start_tabs = deployed.metadata["computer"]["start_tabs"]["chromium"]
    assert [(tab["url"], tab["site"], tab["outcome"]) for tab in start_tabs] == [
        ("https://shop.test/", "shop", "loaded"), ("http://notes.localhost/", "notes", "loaded")], start_tabs
    assert [(tab["url"], tab["title"]) for tab in tabs] == [("https://shop.test/landed", "landed"),
                                                            ("http://notes.localhost/", "notes")], tabs
    assert named.strip() == "http://127.0.0.1:3129"
    assert deployed.metadata["computer"]["sites"]["forward_port"] == {
        "proxy": "http://127.0.0.1:3129", "url": "https://shop.test/", "status": 200}
    assert pages["https://shop.test/alpha"].startswith("200 ") and pages["https://shop.test/alpha"].endswith(
        "<title>alpha</title>"), pages
    assert pages["http://notes.localhost/bravo"].endswith("<title>bravo</title>"), pages
    assert pages["http://shop.test/charlie?x=1"].endswith("<title>charlie?x=1</title>"), pages
    status, seconds, line = pages["https://shop.test/events"].split(" ", 2)
    assert (status, line) == ("200", "data: first") and int(seconds) < 15, pages
    assert all(pages[url].startswith("error") and "403" in pages[url] for url in (
        "https://example.com/", "http://127.0.0.1:18765/", "https://127.0.0.1:18765/")), pages
    refused = [json.loads(line)["refused"] for line in logged.splitlines() if line.startswith('{"refused"')]
    assert {"method": "CONNECT", "host": "example.com:443", "reason": "no site of this computer"} in refused, logged
    assert {"method": "GET", "host": "127.0.0.1:18765", "reason": "no site of this computer"} in refused, logged
    assert stopped["exit_code"] != 0 and still == serving and after_stop.startswith("200 "), (stopped, after_stop)
    assert back.startswith("200 ") and restarted and restarted != serving, (back, restarted)
    assert after_reset.startswith("200 ") and after_reset.endswith("<title>alpha</title>"), after_reset
    assert refused_after_reset.startswith("error") and "403" in refused_after_reset, refused_after_reset
    assert named_after_reset.strip() == "http://127.0.0.1:3129"
    assert [(tab["url"], tab["title"]) for tab in tabs_after_reset] == [("https://shop.test/landed", "landed"),
                                                                        ("http://notes.localhost/", "notes")]
    assert not report.failed


@pytest.mark.asyncio
async def test_routing_sites_beside_live_ones_its_forward_port_takes_the_live_hosts_straight_to_the_real_site(
        browser_computer):
    live = ("github.com", "*.githubassets.com")
    with host_pages() as port, namespace_routing():
        params = {"site_routing": True,
                  "sites": [{"name": "shop", "url": f"http://127.0.0.1:{port}/website/shop/", "domains": ["shop.test"]}],
                  "app_settings": {"chromium": {"start_urls": ["https://github.com/"]}}}
        deployed = await Env.get(browser_computer.id).deploy(env_params=params, live_hosts=live)
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            pages = {url: await _through_port(deployed, url) for url in (
                "https://github.com/", "http://github.com/", "https://github.githubassets.com/favicons/favicon.svg",
                "https://shop.test/alpha", "https://example.com/")}
            tabs = await tabs_of(deployed)
            entries = await _har_entries(deployed)
        finally:
            report = await teardown_run(context)

    assert deployed.metadata["computer"]["live_hosts"] == list(live)
    assert all(pages[url].startswith("200 ") for url in (
        "https://github.com/", "http://github.com/", "https://github.githubassets.com/favicons/favicon.svg",
        "https://shop.test/alpha")), pages
    assert pages["https://example.com/"].startswith("error") and "403" in pages["https://example.com/"], pages
    assert any(tab["url"].startswith("https://github.com/") and "GitHub" in (tab.get("title") or "") for tab in tabs), tabs
    by_url = {entry["request"]["url"]: entry for entry in entries}
    assert by_url["https://shop.test/alpha"]["response"]["content"]["text"] == "<title>alpha</title>", by_url.keys()
    assert by_url["http://github.com/"]["response"]["status"] in (200, 301, 308) and \
        by_url["http://github.com/"]["request"]["method"] == "GET"
    github = [entry for entry in entries if entry["request"]["url"] == "https://github.com:443/" and "_open" not in entry]
    assert any(entry["request"]["method"] == "CONNECT" and entry["response"]["status"] == 200 and
               entry["_tunnel"]["bytes_received"] > 0 for entry in github), github
    assert by_url["https://example.com:443/"]["_refused"] == {"reason": "no site of this computer"}
    assert not report.failed


#: A shop's pages under its route: a form that signs in by POST, answered with a redirect that sets a cookie, onto a
#: cart whose script saves an item as JSON, answered gzipped, then uploads a file as multipart, each answer saying in
#: the title what came of it.
SHOP_FORM = ("<title>form</title><form method=post action=/login><input name=user value=amy>"
             "<input name=note value='caf\u00e9 & tea'><button>go</button></form>").encode()
SHOP_CART = b"""<title>cart</title><script>
fetch('/api/items', {method: 'POST', headers: {'Content-Type': 'application/json'},
                     body: JSON.stringify({name: 'desk', qty: 2})})
  .then(answer => answer.json()).then(item => {
    const form = new FormData();
    form.append('title', 'Trip');
    form.append('file', new Blob(['notes'], {type: 'text/plain'}), 'a.txt');
    return fetch('/upload', {method: 'POST', body: form}).then(() => document.title = 'saved ' + item.id);
  });
</script>"""
SAVED = json.dumps({"id": 7, "name": "desk"}).encode()
#: What WebArena-Verified's own loader reads of each entry of a HAR it grades, as JSON.
WEBARENA_READS = """import json, sys
from pathlib import Path
from webarena_verified.types.tracing import NetworkTrace
trace = NetworkTrace.from_har(Path(sys.argv[1]))
print(json.dumps([{"method": e.http_method, "url": e.url, "status": e.request_status, "post_data": e.post_data,
                   "content": e.response_content, "redirect": e.redirect_url, "cookies": e.response_cookies,
                   "navigation": e.is_navigation_event, "referer": e.referer} for e in trace.evaluation_events]))
"""
WEBARENA_VERIFIED = "webarena-verified==1.2.3"


@contextlib.contextmanager
def _shop_pages():
    """``SHOP_FORM`` and the rest served on every address of this machine under ``/website/shop/``, at the port it
    yields."""

    class Shop(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, body: bytes = b"", **headers) -> None:
            self.send_response(status)
            for name, value in {"Content-Length": str(len(body)), **headers}.items():
                self.send_header(name.replace("_", "-"), value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            pages = {"/website/shop/form": SHOP_FORM, "/website/shop/cart": SHOP_CART}
            if (path := self.path.partition("?")[0]) in pages:
                self._send(200, pages[path], Content_Type="text/html; charset=utf-8")
            else:
                self._send(404)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/website/shop/login":
                self._send(303, Location="/website/shop/cart", Set_Cookie="sid=s2; Path=/website/shop/; HttpOnly")
            elif self.path == "/website/shop/api/items":
                self._send(201, gzip.compress(SAVED), Content_Type="application/json", Content_Encoding="gzip")
            else:
                self._send(204)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), Shop)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


async def _cdp_agent_shops(ws_url: str) -> str:
    """A CDP agent that opens the shop's form, signs in with a click and waits on the cart to save: the title it ends
    on."""
    async with connect(ws_url, max_size=None, open_timeout=60) as ws:
        ids = itertools.count(1)

        async def call(method: str, params: dict, session: str | None = None) -> dict:
            message_id = next(ids)
            await ws.send(json.dumps({"id": message_id, "method": method, "params": params,
                                      **({"sessionId": session} if session else {})}))
            while (reply := json.loads(await asyncio.wait_for(ws.recv(), 90))).get("id") != message_id:
                pass
            return reply

        async def title() -> str:
            evaluated = await call("Runtime.evaluate", {"expression": "document.title", "returnByValue": True}, session)
            return evaluated.get("result", {}).get("result", {}).get("value") or ""

        target = (await call("Target.createTarget", {"url": "https://shop.test/form"}))["result"]["targetId"]
        session = (await call("Target.attachToTarget", {"targetId": target, "flatten": True}))["result"]["sessionId"]
        deadline = time.monotonic() + 60
        while await title() != "form" and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
        await call("Runtime.evaluate", {"expression": "document.querySelector('button').click()",
                                        "userGesture": True}, session)
        shown = ""
        while not shown.startswith("saved") and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            shown = await title()
        return shown


@pytest.mark.asyncio
async def test_routing_sites_its_network_har_has_what_webarena_verifieds_grader_reads(browser_computer, tmp_path):
    """M8.20: an agent signs in to a site by a form, follows the redirect and saves an item as JSON and a file as
    multipart; WebArena-Verified's own loader, reading the network.har data/get exports, sees each request's method,
    URL, form, JSON and multipart fields and referer, each answer's status, redirect, cookie and decoded JSON, and the
    page loads as navigations."""
    if shutil.which("uvx") is None:
        pytest.skip(f"uvx runs {WEBARENA_VERIFIED} in an environment of its own")
    with _shop_pages() as port, namespace_routing():
        params = {"site_routing": True,
                  "sites": [{"name": "shop", "url": f"http://127.0.0.1:{port}/website/shop/", "domains": ["shop.test"]}],
                  "app_settings": {"chromium": {"start_urls": [DATA_PAGE]}}}
        deployed = await Env.get(browser_computer.id).deploy(env_params=params)
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            shown = await _cdp_agent_shops((await _cdp(deployed, "ada"))["ws_url"])
            await asyncio.sleep(1)
            exported = await _har(deployed)
            lines = (await _network_log(deployed))["lines"]
        finally:
            report = await teardown_run(context)

    (tmp_path / HAR_FILE).write_bytes(exported)
    read = subprocess.run(["uvx", "--from", WEBARENA_VERIFIED, "python", "-I", "-c", WEBARENA_READS,
                           str(tmp_path / HAR_FILE)], capture_output=True, text=True, timeout=600)
    assert read.returncode == 0, read.stderr[-3000:]
    events = {(event["method"], event["url"]): event for event in json.loads(read.stdout)}
    assert shown == "saved 7", shown
    form, login = events[("GET", "https://shop.test/form")], events[("POST", "https://shop.test/login")]
    cart = events[("GET", "https://shop.test/cart")]
    saved, uploaded = events[("POST", "https://shop.test/api/items")], events[("POST", "https://shop.test/upload")]
    assert (form["status"], form["navigation"]) == (200, True), form
    assert (login["post_data"], login["status"], login["redirect"], login["cookies"]) == (
        {"user": "amy", "note": "caf\u00e9 & tea"}, 303, "/cart", {"sid": "s2"}), login
    assert (cart["status"], cart["navigation"], cart["referer"]) == (200, True, "https://shop.test/form"), cart
    assert (saved["post_data"], saved["status"], saved["content"]) == ({"name": "desk", "qty": 2}, 201,
                                                                       {"id": 7, "name": "desk"}), saved
    assert (uploaded["post_data"], uploaded["status"]) == ({"title": "Trip", "file": "notes"}, 204), uploaded
    assert [line["seq"] for line in lines] == list(range(1, len(lines) + 1))
    assert not report.failed


@pytest.mark.asyncio
async def test_alone_on_the_open_web_it_loads_a_public_site_through_its_proxy_and_an_agents_tab_and_logs_each_connection(
        browser_computer):
    public, nowhere = "https://github.com/", "https://agentenv-no-such-host.invalid/"
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [DATA_PAGE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            through = {url: await _through_port(deployed, url, "http://127.0.0.1:3128") for url in (public, nowhere)}
            seen = await _cdp_agent_loads((await _cdp(deployed, "ada"))["ws_url"], [public])
            logged = _as_root(await _container_of(deployed), "cat /var/log/agentenv/site-proxy.log")
            lines = (await _network_log(deployed))["lines"]
            exported = await _har_entries(deployed)
        finally:
            report = await teardown_run(context)

    assert through[public].startswith("200 ") and "502" in through[nowhere], through
    assert seen[public]["title"].startswith("GitHub") and not seen[public]["error"], seen
    entries = [json.loads(line) for line in logged.splitlines() if line.startswith("{")]
    github = [entry["connection"] for entry in entries if entry.get("connection", {}).get("host") == "github.com"]
    assert github and all(entry["bytes_received"] > 0 for entry in github), logged[-2000:]
    assert any(entry.get("failed", {}).get("host") == "agentenv-no-such-host.invalid" for entry in entries), logged[-2000:]
    assert [line["seq"] for line in lines] == list(range(1, len(lines) + 1)), lines[:5]
    tunnels = [entry for entry in exported if entry["request"]["url"] == "https://github.com:443/"]
    assert tunnels and all(entry["request"]["method"] == "CONNECT" and entry["_tunnel"]["bytes_received"] > 0
                           and entry["serverIPAddress"] for entry in tunnels if "_open" not in entry), tunnels
    assert any(entry["request"]["url"] == "https://agentenv-no-such-host.invalid:443/" and
               entry.get("_failed", {}).get("reason") == "doesn't resolve" for entry in exported), exported[-5:]
    assert [entry["startedDateTime"] for entry in exported] == sorted(entry["startedDateTime"] for entry in exported)
    assert not report.failed


#: How long after the browser is ready its proxy log is read: past the component updater's check a minute after launch.
QUIET_S = 75.0
#: The one request the browser makes of its own that nothing stops: a Google account check at launch.
ACCOUNT_CHECK = "accounts.google.com"


@pytest.mark.asyncio
async def test_alone_it_makes_no_request_of_its_own_its_proxy_seeing_only_its_page_until_well_after_it_is_ready(
        browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy()
        ready = time.time()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_SITE, "150"], timeout_s=180))
            await asyncio.sleep(SERVE_START_S)
            seen = await _cdp_agent_loads((await _cdp(deployed, "ada"))["ws_url"], [SITE])
            await asyncio.sleep(max(0.0, ready + QUIET_S - time.time()))
            container = await _container_of(deployed)
            # Chromium ended ends each connection it holds open, which its proxy logs as it ends.
            _as_root(container, "pkill -f '[c]hromium/chrome/chrome'; sleep 3")
            logged = _as_root(container, "cat /var/log/agentenv/site-proxy.log")
            serving.cancel()
        finally:
            report = await teardown_run(context)

    assert seen[SITE]["title"] == "Home" and not seen[SITE]["error"], seen
    entries = [(kind, entry) for line in logged.splitlines() if line.startswith("{")
               for kind, entry in json.loads(line).items() if kind in ("connection", "failed")]
    own = [(kind, entry) for kind, entry in entries if (entry["host"], entry["port"]) != ("127.0.0.1", 8771)]
    assert any(kind == "connection" and (entry["host"], entry["port"]) == ("127.0.0.1", 8771)
               for kind, entry in entries), logged
    assert all(entry["host"] == ACCOUNT_CHECK for _, entry in own), own
    assert all(_epoch(entry["started"]) <= ready for _, entry in own), (ready, own)
    assert not report.failed


def _epoch(stamp: str) -> float:
    return calendar.timegm(time.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ"))


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
            await _kill(deployed, "[s]erve_app")
            await _ready(deployed)
            relaunched = await _clients(cdp_clients, "together", ada)
            await protocol_v1.reset_data(f"{deployed.environment_url}/svc/chromium", headers=deployed.control_headers)
            after_data_reset = await _clients(cdp_clients, "refused", ada, wrong=False)
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
    assert "error" in after_data_reset["playwright"] and "error" in after_data_reset["puppeteer"], after_data_reset
    assert "error" in after_reset["playwright"] and "error" in after_reset["puppeteer"], after_reset
    assert fresh != ada and not report.failed


@pytest.mark.asyncio
async def test_playwright_mcps_tools_drive_its_page_hide_the_code_tools_and_come_back_after_a_kill(browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [PAGE_URL]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        serving = None
        try:
            viewport = (await _screenshot(deployed)).size
            listed = await _tool_names(deployed)
            typed = await _mcp(deployed, _type_hello)
            state = await _state(deployed, lambda s: s["v"] == "hello")
            after_typing = (await _screenshot(deployed)).size
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_SITE, "900"], timeout_s=960))
            await asyncio.sleep(SERVE_START_S)
            confirmed = await _mcp(deployed, _confirm_ask)
            logged = (await _browser_log(deployed))["lines"]
            await deployed.invoke(ENABLE_TOOL_EXTENSION_URI, ENABLE, {"role": "default", "tools": ["browser_evaluate"]})
            shown, evaluated = await _tool_names(deployed), await _mcp(deployed, _evaluate_title)
            running = (await inside.run(deployed, ["sh", "-c", "ps -eo args | grep -m1 '[@]playwright/mcp/cli.js'"],
                                        timeout_s=30))["stdout"].split()
            installed = json.loads((await inside.run(deployed, ["python3", "-c", LISTED_TOOLS, *running[:2]],
                                                     timeout_s=120))["stdout"])
            await _kill(deployed, "[@]playwright/mcp/cli.js")
            after_its_kill = await _mcp(deployed, _snapshot)
            await _kill(deployed, "[c]omputer.serve driver.py:PageDriver")
            await _ready(deployed)
            after_driver_kill = await _mcp(deployed, _snapshot)
            count = (await inside.run(deployed, ["sh", "-c", "pgrep -fc '[@]playwright/mcp/cli.js'"],
                                      timeout_s=30))["stdout"].strip()
            record = await _record(deployed)
        finally:
            if serving is not None:
                serving.cancel()
            report = await teardown_run(context)

    playwright = {t["name"] for t in json.loads(PLAYWRIGHT_MCP_TOOLS.read_text())["tools"]}
    assert playwright - {"browser_evaluate", "browser_run_code_unsafe"} <= set(listed), listed
    assert not {"browser_evaluate", "browser_run_code_unsafe"} & set(listed) and "computer_screenshot" in listed
    assert "hello" in typed and state["v"] == "hello" and after_typing == viewport
    assert "confirmed true" in confirmed, confirmed
    entries = [line for line in logged if "action" in line]
    calls = {line["action"]["tool"]: line["seq"] for line in entries
             if line["action"]["via"] == "mcp" and line["action"]["tool"].startswith("browser_")}
    typing = next(line["action"] for line in entries if line["action"].get("tool") == "browser_type")
    sent = {line["action"]["method"]: line["action"].get("for") for line in entries
            if line["action"]["via"] == "cdp" and line["action"]["method"] in ("Page.navigate", "Page.handleJavaScriptDialog")}
    clicks = {line["action"].get("for") for line in entries if line["action"].get("method") == "Input.dispatchMouseEvent"}
    navigated = next(line["tab"] for line in logged if line.get("tab", {}).get("url") == SITE)
    assert {"browser_snapshot", "browser_type", "browser_navigate", "browser_click", "browser_handle_dialog"} <= set(calls)
    assert (typing["who"], typing["ok"], typing["args"]["text"]) == ("default", True, "hello")
    assert sent == {"Page.navigate": calls["browser_navigate"], "Page.handleJavaScriptDialog": calls["browser_handle_dialog"]}
    assert clicks == {calls["browser_click"]}, clicks
    assert (navigated["who"], navigated["for"]) == ("playwright-mcp", calls["browser_navigate"]), navigated
    assert "browser_evaluate" in shown and "browser_run_code_unsafe" not in shown and "confirmed true" in evaluated
    assert installed == json.loads(PLAYWRIGHT_MCP_TOOLS.read_text())["tools"]
    assert "Page URL" in after_its_kill and "Page URL" in after_driver_kill and count == "1"
    assert "playwright-mcp" in {connection["user"] for connection in record["cdp_connections"]}
    assert not report.failed


@pytest.mark.asyncio
async def test_a_browser_recording_frames_each_burst_of_an_agents_actions_keyed_to_its_log_and_ends_on_its_last_page(
        browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [PAGE_URL]}}})
        context = TaskStepContext(instance_id=f"run-{uuid.uuid4().hex[:8]}", deployed_envs=[deployed])
        try:
            context = await StartBrowserRecordingTaskStep(id="start", version=None, env_id=browser_computer.id,
                                                          name="run").execute(context)
            await _mcp(deployed, _type_hello)
            await _mcp(deployed, lambda session: _called(session, "browser_navigate", {"url": DATA_PAGE}))
            context = await EndBrowserRecordingTaskStep(id="end", version=None, name="run").execute(context)
        finally:
            report = await teardown_run(context)
            grant_server(None, None).close()
        entry = context.metadata[BROWSER_RECORDINGS]["run"]
        universe = Artifact.get(entry["artifact"]["id"], version=entry["artifact"]["version"])
        objects = get_config().get_object_store()
        files = {name: objects.get(Artifact.get(artifact_id).object_url)
                 for name, artifact_id in universe.file_artifact_ids.items()}

    manifest = json.loads(files["manifest.json"])
    log = [json.loads(line) for line in files["log.jsonl"].decode().splitlines()]
    calls = {line["action"]["tool"]: line["seq"] for line in log if line.get("action", {}).get("via") == "mcp"}
    frames = manifest["frames"]
    followed = {seq for frame in frames for seq in frame["after"]}
    assert not report.failed and (entry["state"], entry["missing_frames"]) == ("ended", 0), entry
    assert frames[0]["kind"] == "first" and frames[-1]["kind"] == "last" and "action" in {f["kind"] for f in frames}
    assert {calls["browser_type"], calls["browser_navigate"]} <= followed, (calls, frames)
    assert frames[-1]["url"].startswith("data:text/html,") and "Home" in urllib.parse.unquote(frames[-1]["url"])
    assert all(files[frame["path"].replace("/", "-")][:2] == b"\xff\xd8" for frame in frames)


@pytest.mark.asyncio
async def test_a_browser_recording_frames_a_cdp_agents_own_tab_behind_the_active_one(browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [PAGE_URL]}}})
        context = TaskStepContext(instance_id=f"run-{uuid.uuid4().hex[:8]}", deployed_envs=[deployed])
        try:
            context = await StartBrowserRecordingTaskStep(id="start", version=None, env_id=browser_computer.id,
                                                          name="behind").execute(context)
            tab = await _cdp_agent_acts_behind((await _cdp(deployed, "pi"))["ws_url"])
            active = [t for t in (await _browser_op(deployed, "tabs/list", {}))["tabs"] if t.get("active")]
            context = await EndBrowserRecordingTaskStep(id="end", version=None, name="behind",
                                                        fail_task_on_error=False).execute(context)
        finally:
            report = await teardown_run(context)
            grant_server(None, None).close()
        entry = context.metadata[BROWSER_RECORDINGS]["behind"]
        universe = Artifact.get(entry["artifact"]["id"], version=entry["artifact"]["version"])
        manifest = json.loads(get_config().get_object_store().get(
            Artifact.get(universe.file_artifact_ids["manifest.json"]).object_url))

    frames = manifest["frames"]
    assert active and active[0]["id"] != tab, active
    assert not report.failed and (entry["state"], entry["missing_frames"]) == ("ended", 0), frames
    assert [f["tab"] for f in frames if f["kind"] == "action"] and frames[-1]["tab"] == tab, frames
    assert "Own two" in urllib.parse.unquote(frames[-1]["url"]), frames[-1]


#: The pages a CDP agent moves between in a tab of its own behind the active one.
BEHIND = ("data:text/html,<title>Own one</title><h1>Own one</h1>", "data:text/html,<title>Own two</title><h1>Own two</h1>")


async def _cdp_agent_acts_behind(ws_url: str) -> str:
    """A CDP agent, pi, that opens a tab of its own behind the active one and never brings it to the front: it
    navigates it, changes its page and navigates it again. The tab."""
    async with _cdp_calls(ws_url) as call:
        target = (await call("Target.createTarget", {"url": "about:blank", "background": True}))["result"]["targetId"]
        session = (await call("Target.attachToTarget", {"targetId": target, "flatten": True}))["result"]["sessionId"]
        await call("Page.navigate", {"url": BEHIND[0]}, session)
        await asyncio.sleep(2)
        await call("Runtime.evaluate", {"expression": "document.body.append('typed')"}, session)
        await asyncio.sleep(2)
        await call("Page.navigate", {"url": BEHIND[1]}, session)
        await asyncio.sleep(2)
    return target


async def _mcp(deployed, work):
    """What ``work`` makes of one MCP session on the computer, as a role-less agent opens one."""
    async with streamablehttp_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await work(session)


async def _called(session: ClientSession, tool: str, arguments: dict) -> str:
    result = await session.call_tool(tool, arguments)
    text = "\n".join(part.text for part in result.content if part.type == "text")
    assert not result.isError, f"{tool}: {text}"
    return text


async def _snapshot(session: ClientSession) -> str:
    return await _called(session, "browser_snapshot", {})


async def _type_hello(session: ClientSession) -> str:
    ref = re.search(r"textbox \[ref=(\w+)\]", await _snapshot(session)).group(1)
    return await _called(session, "browser_type", {"target": ref, "text": "hello"})


async def _confirm_ask(session: ClientSession) -> str:
    await _called(session, "browser_navigate", {"url": SITE})
    ref = re.search(r'button "ask" \[ref=(\w+)\]', await _snapshot(session)).group(1)
    await session.call_tool("browser_click", {"target": ref})
    await _called(session, "browser_handle_dialog", {"accept": True})
    return await _snapshot(session)


async def _evaluate_title(session: ClientSession) -> str:
    return await _called(session, "browser_evaluate", {"function": "() => document.title"})


@pytest.mark.asyncio
async def test_behind_a_front_that_ends_tls_an_address_under_the_origin_named_is_https_and_wss_and_clients_connect(
        browser_computer, cdp_clients):
    ca = local_ca()
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [SITE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            serving = asyncio.ensure_future(inside.run(deployed, ["python3", "-c", SERVE_SITE, "900"], timeout_s=960))
            await asyncio.sleep(SERVE_START_S)
            driver = urllib.parse.urlsplit(deployed.environment_url)
            async with tls_front(ca, driver.hostname, driver.port) as front:
                named = await _cdp_through(front, deployed, ca, "ada", origin=front)
                unnamed = await _cdp_through(front, deployed, ca, "bob")
                discovered = await _discovered(named["http_url"], ca)
                together = await _clients(cdp_clients, "together", named, ca=ca)
            serving.cancel()
        finally:
            report = await teardown_run(context)

    host = front.removeprefix("https://")
    assert named["http_url"].startswith(f"{front}/svc/chromium/cdp/"), named
    assert named["ws_url"].startswith(f"wss://{host}/svc/chromium/cdp/") and discovered == named["ws_url"], named
    assert unnamed["ws_url"].startswith(f"ws://{host}/svc/chromium/cdp/"), unnamed
    for client in ("playwright_http", "puppeteer_ws", "playwright_ws"):
        assert together[client].get("ok", {}).get("title") == "Home", together
    assert not report.failed


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
        finally:
            report = await teardown_run(context)

    assert [title for _, title in (first, after_server, after_driver)] == ["Home"] * 3
    assert first[0] == after_server[0] == after_driver[0]
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


async def _clients(folder: Path, check: str, address: dict, wrong: bool = True, ca: LocalCA | None = None) -> dict:
    """What the CDP clients saw doing ``check`` at ``address``; for ``refused``, with its token changed unless not
    ``wrong``; trusting ``ca`` too when given."""
    argv = ["node", "connect.mjs", check, address["http_url"], address["ws_url"], SITE] + ([] if wrong else ["exact"])
    env = {**os.environ, "NODE_EXTRA_CA_CERTS": str(ca.cert_path)} if ca else None
    process = await asyncio.create_subprocess_exec(*argv, cwd=folder, env=env, stdout=asyncio.subprocess.PIPE,
                                                   stderr=asyncio.subprocess.PIPE)
    out, err = await asyncio.wait_for(process.communicate(), 300)
    assert process.returncode == 0, err.decode()[-2000:]
    return json.loads(out.decode().strip().splitlines()[-1])


async def _cdp_through(front: str, deployed, ca: LocalCA, user: str, origin: str | None = None) -> dict:
    """``user``'s CDP address, asked of the browser through ``front`` as agentenv asks, with the control token,
    naming ``origin`` or not."""
    async with httpx.AsyncClient(timeout=120, verify=_trusting(ca)) as http:
        card = (await http.get(f"{front}/.well-known/agent-env.json")).json()
        app = next(child for child in card["children_environments"] if child["name"] == "chromium")
        path = protocol_v1.find_extension_method(app, BROWSER, "cdp")["endpoint"]
        answer = await http.post(f"{front}{path}", json={"user": user, **({"origin": origin} if origin else {})},
                                 headers=deployed.control_headers)
    assert answer.status_code == 200, answer.text
    return answer.json()


async def _discovered(http_url: str, ca: LocalCA) -> str:
    """The WebSocket ``json/version`` under ``http_url`` names."""
    async with httpx.AsyncClient(timeout=60, verify=_trusting(ca)) as http:
        return (await http.get(f"{http_url}json/version")).json()["webSocketDebuggerUrl"]


def _trusting(ca: LocalCA) -> ssl.SSLContext:
    return ssl.create_default_context(cafile=str(ca.cert_path))


async def _get(deployed, path: str, **kwargs) -> httpx.Response:
    async with httpx.AsyncClient(timeout=30) as http:
        return await http.get(f"{deployed.environment_url}{path}", **kwargs)


# --- a multi env's website browser (browser_envs) ---

MULTI_SITE = "https://shop.test/api/health"
MULTI_CHROMIUM = {"sandbox_type": "local",
            "env_params": {"form": "container", "app_settings": {"chromium": {"start_urls": [MULTI_SITE]}}}}
#: How often a multi env's browser is asked through the gateway whether it answers again.
ANSWER_POLL_S = 2.0


@pytest.fixture(scope="module")
def shop(browser_computer):
    """The gateway's and service-db's envs, and a website whose backend is the small agentenv website, named
    shop.test, under a local id, since a multi env naming chromium's local id must be local too."""
    with namespace_routing():
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        return WebsiteEnv.put(id=f"@local/~/browsing-{uid}/shop", environment_name="shop", domains=["shop.test"],
                              backend_docker_image_artifact=_image("agentenv_website", f"browsing-shop-{uid}"))


@pytest.fixture
def made(monkeypatch) -> list[str]:
    """The ids of the local sandboxes this test's process makes, so it finds and counts only its own."""
    return local_sandboxes.recorded_sandbox_ids(monkeypatch)


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
            opened = await session.call_tool("chromium_computer_navigate", {"url": MULTI_SITE})
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
        await asyncio.sleep(ANSWER_POLL_S)
    return False


async def _cdp_at_gateway(deployed, browser: DeployedEnv, *, at: str | None = None, user: str = "ada",
                          origin: str | None = None, ca: LocalCA | None = None) -> dict:
    """The CDP address the browser gives ``user``, asked through the gateway's ``/svc/chromium/`` route, at ``at``, a
    front before it, else its own address, with the browser's own control token, as agentenv asks a computer child,
    naming ``origin`` or not."""
    app = next(child for child in browser.environment_card["children_environments"] if child["name"] == "chromium")
    path = protocol_v1.find_extension_method(app, BROWSER, "cdp")["endpoint"]
    async with httpx.AsyncClient(timeout=180, verify=_trusting(ca) if ca else True) as http:
        answer = await http.post(f"{at or deployed.gateway_url}/svc/chromium{path}",
                                 json={"user": user, **({"origin": origin} if origin else {})},
                                 headers=browser.control_headers)
    assert answer.status_code == 200, answer.text
    return answer.json()


def _browser_universe(env_id: str, cookies: list[dict]) -> EnvironmentUniverseArtifact:
    """An environment universe whose chromium environment holds a browser state setting ``cookies``."""
    run = uuid.uuid4().hex[:8]
    state = FileArtifact.put_bytes(id=f"{env_id}-browser-{run}", description="multi env browser e2e",
                                   filename="browser-state.json", content=json.dumps({"cookies": cookies}).encode(),
                                   content_type="application/json")
    environment = EnvironmentArtifact.put(id=f"{env_id}-browser-env-{run}", environment_name="chromium",
                                          file_artifacts={"browser-state.json": state})
    return EnvironmentUniverseArtifact.put(id=f"{env_id}-universe-{run}", environment_artifacts=[environment])


async def _browser_op(browser: DeployedEnv, op: str, params: dict) -> dict:
    """``op``'s answer from the browser app on ``browser``, a multi env's browser computer, through its own record."""
    app = next(child for child in browser.environment_card["children_environments"] if child["name"] == "chromium")
    assert find_extension_method(app, BROWSER, op) is not None, op
    return await invoke_extension(browser.environment_url, app, BROWSER, params, 60, method=op,
                                  headers=browser.control_headers)


async def _discovery(http_url: str) -> dict:
    async with httpx.AsyncClient(timeout=60) as http:
        answer = await http.get(f"{http_url.rstrip('/')}/json/version")
    assert answer.status_code == 200, answer.text
    return answer.json()


@pytest.mark.asyncio
async def test_chromium_as_a_multi_envs_browser_runs_on_its_own_sandbox_reached_through_the_gateway(
        browser_computer, shop, made):
    with namespace_routing():
        office = _office(shop, [Env.get(browser_computer.id)])
        step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                                 child_envs={"chromium": MULTI_CHROMIUM})
        assert step.preflight() == []
        context = TaskStepContext()
        try:
            [deployed] = (await step.execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            stated = {child.environment_name for child in protocol_v1.children(deployed.environment_card)}
            driven = await _drive(deployed)
            address = await _cdp_at_gateway(deployed, browser)
            discovered = await _discovery(address["http_url"])
            seen = await _cdp_agent_loads(address["ws_url"], [MULTI_SITE])
            reattached = await Env.from_instance_id(deployed.instance_id)
            world = _browser_universe(office.id, [{"name": "sid", "value": "ada", "url": MULTI_SITE, "max_age": 3600}])
            await LoadArtifactTaskStep(id="seed", version=None, env_id=office.id, artifact_id=world.id).execute(context)
            loaded = await _browser_op(browser, "cookies/get", {"urls": [MULTI_SITE]})

            subprocess.run(["docker", "kill", *_containers(browser.sandbox_id)], capture_output=True)
            await ResetEnvTaskStep(id="reset", version=None, env_id=office.id, fail_task_on_error=True).execute(context)
            after_kill = await _answers_again(deployed)
            after_kill_tabs = (await _drive(deployed))["tabs"]

            subprocess.run(["docker", "restart", _gateway(deployed, made)], capture_output=True, check=True)
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
    assert (start_tab["url"], start_tab["site"], start_tab["outcome"]) == (MULTI_SITE, "shop", "loaded"), start_tab
    # An agent drives it through the gateway, every tool under chromium_.
    assert {"chromium_computer_screenshot", "chromium_computer_navigate"} <= driven["tools"], driven
    assert not any(name.startswith("computer_") for name in driven["tools"]), driven
    assert driven["shot"] and driven["opened"] and MULTI_SITE in driven["tabs"], driven
    # Its CDP address names the gateway, its discovery answer too, and an agent there reaches the site by name.
    token = address["http_url"].split("/cdp/")[1].strip("/")
    assert address["http_url"] == f"{deployed.gateway_url}/browser/chromium/cdp/{token}/", address
    assert browser.environment_url.rstrip("/") not in json.dumps([address, discovered]), (address, discovered)
    assert discovered["webSocketDebuggerUrl"].startswith(
        f"ws{deployed.gateway_url.removeprefix('http')}/browser/chromium/cdp/"), discovered
    assert seen[MULTI_SITE]["error"] is None and seen[MULTI_SITE].get("ready") == "complete", seen
    # A later process reattaches it from the instance store, and a universe loaded into the multi env sets its cookies.
    assert reattached._computers.browser._deployed.sandbox_id == browser.sandbox_id
    assert [cookie["value"] for cookie in loaded["cookies"] if cookie["name"] == "sid"] == ["ada"], loaded
    # Through a kill and a gateway restart, a reset brings it back.
    assert after_kill and MULTI_SITE in after_kill_tabs, after_kill_tabs
    assert after_restart, through_restart


@pytest.mark.asyncio
async def test_two_leases_in_a_multi_envs_browser_each_sign_in_to_its_site_by_name_through_the_gateway(
        browser_computer, session_shop):
    """M8.15: in a multi env whose browser is chromium, two leases are each signed in to the env's site as a user of
    their own, and a tab of each opens the site by its name, over https through the gateway, as that user; the
    default context has none of their cookies."""
    with namespace_routing():
        office = _office(session_shop, [Env.get(browser_computer.id)])
        context = TaskStepContext()

        def step(type: str, **fields):
            return get_task_step_registry()[type].from_dict({
                "id": f"{type}-{fields.get('label') or fields.get('context')}", "version": None, "type": type,
                "env_id": office.id, **fields})

        try:
            [deployed] = (await DeployEnvTaskStep(
                id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                child_envs={"chromium": {"sandbox_type": "local", "env_params": {
                    "form": "container", "app_settings": {"chromium": {"start_urls": ["about:blank"]}}}}},
            ).execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            for label, user in (("buyer", "alex"), ("seller", "bo")):
                context = await step("browser_lease_context", label=label).execute(context)
                context = await step("browser_sign_in", site=session_shop.id, user=user, context=label).execute(context)
                context = await step("browser_open_tab", url=SESSION_PAGE, context=label).execute(context)
            address = await _cdp_at_gateway(deployed, browser)
            seen = {label: await _lease_page_text(browser, address["ws_url"], label) for label in ("buyer", "seller")}
            default_cookies = (await _browser_op(browser, "cookies/get", {"urls": ["https://shop.test/"]}))["cookies"]
        finally:
            report = await teardown_run(context)

    assert not report.failed
    assert '"alex"' in seen["buyer"] and '"bo"' in seen["seller"], seen
    assert default_cookies == [], default_cookies


async def _lease_page_text(browser: DeployedEnv, ws_url: str, label: str) -> str:
    """The text of the page the one tab of the lease ``label`` shows once it's on shop.test, read over the CDP
    address ``ws_url``; what it last read if that never says who's signed in."""
    text, deadline = "", time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        [tab] = (await _browser_op(browser, "tabs/list", {"context": label}))["tabs"]
        if tab["url"].startswith("https://shop.test/"):
            async with connect(ws_url, max_size=None, open_timeout=60) as ws:
                ids = itertools.count(1)

                async def call(method: str, params: dict, session: str | None = None) -> dict:
                    message_id = next(ids)
                    await ws.send(json.dumps({"id": message_id, "method": method, "params": params,
                                              **({"sessionId": session} if session else {})}))
                    while (reply := json.loads(await asyncio.wait_for(ws.recv(), 60))).get("id") != message_id:
                        pass
                    return reply

                session = (await call("Target.attachToTarget", {"targetId": tab["id"], "flatten": True}))[
                    "result"]["sessionId"]
                text = (await call("Runtime.evaluate", {"expression": "document.body ? document.body.innerText : ''",
                                                         "returnByValue": True}, session))["result"]["result"].get(
                    "value", "")
            if '"user"' in text:
                return text
        await asyncio.sleep(POLL_S)
    return text


@pytest.mark.asyncio
async def test_a_multi_env_naming_no_browser_takes_the_runs_and_chromium_runs_as_a_computer_child(browser_computer,
                                                                                                  shop):
    with namespace_routing():
        office = _office(shop, None)
        context = TaskStepContext()
        try:
            with run_env_ids({WEBSITE_BROWSER_OVERRIDE: browser_computer.id}):
                [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                      ttl_seconds=900, child_envs={"chromium": MULTI_CHROMIUM}
                                                      ).execute(context)).deployed_envs
            stated = {child.environment_name for child in protocol_v1.children(deployed.environment_card)}
        finally:
            report = await teardown_run(context)

    assert not report.failed and set(deployed.child_envs) == {"chromium"}
    assert PLAYWRIGHT_MCP_ENVIRONMENT_NAME not in stated, stated
    [start_tab] = deployed.child_envs["chromium"]["metadata"]["computer"]["start_tabs"]["chromium"]
    assert (start_tab["url"], start_tab["site"], start_tab["outcome"]) == (MULTI_SITE, "shop", "loaded"), start_tab


@pytest.mark.asyncio
async def test_browser_connect_cdp_hands_an_agent_an_address_at_the_multi_envs_gateway_with_its_tools_hidden(
        browser_computer, shop):
    with namespace_routing():
        agent = put_test_agent(f"cdp-agent-{uuid.uuid4().hex[:8]}")
        office = _office(shop, [Env.get(browser_computer.id)])
        context = TaskStepContext()
        try:
            [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                  ttl_seconds=900, child_envs={"chromium": MULTI_CHROMIUM}
                                                  ).execute(context)).deployed_envs
            await DeployAgentTaskStep(id="deploy-solver", version=None, env_ids=[], a2a_agent_id=agent.id,
                                      a2a_agent_version=agent.version, agent_name="solver",
                                      sandbox_type="local").execute(context)
            await BrowserConnectCdpTaskStep(id="connect", version=None, env_id=office.id,
                                            agent_name="solver").execute(context)
            _, title = await _opened_over_cdp(context, "first", url=MULTI_SITE)
            tools = await _tool_names(deployed)
            async with httpx.AsyncClient(timeout=30) as http:
                lines = (await http.get(f"{deployed.gateway_url}/trajectory")).text.splitlines()
            relayed = [event for event in map(json.loads, lines) if event.get("event_type") == "browser_connection"]
        finally:
            report = await teardown_run(context)

    assert not title.startswith("failed"), title
    assert tools == [], tools
    assert [event["browser"] for event in relayed] and all(event["browser"] == "chromium" for event in relayed), relayed
    assert context.metadata["browser_connections"] == [{
        "step_id": "connect", "env_id": office.id, "agent_name": "solver", "browser": "chromium", "name": "chromium",
        "keep_tools": False, "role": "default", "tools_hidden": "*"}]
    assert not report.failed


@pytest.mark.asyncio
async def test_behind_a_front_that_ends_tls_the_multi_envs_gateway_hands_out_https_and_wss_and_an_agent_connects(
        browser_computer, shop):
    ca = local_ca()
    with namespace_routing():
        office = _office(shop, [Env.get(browser_computer.id)])
        context = TaskStepContext()
        try:
            [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                  ttl_seconds=900, child_envs={"chromium": MULTI_CHROMIUM}
                                                  ).execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            gateway = urllib.parse.urlsplit(deployed.gateway_url)
            async with tls_front(ca, gateway.hostname, gateway.port) as front:
                named = await _cdp_at_gateway(deployed, browser, at=front, origin=front, ca=ca)
                unnamed = await _cdp_at_gateway(deployed, browser, at=front, user="bob", ca=ca)
                discovered = await _discovered(named["http_url"], ca)
                seen = await _cdp_agent_loads(named["ws_url"], [MULTI_SITE], tls=_trusting(ca))
        finally:
            report = await teardown_run(context)

    host = front.removeprefix("https://")
    assert named["http_url"].startswith(f"{front}/browser/chromium/cdp/"), named
    assert named["ws_url"].startswith(f"wss://{host}/browser/chromium/cdp/") and discovered == named["ws_url"], named
    assert unnamed["ws_url"].startswith(f"ws://{host}/browser/chromium/cdp/"), unnamed
    assert seen[MULTI_SITE]["error"] is None and seen[MULTI_SITE].get("ready") == "complete", seen
    assert not report.failed


#: A page the site that offers session serves, which says who a request signs in, and chromium starting there.
SESSION_PAGE = "https://shop.test/api/session"
SESSION_CHROMIUM = {**MULTI_CHROMIUM,
                    "env_params": {"form": "container", "app_settings": {"chromium": {"start_urls": [SESSION_PAGE]}}}}


@pytest.fixture(scope="module")
def session_shop(browser_computer):
    """A site named shop.test that offers website/v1 session, under a local id, as a multi env naming chromium's local
    id must be."""
    with namespace_routing():
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        return WebsiteEnv.put(id=f"@local/~/signin-{uid}/shop", environment_name="shop", domains=["shop.test"],
                              server_docker_image_artifact=_image("session_website", f"signin-shop-{uid}"))


@pytest.mark.asyncio
async def test_under_a_chromium_default_a_site_gets_chromium_unnamed_which_browser_sign_in_signs_in_after_each_reset(
        browser_computer, session_shop):
    config = get_config()
    with namespace_routing():
        office = MultiEnv.put(id=f"{session_shop.id.rsplit('/', 1)[0]}/office", mcp_server_envs=[],
                              website_envs=[session_shop], name="office")
        sign_in = BrowserSignInTaskStep(id="in", version=None, env_id=office.id, site=session_shop.id, user="alex")
        context = TaskStepContext()
        config.default_website_browser_env_id = browser_computer.id  # the config's default, not the run's override
        try:
            [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                  ttl_seconds=900, child_envs={"chromium": SESSION_CHROMIUM}
                                                  ).execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            before = await _signed_in(browser)
            await sign_in.execute(context)
            signed = await _signed_in(browser)
            await ResetEnvTaskStep(id="reset", version=None, env_id=office.id, fail_task_on_error=True).execute(context)
            after_reset = await _signed_in(browser)
            await sign_in.execute(context)
            again = await _signed_in(browser)
        finally:
            config.default_website_browser_env_id = None
            report = await teardown_run(context)

    assert not report.failed
    assert office.browser_envs is None and set(deployed.child_envs) == {"chromium"}
    [start_tab] = deployed.child_envs["chromium"]["metadata"]["computer"]["start_tabs"]["chromium"]
    assert (start_tab["url"], start_tab["site"], start_tab["outcome"]) == (SESSION_PAGE, "shop", "loaded"), start_tab
    assert (before, signed, after_reset, again) == (None, "alex", None, "alex")
    assert context.metadata["browser_steps"]["in"]["names"] == ["shop.test"]


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


# --- a site reached by its names through a multi env's gateway ---

HOST_START = "https://shop.test/api/host"
HOST_CHROMIUM = {**MULTI_CHROMIUM,
                 "env_params": {"form": "container", "app_settings": {"chromium": {"start_urls": [HOST_START]}}}}
#: Where the gateway reaches the site, which is the site's Host when a client reaches it at any other name.
SITE_INTERNAL = f"shop-{AGENT_ENV_WEBSITE_BACKEND_SUFFIX}:{AGENT_ENV_WEBSITE_BACKEND_PORT}"


@pytest.fixture(scope="module")
def host_shops(browser_computer):
    """Two sites named shop.test that build their addresses from their Host, as many real apps do, under local ids:
    one whose env stores the name and that serves no card, and one that stores none and whose card declares it."""
    with namespace_routing():
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        image = _image("host_website", f"host-shop-{uid}")
        return {"stored": WebsiteEnv.put(id=f"@local/~/host-{uid}/stored/shop", environment_name="shop",
                                         domains=["shop.test"], server_docker_image_artifact=image),
                "card": WebsiteEnv.put(id=f"@local/~/host-{uid}/card/shop", environment_name="shop",
                                       settings={"card_domains": "shop.test"}, server_docker_image_artifact=image)}


async def _landed(deployed, browser: DeployedEnv, url: str) -> str:
    """Where chromium lands, driven through the gateway's /mcp to ``url``: the first tab at the site's /home, else
    every tab's URL."""
    async with streamable_http_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            assert not (await session.call_tool("chromium_computer_navigate", {"url": url})).isError
    deadline = asyncio.get_running_loop().time() + WAIT_S
    while True:
        urls = [tab.get("url", "") for tab in await tabs_of(browser)]
        if (home := next((u for u in urls if u.endswith("/home")), None)) or \
                asyncio.get_running_loop().time() > deadline:
            return home or " ".join(urls)
        await asyncio.sleep(POLL_S)


async def _listed(deployed) -> list[str]:
    async with streamable_http_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_website_urls", {})
    assert not result.isError and result.structuredContent, result
    return result.structuredContent["result"]


async def _seen(deployed, root: str | None) -> dict:
    """The Host and X-Forwarded-Host the site sees for a request straight to the gateway's site route naming ``root``
    as the origin it was reached at, or naming none."""
    async with httpx.AsyncClient(timeout=30) as http:
        answer = await http.get(f"{deployed.gateway_url}/website/shop/api/host",
                                headers={"AgentEnv-Site-Root": root} if root else {})
    assert answer.status_code == 200, answer.text
    return answer.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("names", ["stored", "card"])
async def test_a_site_chromium_reaches_by_name_through_the_gateway_takes_that_name_as_its_host_and_no_forged_one(
        browser_computer, host_shops, made, names):
    with namespace_routing():
        office = _office(host_shops[names], [Env.get(browser_computer.id)])
        step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                                 child_envs={"chromium": HOST_CHROMIUM})
        context = TaskStepContext()
        try:
            [deployed] = (await step.execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            landed, listed = await _landed(deployed, browser, "https://shop.test/login"), await _listed(deployed)
            seen = [await _seen(deployed, root) for root in ("https://Shop.Test", "https://evil.example", None)]
            subprocess.run(["docker", "restart", _gateway(deployed, made)], capture_output=True, check=True)
            again = await _answers_again(deployed)
            after_restart = (await _landed(deployed, browser, "https://shop.test/login"), await _listed(deployed),
                             await _seen(deployed, "https://evil.example"))
        finally:
            report = await teardown_run(context)

    assert not report.failed
    # chromium follows the site's redirect, built from its Host, to the name it reached it at.
    assert landed == "https://shop.test/home", landed
    # An agent beside chromium is given the site at the name chromium opens it at.
    assert listed == ["https://shop.test/"], listed
    # A client naming any other origin, or none, leaves the site its internal Host; X-Forwarded-Host is as it was.
    gateway_host = urllib.parse.urlsplit(deployed.gateway_url).netloc
    assert seen == [{"host": "Shop.Test", "forwarded_host": "Shop.Test"},
                    {"host": SITE_INTERNAL, "forwarded_host": "evil.example"},
                    {"host": SITE_INTERNAL, "forwarded_host": gateway_host}], seen
    # A restarted gateway, which reads a card's names again, keeps all of it.
    assert again and after_restart == ("https://shop.test/home", ["https://shop.test/"],
                                       {"host": SITE_INTERNAL, "forwarded_host": "evil.example"}), after_restart
