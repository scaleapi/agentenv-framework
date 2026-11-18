"""The Playwright MCP browser's image, built for this machine and run alone: its recorder fronts the MCP port, so the
browser's card is served there, as every MCP server serves its card, and on the recorder's own port too, where deploys
read it until they read it at the MCP port, the card first and the MCP port only once Playwright MCP answers behind it,
which is what the gateway's health check asks; an MCP client lists and calls the browser's tools through the front; a
recording through the card on the MCP port films the page the client opens; and a killed recorder comes back, the front
with it, while the MCP server behind it stays up. Needs Docker."""

import asyncio
import json
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.computer.card import COMPUTER_RECORDING
from agentenv_protocol.types import WELL_KNOWN_PATH

from agent_env.env.envs import playwright_mcp
from agent_env.providers.sandbox_providers.host_images import host_platform
from tst.util.published_image import pull_published_else_build

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

CONTEXT = Path(playwright_mcp.__file__).parent
IMAGE = "agentenv-playwright-mcp:front-test"
#: How long the image may take to answer on its MCP port, its MCP server's first start included.
READY_S = 180.0
PAGE = "data:text/html,<title>front</title><h1 style='font-size:90px'>front</h1>"
#: The ids of the image's processes whose command line holds the first argument, killed with the second when it's given;
#: the image has no procps, so its recorder's Python reads /proc.
PROCESSES = """import os, sys
found = []
for pid in (p for p in os.listdir("/proc") if p.isdigit() and int(p) != os.getpid()):
    try:
        if sys.argv[1] in open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\\0", b" ").decode(errors="replace"):
            found.append(pid)
    except OSError:
        pass
for pid in found if len(sys.argv) > 2 else ():
    os.kill(int(pid), int(sys.argv[2]))
print(" ".join(sorted(found, key=int)))
"""


#: Run in the image as it starts: watches its card port and MCP port from inside, where nothing answers for a port that
#: isn't listening, and the moment the MCP port first accepts, opens an MCP session through it; prints whether the card
#: answered first and the session's status.
OPENING = """import json, socket, time, urllib.error, urllib.request
def accepts(port):
    try:
        socket.create_connection(("127.0.0.1", port), 0.5).close()
        return True
    except OSError:
        return False
deadline, card_at, mcp_at = time.monotonic() + 180, None, None
while time.monotonic() < deadline and mcp_at is None:
    now = time.monotonic()
    if card_at is None and accepts(18766):
        card_at = now
    if accepts(18765):
        mcp_at = now
    else:
        time.sleep(0.05)
body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "opening", "version": "1"}}}).encode()
request = urllib.request.Request("http://127.0.0.1:18765/mcp", body, {
    "Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
try:
    status = urllib.request.urlopen(request, timeout=30).status
except urllib.error.HTTPError as e:
    status = e.code
print(json.dumps({"card_first": card_at is not None and mcp_at is not None and card_at <= mcp_at, "status": status}))
"""


def _docker(*args: str) -> str:
    return subprocess.run(["docker", *args], check=True, capture_output=True, text=True).stdout


def _processes(container: str, pattern: str, signal: int | None = None) -> list[str]:
    return _docker("exec", container, "/opt/recorder/bin/python", "-c", PROCESSES, pattern,
                   *([str(signal)] if signal is not None else [])).split()


@pytest.fixture(scope="module")
def image() -> str:
    pull_published_else_build("playwright-mcp", CONTEXT / "Dockerfile", CONTEXT, IMAGE, platform=host_platform())
    return IMAGE


@pytest.fixture
def browser(image):
    """The image run alone, its MCP port and its recorder's port published on loopback: (container, the MCP port's
    base URL, the recorder port's)."""
    name = f"playwright-mcp-front-{uuid.uuid4().hex[:8]}"
    _docker("run", "-d", "--name", name, "-e", "MCP_HOST=0.0.0.0", "-e", "MCP_PORT=18765", "-p", "127.0.0.1::18765",
            "-p", "127.0.0.1::18766", image)
    try:
        mcp, recorder = (f"http://{_docker('port', name, f'{port}/tcp').splitlines()[0]}" for port in (18765, 18766))
        asyncio.run(_answering(mcp))
        yield name, mcp, recorder
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


async def _answering(mcp: str) -> None:
    """Wait until the front serves the card and Playwright MCP behind it opens a session."""
    deadline, last = time.monotonic() + READY_S, None
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=10) as http:
                (await http.get(f"{mcp}{WELL_KNOWN_PATH}")).raise_for_status()
            await _tools(mcp)
            return
        except Exception as e:  # noqa: BLE001 - not up yet
            last = e
            await asyncio.sleep(1)
    raise AssertionError(f"the image didn't answer on its MCP port within {READY_S:.0f} s: {last!r}")


async def _tools(mcp: str, call: tuple[str, dict] | None = None) -> tuple[list[str], str]:
    async with streamable_http_client(f"{mcp}/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = [tool.name for tool in (await session.list_tools()).tools]
            text = ""
            if call is not None:
                result = await session.call_tool(*call)
                text = "".join(getattr(part, "text", "") for part in result.content)
            return names, text


@pytest.mark.asyncio
async def test_the_card_is_on_the_mcp_port_and_the_recorders_and_tools_pass_through_the_front(browser):
    _, mcp, recorder = browser
    async with httpx.AsyncClient(timeout=30) as http:
        on_mcp_port = (await http.get(f"{mcp}{WELL_KNOWN_PATH}")).json()
        on_recorder_port = (await http.get(f"{recorder}{WELL_KNOWN_PATH}")).json()
    names, navigated = await _tools(mcp, ("browser_navigate", {"url": PAGE}))

    assert on_mcp_port == on_recorder_port and on_mcp_port["name"] == playwright_mcp.PLAYWRIGHT_MCP_ENVIRONMENT_NAME
    assert [extension["uri"] for extension in on_mcp_port["capabilities"]["extensions"]] == [COMPUTER_RECORDING]
    assert {"browser_navigate", "browser_snapshot", "browser_take_screenshot"} <= set(names)
    assert "front" in navigated


def test_the_mcp_port_opens_only_once_playwright_mcp_answers_behind_it_and_the_card_comes_first(image):
    name = f"playwright-mcp-opening-{uuid.uuid4().hex[:8]}"
    _docker("run", "-d", "--name", name, "-e", "MCP_HOST=0.0.0.0", "-e", "MCP_PORT=18765", image)
    try:
        opened = _docker("exec", name, "/opt/recorder/bin/python", "-c", OPENING).strip().splitlines()[-1]
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)

    assert json.loads(opened) == {"card_first": True, "status": 200}, opened


@pytest.mark.asyncio
async def test_a_recording_through_the_card_on_the_mcp_port_films_the_page_an_agent_opens(browser):
    """One MCP session throughout, as an agent holds one: an isolated browser lives as long as its session."""
    _, mcp, _ = browser
    async with httpx.AsyncClient(timeout=60) as http, streamable_http_client(f"{mcp}/mcp") as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            card = (await http.get(f"{mcp}{WELL_KNOWN_PATH}")).json()
            start, end = (protocol_v1.find_extension_method(card, COMPUTER_RECORDING, op)["endpoint"]
                          for op in ("start", "end"))
            await session.call_tool("browser_navigate", {"url": PAGE})
            started = (await http.post(f"{mcp}{start}", json={"request_id": "front:start", "fps": 5})).json()
            for shade in ("red", "green", "blue"):
                page = f"data:text/html,<body style='background:{shade}'>"
                await session.call_tool("browser_navigate", {"url": page})
                await asyncio.sleep(1)
            manifest = (await http.post(f"{mcp}{end}", json={"recording_id": started["recording_id"]})).json()

    assert manifest["ended_by"] == "end" and manifest["segments"], manifest
    assert sum(segment["frames"] for segment in manifest["segments"]) >= 3, manifest


@pytest.mark.asyncio
async def test_a_killed_recorder_comes_back_the_front_with_it_and_the_mcp_server_behind_it_stays_up(browser):
    name, mcp, _ = browser
    serving = _processes(name, "@playwright/mcp")
    assert serving and _processes(name, "/app/recorder.py", 9)
    deadline, back = time.monotonic() + 30, None
    while time.monotonic() < deadline and back is None:
        await asyncio.sleep(1)
        try:
            async with httpx.AsyncClient(timeout=5) as http:
                back = (await http.get(f"{mcp}{WELL_KNOWN_PATH}")).raise_for_status().json()
        except httpx.HTTPError:
            pass
    names, _ = await _tools(mcp)

    assert back is not None and back["name"] == playwright_mcp.PLAYWRIGHT_MCP_ENVIRONMENT_NAME
    assert "browser_navigate" in names
    assert _processes(name, "@playwright/mcp") == serving
