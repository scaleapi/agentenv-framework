"""The page driver against a fake browser server on the app's socket: it passes the definition checks with a page
screen, turns tool calls into the CDP input they mean, shows a tab that paints nothing as white, and once the browser
has ended on its own, its ready check fails for good and its calls say why. Beside its own tools it serves Playwright
MCP's from their manifest, which is of the version its install downloads and lists only tools looked at, the two that run
the agent's code hidden, on a Playwright MCP it runs on the CDP address the browser issues it under the driver's loopback
origin, a check its tool calls don't credit to the agent: here a stand-in, which starts on the first call, keeps a session
per caller, starts again on a new address or once killed, and never gets the driver's environment; a call it ends
under fails then, a caller's session left idle is closed, and a start that never listens is stopped and tried afresh."""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import re
import signal
import sys
import tomllib
from pathlib import Path
from unittest import mock

import httpx
from mcp.types import CallToolResult, TextContent
import pytest
from pytest_socket import enable_socket

import agent_env
from agentenv_protocol.computer import (DeviceUnavailable, FileChooser, InvalidArgument, PageDialog, PageState,
                                        Platform, Tab)
from agentenv_protocol.computer import driver as computer_driver
from agentenv_protocol.computer.app_runtime import LAUNCHES
from agentenv_protocol.tool_context import ToolContext, bound
from agentenv_protocol.website_browser import BROWSER
from agentenv_protocol.website_browser.log import TOOL_CALL_HEADER

DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "page" / "driver.py"
PLAYWRIGHT_MCP = DRIVER.parent / "playwright_mcp.py"
#: Playwright MCP's tools, each looked at for whether it runs the agent's own code and so is hidden by default: an
#: upgrade that adds a tool fails here until it's been looked at too.
REVIEWED_TOOLS = {
    "browser_click", "browser_close", "browser_console_messages", "browser_drag", "browser_drop",
    "browser_emulate_media", "browser_evaluate", "browser_file_upload", "browser_fill_form", "browser_find",
    "browser_handle_dialog", "browser_hover", "browser_mouse_click_xy", "browser_mouse_down", "browser_mouse_drag_xy",
    "browser_mouse_move_xy", "browser_mouse_up", "browser_mouse_wheel", "browser_navigate", "browser_navigate_back",
    "browser_network_request", "browser_network_requests", "browser_press_key", "browser_resize",
    "browser_run_code_unsafe", "browser_select_option", "browser_snapshot", "browser_tabs", "browser_take_screenshot",
    "browser_type", "browser_wait_for",
}
NO_DISPLAY = Platform(os="linux", distribution="ubuntu", version="24.04", arch="amd64", display="none")
PAGE_OPS = ("page/viewport", "page/capture", "page/input", "page/state", "navigate", "history", "tabs/list",
            "tabs/open", "tabs/activate", "tabs/close", "dialog/answer", "files/choose", "cdp", "calls/begin",
            "calls/end")
GETS = ("page/viewport", "page/capture", "page/state", "tabs/list")
CARD = {"name": "chromium", "capabilities": {"extensions": [{"uri": BROWSER, "params": {"methods": {
    op: {"method": "GET" if op in GETS else "POST", "endpoint": f"/agentenv/ext/browser/{op}"}
    for op in PAGE_OPS}}}]}}
#: A stand-in for Playwright MCP on the port and CDP address it's started with: browser_snapshot says what it was
#: started with and which session called it, and browser_wait_for waits. Started on an address ending /exits, it exits
#: at once; on one ending /silent-once, the first time it writes its pid to the marker it's given and never listens; on
#: one ending /wedged-once, the first time it listens but never answers.
FAKE_PLAYWRIGHT_MCP = """import asyncio, json, os, sys, time
from mcp.server.fastmcp import Context, FastMCP
args = sys.argv[1:]
endpoint, port = args[args.index("--cdp-endpoint") + 1], int(args[args.index("--port") + 1])
if endpoint.endswith("/exits"):
    sys.exit(3)
marker = args[args.index("--marker") + 1]
if endpoint.endswith("/silent-once") and not os.path.exists(marker):
    open(marker, "w").write(str(os.getpid()))
    time.sleep(3600)
if endpoint.endswith("/wedged-once") and not os.path.exists(marker):
    import socket
    open(marker, "w").write(str(os.getpid()))
    held = socket.create_server(("127.0.0.1", port))
    accepted = []
    while True:
        accepted.append(held.accept())
app = FastMCP("fake", host="127.0.0.1", port=port)
@app.tool()
def browser_snapshot(ctx: Context) -> str:
    return json.dumps({"pid": os.getpid(), "endpoint": endpoint, "session": id(ctx.session), "cwd": os.getcwd(),
                       "env": sorted(os.environ)})
@app.tool()
async def browser_wait_for(time: float) -> str:
    await asyncio.sleep(time)
    return "waited"
app.run("streamable-http")
"""
#: A 2×1 PNG's first 24 bytes: its signature, then its IHDR's length, type, width and height.
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (2).to_bytes(4, "big") + (1).to_bytes(4, "big")


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _playwright_mcp():
    return _module("playwright_mcp", PLAYWRIGHT_MCP)


def _load():
    """driver.py, importing its own playwright_mcp.py as it does where it runs."""
    with mock.patch.dict(sys.modules, {"playwright_mcp": _playwright_mcp()}):
        return _module("page_driver", DRIVER)


@pytest.fixture
def page(monkeypatch, tmp_path):
    """The driver, on a computer with no display, its one browser in ``tmp_path``, whose server answers ``served``,
    a dict of answers by op, and records each request it gets in ``sent``."""
    module = _load()
    monkeypatch.setattr(computer_driver, "detect", lambda: NO_DISPLAY)
    driver = module.PageDriver()
    driver._browser = (tmp_path, CARD)
    driver.sent, driver.headers, driver.served = [], [], {"page/viewport": {"width": 1280, "height": 800}}
    driver.list_queries = []

    def handler(request: httpx.Request) -> httpx.Response:
        op = request.url.path.removeprefix("/agentenv/ext/browser/")
        driver.sent.append((op, json.loads(request.content) if request.content else None))
        if op == "tabs/list":
            driver.list_queries.append(request.url.query.decode())
        driver.headers.append(request.headers)
        answer = driver.served.get(op, {})
        return answer if isinstance(answer, httpx.Response) else httpx.Response(200, json=answer)

    monkeypatch.setattr(module, "_http", lambda folder: httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://app"))
    return driver


def _ended(folder: Path) -> None:
    (folder / LAUNCHES).parent.mkdir(parents=True, exist_ok=True)
    (folder / LAUNCHES).write_text(json.dumps({"event": "launch", "cause": "boot", "at": "t0"}) + "\n"
                                   + json.dumps({"event": "exit", "signal": "SIGKILL", "at": "t1"}) + "\n")


def _events(page) -> list[dict]:
    return [event for op, body in page.sent if op == "page/input" for event in body["events"]]


def test_its_screen_is_a_page_on_a_computer_with_no_display():
    module = _load()

    assert module.PageDriver.screen_kind == "page"
    assert module.PageDriver.supports == [{"os": {"type": "linux", "display": ["none"]}, "arch": ["amd64", "arm64"]}]


@pytest.mark.asyncio
async def test_a_key_with_control_held_types_nothing_so_it_reaches_the_page_as_a_shortcut(page):
    await page.press_keys(["Control", "a"])

    assert [(e["params"]["type"], e["params"]["key"], e["params"]["modifiers"], e["params"].get("text"))
            for e in _events(page)] == [("rawKeyDown", "Control", 2, None), ("rawKeyDown", "a", 2, None),
                                        ("keyUp", "a", 2, None), ("keyUp", "Control", 0, None)]


@pytest.mark.asyncio
async def test_a_key_held_down_then_let_go_is_released_with_its_own_modifier_off(page):
    await page.keys_down(["Shift"])
    await page.keys_up(["Shift"])

    assert [(e["params"]["type"], e["params"]["modifiers"]) for e in _events(page)] == [("rawKeyDown", 8),
                                                                                        ("keyUp", 0)]


@pytest.mark.asyncio
async def test_with_shift_held_typed_text_is_shifted_and_with_control_held_a_click_carries_it(page):
    await page.keys_down(["Shift"])
    await page.type_in("a1;")
    await page.keys_up(["Shift"])
    await page.keys_down(["Control"])
    await page.click_at(10, 20, "left", 1, [])
    await page.scroll_at(10, 20, "down", 1, ["Alt"])

    events = _events(page)
    typed = [(e["params"]["key"], e["params"]["modifiers"], e["params"]["text"]) for e in events
             if e["params"].get("type") == "keyDown"]
    assert typed == [("A", 8, "A"), ("!", 8, "!"), (":", 8, ":")]
    assert {e["params"]["modifiers"] for e in events if e["params"]["type"] in ("mousePressed", "mouseReleased")} == {2}
    assert events[-1]["params"]["type"] == "mouseWheel" and events[-1]["params"]["modifiers"] == 3


@pytest.mark.asyncio
async def test_a_scroll_is_one_wheel_event_a_notch_sent_together(page):
    await page.scroll_at(10, 20, "left", 3, [])

    assert [op for op, _ in page.sent] == ["page/input"]
    assert [(e["params"]["type"], e["params"]["deltaX"], e["params"]["deltaY"]) for e in _events(page)] == [
        ("mouseWheel", -100, 0)] * 3


@pytest.mark.asyncio
async def test_typed_text_is_keystrokes_with_shift_where_it_takes_one_and_inserted_text_for_the_rest(page):
    await page.type_in("Hi!\né")

    events = _events(page)
    downs = [(e["params"]["key"], e["params"]["modifiers"], e["params"]["text"]) for e in events
             if e["params"].get("type") == "keyDown"]
    assert downs == [("H", 8, "H"), ("i", 0, "i"), ("!", 8, "!"), ("Enter", 0, "\r")]
    assert events[-1] == {"method": "Input.insertText", "params": {"text": "é"}}


@pytest.mark.asyncio
async def test_a_double_click_is_one_move_then_two_presses_and_releases_counted(page):
    await page.click_at(10, 20, "left", 2, [])

    assert [(e["params"]["type"], e["params"]["clickCount"]) for e in _events(page)] == [
        ("mouseMoved", 0), ("mousePressed", 1), ("mouseReleased", 1), ("mousePressed", 2), ("mouseReleased", 2)]


@pytest.mark.asyncio
async def test_its_capture_is_the_pages_png_and_a_tab_that_paints_nothing_is_white(page):
    shown = {"tab": "A1", "url": "https://mail.test/", "title": "Inbox"}
    page.served["page/capture"] = {"png": base64.b64encode(PNG).decode(), "page": shown}
    shot = await page.page()
    page.served["page/capture"] = {"blank": True, "page": shown}
    blank = await page.page()

    assert (shot.width, shot.height, shot.mode, shot.data) == (2, 1, "png", PNG)
    assert (blank.width, blank.height, blank.mode) == (1280, 800, "RGB") and set(blank.data) == {0xff}
    assert shot.page == blank.page == PageState("A1", "https://mail.test/", "Inbox")


@pytest.mark.asyncio
async def test_its_browser_check_passes_until_the_browser_ends_on_its_own_then_fails_for_good(page, tmp_path):
    before = await page.ready_checks()
    _ended(tmp_path)
    after = await page.ready_checks()

    assert [(c.name, c.ok, c.final) for c in before] == [("browser", True, False)]
    assert [(c.name, c.ok, c.final) for c in after] == [("browser", False, True)]
    assert "by signal SIGKILL" in after[0].detail and "infrastructure failure" in after[0].detail


@pytest.mark.asyncio
async def test_once_the_browser_has_ended_its_calls_say_why_and_send_nothing_though_its_server_answers(page, tmp_path):
    page.served["page/capture"] = httpx.Response(503, text="the browser has no tab open")
    with pytest.raises(DeviceUnavailable, match="couldn't page/capture: the browser has no tab open"):
        await page.page()
    _ended(tmp_path)

    with pytest.raises(DeviceUnavailable, match=f"the browser in {tmp_path.name} has failed: .*by signal SIGKILL"):
        await page.page()
    page.served["page/capture"] = {"blank": True}
    sent = len(page.sent)

    with pytest.raises(DeviceUnavailable, match="has failed"):
        await page.click_at(1, 2, "left", 1, [])
    assert len(page.sent) == sent


@pytest.mark.asyncio
async def test_its_page_operations_call_the_browsers_and_a_tab_action_answers_with_the_tabs_after_it(page):
    page.served["tabs/list"] = {"tabs": [{"id": "A1", "url": "https://mail.test/", "title": "Inbox", "active": True}]}
    await page.go_to("https://mail.test/")
    await page.move_in_history("back")
    opened = await page.tab_action("open", None, "https://mail.test/")
    await page.tab_action("switch", "A1", None)
    await page.tab_action("close", "A1", None)
    await page.answer_dialog(False, None)
    await page.answer_dialog(True, "Grace")
    await page.choose_files(["cv.pdf"])

    assert opened == [Tab("A1", "https://mail.test/", "Inbox", True)]
    assert [entry for entry in page.sent if entry[0] != "tabs/list"] == [
        ("navigate", {"url": "https://mail.test/"}), ("history", {"direction": "back"}),
        ("tabs/open", {"url": "https://mail.test/"}), ("tabs/activate", {"tab": "A1"}), ("tabs/close", {"tab": "A1"}),
        ("dialog/answer", {"accept": False}), ("dialog/answer", {"accept": True, "text": "Grace"}),
        ("files/choose", {"paths": ["cv.pdf"]})]


@pytest.mark.asyncio
async def test_on_a_browser_that_leases_contexts_its_tabs_are_every_contexts_and_a_leases_is_acted_on_in_it(page):
    leases = {**CARD["capabilities"]["extensions"][0]["params"]["methods"],
              "contexts/lease": {"method": "POST", "endpoint": "/agentenv/ext/browser/contexts/lease"}}
    page._browser = (page._browser[0], {"name": "chromium", "capabilities": {"extensions": [
        {"uri": BROWSER, "params": {"methods": leases}}]}})
    page.served["tabs/list"] = {"tabs": [
        {"id": "A1", "url": "https://mail.test/", "title": "Inbox", "active": False, "context": None},
        {"id": "B1", "url": "https://mail.test/", "title": "Bo's inbox", "active": True, "context": "bo"}]}

    listed = await page.tab_action("switch", "B1", None)
    await page.tab_action("close", "A1", None)

    assert listed == [Tab("A1", "https://mail.test/", "Inbox", False, None),
                      Tab("B1", "https://mail.test/", "Bo's inbox", True, "bo")]
    assert [entry for entry in page.sent if entry[0] != "tabs/list"] == [
        ("tabs/activate", {"tab": "B1", "context": "bo"}), ("tabs/close", {"tab": "A1"})]
    assert page.list_queries and all(query == "every=true" for query in page.list_queries)


@pytest.mark.asyncio
async def test_its_page_state_is_the_browsers_with_what_is_open_on_it(page):
    page.served["page/state"] = {"tab": "A1", "url": "https://mail.test/", "title": "Inbox",
                                 "dialog": {"type": "confirm", "message": "Delete?", "default_text": ""},
                                 "file_chooser": {"multiple": True}}
    open_on_it = await page.page_now()
    page.served["page/state"] = {"tab": "A1", "url": "https://mail.test/", "title": "Inbox"}

    assert open_on_it == PageState("A1", "https://mail.test/", "Inbox", PageDialog("confirm", "Delete?"),
                                   FileChooser(True))
    assert await page.page_now() == PageState("A1", "https://mail.test/", "Inbox")


@pytest.mark.asyncio
async def test_a_call_the_browser_refuses_as_given_wrong_is_an_invalid_argument_the_agent_can_put_right(page):
    page.served["files/choose"] = httpx.Response(400, json={"ok": False, "error": {
        "code": "invalid_argument", "message": "there's no file cv.pdf in ~/Uploads"}})

    with pytest.raises(InvalidArgument, match="there's no file cv.pdf in ~/Uploads"):
        await page.choose_files(["cv.pdf"])


def test_it_serves_playwright_mcps_tools_from_their_manifest_and_hides_the_two_that_run_the_agents_code(page):
    manifest = json.loads((DRIVER.parent / "playwright-mcp-tools.json").read_text())
    forwarded = page.forwarded_tools()

    assert manifest["caps"] == ["vision"] and {t["name"] for t in manifest["tools"]} == REVIEWED_TOOLS
    assert [f.tool.name for f in forwarded] == [t["name"] for t in manifest["tools"]]
    assert [f.tool.inputSchema for f in forwarded] == [t["inputSchema"] for t in manifest["tools"]]
    assert {f.tool.name for f in forwarded if f.hidden_by_default} == {"browser_evaluate", "browser_run_code_unsafe"}


def test_its_manifest_is_of_the_playwright_mcp_its_install_downloads():
    manifest = json.loads((DRIVER.parent / "playwright-mcp-tools.json").read_text())
    downloads = tomllib.loads((DRIVER.parent / "env.toml").read_text())["install"]["linux"]["downloads"]

    versions = [m.group(1) for d in downloads if (m := re.search(r"/@playwright/mcp/-/mcp-(.+)\.tgz$", d["url"]))]

    assert versions == [manifest["version"]]


@pytest.mark.asyncio
async def test_each_playwright_mcp_call_is_an_entry_in_the_browsers_log_begun_as_the_agents_call_ended_with_how_it_went(
        page, monkeypatch):
    page.served["calls/begin"] = {"entry": 7}
    results = iter([CallToolResult(content=[TextContent(type="text", text="clicked")]),
                    CallToolResult(isError=True, content=[TextContent(type="text", text="no such element")])])

    async def call(tool, **arguments):
        return next(results)

    monkeypatch.setattr(page._playwright, "call", call)
    with bound(ToolContext(tool="browser_click", call_id="c1", mcp=True)):
        clicked = await page._playwright_call("browser_click", target="e3")
    with bound(ToolContext(tool="browser_snapshot", call_id="c2", mcp=True)):
        failed = await page._playwright_call("browser_snapshot")

    assert page.sent == [("calls/begin", {"user": "playwright-mcp", "acts": True, "within_s": 400.0}),
                         ("calls/end", {"entry": 7, "ok": True}),
                         ("calls/begin", {"user": "playwright-mcp", "acts": False, "within_s": 400.0}),
                         ("calls/end", {"entry": 7, "ok": False, "error": "no such element"})]
    assert TOOL_CALL_HEADER in page.headers[0] and TOOL_CALL_HEADER not in page.headers[1]
    assert clicked.content[0].text == "clicked" and failed.isError


@pytest.mark.asyncio
async def test_a_call_the_browser_cant_log_still_goes_and_a_cancelled_one_is_ended_as_cancelled(page, monkeypatch):
    page.served["calls/begin"] = httpx.Response(500, text="the browser is restarting")
    calls = []

    async def call(tool, **arguments):
        calls.append(tool)
        if tool == "browser_wait_for":
            await asyncio.Event().wait()
        return CallToolResult(content=[TextContent(type="text", text="ok")])

    monkeypatch.setattr(page._playwright, "call", call)
    unlogged = await page._playwright_call("browser_snapshot")
    page.served["calls/begin"] = {"entry": 8}
    waiting = asyncio.ensure_future(page._playwright_call("browser_wait_for", time=60))
    await asyncio.sleep(0.05)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert calls == ["browser_snapshot", "browser_wait_for"] and unlogged.content[0].text == "ok"
    assert [op for op, _ in page.sent] == ["calls/begin", "calls/begin", "calls/end"]
    assert page.sent[-1] == ("calls/end", {"entry": 8, "ok": False, "error": "cancelled"})


@pytest.mark.asyncio
async def test_it_asks_the_browser_for_playwright_mcps_address_under_its_own_loopback_origin_as_its_own_call(
        page, monkeypatch):
    monkeypatch.setenv("MCP_PORT", "18765")
    page.served["cdp"] = {"http_url": "http://127.0.0.1:18765/svc/chromium/cdp/t/",
                          "ws_url": "ws://127.0.0.1:18765/svc/chromium/cdp/t/devtools/browser"}

    with bound(ToolContext(tool="browser_click", call_id="c1", mcp=True)):
        endpoint = await page._playwright_endpoint()
        await page.go_to("https://example.com/")

    assert endpoint == "ws://127.0.0.1:18765/svc/chromium/cdp/t/devtools/browser"
    assert page.sent[0] == ("cdp", {"user": "playwright-mcp", "origin": f"http://127.0.0.1:18765/svc/{page._browser[0].name}"})
    assert TOOL_CALL_HEADER not in page.headers[0]
    assert TOOL_CALL_HEADER in page.headers[1]


def test_playwright_mcp_attaches_to_the_browser_never_launches_one_and_ends_with_the_driver():
    command = _playwright_mcp()._command("ws://127.0.0.1:18765/svc/chromium/cdp/t/devtools/browser", 40123)

    assert command[:4] == ["setpriv", "--pdeathsig", "TERM", "--"]
    assert command[command.index("--cdp-endpoint") + 1] == "ws://127.0.0.1:18765/svc/chromium/cdp/t/devtools/browser"
    assert {"--caps", "vision", "--ignore-roots", "127.0.0.1", "40123"} <= set(command)
    assert not {"--isolated", "--headless", "--browser", "--allow-unrestricted-file-access"} & set(command)


@pytest.mark.asyncio
async def test_playwright_mcp_starts_on_the_first_call_keeps_a_session_per_caller_and_starts_again_when_it_must(
        stand_in, monkeypatch, tmp_path):
    _, playwright, endpoints, called = stand_in
    monkeypatch.setenv("AGENTENV_CONTROL_TOKEN", "secret")

    async def snapshot(caller: str) -> dict:
        result = await called(caller)
        assert not result.isError, result.content
        return json.loads(result.content[0].text)

    try:
        first, again, other = await snapshot("ada"), await snapshot("ada"), await snapshot("bob")
        endpoints.append("ws://127.0.0.1:18765/svc/chromium/cdp/two/devtools/browser")
        moved = await snapshot("ada")
        os.kill(moved["pid"], signal.SIGKILL)
        await asyncio.sleep(1)
        revived = await snapshot("ada")
        endpoints.append("ws://127.0.0.1:18765/svc/chromium/cdp/exits")
        refused = await called("ada")
    finally:
        await playwright._stop()

    assert first["pid"] == again["pid"] == other["pid"] and first["session"] == again["session"] != other["session"]
    assert first["endpoint"].endswith("/cdp/one/devtools/browser") and first["cwd"] == str(tmp_path / "home" / "Uploads")
    assert "AGENTENV_CONTROL_TOKEN" not in first["env"] and "HOME" in first["env"]
    assert moved["pid"] != first["pid"] and moved["endpoint"].endswith("/cdp/two/devtools/browser")
    assert revived["pid"] != moved["pid"]
    assert refused.isError and "exited at start with 3" in refused.content[0].text


@pytest.fixture
def stand_in(monkeypatch, tmp_path):
    """A ``PlaywrightMcp`` on the stand-in, its address the last of ``endpoints``, and a way to call it as a caller."""
    enable_socket()
    module = _playwright_mcp()
    fake = tmp_path / "fake_playwright_mcp.py"
    fake.write_text(FAKE_PLAYWRIGHT_MCP)
    monkeypatch.setattr(module, "_command", lambda endpoint, port: [
        sys.executable, str(fake), "--cdp-endpoint", endpoint, "--port", str(port), "--marker", str(tmp_path / "m")])
    monkeypatch.setattr(module, "OUTPUT", tmp_path / "out")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    endpoints = ["ws://127.0.0.1:18765/svc/chromium/cdp/one/devtools/browser"]

    async def endpoint() -> str:
        return endpoints[-1]

    playwright = module.PlaywrightMcp(endpoint)

    async def called(caller: str, tool: str = "browser_snapshot", arguments: dict | None = None):
        with bound(ToolContext.as_caller(session=caller)):
            return await playwright.call(tool, **(arguments or {}))

    return module, playwright, endpoints, called


@pytest.mark.asyncio
async def test_a_call_playwright_mcp_ends_under_fails_then_and_isnt_sent_again(stand_in):
    _, playwright, _, called = stand_in
    try:
        pid = json.loads((await called("ada")).content[0].text)["pid"]
        waiting = asyncio.ensure_future(called("ada", "browser_wait_for", {"time": 60}))
        await asyncio.sleep(1)
        os.kill(pid, signal.SIGKILL)
        ended = await asyncio.wait_for(waiting, 10)
        after = await called("ada")
    finally:
        await playwright._stop()

    assert ended.isError and "ended during browser_wait_for, which may have acted" in ended.content[0].text
    assert not after.isError and json.loads(after.content[0].text)["pid"] != pid


@pytest.mark.asyncio
async def test_a_call_its_caller_cancels_ends_its_call_on_playwright_mcp_too(stand_in):
    _, playwright, _, called = stand_in
    try:
        await called("ada")
        waiting = asyncio.ensure_future(called("ada", "browser_wait_for", {"time": 60}))
        await asyncio.sleep(1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        await asyncio.sleep(0.5)
        left = [task for task in asyncio.all_tasks() if "call_tool" in repr(task.get_coro())]
        after = await called("ada")
    finally:
        await playwright._stop()

    assert left == [] and not after.isError


@pytest.mark.asyncio
async def test_a_callers_session_left_idle_is_closed_when_another_caller_comes(stand_in, monkeypatch):
    module, playwright, _, called = stand_in
    monkeypatch.setattr(module, "SESSION_IDLE_S", 0.5)
    try:
        await called("ada")
        ada = playwright._sessions["ada"]
        await asyncio.sleep(1)
        await called("bob")
        await asyncio.wait_for(ada.owner, 10)
        kept = sorted(playwright._sessions)
    finally:
        await playwright._stop()

    assert kept == ["bob"] and ada.close.is_set()


@pytest.mark.asyncio
async def test_a_start_that_never_listens_is_stopped_and_the_next_call_starts_afresh(stand_in, monkeypatch, tmp_path):
    module, playwright, endpoints, called = stand_in
    monkeypatch.setattr(module, "START_S", 2.0)
    endpoints.append("ws://127.0.0.1:18765/svc/chromium/cdp/silent-once")
    try:
        refused = await called("ada")
        hung = int((tmp_path / "m").read_text())
        gone = _gone(hung)
        answered = await called("ada")
    finally:
        await playwright._stop()

    assert refused.isError and "didn't listen within 2 s" in refused.content[0].text
    assert gone and not answered.isError


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.asyncio
async def test_a_call_slower_than_its_bound_is_refused_and_its_session_opened_afresh(stand_in, monkeypatch):
    module, playwright, _, called = stand_in
    monkeypatch.setattr(module, "CALL_TIMEOUT_S", 1.0)
    try:
        await called("ada")
        first = playwright._sessions["ada"]
        slow = await asyncio.wait_for(called("ada", "browser_wait_for", {"time": 30}), 10)
        after = await called("ada")
    finally:
        await playwright._stop()

    assert slow.isError and "didn't answer browser_wait_for" in slow.content[0].text
    assert first.close.is_set() and not after.isError


@pytest.mark.asyncio
async def test_a_call_is_bounded_from_when_it_asks_for_its_session_and_leaves_no_session_half_open(stand_in, monkeypatch):
    module, playwright, endpoints, called = stand_in
    monkeypatch.setattr(module, "CALL_BOUND_S", 1.0)
    try:
        async with playwright._lock:
            waited = await asyncio.wait_for(called("ada"), 10)
        endpoints.append("ws://127.0.0.1:18765/svc/chromium/cdp/wedged-once")
        opening = await asyncio.wait_for(called("bo"), 10)
        await asyncio.sleep(0.1)
        owners = [task for task in asyncio.all_tasks() if task.get_coro().__name__ == "_own"]
    finally:
        await playwright._stop()

    assert waited.isError and "didn't answer browser_snapshot within 1 s" in waited.content[0].text
    assert opening.isError and "within 1 s" in opening.content[0].text and owners == []


@pytest.mark.asyncio
async def test_a_playwright_mcp_that_never_opens_a_session_is_stopped_and_the_next_call_starts_afresh(
        stand_in, monkeypatch, tmp_path):
    module, playwright, endpoints, called = stand_in
    monkeypatch.setattr(module, "START_S", 2.0)
    endpoints.append("ws://127.0.0.1:18765/svc/chromium/cdp/wedged-once")
    try:
        refused = await asyncio.wait_for(called("ada"), 20)
        wedged = int((tmp_path / "m").read_text())
        gone = _gone(wedged)
        answered = await called("ada")
    finally:
        await playwright._stop()

    assert refused.isError and "didn't open a session within 2 s" in refused.content[0].text
    assert gone and not answered.isError
