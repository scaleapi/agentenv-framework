"""The page driver against a fake browser server on the app's socket: it passes the definition checks with a page
screen, turns tool calls into the CDP input they mean, shows a tab that paints nothing as white, and once the browser
has ended on its own, its ready check fails for good and its calls say why."""
from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

import agent_env
from agentenv_protocol.computer import (DeviceUnavailable, FileChooser, InvalidArgument, PageDialog, PageState,
                                        Platform, Tab)
from agentenv_protocol.computer import driver as computer_driver
from agentenv_protocol.computer.app_runtime import LAUNCHES
from agentenv_protocol.website_browser import BROWSER

DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "page" / "driver.py"
NO_DISPLAY = Platform(os="linux", distribution="ubuntu", version="24.04", arch="amd64", display="none")
PAGE_OPS = ("page/viewport", "page/capture", "page/input", "page/state", "navigate", "history", "tabs/list",
            "tabs/open", "tabs/activate", "tabs/close", "dialog/answer", "files/choose")
GETS = ("page/viewport", "page/capture", "page/state", "tabs/list")
CARD = {"name": "chromium", "capabilities": {"extensions": [{"uri": BROWSER, "params": {"methods": {
    op: {"method": "GET" if op in GETS else "POST", "endpoint": f"/agentenv/ext/browser/{op}"}
    for op in PAGE_OPS}}}]}}
#: A 2×1 PNG's first 24 bytes: its signature, then its IHDR's length, type, width and height.
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (2).to_bytes(4, "big") + (1).to_bytes(4, "big")


def _load():
    spec = importlib.util.spec_from_file_location("page_driver", DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def page(monkeypatch, tmp_path):
    """The driver, on a computer with no display, its one browser in ``tmp_path``, whose server answers ``served``,
    a dict of answers by op, and records each request it gets in ``sent``."""
    module = _load()
    monkeypatch.setattr(computer_driver, "detect", lambda: NO_DISPLAY)
    driver = module.PageDriver()
    driver._browser = (tmp_path, CARD)
    driver.sent, driver.served = [], {"page/viewport": {"width": 1280, "height": 800}}

    def handler(request: httpx.Request) -> httpx.Response:
        op = request.url.path.removeprefix("/agentenv/ext/browser/")
        driver.sent.append((op, json.loads(request.content) if request.content else None))
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

