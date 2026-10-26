"""The X11 example driver against fake mss and Xlib modules: it passes the definition checks, serves the whole
computer contract, and turns tool calls into the XTEST events they mean."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import LINUX, http_client, mcp_session  # noqa: E402

from agentenv_protocol.computer import COMPUTER_EXEC, COMPUTER_KEYBOARD, COMPUTER_POINTER  # noqa: E402

DRIVER = Path(__file__).resolve().parents[1] / "examples" / "computer_x11" / "driver.py"
WIDTH, HEIGHT = 640, 400


class ConnectionClosedError(Exception):
    pass


class FakeX:
    """An X display: its screen and pointer, the XTEST events sent to it, and ``drops`` syncs that fail."""

    def __init__(self) -> None:
        self.events: list = []
        self.pointer = (0, 0)
        self.opened = 0
        self.drops = 0
        self.remapped = None
        self.display = types.SimpleNamespace(info=types.SimpleNamespace(min_keycode=8, max_keycode=255))

    def screen(self):
        query = lambda: types.SimpleNamespace(root_x=self.pointer[0], root_y=self.pointer[1])  # noqa: E731
        return types.SimpleNamespace(width_in_pixels=WIDTH, height_in_pixels=HEIGHT,
                                     root=types.SimpleNamespace(query_pointer=query))

    def sync(self) -> None:
        if self.drops:
            self.drops -= 1
            raise ConnectionClosedError("display")

    def has_extension(self, name: str) -> bool:
        return name == "XTEST"

    def keysym_to_keycode(self, keysym: int) -> int:
        return keysym + 8 if keysym < 0x10000 else 0

    def keycode_to_keysym(self, code: int, index: int) -> int:
        return code - 8 if index == 0 else 0

    def get_keyboard_mapping(self, first: int, count: int) -> list:
        return [(0, 0)] * count

    def change_keyboard_mapping(self, first: int, keysyms: list) -> None:
        self.remapped = (first, keysyms)


class FakeMSS:
    monitors = [{"left": 0, "top": 0, "width": WIDTH, "height": HEIGHT}]

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def grab(self, monitor: dict):
        return types.SimpleNamespace(width=WIDTH, height=HEIGHT, bgra=bytes(WIDTH * HEIGHT * 4))


@pytest.fixture
def x11(monkeypatch):
    """The example's driver on a fake X display, and the keysyms the fake XK gave out by name."""
    fake, keysyms = FakeX(), {}

    def open_display():
        fake.opened += 1
        return fake

    def fake_input(d, kind, detail=0, x=0, y=0):
        d.events.append((kind, (x, y)) if kind == "move" else (kind, detail))

    def module(name: str, **attrs) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    X = module("Xlib.X", MotionNotify="move", ButtonPress="button-down", ButtonRelease="button-up",
               KeyPress="key-down", KeyRelease="key-up")
    XK = module("Xlib.XK", string_to_keysym=lambda name: keysyms.setdefault(name, 0xff00 + len(keysyms)))
    display = module("Xlib.display", Display=open_display)
    module("Xlib.error", ConnectionClosedError=ConnectionClosedError, DisplayConnectionError=ConnectionClosedError)
    xtest = module("Xlib.ext.xtest", fake_input=fake_input)
    module("Xlib.ext", xtest=xtest)
    module("Xlib", X=X, XK=XK, display=display)
    module("mss", MSS=FakeMSS)
    monkeypatch.setattr("agentenv_protocol.computer.driver.detect", lambda: LINUX)
    monkeypatch.delenv("ENVIRONMENT_NAME", raising=False)
    monkeypatch.delenv("AGENTENV_COMPUTER_SETTINGS", raising=False)
    spec = importlib.util.spec_from_file_location("computer_x11_driver", DRIVER)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example.X11Driver(), fake, keysyms


@pytest.mark.asyncio
async def test_the_example_serves_the_whole_computer_contract(x11):
    driver, _x, _keysyms = x11
    async with http_client(driver) as http, mcp_session(driver) as session:
        card = (await http.get("/.well-known/agent-env.json")).json()
        tools = {t.name for t in (await session.list_tools()).tools}
        ready = (await http.get("/agentenv/ext/computer-ready/status")).json()
    extensions = {e["uri"]: e for e in card["capabilities"]["extensions"]}
    assert card["name"] == "x11-desktop" and len(tools) == 14 and "computer_keyboard_down" in tools
    assert {COMPUTER_POINTER, COMPUTER_KEYBOARD, COMPUTER_EXEC} <= set(extensions)
    assert extensions["urn:agentenv:computer/v1"]["params"]["max_click_count"] == 3
    assert ready["ready"] and [c["name"] for c in ready["checks"]] == ["display", "capture", "xtest"]


@pytest.mark.asyncio
async def test_tool_calls_become_the_xtest_events_they_mean(x11):
    driver, x, keysyms = x11

    async def events(session, tool: str, args: dict) -> list:
        x.events.clear()
        result = await session.call_tool(tool, args)
        assert not result.isError, result.content[0].text
        return list(x.events)

    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        clicked = await events(session, "computer_click", {"x": 10, "y": 20, "button": "right", "modifiers": ["Shift"]})
        typed = await events(session, "computer_type", {"text": "a東\n"})
        dragged = await events(session, "computer_drag", {"path": [[0, 0], [40, 0]], "duration_ms": 0})
        pressed = await events(session, "computer_key", {"keys": ["Control", "s"]})
    shift, control, enter = (keysyms[name] + 8 for name in ("Shift_L", "Control_L", "Return"))
    assert clicked == [("key-down", shift), ("move", (10, 20)), ("button-down", 3), ("button-up", 3), ("key-up", shift)]
    assert typed == [("key-down", 0x69), ("key-up", 0x69), ("key-down", 8), ("key-up", 8),
                     ("key-down", enter), ("key-up", enter)]
    assert x.remapped == (8, [(0x01006771, 0x01006771)])
    assert dragged == [("move", (0, 0)), ("button-down", 1), ("move", (20, 0)), ("move", (40, 0)), ("button-up", 1)]
    assert pressed == [("key-down", control), ("key-down", 0x7b), ("key-up", 0x7b), ("key-up", control)]


@pytest.mark.asyncio
async def test_a_dropped_x_connection_is_reopened_and_a_dead_one_is_device_unavailable(x11):
    driver, x, _keysyms = x11
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        x.drops = 1
        moved = await session.call_tool("computer_pointer_position", {})
        reopened = x.opened
        x.drops = 2
        refused = await session.call_tool("computer_pointer_position", {})
    assert not moved.isError and reopened == 2
    assert refused.structuredContent["code"] == "device_unavailable"
    assert "the X server is unavailable: ConnectionClosedError" in refused.structuredContent["message"]
