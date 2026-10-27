"""The reference driver against fake mss and Xlib modules: it passes the definition checks, serves the whole computer
contract, and turns tool calls into the XTEST events they mean, without resending input, leaving keys or buttons down,
or letting go of keys it didn't press."""
from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import sys
import types
from pathlib import Path

import httpx
import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import agent_env
from agentenv_protocol.computer import COMPUTER_EXEC, COMPUTER_KEYBOARD, COMPUTER_POINTER, Platform

DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "reference" / "driver.py"
LINUX = Platform(os="linux", distribution="ubuntu", version="24.04", arch="amd64", display="x11")
WIDTH, HEIGHT = 640, 400
TOKYO = [0x01000000 | ord(ch) for ch in "東京"]
#: The fake keymap puts the named keys, such as Shift_L, from this keycode up; printable ASCII is its keysym + 8.
NAMED_KEYS = 200


class ConnectionClosedError(Exception):
    pass


class FakeX:
    """An X display: its screen and pointer, a US-like keymap whose capital letters are the second level of their
    keys, the keys down, the XTEST events sent, ``drops`` syncs and the ``fail_at``-th event that fail, and the
    ``spare`` keycodes the keymap leaves empty."""

    def __init__(self) -> None:
        self.events: list = []
        self.down: set[int] = set()
        self.remapped: list = []
        self.pointer = (0, 0)
        self.opened = 0
        self.drops = 0
        self.fail_at = None
        self.spare = {8}
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
        if ord("A") <= keysym <= ord("Z"):
            keysym += 0x20
        if 0xff00 <= keysym < 0xff38:
            return NAMED_KEYS + keysym - 0xff00
        return keysym + 8 if 0x20 <= keysym < 0x7f else 0

    def keycode_to_keysym(self, code: int, index: int) -> int:
        sym = 0xff00 + code - NAMED_KEYS if code >= NAMED_KEYS else code - 8
        if index == 0:
            return sym
        return sym - 0x20 if index == 1 and ord("a") <= sym <= ord("z") else 0

    def get_keyboard_mapping(self, first: int, count: int) -> list:
        return [(0, 0) if first + i in self.spare else (1, 0) for i in range(count)]

    def change_keyboard_mapping(self, first: int, keysyms: list) -> None:
        self.remapped.append((first, keysyms[0][0]))

    def query_keymap(self) -> list[int]:
        return [sum(1 << bit for bit in range(8) if 8 * i + bit in self.down) for i in range(32)]


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
    """The example's driver on a fake X display, and the keycodes the fake keymap gives named keys."""
    fake, keysyms = FakeX(), {}

    def open_display():
        fake.opened += 1
        return fake

    def fake_input(d, kind, detail=0, x=0, y=0):
        if d.fail_at is not None:
            d.fail_at -= 1
            if d.fail_at == 0:
                d.fail_at = None
                raise ConnectionClosedError("display")
        d.events.append((kind, (x, y)) if kind == "move" else (kind, detail))
        if kind == "key-down":
            d.down.add(detail)
        elif kind == "key-up":
            d.down.discard(detail)

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
    spec = importlib.util.spec_from_file_location("reference_driver", DRIVER)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    return example.ReferenceDriver(), fake, lambda name: fake.keysym_to_keycode(XK.string_to_keysym(name))


@contextlib.asynccontextmanager
async def mcp_session(driver):
    async with create_connected_server_and_client_session(driver.mcp or driver.create_app()) as session:
        yield session


@contextlib.asynccontextmanager
async def http_client(driver):
    app = (driver.mcp or driver.create_app()).streamable_http_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        yield client


async def _events(fake: FakeX, session, tool: str, args: dict) -> list:
    fake.events.clear()
    result = await session.call_tool(tool, args)
    assert not result.isError, result.content[0].text
    return list(fake.events)


@pytest.mark.asyncio
async def test_the_example_serves_the_whole_computer_contract(x11):
    driver, _fake, _code = x11
    async with http_client(driver) as http, mcp_session(driver) as session:
        card = (await http.get("/.well-known/agent-env.json")).json()
        tools = {t.name for t in (await session.list_tools()).tools}
        ready = (await http.get("/agentenv/ext/computer-ready/status")).json()
    extensions = {e["uri"]: e for e in card["capabilities"]["extensions"]}
    assert card["name"] == "reference" and len(tools) == 14 and "computer_keyboard_down" in tools
    assert {COMPUTER_POINTER, COMPUTER_KEYBOARD, COMPUTER_EXEC} <= set(extensions)
    assert extensions["urn:agentenv:computer/v1"]["params"]["max_click_count"] == 3
    assert ready["ready"] and [c["name"] for c in ready["checks"]] == ["display", "capture", "xtest"]


@pytest.mark.asyncio
async def test_tool_calls_become_the_xtest_events_they_mean(x11):
    driver, fake, code = x11
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        clicked = await _events(fake, session, "computer_click",
                                {"x": 10, "y": 20, "button": "right", "modifiers": ["Shift"]})
        typed = await _events(fake, session, "computer_type", {"text": "aA\n"})
        dragged = await _events(fake, session, "computer_drag", {"path": [[0, 0], [40, 0]], "duration_ms": 0})
        chord = await _events(fake, session, "computer_key", {"keys": ["Control", "s"]})
        capital = await _events(fake, session, "computer_key", {"keys": ["A"]})
    shift, a = code("Shift_L"), ord("a") + 8
    assert clicked == [("key-down", shift), ("move", (10, 20)), ("button-down", 3), ("button-up", 3), ("key-up", shift)]
    assert typed == [("key-down", a), ("key-up", a), ("key-down", shift), ("key-down", a), ("key-up", a),
                     ("key-up", shift), ("key-down", code("Return")), ("key-up", code("Return"))]
    assert dragged == [("move", (0, 0)), ("button-down", 1), ("move", (20, 0)), ("move", (40, 0)), ("button-up", 1)]
    assert chord == [("key-down", code("Control_L")), ("key-down", ord("s") + 8), ("key-up", ord("s") + 8),
                     ("key-up", code("Control_L"))]
    assert capital == [("key-down", shift), ("key-down", a), ("key-up", a), ("key-up", shift)]
    assert not fake.down


@pytest.mark.asyncio
async def test_keys_held_by_keyboard_down_stay_down_through_other_input(x11):
    driver, fake, code = x11
    shift, a = code("Shift_L"), ord("a") + 8
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        held = await _events(fake, session, "computer_keyboard_down", {"keys": ["Shift"]})
        typed = await _events(fake, session, "computer_type", {"text": "A"})
        clicked = await _events(fake, session, "computer_click", {"x": 1, "y": 1, "modifiers": ["Shift"]})
        chord = await _events(fake, session, "computer_key", {"keys": ["Shift", "a"]})
        released = await _events(fake, session, "computer_keyboard_up", {})
    assert held == [("key-down", shift)]
    assert typed == [("key-down", a), ("key-up", a)] and chord == [("key-down", a), ("key-up", a)]
    assert clicked == [("move", (1, 1)), ("button-down", 1), ("button-up", 1)]
    assert released == [("key-up", shift)] and not fake.down


@pytest.mark.asyncio
async def test_characters_off_the_keymap_share_the_spare_keycodes(x11):
    driver, fake, _code = x11
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        repeated = await _events(fake, session, "computer_type", {"text": "東東"})
        mapped_once = list(fake.remapped)
        recycled = await _events(fake, session, "computer_type", {"text": "京東"})
    assert repeated == [("key-down", 8), ("key-up", 8)] * 2 and mapped_once == [(8, TOKYO[0])]
    assert recycled == [("key-down", 8), ("key-up", 8)] * 2
    assert fake.remapped == [(8, TOKYO[0]), (8, TOKYO[1]), (8, TOKYO[0])]


@pytest.mark.asyncio
async def test_a_held_characters_keycode_is_not_taken_by_another(x11):
    driver, fake, _code = x11
    fake.spare = {8, 9}
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        held = await _events(fake, session, "computer_keyboard_down", {"keys": ["東"]})
        fake.events.clear()
        chord = await session.call_tool("computer_key", {"keys": ["京", "日"]})
        held_too = await session.call_tool("computer_keyboard_down", {"keys": ["京", "日"]})
        refused = list(fake.events)
        typed = await _events(fake, session, "computer_type", {"text": "京"})
        released = await _events(fake, session, "computer_keyboard_up", {})
    assert held == [("key-down", 9)] and typed == [("key-down", 8), ("key-up", 8)]
    for call in (chord, held_too):
        assert call.structuredContent["code"] == "unsupported" and call.structuredContent["applied"] == "no"
    assert refused == [] and released == [("key-up", 9)] and not fake.down


@pytest.mark.asyncio
async def test_text_that_cant_be_typed_whole_is_refused_before_any_of_it(x11):
    driver, fake, _code = x11
    fake.spare = set()
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        fake.events.clear()
        refused = await session.call_tool("computer_type", {"text": "a東"})
    assert refused.structuredContent["code"] == "unsupported" and refused.structuredContent["applied"] == "no"
    assert fake.events == []


@pytest.mark.asyncio
async def test_input_is_not_sent_again_when_the_connection_drops_during_it(x11):
    driver, fake, _code = x11
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        fake.events.clear()
        fake.fail_at = 2
        dropped = await session.call_tool("computer_click", {"x": 5, "y": 5})
    assert dropped.structuredContent["code"] == "delivery_failed"
    assert dropped.structuredContent["applied"] == "unknown"
    assert fake.events == [("move", (5, 5))] and fake.opened == 1


@pytest.mark.asyncio
async def test_a_drag_that_fails_or_is_cancelled_lets_go_of_its_button_and_modifiers(x11):
    driver, fake, code = x11
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        fake.events.clear()
        fake.fail_at = 4
        failed = await session.call_tool("computer_drag", {"path": [[0, 0], [40, 0]], "modifiers": ["Shift"]})
    assert failed.structuredContent["code"] == "delivery_failed"
    assert fake.events[-2:] == [("button-up", 1), ("key-up", code("Shift_L"))] and not fake.down
    fake.events.clear()
    dragging = asyncio.ensure_future(driver.drag_along([[0, 0], [400, 0]], "left", 10_000, ["Control"]))
    await asyncio.sleep(0.2)
    dragging.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dragging
    assert fake.events[-2:] == [("button-up", 1), ("key-up", code("Control_L"))] and not fake.down


@pytest.mark.asyncio
async def test_a_dropped_x_connection_is_reopened_and_a_dead_one_is_device_unavailable(x11):
    driver, fake, _code = x11
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        fake.drops = 1
        moved = await session.call_tool("computer_pointer_position", {})
        reopened = fake.opened
        fake.drops = 2
        refused = await session.call_tool("computer_pointer_position", {})
    assert not moved.isError and reopened == 2
    assert refused.structuredContent["code"] == "device_unavailable"
    assert "the X server is unavailable: ConnectionClosedError" in refused.structuredContent["message"]
