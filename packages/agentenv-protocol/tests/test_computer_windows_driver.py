"""WindowsDriver, the computer/v1 operations on Windows, against a fake user32 that decodes the SendInput events it's
sent: moves aimed at the pixel asked for on the virtual screen, buttons and wheel notches, keys with their scan codes
and extended flags, characters by their keyboard layout's keys or as themselves, held keys released only by what
pressed them, and a refusal when Windows takes fewer events than it was sent."""
from __future__ import annotations

import ctypes
import importlib
import sys
import types

import pytest

pytest.importorskip("mcp.server.fastmcp")

from agentenv_protocol.computer import (  # noqa: E402
    ComputerSettings,
    DeliveryFailed,
    Image,
    Platform,
    Unsupported,
    computer_driver,
)
from agentenv_protocol.computer import _desktop  # noqa: E402

WINDOWS = Platform(os="windows", arch="amd64", display="dwm")
SHIFT, CONTROL, A, ONE = 0xA0, 0xA2, 0x41, 0x31
MOVE_ABSOLUTE = 0x0001 | 0x8000 | 0x4000
LEFT_DOWN, LEFT_UP, X_DOWN, X_UP, WHEEL, HWHEEL = 0x0002, 0x0004, 0x0080, 0x0100, 0x0800, 0x1000
EXTENDED, KEY_UP, UNICODE = 0x0001, 0x0002, 0x0004
#: A US layout's VkKeyScanW: a letter's virtual key, Shift (bit 8) for capitals and shifted digits.
US = {**{ch: ord(ch.upper()) for ch in "abcdefghijklmnopqrstuvwxyz0123456789 "},
      **{ch: 0x100 | ord(ch) for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}, "!": 0x100 | ONE, "/": 0xBF}


class FakeUser32:
    def __init__(self) -> None:
        self.sent: list[tuple] = []
        self.screen = (0, 0, 1920, 1080)
        self.cursor = (0, 0)
        self.takes: int | None = None
        self.desktop = 7
        self.closed: list[int] = []

    def GetSystemMetrics(self, index: int) -> int:
        return dict(zip((76, 77, 78, 79), self.screen))[index]

    def SendInput(self, count, events, size) -> int:
        assert size == ctypes.sizeof(windows._Input)
        for event in events[:count]:
            if event.type == 0:
                mi = event.u.mi
                self.sent.append(("mouse", mi.dx, mi.dy, mi.mouseData, mi.dwFlags))
            else:
                ki = event.u.ki
                self.sent.append(("key", ki.wVk, ki.wScan, ki.dwFlags))
        return count if self.takes is None else self.takes

    def VkKeyScanW(self, ch: str) -> int:
        return US.get(ch, -1)

    def MapVirtualKeyW(self, vk: int, kind: int) -> int:
        return 0x1000 + vk

    def OpenInputDesktop(self, flags, inherit, access) -> int:
        return self.desktop

    def CloseDesktop(self, handle) -> None:
        self.closed.append(handle)

    def GetCursorPos(self, point) -> bool:
        point._obj.x, point._obj.y = self.cursor
        return True


class FakeMSS:
    monitors = [{"left": 0, "top": 0, "width": 2, "height": 1}]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def grab(self, monitor):
        return types.SimpleNamespace(width=2, height=1, bgra=bytearray([1, 2, 3, 255, 4, 5, 6, 255]))


windows: types.ModuleType


@pytest.fixture
def user32(monkeypatch) -> FakeUser32:
    global windows
    monkeypatch.setitem(sys.modules, "mss", types.SimpleNamespace(MSS=FakeMSS))
    monkeypatch.delitem(sys.modules, "agentenv_protocol.computer._windows", raising=False)
    windows = importlib.import_module("agentenv_protocol.computer._windows")
    fake = FakeUser32()
    monkeypatch.setattr(windows, "_user32", lambda: fake)
    monkeypatch.setattr(windows, "_last_error", lambda: "Access is denied.")
    monkeypatch.setattr(windows, "CLICK_GAP_S", 0)
    return fake


@pytest.fixture
def desk(user32):
    @computer_driver(name="desk", supports=[{"os": {"type": "windows", "display": ["dwm"]}, "arch": ["amd64"]}])
    class Desk(windows.WindowsDriver):
        pass

    return Desk(ComputerSettings(tap_hold_ms=0, type_delay_ms=0), WINDOWS)


def keys(sent: list[tuple]) -> list[tuple]:
    return [(vk, flags) for kind, vk, _scan, flags in (e[:4] for e in sent) if kind == "key"]


def pixel(dx: int, dy: int, user32: FakeUser32) -> tuple[int, int]:
    """Where Windows puts a normalized absolute move: its 0 to 65535 range spread over the virtual screen."""
    left, top, width, height = user32.screen
    return left + dx * width // 65536, top + dy * height // 65536


@pytest.mark.asyncio
async def test_typed_text_goes_out_as_its_layouts_keys_and_text_off_the_layout_as_its_characters(desk, user32):
    await desk.type_in("aB\n\tÉ😀")

    assert [(vk, scan, flags) for _, vk, scan, flags in user32.sent] == [
        (A, 0x1041, 0), (A, 0x1041, KEY_UP),
        (SHIFT, 0x10A0, 0), (0x42, 0x1042, 0), (0x42, 0x1042, KEY_UP), (SHIFT, 0x10A0, KEY_UP),
        (0x0D, 0x100D, 0), (0x0D, 0x100D, KEY_UP),
        (0x09, 0x1009, 0), (0x09, 0x1009, KEY_UP),
        (0, ord("É"), UNICODE), (0, ord("É"), UNICODE | KEY_UP),
        (0, 0xD83D, UNICODE), (0, 0xDE00, UNICODE), (0, 0xD83D, UNICODE | KEY_UP), (0, 0xDE00, UNICODE | KEY_UP),
    ]


@pytest.mark.asyncio
async def test_a_key_held_by_key_down_stays_down_while_text_is_typed_and_isnt_pressed_again(desk, user32):
    await desk.keys_down(["Shift"])
    await desk.type_in("aB")
    await desk.click_at(1, 1, "left", 1, ["Shift"])
    await desk.keys_up(["Shift"])
    await desk.type_in("d")

    assert keys(user32.sent) == [(SHIFT, 0), (A, 0), (A, KEY_UP), (0x42, 0), (0x42, KEY_UP), (SHIFT, KEY_UP),
                                 (0x44, 0), (0x44, KEY_UP)]


@pytest.mark.asyncio
async def test_a_chord_presses_the_keys_its_layout_gives_shift_once_and_releases_them_in_reverse(desk, user32):
    await desk.press_keys(["Control", "A"])
    await desk.press_keys(["!"])
    await desk.press_keys(["ArrowLeft"])

    assert keys(user32.sent) == [
        (CONTROL, 0), (SHIFT, 0), (A, 0), (A, KEY_UP), (SHIFT, KEY_UP), (CONTROL, KEY_UP),
        (SHIFT, 0), (ONE, 0), (ONE, KEY_UP), (SHIFT, KEY_UP),
        (0x25, EXTENDED), (0x25, EXTENDED | KEY_UP),
    ]
    assert user32.sent[-1][2] == 0x1025
    with pytest.raises(Unsupported, match="no key on this keyboard layout"):
        await desk.press_keys(["é"])
    with pytest.raises(Unsupported, match="no Windows virtual key"):
        await desk.press_keys(["Hyper"])


@pytest.mark.asyncio
async def test_a_move_lands_on_the_pixel_asked_for_across_the_virtual_screen(desk, user32):
    user32.screen = (-1920, 0, 3840, 1080)
    for x, y in ((0, 0), (1, 1), (1919, 540), (1920, 0), (3839, 1079)):
        await desk.move(x, y)
    moves = [pixel(dx, dy, user32) for _, dx, dy, _, flags in user32.sent if flags == MOVE_ABSOLUTE]

    assert moves == [(-1920, 0), (-1919, 1), (-1, 540), (0, 0), (1919, 1079)]
    user32.cursor = (-1000, 10)
    assert await desk.where() == (920, 10)
    assert (await desk.screen()).width == 3840


@pytest.mark.asyncio
async def test_a_click_holds_its_modifiers_around_its_presses_at_the_point(desk, user32):
    await desk.click_at(100, 50, "left", 2, ["Shift"])
    await desk.click_at(5, 5, "back", 1, [])

    kinds = [e[4] if e[0] == "mouse" else ("key", e[1], e[3]) for e in user32.sent]
    assert kinds == [("key", SHIFT, 0), MOVE_ABSOLUTE, LEFT_DOWN, LEFT_UP, LEFT_DOWN, LEFT_UP, ("key", SHIFT, KEY_UP),
                     MOVE_ABSOLUTE, X_DOWN, X_UP]
    assert pixel(*user32.sent[1][1:3], user32) == (100, 50)
    assert [e[3] for e in user32.sent[-2:]] == [1, 1]


@pytest.mark.asyncio
async def test_a_scroll_sends_a_wheel_notch_per_step_forward_for_up_and_right(desk, user32):
    await desk.scroll_at(7, 8, "down", 3, [])
    await desk.scroll_at(7, 8, "right", 1, [])

    wheels = [(e[3], e[4]) for e in user32.sent if e[4] in (WHEEL, HWHEEL)]
    assert wheels == [(ctypes.c_ulong(-120).value, WHEEL)] * 3 + [(120, HWHEEL)]


@pytest.mark.asyncio
async def test_a_drag_presses_at_its_start_moves_through_its_waypoints_and_releases_at_its_end(desk, user32):
    await desk.drag_along([(0, 0), (40, 0), (40, 30)], "left", 0, [])

    flags = [e[4] for e in user32.sent]
    assert flags[:2] == [MOVE_ABSOLUTE, LEFT_DOWN] and flags[-1] == LEFT_UP
    moved = [pixel(e[1], e[2], user32) for e in user32.sent[2:-1]]
    assert (40, 0) in moved and moved[-1] == (40, 30)
    assert all(abs(x1 - x0) + abs(y1 - y0) <= _desktop.DRAG_STEP_PX for (x0, y0), (x1, y1) in zip(moved, moved[1:]))


@pytest.mark.asyncio
async def test_a_held_key_is_released_only_by_what_pressed_it(desk, user32):
    await desk.keys_down(["Shift"])
    await desk.keys_down(["A"])
    await desk.keys_up(["A"])
    await desk.keys_up(["Shift"])

    assert keys(user32.sent) == [(SHIFT, 0), (A, 0), (A, KEY_UP), (SHIFT, KEY_UP)]


@pytest.mark.asyncio
async def test_a_key_held_twice_is_pressed_once_and_one_another_holds_isnt_released_with_it(desk, user32):
    await desk.keys_down(["a", "a"])
    await desk.keys_up(["a"])
    await desk.keys_down(["A"])
    await desk.keys_down(["Shift"])
    await desk.keys_up(["Shift"])
    await desk.keys_up(["A"])

    assert keys(user32.sent) == [(A, 0), (A, KEY_UP), (SHIFT, 0), (A, 0), (A, KEY_UP), (SHIFT, KEY_UP)]


@pytest.mark.asyncio
async def test_a_refused_release_keeps_its_keys_held_so_the_call_can_be_made_again(desk, user32):
    await desk.keys_down(["A"])
    user32.takes = 0
    with pytest.raises(DeliveryFailed):
        await desk.keys_up(["A"])
    user32.takes = None
    user32.sent.clear()
    await desk.keys_up(["A"])

    assert keys(user32.sent) == [(A, KEY_UP), (SHIFT, KEY_UP)]


@pytest.mark.asyncio
async def test_input_windows_takes_only_part_of_is_refused(desk, user32):
    user32.takes = 0
    with pytest.raises(DeliveryFailed, match="Windows took 0 of 3 input events: Access is denied"):
        await desk.press_keys(["Control", "A"])


@pytest.mark.asyncio
async def test_computer_ready_says_whether_the_input_desktop_is_this_sessions(desk, user32):
    (ready,) = await desk.ready_checks()
    user32.desktop = 0
    (locked,) = await desk.ready_checks()

    assert ready.ok and user32.closed == [7]
    assert not locked.ok and "Access is denied" in locked.detail


@pytest.mark.asyncio
async def test_a_capture_is_the_virtual_screen_as_bgra(desk):
    assert await desk.grab() == Image(2, 1, bytes([1, 2, 3, 255, 4, 5, 6, 255]), "BGRA")
