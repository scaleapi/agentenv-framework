"""An X11 computer driver: mss captures the screen and XTEST, through python-xlib, sends the input. X calls run off the
event loop, one at a time, and a dropped X connection is reopened on the next call, so the driver outlives a
restarted X server."""
from __future__ import annotations

import asyncio
import math
import threading
import time
from typing import Any, Callable

import mss
from Xlib import X, XK
from Xlib import display as xdisplay
from Xlib.error import ConnectionClosedError, DisplayConnectionError
from Xlib.ext import xtest

from agentenv_protocol.computer import (
    ComputerDriverEnvironment,
    DeviceUnavailable,
    Display,
    Image,
    ReadyCheck,
    Unsupported,
    capture,
    click,
    computer_driver,
    display,
    drag,
    key_down,
    key_up,
    pointer_down,
    pointer_move,
    pointer_position,
    pointer_up,
    press,
    scroll,
    type_text,
)

BUTTONS = {"left": 1, "middle": 2, "right": 3, "back": 8, "forward": 9}
SCROLL = {"up": 4, "down": 5, "left": 6, "right": 7}
#: W3C key values and their X keysym names; any other single character is typed as its own keysym.
KEYSYMS = {
    "Enter": "Return", "Tab": "Tab", "Escape": "Escape", "Backspace": "BackSpace", "Delete": "Delete",
    "ArrowUp": "Up", "ArrowDown": "Down", "ArrowLeft": "Left", "ArrowRight": "Right", "Home": "Home", "End": "End",
    "PageUp": "Prior", "PageDown": "Next", "Insert": "Insert", "Control": "Control_L", "Shift": "Shift_L",
    "Alt": "Alt_L", "Meta": "Super_L", "Super": "Super_L", "CapsLock": "Caps_Lock", "ContextMenu": "Menu",
    "PrintScreen": "Print", "Pause": "Pause", " ": "space", **{f"F{n}": f"F{n}" for n in range(1, 25)},
}
TYPED = {"\n": "Return", "\t": "Tab"}
DROPPED = (ConnectionClosedError, DisplayConnectionError, OSError)


@computer_driver(name="x11-desktop",
                 supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
class X11Driver(ComputerDriverEnvironment):
    buttons = ("left", "right", "middle", "back", "forward")
    max_click_count = 3
    max_drag_points = 50

    def __init__(self) -> None:
        super().__init__()
        self._x = None
        self._x_lock = threading.Lock()

    @display
    async def screen(self) -> Display:
        def read(d):
            s = d.screen()
            return Display(s.width_in_pixels, s.height_in_pixels, 1.0)
        return await self._on_x(read)

    @capture
    async def grab(self) -> Image:
        return await asyncio.to_thread(_grab)

    async def ready_checks(self) -> list[ReadyCheck]:
        try:
            has = await self._on_x(lambda d: d.has_extension("XTEST"))
        except DeviceUnavailable as e:
            return [ReadyCheck("xtest", False, e.message)]
        return [ReadyCheck("xtest", bool(has), "XTEST available" if has else "XTEST missing")]

    @click
    async def click_at(self, x, y, button, count, modifiers) -> None:
        def act(d):
            held = _press(d, modifiers)
            _move(d, x, y)
            for i in range(count):
                _button(d, button, True)
                _button(d, button, False)
                if i + 1 < count:
                    d.sync()
                    time.sleep(0.04)
            _release(d, reversed(held))
        await self._on_x(act)

    @drag
    async def drag_along(self, path, button, duration_ms, modifiers) -> None:
        steps = _steps(path)
        pause = duration_ms / 1000 / len(steps)

        def start(d):
            held = _press(d, modifiers)
            _move(d, *path[0])
            _button(d, button, True)
            return held

        def finish(d):
            _button(d, button, False)
            _release(d, reversed(held))

        held = await self._on_x(start)
        for x, y in steps:
            await self._on_x(lambda d, x=x, y=y: _move(d, x, y))
            await asyncio.sleep(pause)
        await self._on_x(finish)

    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers) -> None:
        def act(d):
            held = _press(d, modifiers)
            _move(d, x, y)
            for _ in range(amount):
                xtest.fake_input(d, X.ButtonPress, SCROLL[direction])
                xtest.fake_input(d, X.ButtonRelease, SCROLL[direction])
            _release(d, reversed(held))
        await self._on_x(act)

    @pointer_move
    async def move(self, x, y) -> None:
        await self._on_x(lambda d: _move(d, x, y))

    @pointer_down
    async def down(self, button) -> None:
        await self._on_x(lambda d: _button(d, button, True))

    @pointer_up
    async def up(self, button) -> None:
        await self._on_x(lambda d: _button(d, button, False))

    @pointer_position
    async def where(self) -> tuple[int, int]:
        def read(d):
            p = d.screen().root.query_pointer()
            return p.root_x, p.root_y
        return await self._on_x(read)

    @type_text
    async def type_in(self, text) -> None:
        for ch in text:
            sym = XK.string_to_keysym(TYPED[ch]) if ch in TYPED else _keysym(ch)
            hold = self.settings.tap_hold_ms / 1000
            await self._on_x(lambda d, sym=sym: _tap(d, sym, hold))
            await asyncio.sleep(self.settings.type_delay_ms / 1000)

    @press
    async def press_keys(self, keys) -> None:
        hold = self.settings.tap_hold_ms / 1000

        def act(d):
            codes = _press(d, keys)
            d.sync()
            time.sleep(hold)
            _release(d, reversed(codes))
        await self._on_x(act)

    @key_down
    async def keys_down(self, keys) -> None:
        await self._on_x(lambda d: _press(d, keys))

    @key_up
    async def keys_up(self, keys) -> None:
        await self._on_x(lambda d: _release(d, _keycodes(d, keys)))

    async def _on_x(self, fn: Callable[[Any], Any]) -> Any:
        return await asyncio.to_thread(self._x_call, fn)

    def _x_call(self, fn: Callable[[Any], Any]) -> Any:
        """``fn(display)`` on a live connection, reconnecting once if it dropped."""
        with self._x_lock:
            for attempt in (1, 2):
                try:
                    if self._x is None:
                        self._x = xdisplay.Display()
                    result = fn(self._x)
                    self._x.sync()
                    return result
                except DROPPED as e:
                    self._x = None
                    if attempt == 2:
                        raise DeviceUnavailable(f"the X server is unavailable: {type(e).__name__}") from None


def _grab() -> Image:
    try:
        with mss.MSS() as m:
            shot = m.grab(m.monitors[0])
            return Image(shot.width, shot.height, bytes(shot.bgra), "BGRA")
    except Exception as e:
        raise DeviceUnavailable(f"capture failed: {type(e).__name__}: {e}") from None


def _steps(path: list) -> list[tuple[float, float]]:
    """The points a drag moves through after its start, about every 20 pixels."""
    steps = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / 20))
        steps += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(1, n + 1)]
    return steps


def _move(d, x, y) -> None:
    xtest.fake_input(d, X.MotionNotify, x=int(x), y=int(y))


def _button(d, button: str, down: bool) -> None:
    xtest.fake_input(d, X.ButtonPress if down else X.ButtonRelease, BUTTONS[button])


def _press(d, keys: list[str]) -> list[int]:
    codes = _keycodes(d, keys)
    for code in codes:
        xtest.fake_input(d, X.KeyPress, code)
    return codes


def _release(d, codes) -> None:
    for code in codes:
        xtest.fake_input(d, X.KeyRelease, code)


def _tap(d, keysym: int, hold_s: float) -> None:
    code, shifted = _keycode(d, keysym)
    shift = d.keysym_to_keycode(XK.string_to_keysym("Shift_L"))
    if shifted:
        xtest.fake_input(d, X.KeyPress, shift)
    xtest.fake_input(d, X.KeyPress, code)
    d.sync()
    time.sleep(hold_s)
    xtest.fake_input(d, X.KeyRelease, code)
    if shifted:
        xtest.fake_input(d, X.KeyRelease, shift)


def _keycodes(d, keys: list[str]) -> list[int]:
    return [_keycode(d, _keysym(k))[0] for k in keys]


def _keysym(key: str) -> int:
    if key in KEYSYMS:
        return XK.string_to_keysym(KEYSYMS[key])
    if len(key) == 1:
        code = ord(key)
        return code if code < 0x100 else 0x01000000 | code
    raise Unsupported(f"this driver has no X keysym for {key!r}")


def _keycode(d, keysym: int) -> tuple[int, bool]:
    """The keycode for ``keysym`` and whether it needs Shift; with no keycode for it, a spare one is mapped to it."""
    code = d.keysym_to_keycode(keysym)
    if code:
        return code, d.keycode_to_keysym(code, 0) != keysym and d.keycode_to_keysym(code, 1) == keysym
    first = d.display.info.min_keycode
    mapping = d.get_keyboard_mapping(first, d.display.info.max_keycode - first + 1)
    spare = next((first + i for i, syms in enumerate(mapping) if not any(syms)), None)
    if spare is None:
        raise Unsupported(f"no spare keycode to map keysym {keysym:#x} to")
    d.change_keyboard_mapping(spare, [(keysym,) * len(mapping[0])])
    d.sync()
    return spare, False


if __name__ == "__main__":
    X11Driver().serve()
