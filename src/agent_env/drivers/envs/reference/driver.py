"""The reference computer driver, for an X11 desktop: mss captures the screen and XTEST, through python-xlib, sends the
input. X calls run off the event loop, one at a time, on a connection checked before each call and reopened when it
drops, so the driver outlives a restarted X server. It records the display through FFmpeg's ``x11grab``, by PyAV. It
needs ``agentenv-framework-protocol[computer]``, ``mss`` and ``python-xlib`` where it runs."""
from __future__ import annotations

import asyncio
import contextlib
import math
import os
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Iterator

import av
import mss
from Xlib import X, XK
from Xlib import display as xdisplay
from Xlib.error import ConnectionClosedError, DisplayConnectionError
from Xlib.ext import xtest

from agentenv_protocol.computer import (
    ComputerDriverEnvironment,
    DeliveryFailed,
    DeviceUnavailable,
    Display,
    Image,
    ReadyCheck,
    RecordedFrame,
    RecordHints,
    Recording,
    Unsupported,
    capture,
    click,
    computer_driver,
    display,
    drag,
    frames_in_thread,
    key_down,
    key_up,
    pointer_down,
    pointer_move,
    pointer_position,
    pointer_up,
    press,
    record,
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
#: The pause between the presses of a double or triple click, in seconds.
CLICK_GAP_S = 0.04
#: How far apart a drag's pointer moves are, in pixels.
DRAG_STEP_PX = 20
DROPPED = (ConnectionClosedError, DisplayConnectionError, OSError)


@computer_driver(name="reference",
                 supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
class ReferenceDriver(ComputerDriverEnvironment):
    buttons = ("left", "right", "middle", "back", "forward")
    max_click_count = 3
    max_drag_points = 50
    recording = Recording(max_fps=30)

    def __init__(self) -> None:
        super().__init__()
        self._x: _Connection | None = None
        self._x_lock = threading.Lock()
        #: The keycodes each key held by @key_down pressed, which @key_up releases.
        self._held: dict[str, list[int]] = {}

    @display
    async def screen(self) -> Display:
        def read(c):
            s = c.d.screen()
            return Display(s.width_in_pixels, s.height_in_pixels, 1.0)
        return await self._read(read)

    @capture
    async def grab(self) -> Image:
        return await asyncio.to_thread(_grab)

    @record
    async def stream(self, hints: RecordHints) -> AsyncIterator[RecordedFrame]:
        """The display through FFmpeg's ``x11grab``. Its frames carry the wall clock, so each one's time is when it
        was captured."""
        size = await self.screen()
        options = {"framerate": str(hints.fps), "video_size": f"{size.width}x{size.height}",
                   "draw_mouse": "1" if hints.cursor else "0"}

        def frames() -> Iterator[RecordedFrame]:
            with av.open(os.environ.get("DISPLAY", ":0"), format="x11grab", options=options) as source:
                for frame in source.decode(video=0):
                    yield RecordedFrame(frame, datetime.fromtimestamp(frame.time, timezone.utc))

        async for frame in frames_in_thread(frames):
            yield frame

    async def ready_checks(self) -> list[ReadyCheck]:
        try:
            has = await self._read(lambda c: c.d.has_extension("XTEST"))
        except DeviceUnavailable as e:
            return [ReadyCheck("xtest", False, e.message)]
        return [ReadyCheck("xtest", bool(has), "XTEST available" if has else "XTEST missing")]

    @click
    async def click_at(self, x, y, button, count, modifiers) -> None:
        syms = [_keysym(k) for k in modifiers]

        def act(c):
            held = c.hold(syms)
            c.move(x, y)
            for i in range(count):
                c.button(button, True)
                c.button(button, False)
                if i + 1 < count:
                    c.d.sync()
                    time.sleep(CLICK_GAP_S)
            c.release(reversed(held))
        await self._send(act)

    @drag
    async def drag_along(self, path, button, duration_ms, modifiers) -> None:
        syms = [_keysym(k) for k in modifiers]
        steps = _steps(path)
        pause = duration_ms / 1000 / len(steps)
        held: list[int] = []

        def start(c):
            held.extend(c.hold(syms))
            c.move(*path[0])
            c.button(button, True)

        def finish(c):
            c.button(button, False)
            c.release(reversed(held))

        try:
            await self._send(start)
            for px, py in steps:
                await self._send(lambda c, px=px, py=py: c.move(px, py))
                await asyncio.sleep(pause)
        except BaseException:
            with contextlib.suppress(Exception):
                await self._send(finish)
            raise
        await self._send(finish)

    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers) -> None:
        syms = [_keysym(k) for k in modifiers]

        def act(c):
            held = c.hold(syms)
            c.move(x, y)
            for _ in range(amount):
                xtest.fake_input(c.d, X.ButtonPress, SCROLL[direction])
                xtest.fake_input(c.d, X.ButtonRelease, SCROLL[direction])
            c.release(reversed(held))
        await self._send(act)

    @pointer_move
    async def move(self, x, y) -> None:
        await self._send(lambda c: c.move(x, y))

    @pointer_down
    async def down(self, button) -> None:
        await self._send(lambda c: c.button(button, True))

    @pointer_up
    async def up(self, button) -> None:
        await self._send(lambda c: c.button(button, False))

    @pointer_position
    async def where(self) -> tuple[int, int]:
        def read(c):
            p = c.d.screen().root.query_pointer()
            return p.root_x, p.root_y
        return await self._read(read)

    @type_text
    async def type_in(self, text) -> None:
        syms = [XK.string_to_keysym(TYPED[ch]) if ch in TYPED else _keysym(ch) for ch in text]
        await self._read(lambda c: c.check(syms))
        for sym in syms:
            hold = self.settings.tap_hold_ms / 1000
            await self._send(lambda c, sym=sym: c.tap(sym, hold))
            await asyncio.sleep(self.settings.type_delay_ms / 1000)

    @press
    async def press_keys(self, keys) -> None:
        syms = [_keysym(k) for k in keys]
        hold = self.settings.tap_hold_ms / 1000

        def act(c):
            pressed = c.hold(syms)
            c.d.sync()
            time.sleep(hold)
            c.release(reversed(pressed))
        await self._send(act)

    @key_down
    async def keys_down(self, keys) -> None:
        syms = [_keysym(k) for k in keys]

        def act(c):
            for key, pressed in zip(keys, c.hold_each(syms)):
                self._held.setdefault(key, []).extend(pressed)
        await self._send(act)

    @key_up
    async def keys_up(self, keys) -> None:
        def act(c):
            for key in keys:
                c.release(reversed(self._held.pop(key, None) or c.codes(_keysym(key))[-1:]))
        await self._send(act)

    async def _read(self, fn: Callable[[_Connection], Any]) -> Any:
        return await asyncio.to_thread(self._x_call, fn, False)

    async def _send(self, fn: Callable[[_Connection], Any]) -> Any:
        return await asyncio.to_thread(self._x_call, fn, True)

    def _x_call(self, fn: Callable[[_Connection], Any], sends_input: bool) -> Any:
        """``fn`` on a live connection. When the connection drops, a read is retried once on a new one; input is not
        sent again, since some of it may have arrived."""
        with self._x_lock:
            for attempt in (1, 2):
                c = self._connected()
                try:
                    result = fn(c)
                    c.d.sync()
                    return result
                except DROPPED as e:
                    self._x = None
                    if sends_input:
                        raise DeliveryFailed(f"the X connection dropped while sending input: {type(e).__name__}") from None
                    if attempt == 2:
                        raise DeviceUnavailable(f"the X server is unavailable: {type(e).__name__}") from None

    def _connected(self) -> _Connection:
        """The connection, opened or reopened as needed and checked with a round trip before anything is sent."""
        for attempt in (1, 2):
            try:
                if self._x is None:
                    self._x = _Connection()
                    self._held.clear()
                self._x.d.sync()
                return self._x
            except DROPPED as e:
                self._x = None
                if attempt == 2:
                    raise DeviceUnavailable(f"the X server is unavailable: {type(e).__name__}") from None


class _Connection:
    """One connection to the X server, with XTEST input on it. A character the keymap lacks is typed through a spare
    keycode mapped to it, kept for that character and reused; once all are taken, the least recently used one that
    no key holds down is mapped again."""

    def __init__(self) -> None:
        self.d = xdisplay.Display()
        first = self.d.display.info.min_keycode
        mapping = self.d.get_keyboard_mapping(first, self.d.display.info.max_keycode - first + 1)
        self.width = len(mapping[0])
        self.spare = [first + i for i, syms in enumerate(mapping) if not any(syms)]
        self.mapped: OrderedDict[int, int] = OrderedDict()
        self.shift = self.d.keysym_to_keycode(XK.string_to_keysym("Shift_L"))

    def move(self, x, y) -> None:
        xtest.fake_input(self.d, X.MotionNotify, x=int(x), y=int(y))

    def button(self, button: str, down: bool) -> None:
        xtest.fake_input(self.d, X.ButtonPress if down else X.ButtonRelease, BUTTONS[button])

    def hold(self, keysyms: list[int]) -> list[int]:
        """``hold_each``'s keycodes, in the order they were pressed."""
        return [code for pressed in self.hold_each(keysyms) for code in pressed]

    def hold_each(self, keysyms: list[int]) -> list[list[int]]:
        """Press what the keysyms need that isn't down already, Shift first where a character needs it; for each
        keysym, the keycodes this pressed, so the caller releases only those. Every keycode is found before the first
        press, so keysyms that can't all be held at once are refused with nothing sent."""
        down = self._down()
        planned: list[list[int]] = []
        for keysym in keysyms:
            planned.append(self.codes(keysym, busy=down | {code for codes in planned for code in codes}))
        pressed = []
        for codes in planned:
            pressed.append([code for code in codes if code not in down])
            for code in pressed[-1]:
                xtest.fake_input(self.d, X.KeyPress, code)
                down.add(code)
        return pressed

    def release(self, codes) -> None:
        for code in codes:
            xtest.fake_input(self.d, X.KeyRelease, code)

    def tap(self, keysym: int, hold_s: float) -> None:
        """Type one character: its key pressed and released, with Shift around it when it needs Shift and nothing
        holds Shift already."""
        down = self._down()
        *modifiers, code = self.codes(keysym, busy=down)
        pressed = [m for m in modifiers if m not in down]
        for m in pressed:
            xtest.fake_input(self.d, X.KeyPress, m)
        xtest.fake_input(self.d, X.KeyPress, code)
        self.d.sync()
        time.sleep(hold_s)
        xtest.fake_input(self.d, X.KeyRelease, code)
        self.release(reversed(pressed))

    def codes(self, keysym: int, busy: set[int] = frozenset()) -> list[int]:
        """The keycodes that make ``keysym``: Shift and the key when it is the key's second level, else one key; a
        keysym off the keymap is mapped to a spare keycode not in ``busy``."""
        codes = self._on_keymap(keysym)
        if codes is not None:
            return codes
        code = self._free(busy)
        self.d.change_keyboard_mapping(code, [(keysym,) * self.width])
        self.d.sync()
        self.mapped[keysym] = code
        return [code]

    def check(self, keysyms: list[int]) -> None:
        """Refuse, before anything is sent, text with a character off the keymap when no spare keycode is free."""
        if any(self._on_keymap(k) is None for k in keysyms):
            self._free(self._down(), take=False)

    def _free(self, busy: set[int], take: bool = True) -> int:
        """A spare keycode not in ``busy``, the least recently used mapped one when none is unmapped."""
        if self.spare:
            return self.spare.pop() if take else self.spare[-1]
        for keysym, code in self.mapped.items():
            if code not in busy:
                if take:
                    del self.mapped[keysym]
                return code
        raise Unsupported("no spare keycode is free for a character off the X keymap: the server has none, or keys "
                          "hold them all down")

    def _on_keymap(self, keysym: int) -> list[int] | None:
        if keysym in self.mapped:
            self.mapped.move_to_end(keysym)
            return [self.mapped[keysym]]
        code = self.d.keysym_to_keycode(keysym)
        if code and self.d.keycode_to_keysym(code, 0) == keysym:
            return [code]
        if code and self.d.keycode_to_keysym(code, 1) == keysym:
            return [self.shift, code]
        return None

    def _down(self) -> set[int]:
        return {8 * i + bit for i, byte in enumerate(self.d.query_keymap()) for bit in range(8) if byte >> bit & 1}


def _grab() -> Image:
    try:
        with mss.MSS() as m:
            shot = m.grab(m.monitors[0])
            return Image(shot.width, shot.height, bytes(shot.bgra), "BGRA")
    except Exception as e:
        raise DeviceUnavailable(f"capture failed: {type(e).__name__}: {e}") from None


def _steps(path: list) -> list[tuple[float, float]]:
    """The points a drag moves through after its start, about every DRAG_STEP_PX pixels."""
    steps = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / DRAG_STEP_PX))
        steps += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(1, n + 1)]
    return steps


def _keysym(key: str) -> int:
    if key in KEYSYMS:
        return XK.string_to_keysym(KEYSYMS[key])
    if len(key) == 1:
        code = ord(key)
        return code if code < 0x100 else 0x01000000 | code
    raise Unsupported(f"this driver has no X keysym for {key!r}")


if __name__ == "__main__":
    ReferenceDriver().serve()
