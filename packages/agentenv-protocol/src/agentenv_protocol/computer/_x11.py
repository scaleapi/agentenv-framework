"""computer/v1 on an X11 desktop on Linux, for the reference driver and for a driver whose software serves tools of its
own, as agent-desktop's does: mss captures the screen and XTEST, through python-xlib, sends the input. X calls run off
the event loop, one at a time, on a connection checked before each call and reopened when it drops, so the driver
outlives a restarted X server, and input that lets go of a button returns once the server has let go of it. It records
the display through FFmpeg's ``x11grab``, by PyAV. A driver subclasses ``X11Driver``, declares itself with
``computer_driver``, and serves its own tools beside these. Needs the ``x11`` extra: mss and python-xlib."""
from __future__ import annotations

import asyncio
import contextlib
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

from ._desktop import CLICK_GAP_S, MAX_CLICK_COUNT, MAX_DRAG_POINTS, drag_steps, mss_capture
from .driver import ComputerDriverEnvironment
from .errors import DeliveryFailed, DeviceUnavailable, Unsupported
from .operations import (Display, Image, ReadyCheck, capture, click, display, drag, key_down, key_up, pointer_down,
                         pointer_move, pointer_position, pointer_up, press, record, scroll, type_text)
from .platform import Platform
from .recording import RecordedFrame, RecordHints, Recording, frames_in_thread
from .settings import ComputerSettings

BUTTONS = {"left": 1, "middle": 2, "right": 3, "back": 8, "forward": 9}
#: The pointer state's bit for each button that has one.
BUTTON_MASKS = {"left": X.Button1Mask, "middle": X.Button2Mask, "right": X.Button3Mask}
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
#: How long a button's release may take to reach the app, and how often the driver looks.
RELEASE_WAIT_S = 5.0
RELEASE_POLL_S = 0.005
DROPPED = (ConnectionClosedError, DisplayConnectionError, OSError)


class X11Driver(ComputerDriverEnvironment):
    """The computer/v1 operations, the pointer and keyboard groups among them, on an X11 desktop."""

    buttons = ("left", "right", "middle", "back", "forward")
    max_click_count = MAX_CLICK_COUNT
    max_drag_points = MAX_DRAG_POINTS
    recording = Recording(max_fps=30)

    def __init__(self, settings: ComputerSettings | None = None, platform: Platform | None = None) -> None:
        super().__init__(settings, platform)
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
        return await asyncio.to_thread(mss_capture, mss.MSS)

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
            c.released(button)
            c.release(reversed(held))
        await self._send(act)

    @drag
    async def drag_along(self, path, button, duration_ms, modifiers) -> None:
        syms = [_keysym(k) for k in modifiers]
        steps = drag_steps(path)
        pause = duration_ms / 1000 / len(steps)
        held: list[int] = []

        def start(c):
            held.extend(c.hold(syms))
            c.move(*path[0])
            c.button(button, True)

        def finish(c):
            c.button(button, False)
            c.released(button)
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
        def act(c):
            c.button(button, False)
            c.released(button)
        await self._send(act)

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
                held = self._held.pop(key, None)
                c.release(reversed(c.codes(_keysym(key))[-1:] if held is None else held))
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

    def released(self, button: str) -> None:
        """Wait, up to RELEASE_WAIT_S, until the server has let go of ``button``. A window manager's passive grab holds
        a click on a window until it replays it to the app, while keys reach the app at once, so a key sent before then
        overtakes the click, and so do the next click's moves."""
        deadline = time.monotonic() + RELEASE_WAIT_S
        while self.d.screen().root.query_pointer().mask & BUTTON_MASKS.get(button, 0) and time.monotonic() < deadline:
            time.sleep(RELEASE_POLL_S)

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


def _keysym(key: str) -> int:
    if key in KEYSYMS:
        return XK.string_to_keysym(KEYSYMS[key])
    if len(key) == 1:
        code = ord(key)
        return code if code < 0x100 else 0x01000000 | code
    raise Unsupported(f"this driver has no X keysym for {key!r}")
