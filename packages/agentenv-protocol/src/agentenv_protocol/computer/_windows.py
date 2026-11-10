"""computer/v1 on a Windows desktop, for a driver whose software serves tools of its own, as agent-desktop's does: mss
captures the screen and SendInput sends the input, in the session the driver runs in. The process is aware of each
monitor's DPI, so a point is a pixel of the capture, which spans every monitor. A recording polls those captures, as
the base class does for a driver with no stream of its own. A driver subclasses ``WindowsDriver``, declares itself with
``computer_driver``, and serves its own tools beside these. Needs the ``windows`` extra: mss."""
from __future__ import annotations

import asyncio
import contextlib
import ctypes
import functools
import time

import mss

from ._desktop import CLICK_GAP_S, MAX_CLICK_COUNT, MAX_DRAG_POINTS, drag_steps, mss_capture
from .driver import ComputerDriverEnvironment
from .errors import DeliveryFailed, DeviceUnavailable, Unsupported
from .operations import (Display, Image, ReadyCheck, capture, click, display, drag, key_down, key_up, pointer_down,
                         pointer_move, pointer_position, pointer_up, press, scroll, type_text)
from .platform import Platform
from .settings import ComputerSettings

INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
MOVE, ABSOLUTE, VIRTUAL_DESK = 0x0001, 0x8000, 0x4000
WHEEL, HWHEEL, WHEEL_DELTA = 0x0800, 0x1000, 120
#: Each button's press and release flags, and the X button it is, for the two that are one.
BUTTONS = {"left": (0x0002, 0x0004, 0), "right": (0x0008, 0x0010, 0), "middle": (0x0020, 0x0040, 0),
           "back": (0x0080, 0x0100, 1), "forward": (0x0080, 0x0100, 2)}
EXTENDED, KEY_UP, UNICODE = 0x0001, 0x0002, 0x0004
#: W3C key values and their virtual keys; any other single character is the key its keyboard layout gives it.
VIRTUAL_KEYS = {
    "Enter": 0x0D, "Tab": 0x09, "Escape": 0x1B, "Backspace": 0x08, "Delete": 0x2E, "ArrowUp": 0x26, "ArrowDown": 0x28,
    "ArrowLeft": 0x25, "ArrowRight": 0x27, "Home": 0x24, "End": 0x23, "PageUp": 0x21, "PageDown": 0x22, "Insert": 0x2D,
    "Control": 0xA2, "Shift": 0xA0, "Alt": 0xA4, "Meta": 0x5B, "Super": 0x5B, "CapsLock": 0x14, "ContextMenu": 0x5D,
    "PrintScreen": 0x2C, "Pause": 0x13, " ": 0x20, **{f"F{n}": 0x6F + n for n in range(1, 25)},
}
#: The keys a keyboard sends with the extended flag, without which Windows reads the arrows as the keypad's.
EXTENDED_KEYS = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2C, 0x2D, 0x2E, 0x5B, 0x5D}
#: The modifiers VkKeyScanW says a character needs, as the bits of its high byte.
SHIFT_STATE = ((1, 0xA0), (2, 0xA2), (4, 0xA4))
TYPED = {"\n": 0x0D, "\t": 0x09}
#: SetProcessDpiAwarenessContext's per-monitor (v2) awareness.
PER_MONITOR_AWARE_V2 = -4
#: GetSystemMetrics' indexes of the virtual screen's left, top, width and height.
VIRTUAL_SCREEN = (76, 77, 78, 79)
DESKTOP_SWITCHDESKTOP = 0x0100


class _MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long), ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_size_t)]


class _KeybdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", ctypes.c_size_t)]


class _HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_ushort), ("wParamH", ctypes.c_ushort)]


class _Event(ctypes.Union):
    _fields_ = [("mi", _MouseInput), ("ki", _KeybdInput), ("hi", _HardwareInput)]


class _Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("u", _Event)]


class _Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


@functools.cache
def _user32():
    """Windows' user32, typed, with the per-monitor DPI awareness set once so that points and the capture's pixels
    agree."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.SendInput.argtypes = (ctypes.c_uint, ctypes.POINTER(_Input), ctypes.c_int)
    user32.SendInput.restype = ctypes.c_uint
    user32.VkKeyScanW.argtypes = (ctypes.c_wchar,)
    user32.VkKeyScanW.restype = ctypes.c_short
    user32.MapVirtualKeyW.argtypes = (ctypes.c_uint, ctypes.c_uint)
    user32.MapVirtualKeyW.restype = ctypes.c_uint
    user32.OpenInputDesktop.argtypes = (ctypes.c_ulong, ctypes.c_bool, ctypes.c_ulong)
    user32.OpenInputDesktop.restype = ctypes.c_void_p
    user32.CloseDesktop.argtypes = (ctypes.c_void_p,)
    user32.GetCursorPos.argtypes = (ctypes.POINTER(_Point),)
    with contextlib.suppress(AttributeError):
        user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(PER_MONITOR_AWARE_V2))
    return user32


class WindowsDriver(ComputerDriverEnvironment):
    """The computer/v1 operations, the pointer and keyboard groups among them, on a Windows desktop."""

    buttons = ("left", "right", "middle", "back", "forward")
    max_click_count = MAX_CLICK_COUNT
    max_drag_points = MAX_DRAG_POINTS

    def __init__(self, settings: ComputerSettings | None = None, platform: Platform | None = None) -> None:
        super().__init__(settings, platform)
        #: The virtual keys each key held by @key_down pressed, which @key_up releases.
        self._held: dict[str, list[int]] = {}

    @display
    async def screen(self) -> Display:
        _, _, width, height = _virtual_screen()
        return Display(width, height, 1.0)

    @capture
    async def grab(self) -> Image:
        return await asyncio.to_thread(mss_capture, mss.MSS)

    async def ready_checks(self) -> list[ReadyCheck]:
        user32 = _user32()
        desktop = user32.OpenInputDesktop(0, False, DESKTOP_SWITCHDESKTOP)
        if not desktop:
            return [ReadyCheck("input_desktop", False, f"the input desktop isn't this session's: {_last_error()}")]
        user32.CloseDesktop(desktop)
        return [ReadyCheck("input_desktop", True, "the input desktop is this session's")]

    @click
    async def click_at(self, x, y, button, count, modifiers) -> None:
        held = self._pressing(modifiers)

        def act():
            _send([_key(vk, True) for vk in held] + [_move(x, y)])
            for i in range(count):
                _send([_button(button, True), _button(button, False)])
                if i + 1 < count:
                    time.sleep(CLICK_GAP_S)
            _send([_key(vk, False) for vk in reversed(held)])
        await asyncio.to_thread(act)

    @drag
    async def drag_along(self, path, button, duration_ms, modifiers) -> None:
        held = self._pressing(modifiers)
        steps = drag_steps(path)
        pause = duration_ms / 1000 / len(steps)
        finish = [_button(button, False)] + [_key(vk, False) for vk in reversed(held)]
        try:
            await asyncio.to_thread(_send, [_key(vk, True) for vk in held] + [_move(*path[0]), _button(button, True)])
            for px, py in steps:
                await asyncio.to_thread(_send, [_move(px, py)])
                await asyncio.sleep(pause)
        except BaseException:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(_send, finish)
            raise
        await asyncio.to_thread(_send, finish)

    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers) -> None:
        held = self._pressing(modifiers)
        await asyncio.to_thread(_send, [_key(vk, True) for vk in held] + [_move(x, y)] + [_wheel(direction)] * amount
                                + [_key(vk, False) for vk in reversed(held)])

    @pointer_move
    async def move(self, x, y) -> None:
        await asyncio.to_thread(_send, [_move(x, y)])

    @pointer_down
    async def down(self, button) -> None:
        await asyncio.to_thread(_send, [_button(button, True)])

    @pointer_up
    async def up(self, button) -> None:
        await asyncio.to_thread(_send, [_button(button, False)])

    @pointer_position
    async def where(self) -> tuple[int, int]:
        point = _Point()
        if not _user32().GetCursorPos(ctypes.byref(point)):
            raise DeviceUnavailable(f"the pointer can't be read: {_last_error()}")
        left, top, _, _ = _virtual_screen()
        return point.x - left, point.y - top

    @type_text
    async def type_in(self, text) -> None:
        """Each character as its key on the keyboard layout, with the modifiers it needs that no held key holds, so an
        app sees the keys; a character off the layout as itself."""
        hold = self.settings.tap_hold_ms / 1000
        for ch in text:
            vks = [TYPED[ch]] if ch in TYPED else _layout_keys(ch)
            if vks is None:
                downs, ups = _unicode(ch)
            else:
                pressed = [vk for vk in vks if vk not in self._down()]
                downs, ups = [_key(vk, True) for vk in pressed], [_key(vk, False) for vk in reversed(pressed)]
            await asyncio.to_thread(_tap, downs, ups, hold)
            await asyncio.sleep(self.settings.type_delay_ms / 1000)

    @press
    async def press_keys(self, keys) -> None:
        pressed = self._pressing(keys)
        await asyncio.to_thread(_tap, [_key(vk, True) for vk in pressed], [_key(vk, False) for vk in reversed(pressed)],
                                self.settings.tap_hold_ms / 1000)

    @key_down
    async def keys_down(self, keys) -> None:
        """Press what the keys need that isn't down already; for each key, the virtual keys this pressed, so that
        @key_up releases only those. Every key is looked up before the first press."""
        down = self._down()
        planned = {}
        for key in dict.fromkeys(keys):
            planned[key] = [vk for vk in _virtual_keys(key) if vk not in down]
            down.update(planned[key])
        await asyncio.to_thread(_send, [_key(vk, True) for vks in planned.values() for vk in vks])
        for key, vks in planned.items():
            self._held.setdefault(key, []).extend(vks)

    @key_up
    async def keys_up(self, keys) -> None:
        """Release what each key's @key_down pressed, nothing for one whose keys were down already, and a key none
        holds as itself; the keys stay held until Windows takes the releases, so a refused call can be made again."""
        released = [vk for key in dict.fromkeys(keys)
                    for vk in reversed(self._held[key] if key in self._held else _virtual_keys(key)[-1:])]
        await asyncio.to_thread(_send, [_key(vk, False) for vk in released])
        for key in keys:
            self._held.pop(key, None)

    def _down(self) -> set[int]:
        return {vk for vks in self._held.values() for vk in vks}

    def _pressing(self, keys) -> list[int]:
        """The virtual keys that hold ``keys`` down together, in order, each once, but those held keys hold already."""
        down = self._down()
        return [vk for vk in dict.fromkeys(vk for key in keys for vk in _virtual_keys(key)) if vk not in down]


def _send(events: list[_Input]) -> None:
    """``events`` through SendInput, refused when Windows takes fewer than all, as it does when the input desktop isn't
    this session's or a window of higher integrity is in front."""
    if not events:
        return
    sent = _user32().SendInput(len(events), (_Input * len(events))(*events), ctypes.sizeof(_Input))
    if sent != len(events):
        raise DeliveryFailed(f"Windows took {sent} of {len(events)} input events: {_last_error()}")


def _tap(downs: list[_Input], ups: list[_Input], hold_s: float) -> None:
    _send(downs)
    time.sleep(hold_s)
    _send(ups)


def _virtual_screen() -> tuple[int, int, int, int]:
    user32 = _user32()
    return tuple(user32.GetSystemMetrics(i) for i in VIRTUAL_SCREEN)


def _move(x: float, y: float) -> _Input:
    """A move to the pixel (x, y) of the capture, aimed at its centre in the virtual screen's 0 to 65535 coordinates,
    so that Windows' rounding lands on it."""
    _, _, width, height = _virtual_screen()
    dx = min(65535, round((int(x) + 0.5) * 65536 / width))
    dy = min(65535, round((int(y) + 0.5) * 65536 / height))
    return _Input(INPUT_MOUSE, _Event(mi=_MouseInput(dx, dy, 0, MOVE | ABSOLUTE | VIRTUAL_DESK, 0, 0)))


def _button(button: str, down: bool) -> _Input:
    press_flag, release_flag, which = BUTTONS[button]
    return _Input(INPUT_MOUSE, _Event(mi=_MouseInput(0, 0, which, press_flag if down else release_flag, 0, 0)))


def _wheel(direction: str) -> _Input:
    """One notch: up and right turn the wheel forward, down and left back."""
    flag = HWHEEL if direction in ("left", "right") else WHEEL
    notch = WHEEL_DELTA if direction in ("up", "right") else -WHEEL_DELTA
    return _Input(INPUT_MOUSE, _Event(mi=_MouseInput(0, 0, ctypes.c_ulong(notch).value, flag, 0, 0)))


def _key(vk: int, down: bool) -> _Input:
    """A virtual key with its scan code, which apps that read the physical key, as Chromium's ``code`` does, need."""
    flags = (EXTENDED if vk in EXTENDED_KEYS else 0) | (0 if down else KEY_UP)
    return _Input(INPUT_KEYBOARD, _Event(ki=_KeybdInput(vk, _user32().MapVirtualKeyW(vk, 0), flags, 0, 0)))


def _unicode(ch: str) -> tuple[list[_Input], list[_Input]]:
    """A character typed as itself, whatever the keyboard layout: a key for each of its UTF-16 units, two for one
    beyond the BMP."""
    data = ch.encode("utf-16-le")
    units = [int.from_bytes(data[i:i + 2], "little") for i in range(0, len(data), 2)]
    return [_unit(u, 0) for u in units], [_unit(u, KEY_UP) for u in units]


def _unit(unit: int, flags: int) -> _Input:
    return _Input(INPUT_KEYBOARD, _Event(ki=_KeybdInput(0, unit, UNICODE | flags, 0, 0)))


def _virtual_keys(key: str) -> list[int]:
    """The virtual keys that make ``key``: a named key's, or a character's on the keyboard layout with the modifiers
    it needs first."""
    if key in VIRTUAL_KEYS:
        return [VIRTUAL_KEYS[key]]
    if len(key) == 1:
        if (vks := _layout_keys(key)) is None:
            raise Unsupported(f"{key!r} has no key on this keyboard layout; type it as text instead")
        return vks
    raise Unsupported(f"this driver has no Windows virtual key for {key!r}")


def _layout_keys(ch: str) -> list[int] | None:
    """The modifiers and key that type ``ch`` on the keyboard layout, None for a character it has no key for, as one
    beyond the BMP, which VkKeyScanW can't take."""
    scan = _user32().VkKeyScanW(ch) if ord(ch) <= 0xFFFF else -1
    return None if scan == -1 else [vk for bit, vk in SHIFT_STATE if scan >> 8 & bit] + [scan & 0xFF]


def _last_error() -> str:
    return ctypes.WinError(ctypes.get_last_error()).strerror


