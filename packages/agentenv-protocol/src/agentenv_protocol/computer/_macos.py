"""computer/v1 on macOS for a driver whose software serves tools of its own, as agent-desktop's and macos-harness's do:
ScreenCaptureKit captures the display and Quartz events, posted where the hardware's are, send the pointer and the
keyboard, from the driver's process in the login session. That process needs Accessibility, Post Event and Screen
Recording, which its layer grants the Python it runs on (``[install.macos]``'s ``grants``); ``computer-ready`` names
any it lacks. A driver subclasses ``MacOSDriver``, declares itself with ``computer_driver``, and serves its own tools
beside these. Needs the ``macos`` extra: PyObjC's Quartz and ScreenCaptureKit."""
from __future__ import annotations

import asyncio
import ctypes
import math
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from .driver import ComputerDriverEnvironment
from .errors import DeliveryFailed, DeviceUnavailable, Unsupported
from .keys import PRINTABLE_KEYS
from .operations import (Display, Image, ReadyCheck, capture, click, display, drag, key_down, key_up, pointer_down,
                         pointer_move, pointer_position, pointer_up, press, scroll, type_text)
from .platform import Platform
from .settings import ComputerSettings

#: Each W3C modifier key value: its virtual key code (kVK_*) and the Quartz event flag it holds; Meta and Super are
#: Command.
MODIFIERS = {"Shift": (56, 0x20000), "Control": (59, 0x40000), "Alt": (58, 0x80000), "Meta": (55, 0x100000),
             "Super": (55, 0x100000)}
SHIFT = "Shift"
#: Named W3C key values and their virtual key codes.
NAMED = {
    "Enter": 36, "Tab": 48, " ": 49, "Backspace": 51, "Escape": 53, "Delete": 117, "Help": 114, "Home": 115,
    "End": 119, "PageUp": 116, "PageDown": 121, "ArrowLeft": 123, "ArrowRight": 124, "ArrowDown": 125,
    "ArrowUp": 126, "CapsLock": 57,
    "F1": 122, "F2": 120, "F3": 99, "F4": 118, "F5": 96, "F6": 97, "F7": 98, "F8": 100, "F9": 101, "F10": 109,
    "F11": 103, "F12": 111, "F13": 105, "F14": 107, "F15": 113, "F16": 106, "F17": 64, "F18": 79, "F19": 80,
    "F20": 90,
}
#: The virtual key code of each key on a US keyboard, by the character it types unshifted.
US = {
    **dict(zip("asdfhgzxcvb", (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11))),
    **dict(zip("qwertyuiop", (12, 13, 14, 15, 17, 16, 32, 34, 31, 35))),
    **dict(zip("1234567890", (18, 19, 20, 21, 23, 22, 26, 28, 25, 29))),
    **dict(zip("lkjnm", (37, 40, 38, 45, 46))),
    "=": 24, "-": 27, "]": 30, "[": 33, "'": 39, ";": 41, "\\": 42, ",": 43, "/": 44, ".": 47, "`": 50,
}
#: Each character a US keyboard types with Shift, and the key's unshifted character.
SHIFTED = {**{c.upper(): c for c in "abcdefghijklmnopqrstuvwxyz"},
           **dict(zip('!@#$%^&*()_+{}|:"<>?~', "1234567890-=[]\\;',./`"))}
#: How typing sends a space, a tab or a newline.
TYPED = {" ": " ", "\t": "Tab", "\n": "Enter"}
#: The key code a character off the keyboard is typed on, carrying the character itself.
UNMAPPED = 0
#: Quartz's event types for each button: down, up and dragged; and the pointer moved with no button down.
BUTTONS = {"left": (0, 1, 2, 6), "right": (1, 3, 4, 7), "middle": (2, 25, 26, 27)}
MOVED = 5
#: How long the pointer rests where a click or a drag presses before the press: an app such as Tk reads where the
#: pointer is when it handles the press, and a button the pointer hasn't yet entered loses the click.
SETTLE_S = 0.2
#: How long a click holds its button down, as a hand does: a press released at once is lost now and then.
CLICK_HOLD_S = 0.03
#: The pause between the presses of a double or triple click.
CLICK_GAP_S = 0.04
#: How far apart a drag's pointer moves are, in pixels.
DRAG_STEP_PX = 20
#: The pause between a scroll's wheel notches, as a wheel turned by hand has: notches posted back to back reach an app
#: as fewer.
SCROLL_GAP_S = 0.05
#: How long a posted move may take to become where macOS says the pointer is, and how often that's looked at: a press
#: at the pointer right after a move otherwise lands where it was.
MOVE_SETTLE_S, MOVE_POLL_S = 0.2, 0.01
#: How long ScreenCaptureKit may take to answer.
CAPTURE_TIMEOUT_S = 10.0
#: The TCC permissions the driver's process needs, and how each is named to a person.
PERMISSIONS = {"accessibility": "Accessibility", "post_event": "Post Event", "screen_capture": "Screen Recording"}


class MacOSDriver(ComputerDriverEnvironment):
    """The computer/v1 operations, the pointer and keyboard groups among them, on macOS. Input goes out one call at a
    time; keys and buttons held by ``@key_down`` and ``@pointer_down`` stay held, as their flags on every event, until
    released."""

    buttons = ("left", "right", "middle")
    max_click_count = 3
    max_drag_points = 50
    keys = (*MODIFIERS, *NAMED, *PRINTABLE_KEYS)

    def __init__(self, settings: ComputerSettings | None = None, platform: Platform | None = None) -> None:
        super().__init__(settings, platform)
        self._mac: Quartz | None = None
        self._mac_input = threading.Lock()
        #: The keys ``@key_down`` holds, each with its key code, and the buttons ``@pointer_down`` holds.
        self._mac_keys: dict[str, int] = {}
        self._mac_buttons: list[str] = []

    @display
    async def screen(self) -> Display:
        width, height, scale = await asyncio.to_thread(self._mac_backend().display)
        return Display(width, height, scale)

    @capture
    async def grab(self) -> Image:
        width, height, data = await asyncio.to_thread(self._mac_backend().capture)
        return Image(width, height, data, "BGRA")

    async def ready_checks(self) -> list[ReadyCheck]:
        missing = await asyncio.to_thread(self._mac_backend().missing)
        return [ReadyCheck("macOS permissions", not missing, "Accessibility, Post Event and Screen Recording granted"
                           if not missing else f"macOS hasn't granted this driver {', '.join(missing)}, which its "
                           "layer grants the Python it runs on")]

    @click
    async def click_at(self, x, y, button, count, modifiers) -> None:
        number, down, up, _ = BUTTONS[button]

        def act(mac):
            with self._mac_holding(mac, modifiers) as flags:
                mac.mouse(MOVED, x, y, 0, 1, flags)
                time.sleep(SETTLE_S)
                for n in range(1, count + 1):
                    mac.mouse(down, x, y, number, n, flags)
                    time.sleep(CLICK_HOLD_S)
                    mac.mouse(up, x, y, number, n, flags)
                    if n < count:
                        time.sleep(CLICK_GAP_S)
        await self._mac_send(act)

    @drag
    async def drag_along(self, path, button, duration_ms, modifiers) -> None:
        number, down, up, dragged = BUTTONS[button]
        steps = _steps(path)
        pause = duration_ms / 1000 / len(steps)

        def act(mac):
            with self._mac_holding(mac, modifiers) as flags:
                mac.mouse(MOVED, *path[0], 0, 1, flags)
                time.sleep(SETTLE_S)
                mac.mouse(down, *path[0], number, 1, flags)
                try:
                    for x, y in steps:
                        time.sleep(pause)
                        mac.mouse(dragged, x, y, number, 1, flags)
                finally:
                    mac.mouse(up, *path[-1], number, 1, flags)
        await self._mac_send(act)

    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers) -> None:
        lines, across = {"up": (1, 0), "down": (-1, 0), "left": (0, 1), "right": (0, -1)}[direction]

        def act(mac):
            with self._mac_holding(mac, modifiers) as flags:
                mac.mouse(MOVED, x, y, 0, 1, flags)
                for notch in range(amount):
                    if notch:
                        time.sleep(SCROLL_GAP_S)
                    mac.wheel(x, y, lines, across, flags)
        await self._mac_send(act)

    @type_text
    async def type_in(self, text) -> None:
        hold, delay = self.settings.tap_hold_ms / 1000, self.settings.type_delay_ms / 1000
        for character in text:
            await self._mac_send(lambda mac, character=character: self._mac_type(mac, character, hold))
            await asyncio.sleep(delay)

    @press
    async def press_keys(self, keys) -> None:
        *held, key = keys
        code = _code(key)
        hold = self.settings.tap_hold_ms / 1000
        if key in SHIFTED and SHIFT not in held:
            held.append(SHIFT)

        def act(mac):
            with self._mac_holding(mac, held) as flags:
                if key in MODIFIERS:
                    with self._mac_holding(mac, [key]):
                        time.sleep(hold)
                    return
                mac.key(code, True, flags)
                time.sleep(hold)
                mac.key(code, False, flags)
        await self._mac_send(act)

    @key_down
    async def keys_down(self, keys) -> None:
        codes = [_code(key) for key in keys]

        def act(mac):
            for key, code in zip(keys, codes):
                if key not in self._mac_keys:
                    self._mac_keys[key] = code
                    mac.key(code, True, self._mac_flags())
        await self._mac_send(act)

    @key_up
    async def keys_up(self, keys) -> None:
        def act(mac):
            for key in keys:
                code = self._mac_keys.pop(key, None)
                mac.key(_code(key) if code is None else code, False, self._mac_flags())
        await self._mac_send(act)

    @pointer_move
    async def move(self, x, y) -> None:
        def act(mac):
            if self._mac_buttons:
                number, _, _, dragged = BUTTONS[self._mac_buttons[0]]
                mac.mouse(dragged, x, y, number, 1, self._mac_flags())
            else:
                mac.mouse(MOVED, x, y, 0, 1, self._mac_flags())
            deadline = time.monotonic() + MOVE_SETTLE_S
            while max(abs(a - b) for a, b in zip(mac.location(), (x, y))) >= 1 and time.monotonic() < deadline:
                time.sleep(MOVE_POLL_S)
        await self._mac_send(act)

    @pointer_down
    async def down(self, button) -> None:
        def act(mac):
            number, down, _, _ = BUTTONS[button]
            mac.mouse(down, *mac.location(), number, 1, self._mac_flags())
            if button not in self._mac_buttons:
                self._mac_buttons.append(button)
        await self._mac_send(act)

    @pointer_up
    async def up(self, button) -> None:
        def act(mac):
            number, _, up, _ = BUTTONS[button]
            mac.mouse(up, *mac.location(), number, 1, self._mac_flags())
            if button in self._mac_buttons:
                self._mac_buttons.remove(button)
        await self._mac_send(act)

    @pointer_position
    async def where(self) -> tuple[int, int]:
        x, y = await asyncio.to_thread(self._mac_backend().location)
        return round(x), round(y)

    def _mac_type(self, mac: Any, character: str, hold: float) -> None:
        """One character: its key, with Shift around it when it needs Shift and no key holds Shift; or, off the
        keyboard, a key carrying the character itself."""
        if character in TYPED:
            code, shifted, carried = NAMED[TYPED[character]], False, None
        elif character in US or character in SHIFTED:
            code, shifted, carried = US[SHIFTED.get(character, character)], character in SHIFTED, None
        else:
            code, shifted, carried = UNMAPPED, False, character
        with self._mac_holding(mac, [SHIFT] if shifted and SHIFT not in self._mac_keys else []) as flags:
            mac.key(code, True, flags, carried)
            time.sleep(hold)
            mac.key(code, False, flags, carried)

    @contextmanager
    def _mac_holding(self, mac: Any, modifiers: list[str]) -> Iterator[int]:
        """The modifier keys pressed in order and released in reverse, as a keyboard sends them; yields the flags every
        event between carries, those of the keys ``@key_down`` holds too."""
        pressed: list[str] = []
        try:
            for modifier in modifiers:
                if modifier not in MODIFIERS:
                    raise Unsupported(f"{modifier!r} isn't a modifier key, so it can't be held with another")
                if modifier in self._mac_keys or modifier in pressed:
                    continue
                pressed.append(modifier)
                mac.key(MODIFIERS[modifier][0], True, self._mac_flags(pressed))
            yield self._mac_flags(pressed)
        finally:
            for modifier in reversed(pressed):
                pressed.remove(modifier)
                mac.key(MODIFIERS[modifier][0], False, self._mac_flags(pressed))

    def _mac_flags(self, pressed: list[str] = ()) -> int:
        flags = 0
        for key in (*self._mac_keys, *pressed):
            if key in MODIFIERS:
                flags |= MODIFIERS[key][1]
        return flags

    async def _mac_send(self, act: Callable[[Any], None]) -> None:
        """``act`` on the Mac, one input at a time. Quartz posts an event or raises, and a raise mid-input may have
        sent some of it."""
        def run():
            with self._mac_input:
                act(self._mac_backend())
        try:
            await asyncio.to_thread(run)
        except (Unsupported, DeviceUnavailable):
            raise
        except Exception as e:
            raise DeliveryFailed(f"Quartz failed while sending input: {type(e).__name__}: {e}") from None

    def _mac_backend(self) -> Quartz:
        if self._mac is None:
            self._mac = Quartz()
        return self._mac


def _code(key: str) -> int:
    """The virtual key code of a key value this driver presses: a modifier, a named key, or a character on a US
    keyboard, its unshifted key for one it types with Shift."""
    if key in MODIFIERS:
        return MODIFIERS[key][0]
    if key in NAMED:
        return NAMED[key]
    if key in US or key in SHIFTED:
        return US[SHIFTED.get(key, key)]
    raise Unsupported(f"no key on a US keyboard presses {key!r}")


def _steps(path: list) -> list[tuple[float, float]]:
    """The points a drag moves through: each of ``path`` after its first, with points about ``DRAG_STEP_PX`` apart
    between them."""
    steps = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        count = max(1, math.ceil(math.hypot(x1 - x0, y1 - y0) / DRAG_STEP_PX))
        steps += [(x0 + (x1 - x0) * i / count, y0 + (y1 - y0) * i / count) for i in range(1, count + 1)]
    return steps or [tuple(path[-1])]


class Quartz:
    """The Mac through PyObjC: ScreenCaptureKit for the display, Quartz events for input, and TCC's answers for this
    process, read through ctypes."""

    def __init__(self) -> None:
        try:
            import Quartz as quartz
            import ScreenCaptureKit
        except ImportError as e:
            raise DeviceUnavailable(f"PyObjC's Quartz and ScreenCaptureKit aren't installed: {e}") from None
        self.q, self.sck = quartz, ScreenCaptureKit
        self._shot: tuple | None = None
        services = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices")
        services.AXIsProcessTrusted.restype = ctypes.c_bool
        graphics = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
        graphics.CGPreflightPostEventAccess.restype = ctypes.c_bool
        graphics.CGPreflightScreenCaptureAccess.restype = ctypes.c_bool
        self._granted = {"accessibility": services.AXIsProcessTrusted,
                         "post_event": graphics.CGPreflightPostEventAccess,
                         "screen_capture": graphics.CGPreflightScreenCaptureAccess}

    def missing(self) -> list[str]:
        return [PERMISSIONS[name] for name, granted in self._granted.items() if not granted()]

    def display(self) -> tuple[int, int, float]:
        q = self.q
        main = q.CGMainDisplayID()
        bounds = q.CGDisplayBounds(main)
        width, height = int(bounds.size.width), int(bounds.size.height)
        pixels = q.CGDisplayModeGetPixelWidth(q.CGDisplayCopyDisplayMode(main))
        return width, height, pixels / width if width and pixels else 1.0

    def capture(self) -> tuple[int, int, bytes]:
        """The main display at its native resolution, as BGRA rows. Its filter is made once a display mode, and again
        when a capture fails, as one does after the mode changes."""
        try:
            image = self._screenshot()
        except DeviceUnavailable:
            self._shot = None
            image = self._screenshot()
        q = self.q
        width, height, row = q.CGImageGetWidth(image), q.CGImageGetHeight(image), q.CGImageGetBytesPerRow(image)
        data = bytes(q.CGDataProviderCopyData(q.CGImageGetDataProvider(image)))
        if row != width * 4:
            data = b"".join(data[y * row:y * row + width * 4] for y in range(height))
        return width, height, data

    def _screenshot(self) -> Any:
        return self._wait(lambda done: self.sck.SCScreenshotManager
                          .captureImageWithFilter_configuration_completionHandler_(*self._filter(), done))

    def _filter(self) -> tuple:
        q, sck = self.q, self.sck
        main = q.CGMainDisplayID()
        mode = q.CGDisplayCopyDisplayMode(main)
        size = (q.CGDisplayModeGetPixelWidth(mode), q.CGDisplayModeGetPixelHeight(mode))
        if self._shot is None or self._shot[0] != size:
            content = self._wait(sck.SCShareableContent.getShareableContentWithCompletionHandler_)
            screen = next((d for d in content.displays() if d.displayID() == main), None)
            if screen is None:
                raise DeviceUnavailable("ScreenCaptureKit lists no main display")
            config = sck.SCStreamConfiguration.alloc().init()
            config.setWidth_(size[0])
            config.setHeight_(size[1])
            config.setShowsCursor_(False)
            self._shot = (size, sck.SCContentFilter.alloc().initWithDisplay_excludingWindows_(screen, []), config)
        return self._shot[1:]

    @staticmethod
    def _wait(start: Callable[[Callable], None]) -> Any:
        """What ScreenCaptureKit hands the completion handler ``start`` passes it: its result, or its error raised."""
        done, answer = threading.Event(), {}

        def handler(result, error):
            answer.update(result=result, error=error)
            done.set()

        start(handler)
        if not done.wait(CAPTURE_TIMEOUT_S):
            raise DeviceUnavailable(f"ScreenCaptureKit didn't answer within {CAPTURE_TIMEOUT_S:.0f} s")
        if answer["error"] is not None:
            raise DeviceUnavailable(f"ScreenCaptureKit: {answer['error'].localizedDescription()}")
        return answer["result"]

    def mouse(self, kind: int, x: float, y: float, button: int, count: int, flags: int) -> None:
        q = self.q
        event = q.CGEventCreateMouseEvent(None, kind, (x, y), button)
        q.CGEventSetIntegerValueField(event, q.kCGMouseEventClickState, count)
        self._post(event, flags)

    def wheel(self, x: float, y: float, lines: int, across: int, flags: int) -> None:
        q = self.q
        event = q.CGEventCreateScrollWheelEvent(None, q.kCGScrollEventUnitLine, 2, lines, across)
        q.CGEventSetLocation(event, (x, y))
        self._post(event, flags)

    def key(self, code: int, down: bool, flags: int, character: str | None = None) -> None:
        event = self.q.CGEventCreateKeyboardEvent(None, code, down)
        if character is not None:
            self.q.CGEventKeyboardSetUnicodeString(event, len(character.encode("utf-16-le")) // 2, character)
        self._post(event, flags)

    def location(self) -> tuple[float, float]:
        point = self.q.CGEventGetLocation(self.q.CGEventCreate(None))
        return point.x, point.y

    def _post(self, event: Any, flags: int) -> None:
        if event is None:
            raise DeliveryFailed("Quartz made no event: this process may lack Post Event")
        self.q.CGEventSetFlags(event, flags)
        self.q.CGEventPost(self.q.kCGHIDEventTap, event)

