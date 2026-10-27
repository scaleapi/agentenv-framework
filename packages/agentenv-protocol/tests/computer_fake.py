"""An in-memory screen and a driver on it that implements every operation, so the tests exercise the whole base
class with no display. A click paints its button's colour at its point; everything that reaches the screen is
logged; ``fail`` makes the next call of an operation raise."""
from __future__ import annotations

import asyncio
import contextlib

import httpx
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image as PILImage

from agentenv_protocol.computer import (
    ComputerDriverEnvironment,
    ComputerSettings,
    Display,
    Effect,
    Platform,
    capture,
    click,
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
    set_display,
    type_text,
)

LINUX = Platform(os="linux", distribution="ubuntu", version="24.04", arch="amd64", display="x11")
BACKGROUND = (255, 255, 255)
COLOURS = {"left": (255, 0, 0), "right": (0, 0, 255), "middle": (0, 255, 0)}


class FakeScreen:
    def __init__(self, width: int = 1280, height: int = 800, scale: float = 1.0) -> None:
        self.log: list = []
        self.pointer = (0, 0)
        self.held: list = []
        self.scrolled = [0, 0]
        self.resize(width, height, scale)

    def resize(self, width: int, height: int, scale: float) -> None:
        self.display = Display(width, height, scale)
        self.image = PILImage.new("RGB", (round(width * scale), round(height * scale)), BACKGROUND)

    def paint(self, x: int, y: int, colour: tuple) -> None:
        """Colour the image pixels under screen pixel (x, y)."""
        s = self.display.scale
        for px in range(int(x * s), max(int((x + 1) * s), int(x * s) + 1)):
            for py in range(int(y * s), max(int((y + 1) * s), int(y * s) + 1)):
                self.image.putpixel((px, py), colour)

    def colour_at(self, x: int, y: int) -> tuple:
        s = self.display.scale
        return self.image.getpixel((int(x * s), int(y * s)))


class FakeDriver(ComputerDriverEnvironment):
    buttons = ("left", "right", "middle")
    max_click_count = 3
    max_drag_points = 10
    display_modes = ((1280, 800), (1920, 1080), (640, 400))

    def __init__(self, screen: FakeScreen | None = None, settings: ComputerSettings | None = None,
                 platform: Platform = LINUX) -> None:
        super().__init__(settings=settings or ComputerSettings(), platform=platform)
        self.screen = screen or FakeScreen()
        self.fail: dict = {}
        self.slow: dict = {}

    async def _step(self, op: str) -> None:
        if op in self.slow:
            await asyncio.sleep(self.slow[op])
        error = self.fail.pop(op, None)
        if error is not None:
            raise error

    @display
    async def screen_size(self) -> Display:
        await self._step("display")
        return self.screen.display

    @set_display
    async def resize(self, width: int, height: int, scale: float) -> Display:
        await self._step("set_display")
        self.screen.resize(width, height, scale)
        return self.screen.display

    @capture
    async def grab(self):
        await self._step("capture")
        return self.screen.image.copy()

    @click
    async def click_at(self, x, y, button, count, modifiers):
        await self._step("click")
        self.screen.log.append(("click", x, y, button, count, tuple(modifiers)))
        self.screen.pointer = (x, y)
        self.screen.paint(x, y, COLOURS[button])
        return Effect(pointer_moved=True)

    @drag
    async def drag_along(self, path, button, duration_ms, modifiers):
        await self._step("drag")
        self.screen.log.append(("drag", tuple(map(tuple, path)), button, duration_ms, tuple(modifiers)))
        self.screen.pointer = tuple(path[-1])

    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers):
        await self._step("scroll")
        self.screen.log.append(("scroll", x, y, direction, amount, tuple(modifiers)))
        dx, dy = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}[direction]
        self.screen.scrolled = [self.screen.scrolled[0] + dx * amount, self.screen.scrolled[1] + dy * amount]

    @type_text
    async def type_in(self, text):
        await self._step("type_text")
        self.screen.log.append(("type", text))

    @press
    async def press_keys(self, keys):
        await self._step("press")
        self.screen.log.append(("press", tuple(keys), self.settings.tap_hold_ms))

    @pointer_move
    async def move(self, x, y):
        await self._step("pointer_move")
        self.screen.log.append(("move", x, y))
        self.screen.pointer = (x, y)

    @pointer_down
    async def down(self, button):
        await self._step("pointer_down")
        self.screen.log.append(("down", button))

    @pointer_up
    async def up(self, button):
        await self._step("pointer_up")
        self.screen.log.append(("up", button))

    @pointer_position
    async def where(self):
        await self._step("pointer_position")
        return self.screen.pointer

    @key_down
    async def keys_down(self, keys):
        await self._step("key_down")
        self.screen.log.append(("key_down", tuple(keys)))
        self.screen.held += [k for k in keys if k not in self.screen.held]

    @key_up
    async def keys_up(self, keys):
        await self._step("key_up")
        self.screen.log.append(("key_up", tuple(keys)))
        self.screen.held = [k for k in self.screen.held if k not in keys]


class MinimalDriver(ComputerDriverEnvironment):
    """Only the required operations, synchronous ones: no pointer or keyboard group, no display modes."""

    def __init__(self, screen: FakeScreen | None = None, platform: Platform = LINUX) -> None:
        super().__init__(settings=ComputerSettings(), platform=platform)
        self.screen = screen or FakeScreen(640, 400)

    @display
    def screen_size(self) -> Display:
        return self.screen.display

    @capture
    def grab(self):
        return self.screen.image.copy()

    @click
    def click_at(self, x, y, button, count, modifiers):
        self.screen.log.append(("click", x, y, button, count, tuple(modifiers)))
        self.screen.paint(x, y, COLOURS[button])

    @drag
    def drag_along(self, path, button, duration_ms, modifiers):
        self.screen.log.append(("drag", tuple(map(tuple, path))))

    @scroll
    def scroll_at(self, x, y, direction, amount, modifiers):
        self.screen.log.append(("scroll", x, y, direction, amount))

    @type_text
    def type_in(self, text):
        self.screen.log.append(("type", text))

    @press
    def press_keys(self, keys):
        self.screen.log.append(("press", tuple(keys)))


@contextlib.asynccontextmanager
async def mcp_session(driver: ComputerDriverEnvironment):
    """An MCP client session on the driver's app, in memory."""
    async with create_connected_server_and_client_session(driver.mcp or driver.create_app()) as session:
        yield session


@contextlib.asynccontextmanager
async def http_client(driver: ComputerDriverEnvironment):
    """An HTTP client on the driver's app, for its card, extensions and data plane."""
    app = (driver.mcp or driver.create_app()).streamable_http_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        yield client


def rpc(method: str, params: dict | None = None) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
