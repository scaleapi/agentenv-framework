"""An input's screenshot is taken ``screenshot_delay_ms`` after the input, so it shows what the input did: a browser's
smooth scroll or the page a key opens renders a couple of hundred milliseconds after the input returns."""
from __future__ import annotations

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

import asyncio  # noqa: E402
import base64  # noqa: E402
import io  # noqa: E402
import time  # noqa: E402

from computer_fake import BACKGROUND, FakeDriver, mcp_session  # noqa: E402
from PIL import Image as PILImage  # noqa: E402

from agentenv_protocol.computer import ComputerSettings, scroll  # noqa: E402

RED = (255, 0, 0)


class LateScreen(FakeDriver):
    """A fake whose scroll reaches the screen 0.2 s after the scroll returns, filling it red."""

    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers):
        await super().scroll_at(x, y, direction, amount, modifiers)
        asyncio.get_running_loop().call_later(0.2, self.fill)

    def fill(self) -> None:
        self.screen.image = PILImage.new("RGB", self.screen.image.size, RED)


def _colour(result) -> tuple:
    [image] = [c for c in result.content if c.type == "image"]
    return PILImage.open(io.BytesIO(base64.b64decode(image.data))).convert("RGB").getpixel((10, 10))


async def _scroll(session, screenshot: bool = True):
    return await session.call_tool("computer_scroll", {"x": 5, "y": 5, "direction": "down", "screenshot": screenshot})


@pytest.mark.asyncio
async def test_the_screenshot_shows_a_change_that_reaches_the_screen_after_the_input():
    driver = LateScreen()
    assert driver.settings.screenshot_delay_ms == 500
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        started = time.monotonic()
        scrolled = await _scroll(session)
        took = time.monotonic() - started
    assert _colour(scrolled) == RED
    assert took >= 0.5
    assert scrolled.structuredContent["frame"]["frame_id"] == "f2"


@pytest.mark.asyncio
async def test_with_no_delay_the_screenshot_is_taken_at_once():
    async with mcp_session(LateScreen(settings=ComputerSettings(screenshot_delay_ms=0))) as session:
        await session.call_tool("computer_screenshot", {})
        scrolled = await _scroll(session)
    assert _colour(scrolled) == BACKGROUND


@pytest.mark.asyncio
async def test_an_input_without_a_screenshot_doesnt_wait():
    async with mcp_session(LateScreen(settings=ComputerSettings(screenshot_delay_ms=5000))) as session:
        await session.call_tool("computer_screenshot", {})
        started = time.monotonic()
        scrolled = await _scroll(session, screenshot=False)
        took = time.monotonic() - started
    assert not scrolled.isError and "frame" not in scrolled.structuredContent
    assert took < 2.5


@pytest.mark.asyncio
async def test_the_wait_before_the_screenshot_doesnt_hold_other_input():
    driver = LateScreen(settings=ComputerSettings(screenshot_delay_ms=3000))
    finished = []

    async def call(name: str, tool: str, args: dict) -> None:
        await session.call_tool(tool, args)
        finished.append(name)

    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        scrolling = asyncio.ensure_future(call("scroll", "computer_scroll",
                                               {"x": 5, "y": 5, "direction": "down", "screenshot": True}))
        await asyncio.sleep(0.5)
        await call("click", "computer_click", {"x": 7, "y": 7})
        await scrolling
    assert finished == ["click", "scroll"]
    assert [entry[0] for entry in driver.screen.log] == ["scroll", "click"]
