"""An input's screenshot shows what the input did: it waits for the screen to settle, as a browser's smooth scroll or
the page a key opens renders a couple of hundred milliseconds after the input returns."""
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

from agentenv_protocol.computer import ComputerSettings, capture, scroll  # noqa: E402
from agentenv_protocol.computer.errors import Internal  # noqa: E402
from agentenv_protocol.computer.runtime import SETTLE_QUIET_S  # noqa: E402

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)


class LateScreen(FakeDriver):
    """A fake whose scroll reaches the screen late: each (delay, colour) of ``steps`` fills the screen ``delay``
    seconds after the scroll returns."""

    def __init__(self, steps=(), settings: ComputerSettings | None = None) -> None:
        super().__init__(settings=settings)
        self.steps = steps
        self.captures = 0
        self.since_scroll: int | None = None

    def fill(self, colour: tuple) -> None:
        self.screen.image = PILImage.new("RGB", self.screen.image.size, colour)

    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers):
        await super().scroll_at(x, y, direction, amount, modifiers)
        self.since_scroll = 0
        loop = asyncio.get_running_loop()
        for delay, colour in self.steps:
            loop.call_later(delay, self.fill, colour)

    @capture
    async def grab(self):
        self.captures += 1
        if self.since_scroll is not None:
            self.since_scroll += 1
        return await super().grab()


def _colour(result) -> tuple:
    [image] = [c for c in result.content if c.type == "image"]
    return PILImage.open(io.BytesIO(base64.b64decode(image.data))).convert("RGB").getpixel((10, 10))


async def _scroll(driver: LateScreen) -> tuple:
    """The scroll's screenshot, and how long the call took."""
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        started = time.monotonic()
        scrolled = await session.call_tool("computer_scroll", {"x": 5, "y": 5, "direction": "down", "screenshot": True})
    return scrolled, time.monotonic() - started


@pytest.mark.asyncio
async def test_the_screenshot_shows_a_change_that_reaches_the_screen_after_the_input():
    scrolled, _ = await _scroll(LateScreen(steps=[(0.2, RED)]))
    assert _colour(scrolled) == RED
    assert scrolled.structuredContent["frame"]["frame_id"] == "f2"


@pytest.mark.asyncio
async def test_with_no_settle_the_screenshot_is_taken_at_once():
    scrolled, _ = await _scroll(LateScreen(steps=[(0.2, RED)], settings=ComputerSettings(screenshot_settle_ms=0)))
    assert _colour(scrolled) == BACKGROUND


@pytest.mark.asyncio
async def test_the_screenshot_waits_out_an_animation():
    scrolled, _ = await _scroll(LateScreen(steps=[(0.15, RED), (0.27, GREEN), (0.39, BLUE)]))
    assert _colour(scrolled) == BLUE


@pytest.mark.asyncio
async def test_an_input_that_changes_nothing_is_answered_after_the_quiet_wait():
    driver = LateScreen()
    scrolled, took = await _scroll(driver)
    assert _colour(scrolled) == BACKGROUND
    assert SETTLE_QUIET_S <= took < SETTLE_QUIET_S + 0.5
    assert driver.captures < 12


@pytest.mark.asyncio
async def test_a_screen_that_never_settles_is_answered_when_the_wait_runs_out():
    driver = LateScreen(settings=ComputerSettings(screenshot_settle_ms=700))

    async def flicker():
        shade = 0
        while True:
            shade = (shade + 1) % 250
            driver.fill((shade, 0, 0))
            await asyncio.sleep(0.03)

    task = asyncio.ensure_future(flicker())
    try:
        scrolled, took = await _scroll(driver)
    finally:
        task.cancel()
    assert _colour(scrolled)[1:] == (0, 0)
    assert 0.7 <= took < 1.2


@pytest.mark.asyncio
async def test_the_input_goes_ahead_when_the_capture_before_it_fails():
    driver = LateScreen(steps=[(0.2, RED)])
    async with mcp_session(driver) as session:
        await session.call_tool("computer_screenshot", {})
        driver.fail["capture"] = Internal("the display went away for a moment")
        scrolled = await session.call_tool("computer_scroll", {"x": 5, "y": 5, "direction": "down",
                                                               "screenshot": True})
    assert not scrolled.isError
    assert driver.screen.log[-1][0] == "scroll"
    assert scrolled.structuredContent["frame"]["frame_id"] == "f2"


class FailsAfter(LateScreen):
    """A fake whose captures fail from its scroll on, while the screen settles."""

    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers):
        await super().scroll_at(x, y, direction, amount, modifiers)
        self.fail["capture"] = RuntimeError("the screen grabber died")


@pytest.mark.asyncio
async def test_a_capture_that_fails_while_the_screen_settles_reports_the_input_as_done():
    driver = FailsAfter(steps=[(0.2, RED)])
    scrolled, _ = await _scroll(driver)
    assert not scrolled.isError and scrolled.structuredContent["applied"] == "yes"
    assert "frame" not in scrolled.structuredContent
    assert "the screenshot after it failed (internal: @capture failed: RuntimeError: the screen grabber died)" in (
        scrolled.content[0].text)


@pytest.mark.asyncio
async def test_a_slow_capture_doesnt_count_as_time_the_screen_held_still():
    driver = LateScreen(steps=[(0.05, RED), (0.42, BLUE)])
    driver.slow["capture"] = 0.15
    scrolled, _ = await _scroll(driver)
    assert _colour(scrolled) == BLUE


class SlowLater(LateScreen):
    """A fake whose captures take a second from the second after its scroll on."""

    @capture
    async def grab(self):
        if self.since_scroll:
            await asyncio.sleep(1.0)
        return await super().grab()


@pytest.mark.asyncio
async def test_a_slow_capture_doesnt_outlast_the_settle_limit():
    scrolled, took = await _scroll(SlowLater(settings=ComputerSettings(screenshot_settle_ms=400)))
    assert not scrolled.isError and scrolled.structuredContent["frame"]["frame_id"] == "f2"
    assert took < 0.8
