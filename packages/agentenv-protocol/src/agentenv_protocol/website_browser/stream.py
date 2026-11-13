"""The frames of one browser context as its recording's video takes them (browser-recording/v1): of the tab it
follows, the one an agent last acted on there, a tab that one opens as soon as it opens, and once it closes the tab
followed before it that's still open, else the context's newest; a blank frame while the context has no tab. The
browser's own connection screencasts that tab; a tab that sends no frame for ``SILENT_S`` and isn't visible, as one
behind another in its window, is captured instead, at most ``FALLBACK_FPS`` times a second, until it sends one again,
while a visible one is a still page its last frame shows. Every ``1/fps`` seconds the latest frame is handed on, so a
still page doesn't split the video."""
from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Optional

from PIL import Image

from ..computer.recording import RecordedFrame, RecordHints

if TYPE_CHECKING:
    from .control import BrowserControl

logger = logging.getLogger(__name__)

#: How long the followed tab may send no screencast frame before it's captured instead, and how often then.
SILENT_S = 1.0
FALLBACK_FPS = 2
#: How long a recording's start waits on its first frame's capture of the tab it follows.
FIRST_CAPTURE_S = 2.0
#: The size of the blank frame shown while the context has no tab and no frame of one yet.
BLANK_SIZE = (1280, 800)
SCREENCAST, SCREENSHOTS, BLANK = "screencast", "screenshots", "blank"


class TabStream:
    """The frames of the context ``context_id``, the browser's id for a lease, None for the default context, of the
    browser that ``control`` is. ``followed`` lists each tab it showed from when, and how."""

    def __init__(self, control: BrowserControl, context_id: Optional[str], blank_size: tuple[int, int] = BLANK_SIZE
                 ) -> None:
        self.control, self.context_id, self.blank_size = control, context_id, blank_size
        self.tab: Optional[str] = None
        self.followed: list[dict] = []
        self._history: list[str] = []
        self._jpeg: Optional[bytes] = None
        self._image: Optional[Image.Image] = None
        self._loop = asyncio.get_running_loop()
        self._heard_at = 0.0
        #: The screencast it runs: the tab, its session and what hears that tab's frames.
        self._casting: Optional[tuple[str, str, Callable[[bytes], None]]] = None
        self._capturing: Optional[asyncio.Task] = None
        self._seeding: Optional[asyncio.Future] = None
        self._max_edge: Optional[int] = None
        self._via: Optional[str] = None

    def sees(self, target: Optional[str]) -> bool:
        """Whether the tab ``target`` is in its context."""
        return bool(target) and self.control.in_context(self.control.context_of_tab(target), self.context_id)

    def follow(self, target: Optional[str]) -> None:
        """An agent acted on the tab ``target``: one in its context is followed from now."""
        if self.sees(target) and target != self.tab:
            self._move(target)

    def tab_event(self, event: str, tab: dict) -> None:
        """A tab opened or closed in the browser: one the followed tab opens, or the first its context has, is
        followed; once the followed tab closes, the one followed before it that's still open, else the newest."""
        target = tab.get("id")
        if not self.control.in_context(tab.get("context_id"), self.context_id):
            return
        if event == "opened" and (self.tab is None or tab.get("opener") == self.tab):
            self._move(target)
        elif event == "closed" and target == self.tab:
            self._move(next((t for t in reversed(self._history) if t in self.control.tabs and self.sees(t)), None) or
                       next((t for t in reversed(self.control.tabs) if self.sees(t)), None))

    async def start(self) -> None:
        """Following the context's active tab from now, else its newest, its first frame a capture of it."""
        if self.tab is None:
            active = await self.control.active_tab(self.context_id)
            self._move(active or next((t for t in reversed(self.control.tabs) if self.sees(t)), None))
        if self._seeding is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._seeding

    async def frames(self, hints: RecordHints) -> AsyncIterator[RecordedFrame]:
        """Its frames, ``hints.fps`` a second, each the latest the followed tab showed, until the recording ends."""
        self._max_edge = hints.max_edge
        interval, due = 1 / hints.fps, self._loop.time()
        try:
            while True:
                await self._keep_up()
                yield RecordedFrame(await self._latest(), datetime.now(timezone.utc))
                due = max(due + interval, self._loop.time())
                await asyncio.sleep(due - self._loop.time())
        finally:
            await self.stop()

    async def stop(self) -> None:
        """No screencast or capture of its own left running."""
        if self._capturing is not None:
            self._capturing.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._capturing
            self._capturing = None
        if self._seeding is not None:
            self._seeding.cancel()
            self._seeding = None
        if (casting := self._casting) is not None:
            self._casting = None
            await self.control.unscreencast(casting[1], casting[2])

    def _move(self, target: Optional[str]) -> None:
        """Following ``target`` from now: what showed of the tab before it goes, and a capture of it is asked for at
        once, so the video never shows one tab while ``followed`` names another."""
        if target != self.tab:
            self._jpeg, self._image = None, None
            if self._seeding is not None:
                self._seeding.cancel()
            self._seeding = asyncio.ensure_future(self._seed(target)) if target is not None else None
        self.tab = target
        if target is not None:
            if target in self._history:
                self._history.remove(target)
            self._history.append(target)
        self._noted(None)

    async def _seed(self, target: str) -> None:
        """A first frame of ``target``, captured, unless its screencast sends one first or it's left meanwhile."""
        with contextlib.suppress(asyncio.TimeoutError):
            jpeg = await asyncio.wait_for(self.control.capture_tab(target), FIRST_CAPTURE_S)
            if jpeg is not None and self.tab == target and self._jpeg is None:
                self._jpeg, self._image = jpeg, None

    def _noted(self, via: Optional[str]) -> None:
        """What the video shows from now in ``followed``: the tab and how its frames come, once either changes."""
        via = BLANK if self.tab is None else via
        last = self.followed[-1] if self.followed else None
        if via is None or (last is not None and last["tab"] == self.tab and last["via"] == via):
            return
        self._via = via
        self.followed.append({"tab": self.tab, "since": _iso(datetime.now(timezone.utc)), "via": via})

    async def _keep_up(self) -> None:
        """The screencast on the followed tab, a capture of it while it sends nothing, and none of either left on a
        tab it no longer follows."""
        if self._casting is not None and (self._casting[0] != self.tab or
                                          self.control.session_of(self.tab) != self._casting[1]):
            old, self._casting = self._casting, None
            await self.control.unscreencast(old[1], old[2])
        if self.tab is None:
            self._jpeg, self._image = None, None
            self._noted(BLANK)
            return
        target = self.tab

        def heard(jpeg: bytes) -> None:
            self._heard(target, jpeg)

        if self._casting is None and (session := await self.control.screencast(target, heard,
                                                                                self._max_edge)) is not None:
            self._casting = (target, session, heard)
            self._heard_at = self._loop.time()
            self._noted(SCREENCAST)
        silent = self._loop.time() - self._heard_at >= SILENT_S
        if silent and self._capturing is None:
            self._capturing = asyncio.ensure_future(self._capture(self.tab))
        elif not silent and self._via != SCREENCAST and self._casting is not None:
            self._noted(SCREENCAST)

    def _heard(self, target: str, jpeg: bytes) -> None:
        if target == self.tab:
            self._jpeg, self._image, self._heard_at = jpeg, None, self._loop.time()

    async def _capture(self, target: str) -> None:
        """Captures of ``target`` while it's followed, sends no screencast frame and isn't visible."""
        try:
            while self.tab == target and self._loop.time() - self._heard_at >= SILENT_S:
                if await self.control.visible(target):
                    await asyncio.sleep(1 / FALLBACK_FPS)
                    continue
                self._noted(SCREENSHOTS)
                if (jpeg := await self.control.capture_tab(target)) is not None and self.tab == target and \
                        self._loop.time() - self._heard_at >= SILENT_S:
                    self._jpeg, self._image = jpeg, None
                await asyncio.sleep(1 / FALLBACK_FPS)
        finally:
            self._capturing = None

    async def _latest(self) -> Image.Image:
        """The latest frame decoded, in a thread, as each recorded context's frames would hold up the event loop; or
        blank at the window's size while there's none."""
        if self._image is not None:
            return self._image
        jpeg = self._jpeg
        image = await asyncio.to_thread(_decoded, jpeg) if jpeg else None
        if image is None:
            if jpeg is not None and self._jpeg is jpeg:
                logger.warning(f"a frame of tab {self.tab} didn't decode")
                self._jpeg = None
            image = Image.new("RGB", self.blank_size, "white")
        if self._jpeg is jpeg:
            self._image = image
        return image


def _decoded(jpeg: bytes) -> Optional[Image.Image]:
    try:
        return Image.open(io.BytesIO(jpeg)).convert("RGB")
    except OSError:
        return None


def _iso(at: datetime) -> str:
    return at.isoformat(timespec="milliseconds").replace("+00:00", "Z")
