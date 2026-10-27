"""The website browser's recorder: computer-recording/v1 of the page Playwright MCP drives, from Chrome DevTools
screencast frames, on a card and port of its own beside the MCP server. It is idle until a recording starts, which
finds the browser through the remote debugging port Playwright opens for an ``--isolated`` Chromium and follows the
page the agent is on. It runs standalone in the website browser's image, so it imports only
``agentenv-framework-protocol[computer]`` and, to talk to the browser, ``websockets``."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import itertools
import json
import logging
import os
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncContextManager, Optional

import httpx
from PIL import Image as PILImage

from agentenv_protocol.agent_env_environment import (
    AgentEnvEnvironment,
    AgentEnvStarletteApplication,
    environment_card,
    extension,
)
from agentenv_protocol.computer.card import COMPUTER_RECORDING
from agentenv_protocol.computer.recording import RecordedFrame, RecordHints, Recording, Recordings, recording_params

logger = logging.getLogger("website-browser-recorder")

#: The port the recorder serves its card on, unless RECORDER_PORT says another; the gateway reads the browser's card
#: there (agent_env.env.envs.website_browser.WEBSITE_BROWSER_RECORDER_PORT).
DEFAULT_PORT = 18766
#: How often the browser is looked for while a recording has none.
FIND_S = 0.5
#: A page that hasn't changed is filmed again this often, so a still page doesn't split the recording at a gap.
HOLD_S = 1.0
#: How long one DevTools command may take.
CALL_S = 10.0
JPEG_QUALITY = 80


class CdpError(Exception):
    """A DevTools command the browser refused, or one it can no longer answer."""


class Cdp:
    """One DevTools connection to the browser: commands answered by id, and everything else, in order, as events."""

    def __init__(self, ws: Any) -> None:
        self._ws = ws
        self._ids = itertools.count(1)
        self._waiting: dict[int, asyncio.Future] = {}
        self._events: asyncio.Queue = asyncio.Queue()

    async def call(self, method: str, params: Optional[dict] = None, session: Optional[str] = None) -> dict:
        request_id = next(self._ids)
        answer = asyncio.get_running_loop().create_future()
        self._waiting[request_id] = answer
        message = {"id": request_id, "method": method, "params": params or {}}
        if session is not None:
            message["sessionId"] = session
        try:
            await self._ws.send(json.dumps(message))
            return await asyncio.wait_for(answer, CALL_S)
        finally:
            self._waiting.pop(request_id, None)

    async def events(self) -> AsyncIterator[tuple[str, dict, Optional[str]]]:
        """(method, params, session) of each event, until the connection closes."""
        while (event := await self._events.get()) is not None:
            yield event

    async def read(self) -> None:
        """Hand each message to its command or the events, until the connection closes."""
        try:
            async for raw in self._ws:
                message = json.loads(raw)
                if "id" in message:
                    answer = self._waiting.get(message["id"])
                    if answer is not None and not answer.done():
                        if "error" in message:
                            answer.set_exception(CdpError(message["error"].get("message", message["error"])))
                        else:
                            answer.set_result(message.get("result") or {})
                elif "method" in message:
                    self._events.put_nowait((message["method"], message.get("params") or {}, message.get("sessionId")))
        finally:
            for answer in self._waiting.values():
                if not answer.done():
                    answer.set_exception(CdpError("the browser closed the connection"))
            self._events.put_nowait(None)


@contextlib.asynccontextmanager
async def connect_cdp(port: int) -> AsyncIterator[Cdp]:
    """A DevTools connection to the browser debugging on ``port`` of this host."""
    from websockets.asyncio.client import connect

    async with httpx.AsyncClient() as client:
        version = await client.get(f"http://127.0.0.1:{port}/json/version", timeout=CALL_S)
    url = version.raise_for_status().json()["webSocketDebuggerUrl"]
    async with connect(url, max_size=None, ping_interval=None, compression=None, open_timeout=CALL_S) as ws:
        cdp = Cdp(ws)
        reader = asyncio.create_task(cdp.read())
        try:
            yield cdp
        finally:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await reader


def find_browser(proc: Path = Path("/proc")) -> Optional[int]:
    """The remote debugging port of the newest Chromium browser process here, or None while there is none."""
    newest: Optional[tuple[int, int]] = None
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
            started = int((entry / "stat").read_text().rsplit(")", 1)[1].split()[19])
        except (OSError, ValueError, IndexError):
            continue
        if any(arg.startswith(b"--type=") for arg in argv):  # a renderer, GPU or utility process
            continue
        port = next((arg.split(b"=", 1)[1] for arg in argv if arg.startswith(b"--remote-debugging-port=")), b"")
        if port.isdigit() and int(port) > 0 and (newest is None or started > newest[0]):
            newest = (started, int(port))
    return newest[1] if newest else None


class Screencast:
    """The screencast of the page the agent is on, in the newest browser: the page opened last, or the last one to
    navigate. ``latest`` is its last frame as JPEG, or None while there is no page."""

    def __init__(self, max_edge: Optional[int], find: Callable[[], Optional[int]],
                 connect: Callable[[int], AsyncContextManager[Cdp]]) -> None:
        self.max_edge = max_edge
        self.latest: Optional[bytes] = None
        self.frames = 0
        self.page: Optional[str] = None
        self._find, self._connect = find, connect
        self._session: Optional[str] = None
        self._pages: OrderedDict[str, str] = OrderedDict()

    async def run(self) -> None:
        """Watch the browser, and the next one when it goes, until cancelled."""
        while True:
            port = await asyncio.to_thread(self._find)
            if port is not None:
                try:
                    async with self._connect(port) as cdp:
                        await self.watch(cdp)
                except (OSError, CdpError, httpx.HTTPError, KeyError, ValueError) as e:
                    logger.info("the browser on port %d: %s: %s", port, type(e).__name__, e)
                except Exception as e:  # the websockets library's own errors, a closed connection among them
                    logger.warning("the browser on port %d went: %s: %s", port, type(e).__name__, e)
                self.latest, self.page, self._session = None, None, None
            await asyncio.sleep(FIND_S)

    async def watch(self, cdp: Cdp) -> None:
        """Follow the browser's pages over ``cdp`` until it closes."""
        self._pages.clear()
        await cdp.call("Target.setDiscoverTargets", {"discover": True})  # announces each page there already is
        async for method, params, session in cdp.events():
            await self._event(cdp, method, params, session)

    async def _event(self, cdp: Cdp, method: str, params: dict, session: Optional[str]) -> None:
        if method == "Page.screencastFrame":
            if session is not None and session == self._session:
                self.latest, self.frames = base64.b64decode(params["data"]), self.frames + 1
            with contextlib.suppress(CdpError, asyncio.TimeoutError):
                await cdp.call("Page.screencastFrameAck", {"sessionId": params["sessionId"]}, session)
        elif method in ("Target.targetCreated", "Target.targetInfoChanged"):
            info = params.get("targetInfo") or {}
            target, url = info.get("targetId"), info.get("url")
            if info.get("type") != "page" or not target:
                return
            if method == "Target.targetCreated" or self._pages.get(target) != url:
                self._pages[target] = url
                self._pages.move_to_end(target)
                await self._show(cdp, target)
        elif method == "Target.targetDestroyed":
            target = params.get("targetId")
            if self._pages.pop(target, None) is not None and target == self.page:
                await self._show(cdp, next(reversed(self._pages), None))
        elif method == "Target.detachedFromTarget" and params.get("sessionId") == self._session:
            self.page, self._session = None, None
            await self._show(cdp, next(reversed(self._pages), None))

    async def _show(self, cdp: Cdp, target: Optional[str]) -> None:
        """Screencast ``target`` instead; the last frame stays until its first arrives."""
        if target == self.page and self._session is not None:
            return
        old, self.page, self._session = self._session, target, None
        if old is not None:
            with contextlib.suppress(CdpError, asyncio.TimeoutError):
                await cdp.call("Target.detachFromTarget", {"sessionId": old})
        if target is None:
            self.latest = None
            return
        options: dict[str, Any] = {"format": "jpeg", "quality": JPEG_QUALITY, "everyNthFrame": 1}
        if self.max_edge is not None:
            options.update(maxWidth=self.max_edge, maxHeight=self.max_edge)
        try:
            session = (await cdp.call("Target.attachToTarget", {"targetId": target, "flatten": True}))["sessionId"]
            await cdp.call("Page.startScreencast", options, session)
        except (CdpError, asyncio.TimeoutError, KeyError) as e:  # the page went meanwhile
            logger.info("screencasting page %s: %s", target, e)
            if self.page == target:  # nothing took over since: back to the newest page left
                self.page = None
                self._pages.pop(target, None)
                await self._show(cdp, next(reversed(self._pages), None))
            return
        if self.page == target:
            self._session = session


@environment_card(name="website-browser")
class BrowserRecorder(AgentEnvEnvironment):
    """computer-recording/v1 of the website browser's page; the browser's tools stay Playwright MCP's."""

    recording = Recording()

    def __init__(self, find: Callable[[], Optional[int]] = find_browser,
                 connect: Callable[[int], AsyncContextManager[Cdp]] = connect_cdp) -> None:
        self._find, self._connect = find, connect
        self.recordings = Recordings(_Host(self))

    def extension_params(self, uri: str) -> dict:
        if uri == COMPUTER_RECORDING:
            return {**recording_params(self.recording, streams=True), "cursor": False}
        return {}

    @extension(COMPUTER_RECORDING, op="start",
               description="Record the browser's page: start returns a recording whose segments upload through its "
                           "grant as they close, and end seals it and returns its manifest; repeats answer the same.")
    async def _recording_start(self, request_id: str, scope: str = "display", window: Optional[str] = None,
                               fps: Optional[int] = None, max_edge: Optional[int] = None, cursor: bool = True,
                               upload: Optional[dict] = None) -> dict:
        return await self.recordings.start(request_id, scope, window, fps, max_edge, False, upload)

    @extension(COMPUTER_RECORDING, op="end")
    async def _recording_end(self, recording_id: str, upload: Optional[dict] = None, reason: str = "end") -> dict:
        return await self.recordings.end(recording_id, upload, reason)

    async def frames(self, hints: RecordHints) -> AsyncIterator[RecordedFrame]:
        """The page at up to ``fps``: a frame each time it changes, and the last one again after ``HOLD_S``."""
        screencast = Screencast(hints.max_edge, self._find, self._connect)
        watching = asyncio.create_task(screencast.run())
        loop = asyncio.get_running_loop()
        interval, due = 1 / hints.fps, loop.time()
        shown: Optional[bytes] = None
        shown_at = 0.0
        try:
            while True:
                due = max(due + interval, loop.time())
                await asyncio.sleep(due - loop.time())
                if watching.done():
                    watching.result()
                frame = screencast.latest
                if frame is not None and (frame is not shown or loop.time() - shown_at >= HOLD_S):
                    shown, shown_at = frame, loop.time()
                    yield RecordedFrame(PILImage.open(io.BytesIO(frame)), datetime.now(timezone.utc))
        finally:
            watching.cancel()


class _Host:
    """What Recordings reads of a computer's runtime: the limits, and a stream to record from."""

    def __init__(self, recorder: BrowserRecorder) -> None:
        self.driver = recorder

    def has(self, capability: str) -> bool:
        return capability == "record"

    def stream(self, hints: RecordHints) -> AsyncIterator[RecordedFrame]:
        return self.driver.frames(hints)


def create_app(recorder: BrowserRecorder) -> Any:
    from starlette.applications import Starlette

    app = Starlette()
    AgentEnvStarletteApplication(recorder._build_card(), recorder).add_routes_to_app(app)
    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s recorder %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # a request each time it finds a browser
    uvicorn.run(create_app(BrowserRecorder()), host=os.environ.get("MCP_HOST", "0.0.0.0"),
                port=int(os.environ.get("RECORDER_PORT", DEFAULT_PORT)), log_level="warning")


if __name__ == "__main__":
    main()
