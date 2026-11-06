"""The Playwright MCP browser's recorder: computer-recording/v1 of the page Playwright MCP drives, from Chrome DevTools
screencast frames, on a card and port of its own beside the MCP server. It is idle until a recording starts, which
finds the browser through the remote debugging port Playwright opens for an ``--isolated`` Chromium and follows the
page the agent is on. It runs standalone in the browser's image, so it imports only
``agentenv-framework-protocol[computer]`` and, to talk to the browser, ``websockets``.

The same card serves browser/v1's ``cookies/set`` and ``cookies/delete``, in the shapes a website browser's take, so a
step can start the browser signed in to a site. Playwright MCP starts each browser context it opens from the storage state file ``--storage-state`` names
(``PLAYWRIGHT_STORAGE_STATE``), and launches its browser only at the agent's first tool call, so the cookies go into
that file, and through DevTools into every context of a browser already running."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import itertools
import json
import logging
import os
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncContextManager, Optional
from urllib.parse import urlsplit

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
from agentenv_protocol.errors import AgentEnvError
from agentenv_protocol.website_browser.shapes import CookieToSet
from pydantic import TypeAdapter, ValidationError

logger = logging.getLogger("playwright-mcp-recorder")

#: The port the recorder serves its card on, unless RECORDER_PORT says another; the gateway reads the browser's card
#: there (agent_env.env.envs.playwright_mcp.PLAYWRIGHT_MCP_RECORDER_PORT).
DEFAULT_PORT = 18766
#: How often the browser is looked for while a recording has none.
FIND_S = 0.5
#: A page that hasn't changed is filmed again this often, so a still page doesn't split the recording at a gap.
HOLD_S = 1.0
#: How long one DevTools command may take.
CALL_S = 10.0
JPEG_QUALITY = 80
DEFAULT_STORAGE_STATE = "/app/storage-state.json"
BROWSER_EXTENSION_URI = "urn:agentenv:browser/v1"


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


class BrowserCookies:
    """browser/v1's cookies: kept in the storage state file each new browser context starts from, and set in, or
    deleted from, every context of the browser that is running, if one is. One change runs at a time, its file
    write and DevTools calls together, so a concurrent set or delete never restores what another removed. A cookie is
    kept as a storage state file lists it: ``expires`` in epoch seconds, -1 for one that ends with the session."""

    def __init__(self, path: Path, find: Callable[[], Optional[int]],
                 connect: Callable[[int], AsyncContextManager[Cdp]]) -> None:
        self.path, self._find, self._connect = path, find, connect
        self._lock = asyncio.Lock()

    async def set(self, given: list[CookieToSet]) -> dict:
        async with self._lock:
            placed = [_placed(c, time.time()) for c in given]
            kept = {_key(c): c for c in self._read()}
            kept.update({_key(c): c for c in placed})
            self._write(list(kept.values()))

            async def add(cdp: Cdp, context: dict) -> None:
                await cdp.call("Storage.setCookies", {"cookies": [_settable(c) for c in placed], **context})

            return {"set": len(placed), "open_browser": await self._each_context(add)}

    async def delete(self, name: Optional[str], domain: Optional[str], path: Optional[str]) -> dict:
        async with self._lock:
            def gone(c: dict) -> bool:
                return ((name is None or c.get("name") == name) and (domain is None or c.get("domain") == domain)
                        and (path is None or c.get("path") == path))

            before = self._read()
            self._write([c for c in before if not gone(c)])

            async def remove(cdp: Cdp, context: dict) -> None:
                current = (await cdp.call("Storage.getCookies", context)).get("cookies") or []
                if not any(gone(c) for c in current):
                    return
                await cdp.call("Storage.clearCookies", context)
                if others := [_settable(c) for c in current if not gone(c)]:
                    await cdp.call("Storage.setCookies", {"cookies": others, **context})

            return {"deleted": sum(gone(c) for c in before), "open_browser": await self._each_context(remove)}

    async def _each_context(self, act: Callable[[Cdp, dict], Any]) -> bool:
        """``act`` on the default context and each other one of the running browser; False when none runs. An error
        names the DevTools call's kind only, since its message may repeat a cookie."""
        port = await asyncio.to_thread(self._find)
        if port is None:
            return False
        try:
            async with self._connect(port) as cdp:
                contexts = (await cdp.call("Target.getBrowserContexts")).get("browserContextIds") or []
                for context in [{}, *({"browserContextId": c} for c in contexts)]:
                    await act(cdp, context)
        except (OSError, CdpError, httpx.HTTPError, KeyError, ValueError, asyncio.TimeoutError) as e:
            raise AgentEnvError("browser_unreachable", f"the running browser didn't take the change "
                                f"({type(e).__name__}); the next context it opens starts with it", applied="unknown",
                                status=502) from None
        return True

    def _read(self) -> list[dict]:
        try:
            return list(json.loads(self.path.read_text(encoding="utf-8")).get("cookies") or [])
        except FileNotFoundError:
            return []

    def _write(self, kept: list[dict]) -> None:
        try:
            origins = json.loads(self.path.read_text(encoding="utf-8")).get("origins") or []
        except (FileNotFoundError, ValueError):
            origins = []
        staged = self.path.with_name(self.path.name + ".tmp")
        staged.write_text(json.dumps({"cookies": kept, "origins": origins}), encoding="utf-8")
        os.chmod(staged, 0o600)
        os.replace(staged, self.path)


def _placed(cookie: CookieToSet, now: float) -> dict:
    """``cookie`` as a storage state file lists it: at its ``url``'s host and ``/``, or its ``domain`` and ``path``,
    a ``max_age`` turned into an expiry from ``now``."""
    split = urlsplit(cookie.url) if cookie.url else None
    expires = cookie.expires if cookie.expires is not None else (
        now + cookie.max_age if cookie.max_age is not None else -1)
    placed = {"name": cookie.name, "value": cookie.value, "domain": cookie.domain or split.hostname,
              "path": cookie.path or "/", "expires": expires, "httpOnly": cookie.httpOnly,
              "secure": cookie.secure or (split is not None and split.scheme == "https"),
              "sameSite": cookie.sameSite or "Lax"}
    if cookie.partitionKey is not None:
        placed["partitionKey"] = cookie.partitionKey.model_dump()
    return placed


def _key(cookie: dict) -> tuple:
    """What a browser keeps one cookie under, a partitioned one's partition among it: a later one with the same key
    replaces it."""
    partition = cookie.get("partitionKey") or {}
    return (cookie["name"], str(cookie["domain"]).lower(), cookie.get("path") or "/", partition.get("topLevelSite"),
            partition.get("hasCrossSiteAncestor"))


def _settable(cookie: dict) -> dict:
    """A cookie as ``Storage.setCookies`` takes it: one kept in the file, or one DevTools listed."""
    param = {k: cookie[k] for k in ("name", "value", "domain", "path", "secure", "httpOnly", "sameSite", "partitionKey")
             if k in cookie}
    if not cookie.get("session", False) and cookie.get("expires", -1) not in (-1, None):
        param["expires"] = cookie["expires"]
    return param


@environment_card(name="playwright-mcp")
class BrowserRecorder(AgentEnvEnvironment):
    """computer-recording/v1 of the Playwright MCP browser's page, and browser/v1's cookies; the browser's tools stay
    Playwright MCP's."""

    recording = Recording()

    def __init__(self, find: Callable[[], Optional[int]] = find_browser,
                 connect: Callable[[int], AsyncContextManager[Cdp]] = connect_cdp,
                 storage_state_path: Optional[Path] = None) -> None:
        self._find, self._connect = find, connect
        self.recordings = Recordings(_Host(self))
        path = storage_state_path or Path(os.environ.get("PLAYWRIGHT_STORAGE_STATE", DEFAULT_STORAGE_STATE))
        self.cookies = BrowserCookies(path, find, connect)

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

    @extension(BROWSER_EXTENSION_URI, op="cookies/set",
               description="The browser's cookies, for agentenv's steps, never its agent, in a website browser's "
                           "shapes: cookies/set adds cookies (name, value, url or domain and path, expires or max_age, "
                           "httpOnly, secure, sameSite) to every context, open and to come, and cookies/delete removes "
                           "those with all of name, domain and path given, or all of them.")
    async def _cookies_set(self, cookies: list[dict]) -> dict:
        try:
            given = TypeAdapter(list[CookieToSet]).validate_python(cookies)
        except ValidationError as e:
            raise AgentEnvError("invalid_params", "; ".join(
                f"cookies[{'.'.join(map(str, err['loc']))}]: {err['msg']}" for err in e.errors())) from None
        return await self.cookies.set(given)

    @extension(BROWSER_EXTENSION_URI, op="cookies/delete")
    async def _cookies_delete(self, name: Optional[str] = None, domain: Optional[str] = None,
                              path: Optional[str] = None) -> dict:
        return await self.cookies.delete(name, domain, path)

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
