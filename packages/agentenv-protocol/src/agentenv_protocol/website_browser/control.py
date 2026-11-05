"""agentenv's control of a browser over its pipe, and the record it keeps."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Optional

from ..computer.errors import DeviceUnavailable, InvalidArgument
from .devtools import DevToolsError, DevToolsPipe

logger = logging.getLogger(__name__)

#: Who closed a tab: agentenv over its own connection, or someone it can't tell; or how its browser ended, on its own
#: or stopped by its server.
AGENTENV, UNKNOWN, EXIT, STOP = "agentenv", "unknown", "exit", "stop"
#: How long the active tab's check waits on each page.
ACTIVE_CHECK_TIMEOUT_S = 2.0
#: How long a tab opened headless may take to be attached and held to its window's size.
PIN_TIMEOUT_S = 10.0
#: How much of a download each read hashes.
HASH_CHUNK_BYTES = 1 << 20
#: Who answered a dialog when agentenv didn't.
AGENT = "agent"
#: The targets agentenv's connection follows.
PAGES = [{"type": "page"}]
#: The CDP input a computer's page driver may send to the active tab, and how long its screenshot may take.
PAGE_INPUT = frozenset({"Input.dispatchMouseEvent", "Input.dispatchKeyEvent", "Input.insertText"})
CAPTURE_TIMEOUT_S, UNNAVIGATED_CAPTURE_TIMEOUT_S = 3.0, 1.0
#: Where an agent's files to upload are, and where downloads land, under the session user's home.
UPLOADS = "Uploads"
DOWNLOADS = "Downloads"


class BrowserControl:
    """agentenv's control of a browser over its pipe: at each launch it attaches to every page with flattened sessions
    and keeps a record of the browser since this server started, its open tabs in opened order, those closed and who
    closed them, its downloads with their size and sha256, and its JavaScript dialogs, which it answers when
    ``dialogs`` says to: always leaving a page that asks before it's left, so no answer of agentenv's blocks a
    navigation."""

    def __init__(self, devtools: DevToolsPipe, dialogs: str, viewport: Optional[tuple[int, int]] = None) -> None:
        self.devtools, self.dialogs, self.viewport = devtools, dialogs, viewport
        #: When this record starts: a server that restarts starts another.
        self.since = _now()
        self.tabs: dict[str, dict] = {}
        self.closed_tabs: list[dict] = []
        self.downloads: dict[str, dict] = {}
        self.dialog_log: list[dict] = []
        self._sessions: dict[str, str] = {}
        self._hidden: set[str] = set()
        self._closing: set[str] = set()
        self._open_dialogs: dict[str, dict] = {}
        self._answering: set[str] = set()
        #: The file chooser each page's session opened that waits, headless, and the tab agentenv last brought to the
        #: front.
        self._choosers: dict[str, dict] = {}
        self._front: Optional[str] = None
        #: Each tab's latest capture, which shows while a dialog keeps its page from painting; and, by session, what
        #: input sent there waits on to learn a dialog opened.
        self._shots: dict[str, str] = {}
        self._dialog_opened: dict[str, asyncio.Event] = {}
        self._pinned: dict[str, asyncio.Event] = {}
        self._tasks: set[asyncio.Future] = set()
        devtools.listen(self._event)

    async def attach(self) -> None:
        """Follow a browser just launched: any tabs the one before left open closed by its server's stop, every page
        this one has or opens attached, and its downloads reported, which go where its own policies say, as Chrome
        downloads unattended: into ``~/Downloads``, a name downloaded again kept beside the first."""
        self.ended(STOP)
        await self.devtools.call("Target.setDiscoverTargets", {"discover": True, "filter": PAGES})
        await self.devtools.call("Target.setAutoAttach", {"autoAttach": True,
                                                          "waitForDebuggerOnStart": self.viewport is not None,
                                                          "flatten": True, "filter": PAGES})
        await self.devtools.call("Browser.setDownloadBehavior", {"behavior": "default", "eventsEnabled": True})

    def ended(self, how: str) -> None:
        """The browser is gone, ``how`` it ended: its open tabs are closed by that, and its sessions with them."""
        now = _now()
        self.closed_tabs += [{**tab, "closed_at": now, "closed_by": how} for tab in self.tabs.values()]
        self.tabs.clear()
        self._sessions.clear()
        self._open_dialogs.clear()
        self._choosers.clear()
        self._shots.clear()
        self._dialog_opened.clear()
        self._pinned.clear()

    async def close_tab(self, target: str) -> None:
        """Close the tab ``target``, which the record says agentenv closed."""
        self._closing.add(target)
        await self.devtools.call("Target.closeTarget", {"targetId": target})

    async def tab_list(self) -> list[dict]:
        """The open tabs in opened order, each with its URL and title now and whether it's active."""
        await self._refresh()
        active = await self.active_tab()
        return [{"id": tab["id"], "url": tab["url"], "title": tab["title"], "active": tab["id"] == active}
                for tab in self.tabs.values()]

    async def open_tab(self, url: Optional[str] = None) -> str:
        """A new tab at ``url``, else blank, brought to the front; its id. Headless, it opens blank and goes to ``url``
        once it's held to the window's size, so its page loads at that size."""
        held = self.viewport is not None and url
        start = "about:blank" if held else url or "about:blank"
        target = (await self.devtools.call("Target.createTarget", {"url": start}))["targetId"]
        if held:
            await asyncio.wait_for(self._pinned.setdefault(target, asyncio.Event()).wait(), PIN_TIMEOUT_S)
            session = next(s for s, t in self._sessions.items() if t == target)
            await self.devtools.call("Page.navigate", {"url": url}, session_id=session)
        await self._bring_to_front(target)
        return target

    async def activate(self, target: str) -> None:
        """The open tab ``target`` brought to the front, the active one."""
        self._open(target)
        await self._bring_to_front(target)

    async def close(self, target: str) -> None:
        """The open tab ``target`` closed, which the record says agentenv closed."""
        self._open(target)
        await self.close_tab(target)

    async def navigate(self, url: str) -> None:
        """The active tab opened at ``url``; InvalidArgument, with the browser's reason, when it can't."""
        answer = await self.devtools.call("Page.navigate", {"url": url}, session_id=await self.active_session())
        if error := answer.get("errorText"):
            raise InvalidArgument(f"the browser couldn't open {url}: {error}")

    async def history(self, direction: str) -> None:
        """The active tab's page reloaded, or the one before or after it in its history opened."""
        session = await self.active_session()
        if direction == "reload":
            await self.devtools.call("Page.reload", {}, session_id=session)
            return
        if direction not in ("back", "forward"):
            raise InvalidArgument(f"direction is back, forward or reload, not {direction!r}")
        history = await self.devtools.call("Page.getNavigationHistory", session_id=session)
        index = history["currentIndex"] + (-1 if direction == "back" else 1)
        if not 0 <= index < len(history["entries"]):
            raise InvalidArgument(f"the active tab has no page to go {direction} to")
        await self.devtools.call("Page.navigateToHistoryEntry", {"entryId": history["entries"][index]["id"]},
                                 session_id=session)

    async def answer_dialog(self, accept: bool, text: Optional[str] = None) -> None:
        """The dialog open on the active tab answered, as the agent's answer; InvalidArgument with none open."""
        active = await self.active_tab()
        session = next((s for s, dialog in self._open_dialogs.items() if dialog["tab"] == active), None)
        if session is None:
            raise InvalidArgument("no dialog is open on the page")
        await self.devtools.call("Page.handleJavaScriptDialog", {
            "accept": accept, **({"promptText": text} if text is not None else {})}, session_id=session)

    async def choose_files(self, paths: list[str]) -> None:
        """The file chooser open on the active tab given ``paths``, files in the session user's ``~/Uploads``;
        InvalidArgument for one outside it or missing, for several where it takes one, or with no chooser open."""
        active = await self.active_tab()
        session = next((s for s in self._choosers if self._sessions.get(s) == active), None)
        if session is None:
            raise InvalidArgument("no file chooser is open on the page")
        uploads = (Path.home() / UPLOADS).resolve()
        files = []
        for path in paths:
            file = (uploads / path).resolve()
            if not file.is_relative_to(uploads):
                raise InvalidArgument(f"{path} isn't in ~/{UPLOADS}")
            if not file.is_file():
                raise InvalidArgument(f"there's no file {path} in ~/{UPLOADS}")
            files.append(str(file))
        chooser = self._choosers[session]
        if len(files) > 1 and not chooser["multiple"]:
            raise InvalidArgument("the file chooser takes one file")
        await self.devtools.call("DOM.setFileInputFiles", {"files": files, "backendNodeId": chooser["node"]},
                                 session_id=session)
        self._choosers.pop(session, None)

    async def state(self) -> dict:
        """The active tab's id, URL and title now, and the dialog and file chooser open on it."""
        await self._refresh()
        active = await self.active_tab()
        if active is None:
            raise DeviceUnavailable("the browser has no tab open")
        return self._state_of(active)

    def _state_of(self, target: str) -> dict:
        tab = self.tabs.get(target) or {"url": "", "title": ""}
        state: dict[str, Any] = {"tab": target, "url": tab["url"], "title": tab["title"]}
        if dialog := next((d for d in self._open_dialogs.values() if d["tab"] == target), None):
            state["dialog"] = {"type": dialog["type"], "message": dialog["message"],
                               "default_text": dialog["default_text"]}
        if chooser := next((c for s, c in self._choosers.items() if self._sessions.get(s) == target), None):
            state["file_chooser"] = {"multiple": chooser["multiple"]}
        return state

    def _open(self, target: str) -> None:
        if target not in self.tabs:
            raise InvalidArgument(f"no tab {target} is open; list the tabs for their ids")

    async def _bring_to_front(self, target: str) -> None:
        await self.devtools.call("Target.activateTarget", {"targetId": target})
        if (session := next((s for s, t in self._sessions.items() if t == target), None)) is not None:
            with contextlib.suppress(DevToolsError):
                await self.devtools.call("Page.bringToFront", session_id=session)
        self._front = target

    @contextlib.asynccontextmanager
    async def hidden_page(self, url: str = "about:blank") -> AsyncIterator[str]:
        """A page of agentenv's own for reading state, in no window and never in the record; its session's id. It's
        closed after."""
        target = (await self.devtools.call("Target.createTarget",
                                           {"url": url, "hidden": True, "background": True}))["targetId"]
        self._hidden.add(target)
        self.tabs.pop(target, None)
        try:
            yield (await self.devtools.call("Target.attachToTarget", {"targetId": target, "flatten": True}))["sessionId"]
        finally:
            with contextlib.suppress(ConnectionError, DevToolsError):
                await self.devtools.call("Target.closeTarget", {"targetId": target})

    async def active_tab(self) -> Optional[str]:
        """The tab a dialog is open on, whose page it blocks from saying; else the visible tab whose page has focus, as
        a hidden tab's page that had input still says it has; else the one agentenv last brought to the front, else the
        first visible one: CDP has no event for a tab made active."""
        if dialogs := [dialog["tab"] for dialog in self._open_dialogs.values() if dialog["tab"] in self.tabs]:
            return dialogs[-1]
        visible = None
        for session, target in list(self._sessions.items()):
            if target not in self.tabs:
                continue
            try:
                state = (await self.devtools.call(
                    "Runtime.evaluate", {"expression": "[document.visibilityState, document.hasFocus()]",
                                         "returnByValue": True}, session_id=session,
                    timeout_s=ACTIVE_CHECK_TIMEOUT_S))["result"]["value"]
            except (ConnectionError, DevToolsError, asyncio.TimeoutError, KeyError):
                continue
            if state[0] == "visible" and state[1]:
                return target
            if state[0] == "visible" and visible is None:
                visible = target
        return self._front if self._front in self.tabs else visible

    async def record(self) -> dict:
        """The record ``since`` it started: open ``tabs``, each with its URL and title now and whether it's ``active``,
        ``closed_tabs``, ``downloads`` and ``dialogs``."""
        await self._refresh()
        active = await self.active_tab()
        return {"since": self.since, "tabs": [{**tab, "active": tab["id"] == active} for tab in self.tabs.values()],
                "closed_tabs": list(self.closed_tabs), "downloads": list(self.downloads.values()),
                "dialogs": list(self.dialog_log)}

    async def active_session(self) -> str:
        """The session of the active tab, else of the first open one; DeviceUnavailable with no tab open."""
        active = await self.active_tab() or next(iter(self.tabs), None)
        session = next((session for session, target in self._sessions.items() if target == active), None)
        if session is None:
            raise DeviceUnavailable("the browser has no tab open")
        return session

    async def capture(self) -> Optional[str]:
        """The active tab's page as it shows now, a PNG, base64; None for a tab that paints nothing within
        ``CAPTURE_TIMEOUT_S``, which a desktop browser shows blank, or sooner for a tab with no URL yet, such as a new
        tab whose navigation the browser blocked, which may never paint."""
        return await self._capture(await self.active_session())

    async def capture_page(self) -> dict:
        """The active tab's capture, ``png``, else ``blank``, with its ``page`` state, of that tab whatever becomes
        active meanwhile."""
        session = await self.active_session()
        target = self._sessions[session]
        png = await self._capture(session)
        await self._refresh(target)
        return {**({"png": png} if png is not None else {"blank": True}), "page": self._state_of(target)}

    async def _capture(self, session: str) -> Optional[str]:
        """The tab's page as it shows now; with a dialog open on it, which keeps it from painting, the last it took."""
        target = self._sessions[session]
        if session in self._open_dialogs:
            return self._shots.get(target)
        navigated = self.tabs.get(target, {}).get("url")
        try:
            shot = await self.devtools.call("Page.captureScreenshot", {"format": "png", "optimizeForSpeed": True},
                                            session_id=session,
                                            timeout_s=CAPTURE_TIMEOUT_S if navigated else UNNAVIGATED_CAPTURE_TIMEOUT_S)
        except asyncio.TimeoutError:
            return self._shots.get(target) if session in self._open_dialogs else None
        self._shots[target] = shot["data"]
        return shot["data"]

    async def dispatch(self, events: list[dict]) -> None:
        """Each of ``events``, a CDP input call, sent to the active tab in order, until one opens a dialog left to the
        agent: the page answers that one only once the dialog is answered, so it finishes then, and the rest aren't
        sent. A dialog answered for the agent is answered at once, and the rest go on. InvalidArgument for any other
        call."""
        if refused := [event.get("method") for event in events if event.get("method") not in PAGE_INPUT]:
            raise InvalidArgument(f"page input takes {', '.join(sorted(PAGE_INPUT))}, not {refused}")
        session = await self.active_session()
        opened = self._dialog_opened.setdefault(session, asyncio.Event())
        for event in events:
            if self.dialogs == AGENT and session in self._open_dialogs:
                return
            opened.clear()
            sent = asyncio.ensure_future(self.devtools.call(event["method"], event.get("params") or {},
                                                            session_id=session))
            waiting = asyncio.ensure_future(opened.wait())
            try:
                await asyncio.wait({sent, waiting}, return_when=asyncio.FIRST_COMPLETED)
            finally:
                waiting.cancel()
            if not sent.done():
                if self.dialogs != AGENT:
                    await sent
                    continue
                self._tasks.add(sent)
                sent.add_done_callback(self._finished)
                return
            sent.result()

    async def _refresh(self, target: Optional[str] = None) -> None:
        """The URL and title of the tab ``target``, or of every open tab, as the browser has them now: it sends no
        event when a page's title changes."""
        try:
            infos = (await self.devtools.call("Target.getTargets", {"filter": PAGES},
                                              timeout_s=ACTIVE_CHECK_TIMEOUT_S))["targetInfos"]
        except (ConnectionError, DevToolsError, asyncio.TimeoutError, KeyError):
            return
        for info in infos:
            if target in (None, info["targetId"]) and (tab := self.tabs.get(info["targetId"])) is not None:
                tab.update(url=info.get("url", tab["url"]), title=info.get("title", tab["title"]))

    def _event(self, message: dict) -> None:
        """An event on this connection's own sessions; one on a session an agent's CDP connection holds is the
        agent's."""
        session = message.get("sessionId")
        handler = getattr(self, "_on_" + message.get("method", "").replace(".", "_"), None)
        if handler is not None and (session is None or session in self._sessions):
            handler(message.get("params", {}), session)

    def _on_Target_targetCreated(self, params: dict, session: Optional[str]) -> None:
        info = params["targetInfo"]
        if info.get("type") == "page" and info["targetId"] not in self._hidden and info["targetId"] not in self.tabs:
            self.tabs[info["targetId"]] = {"id": info["targetId"], "url": info.get("url", ""),
                                           "title": info.get("title", ""), "opened_at": _now()}

    def _on_Target_targetInfoChanged(self, params: dict, session: Optional[str]) -> None:
        info = params["targetInfo"]
        if (tab := self.tabs.get(info["targetId"])) is not None:
            tab.update(url=info.get("url", tab["url"]), title=info.get("title", tab["title"]))

    def _on_Target_targetDestroyed(self, params: dict, session: Optional[str]) -> None:
        target = params["targetId"]
        if (tab := self.tabs.pop(target, None)) is not None:
            self.closed_tabs.append({**tab, "closed_at": _now(),
                                     "closed_by": AGENTENV if target in self._closing else UNKNOWN})
        self._closing.discard(target)
        self._hidden.discard(target)

    def _on_Target_attachedToTarget(self, params: dict, session: Optional[str]) -> None:
        if params["targetInfo"].get("type") == "page":
            self._sessions[params["sessionId"]] = params["targetInfo"]["targetId"]
            self._spawn(self._let_run(params["sessionId"]))

    async def _let_run(self, session: str) -> None:
        """A page it attached to, enabled and, headless, held to its window's size, then told to run: a page attached as
        it opens, a popup say, or any new page headless, waits for that before it runs."""
        target = self._sessions.get(session)
        try:
            await self.devtools.call("Page.enable", session_id=session)
            if self.viewport is not None:
                width, height = self.viewport
                await self.devtools.call("Emulation.setDeviceMetricsOverride", {
                    "width": width, "height": height, "deviceScaleFactor": 1, "mobile": False}, session_id=session)
                await self.devtools.call("Page.setInterceptFileChooserDialog", {"enabled": True}, session_id=session)
        finally:
            try:
                await self.devtools.call("Runtime.runIfWaitingForDebugger", session_id=session)
            finally:
                if target is not None:
                    self._pinned.setdefault(target, asyncio.Event()).set()

    def _on_Target_detachedFromTarget(self, params: dict, session: Optional[str]) -> None:
        if (target := self._sessions.pop(params["sessionId"], None)) is not None:
            self._shots.pop(target, None)
            self._pinned.pop(target, None)
        self._choosers.pop(params["sessionId"], None)
        self._dialog_opened.pop(params["sessionId"], None)

    def _on_Page_loadEventFired(self, params: dict, session: Optional[str]) -> None:
        self._choosers.pop(session, None)
        if (target := self._sessions.get(session)) is not None:
            self._spawn(self._refresh(target))

    def _on_Page_fileChooserOpened(self, params: dict, session: Optional[str]) -> None:
        if session in self._sessions:
            self._choosers[session] = {"node": params.get("backendNodeId"),
                                       "multiple": params.get("mode") == "selectMultiple"}

    def _on_Page_javascriptDialogOpening(self, params: dict, session: Optional[str]) -> None:
        dialog = {"tab": self._sessions.get(session), "type": params.get("type"), "message": params.get("message", ""),
                  "default_text": params.get("defaultPrompt", ""), "url": params.get("url", ""), "opened_at": _now(),
                  "answer": None, "by": None}
        self.dialog_log.append(dialog)
        self._open_dialogs[session] = dialog
        if (opened := self._dialog_opened.get(session)) is not None:
            opened.set()
        if self.dialogs != AGENT:
            self._answering.add(session)
            leave = params.get("type") == "beforeunload"
            self._spawn(self.devtools.call("Page.handleJavaScriptDialog", {
                "accept": leave or self.dialogs == "accept", "promptText": params.get("defaultPrompt", "")},
                session_id=session))

    def _on_Page_javascriptDialogClosed(self, params: dict, session: Optional[str]) -> None:
        if (dialog := self._open_dialogs.pop(session, None)) is not None:
            dialog.update(answer="accepted" if params.get("result") else "dismissed", answered_at=_now(),
                          by=AGENTENV if session in self._answering else AGENT,
                          **({"input": params["userInput"]} if params.get("userInput") else {}))
        self._answering.discard(session)

    def _on_Browser_downloadWillBegin(self, params: dict, session: Optional[str]) -> None:
        self.downloads[params["guid"]] = {"guid": params["guid"], "url": params.get("url", ""),
                                          "filename": params.get("suggestedFilename", ""), "state": "inProgress",
                                          "started_at": _now()}

    def _on_Browser_downloadProgress(self, params: dict, session: Optional[str]) -> None:
        if (download := self.downloads.get(params["guid"])) is None:
            return
        download["state"] = params["state"]
        if params["state"] == "completed":
            if not params.get("filePath"):
                download["ended_at"] = _now()
                return
            download["path"] = params["filePath"]
            self._spawn(self._hashed(download, Path(params["filePath"])))
        elif params["state"] == "canceled":
            download["ended_at"] = _now()

    async def _hashed(self, download: dict, path: Path) -> None:
        size, sha256 = await asyncio.to_thread(_size_and_sha256, path)
        download.update(size=size, sha256=sha256, ended_at=_now())

    def _spawn(self, work: Awaitable) -> None:
        task = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Future) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.warning(f"the browser's control connection: {type(error).__name__}: {error}")


def _size_and_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return path.stat().st_size, digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

