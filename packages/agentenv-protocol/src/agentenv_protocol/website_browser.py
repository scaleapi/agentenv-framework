"""WebsiteBrowserEnvironment: a website browser as a computer app, run inside a computer or alone. A subclass says how
its build installs and starts; this class adds what agentenv owns, the same in both: its flags after the subclass's
command, the unattended-use policies, a CDP connection over the browser's debugging pipe, the settings every browser
takes, a fresh cache and the deploy's prompt answers at every launch, another window for a launch while it runs, its
skill and ``browser/v1`` on its card. It runs alone, headless, on a computer with no display, a browser computer's,
whose screen is its page; which site proxy it goes through comes from a variable agentenv sets, never from its
settings."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, ClassVar, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .agent_env_environment import agent_skill, extension
from .computer import AppDeclaration, ComputerAppEnvironment, Launch, install_step, ready_check
from .computer import app as computer_app
from .computer.errors import DeviceUnavailable, InvalidArgument

logger = logging.getLogger(__name__)

SITE_PROXY_ENV = "AGENTENV_SITE_PROXY"
BROWSER = "urn:agentenv:browser/v1"
#: The flags agentenv sets, by prefix, which a subclass's command may not carry: among them every way to open a debugging
#: port, since the browser is reached only over its pipe.
OWNED_FLAGS = ("--user-data-dir", "--remote-debugging-", "--remote-allow-origins", "--proxy-", "--no-proxy-server",
               "--headless", "--window-size", "--lang", "--force-webrtc-ip-handling-policy", "--disk-cache-dir",
               "--disable-dev-shm-usage")
#: Below this much /dev/shm, as Docker's 64 MiB default, the browser keeps its shared memory in files instead.
SHM_MIN_BYTES = 1 << 30
#: Chromium's content setting that denies a permission without a prompt.
BLOCK = 2
#: Who closed a tab: agentenv over its own connection, or someone it can't tell; or how its browser ended, on its own
#: or stopped by its server.
AGENTENV, UNKNOWN, EXIT, STOP = "agentenv", "unknown", "exit", "stop"
#: How long the active tab's check waits on each page.
ACTIVE_CHECK_TIMEOUT_S = 2.0
#: How much of a download each read hashes.
HASH_CHUNK_BYTES = 1 << 20

#: Who answered a dialog when agentenv didn't.
AGENT = "agent"
#: The targets agentenv's connection follows.
PAGES = [{"type": "page"}]
#: The CDP input a computer's page driver may send to the active tab, and how long its screenshot may take.
PAGE_INPUT = frozenset({"Input.dispatchMouseEvent", "Input.dispatchKeyEvent", "Input.insertText"})
CAPTURE_TIMEOUT_S, UNNAVIGATED_CAPTURE_TIMEOUT_S = 3.0, 1.0
#: Where files a page's file chooser may be given come from, under the session user's home.
UPLOADS = "Uploads"
#: Alone, loopback goes through the proxy too. In a computer, the apps on loopback beside it stay reachable; ``<-loopback>``
#: removes Chromium's own rule, so ``*.localhost`` names still don't.
ALONE_BYPASS = "<-loopback>"
COMPUTER_BYPASS = "<-loopback>;127.0.0.1;[::1];localhost"
#: Managed policies for unattended use: no sign-in, sync, password manager, autofill, metrics, promotions, translation,
#: background mode, component updates, default-browser check, download prompt or warning bar for its flags.
POLICIES: dict[str, Any] = {
    "BrowserSignin": 0,
    "SyncDisabled": True,
    "PasswordManagerEnabled": False,
    "AutofillSettings": [{"url_pattern": "*", "blocked_types": ["all"]}],
    "MetricsReportingEnabled": False,
    "PromotionalTabsEnabled": False,
    "TranslateEnabled": False,
    "BackgroundModeEnabled": False,
    "ComponentUpdatesEnabled": False,
    "DefaultBrowserSettingEnabled": False,
    "PromptForDownloadLocation": False,
    "CommandLineFlagSecurityWarningsEnabled": False,
    "DownloadDirectory": "${user_home}/Downloads",
}
#: Where an agent's files to upload are, and where downloads land, under the session user's home.
UPLOADS = "Uploads"
DOWNLOADS = "Downloads"
SKILL = f"""# Website browser

- Open sites by their real names, such as `https://slack.com`: the environment serves its sites under those names.
- Files to upload are in `~/{UPLOADS}`; downloads land in `~/{DOWNLOADS}`.
"""
#: The fds a Chromium on ``--remote-debugging-pipe`` reads its CDP messages from and writes its answers to.
PIPE_IN, PIPE_OUT = 3, 4
#: The largest CDP message the pipe takes, such as a screenshot's.
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


class DevToolsError(RuntimeError):
    """A CDP call the browser answered with an error."""


class DevToolsPipe:
    """A CDP connection over a Chromium's debugging pipe, each message JSON ended by a NUL. It connects anew at every
    launch; a call with no browser connected, or whose browser exits before answering, raises ConnectionError. Each
    event, a message with no ``id``, goes to every listener, and each closer hears when the browser closes the pipe."""

    #: How long a call waits for the browser's answer when it names no wait of its own.
    call_timeout_s: ClassVar[float] = 30.0

    def __init__(self) -> None:
        self._ids = itertools.count(1)
        self._waiting: dict[int, asyncio.Future] = {}
        self._writer: Optional[asyncio.WriteTransport] = None
        self._reading: Optional[asyncio.Task] = None
        self._listeners: list[Callable[[dict], None]] = []
        self._closers: list[Callable[[], None]] = []

    def listen(self, listener: Callable[[dict], None]) -> None:
        """Hand ``listener`` each event: its ``method``, ``params`` and, on a page's session, ``sessionId``."""
        self._listeners.append(listener)

    def on_close(self, closer: Callable[[], None]) -> None:
        """Call ``closer`` when the browser closes its end of the pipe, not when this end closes it."""
        self._closers.append(closer)

    async def connect(self, to_browser: int, from_browser: int) -> None:
        """Take over this end of a new pipe, ending any connection to a browser launched before."""
        self.close()
        loop = asyncio.get_running_loop()
        reader = asyncio.StreamReader(limit=MAX_MESSAGE_BYTES)
        await loop.connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), os.fdopen(from_browser, "rb", 0))
        self._writer, _ = await loop.connect_write_pipe(asyncio.Protocol, os.fdopen(to_browser, "wb", 0))
        self._reading = asyncio.ensure_future(self._read(reader))

    async def call(self, method: str, params: Optional[dict] = None, *, session_id: Optional[str] = None,
                   timeout_s: Optional[float] = None) -> dict:
        """``method``'s result, or DevToolsError with the browser's message."""
        if self._writer is None or self._writer.is_closing():
            raise ConnectionError("no browser is connected over its debugging pipe")
        message_id = next(self._ids)
        answer = asyncio.get_running_loop().create_future()
        self._waiting[message_id] = answer
        message = {"id": message_id, "method": method, "params": params or {},
                   **({"sessionId": session_id} if session_id else {})}
        self._writer.write(json.dumps(message).encode() + b"\0")
        try:
            answered = await asyncio.wait_for(answer, self.call_timeout_s if timeout_s is None else timeout_s)
        finally:
            self._waiting.pop(message_id, None)
        if "error" in answered:
            raise DevToolsError(f"{method}: {answered['error'].get('message', answered['error'])}")
        return answered.get("result", {})

    def close(self) -> None:
        if self._reading is not None:
            self._reading.cancel()
        if self._writer is not None:
            self._writer.close()
        self._fail_waiting()
        self._writer = self._reading = None

    async def _read(self, reader: asyncio.StreamReader) -> None:
        try:
            while True:
                message = json.loads((await reader.readuntil(b"\0"))[:-1])
                if "id" not in message:
                    for listener in self._listeners:
                        _heard(listener, message)
                    continue
                answer = self._waiting.get(message.get("id"))
                if answer is not None and not answer.done():
                    answer.set_result(message)
        except (asyncio.IncompleteReadError, ConnectionError):
            if self._writer is not None:
                self._writer.close()
            self._fail_waiting()
            for closer in self._closers:
                _heard(closer)

    def _fail_waiting(self) -> None:
        for answer in self._waiting.values():
            if not answer.done():
                answer.set_exception(ConnectionError("the browser closed its debugging pipe"))


def _heard(callback: Callable, *args: Any) -> None:
    try:
        callback(*args)
    except Exception:
        logger.exception("a DevToolsPipe listener failed")

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
        """A new tab at ``url``, else blank, brought to the front; its id."""
        target = (await self.devtools.call("Target.createTarget", {"url": url or "about:blank"}))["targetId"]
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
        """The tab a dialog is open on, whose page it blocks from saying; else the tab whose page has focus, else the
        one agentenv last brought to the front, else the first visible one: CDP has no event for a tab made active."""
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
            if state[1]:
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
        target = self._sessions[session]
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
        handler = getattr(self, "_on_" + message.get("method", "").replace(".", "_"), None)
        if handler is not None:
            handler(message.get("params", {}), message.get("sessionId"))

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
        try:
            await self.devtools.call("Page.enable", session_id=session)
            if self.viewport is not None:
                width, height = self.viewport
                await self.devtools.call("Emulation.setDeviceMetricsOverride", {
                    "width": width, "height": height, "deviceScaleFactor": 1, "mobile": False}, session_id=session)
                await self.devtools.call("Page.setInterceptFileChooserDialog", {"enabled": True}, session_id=session)
        finally:
            await self.devtools.call("Runtime.runIfWaitingForDebugger", session_id=session)

    def _on_Target_detachedFromTarget(self, params: dict, session: Optional[str]) -> None:
        if (target := self._sessions.pop(params["sessionId"], None)) is not None:
            self._shots.pop(target, None)
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


class Window(BaseModel):
    width: int = Field(1280, gt=0)
    height: int = Field(800, gt=0)


class BrowserSettings(BaseModel):
    """The settings every website browser takes, per deploy; a browser's own ``Settings`` adds to them."""

    #: Opened in tabs at launch, the first active.
    start_urls: list[str] = Field(default_factory=lambda: ["about:blank"])
    #: Its window when it runs alone; in a computer the screen sets it.
    window: Window = Field(default_factory=Window)
    locale: str = "en-US"
    #: The IANA time zone its pages see.
    time_zone: str = "UTC"
    #: Whether a page asking for the location, or to show notifications, is denied without a prompt or asks.
    geolocation: Literal["deny", "ask"] = "deny"
    notifications: Literal["deny", "ask"] = "deny"
    #: Who answers a page's JavaScript dialogs: the agent, or agentenv accepting or dismissing each.
    dialogs: Literal["agent", "accept", "dismiss"] = "agent"

    @field_validator("start_urls")
    @classmethod
    def _no_flags(cls, urls: list[str]) -> list[str]:
        if flags := [url for url in urls if url.startswith("-")]:
            raise ValueError(f"start_urls are URLs, not flags: {flags}")
        return urls


class WebsiteBrowserEnvironment(ComputerAppEnvironment):
    """Subclass it as any computer app, with ``@computer_app`` and ``@linux``, its launch running the browser itself,
    and set ``version``, its build, and ``policy_dir``, where that build reads managed policies. ``policies`` adds its
    own to agentenv's, and ``profile`` is its profile folder under the session user's home, holding what its install
    steps put there."""

    Settings = BrowserSettings
    version: ClassVar[str] = ""
    policy_dir: ClassVar[Path] = Path("/etc/chromium/policies/managed")
    policies: ClassVar[dict[str, Any]] = {}
    profile: ClassVar[str] = ".config/chromium"
    #: How long its ready check waits on the browser's answer over its pipe.
    devtools_timeout_s: ClassVar[float] = 5.0

    def __init__(self) -> None:
        #: Its CDP connection, over the pipe its server launches it with.
        self.devtools = DevToolsPipe()
        self._following: Optional[asyncio.Future] = None
        super().__init__()
        for folder in (UPLOADS, DOWNLOADS):
            (Path.home() / folder).mkdir(exist_ok=True)
        #: agentenv's control of it over that connection, and the record it keeps.
        self.control = BrowserControl(self.devtools, self.settings.dialogs,
                                      (self.settings.window.width, self.settings.window.height) if _alone()
                                      else None)
        self.devtools.on_close(self._pipe_closed)

    @classmethod
    def declaration(cls) -> AppDeclaration:
        """What the class declares, a website browser; ValueError for a launch whose command carries a flag agentenv
        owns."""
        for declared in cls._installs.values():
            if declared.launch is not None:
                _refuse_owned(cls.__name__, declared.launch)
        return super().declaration().model_copy(update={"browser": True})

    def profile_dir(self) -> Path:
        return Path.home() / self.profile

    def cache_dir(self) -> Path:
        """Its HTTP cache, kept out of its profile, which holds only its data."""
        return Path.home() / ".cache" / Path(self.profile).name

    def agentenv_flags(self) -> list[str]:
        """The flags agentenv owns. A browser alone goes out only through its site proxy, so it won't launch without
        one; in a computer it goes through one when the computer has one, loopback aside."""
        settings, alone, proxy = self.settings, _alone(), os.environ.get(SITE_PROXY_ENV)
        flags = [f"--user-data-dir={self.profile_dir()}", f"--disk-cache-dir={self.cache_dir()}",
                 "--remote-debugging-pipe", "--no-first-run", "--no-default-browser-check", f"--lang={settings.locale}"]
        if (shm := _shm_bytes()) < SHM_MIN_BYTES:
            logger.info(f"/dev/shm holds {shm // 2**20} MiB, under {SHM_MIN_BYTES // 2**20} MiB: the browser keeps its "
                        "shared memory in files")
            flags.append("--disable-dev-shm-usage")
        if proxy:
            flags += [f"--proxy-server={proxy}", f"--proxy-bypass-list={ALONE_BYPASS if alone else COMPUTER_BYPASS}",
                      "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"]
        elif alone:
            raise ValueError(f"a website browser alone goes out only through its site proxy, and {SITE_PROXY_ENV} "
                             "names none")
        if alone:
            flags.append(f"--window-size={settings.window.width},{settings.window.height}")
        if alone:
            flags.append("--headless=new")
        return flags

    def _launch_for(self, launch: Launch) -> Launch:
        """The subclass's launch with agentenv's flags and the start tabs after its command, and the time zone in its
        environment; alone, with no window to match or wait for, its ``ready`` keeping its port and command, and
        none when that leaves it nothing, so its ready checks say when it's ready."""
        _refuse_owned(type(self).__name__, launch)
        update: dict[str, Any] = {"command": [*launch.command, *self.agentenv_flags(), *self._command_line_tabs()],
                                  "env": {**(launch.env or {}), "TZ": self.settings.time_zone}}
        if _alone():
            ready = launch.ready
            waits = ready is not None and (ready.port is not None or bool(ready.command))
            update.update(window=None, ready=ready.model_copy(update={"window": False}) if waits else None)
        return launch.model_copy(update=update)

    def _on_desktop(self) -> bool:
        return not _alone()

    def _command_line_tabs(self) -> list[str]:
        """The start tabs its command line opens: headless Chromium takes one, and the rest open over its pipe."""
        return self.settings.start_urls[:1] if _alone() else self.settings.start_urls

    async def _launch_fds(self) -> dict[int, int]:
        """Before each launch, an empty cache and the deploy's prompt answers in its profile; and a new debugging pipe:
        the browser's ends as its fds 3 and 4, this server's ends its CDP connection, which follows the browser and
        then opens the start tabs its command line can't."""
        shutil.rmtree(self.cache_dir(), ignore_errors=True)
        self._write_prompt_answers()
        to_browser_read, to_browser_write = os.pipe()
        from_browser_read, from_browser_write = os.pipe()
        await self.devtools.connect(to_browser_write, from_browser_read)
        self._following = asyncio.ensure_future(self._follow(self.settings.start_urls[len(self._command_line_tabs()):]))
        self._following.add_done_callback(lambda done: done.cancelled() or done.exception())
        return {PIPE_IN: to_browser_read, PIPE_OUT: from_browser_write}

    def _pipe_closed(self) -> None:
        """The browser closed its pipe: its tabs closed by its exit when it ended on its own, else by its server's
        stop."""
        self.control.ended(EXIT if self._computer_app.exited() else STOP)

    async def _follow(self, urls: list[str]) -> None:
        """Its control attached, then each of ``urls`` in a tab behind the active one, in order."""
        await self.control.attach()
        for url in urls:
            await self.devtools.call("Target.createTarget", {"url": url, "background": True})

    def _write_prompt_answers(self) -> None:
        """Its profile's preferences, which it reads as it starts, denying location and notification prompts or
        leaving them to ask, as the deploy's settings say."""
        preferences = self.profile_dir() / "Default" / "Preferences"
        try:
            current = json.loads(preferences.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            current = {}
        current = current if isinstance(current, dict) else {}
        defaults = current.setdefault("profile", {}).setdefault("default_content_setting_values", {})
        for name in ("geolocation", "notifications"):
            if getattr(self.settings, name) == "deny":
                defaults[name] = BLOCK
            else:
                defaults.pop(name, None)
        preferences.parent.mkdir(parents=True, exist_ok=True)
        preferences.write_text(json.dumps(current), encoding="utf-8")

    def _relaunch_on_start(self) -> bool:
        return True

    def _launch_running(self) -> Callable[[], Awaitable[None]]:
        return self._open_window

    async def _open_window(self) -> None:
        """Another window, which a launch of the browser while it runs opens over its pipe: a second copy would only
        hand its window to the first and exit, taking its new pipe with it."""
        await self.devtools.call("Target.createTarget", {"url": "about:blank", "newWindow": True})

    @install_step(scope="system")
    def write_policies(self) -> None:
        """agentenv's unattended-use policies, with the subclass's over them, where its build reads them."""
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        (self.policy_dir / "agentenv.json").write_text(json.dumps({**POLICIES, **self.policies}, indent=1) + "\n",
                                                       encoding="utf-8")

    @ready_check
    async def devtools_answers(self) -> bool:
        """It answers over its pipe, its control follows it and it has opened its start tabs."""
        try:
            await self.devtools.call("Browser.getVersion", timeout_s=self.devtools_timeout_s)
        except (ConnectionError, DevToolsError, asyncio.TimeoutError):
            return False
        following = self._following
        return following is not None and following.done() and not following.cancelled() and \
            following.exception() is None

    @extension(BROWSER, op="record", method="GET", description=(
        "A website browser agentenv runs: its build, the clock its pages see and, alone, its window. record returns "
        "what agentenv's own connection to it saw: its open tabs, each with whether it's active, the tabs closed and "
        "who closed them, its downloads with their size and sha256, its JavaScript dialogs and who answered each, and "
        "each launch, stop and exit of it since the computer booted, with its cause. page/viewport, page/capture and "
        "page/input are its active tab's page as a computer's page driver sees it: its size in pixels, which headless "
        "is its window's, a PNG of it, and CDP input calls sent to it in order. navigate and history open a URL in "
        "the active tab, or go back, forward or reload; tabs/list, tabs/open, tabs/activate and tabs/close manage "
        "its tabs by their ids; dialog/answer answers the dialog open on the active tab; files/choose gives the file "
        "chooser open on it files in ~/Uploads, headless; and page/state is the active tab's id, URL and title and "
        "what is open on it."))
    async def browser_record(self) -> dict:
        return {**await self.control.record(), "launches": self._computer_app.launches()}

    @extension(BROWSER, op="page/viewport", method="GET")
    async def browser_page_viewport(self) -> dict:
        window = self.settings.window
        return {"width": window.width, "height": window.height}

    @extension(BROWSER, op="page/capture", method="GET")
    async def browser_page_capture(self) -> dict:
        return await self.control.capture_page()

    @extension(BROWSER, op="page/input")
    async def browser_page_input(self, events: list[dict]) -> dict:
        await self.control.dispatch(events)
        return {}

    @extension(BROWSER, op="page/state", method="GET")
    async def browser_page_state(self) -> dict:
        return await self.control.state()

    @extension(BROWSER, op="navigate")
    async def browser_navigate(self, url: str) -> dict:
        await self.control.navigate(url)
        return {}

    @extension(BROWSER, op="history")
    async def browser_history(self, direction: str) -> dict:
        await self.control.history(direction)
        return {}

    @extension(BROWSER, op="tabs/list", method="GET")
    async def browser_tabs_list(self) -> dict:
        return {"tabs": await self.control.tab_list()}

    @extension(BROWSER, op="tabs/open")
    async def browser_tabs_open(self, url: Optional[str] = None) -> dict:
        return {"tab": await self.control.open_tab(url)}

    @extension(BROWSER, op="tabs/activate")
    async def browser_tabs_activate(self, tab: str) -> dict:
        await self.control.activate(tab)
        return {}

    @extension(BROWSER, op="tabs/close")
    async def browser_tabs_close(self, tab: str) -> dict:
        await self.control.close(tab)
        return {}

    @extension(BROWSER, op="dialog/answer")
    async def browser_dialog_answer(self, accept: bool, text: Optional[str] = None) -> dict:
        await self.control.answer_dialog(accept, text)
        return {}

    @extension(BROWSER, op="files/choose")
    async def browser_files_choose(self, paths: list[str]) -> dict:
        await self.control.choose_files(paths)
        return {}

    def extension_params(self, uri: str) -> dict:
        if uri == BROWSER:
            params: dict[str, Any] = {"chromium": self.version, "clock": "real"}
            if _alone():
                params["window"] = self.settings.window.model_dump()
            return params
        return super().extension_params(uri)

    @agent_skill
    def browser_skill(self) -> tuple[str, str]:
        return "How to use this website browser.", SKILL


def _alone() -> bool:
    """Whether it runs on a computer with no display, a browser computer's, alone and headless."""
    return computer_app.detect().display == "none"


def _shm_bytes(path: str = "/dev/shm") -> int:
    try:
        stats = os.statvfs(path)
    except OSError:
        return 0
    return stats.f_frsize * stats.f_blocks


def _refuse_owned(name: str, launch: Launch) -> None:
    if owned := [arg for arg in launch.command if arg.startswith(OWNED_FLAGS)]:
        raise ValueError(f"{name}'s launch sets {', '.join(owned)}, which agentenv sets for every website browser")
