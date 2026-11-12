"""agentenv's control of a browser over its pipe, and the record it keeps."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import urlsplit

from .log import ACTOR, KEY_FIELDS, TYPING_METHODS, BrowserLog, Causes, caused, log_time, redact_url
from ..computer.errors import DeviceUnavailable, InvalidArgument
from .devtools import DevToolsError, DevToolsPipe
from .shapes import (ClosedTab, Cookie, CookieToSet, Dialog, Download, OpenTab, OriginStorage, StartTab, StorageItem,
                     origin_of)

logger = logging.getLogger(__name__)

#: Who closed a tab: agentenv over its own connection, or someone it can't tell; or how its browser ended, on its own
#: or stopped by its server.
AGENTENV, UNKNOWN, EXIT, STOP = "agentenv", "unknown", "exit", "stop"
#: How long the active tab's check waits on each page.
ACTIVE_CHECK_TIMEOUT_S = 2.0
#: How long a tab opened alone may take to be attached and held to its window's size, which is also how long a page
#: of agentenv's own may take to reach an origin, and how often it's asked meanwhile.
PIN_TIMEOUT_S = 10.0
ARRIVAL_POLL_S = 0.05
#: How long a start tab on a site the env doesn't serve may take to load before it's left loading.
LIVE_START_TAB_TIMEOUT_S = 30.0
#: How a start tab settled.
LOADED, FAILED, TIMED_OUT = "loaded", "failed", "timed_out"
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
#: The document agentenv serves a page of its own that it sends to an origin, so the site gets no request.
SERVED_PAGE = base64.b64encode(b"<!doctype html>").decode()
#: Marks the document a page of agentenv's own leaves, and reads the origin of the one it goes to once it's there, as
#: the browser names it.
LEAVING, ARRIVED = "window.__agentenvLeft = true", "window.__agentenvLeft ? '' : location.origin"
#: Hostnames a secure cookie is sent to over http, as browsers treat them.
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")
#: Whether the focused element is a password field, through shadow roots and frames of the page's own origin; a frame
#: of another origin can't be looked into.
PASSWORD_FOCUSED = """(() => {
  let element = document.activeElement;
  for (;;) {
    if (element && element.shadowRoot && element.shadowRoot.activeElement) element = element.shadowRoot.activeElement;
    else if (element && (element.tagName === 'IFRAME' || element.tagName === 'FRAME')) {
      try { element = element.contentDocument.activeElement; } catch (e) { return false; }
    } else break;
  }
  return !!element && element.tagName === 'INPUT' && (element.type || '').toLowerCase() === 'password';
})()"""
#: How long the password-field check waits on a page.
PASSWORD_CHECK_TIMEOUT_S = 2.0


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
        #: The file chooser each page's session opened that waits, alone, and the tab agentenv last brought to the
        #: front.
        self._choosers: dict[str, dict] = {}
        self._front: Optional[str] = None
        #: Each tab's latest capture, which shows while a dialog keeps its page from painting; and, by session, what
        #: input sent there waits on to learn a dialog opened.
        self._shots: dict[str, str] = {}
        self._dialog_opened: dict[str, asyncio.Event] = {}
        self._pinned: dict[str, asyncio.Event] = {}
        self._tasks: set[asyncio.Future] = set()
        #: Each start tab loading, by its session.
        self._loading: dict[str, _StartLoad] = {}
        #: Every http and https origin a page or frame it follows loaded since this server started, and the sessions
        #: of agentenv's own pages whose requests it answers itself.
        self.origins: set[str] = set()
        self._serving: set[str] = set()
        #: The browser's log, where its tab events go, and the commands in flight that cause them.
        self.log: Optional[BrowserLog] = None
        self.causes = Causes()
        self._launching = False
        devtools.listen(self._event)

    async def attach(self) -> None:
        """Follow a browser just launched: any tabs the one before left open closed by its server's stop, every page
        this one has or opens attached, and its downloads reported, which go where its own policies say, as Chrome
        downloads unattended: into ``~/Downloads``, a name downloaded again kept beside the first."""
        self.ended(STOP)
        self._launching = True
        try:
            await self.devtools.call("Target.setDiscoverTargets", {"discover": True, "filter": PAGES})
        finally:
            self._launching = False
        await self.devtools.call("Target.setAutoAttach", {"autoAttach": True,
                                                          "waitForDebuggerOnStart": self.viewport is not None,
                                                          "flatten": True, "filter": PAGES})
        await self.devtools.call("Browser.setDownloadBehavior", {"behavior": "default", "eventsEnabled": True})

    def ended(self, how: str) -> None:
        """The browser is gone, ``how`` it ended: its open tabs are closed by that, and its sessions with them."""
        now = _now()
        self.closed_tabs += [{**tab, "closed_at": now, "closed_by": how} for tab in self.tabs.values()]
        if self.log is not None:
            for tab in self.tabs.values():
                self.log.note("tab", {"event": "closed", "id": tab["id"], "url": redact_url(tab["url"]), "by": how})
        self.tabs.clear()
        self._sessions.clear()
        self._open_dialogs.clear()
        self._choosers.clear()
        self._shots.clear()
        self._dialog_opened.clear()
        self._pinned.clear()
        for load in self._loading.values():
            load.error = load.error or "the browser ended"
            load.settled.set()

    async def close_tab(self, target: str) -> None:
        """Close the tab ``target``, which the record says agentenv closed, and the log says who asked."""
        self._closing.add(target)
        self.causes.closing(target, _cause())
        await self.devtools.call("Target.closeTarget", {"targetId": target})

    async def create_target(self, params: dict) -> str:
        """A new tab, ``Target.createTarget`` with ``params``, which the log says who asked for; its id."""
        landed = asyncio.get_running_loop().create_future()
        self.causes.creating(landed, _cause())
        try:
            target = (await self.devtools.call("Target.createTarget", params))["targetId"]
        except BaseException:
            landed.cancel()
            raise
        landed.set_result(target)
        return target

    async def _navigate(self, target: str, method: str, params: dict, session: str) -> dict:
        """``Page.navigate`` of ``target`` on ``session``, which the log says who asked for; its answer."""
        landed = asyncio.get_running_loop().create_future()
        self.causes.navigating(target, landed, _cause())
        try:
            answer = await self.devtools.call(method, params, session_id=session)
        except BaseException:
            landed.cancel()
            raise
        if answer.get("errorText"):
            landed.cancel()
        else:
            landed.set_result(answer.get("loaderId"))
        return answer

    def url_of(self, target: Optional[str]) -> Optional[str]:
        """The URL of the open tab ``target`` as the log keeps it; None for one it doesn't know."""
        tab = self.tabs.get(target) if target else None
        return redact_url(tab["url"]) if tab is not None else None

    async def password_focused(self, target: str) -> bool:
        """Whether a password field has focus in the tab ``target``, so what is typed there isn't logged; True when it
        can't be told, as for a tab it has no session on."""
        session = next((s for s, t in self._sessions.items() if t == target), None)
        if session is None:
            return True
        try:
            answer = await self.devtools.call("Runtime.evaluate", {"expression": PASSWORD_FOCUSED,
                                                                   "returnByValue": True, "silent": True},
                                              session_id=session, timeout_s=PASSWORD_CHECK_TIMEOUT_S)
            return answer["result"]["value"] is not False
        except (ConnectionError, DevToolsError, asyncio.TimeoutError, KeyError, TypeError):
            return True

    def _acted_on(self, target: Optional[str]) -> None:
        """The tab the op under way acts on or reads, for its entry; an acting op's becomes that tab's latest."""
        actor = ACTOR.get()
        if actor is None or target is None:
            return
        actor["tab"] = target
        if actor.get("acts") and self.log is not None and self.log.last_acting.get(target, 0) < actor["seq"]:
            self.log.last_acting[target] = actor["seq"]

    async def tab_list(self) -> list[dict]:
        """The open tabs in opened order, each with its URL and title now and whether it's active."""
        await self._refresh()
        active = await self.active_tab()
        return [OpenTab(**tab, active=tab["id"] == active).model_dump() for tab in self.tabs.values()]

    async def open_tab(self, url: Optional[str] = None, background: bool = False) -> str:
        """A new tab at ``url``, else blank, brought to the front, or with ``background`` behind the active tab, which
        stays active, brought back only if the new tab took its place; its id. Alone, it opens blank and goes to
        ``url`` once it's held to the window's size, so its page loads at that size."""
        active = await self.active_tab() if background else None
        held = self.viewport is not None and url
        start = "about:blank" if held else url or "about:blank"
        target = await self.create_target({"url": start, "background": background})
        self._acted_on(target)
        if held:
            await asyncio.wait_for(self._pinned.setdefault(target, asyncio.Event()).wait(), PIN_TIMEOUT_S)
            session = next(s for s, t in self._sessions.items() if t == target)
            await self._navigate(target, "Page.navigate", {"url": url}, session)
        if not background:
            await self._bring_to_front(target)
        elif active is not None and active in self.tabs and await self.active_tab() == target:
            await self._bring_to_front(active)
        return target

    async def load_start_tabs(self, urls: list[str], site_of: Callable[[str], Optional[str]], own_timeout_s: float,
                              live_timeout_s: float = LIVE_START_TAB_TIMEOUT_S) -> list[dict]:
        """Each of ``urls`` in a start tab, in order, the first then active: in the tabs open now, in opened order, and
        in tabs it opens blank after them; each held to the window's size, alone, before it goes to its URL, and
        all loading at once. A tab settles once it has loaded, once it has failed, which on one of the env's own sites,
        as ``site_of`` names them, an error status is too, or once it has been waited on ``own_timeout_s`` on such a
        site, else ``live_timeout_s``. Each tab's outcome, in order, kept though the browser has gone and the first
        can't come to the front. A page already loaded doesn't read cookies set after it, so a tab a load signed in
        shows it only once it loads again."""
        if not urls:
            return []
        targets = list(self.tabs)[:len(urls)] or [await self._first_tab()]
        while len(targets) < len(urls):
            targets.append(await self.create_target({"url": "about:blank", "background": True}))
        for target in targets:
            await asyncio.wait_for(self._pinned.setdefault(target, asyncio.Event()).wait(), PIN_TIMEOUT_S)
        loading = []
        for target, url in zip(targets, urls):
            site = site_of(url)
            loading.append(await self._start_load(target, url, site))
        outcomes = await asyncio.gather(*(self._settled(target, load, own_timeout_s if load.site is not None
                                                        else live_timeout_s) for target, load in zip(targets, loading)))
        with contextlib.suppress(ConnectionError, DevToolsError):
            await self._bring_to_front(targets[0])
        return outcomes

    async def _first_tab(self) -> str:
        """The tab a browser just launched opened, once it has one."""
        deadline = asyncio.get_running_loop().time() + PIN_TIMEOUT_S
        while not self.tabs:
            if asyncio.get_running_loop().time() > deadline:
                raise DeviceUnavailable(f"the browser opened no tab within {PIN_TIMEOUT_S:g} s")
            await asyncio.sleep(ARRIVAL_POLL_S)
        return next(iter(self.tabs))

    async def _start_load(self, target: str, url: str, site: Optional[str]) -> _StartLoad:
        """The tab ``target`` sent to ``url``, its navigation followed by what the browser says of it. The browser
        answers a navigation once it commits, which a page that never answers never does, so its answer is awaited
        apart, for as long as the tab is."""
        session = next((s for s, t in self._sessions.items() if t == target), None)
        if session is None:
            raise DeviceUnavailable(f"the browser's tab {target} has no session to load a start tab in")
        load = self._loading[session] = _StartLoad(session, url, site, asyncio.get_running_loop().time())
        try:
            await self.devtools.call("Network.enable", session_id=session)
        except (ConnectionError, DevToolsError, asyncio.TimeoutError) as e:
            load.error = f"{type(e).__name__}: {e}"
            load.check()
            return load
        landed = asyncio.get_running_loop().create_future()
        self.causes.navigating(target, landed, _cause())
        load.navigating = asyncio.ensure_future(self.devtools.request("Page.navigate", {"url": url},
                                                                      session_id=session))
        load.navigating.add_done_callback(load.answered)
        load.navigating.add_done_callback(lambda done: _landed_at(landed, done))
        return load

    async def _settled(self, target: str, load: _StartLoad, timeout_s: float) -> dict:
        """``load``'s outcome once it settles, or once it has been waited on ``timeout_s``."""
        try:
            await asyncio.wait_for(load.settled.wait(), timeout_s)
        except asyncio.TimeoutError:
            pass
        finally:
            if load.navigating is not None:
                load.navigating.cancel()
            self._loading.pop(load.session, None)
            with contextlib.suppress(ConnectionError, DevToolsError, asyncio.TimeoutError):
                await self.devtools.call("Network.disable", session_id=load.session)
        document = load.current() or load.loader or ""
        status, final_url = load.responses.get(document, (None, None))
        error = load.error or load.failures.get(load.loader or "") or load.failures.get(document)
        outcome = FAILED if error or (load.settled.is_set() and status is not None and status >= 400) else \
            LOADED if load.settled.is_set() else TIMED_OUT
        if final_url is None and outcome == LOADED:
            await self._refresh(target)
            final_url = self.tabs.get(target, {}).get("url")
        return StartTab(tab=target, url=load.url, final_url=final_url, status=status, error=error,
                        load_ms=round((asyncio.get_running_loop().time() - load.started) * 1000), site=load.site,
                        outcome=outcome).model_dump()

    async def replace_tabs(self, tabs: list) -> None:
        """``tabs``, each with its ``url`` and whether it's ``active``, opened in order in place of the tabs open now,
        the active one, else the first, in front. The new ones open before the old close, so the window stays."""
        before = list(self.tabs)
        opened = [await self.open_tab(tab.url) for tab in tabs]
        for target in before:
            await self.close_tab(target)
        if opened:
            await self._bring_to_front(next((target for target, tab in zip(opened, tabs) if tab.active), opened[0]))

    def forget(self) -> None:
        """A record that starts now, as a reset leaves the browser: no tabs closed, downloads or dialogs before it, and
        no origin seen."""
        self.since = _now()
        self.closed_tabs, self.dialog_log = [], []
        self.downloads.clear()
        self.origins.clear()

    async def activate(self, target: str) -> None:
        """The open tab ``target`` brought to the front, the active one."""
        self._open(target)
        self._acted_on(target)
        await self._bring_to_front(target)

    async def close(self, target: str) -> None:
        """The open tab ``target`` closed, which the record says agentenv closed."""
        self._open(target)
        self._acted_on(target)
        await self.close_tab(target)

    async def navigate(self, url: str) -> None:
        """The active tab opened at ``url``; InvalidArgument, with the browser's reason, when it can't."""
        session = await self.active_session()
        self._acted_on(target := self._sessions[session])
        answer = await self._navigate(target, "Page.navigate", {"url": url}, session)
        if error := answer.get("errorText"):
            raise InvalidArgument(f"the browser couldn't open {url}: {error}")

    async def history(self, direction: str) -> None:
        """The active tab's page reloaded, or the one before or after it in its history opened."""
        session = await self.active_session()
        self._acted_on(target := self._sessions[session])
        if direction == "reload":
            await self._moves(target, "Page.reload", {}, session)
            return
        if direction not in ("back", "forward"):
            raise InvalidArgument(f"direction is back, forward or reload, not {direction!r}")
        history = await self.devtools.call("Page.getNavigationHistory", session_id=session)
        index = history["currentIndex"] + (-1 if direction == "back" else 1)
        if not 0 <= index < len(history["entries"]):
            raise InvalidArgument(f"the active tab has no page to go {direction} to")
        await self._moves(target, "Page.navigateToHistoryEntry", {"entryId": history["entries"][index]["id"]},
                          session)

    async def _moves(self, target: str, method: str, params: dict, session: str) -> None:
        """A reload or move in history of ``target``, whose next navigation the log says the op under way caused,
        unless it fails."""
        cause = _cause()
        self.causes.navigates_next(target, cause)
        try:
            await self.devtools.call(method, params, session_id=session)
        except BaseException:
            self.causes.navigated_nowhere(target, cause)
            raise

    async def answer_dialog(self, accept: bool, text: Optional[str] = None) -> None:
        """The dialog open on the active tab answered, as the agent's answer; InvalidArgument with none open."""
        active = await self.active_tab()
        self._acted_on(active)
        session = next((s for s, dialog in self._open_dialogs.items() if dialog["tab"] == active), None)
        if session is None:
            raise InvalidArgument("no dialog is open on the page")
        await self.devtools.call("Page.handleJavaScriptDialog", {
            "accept": accept, **({"promptText": text} if text is not None else {})}, session_id=session)

    async def choose_files(self, paths: list[str]) -> None:
        """The file chooser open on the active tab given ``paths``, files in the session user's ``~/Uploads``;
        InvalidArgument for one outside it or missing, for several where it takes one, or with no chooser open."""
        active = await self.active_tab()
        self._acted_on(active)
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
        self._acted_on(active)
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
        return {"since": self.since, "tabs": await self.tab_list(),
                "closed_tabs": [ClosedTab(**tab).model_dump() for tab in self.closed_tabs],
                "downloads": self.download_list(), "dialogs": [Dialog(**dialog).model_dump() for dialog in self.dialog_log]}

    async def cookies(self, urls: Optional[list[str]] = None) -> list[dict]:
        """The default context's cookies, or those a request to any of ``urls`` would send, as Playwright picks them."""
        cookies = [Cookie(**{field: raw[field] for field in Cookie.model_fields if field in raw})
                   for raw in (await self.devtools.call("Storage.getCookies"))["cookies"]]
        return [cookie.model_dump(exclude_none=True) for cookie in cookies
                if not urls or any(_sent(cookie, url) for url in urls)]

    async def set_cookies(self, cookies: list[CookieToSet]) -> None:
        """``cookies`` set in the default context at once, a ``max_age`` turned into an expiry on this machine's clock,
        which the browser's cookies are judged by; InvalidArgument when the browser refuses one, which sets none."""
        try:
            await self.devtools.call("Storage.setCookies", {"cookies": [_cdp_cookie(cookie, time.time())
                                                                        for cookie in cookies]})
        except DevToolsError as e:
            raise InvalidArgument(f"the browser refused the cookies, so it set none of them: {e}")

    async def delete_cookies(self, name: Optional[str] = None, domain: Optional[str] = None,
                             path: Optional[str] = None) -> int:
        """Every cookie of the default context with all of ``name``, ``domain`` and ``path`` given, every cookie with
        none given, deleted; how many."""
        gone = [cookie for cookie in await self.cookies() if (name is None or cookie["name"] == name)
                and (domain is None or cookie["domain"] == domain) and (path is None or cookie["path"] == path)]
        if gone:
            async with self.hidden_page() as session:
                for cookie in gone:
                    await self.devtools.call("Network.deleteCookies", {
                        "name": cookie["name"], "domain": cookie["domain"], "path": cookie["path"],
                        **({"partitionKey": cookie["partitionKey"]} if cookie.get("partitionKey") else {})},
                        session_id=session)
        return len(gone)

    async def local_storage(self, origins: Optional[list[str]] = None) -> list[dict]:
        """The local storage of each of ``origins``, else of every origin it has seen, as ``storageState`` has it."""
        wanted = _origins(origins) if origins is not None else sorted(self.origins)
        found = []
        async with self._at_origins() as visit:
            for asked in wanted:
                session, origin = await visit(asked)
                entries = (await self.devtools.call("DOMStorage.getDOMStorageItems", {"storageId": _storage(origin)},
                                                    session_id=session))["entries"]
                found.append(OriginStorage(origin=origin, localStorage=[StorageItem(name=name, value=value)
                                                                        for name, value in entries]).model_dump())
        return found

    async def set_local_storage(self, origins: list[OriginStorage]) -> None:
        """Each origin's items set in its local storage, beside those it has."""
        async with self._at_origins() as visit:
            for origin in origins:
                session, named = await visit(origin.origin)
                for item in origin.localStorage:
                    await self.devtools.call("DOMStorage.setDOMStorageItem", {
                        "storageId": _storage(named), "key": item.name, "value": item.value}, session_id=session)

    async def clear_local_storage(self, origins: Optional[list[str]] = None) -> list[str]:
        """The local storage of each of ``origins``, else of every origin it has seen, cleared; those origins."""
        wanted, cleared = _origins(origins) if origins is not None else sorted(self.origins), []
        async with self._at_origins() as visit:
            for asked in wanted:
                session, origin = await visit(asked)
                await self.devtools.call("DOMStorage.clear", {"storageId": _storage(origin)}, session_id=session)
                cleared.append(origin)
        return cleared

    def download_list(self) -> list[dict]:
        return [Download(**download).model_dump() for download in self.downloads.values()]

    def download(self, guid: str) -> dict:
        """The download ``guid``; InvalidArgument for one it hasn't seen."""
        if guid not in self.downloads:
            raise InvalidArgument(f"no download {guid} has begun; list the downloads for their guids")
        return Download(**self.downloads[guid]).model_dump()

    @contextlib.asynccontextmanager
    async def _at_origins(self) -> AsyncIterator[Callable[[str], Awaitable[tuple[str, str]]]]:
        """A page of agentenv's own, ``visit``, that goes to an origin and returns its session and the origin as the
        browser names it, once it's that origin's page. agentenv serves it the document, and a service worker of the
        origin is bypassed, so the site gets no request and runs no script: CDP reaches an origin's local storage only
        through a page of that origin."""
        async with self.hidden_page() as session:
            self._serving.add(session)
            try:
                await self.devtools.call("Network.enable", session_id=session)
                await self.devtools.call("Network.setBypassServiceWorker", {"bypass": True}, session_id=session)
                await self.devtools.call("Fetch.enable", {"patterns": [{"urlPattern": "*"}]}, session_id=session)
                await self.devtools.call("DOMStorage.enable", session_id=session)

                async def visit(origin: str) -> tuple[str, str]:
                    await self.devtools.call("Runtime.evaluate", {"expression": LEAVING}, session_id=session)
                    answer = await self.devtools.call("Page.navigate", {"url": f"{origin}/"}, session_id=session)
                    if error := answer.get("errorText"):
                        raise InvalidArgument(f"the browser couldn't open a page of {origin}: {error}")
                    deadline = asyncio.get_running_loop().time() + PIN_TIMEOUT_S
                    while not (named := (await self.devtools.call("Runtime.evaluate", {
                            "expression": ARRIVED, "returnByValue": True}, session_id=session))["result"].get("value")):
                        if asyncio.get_running_loop().time() > deadline:
                            raise DeviceUnavailable(f"a page of {origin} didn't open within {PIN_TIMEOUT_S:g} s")
                        await asyncio.sleep(ARRIVAL_POLL_S)
                    self.origins.add(named)
                    return session, named

                yield visit
            finally:
                self._serving.discard(session)

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
        self._acted_on(target)
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
        self._acted_on(target := self._sessions.get(session))
        actor = ACTOR.get()
        opened = self._dialog_opened.setdefault(session, asyncio.Event())
        for event in events:
            if actor is not None and target is not None and actor["password"] is not True and _types(event):
                actor["password"] = await self.password_focused(target)
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
        if handler is not None and (session is None or session in self._sessions or session in self._serving):
            handler(message.get("params", {}), session)

    def _on_Target_targetCreated(self, params: dict, session: Optional[str]) -> None:
        info = params["targetInfo"]
        if info.get("type") == "page" and info["targetId"] not in self._hidden and info["targetId"] not in self.tabs:
            self.tabs[info["targetId"]] = {"id": info["targetId"], "url": info.get("url", ""),
                                           "title": info.get("title", ""), "opened_at": _now(),
                                           "opener": info.get("openerId")}
            self._saw(info.get("url", ""))
            if self.log is not None:
                self._spawn(self._log_opened(self.log.reserve(), log_time(), info["targetId"], info.get("url", ""),
                                             info.get("openerId"), self._launching))

    async def _log_opened(self, seq: int, at: str, target: str, url: str, opener: Optional[str],
                          launching: bool) -> None:
        """The tab ``target`` opened, in the log as ``seq``, once the commands in flight that may have opened it have
        answered: by the command that did, by agentenv as the browser launched, else by its page, its opener's."""
        cause = None
        try:
            cause = {"via": "agentenv"} if launching else await self.causes.opened(target)
        finally:
            self.log.write(seq, {"tab": {"event": "opened", "id": target, "url": redact_url(url),
                                         **({"opener": opener} if opener else {}),
                                         **caused(cause, self.log, opener or target)}}, at=at)

    async def _log_navigated(self, seq: int, at: str, target: str, url: str, loader: Optional[str] | bool) -> None:
        """The tab ``target`` navigated to ``url``, in the log as ``seq``, once its navigations in flight have
        answered: by the command whose loader it is, else by its page; with ``loader`` False, within its
        document."""
        cause = None
        try:
            cause = await (self.causes.navigated(target, loader) if loader is not False else
                           self.causes.navigated_within(target))
        finally:
            self.log.write(seq, {"tab": {"event": "navigated", "id": target, "url": redact_url(url),
                                         **({"within_document": True} if loader is False else {}),
                                         **caused(cause, self.log, target)}}, at=at)

    def _on_Target_targetInfoChanged(self, params: dict, session: Optional[str]) -> None:
        info = params["targetInfo"]
        if (tab := self.tabs.get(info["targetId"])) is not None:
            tab.update(url=info.get("url", tab["url"]), title=info.get("title", tab["title"]))
            self._saw(tab["url"])

    def _on_Page_frameNavigated(self, params: dict, session: Optional[str]) -> None:
        if session in self._sessions:
            self._saw(params.get("frame", {}).get("url", ""))
        frame = params.get("frame", {})
        if self.log is not None and not frame.get("parentId") and (target := self._sessions.get(session)) in self.tabs:
            self._spawn(self._log_navigated(self.log.reserve(), log_time(), target, frame.get("url", ""),
                                            frame.get("loaderId")))
        if (load := self._loading.get(session)) is not None and not frame.get("parentId") and frame.get("loaderId"):
            load.commits.append(frame["loaderId"])
            load.check()

    def _on_Page_navigatedWithinDocument(self, params: dict, session: Optional[str]) -> None:
        """A tab's page moving to another URL of its document, a fragment or its history API: a command's, when one
        asked for it, else its page's."""
        target = self._sessions.get(session)
        if self.log is not None and target in self.tabs and params.get("frameId") == target:
            self._spawn(self._log_navigated(self.log.reserve(), log_time(), target, params.get("url", ""), False))

    def _on_Network_requestWillBeSent(self, params: dict, session: Optional[str]) -> None:
        if (load := self._loading.get(session)) is not None and params.get("type") == "Document":
            load.requests[params["requestId"]] = params.get("loaderId", "")

    def _on_Network_responseReceived(self, params: dict, session: Optional[str]) -> None:
        if (load := self._loading.get(session)) is not None and params.get("type") == "Document" and \
                params.get("requestId") in load.requests:
            response = params.get("response", {})
            load.responses[load.requests[params["requestId"]]] = (response.get("status"), response.get("url"))
            load.check()

    def _on_Network_loadingFailed(self, params: dict, session: Optional[str]) -> None:
        if (load := self._loading.get(session)) is not None and (loader := load.requests.get(params.get("requestId"))):
            load.failures[loader] = params.get("errorText") or "the load failed"
            load.check()

    def _on_Fetch_requestPaused(self, params: dict, session: Optional[str]) -> None:
        if session in self._serving:
            self._spawn(self.devtools.call("Fetch.fulfillRequest", {
                "requestId": params["requestId"], "responseCode": 200,
                "responseHeaders": [{"name": "Content-Type", "value": "text/html"}], "body": SERVED_PAGE},
                session_id=session))

    def _saw(self, url: str) -> None:
        if (origin := origin_of(url)) is not None:
            self.origins.add(origin)

    def _on_Target_targetDestroyed(self, params: dict, session: Optional[str]) -> None:
        target = params["targetId"]
        if (tab := self.tabs.pop(target, None)) is not None:
            self.closed_tabs.append({**tab, "closed_at": _now(),
                                     "closed_by": AGENTENV if target in self._closing else UNKNOWN})
            if self.log is not None:
                self.log.note("tab", {"event": "closed", "id": target, "url": redact_url(tab["url"]),
                                      **caused(self.causes.closed(target), self.log, target)})
        else:
            self.causes.closed(target)
        self._closing.discard(target)
        self._hidden.discard(target)

    def _on_Target_attachedToTarget(self, params: dict, session: Optional[str]) -> None:
        if params["targetInfo"].get("type") == "page":
            self._sessions[params["sessionId"]] = params["targetInfo"]["targetId"]
            self._spawn(self._let_run(params["sessionId"]))

    async def _let_run(self, session: str) -> None:
        """A page it attached to, enabled and, alone, held to its window's size, then told to run: a page attached as
        it opens, a popup say, or any new page alone, waits for that before it runs."""
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
        if (load := self._loading.get(session)) is not None and load.commits:
            load.loaded.add(load.commits[-1])
            load.check()

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
        try:
            size, sha256 = await asyncio.to_thread(_size_and_sha256, path)
            download.update(size=size, sha256=sha256)
        finally:
            download["ended_at"] = _now()

    def _spawn(self, work: Awaitable) -> None:
        task = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Future) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.warning(f"the browser's control connection: {type(error).__name__}: {error}")


@dataclass
class _StartLoad:
    """What the browser has said of a start tab's navigation, by loader, since it was sent to ``url``: its main frame's
    commits in order, the loads that fired after a commit, and its documents' requests, final responses and failures.
    Its document is the last its main frame committed from its own navigation on, which a page that sends itself
    elsewhere before it loads replaces. It's ``settled`` once its navigation or its document has failed, its document
    has loaded or, on one of the env's ``site``, answered an error."""
    session: str
    url: str
    site: Optional[str]
    started: float
    loader: Optional[str] = None
    error: Optional[str] = None
    commits: list[str] = field(default_factory=list)
    loaded: set[str] = field(default_factory=set)
    requests: dict[str, str] = field(default_factory=dict)
    responses: dict[str, tuple[Optional[int], Optional[str]]] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    settled: asyncio.Event = field(default_factory=asyncio.Event)
    navigating: Optional[asyncio.Future] = None

    def answered(self, navigating: asyncio.Future) -> None:
        """The browser's answer to its navigation: its loader, or why it couldn't navigate."""
        if navigating.cancelled():
            return
        if (error := navigating.exception()) is not None:
            self.error = f"{type(error).__name__}: {error}"
        else:
            answer = navigating.result()
            if "error" in answer:
                self.error = str(answer["error"].get("message", answer["error"]))
            else:
                self.error = answer.get("result", {}).get("errorText") or None
                self.loader = answer.get("result", {}).get("loaderId")
        self.check()

    def current(self) -> Optional[str]:
        """The loader of its document: the last its main frame committed, once its own navigation has; else None."""
        return self.commits[-1] if self.loader is not None and self.loader in self.commits else None

    def check(self) -> None:
        if self.error or (self.loader is not None and self.loader in self.failures):
            self.settled.set()
            return
        if (document := self.current()) is None:
            return
        status = self.responses.get(document, (None, None))[0]
        if document in self.failures or document in self.loaded or (
                self.site is not None and status is not None and status >= 400):
            self.settled.set()


def _landed_at(landed: asyncio.Future, navigating: asyncio.Future) -> None:
    """``landed`` given the loader a start tab's navigation answered with; cancelled when it failed or named none."""
    if landed.done():
        return
    answer = None if navigating.cancelled() or navigating.exception() is not None else navigating.result()
    loader = ((answer or {}).get("result") or {}).get("loaderId")
    if isinstance(loader, str):
        landed.set_result(loader)
    else:
        landed.cancel()


def _types(event: dict) -> bool:
    """Whether a page input event types something."""
    return event.get("method") in TYPING_METHODS and any(field in (event.get("params") or {}) for field in KEY_FIELDS)


def _cause() -> dict:
    """Who the control's command is for: the op under way, else agentenv itself."""
    actor = ACTOR.get()
    return {"seq": actor["seq"], "via": actor["via"], "who": actor.get("who")} if actor is not None else \
        {"via": "agentenv"}


def _size_and_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return path.stat().st_size, digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _origins(origins: list[str]) -> list[str]:
    try:
        return [origin_of(origin, strict=True) for origin in origins]
    except ValueError as e:
        raise InvalidArgument(str(e))


def _storage(origin: str) -> dict:
    return {"storageKey": f"{origin}/", "isLocalStorage": True}


def _cdp_cookie(cookie: CookieToSet, now: float) -> dict:
    """``cookie`` as CDP's ``Storage.setCookies`` takes it."""
    param: dict[str, Any] = {"name": cookie.name, "value": cookie.value, "httpOnly": cookie.httpOnly,
                             "secure": cookie.secure}
    if cookie.url:
        param["url"] = cookie.url
    if cookie.domain:
        param["domain"] = cookie.domain
    if cookie.path or cookie.domain:
        param["path"] = cookie.path or "/"
    if cookie.sameSite:
        param["sameSite"] = cookie.sameSite
    if cookie.partitionKey:
        param["partitionKey"] = cookie.partitionKey.model_dump()
    if cookie.max_age is not None:
        param["expires"] = now + cookie.max_age
    elif cookie.expires is not None and cookie.expires >= 0:
        param["expires"] = cookie.expires
    return param


def _sent(cookie: Cookie, url: str) -> bool:
    """Whether a request to ``url`` sends ``cookie``, as Playwright's ``cookies(urls)`` decides: its domain, with or
    without a leading dot, is the URL's host or a parent of it, its path a prefix of the URL's, and a secure cookie goes
    over https or to a local host."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    domain = cookie.domain.lower() if cookie.domain.startswith(".") else f".{cookie.domain.lower()}"
    if not f".{host}".endswith(domain) or not (parts.path or "/").startswith(cookie.path):
        return False
    return not cookie.secure or parts.scheme == "https" or host in LOCAL_HOSTS or host == "::1"

