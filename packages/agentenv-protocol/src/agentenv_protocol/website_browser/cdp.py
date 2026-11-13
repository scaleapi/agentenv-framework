"""The CDP addresses a website browser hands agents: a token agentenv mints for each user, of which only a hash is kept
where the computer's session can read it, discovery answered from the browser, and a relay that gives each WebSocket
connection its own browser-level session over the browser's one debugging pipe, with a guard on every command."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

from starlette.requests import Request
from starlette.websockets import WebSocket, WebSocketDisconnect

from .devtools import DevToolsError
from .log import (ERROR_CHARS, KEY_FIELDS, TYPING_METHODS, BrowserLog, acts, redact_text, redacted_args,
                  redacted_params)
from .scope import ENDS_CONTEXT, MAKES_CONTEXT, WITH_INFO, Scope, Targets

if TYPE_CHECKING:
    from .contexts import Contexts
    from .control import BrowserControl
    from .devtools import DevToolsPipe

logger = logging.getLogger(__name__)

#: How many random bytes a token holds.
TOKEN_BYTES = 32
#: The route agents reach the relay under, through the driver's /svc/<app>/.
ROUTE = "/cdp/"
#: The close code and reason a connection gets when its browser goes away, which a client may connect again after.
RELAUNCHED = (1012, "the browser relaunched: connect again")
#: The close code and reason a connection scoped to a lease gets when the lease ends, which its address ends with.
LEASE_ENDED = (1001, "the lease ended: ask for an address again")
#: The close code and reason a connection gets when its client falls behind by more than ``OUTBOX_LIMIT_BYTES``.
TOO_SLOW = (1013, "the client fell too far behind: connect again")
#: How much a connection's client may have waiting to be sent to it before it is closed; a single answer bigger than
#: this still goes when nothing else waits.
OUTBOX_LIMIT_BYTES = 64 << 20
#: The close code and reason a connection gets after it closes or crashes the browser, which ends only it.
CLOSED = (1000, "the browser stays up for the env: only this connection ended")
#: CDP's own answer to a command for a session the connection doesn't have.
NO_SESSION = {"code": -32001, "message": "Session with given id not found."}
#: The code of a CDP error for a command not carried out, as Chromium answers its own refusals.
SERVER_ERROR = -32000
#: The answer to a message that isn't a command the relay can read, with JSON-RPC's code for an invalid request.
MALFORMED = {"code": -32600, "message": "a command is an object with a string method, and object params and a "
                                        "string sessionId if any"}
#: The commands that would end the browser every connection shares: each ends only the connection that sends it.
ENDS_BROWSER = frozenset({"Browser.close", "Browser.crash", "Browser.crashGpuProcess"})
#: The commands that set how and where downloads are saved, which agentenv keeps as its own.
SETS_DOWNLOADS = frozenset({"Browser.setDownloadBehavior", "Page.setDownloadBehavior"})
#: The schemes of the browser's own files, settings and tools, which an agent's navigation or new tab may not open.
OWN_SCHEMES = frozenset({"file", "chrome", "chrome-extension", "chrome-untrusted", "view-source", "devtools"})
#: The ``about:`` pages that aren't the browser's own under another name.
ABOUT_PAGES = frozenset({"about:blank", "about:srcdoc"})
#: How many of a connection's guarded commands its record keeps.
GUARDED_KEPT = 100
#: What the record says the guard did, by its action.
GUARDED = {"refuse": "refused", "send": "changed", "answer": "answered", "close": "closed"}
#: The commands that navigate a tab without naming the loader they navigate with, whose next navigation is theirs.
NAVIGATES_NEXT = frozenset({"Page.reload", "Page.navigateToHistoryEntry"})
#: What a command's entry says when its connection ended before the browser answered it.
ENDED_UNANSWERED = "the connection ended before the browser answered"


@dataclass
class Address:
    """A user's CDP address: ``http_url``, the base clients discover from, and ``ws_url``, the browser's WebSocket;
    ``named`` when it's under an origin the caller named, where its agent reaches the env."""

    user: str
    http_url: str
    ws_url: str
    named: bool = False
    #: The label of the lease it's scoped to, None for the browser's default context.
    context: Optional[str] = None


def reached(request: Request) -> str:
    """The address a request reached the browser at, scheme, host and path prefix: what the driver forwarded, else the
    request's own."""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}{request.headers.get('x-forwarded-prefix', '')}".rstrip("/")


def under(origin: str, request: Request) -> str:
    """The address a request reached the browser at, under ``origin``, where the caller's agent reaches the env: the
    origin, then the path prefix the driver forwarded."""
    return f"{origin.rstrip('/')}{request.headers.get('x-forwarded-prefix', '')}".rstrip("/")


def websocket_form(url: str) -> str:
    return "wss://" + url.removeprefix("https://") if url.startswith("https://") else \
        "ws://" + url.removeprefix("http://")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _address(user: str, base: str, named: bool, token: str, context: Optional[str] = None) -> Address:
    root = f"{base}{ROUTE}{token}"
    return Address(user, f"{root}/", f"{websocket_form(root)}/devtools/browser", named, context)


class Tokens:
    """The tokens minted, one per user: in memory, so asking twice gives the same address, and in ``path`` only by
    hash, with the user, the base address and whether its caller named it, so an address still works after the server
    restarts while the computer's session, which can read the file, can't use one. A token scoped to a lease is one per
    user and lease, in memory only, as no lease outlives the server; the lease's end revokes it."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._minted: dict[str, Address] = {}
        self._scoped: dict[tuple[str, str], Address] = {}
        self._scoped_known: dict[str, tuple[str, str, bool, str]] = {}
        try:
            stored = json.loads(path.read_text())
            self._known = {digest: (entry["user"], entry["base"], entry["named"])
                           for digest, entry in stored.items()}
        except (OSError, ValueError, TypeError, KeyError):
            self._known = {}

    def mint(self, user: str, base: str, *, named: bool = False, context: Optional[str] = None) -> Address:
        """``user``'s address under ``base``, ``named`` when its caller named where its agent reaches the env, scoped
        to the lease ``context`` when it names one: the one minted before, else a new one."""
        if context is not None:
            if (user, context) not in self._scoped:
                token = secrets.token_urlsafe(TOKEN_BYTES)
                self._scoped[(user, context)] = _address(user, base, named, token, context)
                self._scoped_known[_hash(token)] = (user, base, named, context)
            return self._scoped[(user, context)]
        if user not in self._minted:
            token = secrets.token_urlsafe(TOKEN_BYTES)
            self._minted[user] = _address(user, base, named, token)
            self._known[_hash(token)] = (user, base, named)
            self._save()
        return self._minted[user]

    def revoke_all(self) -> None:
        """Every address minted so far ended, as a reset ends them."""
        self._minted, self._known, self._scoped, self._scoped_known = {}, {}, {}, {}
        self._save()

    def revoke(self, context: str) -> None:
        """Every address scoped to the lease ``context`` ended, as the lease has."""
        self._scoped = {key: address for key, address in self._scoped.items() if key[1] != context}
        self._scoped_known = {digest: known for digest, known in self._scoped_known.items() if known[3] != context}

    def address(self, token: str) -> Optional[Address]:
        """The address ``token`` was minted for, or None for one that wasn't, or whose lease has ended."""
        digest = _hash(token)
        if (known := self._known.get(digest)) is not None:
            return _address(*known, token)
        if (scoped := self._scoped_known.get(digest)) is not None:
            return _address(*scoped[:3], token, scoped[3])
        return None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part = self.path.with_name(f".{self.path.name}.part")
        part.write_text(json.dumps({digest: {"user": user, "base": base, "named": named}
                                    for digest, (user, base, named) in self._known.items()}))
        os.replace(part, self.path)


@dataclass
class Verdict:
    """What the guard does with an agent's command: ``send`` it on with ``params``, ``changed`` or not; ``answer`` it
    with ``{}`` itself; answer it and ``close`` the connection; or ``refuse`` it for ``reason``."""

    action: Literal["send", "answer", "close", "refuse"]
    params: dict
    reason: str = ""
    changed: bool = False
    #: The error a refusal answers with in place of agentenv's own, as for a target the connection doesn't see.
    error: Optional[dict] = None


def guard(method: str, params: dict, *, dialogs_answered: bool, download_path: Optional[str] = None) -> Verdict:
    """What the relay does with an agent's ``method``, so it can't end the browser others share, route around the site
    proxy, save downloads elsewhere than its context's folder, ``download_path`` for a lease's, open the browser's own
    pages or race the dialogs agentenv answers when ``dialogs_answered``: a guardrail, since the sandbox's network is
    the boundary."""
    if method in ENDS_BROWSER:
        return Verdict("close", params)
    if method == "Target.exposeDevToolsProtocol":
        return Verdict("refuse", params, "it would give a page's script the browser's own CDP channel")
    if method == "Target.sendMessageToTarget":
        return Verdict("refuse", params, "only flattened sessions are served, whose commands the guard sees")
    if method == "Target.createBrowserContext" and {"proxyServer", "proxyBypassList"} & params.keys():
        return Verdict("refuse", params, "a context's traffic goes through the env's site proxy")
    if method in ("Page.navigate", "Target.createTarget") and (scheme := _own_scheme(params.get("url"))):
        return Verdict("refuse", params, f"{scheme}: URLs open the browser's own files, settings and tools")
    if method in SETS_DOWNLOADS:
        pinned = {"behavior": "default"} if download_path is None else {"behavior": "allow", "downloadPath": download_path}
        pinned.update({key: params[key] for key in ("browserContextId", "eventsEnabled") if key in params})
        return Verdict("send", pinned, changed=pinned != params)
    if method == "Page.handleJavaScriptDialog" and dialogs_answered:
        return Verdict("answer", params)
    return Verdict("send", params)


def _own_scheme(url: Any) -> Optional[str]:
    """``url``'s scheme when it's one of the browser's own, read as the browser reads it, without tabs and newlines and
    with spaces and controls trimmed from its ends; an ``about:`` page other than a blank one counts."""
    if not isinstance(url, str):
        return None
    url = url.translate({ord(c): None for c in "\t\n\r"}).strip("".join(map(chr, range(0x21))))
    scheme, colon, _ = url.partition(":")
    scheme = scheme.lower()
    if not colon:
        return None
    if scheme == "about":
        return None if url.lower().split("#")[0].split("?")[0] in ABOUT_PAGES else scheme
    return scheme if scheme in OWN_SCHEMES else None


class _Connection:
    """One client's WebSocket, its sessions over the pipe (its browser-level one and those it attached), its unanswered
    commands, what waits to be sent to it, and its entry in the record."""

    def __init__(self, websocket: WebSocket, user: str, session: str, record: dict, number: int,
                 scope: Optional[Scope] = None) -> None:
        self.websocket, self.user, self.session, self.record, self.number = websocket, user, session, record, number
        #: What it sees and acts in, a lease or the default context; None where the relay keeps no contexts.
        self.scope = scope
        self.sessions = {session}
        #: The tab each session it attached is on.
        self.session_tabs: dict[str, str] = {}
        #: Held by each of its commands from when it arrives until it's sent, so they're sent in the order they came,
        #: whatever one waits on before it's sent.
        self.order = asyncio.Lock()
        self.commands: set[asyncio.Future] = set()
        self.outbox: asyncio.Queue[str | tuple[int, str]] = asyncio.Queue()
        self.waiting_bytes = 0
        self.closing = False
        #: Whether it set how downloads are saved, and the contexts it made that end as it detaches.
        self.set_downloads = False
        self.ends_on_detach: set[str] = set()


class _Call:
    """An agent's tool call a driver serves through one CDP user's connections, open from its ``calls/begin`` to its
    ``calls/end``: its entry; the tab its commands last acted on, else the one they last read; whether a password field
    had focus for its typing, None until a typing command of its own is checked; and when it's given up on."""

    def __init__(self, begun: dict, user: str, call: dict) -> None:
        self.begun, self.user, self.tool, self.args = begun, user, call["tool"], call["args"]
        self.tab: Optional[str] = None
        self.acted = False
        self.password: Optional[bool] = None
        self.expiry: Optional[asyncio.TimerHandle] = None

    def credit(self, target: Optional[str], acting: bool, typed: bool, password: bool, alone: bool) -> None:
        """One of its user's commands credited to it, ``alone`` or with other calls open: the tab it was on, and for
        one that typed, whether a password field had focus there, taken as so when other calls shared it."""
        if target and alone and (acting or not self.acted):
            self.tab, self.acted = target, self.acted or acting
        if typed:
            self.password = True if password or not alone else bool(self.password)


class Relay:
    """Each WebSocket connection given its own browser-level session over ``devtools``, its commands sent on it past the
    guard and the answers and events of its sessions sent back, so several clients share one browser; and a record of
    who connected when, and what the guard did to their commands. ``dialogs_answered`` says agentenv answers the
    browser's dialogs itself. With a ``log``, each command is an entry in it, on the tab its session is on, and
    ``control``, agentenv's own connection, tells its tab's URL, whether a password field has focus there, and what
    causes the tab events a command brings. An agent's tool call a driver serves through a CDP user's connections,
    such as Playwright MCP's, is an entry too, which that user's commands are credited to while it's open. With
    ``contexts``, a context a connection makes is leased to its user, only one it made may it dispose, and a lease's
    downloads stay in its folder. A connection is scoped to a lease or to the default context, and sees only its scope
    and the contexts its user made (``scope``)."""

    def __init__(self, devtools: DevToolsPipe, *, dialogs_answered: bool = False, log: Optional[BrowserLog] = None,
                 control: Optional[BrowserControl] = None, contexts: Optional[Contexts] = None) -> None:
        self.devtools, self.dialogs_answered, self.log, self.control = devtools, dialogs_answered, log, control
        self.contexts = contexts
        #: Each target's context, as Chromium reports it.
        self.target_contexts = Targets()
        self._tasks: set[asyncio.Future] = set()
        self.connections: list[dict] = []
        self._by_session: dict[str, _Connection] = {}
        self._open: set[_Connection] = set()
        #: The tool calls open now, by their entry's sequence number.
        self.calls: dict[int, _Call] = {}
        devtools.listen(self._event)

    def begin_call(self, user: str, who: str, call: dict, *, acts: bool, within_s: float) -> int:
        """``who``'s tool ``call``, which a driver serves through ``user``'s connections, begun as an entry in the log:
        each of ``user``'s commands is credited to it until ``end_call`` ends it, or ``within_s`` passes and it's
        written as never said to have ended. Its entry's sequence number."""
        begun = self.log.begin({"via": "mcp", "who": who, "tool": call["tool"], "call": call["call"], "acts": acts})
        opened = self.calls[begun["seq"]] = _Call(begun, user, call)
        opened.expiry = asyncio.get_running_loop().call_later(
            within_s, self.end_call, begun["seq"], False, f"its driver didn't say it ended within {within_s:g} s")
        return begun["seq"]

    def end_call(self, seq: int, ok: bool, error: Optional[str] = None) -> bool:
        """The call begun as entry ``seq`` ended: its entry written with whether it worked and why not, the tab its
        commands acted on and that tab's URL, and, acting, its arguments, what it typed kept only once every command of
        its own that typed found no password field focused. False when no such call is open."""
        call = self.calls.pop(seq, None)
        if call is None:
            return False
        call.expiry.cancel()
        fields: dict[str, Any] = {"tab": call.tab, "url": self.control.url_of(call.tab)
                                  if self.control is not None and call.tab else None, "ok": ok}
        if error:
            fields["error"] = redact_text(error)[:ERROR_CHARS]
        if call.begun["action"]["acts"]:
            fields["args"] = redacted_args(call.tool, call.args, password=call.password is not False)
        self.log.finish(call.begun, **fields)
        return True

    async def serve(self, websocket: WebSocket, user: str, context: Optional[str] = None) -> None:
        """``websocket``, a client of ``user``'s, relayed over a browser-level session of its own until either ends,
        scoped to the lease ``context`` when it names one."""
        lease_id = None
        if context is not None:
            lease = self.contexts.get(context) if self.contexts is not None else None
            if lease is None:
                await websocket.accept()
                await websocket.close(code=LEASE_ENDED[0], reason=LEASE_ENDED[1])
                return
            lease_id = lease.id
        try:
            session = (await self.devtools.call("Target.attachToBrowserTarget"))["sessionId"]
        except Exception as e:
            await websocket.close(code=1013, reason=f"the browser isn't answering: {type(e).__name__}")
            return
        if context is not None and ((lease := self.contexts.get(context)) is None or lease.id != lease_id):
            with contextlib.suppress(Exception):
                await self.devtools.call("Target.detachFromTarget", {"sessionId": session})
            await websocket.accept()
            await websocket.close(code=LEASE_ENDED[0], reason=LEASE_ENDED[1])
            return
        record = {"user": user, "opened_at": _now(), "guarded": [], **({"context": context} if context else {})}
        scope = Scope(self.contexts, self.target_contexts, user, lease_id, self._frame_context) \
            if self.contexts is not None else None
        connection = _Connection(websocket, user, session, record, len(self.connections) + 1, scope)
        self._by_session[session] = connection
        self._open.add(connection)
        self.connections.append(record)
        await websocket.accept()
        sending = asyncio.ensure_future(self._send(connection))
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                text = message.get("text") if message.get("text") is not None else message["bytes"].decode()
                command = asyncio.ensure_future(self._command(connection, json.loads(text)))
                connection.commands.add(command)
                command.add_done_callback(connection.commands.discard)
        except (WebSocketDisconnect, ValueError):
            pass
        finally:
            record["closed_at"] = _now()
            connection.closing = True
            self._end(connection)
            for task in [*connection.commands, sending]:
                task.cancel()
            await asyncio.gather(*connection.commands, sending, return_exceptions=True)
            with contextlib.suppress(Exception):
                await self.devtools.call("Target.detachFromTarget", {"sessionId": session})
            self._detached(connection)

    def ended(self) -> None:
        """The browser has gone: every connection is closed with a reason a client may connect again after."""
        for connection in list(self._open):
            self._close(connection, RELAUNCHED)
        self.target_contexts.forget()

    def lease_ended(self, context_id: str) -> None:
        """The lease the browser knows as ``context_id`` has ended: each connection scoped to it is closed, saying
        so."""
        for connection in list(self._open):
            if connection.scope is not None and connection.scope.lease == context_id:
                self._close(connection, LEASE_ENDED)

    async def _command(self, connection: _Connection, message: Any) -> None:
        if connection.closing:
            return
        if not _well_formed(message):
            self._put(connection, {"id": message.get("id") if isinstance(message, dict) else None, "error": MALFORMED})
            return
        method, params = message["method"], message.get("params") or {}
        session = message.get("sessionId") or connection.session
        verdict = self._verdict(connection, session, method, params)
        logged = self._begin(connection, session, method, params)
        answer, guarded = {"error": {"code": SERVER_ERROR, "message": ENDED_UNANSWERED}}, None
        sending: Optional[asyncio.Future] = None
        replied = False

        def reply(answer: dict) -> None:
            nonlocal replied
            replied = True
            if sending is not None and isinstance(answer.get("result"), dict):
                self._answered(connection, method, params, answer["result"])
                if connection.scope is not None:
                    answer = {**answer, "result": connection.scope.answer(method, answer["result"])}
            self._put(connection, _reply(message, answer))

        try:
            async with connection.order:
                if logged is not None:
                    await logged.check_password()
                if session not in connection.sessions:
                    verdict, answer = None, {"error": NO_SESSION}
                elif verdict.action == "refuse":
                    answer = {"error": verdict.error or {"code": SERVER_ERROR,
                                                         "message": f"agentenv refuses {method}: {verdict.reason}"}}
                elif verdict.action != "send":
                    answer = {"result": {}}
                else:
                    sending = asyncio.ensure_future(self.devtools.request(method, verdict.params, session_id=session,
                                                                          answered=reply))
                    await asyncio.sleep(0)
            if sending is not None:
                try:
                    answer = await sending
                except ConnectionError as e:
                    answer = {"error": {"code": SERVER_ERROR, "message": str(e)}}
            if verdict is not None and (verdict.action != "send" or verdict.changed):
                guarded = GUARDED[verdict.action]
                self._guarded(connection, method, verdict)
            if not replied:
                reply(answer)
            if verdict is not None and verdict.action == "close":
                self._close(connection, CLOSED, dropping=False)
        finally:
            if sending is not None and not sending.done():
                sending.cancel()
            if logged is not None:
                logged.finish(answer, guarded)

    def _verdict(self, connection: _Connection, session: str, method: str, params: dict) -> Verdict:
        """The guard's verdict on ``connection``'s command, after its scope's (``Scope.verdict``), and with where a
        lease's downloads go."""
        if connection.scope is None:
            return guard(method, params, dialogs_answered=self.dialogs_answered)
        ruling = connection.scope.verdict(method, params, on_browser_session=session == connection.session)
        if ruling.refused:
            return Verdict("refuse", params, ruling.refused, error=ruling.error)
        params = ruling.params
        download_path = None
        if method in SETS_DOWNLOADS:
            connection.set_downloads = True
            context = params.get("browserContextId") if method.startswith("Browser.") else \
                self._context_of(connection.session_tabs.get(session))
            download_path = self.contexts.download_path(context)
        return guard(method, params, dialogs_answered=self.dialogs_answered, download_path=download_path)

    def _context_of(self, target: Optional[str]) -> Optional[str]:
        return self.control.tabs.get(target, {}).get("context_id") if self.control is not None and target else None

    def _frame_context(self, frame: object) -> Optional[str]:
        return self.control.context_of_frame(frame) if self.control is not None else None

    def _answered(self, connection: _Connection, method: str, params: dict, result: dict) -> None:
        """A context ``connection`` made leased to its user before it hears the answer, and one it disposed ended."""
        if self.contexts is None:
            return
        if method == MAKES_CONTEXT and isinstance(made := result.get("browserContextId"), str):
            self.contexts.made_by_agent(connection.user, made)
            if params.get("disposeOnDetach"):
                connection.ends_on_detach.add(made)
        elif method == ENDS_CONTEXT:
            self.contexts.disposed(params.get("browserContextId"))

    def _detached(self, connection: _Connection) -> None:
        """``connection`` gone: the contexts it made to end as it detached ended, and the leases' downloads sent to
        their folders again if it set downloads, which its session's end may set otherwise."""
        if self.contexts is None:
            return
        for context in connection.ends_on_detach:
            self.contexts.disposed(context)
        if connection.set_downloads:
            self.contexts.resend_downloads()

    def _begin(self, connection: _Connection, session: str, method: str, params: dict) -> Optional[_Logged]:
        """``connection``'s command, begun in the log, if there is one, on the tab ``session`` is on, or for a browser
        command the tab it names, ``for`` the tool calls of its user's open now; what it causes made known to the
        control."""
        if self.log is None:
            return None
        target = connection.session_tabs.get(session) if session != connection.session else None
        if target is None and isinstance(params.get("targetId"), str):
            target = params["targetId"]
        acting = acts(method)
        url = self.control.url_of(target) if self.control is not None and target else None
        crediting = [call for call in self.calls.values() if call.user == connection.user]
        action = {"via": "cdp", "who": connection.user, "connection": connection.number, "method": method,
                  "acts": acting, "tab": target, "url": url}
        if crediting:
            action["for"] = crediting[0].begun["seq"] if len(crediting) == 1 else [c.begun["seq"] for c in crediting]
        return _Logged(self, self.log.begin(action), method, params, target, acting, crediting)

    @staticmethod
    def _guarded(connection: _Connection, method: str, verdict: Verdict) -> None:
        """What the guard did to ``connection``'s ``method``, in its record, up to ``GUARDED_KEPT`` of them."""
        if len(guarded := connection.record["guarded"]) < GUARDED_KEPT:
            guarded.append({"method": method, "did": GUARDED[verdict.action], "at": _now()})

    def _event(self, message: dict) -> None:
        params = message.get("params") or {}
        if message.get("method") in WITH_INFO:
            self.target_contexts.saw(params.get("targetInfo"))
        connection = self._by_session.get(message.get("sessionId"))
        if connection is None:
            return
        if message.get("sessionId") == connection.session and not self._hears(connection, message):
            return
        if message.get("method") == "Target.attachedToTarget" and params.get("sessionId"):
            connection.sessions.add(params["sessionId"])
            self._by_session[params["sessionId"]] = connection
            if isinstance(target := (params.get("targetInfo") or {}).get("targetId"), str):
                connection.session_tabs[params["sessionId"]] = target
        elif message.get("method") == "Target.detachedFromTarget" and params.get("sessionId"):
            connection.sessions.discard(params["sessionId"])
            connection.session_tabs.pop(params["sessionId"], None)
            self._by_session.pop(params["sessionId"], None)
        if message["sessionId"] == connection.session:
            message = {key: value for key, value in message.items() if key != "sessionId"}
        self._put(connection, message)

    def _hears(self, connection: _Connection, message: dict) -> bool:
        """Whether ``connection`` hears an event on its browser session, as its scope says: never one about a session it
        doesn't have, and a target it doesn't see that auto-attached to it let go."""
        if connection.scope is None:
            return True
        method, params = message.get("method"), message.get("params") or {}
        if method == "Target.detachedFromTarget":
            return params.get("sessionId") in connection.sessions
        heard, let_go = connection.scope.hears(method, params)
        if let_go is not None:
            self._spawn(self._let_go(connection.session, let_go, params.get("waitingForDebugger")))
        return heard

    async def _let_go(self, owner: str, session: str, waiting: object) -> None:
        """A target auto-attached to the browser session ``owner`` as ``session`` that its connection doesn't see: told
        to run when it waits for its debugger, so it isn't held, and detached."""
        with contextlib.suppress(ConnectionError, DevToolsError):
            if waiting:
                await self.devtools.call("Runtime.runIfWaitingForDebugger", session_id=session)
            await self.devtools.call("Target.detachFromTarget", {"sessionId": session}, session_id=owner)

    def _spawn(self, work) -> None:
        task = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _put(self, connection: _Connection, message: dict) -> None:
        """``message`` waiting to be sent to ``connection``'s client, which is closed instead, with what waited dropped,
        once it has fallen too far behind."""
        if connection.closing:
            return
        text = json.dumps(message)
        if connection.waiting_bytes and connection.waiting_bytes + len(text) > OUTBOX_LIMIT_BYTES:
            self._close(connection, TOO_SLOW)
            return
        connection.waiting_bytes += len(text)
        connection.outbox.put_nowait(text)

    def _close(self, connection: _Connection, close: tuple[int, str], *, dropping: bool = True) -> None:
        """``connection`` closed with ``close`` after what waits to be sent to it, or with that dropped when
        ``dropping``; nothing is sent after it."""
        connection.closing = True
        self._end(connection)
        if dropping:
            while not connection.outbox.empty():
                connection.outbox.get_nowait()
            connection.waiting_bytes = 0
        connection.outbox.put_nowait(close)

    def _end(self, connection: _Connection) -> None:
        self._open.discard(connection)
        for session in connection.sessions:
            self._by_session.pop(session, None)

    async def _send(self, connection: _Connection) -> None:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            while True:
                message = await connection.outbox.get()
                if isinstance(message, tuple):
                    await connection.websocket.close(code=message[0], reason=message[1])
                    return
                connection.waiting_bytes -= len(message)
                await connection.websocket.send_text(message)


class _Logged:
    """A relayed command's entry in the log, from when it arrives until it's answered or its connection ends: what it
    causes made known to the control as it's sent, whether a password field has focus on its tab before it types,
    and its answer."""

    def __init__(self, relay: Relay, begun: dict, method: str, params: dict, target: Optional[str],
                 acting: bool, calls: list[_Call]) -> None:
        self.relay, self.begun, self.method, self.params, self.target, self.acting, self.calls = \
            relay, begun, method, params, target, acting, calls
        control = relay.control
        self._cause = cause = {"seq": begun["seq"], "via": "cdp", "who": begun["action"]["who"],
                               **({"for": begun["action"]["for"]} if "for" in begun["action"] else {})}
        self._landed: Optional[asyncio.Future] = None
        self._checks_password = self.password = False
        if control is None or not acting:
            return
        if method == "Target.createTarget":
            self._landed = asyncio.get_running_loop().create_future()
            control.causes.creating(self._landed, cause)
        elif method == "Target.closeTarget" and isinstance(params.get("targetId"), str):
            control.causes.closing(params["targetId"], cause)
        elif method == "Page.navigate" and target:
            self._landed = asyncio.get_running_loop().create_future()
            control.causes.navigating(target, self._landed, cause)
        elif method in NAVIGATES_NEXT and target:
            control.causes.navigates_next(target, cause)
        self._checks_password = method in TYPING_METHODS and bool(target) and any(field in params
                                                                                 for field in KEY_FIELDS)
        #: What a command that types types is withheld until its check says no password field has focus, so one
        #: that never gets checked, its connection gone first, keeps it withheld.
        self.password = self._checks_password

    async def check_password(self) -> None:
        """Before a command that types is sent: whether a password field has focus on its tab, true when that can't
        be told."""
        if self._checks_password:
            self.password = await self.relay.control.password_focused(self.target)

    def finish(self, answer: dict, guarded: Optional[str]) -> None:
        """The entry written with its answer, or with why there is none: whether it worked, why not, what the guard
        did, and, acting, its parameters as the log keeps them; what it caused made known."""
        result = answer.get("result") or {}
        if self._landed is not None and not self._landed.done():
            if "error" in answer or result.get("errorText"):
                self._landed.cancel()
            else:
                self._landed.set_result(result.get("targetId") if self.method == "Target.createTarget"
                                        else result.get("loaderId"))
        fields: dict[str, Any] = {"ok": "error" not in answer}
        if "error" in answer:
            fields["error"] = redact_text(str((answer["error"] or {}).get("message", answer["error"])))[:ERROR_CHARS]
            if self.method in NAVIGATES_NEXT and self.target and self.relay.control is not None:
                self.relay.control.causes.navigated_nowhere(self.target, self._cause)
        if guarded is not None:
            fields["guard"] = guarded
        if self.acting:
            fields["params"] = redacted_params(self.method, self.params, password=self.password)
        opened = result.get("targetId") if self.method == "Target.createTarget" else None
        if isinstance(opened, str):
            fields["tab"] = self.target = opened
        if self.target and self.relay.control is not None and self.begun["action"].get("url") is None:
            fields["url"] = self.relay.control.url_of(self.target)
        for call in self.calls:
            call.credit(self.target, self.acting, self._checks_password, self.password, alone=len(self.calls) == 1)
        self.relay.log.finish(self.begun, **fields)


def _reply(message: dict, answer: dict) -> dict:
    """The client's answer to its command ``message``: the browser's ``answer``, or the relay's own, under its id and
    on its session."""
    return {"id": message.get("id"), **{key: answer[key] for key in ("result", "error") if key in answer},
            **({"sessionId": message["sessionId"]} if message.get("sessionId") else {})}


def _well_formed(message: Any) -> bool:
    """Whether ``message`` is a command the relay can read: an object with a string ``method``, and object ``params``
    and a string ``sessionId`` if any."""
    return isinstance(message, dict) and isinstance(message.get("method"), str) and \
        isinstance(message.get("params") or {}, dict) and isinstance(message.get("sessionId") or "", str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
