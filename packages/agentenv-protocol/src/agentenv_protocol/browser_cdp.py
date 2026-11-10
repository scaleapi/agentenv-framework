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

if TYPE_CHECKING:
    from .website_browser import DevToolsPipe

logger = logging.getLogger(__name__)

#: How many random bytes a token holds.
TOKEN_BYTES = 32
#: The route agents reach the relay under, through the driver's /svc/<app>/.
ROUTE = "/cdp/"
#: The close code and reason a connection gets when its browser goes away, which a client may connect again after.
RELAUNCHED = (1012, "the browser relaunched: connect again")
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


@dataclass
class Address:
    """A user's CDP address: ``http_url``, the base clients discover from, and ``ws_url``, the browser's WebSocket."""

    user: str
    http_url: str
    ws_url: str


def reached(request: Request) -> str:
    """The address a request reached the browser at, scheme, host and path prefix: what the driver forwarded, else the
    request's own."""
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}{request.headers.get('x-forwarded-prefix', '')}".rstrip("/")


def websocket_form(url: str) -> str:
    return "wss://" + url.removeprefix("https://") if url.startswith("https://") else \
        "ws://" + url.removeprefix("http://")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _address(user: str, base: str, token: str) -> Address:
    root = f"{base}{ROUTE}{token}"
    return Address(user, f"{root}/", f"{websocket_form(root)}/devtools/browser")


class Tokens:
    """The tokens minted, one per user: in memory, so asking twice gives the same address, and in ``path`` only by
    hash, with the user and the base address, so an address still works after the server restarts while the
    computer's session, which can read the file, can't use one."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._minted: dict[str, Address] = {}
        try:
            stored = json.loads(path.read_text())
            self._known = {digest: (entry["user"], entry["base"]) for digest, entry in stored.items()}
        except (OSError, ValueError, TypeError, KeyError):
            self._known = {}

    def mint(self, user: str, base: str) -> Address:
        """``user``'s address under ``base``: the one minted before, else a new one."""
        if user not in self._minted:
            token = secrets.token_urlsafe(TOKEN_BYTES)
            self._minted[user] = _address(user, base, token)
            self._known[_hash(token)] = (user, base)
            self._save()
        return self._minted[user]

    def revoke_all(self) -> None:
        """Every address minted so far ended, as a reset ends them."""
        self._minted, self._known = {}, {}
        self._save()

    def address(self, token: str) -> Optional[Address]:
        """The address ``token`` was minted for, or None for one that wasn't."""
        known = self._known.get(_hash(token))
        return _address(*known, token) if known is not None else None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        part = self.path.with_name(f".{self.path.name}.part")
        part.write_text(json.dumps({digest: {"user": user, "base": base}
                                    for digest, (user, base) in self._known.items()}))
        os.replace(part, self.path)


@dataclass
class Verdict:
    """What the guard does with an agent's command: ``send`` it on with ``params``, ``changed`` or not; ``answer`` it
    with ``{}`` itself; answer it and ``close`` the connection; or ``refuse`` it for ``reason``."""

    action: Literal["send", "answer", "close", "refuse"]
    params: dict
    reason: str = ""
    changed: bool = False


def guard(method: str, params: dict, *, dialogs_answered: bool) -> Verdict:
    """What the relay does with an agent's ``method``, so it can't end the browser others share, route around the site
    proxy, save downloads elsewhere, open the browser's own pages or race the dialogs agentenv answers when
    ``dialogs_answered``: a guardrail, since the sandbox's network is the boundary."""
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
        pinned = {"behavior": "default",
                  **{key: params[key] for key in ("browserContextId", "eventsEnabled") if key in params}}
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

    def __init__(self, websocket: WebSocket, user: str, session: str, record: dict) -> None:
        self.websocket, self.user, self.session, self.record = websocket, user, session, record
        self.sessions = {session}
        self.commands: set[asyncio.Future] = set()
        self.outbox: asyncio.Queue[str | tuple[int, str]] = asyncio.Queue()
        self.waiting_bytes = 0
        self.closing = False


class Relay:
    """Each WebSocket connection given its own browser-level session over ``devtools``, its commands sent on it past the
    guard and the answers and events of its sessions sent back, so several clients share one browser; and a record of
    who connected when, and what the guard did to their commands. ``dialogs_answered`` says agentenv answers the
    browser's dialogs itself."""

    def __init__(self, devtools: DevToolsPipe, *, dialogs_answered: bool = False) -> None:
        self.devtools, self.dialogs_answered = devtools, dialogs_answered
        self.connections: list[dict] = []
        self._by_session: dict[str, _Connection] = {}
        self._open: set[_Connection] = set()
        devtools.listen(self._event)

    async def serve(self, websocket: WebSocket, user: str) -> None:
        """``websocket``, a client of ``user``'s, relayed over a browser-level session of its own until either ends."""
        try:
            session = (await self.devtools.call("Target.attachToBrowserTarget"))["sessionId"]
        except Exception as e:
            await websocket.close(code=1013, reason=f"the browser isn't answering: {type(e).__name__}")
            return
        record = {"user": user, "opened_at": _now(), "guarded": []}
        connection = _Connection(websocket, user, session, record)
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

    def ended(self) -> None:
        """The browser has gone: every connection is closed with a reason a client may connect again after."""
        for connection in list(self._open):
            self._close(connection, RELAUNCHED)

    async def _command(self, connection: _Connection, message: Any) -> None:
        if connection.closing:
            return
        if not _well_formed(message):
            self._put(connection, {"id": message.get("id") if isinstance(message, dict) else None, "error": MALFORMED})
            return
        method, params = message["method"], message.get("params") or {}
        session = message.get("sessionId") or connection.session
        verdict = guard(method, params, dialogs_answered=self.dialogs_answered)
        if session not in connection.sessions:
            verdict, answer = None, {"error": NO_SESSION}
        elif verdict.action == "refuse":
            answer = {"error": {"code": SERVER_ERROR, "message": f"agentenv refuses {method}: {verdict.reason}"}}
        elif verdict.action != "send":
            answer = {"result": {}}
        else:
            try:
                answer = await self.devtools.request(method, verdict.params, session_id=session)
            except ConnectionError as e:
                answer = {"error": {"code": SERVER_ERROR, "message": str(e)}}
        if verdict is not None and (verdict.action != "send" or verdict.changed):
            self._guarded(connection, method, verdict)
        reply = {"id": message.get("id"), **{key: answer[key] for key in ("result", "error") if key in answer}}
        if message.get("sessionId"):
            reply["sessionId"] = message["sessionId"]
        self._put(connection, reply)
        if verdict is not None and verdict.action == "close":
            self._close(connection, CLOSED, dropping=False)

    @staticmethod
    def _guarded(connection: _Connection, method: str, verdict: Verdict) -> None:
        """What the guard did to ``connection``'s ``method``, in its record, up to ``GUARDED_KEPT`` of them."""
        if len(guarded := connection.record["guarded"]) < GUARDED_KEPT:
            guarded.append({"method": method, "did": GUARDED[verdict.action], "at": _now()})

    def _event(self, message: dict) -> None:
        connection = self._by_session.get(message.get("sessionId"))
        if connection is None:
            return
        params = message.get("params") or {}
        if message.get("method") == "Target.attachedToTarget" and params.get("sessionId"):
            connection.sessions.add(params["sessionId"])
            self._by_session[params["sessionId"]] = connection
        elif message.get("method") == "Target.detachedFromTarget" and params.get("sessionId"):
            connection.sessions.discard(params["sessionId"])
            self._by_session.pop(params["sessionId"], None)
        if message["sessionId"] == connection.session:
            message = {key: value for key, value in message.items() if key != "sessionId"}
        self._put(connection, message)

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


def _well_formed(message: Any) -> bool:
    """Whether ``message`` is a command the relay can read: an object with a string ``method``, and object ``params``
    and a string ``sessionId`` if any."""
    return isinstance(message, dict) and isinstance(message.get("method"), str) and \
        isinstance(message.get("params") or {}, dict) and isinstance(message.get("sessionId") or "", str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
