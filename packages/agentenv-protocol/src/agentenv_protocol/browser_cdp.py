"""The CDP addresses a website browser hands agents: a token agentenv mints for each user, of which only a hash is kept
where the computer's session can read it, discovery answered from the browser, and a relay that gives each WebSocket
connection its own browser-level session over the browser's one debugging pipe."""
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
from typing import TYPE_CHECKING, Optional

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
#: CDP's own answer to a command for a session the connection doesn't have.
NO_SESSION = {"code": -32001, "message": "Session with given id not found."}


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


class _Connection:
    """One client's WebSocket, the sessions it holds over the pipe (its browser-level one and those it attached) and
    its commands the browser hasn't answered yet."""

    def __init__(self, websocket: WebSocket, user: str, session: str) -> None:
        self.websocket, self.user, self.session = websocket, user, session
        self.sessions = {session}
        self.commands: set[asyncio.Future] = set()
        self.outbox: asyncio.Queue[Optional[dict]] = asyncio.Queue()


class Relay:
    """Each WebSocket connection given its own browser-level session over ``devtools``, its commands sent on it and the
    answers and events of its sessions sent back, so several clients share one browser; and a record of who connected
    when."""

    def __init__(self, devtools: DevToolsPipe) -> None:
        self.devtools = devtools
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
        connection = _Connection(websocket, user, session)
        self._by_session[session] = connection
        self._open.add(connection)
        record = {"user": user, "opened_at": _now()}
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
            self._end(connection)
            for command in list(connection.commands):
                command.cancel()
            await asyncio.gather(*connection.commands, return_exceptions=True)
            connection.outbox.put_nowait(None)
            await asyncio.gather(sending, return_exceptions=True)
            with contextlib.suppress(Exception):
                await self.devtools.call("Target.detachFromTarget", {"sessionId": session})

    def ended(self) -> None:
        """The browser has gone: every connection is closed with a reason a client may connect again after."""
        for connection in list(self._open):
            self._end(connection)
            connection.outbox.put_nowait({"close": RELAUNCHED})

    async def _command(self, connection: _Connection, message: dict) -> None:
        session = message.get("sessionId") or connection.session
        if session not in connection.sessions:
            answer = {"error": NO_SESSION}
        else:
            try:
                answer = await self.devtools.request(message.get("method", ""), message.get("params") or {},
                                                     session_id=session)
            except ConnectionError as e:
                answer = {"error": {"code": -32000, "message": str(e)}}
        reply = {"id": message.get("id"), **{key: answer[key] for key in ("result", "error") if key in answer}}
        if message.get("sessionId"):
            reply["sessionId"] = message["sessionId"]
        connection.outbox.put_nowait(reply)

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
        connection.outbox.put_nowait(message)

    def _end(self, connection: _Connection) -> None:
        self._open.discard(connection)
        for session in connection.sessions:
            self._by_session.pop(session, None)

    async def _send(self, connection: _Connection) -> None:
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            while (message := await connection.outbox.get()) is not None:
                if "close" in message:
                    code, reason = message["close"]
                    await connection.websocket.close(code=code, reason=reason)
                    return
                await connection.websocket.send_text(json.dumps(message))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
