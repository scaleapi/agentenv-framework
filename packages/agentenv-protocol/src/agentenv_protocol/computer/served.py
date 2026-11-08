"""How a computer app's own server is reached, over HTTP or a WebSocket, and how it serves (``serve_app``, ``apps``,
``svc``, ``launch``). On Linux and macOS it serves on the Unix socket ``SOCKET`` in the app's folder, which only the
computer reaches, so it takes no token. On Windows, whose asyncio serves no Unix socket, it serves on a free port of
127.0.0.1 and writes that port, with a token it draws, to ``ADDRESS`` there, in a folder its computer keeps the session
user's own; it answers only a request carrying the token, so no other user of the computer gets in, and whatever
reaches the app reads both from that file. Needs the ``app`` or ``computer`` extra; an app's own code imports none of
it."""
from __future__ import annotations

import hmac
import json
import os
import secrets
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Generator, Optional

import httpx
from websockets.asyncio.client import ClientConnection, connect, unix_connect

from .app import ADDRESS, SOCKET

#: The header a request to an app's server on loopback carries its token in.
TOKEN_HEADER = "x-agentenv-app-token"
#: Whether this OS's app servers serve on loopback rather than a Unix socket.
LOOPBACK = sys.platform == "win32"

Asgi = Callable[[dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]], Awaitable[None]]


@dataclass(frozen=True)
class Served:
    """An app's server as its computer reaches it: on its Unix ``socket``, or on ``port`` of 127.0.0.1 with ``token``,
    through no proxy the environment names; over HTTP, or a WebSocket."""

    socket: Optional[Path] = None
    port: Optional[int] = None
    token: Optional[str] = None

    def client(self, timeout: Optional[float]) -> httpx.AsyncClient:
        return httpx.AsyncClient(**self._client_args(httpx.AsyncHTTPTransport), timeout=timeout)

    def sync_client(self, timeout: Optional[float]) -> httpx.Client:
        return httpx.Client(**self._client_args(httpx.HTTPTransport), timeout=timeout)

    async def websocket(self, uri: str, headers: list[tuple[str, str]], **options: Any) -> ClientConnection:
        """A WebSocket opened to ``uri`` on the app's server, sending ``headers``, and the token where it takes one,
        with websockets' ``options``."""
        if self.socket is not None:
            return await unix_connect(str(self.socket), uri, additional_headers=headers, **options)
        return await connect(uri, host="127.0.0.1", port=self.port, proxy=None,
                             additional_headers=[*headers, (TOKEN_HEADER, self.token)], **options)

    def _client_args(self, transport: type) -> dict[str, Any]:
        if self.socket is not None:
            return {"transport": transport(uds=str(self.socket)), "base_url": "http://app"}
        return {"base_url": f"http://127.0.0.1:{self.port}", "auth": _Token(self.token), "trust_env": False}


def served(folder: Path) -> Served:
    """How the server of the app in ``folder`` is reached: on Windows, on the port and with the token it wrote there,
    raising ``FileNotFoundError`` until it has; elsewhere on its socket. Nothing answering is an ``OSError`` either
    way, once it's asked or here."""
    if not LOOPBACK:
        return Served(socket=folder / SOCKET)
    written = json.loads((folder / ADDRESS).read_text(encoding="utf-8"))
    return Served(port=written["port"], token=written["token"])


def bind(folder: Path) -> tuple[socket.socket, str]:
    """A socket listening on a free port of 127.0.0.1 and a token drawn for it, both written to ``ADDRESS`` in
    ``folder`` once it listens, so a client that reads them finds the server, its connection waiting until the server
    takes it."""
    listening = socket.create_server(("127.0.0.1", 0))
    token = secrets.token_urlsafe(32)
    address = folder / ADDRESS
    written = address.with_name(f"{address.name}.part")
    written.write_text(json.dumps({"port": listening.getsockname()[1], "token": token}), encoding="utf-8")
    os.replace(written, address)
    return listening, token


def guarded(app: Asgi, token: str) -> Asgi:
    """``app`` answering only a request or WebSocket that carries ``token`` in ``TOKEN_HEADER``, which it takes out
    before ``app`` sees the request; any other is refused with 403, or a WebSocket closed with 1008 before it opens."""
    expected, header = token.encode(), TOKEN_HEADER.encode()

    async def guard(scope: dict, receive: Callable[[], Awaitable[dict]], send: Callable[[dict], Awaitable[None]]):
        if scope["type"] not in ("http", "websocket"):
            return await app(scope, receive, send)
        given = [value for name, value in scope["headers"] if name == header]
        if len(given) != 1 or not hmac.compare_digest(given[0], expected):
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            await send({"type": "http.response.start", "status": 403,
                        "headers": [(b"content-type", b"text/plain; charset=utf-8")]})
            return await send({"type": "http.response.body", "body": b"this app's server takes its token alone"})
        return await app({**scope, "headers": [(name, value) for name, value in scope["headers"] if name != header]},
                         receive, send)

    return guard


class _Token(httpx.Auth):
    """``TOKEN_HEADER`` set on every request a client sends, in place of any it carried."""

    def __init__(self, token: str) -> None:
        self.token = token

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers[TOKEN_HEADER] = self.token
        yield request
