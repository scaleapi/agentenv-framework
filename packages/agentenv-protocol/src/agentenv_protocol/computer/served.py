"""Where a computer app's own server answers, and how its computer reaches it (``serve_app``, ``apps``, ``svc``,
``launch``). On Linux and macOS it serves on the Unix socket ``SOCKET`` in the app's folder, which only the computer
reaches, so it takes no token. On Windows, whose asyncio serves no Unix socket, it serves on a free port of 127.0.0.1
and writes that port, with a token it draws, to ``ADDRESS`` there, in a folder its computer keeps the session user's
own; it answers only a request carrying the token, so no other user of the computer gets in, and whatever reaches the
app reads both from that file."""
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

#: Where an app's own server listens on Linux and macOS, in the app's folder.
SOCKET = Path(".agentenv/server.sock")
#: Where an app's server on Windows writes the port it serves on and its token, in the app's folder.
ADDRESS = Path(".agentenv/server.json")
#: The header a request to an app's server on loopback carries its token in.
TOKEN_HEADER = "x-agentenv-app-token"
#: Whether this OS's app servers serve on loopback rather than a Unix socket.
LOOPBACK = sys.platform == "win32"

Asgi = Callable[[dict, Callable[[], Awaitable[dict]], Callable[[dict], Awaitable[None]]], Awaitable[None]]


@dataclass(frozen=True)
class Served:
    """An app's server as its computer reaches it: on its Unix ``socket``, or on ``port`` of 127.0.0.1 with ``token``,
    through no proxy the environment names."""

    socket: Optional[Path] = None
    port: Optional[int] = None
    token: Optional[str] = None

    def client(self, timeout: Optional[float]) -> httpx.AsyncClient:
        return httpx.AsyncClient(**self._client_args(httpx.AsyncHTTPTransport), timeout=timeout)

    def sync_client(self, timeout: Optional[float]) -> httpx.Client:
        return httpx.Client(**self._client_args(httpx.HTTPTransport), timeout=timeout)

    def _client_args(self, transport: type) -> dict[str, Any]:
        if self.socket is not None:
            return {"transport": transport(uds=str(self.socket)), "base_url": "http://app"}
        return {"base_url": f"http://127.0.0.1:{self.port}", "auth": _Token(self.token), "trust_env": False}


def served(folder: Path) -> Served:
    """How the server of the app in ``folder`` is reached: on the port and with the token it wrote, where it serves on
    loopback, else on its socket. Nothing answering either way is an ``OSError`` once it's asked."""
    try:
        written = json.loads((folder / ADDRESS).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Served(socket=folder / SOCKET)
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
