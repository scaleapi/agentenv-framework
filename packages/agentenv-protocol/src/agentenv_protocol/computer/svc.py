"""The driver's ``/svc/<name>/`` route to each app on its computer: a request sent on to the app's server with its body
and its answer streamed, and a WebSocket relayed to it both ways. It needs the ``computer`` extra's ``websockets``, so
only the driver imports it; an app's venv, with the ``app`` extra, imports ``apps`` and never this."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections.abc import AsyncIterator, Iterable
from pathlib import Path
from urllib.parse import quote, unquote

import httpx
import websockets
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import ClientConnection, unix_connect

from ..agent_env_environment import _error_response
from .app import SOCKET
from .errors import DeliveryFailed

#: The methods the route forwards.
METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
#: The headers that belong to one connection, which a proxy doesn't pass on.
HOP_HEADERS = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "proxy-connection",
                         "te", "trailer", "transfer-encoding", "upgrade"})
#: The address a request reached the driver at, which the driver sets for the app in place of any the caller sent.
FORWARDED = ("x-forwarded-host", "x-forwarded-proto", "x-forwarded-prefix")
#: A WebSocket close code that means the connection ended without one, which a close can't send on.
UNSENDABLE_CLOSES = (1005, 1006, 1015)
#: A Host header that names a host, and its port or not, and nothing else.
HOST = re.compile(r"(?:[A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+\])(?::[0-9]{1,5})?")
#: What a path's segments may hold unescaped, besides letters, digits and ``_.-~``.
PATH_SAFE = "/:@!$&'()*+,;="
#: The loggers the driver's server writes each request's path to: HTTP requests, and WebSocket handshakes.
SERVER_LOGGERS = ("uvicorn.access", "uvicorn.error")
#: What a logged path shows past a route an app opens to agents.
REDACTED = "<redacted>"
#: The logger of the client requests are forwarded with, which logs each one's URL, an agent's token in it, at INFO.
FORWARDER_LOGGER = "httpx"


def passed_on(headers: Iterable[tuple[str, str]], *dropped: str) -> list[tuple[str, str]]:
    """``headers`` as a proxy passes them on: without the hop-by-hop ones, those a ``Connection`` header names, and
    ``dropped``."""
    headers = list(headers)
    named = {token.strip().lower() for name, value in headers if name.lower() == "connection"
             for token in value.split(",")}
    skipped = HOP_HEADERS | named | set(dropped)
    return [(name, value) for name, value in headers if name.lower() not in skipped]


def to_app(headers: Iterable[tuple[str, str]], host: str, scheme: str, prefix: str) -> list[tuple[str, str]]:
    """A caller's headers as the app gets them: passed on without the control token's ``Authorization`` or any address
    the caller forwarded, and with the one it reached the driver at, its ``host``, ``scheme`` and the app's
    ``prefix``."""
    return passed_on(headers, "authorization", *FORWARDED) + [
        ("x-forwarded-host", host), ("x-forwarded-proto", scheme), ("x-forwarded-prefix", prefix)]


def target(connection: Request | WebSocket) -> str:
    """The path and query a request under ``/svc/<name>/`` asks its app for, as it came: still percent-encoded, so an
    escaped ``?``, ``#`` or ``/`` stays one. The raw path's segments past any the driver is mounted under must be
    ``svc`` and the app's name, else the routed path is escaped again."""
    mounted = len([segment for segment in connection.scope.get("root_path", "").split("/") if segment])
    parts = (connection.scope.get("raw_path") or b"").decode("latin-1").split("/")[1 + mounted:]
    if len(parts) > 2 and [unquote(part) for part in parts[:2]] == ["svc", connection.path_params["name"]]:
        tail = "/".join(parts[2:])
    else:
        tail = quote(connection.path_params["path"], safe=PATH_SAFE)
    query = connection.scope.get("query_string", b"").decode("latin-1")
    return f"/{tail}" + (f"?{query}" if query else "")


def _bytes(headers: Iterable[tuple[str, str]]) -> list[tuple[bytes, bytes]]:
    """Headers decoded as Latin-1, as Starlette and the helpers here hold them, as the bytes they came as."""
    return [(name.encode("latin-1"), value.encode("latin-1")) for name, value in headers]


def open_to_agents(routes: list[str], path: str) -> bool:
    """Whether ``path``, a request's target, is under one of the app's ``routes`` agents reach without the control
    token, and has no ``.`` or ``..`` segment, escaped or not, that could step out of it."""
    path = path.partition("?")[0]
    if any(segment in (".", "..") for segment in unquote(path).split("/")):
        return False
    return any(path.startswith(route) for route in routes)


class AgentRouteRedaction(logging.Filter):
    """The paths the driver's server logs, cut short past a route an app opens to agents, with ``routes`` each app's
    by name: what follows such a route, as a CDP token does, is a secret the app checks, and the computer's session can
    read the driver's log."""

    def __init__(self, routes: dict[str, list[str]]) -> None:
        super().__init__()
        self.routes = routes

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(self.redacted(arg) if isinstance(arg, str) else arg for arg in record.args)
        return True

    def redacted(self, path: str) -> str:
        parts = path.split("/", 3)
        if len(parts) < 4 or parts[0] or parts[1] != "svc":
            return path
        route = next((route for route in self.routes.get(unquote(parts[2]), []) if f"/{parts[3]}".startswith(route)),
                     None)
        return path if route is None else f"/svc/{parts[2]}{route}{REDACTED}"


async def forward(folder: Path, name: str, request: Request) -> Response:
    """``request`` sent on to the app ``name``'s server in ``folder`` with the caller's headers and none of the
    client's own, its body and its answer streamed; it takes as long as the app does."""
    headers = to_app(request.headers.items(), request.headers.get("host", ""), request.url.scheme, f"/svc/{name}")
    has_body = request.headers.get("content-length", "0") != "0" or "transfer-encoding" in request.headers
    http = httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(folder / SOCKET)), base_url="http://app",
                             timeout=None)
    http.headers.clear()
    try:
        answer = await http.send(http.build_request(request.method, target(request), headers=_bytes(headers),
                                                    content=request.stream() if has_body else None), stream=True)
    except (httpx.HTTPError, OSError) as e:
        await http.aclose()
        return _error_response(DeliveryFailed(f"app {name}'s server isn't answering: {type(e).__name__}: {e}"))
    except BaseException:
        await http.aclose()
        raise

    async def close() -> None:
        await answer.aclose()
        await http.aclose()

    async def streamed() -> AsyncIterator[bytes]:
        try:
            async for chunk in answer.aiter_raw():
                yield chunk
        finally:
            await close()

    response = StreamingResponse(streamed(), status_code=answer.status_code, background=BackgroundTask(close))
    response.raw_headers = _bytes(passed_on((name.decode("latin-1"), value.decode("latin-1"))
                                            for name, value in answer.headers.raw))
    return response


async def relay(folder: Path, name: str, websocket: WebSocket) -> None:
    """``websocket`` relayed both ways to the app ``name``'s server in ``folder`` until either side closes; refused,
    before it opens, with a Host that isn't a host and port, or with the app's server not answering."""
    host = websocket.headers.get("host", "")
    if host and not HOST.fullmatch(host):
        await websocket.close(code=1008)
        return
    headers = [(key, _utf8(value)) for key, value in to_app(
        websocket.headers.items(), host, "https" if websocket.url.scheme == "wss" else "http", f"/svc/{name}")
        if key.lower() != "host" and not key.lower().startswith("sec-websocket-")]
    offered = [p.strip() for p in websocket.headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
    try:
        upstream = await unix_connect(str(folder / SOCKET), f"ws://{host or 'app'}{target(websocket)}",
                                      additional_headers=headers, subprotocols=offered or None, max_size=None,
                                      compression=None, ping_interval=None, user_agent_header=None)
    except (OSError, ValueError, asyncio.TimeoutError, websockets.WebSocketException):
        await websocket.close(code=1011)
        return
    async with upstream:
        await websocket.accept(subprotocol=upstream.subprotocol)
        await _both_ways(websocket, upstream)


def _utf8(value: str) -> str:
    """A header value decoded as Latin-1 as the string ``websockets`` sends as the same bytes, which it encodes as
    UTF-8; bytes that aren't UTF-8 become replacement characters."""
    return value.encode("latin-1").decode("utf-8", "replace")


async def _both_ways(client: WebSocket, app: ClientConnection) -> None:
    """Each message from either side sent to the other until one side closes, then the other closed with its code."""
    async def inbound() -> None:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                code = message.get("code") or 1000
                await app.close(1000 if code in UNSENDABLE_CLOSES else code)
                return
            await app.send(message["text"] if message.get("text") is not None else message["bytes"])

    async def outbound() -> None:
        with contextlib.suppress(websockets.ConnectionClosed):
            async for message in app:
                if isinstance(message, str):
                    await client.send_text(message)
                else:
                    await client.send_bytes(message)
        code = app.close_code or 1000
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await client.close(1000 if code in UNSENDABLE_CLOSES else code)

    tasks = [asyncio.ensure_future(inbound()), asyncio.ensure_future(outbound())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
