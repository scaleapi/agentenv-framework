"""A slot's door: the one container of a slot of a shared VM that publishes ports, each forwarded to one of the slot's
services (``python -m gateway.door``). It runs from the gateway image, so it needs nothing the gateway doesn't.

A request gets through only with the slot's token as the ``X-Agentenv-Slot-Token`` header, compared in constant time;
the door removes that header and forwards the rest, ``Authorization`` and ``Host`` included, untouched. HTTP bodies
stream both ways, so server-sent events and long downloads pass as they come, and WebSockets are relayed frame by
frame. Every other request gets 401. Browsers can't set the header, and no cookie or ticket sign-in exists yet, so a
browser is refused too.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import logging
import os

import httpx
import uvicorn
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from .constants import DOOR_ROUTES_ENV, DOOR_TOKEN_ENV, SLOT_TOKEN_HEADER

logger = logging.getLogger(__name__)

_TOKEN_HEADER = SLOT_TOKEN_HEADER.lower().encode()
# Headers that describe one connection, not the request, so a proxy never forwards them (RFC 9110 7.6.1).
_HOP_BY_HOP = frozenset({b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization", b"te",
                         b"trailer", b"transfer-encoding", b"upgrade", b"proxy-connection"})
_REFUSAL = (f"401: this slot's door takes only the {SLOT_TOKEN_HEADER} header. Browser sign-in (a cookie or a "
            f"ticket) isn't supported yet.\n").encode()
_UPSTREAM_CONNECT_TIMEOUT_S = 10.0


def parse_routes(spec: str) -> dict[int, str]:
    """``<listen port>=<host>:<port>,...`` as each listen port's upstream ``host:port``."""
    routes: dict[int, str] = {}
    for entry in filter(None, (part.strip() for part in spec.split(","))):
        port, _, upstream = entry.partition("=")
        host, _, upstream_port = upstream.rpartition(":")
        if not (port.isdigit() and host and upstream_port.isdigit()) or int(port) in routes:
            raise ValueError(f"bad or repeated door route {entry!r}; expected <listen port>=<host>:<port>")
        routes[int(port)] = upstream
    if not routes:
        raise ValueError("the door has no routes")
    return routes


class Door:
    """The ASGI app in front of one upstream ``host:port``, letting in only requests that carry ``token``."""

    def __init__(self, upstream: str, token: str, client: httpx.AsyncClient):
        if not token:
            raise ValueError("a door needs a token")
        self._upstream = upstream
        self._token = token.encode()
        self._client = client

    async def __call__(self, scope, receive, send) -> None:
        admitted = self._admitted(scope["headers"])
        headers = [(k, v) for k, v in scope["headers"] if k != _TOKEN_HEADER and k not in _HOP_BY_HOP]
        if scope["type"] == "websocket":
            await self._websocket(scope, receive, send, headers, admitted)
        elif admitted:
            await self._http(scope, receive, send, headers)
        else:
            await _respond(send, 401, _REFUSAL)

    def _admitted(self, headers) -> bool:
        tokens = [value for key, value in headers if key == _TOKEN_HEADER]
        return len(tokens) == 1 and hmac.compare_digest(tokens[0], self._token)

    def _target(self, scope, scheme: str) -> str:
        path = scope.get("raw_path") or scope["path"].encode()
        query = f"?{scope['query_string'].decode('latin-1')}" if scope.get("query_string") else ""
        return f"{scheme}://{self._upstream}{path.decode('latin-1')}{query}"

    async def _http(self, scope, receive, send, headers) -> None:
        has_body = any(key in (b"content-length", b"transfer-encoding") for key, _ in scope["headers"])

        async def body():
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                yield message.get("body", b"")
                if not message.get("more_body"):
                    return

        request = self._client.build_request(scope["method"], self._target(scope, "http"), headers=headers,
                                             content=body() if has_body else None)
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError as e:
            logger.warning("door: upstream %s unreachable: %r", self._upstream, e)
            await _respond(send, 502, b"502: the slot's service isn't answering\n")
            return
        try:
            await send({"type": "http.response.start", "status": response.status_code,
                        "headers": [(k, v) for k, v in response.headers.raw if k.lower() not in _HOP_BY_HOP]})
            async for chunk in response.aiter_raw():
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            await response.aclose()

    async def _websocket(self, scope, receive, send, headers, admitted: bool) -> None:
        if (await receive())["type"] != "websocket.connect":
            return
        if not admitted:
            await _deny_websocket(scope, send, 401, _REFUSAL)
            return
        forwarded = [(k.decode("latin-1"), v.decode("latin-1")) for k, v in headers
                     if k != b"host" and not k.startswith(b"sec-websocket-")]
        try:
            upstream = await connect(self._target(scope, "ws"), additional_headers=forwarded,
                                     subprotocols=scope.get("subprotocols") or None, max_size=None,
                                     open_timeout=_UPSTREAM_CONNECT_TIMEOUT_S)
        except Exception as e:
            logger.warning("door: WebSocket to %s failed: %r", self._upstream, e)
            await _deny_websocket(scope, send, 502, b"502: the slot's service refused the WebSocket\n")
            return
        await send({"type": "websocket.accept", "subprotocol": upstream.subprotocol})

        async def downstream():
            with contextlib.suppress(ConnectionClosed):
                async for frame in upstream:
                    await send({"type": "websocket.send", **({"text": frame} if isinstance(frame, str) else {"bytes": frame})})
            await send({"type": "websocket.close", "code": upstream.close_code or 1000})

        async def upstream_bound():
            while (message := await receive())["type"] == "websocket.receive":
                await upstream.send(message["text"] if message.get("text") is not None else message.get("bytes") or b"")
            await upstream.close()

        tasks = [asyncio.ensure_future(downstream()), asyncio.ensure_future(upstream_bound())]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await upstream.close()


async def _respond(send, status: int, body: bytes) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


async def _deny_websocket(scope, send, status: int, body: bytes) -> None:
    """Refuse a WebSocket handshake with an HTTP ``status`` where the server can send one, else close it."""
    if "websocket.http.response" in (scope.get("extensions") or {}):
        await send({"type": "websocket.http.response.start", "status": status,
                    "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "websocket.http.response.body", "body": body})
    else:
        await send({"type": "websocket.close", "code": 1008})


def door_client() -> httpx.AsyncClient:
    """The client a door forwards with: no timeout once connected, as a stream may stay open, no redirects followed,
    and no proxy from the environment."""
    return httpx.AsyncClient(timeout=httpx.Timeout(None, connect=_UPSTREAM_CONNECT_TIMEOUT_S),
                             follow_redirects=False, trust_env=False)


async def serve(routes: dict[int, str], token: str) -> None:
    async with door_client() as client:
        servers = [uvicorn.Server(uvicorn.Config(Door(upstream, token, client), host="0.0.0.0", port=port,
                                                 lifespan="off", proxy_headers=False, server_header=False,
                                                 date_header=False, log_level="warning"))
                   for port, upstream in routes.items()]
        await asyncio.gather(*(server.serve() for server in servers))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    routes = parse_routes(os.environ.get(DOOR_ROUTES_ENV, ""))
    token = os.environ.get(DOOR_TOKEN_ENV, "")
    if not token:
        raise SystemExit(f"{DOOR_TOKEN_ENV} is unset, and a door without a token would let anyone in")
    asyncio.run(serve(routes, token))


if __name__ == "__main__":
    main()
