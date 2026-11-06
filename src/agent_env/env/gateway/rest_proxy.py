"""The gateway's REST reverse proxy: what it forwards to a backing server, and how the backing servers' cards name
the proxied routes.

A request to ``/svc/{service}/{path}`` or ``/website/{service}/{path}`` is forwarded to the service's base URL and
its response streamed through, under five invariants:

- the path goes upstream as received: uvicorn decodes ``scope["path"]`` for routing, so the still-encoded tail is
  recovered from ``raw_path`` and ``%2F``, ``%3F`` and ``%23`` reach the upstream as data, not separators;
- headers are forwarded one entry per raw header (repeated fields keep their multiplicity, non-latin-1 values never
  round-trip through str), minus the hop-by-hop set, in both directions;
- the role header is the gateway's to write: the client's copies are dropped and the role the gateway attributed
  the request to goes in their place, and a POST to a backing MCP server's own endpoint gets the same treatment
  inside its JSON-RPC ``params._meta``;
- the upstream learns the address the client reached it at: ``X-Forwarded-Host`` and ``X-Forwarded-Proto`` (a front's
  own, when one before the gateway sent them, else the request's ``Host`` and scheme) and ``X-Forwarded-Prefix``, the
  route prefix (``/svc/{service}`` or ``/website/{service}``) the gateway strips, which a client never sets: a client
  that serves the route at a root of its own (a computer's site proxy, at ``https://notes.test/``) says so with
  ``AgentEnv-Site-Root``, and the upstream is then told no prefix;
- the upstream response and the client it came through are closed when the response ends, iterated or not.

A WebSocket on ``/website/{service}/{path}`` is relayed under the same rules (:func:`relay_websocket`): the gateway
opens one of its own to the service, with the headers an HTTP request would carry upstream, and relays each message
and the close, either side's ending ending both.

A backing server's env card names its endpoints relative to itself; :func:`rewrite_child_card` roots them under
``/svc/{key}`` so they resolve against the env's address, keeping the gateway's own routes where the gateway offers
the same operation. A website's card names its routes on ``urn:agentenv:website/v1``; :func:`website_routes` reads
them so :func:`route_name` can name the route a proxied request reached.
"""
from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import posixpath
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, NamedTuple
from urllib.parse import unquote, urlsplit

import httpx
from starlette.requests import HTTPConnection, Request
from starlette.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketState
from websockets.asyncio.client import ClientConnection, connect as connect_websocket
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus

# Relative import — gateway runs as a standalone container with no agent_env.*
from .constants import (
    AGENT_ENV_ROLE_HEADER,
    AGENT_ENV_ROLE_META_KEY,
    AGENT_ENV_SESSION_META_KEY,
    AGENT_ENV_SITE_ROOT_HEADER,
    CARD_FETCH_TIMEOUT_S,
    EXT_WEBSITE_URI,
    WELL_KNOWN_PATH,
)

logger = logging.getLogger(__name__)

# HTTP (RFC 7230) hop-by-hop headers, never proxied in either direction.
HOP_BY_HOP = frozenset({
    b"host", b"connection", b"keep-alive", b"transfer-encoding",
    b"te", b"trailer", b"upgrade", b"proxy-authorization", b"proxy-authenticate",
})

# The role header as it appears in the raw ASGI header list (lower-cased by the server).
ROLE_HEADER_BYTES = AGENT_ENV_ROLE_HEADER.lower().encode()

FORWARDED_HOST, FORWARDED_PROTO, FORWARDED_PREFIX = "x-forwarded-host", "x-forwarded-proto", "x-forwarded-prefix"
_FORWARDED = frozenset(name.encode() for name in (FORWARDED_HOST, FORWARDED_PROTO, FORWARDED_PREFIX,
                                                   AGENT_ENV_SITE_ROOT_HEADER.lower()))
# A WebSocket's handshake fields, which the gateway's own connection upstream writes for itself.
_WEBSOCKET_HANDSHAKE = frozenset({"sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions",
                                  "sec-websocket-protocol"})
# What the gateway's own WebSocket handshake answer writes for itself, so a site's copies aren't relayed.
_WEBSOCKET_ANSWER_DROP = HOP_BY_HOP | {b"date", b"server", b"content-length", b"sec-websocket-accept",
                                      b"sec-websocket-protocol", b"sec-websocket-extensions"}
_WEBSOCKET_REFUSAL_DROP = HOP_BY_HOP | {b"date", b"server", b"content-length"}
_HTTP_SCHEME = {"ws": "http", "wss": "https"}
WEBSOCKET_MESSAGE_MAX_BYTES = 16 * 1024 * 1024


async def forward(request: Request, base_url: str, *, role: str, timeout: httpx.Timeout, mcp_server: bool,
                  on_done: Callable[[int], Awaitable[None]] | None = None,
                  withheld: frozenset[bytes] = frozenset()) -> Response:
    """Forward `request` to the service at `base_url` as `role`, without the headers named in `withheld`, and stream
    the response through. `mcp_server` says
    the service is a backing MCP server, whose own endpoint takes the role in its JSON-RPC `_meta` as well as in the
    header. A 400 names a request-target the proxy cannot route; a 502 names an upstream that never answered.
    `on_done` is awaited with the status once the response has ended, after its last byte when streamed."""
    try:
        target_url = upstream_url(base_url, request)
    except ValueError as e:  # an escaped slash inside the route prefix; refuse rather than forward a wrong path
        return await _done(_json_response({"error": str(e)}, 400), on_done)
    content = await request.body()  # /svc and /website request bodies are small (JSON-RPC, forms)
    headers = upstream_headers(request, role, withheld=withheld)
    if mcp_server and request.method == "POST" and is_child_mcp_path(request.path_params.get("path", "")):
        # The gateway is the trust boundary for identity: a child's proxied MCP endpoint stays reachable, but the
        # role in its JSON-RPC `_meta` is the gateway's to write, like the header above.
        content = stamp_jsonrpc_role(content, role)
        headers["content-length"] = str(len(content))
    client = httpx.AsyncClient(timeout=timeout)
    upstream_request = client.build_request(request.method, target_url, content=content, headers=headers)
    try:
        upstream = await client.send(upstream_request, stream=True)  # returns once the headers are in
    except Exception as e:
        await client.aclose()
        logger.error(f"Proxy to {target_url} failed: {e!r}")
        return await _done(_json_response({"error": f"Proxy request failed: {e!r}"}, 502), on_done)
    return UpstreamResponse(upstream, client, target_url, on_done)


async def _done(response: Response, on_done: Callable[[int], Awaitable[None]] | None) -> Response:
    if on_done is not None:
        await on_done(response.status_code)
    return response


def upstream_url(base_url: str, request: HTTPConnection) -> str:
    # Forward the path as received: uvicorn decodes scope["path"] for routing, so Starlette's {path} param
    # would send %2F, %3F and %23 to the upstream as separators. latin-1 keeps one char per wire byte.
    url = base_url
    decoded_rest = request.path_params.get("path", "")
    if decoded_rest:
        raw_rest = raw_remainder(request.scope["raw_path"], request.scope["path"], decoded_rest)
        url += "/" + raw_rest.decode("latin-1")
    query = request.scope.get("query_string", b"")
    if query:
        url += "?" + query.decode("latin-1")
    return url


def client_headers(connection: HTTPConnection, withheld: frozenset[bytes] = frozenset()) -> list[tuple[bytes, bytes]]:
    """What any route of the gateway may pass on of a connection's headers: one entry per raw header, so repeated
    fields (cookie, x-forwarded-for) keep their multiplicity, minus the hop-by-hop set and the headers named in
    `withheld` (lower case), the credentials of the provider in front of the gateway, so no service behind it reads
    them."""
    return [(k, v) for k, v in connection.headers.raw if k.lower() not in HOP_BY_HOP and k.lower() not in withheld]


def upstream_headers(request: HTTPConnection, role: str, *, withheld: frozenset[bytes] = frozenset()) -> httpx.Headers:
    """The client's headers (``client_headers``) as a proxied request sends them. The role header is the gateway's to
    set: the client's copies are dropped and the role the gateway attributed the request to goes in their place, so
    the upstream never reads a value the gateway did not write. So are the forwarded-address headers
    (`forwarded_address`)."""
    headers = httpx.Headers([(k, v) for k, v in client_headers(request, withheld)
                             if k.lower() != ROLE_HEADER_BYTES and k.lower() not in _FORWARDED])
    headers[AGENT_ENV_ROLE_HEADER] = role
    headers.update(forwarded_address(request))
    # Bytes are forwarded undecoded, so only ask upstream for encodings the caller can take.
    headers.setdefault("accept-encoding", "identity")
    return headers


def forwarded_address(request: HTTPConnection) -> dict[str, str]:
    """Where the client reached the proxied service: the host and scheme it addressed (as a front before the gateway
    names them in its own ``X-Forwarded-Host``/``X-Forwarded-Proto``, else the request's ``Host`` and scheme; the
    client could set ``Host`` anyway, so believing its front names nothing it couldn't), and the route prefix as sent,
    or none for a client that says with ``AgentEnv-Site-Root`` it serves the route at a root of its own: a computer's
    site proxy, whose browser would otherwise get the route's prefix twice. A ``AgentEnv-Site-Root`` naming an origin
    (``https://notes.test``) is that root's address, ahead of any ``X-Forwarded-Host``/``-Proto``, which a front
    between the client and the gateway may have rewritten to its own. Like the host, that only changes the links the
    client's own answer names; what the route reaches is the path's alone. A header with nothing to say is left
    out."""
    root = request.headers.get(AGENT_ENV_SITE_ROOT_HEADER, "").strip()
    origin = _origin(root)
    host = origin[1] if origin else _first(request.headers.get(FORWARDED_HOST, "")) or request.headers.get("host", "")
    proto = origin[0] if origin else _first(request.headers.get(FORWARDED_PROTO, "")).lower()
    at_root = bool(root)
    scheme = _HTTP_SCHEME.get(request.url.scheme, request.url.scheme)
    address = {FORWARDED_HOST: host, FORWARDED_PROTO: proto if proto in ("http", "https") else scheme,
               FORWARDED_PREFIX: "" if at_root else route_prefix(request)}
    return {name: value for name, value in address.items() if value}


def _origin(value: str) -> tuple[str, str] | None:
    """The scheme and host of ``value`` when it is an origin (``https://notes.test``, a port allowed), else None."""
    parts = urlsplit(value)
    if (parts.scheme in ("http", "https") and parts.netloc and parts.path in ("", "/") and not parts.query
            and not parts.fragment and "@" not in parts.netloc and value.isprintable()):
        return parts.scheme, parts.netloc
    return None


def _first(value: str) -> str:
    return value.split(",")[0].strip()


def route_prefix(request: HTTPConnection) -> str:
    """The path the client reached the proxied service under, as it sent it: the request-target up to the `{path}`
    remainder, without its trailing slash, e.g. ``/website/notes``, or under a root path ``/env/website/notes``."""
    raw_path = request.scope.get("raw_path") or request.scope["path"].encode()
    decoded_rest = request.path_params.get("path", "")
    if decoded_rest:
        raw_path = raw_path[:-len(raw_remainder(raw_path, request.scope["path"], decoded_rest))]
    return raw_path.decode("latin-1").rstrip("/")


def is_child_mcp_path(decoded_rest: str) -> bool:
    """Whether a /svc/{service}/{path} remainder reaches the backing server's MCP endpoint (`mcp` or anything under
    it), judged after dot-segment normalisation, since httpx collapses `x/../mcp` to `/mcp` on the way upstream.

    >>> [is_child_mcp_path(p) for p in ("mcp", "mcp/", "mcp/x", "x/../mcp", "./mcp", "/mcp", "mcpx", "agentenv", "")]
    [True, True, True, True, True, True, False, False, False]
    """
    normalized = posixpath.normpath("/" + decoded_rest.lstrip("/"))
    return normalized == "/mcp" or normalized.startswith("/mcp/")


def stamp_jsonrpc_role(body: bytes, role: str) -> bytes:
    """A JSON-RPC body (one request or a batch) with every request's `params._meta` carrying the gateway's role
    stamp in place of whatever the client wrote there; a body that is not JSON-RPC is returned as it came.

    >>> json.loads(stamp_jsonrpc_role(b'{"method": "tools/call", "params": {"_meta": {"agentenv.io/role": "x", "agentenv.io/session": "s"}}}', "r"))["params"]
    {'_meta': {'agentenv.io/role': 'r'}}
    >>> stamp_jsonrpc_role(b"not json", "r")
    b'not json'
    """
    try:
        parsed = json.loads(body)
    except ValueError:
        return body
    stamped = False
    for item in parsed if isinstance(parsed, list) else [parsed]:
        params = item.get("params") if isinstance(item, dict) else None
        if isinstance(params, dict):
            meta = dict(params["_meta"]) if isinstance(params.get("_meta"), dict) else {}
            meta.pop(AGENT_ENV_SESSION_META_KEY, None)
            meta[AGENT_ENV_ROLE_META_KEY] = role
            params["_meta"] = meta
            stamped = True
    return json.dumps(parsed).encode() if stamped else body


def raw_remainder(raw_path: bytes, decoded_path: str, decoded_rest: str) -> bytes:
    """The still-encoded tail of `raw_path` that Starlette's decoded `{path}` param came from.

    >>> raw_remainder(b"/svc/mcp-svc/x%2Fy%3Fz", "/svc/mcp-svc/x/y?z", "x/y?z")
    b'x%2Fy%3Fz'
    """
    prefix = decoded_path.removesuffix(decoded_rest)  # "/svc/mcp-svc/"
    rest = raw_path.split(b"/", prefix.count("/"))[-1]  # 3 slashes in both forms, so the tail starts after the 3rd
    if unquote(rest.decode("latin-1")) != decoded_rest:  # only false when an escaped slash hid inside the prefix
        raise ValueError(f"cannot route request-target {raw_path.decode('latin-1')!r}: percent-encoded slash in the route prefix")
    return rest


async def relay_websocket(websocket: WebSocket, base_url: str, *, role: str,
                          open_timeout: float = 10.0, withheld: frozenset[bytes] = frozenset()) -> None:
    """Relay `websocket` to the service at `base_url` as `role`: a WebSocket of the gateway's own is opened to the same
    path, with the headers ``upstream_headers`` sends, asking for the subprotocols the client offered; `websocket` is
    accepted with the one the service chose and the headers its answer carried (a ``Set-Cookie`` among them, each
    field kept apart), but those the gateway's own answer writes. Each text and binary message is relayed as it came, and either side's
    close, with its code, closes the other; a message from the service larger than ``WEBSOCKET_MESSAGE_MAX_BYTES``, the
    size uvicorn takes from a client, closes both with 1009 rather than be held whole. A service that refuses the
    handshake has its answer relayed as the refusal, its headers as an accepted one's are; one that never answers is a
    502. The headers named in `withheld` don't go upstream, as with ``upstream_headers``."""
    try:
        target_url = upstream_url(base_url, websocket)
    except ValueError as e:
        await refuse_websocket(websocket, _json_response({"error": str(e)}, 400))
        return
    headers = [(key, value) for key, value in upstream_headers(websocket, role, withheld=withheld).multi_items()
               if key.lower() not in _WEBSOCKET_HANDSHAKE]
    offered = [protocol.strip() for value in websocket.headers.getlist("sec-websocket-protocol")
               for protocol in value.split(",") if protocol.strip()]
    try:
        upstream = await connect_websocket("ws" + target_url.removeprefix("http"), additional_headers=headers,
                                           subprotocols=offered or None, user_agent_header=None,
                                           open_timeout=open_timeout, max_size=WEBSOCKET_MESSAGE_MAX_BYTES)
    except InvalidStatus as e:
        refusal = Response(content=bytes(e.response.body or b""), status_code=e.response.status_code)
        refusal.raw_headers = [*(item for item in refusal.raw_headers if item[0] == b"content-length"),
                               *_answer_headers(e.response.headers, drop=_WEBSOCKET_REFUSAL_DROP)]
        await refuse_websocket(websocket, refusal)
        return
    except (InvalidHandshake, OSError, TimeoutError) as e:
        logger.error(f"WebSocket proxy to {target_url} failed: {e!r}")
        await refuse_websocket(websocket, _json_response({"error": f"Proxy request failed: {e!r}"}, 502))
        return
    async with upstream:
        await websocket.accept(subprotocol=upstream.subprotocol,
                               headers=_answer_headers(upstream.response.headers, drop=_WEBSOCKET_ANSWER_DROP))
        await _relay_messages(websocket, upstream)


def _answer_headers(headers: Any, *, drop: frozenset[bytes]) -> list[tuple[bytes, bytes]]:
    """A site's handshake answer's headers as the gateway relays them, each field apart (a ``Set-Cookie`` per cookie),
    but those named in `drop`, which the gateway's own answer writes."""
    return [(key.lower(), value) for key, value in
            ((key.encode("ascii", "surrogateescape"), value.encode("ascii", "surrogateescape"))
             for key, value in headers.raw_items()) if key.lower() not in drop]


async def refuse_websocket(websocket: WebSocket, response: Response) -> None:
    """Refuse `websocket`'s handshake with `response`, or, on a server that can't send one, a 403."""
    if "websocket.http.response" in websocket.scope.get("extensions", {}):
        await websocket.send_denial_response(response)
    else:
        await websocket.close(1008)


async def _relay_messages(websocket: WebSocket, upstream: ClientConnection) -> None:
    """Both directions of an accepted WebSocket until either ends, then both closed: the client's close goes upstream
    with its code, and the service's to the client."""

    async def from_client() -> None:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                await upstream.close(_sendable(message.get("code"), abnormal=1001), message.get("reason") or "")
                return
            if message.get("text") is not None:
                await upstream.send(message["text"])
            elif message.get("bytes") is not None:
                await upstream.send(message["bytes"])

    async def from_upstream() -> None:
        async for data in upstream:
            if isinstance(data, str):
                await websocket.send_text(data)
            else:
                await websocket.send_bytes(data)

    tasks = [asyncio.create_task(from_client()), asyncio.create_task(from_upstream())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await upstream.close()
        if WebSocketState.DISCONNECTED not in (websocket.client_state, websocket.application_state):
            sent = upstream.protocol.close_sent
            code = sent.code if sent is not None and sent.code == 1009 else upstream.close_code
            with contextlib.suppress(RuntimeError, OSError, ConnectionClosed):
                await websocket.close(_sendable(code, abnormal=1011), upstream.close_reason or "")


def _sendable(code: int | None, *, abnormal: int) -> int:
    """A close code one side reported, as one the other may be sent: none (1005) as a normal close, and a closure
    with no close frame (1006, 1015) or a code no endpoint may send as `abnormal`.

    >>> [_sendable(c, abnormal=1011) for c in (None, 1005, 1000, 1006, 4001, 1004)]
    [1000, 1000, 1000, 1011, 4001, 1011]
    """
    if code in (None, 1005):
        return 1000
    if 1000 <= code <= 1003 or 1007 <= code <= 1014 or 3000 <= code <= 4999:
        return code
    return abnormal


class UpstreamResponse(StreamingResponse):
    """Streams an httpx response through untouched and closes it when the response ends, iterated or not.

    The close lives in `__call__` rather than the body generator because a client that leaves before the first
    chunk makes Starlette cancel before the generator's first step, and a never-started generator runs no `finally`."""

    _DROP = HOP_BY_HOP | {b"date", b"server"}  # uvicorn adds the gateway's own; forwarding upstream's duplicates them

    def __init__(self, upstream: httpx.Response, client: httpx.AsyncClient, target_url: str,
                 on_done: Callable[[int], Awaitable[None]] | None = None) -> None:
        super().__init__(self._body(upstream, target_url), status_code=upstream.status_code)
        # Header block forwarded byte for byte: repeated names (Set-Cookie) stay separate, non-latin-1 values
        # never hit a str round-trip, and Content-Length/Content-Encoding stay valid for the undecoded body.
        self.raw_headers = [(k.lower(), v) for k, v in upstream.headers.raw if k.lower() not in self._DROP]
        self._upstream = upstream
        self._client = client
        self._on_done = on_done

    @staticmethod
    async def _body(upstream: httpx.Response, target_url: str) -> AsyncIterator[bytes]:
        sent = 0
        try:
            async for chunk in upstream.aiter_raw():
                sent += len(chunk)
                yield chunk
        except Exception as e:  # past the headers a failure can only reach the client as a cut body; name it here
            logger.error(f"Proxy to {target_url} failed after {sent} bytes: {e!r}")
            raise

    def after_body(self, then: Callable[[int], Awaitable[Any]]) -> None:
        """Await `then` with the status once the response has ended, after `on_done`."""
        on_done = self._on_done

        async def both(status: int) -> None:
            if on_done is not None:
                await on_done(status)
            await then(status)
        self._on_done = both

    async def read_ahead_if_sized(self, max_bytes: int) -> None:
        """Read a body sized at most `max_bytes` to its end now and run `on_done` there, so the response then replays it."""
        size = self._upstream.headers.get("content-length", "")
        if self._on_done is None or not size.isdigit() or int(size) > max_bytes:
            return
        chunks: list[bytes] = []
        failure: Exception | None = None
        try:
            async for chunk in self.body_iterator:
                chunks.append(chunk)
        except Exception as e:  # already logged by `_body`; the client still gets the cut body when it is sent
            failure = e
        except BaseException:  # cancelled before `__call__`, which would otherwise close them, ever runs
            await self._upstream.aclose()
            await self._client.aclose()
            raise
        self.body_iterator = _replay(chunks, failure)
        on_done, self._on_done = self._on_done, None
        await on_done(self.status_code)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:  # normal end, upstream error, or the downstream client going away
            await self._upstream.aclose()
            await self._client.aclose()
            if self._on_done is not None:
                await self._on_done(self.status_code)


async def _replay(chunks: list[bytes], failure: Exception | None) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk
    if failure is not None:
        raise failure


async def fetch_child_card(client: httpx.AsyncClient, key: str, base_url: str) -> dict | None:
    """One backing server's card as it serves it; None (omit) on 404 / error / timeout / malformed JSON."""
    url = f"{base_url.rstrip('/')}{WELL_KNOWN_PATH}"
    try:
        response = await client.get(url, timeout=CARD_FETCH_TIMEOUT_S)
    except Exception as e:
        logger.warning(f"env card: fetching {key} failed ({type(e).__name__}: {e}); omitting")
        return None
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        logger.warning(f"env card: {key} returned HTTP {response.status_code}; omitting")
        return None
    try:
        return response.json()
    except Exception:
        logger.warning(f"env card: {key} returned malformed JSON; omitting")
        return None


def rewrite_child_card(key: str, card: dict, gateway_extensions: list[dict]) -> dict:
    """Rewrite every path the child names (its `url`, each extension's endpoint and each method's
    own) to the gateway-rooted `/svc/{key}/...`, so it resolves against the env's address. An
    operation the gateway's own extension with the same `uri` also offers (same method name, HTTP
    verb and endpoint, e.g. `disable`, POST `/tools/disable`) is the gateway's route and keeps its
    path; a child method falling back to that endpoint gets one of its own under the child's prefix. Backing
    servers are leaves, so any nested `children_environments` is dropped, and so are the child's
    interfaces: agents reach its tools through the gateway's aggregate MCP endpoint."""
    gateway_operations, gateway_requests, gateway_paths = set(), set(), set()
    for gateway_ext in gateway_extensions:
        for name, _, _, verb, path in operations(gateway_ext.get("params") or {}):
            gateway_operations.add((gateway_ext["uri"], name, verb, path))
            gateway_requests.add((gateway_ext["uri"], verb, path))
            gateway_paths.add((gateway_ext["uri"], path))

    def child_path(path: str) -> str:
        return f"/svc/{key}{path}" if path.startswith("/") else path

    card = copy.deepcopy(card)
    if card.get("children_environments"):
        logger.warning(f"env card: dropping nested children_environments from leaf {key} (multi-level composition unsupported)")
    card["children_environments"] = None
    card["additionalInterfaces"] = []
    url = card.get("url")
    if isinstance(url, str) and url:
        card["url"] = child_path(url.rstrip("/") or "/")
    for ext in (card.get("capabilities") or {}).get("extensions") or []:
        params = ext.get("params") if isinstance(ext, dict) else None
        if not isinstance(params, dict):
            continue
        uri = ext["uri"] if isinstance(ext.get("uri"), str) else None
        endpoint = params.get("endpoint") if isinstance(params.get("endpoint"), str) else None
        # A method is the gateway's when the gateway offers it under the same name; an extension
        # listing no methods is a bare request, the gateway's when the gateway serves it.
        ops = [(method, holder, (uri, name, verb, path) in gateway_operations
                if name is not None else (uri, verb, path) in gateway_requests)
               for name, method, holder, verb, path in operations(params)]
        # The extension's endpoint stays when a gateway operation falls back to it or, used by no
        # operation, when the gateway's extension with this uri names it.
        keep_endpoint = any(on_gateway and holder is params for _, holder, on_gateway in ops) or (
            all(holder is not params for _, holder, _ in ops) and (uri, endpoint) in gateway_paths)
        for method, holder, on_gateway in ops:
            if on_gateway:
                continue
            if holder is not params:
                holder["endpoint"] = child_path(holder["endpoint"])
            elif keep_endpoint and method is not None:
                method["endpoint"] = child_path(params["endpoint"])
        if endpoint is not None and not keep_endpoint:
            params["endpoint"] = child_path(endpoint)
    return card


def operations(params: dict) -> list[tuple[str | None, dict | None, dict, str, str]]:
    """Each operation an extension offers, read the way `invoke_extension` reads it: (method name, method, the
    object naming its endpoint, HTTP verb, endpoint). A method without an endpoint of its own uses
    the extension's, its verb defaults to POST, and an extension listing no methods offers a POST
    to its endpoint. One without a text endpoint is skipped, so it stays as stored."""
    methods = params.get("methods")
    if isinstance(methods, dict) and methods:
        found = [(name, method, method if method.get("endpoint") else params, str(method.get("method") or "POST").upper())
                 for name, method in methods.items() if isinstance(method, dict)]
    else:
        found = [(None, None, params, "POST")]
    return [(name, method, holder, verb, holder["endpoint"]) for name, method, holder, verb in found
            if isinstance(holder.get("endpoint"), str) and holder["endpoint"]]


class WebsiteRoute(NamedTuple):
    """One route a website's card names: its verb, the pattern its path matches (a Starlette converter's own, one
    path segment for any other converter or none), and its name, None for a route the site builds in."""

    method: str
    pattern: re.Pattern
    name: str | None


_CONVERTERS = {"path": ".*", "int": "[0-9]+", "float": r"[0-9]+(?:\.[0-9]+)?",
               "uuid": "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"}
_PATH_PARAM = re.compile(r"{\w+(?::(\w+))?}")


def website_routes(card: dict | None) -> list[WebsiteRoute]:
    """The routes a website's card names on ``urn:agentenv:website/v1``, in its order: each one's method, its path
    (``/api/notes/{note_id}``, with or without a Starlette converter such as ``{note_id:int}``) and its name. An entry
    missing any of the three is skipped; a card naming none, or no card, has none.

    Before them come, unnamed, the built-in routes the card names, which answer before any of its routes would: a GET
    to its ``health`` path, and a GET or DELETE to its ``session_path`` unless one of its routes is at that very path
    (then the site serves its own there). A request one of them answers is named by no route.

    >>> [(r.method, r.pattern.pattern, r.name) for r in website_routes({"capabilities": {"extensions": [
    ...     {"uri": "urn:agentenv:website/v1", "params": {"health": "/healthz", "routes": [
    ...         {"method": "get", "path": "/api/notes/{note_id:int}", "name": "get_note"}, {"path": "/x"}]}}]}})]
    [('GET', '/healthz', None), ('GET', '/api/notes/(?:[0-9]+)', 'get_note')]
    """
    found: list[WebsiteRoute] = []
    for ext in ((card or {}).get("capabilities") or {}).get("extensions") or []:
        params = ext.get("params") if isinstance(ext, dict) and ext.get("uri") == EXT_WEBSITE_URI else None
        routes = params.get("routes") if isinstance(params, dict) else None
        named = [route for route in (routes if isinstance(routes, list) else [])
                 if isinstance(route, dict) and all(isinstance(route.get(k), str) and route[k]
                                                    for k in ("method", "path", "name"))]
        health, session = (params.get(k) if isinstance(params, dict) else None for k in ("health", "session_path"))
        if isinstance(health, str) and health:
            found.append(WebsiteRoute("GET", _path_pattern(health), None))
        if isinstance(session, str) and session and all(route["path"] != session for route in named):
            found += [WebsiteRoute(verb, _path_pattern(session), None) for verb in ("GET", "DELETE")]
        found += [WebsiteRoute(route["method"].upper(), _path_pattern(route["path"]), route["name"]) for route in named]
    return found


def _path_pattern(path: str) -> re.Pattern:
    parts, end = [], 0
    for param in _PATH_PARAM.finditer(path):
        parts += [re.escape(path[end:param.start()]), f"(?:{_CONVERTERS.get(param.group(1) or '', '[^/]+')})"]
        end = param.end()
    return re.compile("".join(parts) + re.escape(path[end:]))


def route_name(routes: list[WebsiteRoute], method: str, path: str) -> str | None:
    """The name of the first of `routes` a `method` request to `path` reaches (a HEAD reaches a GET route), else None,
    as it is when that first one is a built-in route.

    >>> route_name(website_routes({"capabilities": {"extensions": [{"uri": "urn:agentenv:website/v1", "params": {
    ...     "routes": [{"method": "GET", "path": "/api/notes/{note_id}", "name": "get_note"}]}}]}}), "HEAD", "/api/notes/7")
    'get_note'
    """
    verbs = {method.upper(), "GET"} if method.upper() == "HEAD" else {method.upper()}
    return next((route.name for route in routes if route.method in verbs and route.pattern.fullmatch(path)), None)


def _json_response(data: dict | list, status_code: int) -> Response:
    return Response(content=json.dumps(data), status_code=status_code, media_type="application/json")
