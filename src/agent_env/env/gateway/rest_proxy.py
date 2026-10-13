"""The gateway's REST reverse proxy: what it forwards to a backing server, and how the backing servers' cards name
the proxied routes.

A request to ``/svc/{service}/{path}`` or ``/website/{service}/{path}`` is forwarded to the service's base URL and
its response streamed through, under four invariants:

- the path goes upstream as received: uvicorn decodes ``scope["path"]`` for routing, so the still-encoded tail is
  recovered from ``raw_path`` and ``%2F``, ``%3F`` and ``%23`` reach the upstream as data, not separators;
- headers are forwarded one entry per raw header (repeated fields keep their multiplicity, non-latin-1 values never
  round-trip through str), minus the hop-by-hop set, in both directions;
- the role header is the gateway's to write: the client's copies are dropped and the role the gateway attributed
  the request to goes in their place, and a POST to a backing MCP server's own endpoint gets the same treatment
  inside its JSON-RPC ``params._meta``;
- the upstream response and the client it came through are closed when the response ends, iterated or not.

A backing server's env card names its endpoints relative to itself; :func:`rewrite_child_card` roots them under
``/svc/{key}`` so they resolve against the env's address, keeping the gateway's own routes where the gateway offers
the same operation.
"""
from __future__ import annotations

import copy
import json
import logging
import posixpath
from collections.abc import AsyncIterator
from urllib.parse import unquote

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

# Relative import — gateway runs as a standalone container with no agent_env.*
from .constants import (
    AGENT_ENV_ROLE_HEADER,
    AGENT_ENV_ROLE_META_KEY,
    AGENT_ENV_SESSION_META_KEY,
    CARD_FETCH_TIMEOUT_S,
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


async def forward(request: Request, base_url: str, *, role: str, timeout: httpx.Timeout, mcp_server: bool) -> Response:
    """Forward `request` to the service at `base_url` as `role` and stream the response through. `mcp_server` says
    the service is a backing MCP server, whose own endpoint takes the role in its JSON-RPC `_meta` as well as in the
    header. A 400 names a request-target the proxy cannot route; a 502 names an upstream that never answered."""
    try:
        target_url = upstream_url(base_url, request)
    except ValueError as e:  # an escaped slash inside the route prefix; refuse rather than forward a wrong path
        return _json_response({"error": str(e)}, 400)
    content = await request.body()  # /svc and /website request bodies are small (JSON-RPC, forms)
    headers = upstream_headers(request, role)
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
        return _json_response({"error": f"Proxy request failed: {e!r}"}, 502)
    return UpstreamResponse(upstream, client, target_url)


def upstream_url(base_url: str, request: Request) -> str:
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


def upstream_headers(request: Request, role: str) -> httpx.Headers:
    """One entry per raw header, so repeated fields (cookie, x-forwarded-for) keep their multiplicity. The role
    header is the gateway's to set: the client's copies are dropped and the role the gateway attributed the
    request to goes in their place, so the upstream never reads a value the gateway did not write."""
    headers = httpx.Headers([(k, v) for k, v in request.headers.raw
                             if k.lower() not in HOP_BY_HOP and k.lower() != ROLE_HEADER_BYTES])
    headers[AGENT_ENV_ROLE_HEADER] = role
    # Bytes are forwarded undecoded, so only ask upstream for encodings the caller can take.
    headers.setdefault("accept-encoding", "identity")
    return headers


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


class UpstreamResponse(StreamingResponse):
    """Streams an httpx response through untouched and closes it when the response ends, iterated or not.

    The close lives in `__call__` rather than the body generator because a client that leaves before the first
    chunk makes Starlette cancel before the generator's first step, and a never-started generator runs no `finally`."""

    _DROP = HOP_BY_HOP | {b"date", b"server"}  # uvicorn adds the gateway's own; forwarding upstream's duplicates them

    def __init__(self, upstream: httpx.Response, client: httpx.AsyncClient, target_url: str) -> None:
        super().__init__(self._body(upstream, target_url), status_code=upstream.status_code)
        # Header block forwarded byte for byte: repeated names (Set-Cookie) stay separate, non-latin-1 values
        # never hit a str round-trip, and Content-Length/Content-Encoding stay valid for the undecoded body.
        self.raw_headers = [(k.lower(), v) for k, v in upstream.headers.raw if k.lower() not in self._DROP]
        self._upstream = upstream
        self._client = client

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

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:  # normal end, upstream error, or the downstream client going away
            await self._upstream.aclose()
            await self._client.aclose()


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


def _json_response(data: dict | list, status_code: int) -> Response:
    return Response(content=json.dumps(data), status_code=status_code, media_type="application/json")
