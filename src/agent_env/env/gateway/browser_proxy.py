"""The gateway's CDP route: ``/browser/<key>/cdp/<token>/...`` reaches the website browser behind the ``/svc`` child
``<key>``, its discovery and its WebSocket both, and every CDP address the browser hands out, in discovery or in its
``cdp`` answer, comes back naming the gateway at the address the caller reached it at.

The browser builds those addresses from the address its own server was reached at, so behind the gateway they would
name the browser's host and a client following one would leave the gateway. The gateway rewrites them by path,
replacing everything before ``/cdp/<token>/`` with its own address and ``/browser/<key>``, so a rewrite holds whatever
host an address was minted at. It keeps the https or wss of an address the browser minted under an origin its caller
named, since a tunnel in front of the gateway that ends TLS may not say how it was reached."""
from __future__ import annotations

import asyncio
import contextlib
import logging
import posixpath
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlsplit

import httpx
import websockets
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets.asyncio.client import ClientConnection, connect

# Relative import — gateway runs as a standalone container with no agent_env.*
from .constants import EXT_BROWSER_URI, EXT_COMPUTER_APP_URI
from .rest_proxy import client_headers, fetch_child_card, operations, upstream_headers, upstream_url

logger = logging.getLogger(__name__)

#: The route under a browser's own that its CDP addresses sit under, which its card opens to agents.
CDP_ROUTE = "/cdp/"
#: How often each leg of a relayed connection is pinged, so an idle one outlives a tunnel's idle timeout.
PING_INTERVAL_S = 20.0
#: The fields of a discovery answer, ``json/version`` or each ``json/list`` entry, that carry a CDP address.
DISCOVERY_FIELDS = frozenset({"webSocketDebuggerUrl", "devtoolsFrontendUrl"})
#: The fields of a browser's ``cdp`` answer that do.
CDP_ANSWER_FIELDS = frozenset({"http_url", "ws_url"})
#: A Host header that names a host, and its port or not, and nothing else.
HOST = re.compile(r"(?:[A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+\])(?::[0-9]{1,5})?")
#: A CDP address's path from its token on.
ADDRESS = re.compile(r"/cdp/[^/?#]+(?:/[^?#]*)?$")
#: A CDP address's token, which no log line shows.
TOKEN = re.compile(r"(/cdp/)[^/\s?#\"']+")
#: A WebSocket close code that means the connection ended without one, which a close can't send on.
UNSENDABLE_CLOSES = (1005, 1006, 1015)
#: A close for a connection the browser can't be found for, or refuses.
REFUSED = 1008
#: A close for a connection the browser didn't answer, or that failed in between.
UNREACHABLE = 1011


@dataclass(frozen=True)
class Browser:
    """A website browser behind a ``/svc`` child: where its routes sit under the child's (``prefix``), and the path
    of its ``cdp`` method, as the child's card names them."""

    prefix: str
    cdp_path: str


def browsers(card: dict) -> list[Browser]:
    """The website browsers a ``/svc`` child's card, as the child serves it, offers: the child itself, or each app
    among its children, as a computer's are, its routes under ``/svc/<app>``. One counts only when its card opens
    its CDP route to agents, the only one the gateway relays without the child's control token."""
    offered = [("", card)] + [(f"/svc/{child['name']}", child) for child in card.get("children_environments") or []
                              if isinstance(child, dict) and isinstance(child.get("name"), str)]
    found = []
    for prefix, browser_card in offered:
        cdp_path = _method_path(browser_card, EXT_BROWSER_URI, "cdp")
        if cdp_path is not None and CDP_ROUTE in (_params(browser_card, EXT_COMPUTER_APP_URI).get("agent_routes") or []):
            found.append(Browser(prefix, cdp_path))
    return found


def at_gateway(url: str, base: str, key: str) -> str:
    """``url``, a CDP address a browser handed out, at the gateway: everything before its ``/cdp/<token>/`` becomes
    ``base`` and ``/browser/<key>``, a WebSocket URL staying one, secure when either is: a browser the gateway reaches
    over plain http names https only under an origin its caller named. A value that isn't such an address is kept."""
    parts = urlsplit(url)
    match = ADDRESS.search(parts.path)
    if parts.scheme not in ("http", "https", "ws", "wss") or match is None:
        return url
    secure = base.startswith("https://") or parts.scheme in ("https", "wss")
    host = base.split("://", 1)[1]
    scheme = ("wss" if secure else "ws") if parts.scheme in ("ws", "wss") else ("https" if secure else "http")
    return f"{scheme}://{host}/browser/{key}{match.group(0)}" + (f"?{parts.query}" if parts.query else "")


def rewritten(body: Any, fields: frozenset[str], base: str, key: str) -> Any:
    """A JSON answer, an object or a list of them, with each of ``fields`` holding an address at the gateway."""
    if isinstance(body, list):
        return [rewritten(item, fields, base, key) for item in body]
    if isinstance(body, dict):
        return {name: at_gateway(value, base, key) if name in fields and isinstance(value, str) else value
                for name, value in body.items()}
    return body


def public_base(connection: Request | WebSocket) -> str | None:
    """The address the client reached the gateway at, scheme, host and port: the Host it sent, with the scheme a
    TLS-terminating tunnel in front says it was reached by (``X-Forwarded-Proto``), else the connection's own.
    None for a Host that isn't a host and port. A rewrite with it goes back only to the client that sent it."""
    host = connection.headers.get("host", "")
    if not HOST.fullmatch(host):
        return None
    proto = connection.headers.get("x-forwarded-proto", "").split(",")[0].strip().lower()
    if proto not in ("http", "https"):
        proto = "https" if connection.url.scheme in ("https", "wss") else "http"
    return f"{proto}://{host}"


class TokenRedaction(logging.Filter):
    """Takes the token out of each CDP address a log line names: a request's path, or a URL forwarded to."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if CDP_ROUTE in message:
            record.msg, record.args = TOKEN.sub(r"\1<redacted>", message), ()
        return True


class BrowserRoutes:
    """The ``/browser/<key>/cdp/<token>/`` route over a gateway's ``/svc`` children, and the rewrite of a browser's
    ``cdp`` answer on their ``/svc`` route. ``log_event`` writes a trajectory event, ``attribution`` names a role's
    caller for one, and ``role_of`` reads the role a connection's ``AgentEnv-Role`` header names. The headers named in
    ``withheld`` (lower case), the sandbox provider's credentials, reach no browser."""

    def __init__(self, rest_proxy_urls: dict[str, str], timeout: httpx.Timeout,
                 log_event: Callable[[dict], Awaitable[Any]], attribution: Callable[[str], dict],
                 role_of: Callable[[Request | WebSocket], str], *, withheld: frozenset[bytes] = frozenset()):
        self._urls = rest_proxy_urls
        self._withheld = withheld
        self._timeout = timeout
        self._log_event = log_event
        self._attribution = attribution
        self._role_of = role_of
        self._found: dict[str, list[Browser]] = {}

    async def browser(self, key: str) -> Browser | Response:
        """The website browser behind the ``/svc`` child ``key``, from its card, read on first use and kept once it
        names one: a 404 for a key no child has or whose card offers none, a 409 for one offering several, and a 503
        while its card can't be read. A card naming none is read again next time, since a computer lists an app
        whose server isn't answering yet by its name alone."""
        base = self._urls.get(key)
        if base is None:
            return _error(404, f"no browser {key!r}")
        found = self._found.get(key)
        if found is None:
            async with httpx.AsyncClient() as client:
                card = await fetch_child_card(client, key, base)
            if card is None:
                return _error(503, f"{key}'s card can't be read")
            found = browsers(card)
            if found:
                self._found[key] = found
        if len(found) != 1:
            return _error(409 if found else 404, f"{key} has {len(found) or 'no'} website browsers")
        return found[0]

    async def discovery(self, request: Request) -> Response:
        """A request under a CDP address, discovery's ``json/version`` and ``json/list``, forwarded to the browser,
        with each address in a JSON answer at the gateway."""
        key = request.path_params["key"]
        browser = await self.browser(key)
        if isinstance(browser, Response):
            return browser
        base = public_base(request)
        if base is None:
            return _error(400, "the Host header isn't a host and port")
        headers = client_headers(request, self._withheld)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                answer = await client.request(request.method, self._target(key, browser, request), headers=headers)
        except httpx.HTTPError as e:
            return _error(502, f"the browser didn't answer: {type(e).__name__}")
        return _answered(answer, DISCOVERY_FIELDS, base, key)

    async def cdp_answer(self, request: Request, base_url: str, role: str) -> Response | None:
        """A browser's ``cdp`` answer to a call through its child's ``/svc/<key>/`` route, with its addresses at the
        gateway; None for any other ``/svc`` request, which goes through as it is."""
        path = posixpath.normpath("/" + request.path_params.get("path", ""))
        if request.method != "POST" or not path.endswith("/cdp"):
            return None
        key = request.path_params["service_name"]
        browser = await self.browser(key)
        if not isinstance(browser, Browser) or path != browser.cdp_path:
            return None
        base = public_base(request)
        if base is None:
            return _error(400, "the Host header isn't a host and port")
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                answer = await client.post(upstream_url(base_url, request), content=await request.body(),
                                           headers=upstream_headers(request, role, withheld=self._withheld))
        except httpx.HTTPError as e:
            return _error(502, f"the browser didn't answer: {type(e).__name__}")
        return _answered(answer, CDP_ANSWER_FIELDS, base, key)

    async def relay(self, websocket: WebSocket) -> None:
        """A WebSocket under a CDP address relayed both ways to the browser until either side closes, each leg pinged
        every ``PING_INTERVAL_S``; refused before it opens when its browser can't be found, refuses it or doesn't
        answer. Each attempt is a ``browser_connection`` trajectory event once it ends, without its token."""
        key = websocket.path_params["key"]
        event = {"event_type": "browser_connection", "browser": key, "path": websocket.path_params.get("path", ""),
                 **self._attribution(self._role_of(websocket))}
        browser = await self.browser(key)
        if isinstance(browser, Response):
            await websocket.close(code=REFUSED)
            await self._log_event({**event, "opened_at": None, "closed_at": _now(), "close_code": REFUSED,
                                   "closed_by": "gateway", "status": browser.status_code})
            return
        headers = [(name.decode("latin-1"), value.decode("latin-1"))
                   for name, value in client_headers(websocket, self._withheld)
                   if not name.lower().startswith(b"sec-websocket-")]
        offered = [p.strip() for p in websocket.headers.get("sec-websocket-protocol", "").split(",") if p.strip()]
        try:
            upstream = await connect(
                _websocket_form(self._target(key, browser, websocket)), additional_headers=headers,
                subprotocols=offered or None, max_size=None, compression=None, user_agent_header=None,
                ping_interval=PING_INTERVAL_S, ping_timeout=PING_INTERVAL_S, open_timeout=self._timeout.connect)
        except websockets.InvalidStatus as e:  # an address the browser didn't mint, or a route it refuses
            await websocket.close(code=REFUSED)
            await self._log_event({**event, "opened_at": None, "closed_at": _now(), "close_code": REFUSED,
                                   "closed_by": "browser", "status": e.response.status_code})
            return
        except (OSError, ValueError, asyncio.TimeoutError, websockets.WebSocketException) as e:
            await websocket.close(code=UNREACHABLE)
            await self._log_event({**event, "opened_at": None, "closed_at": _now(), "close_code": UNREACHABLE,
                                   "closed_by": "gateway", "error": type(e).__name__})
            return
        opened, closed_by, code, error = _now(), "gateway", 1001, None
        try:
            async with upstream:
                await websocket.accept(subprotocol=upstream.subprotocol)
                closed_by, code, error = await _both_ways(websocket, upstream)
        finally:
            if error is not None:
                logger.warning(f"browser {key}: a relayed connection failed with {error}")
            await self._log_event({**event, "opened_at": opened, "closed_at": _now(), "close_code": code,
                                   "closed_by": closed_by, **({"error": error} if error else {})})

    def _target(self, key: str, browser: Browser, connection: Request | WebSocket) -> str:
        """The browser's own URL for a request under ``/browser/<key>/``: the child's, the browser's prefix and the
        path from ``cdp/`` on as it came, still percent-encoded, and its query."""
        raw = (connection.scope.get("raw_path") or connection.url.path.encode()).decode("latin-1")
        tail = raw.split("/", 3)[3]  # "/browser/<key>/cdp/..." -> "cdp/..."
        query = connection.scope.get("query_string", b"").decode("latin-1")
        return f"{self._urls[key].rstrip('/')}{browser.prefix}/{tail}" + (f"?{query}" if query else "")


async def _both_ways(client: WebSocket, browser: ClientConnection) -> tuple[str, int, str | None]:
    """Each message from either side sent to the other until one side closes, then the other closed with its code:
    which side closed first, and its code. When neither closed but relaying failed, both are closed as ``UNREACHABLE``
    and the failure's type is the third value, never its message, which may hold the page's data."""
    ended: list[tuple[str, int]] = []

    async def inbound() -> None:
        while True:
            message = await client.receive()
            if message["type"] == "websocket.disconnect":
                code = message.get("code") or 1000
                ended.append(("client", code))
                await browser.close(1000 if code in UNSENDABLE_CLOSES else code)
                return
            await browser.send(message["text"] if message.get("text") is not None else message["bytes"])

    async def outbound() -> None:
        with contextlib.suppress(websockets.ConnectionClosed):
            async for message in browser:
                if isinstance(message, str):
                    await client.send_text(message)
                else:
                    await client.send_bytes(message)
        code = browser.close_code or 1000
        ended.append(("browser", code))
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await client.close(1000 if code in UNSENDABLE_CLOSES else code)

    tasks = [asyncio.ensure_future(inbound()), asyncio.ensure_future(outbound())]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    if ended:
        return (*ended[0], None)
    failure = next((task.exception() for task in done if not task.cancelled() and task.exception()), None)
    if failure is None:
        return "gateway", 1001, None
    with contextlib.suppress(RuntimeError, WebSocketDisconnect):
        await client.close(UNREACHABLE)
    with contextlib.suppress(websockets.WebSocketException, OSError):
        await browser.close(UNREACHABLE)
    return "gateway", UNREACHABLE, type(failure).__name__


def _answered(answer: httpx.Response, fields: frozenset[str], base: str, key: str) -> Response:
    """The browser's ``answer``, a JSON one with each of ``fields`` at the gateway, anything else as it came."""
    media_type = answer.headers.get("content-type")
    if media_type and media_type.split(";")[0].strip() == "application/json":
        try:
            return JSONResponse(rewritten(answer.json(), fields, base, key), status_code=answer.status_code)
        except ValueError:
            pass
    return Response(answer.content, status_code=answer.status_code, media_type=media_type)


def _method_path(card: dict, uri: str, name: str) -> str | None:
    params = _params(card, uri)
    return next((path for method, _, _, _, path in operations(params) if method == name), None) if params else None


def _params(card: dict, uri: str) -> dict:
    for extension in (card.get("capabilities") or {}).get("extensions") or []:
        if isinstance(extension, dict) and extension.get("uri") == uri and isinstance(extension.get("params"), dict):
            return extension["params"]
    return {}


def _websocket_form(url: str) -> str:
    return "wss://" + url.removeprefix("https://") if url.startswith("https://") else "ws://" + url.removeprefix("http://")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error(status: int, message: str) -> Response:
    return JSONResponse({"error": message}, status_code=status)
