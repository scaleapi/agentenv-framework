"""A site whose pages an app beside it serves: ``ForwardingSite``, a ``WebsiteEnvironment`` that answers its card, its
data plane and ``website/v1`` itself and sends every other request on to ``upstream``, an app in the same container
such as a third-party web app it wraps. The request goes on as it came: its method, the site's own path with its
encoding intact and its query, its headers but the hop-by-hop ones and the site's ``sign_in_headers``, the ``Host``
the browser named and its body, streamed. The answer comes back as the app gave it, streamed and still encoded, but
for a cookie the app ties to one of the site's own names (``Domain``), which comes back the host's alone, as agentenv
sets its own, so the cookie a sign-in sets is the app's cookie, not one beside it. Such a site usually declares
``@website(builtin_routes=False, clock="real")``, so every path but agentenv's own reaches the app and nothing
expects the app to follow the universe's clock; it signs a browser in through a ``@session_state`` of its own, which
asks the app with ``ask_app`` as its ready checks do. Its app keeps its own data, so its data plane serves
``data/get`` alone, which exports nothing: it takes no load and can't reset, which agentenv reads off its card, as it
reads any site's that serves no ``data/reset``. Its card names its ``sign_in_headers``, whose values the browser's
network log takes out.

It forwards HTTP alone: a WebSocket is refused (close code 1011)."""
from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from functools import cached_property
from http.cookiejar import DefaultCookiePolicy
from typing import Any, ClassVar, Optional
from urllib.parse import quote

import httpx
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import StreamingResponse

from ..agent_env_environment import get_data
from ..passthrough import passed_on
from .environment import SITE_RAW_PATH, WebsiteEnvironment, ready_check, route_path
from .errors import ApiError
from .readiness import WEBSITE_EXTENSION_URI, WEBSITE_SIGN_IN_HEADERS_PARAM
from .routes import error_response

#: How long the forwarder waits to connect to the app; once connected, it waits as long as the app takes to answer.
CONNECT_TIMEOUT_S = 10.0
#: How long the ready check waits for the app's answer, inside the ready check's own bound.
READY_ANSWER_S = 8.0
#: How long ``ask_app`` waits for the app's answer unless told otherwise.
ASK_TIMEOUT_S = 60.0
#: What a path's segments may hold unescaped, besides letters, digits and ``_.-~``.
PATH_SAFE = "/:@!$&'()*+,;="
#: The close code a WebSocket gets: the forwarder relays HTTP alone.
WEBSOCKET_REFUSED = 1011
#: The forwarder's client keeps no cookie, so each request carries the cookies its own browser sent and no others.
KEEPS_NO_COOKIES = DefaultCookiePolicy(allowed_domains=[])


class ForwardingSite(WebsiteEnvironment):
    """A site whose pages ``upstream`` serves: set ``upstream`` to the app's address in the container, ``ready_path``
    to a path it answers once it's up (the site is ready when that answers below 500), and ``sign_in_headers`` to
    the request headers only the site's own calls may send the app, such as the sign-in header its ``@session_state``
    uses: the forwarder drops them from every browser's request, and the browser's network log takes their values out.
    A site's own calls go through ``ask_app``."""

    upstream: ClassVar[str] = "http://127.0.0.1:80"
    ready_path: ClassVar[str] = "/"
    sign_in_headers: ClassVar[tuple[str, ...]] = ()

    def fallback_app(self) -> Any:
        return self._forwarder

    def extension_params(self, uri: str) -> dict:
        params = super().extension_params(uri)
        if uri == WEBSITE_EXTENSION_URI and self.sign_in_headers:
            return {**params, WEBSITE_SIGN_IN_HEADERS_PARAM: list(self.sign_in_headers)}
        return params

    async def ask_app(self, path: str, *, method: str = "GET", headers: Optional[dict[str, str]] = None,
                      content: Optional[bytes] = None, timeout: float = ASK_TIMEOUT_S) -> httpx.Response:
        """The app's answer to ``method`` ``path`` asked as the site: under its first name as ``Host``, with
        ``headers``, such as one of its ``sign_in_headers``, which only the site's own calls send, and
        ``content``; through the forwarder's client, so with no cookie but those in ``headers`` and no redirect
        followed."""
        names = self._own_names()
        return await self._forwarder.client.request(
            method, self._forwarder.url(path), content=content, timeout=timeout,
            headers={**({"host": names[0]} if names else {}), **(headers or {})})

    @cached_property
    def _forwarder(self) -> Forwarder:
        return Forwarder(self.upstream, self.sign_in_headers, self._own_names())

    def _own_names(self) -> list[str]:
        """The site's exact names, the first the one its own calls ask the app under."""
        return [name for name in self.declaration().domains if not name.startswith("*.")]

    @get_data
    async def exported(self) -> list:
        """Nothing: the app keeps its own data, which a fresh deploy starts afresh."""
        return []

    @ready_check
    async def upstream_answers(self) -> bool:
        """The app answers ``ready_path``, for the site's first name, below 500."""
        answer = await self.ask_app(self.ready_path, timeout=READY_ANSWER_S)
        if answer.status_code >= 500:
            raise RuntimeError(f"{self.ready_path} answered {answer.status_code}")
        return True


class Forwarder:
    """The ASGI app that sends each HTTP request on to ``upstream`` and streams its answer back, a cookie tied to one
    of ``names`` the host's alone."""

    def __init__(self, upstream: str, dropped: Iterable[str] = (), names: Iterable[str] = ()) -> None:
        self.upstream = upstream.rstrip("/")
        self.dropped = tuple(dropped)
        self.names = frozenset(name.lower() for name in names)
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        """The client requests go on with: straight to the app, whatever proxy the environment names, following no
        redirect, so the browser gets each one, and with no headers or cookies of its own, so the app gets only the
        request's: a cookie one browser was answered with never goes on with another's request."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=CONNECT_TIMEOUT_S), trust_env=False,
                                             follow_redirects=False)
            self._client.headers.clear()
            self._client.cookies.jar.set_policy(KEEPS_NO_COOKIES)
        return self._client

    def url(self, path: str) -> str:
        """Where ``path`` is on the app: a path as a request names it, ``/`` first, or one given without it."""
        return self.upstream + (path if path.startswith("/") else f"/{path}")

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": WEBSOCKET_REFUSED, "reason": "the site forwards HTTP alone"})
            return
        if scope["type"] != "http":
            return
        request = Request(scope, receive)
        headers = passed_on(_text(scope["headers"]), *self.dropped)
        has_body = request.headers.get("content-length", "0") != "0" or "transfer-encoding" in request.headers
        try:
            answer = await self.client.send(self.client.build_request(
                scope["method"], self.url(target(scope)), headers=_bytes(headers),
                content=request.stream() if has_body else None), stream=True)
        except httpx.HTTPError as e:
            await error_response(ApiError(f"the app at {self.upstream} isn't answering: {type(e).__name__}: {e}",
                                          code="upstream_unavailable", status=502))(scope, receive, send)
            return

        async def streamed() -> AsyncIterator[bytes]:
            try:
                async for chunk in answer.aiter_raw():
                    yield chunk
            finally:
                await answer.aclose()

        response = StreamingResponse(streamed(), status_code=answer.status_code, background=BackgroundTask(answer.aclose))
        response.raw_headers = _bytes((name, host_only(value, self.names) if name.lower() == "set-cookie" else value)
                                      for name, value in passed_on(_text(answer.headers.raw)))
        await response(scope, receive, send)


def target(scope: dict) -> str:
    """The path and query a request asks the app for: the site's own path as it reached the server, still
    percent-encoded, so an escaped ``/`` stays one, before any forwarded prefix was put before it."""
    raw = scope.get(SITE_RAW_PATH)
    path = raw.decode("latin-1") if raw else quote(route_path(scope), safe=PATH_SAFE)
    query = scope.get("query_string", b"").decode("latin-1")
    return path + (f"?{query}" if query else "")


def host_only(set_cookie: str, names: frozenset[str]) -> str:
    """``set_cookie`` without a ``Domain`` attribute naming one of ``names``, so the cookie is the host's alone."""
    name_value, *attributes = set_cookie.split(";")
    return ";".join([name_value, *(attribute for attribute in attributes if not _names(attribute, names))])


def _names(attribute: str, names: frozenset[str]) -> bool:
    key, _, value = attribute.partition("=")
    return key.strip().lower() == "domain" and value.strip().lstrip(".").lower() in names


def _text(headers: Iterable[tuple[bytes, bytes]]) -> list[tuple[str, str]]:
    return [(name.decode("latin-1"), value.decode("latin-1")) for name, value in headers]


def _bytes(headers: Iterable[tuple[str, str]]) -> list[tuple[bytes, bytes]]:
    return [(name.encode("latin-1"), value.encode("latin-1")) for name, value in headers]
