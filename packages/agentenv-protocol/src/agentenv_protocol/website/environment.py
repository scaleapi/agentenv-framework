"""WebsiteEnvironment: an AgentEnvEnvironment that is a site's server, meeting ``urn:agentenv:website/v1``. It serves
plain HTTP on ``PORT`` for any host name: its ``@api`` routes, its static folder at ``/`` with ``index.html`` for the
pages a single-page app routes itself, and the data plane and card the base class serves, the card's ``website/v1``
``ready`` saying whether the site is usable yet by its ``@ready_check`` methods. Every answer
carries ``X-Universe-Now``, the universe's time, which ``self.now()`` reads and ``clock/v1``'s ``sync_time`` ties to
the gateway's virtual clock, read once per request as MCP servers read it. A site keeps its data where it likes,
usually its schema of the env's database (``database_url``), and serves it through the data plane's hooks
(``@reset_data``, ``@add_data``, ``@get_data``) as any AgentEnvEnvironment does: ``data/get`` is how agentenv reads
what a task left a site with.

One route is built in, replaced by a site's own ``@api`` route at its very path and answering before one whose path
parameters would also match it: ``/api/session``, which a frontend calls to learn who is signed in, signing the acting
user in when nobody is (GET), or to sign out (DELETE)."""
from __future__ import annotations

import asyncio
import inspect
import math
import os
import re
import time
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, Optional
from urllib.parse import unquote

from starlette.applications import Starlette
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from ..agent_env_environment import (
    AgentEnvEnvironment,
    AgentEnvStarletteApplication,
    _marked_members,
    _maybe_await,
    extension,
)
from .clock import CLOCK_EXTENSION_URI, UNIVERSE_NOW_HEADER, ClockUnavailable, UniverseClock, clock_patch_script, iso
from .errors import ApiError, NotFound
from .readiness import SITE_READY_TIMEOUT_S, WEBSITE_EXTENSION_URI, WEBSITE_READY_OP, WebsiteReadiness
from .routes import (ApiContext, ApiRoute, endpoint, error_response, forwarded_prefix, openapi, openapi_path, request_role,
                     reserved_path, routes_of)
from .session import SESSION_COOKIE, SessionSigner

SESSION_PATH = "/api/session"
CHECK_TIMEOUT_S = 10.0
_WEBSITE_ATTR = "_aee_website"
_READY_ATTR = "_aee_website_ready"
_HEAD = re.compile(rb"<head(?:\s[^>]*)?>", re.IGNORECASE)
_PREAMBLE = re.compile(rb"\A(?:\s|<!--.*?-->|<!doctype[^>]*>|<html(?:\s[^>]*)?>)*", re.IGNORECASE | re.DOTALL)
_HOST = re.compile(r"(\*\.)?[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?)*")


class WebsiteDeclaration(NamedTuple):
    """What ``@website`` declares: the site's domains, its static folder (relative to the class's file), how long it
    may take to be ready after its server starts, and whether its HTML pages read the universe's time."""

    domains: tuple[str, ...] = ()
    static: Optional[str] = None
    ready_timeout_s: float = SITE_READY_TIMEOUT_S
    clock_patch: bool = False


def website(_cls: type | None = None, *, domains: list[str] | tuple[str, ...] = (), static: str | None = None,
            ready_timeout_s: float = SITE_READY_TIMEOUT_S, clock_patch: bool = False) -> Any:
    """Declare a WebsiteEnvironment's site."""
    if _cls is not None:
        raise TypeError("use @website(...) with parentheses")
    if isinstance(domains, str):
        raise ValueError(f"domains is a list of host names, not the string {domains!r}")
    if problems := [problem for host in domains if (problem := host_problem(host))]:
        raise ValueError(f"domains: {problems[0]}")
    if isinstance(ready_timeout_s, bool) or not isinstance(ready_timeout_s, (int, float)) or not (
            0 < ready_timeout_s < math.inf):
        raise ValueError(f"ready_timeout_s is a number of seconds above 0, not {ready_timeout_s!r}")

    def deco(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, WebsiteEnvironment)):
            raise TypeError(f"@website declares a WebsiteEnvironment subclass, not {cls!r}")
        setattr(cls, _WEBSITE_ATTR, WebsiteDeclaration(tuple(domains), static, float(ready_timeout_s), clock_patch))
        return cls

    return deco


def host_problem(host: Any) -> str | None:
    """Why ``host`` isn't a host name a site is reached at, None when it is: a name like ``github.com``, or
    ``*.github.com`` for its subdomains, with no scheme, port or path."""
    if not isinstance(host, str) or not _HOST.fullmatch(host):
        return f"{host!r} is not a host name such as example.com or *.example.com"
    return None


def ready_check(fn: Any) -> Any:
    """Mark a method that says whether the site is usable yet: ``website/v1``'s ``ready`` says ready once every one
    returns a truthy value. They run at once, each within ``CHECK_TIMEOUT_S``, a plain ``def`` in a worker thread."""
    setattr(fn, _READY_ATTR, True)
    return fn


class WebsiteEnvironment(AgentEnvEnvironment):
    """Subclass it, declare the site with ``@website(...)``, mark its API with ``@api``, and ``serve()`` it. A
    subclass's ``__init__`` calls ``super().__init__()``. ``app`` may be a Starlette or FastAPI app answering what
    ``@api`` and the static folder don't (streaming, uploads); its lifespan is not run."""

    app: ClassVar[Any] = None
    asgi: Any = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        routes_of(cls)

    def __init__(self) -> None:
        self.clock = UniverseClock()
        self._signer = SessionSigner()
        self._started = time.monotonic()
        self._checking: dict[str, asyncio.Future] = {}

    @classmethod
    def declaration(cls) -> WebsiteDeclaration:
        return getattr(cls, _WEBSITE_ATTR, None) or WebsiteDeclaration()

    @classmethod
    def routes(cls) -> list[ApiRoute]:
        return routes_of(cls)

    @property
    def database_url(self) -> str | None:
        """Where the site keeps its task state: its own schema of the env's database."""
        return os.environ.get("DATABASE_URL")

    def now(self) -> Any:
        """The universe's time, an aware UTC datetime: the instant the request being answered read from the gateway's
        virtual clock once synced, else wall time."""
        return self._clock().now()

    @property
    def timezone(self) -> str | None:
        """The universe's time zone, an IANA name such as ``America/Chicago``, as the last ``sync_time`` named it;
        None when it named none. ``now()`` is UTC either way."""
        return self._clock().timezone

    def acting_user(self) -> str | None:
        """Who a browser with no session is signed in as: nobody unless a site says else, from its data say."""
        return None

    def context(self, request: Request) -> ApiContext:
        """The ``ApiContext`` of ``request``, for a mounted ``app``'s routes as for ``@api`` ones."""
        session = self._session_signer().decode(request.cookies.get(SESSION_COOKIE))
        user = session["user"] if session is not None else self.acting_user()
        return ApiContext(request, user, request_role(request), session, self._session_signer())

    def create_app(self) -> Starlette:
        if self._mounted:
            raise RuntimeError(f"{type(self).__name__} is already mounted")
        declaration = self.declaration()
        api_routes = self.routes()
        taken = {openapi_path(r.path) for r in api_routes}
        root = self._site_path(declaration.static) if declaration.static is not None else None
        asgi = Starlette(middleware=[Middleware(_Forwarded), Middleware(_Universe, env=self)])
        AgentEnvStarletteApplication(self._build_card(), self).add_routes_to_app(asgi)
        if SESSION_PATH not in taken:
            asgi.router.routes.append(Route(SESSION_PATH, self._session, methods=["GET", "DELETE"]))
        asgi.router.routes.extend(Route(r.path, _unreserved(endpoint(getattr(self, r.fn.__name__), r, self.context)),
                                        methods=[r.method]) for r in api_routes)
        asgi.router.default = _Fallback(root, type(self).app)
        self.asgi = asgi
        self._mounted = True
        return asgi

    def mount(self, app: Any) -> Any:
        raise TypeError(f"{type(self).__name__} builds its own app with create_app(); set app = ... to serve more")

    def serve(self) -> None:
        """Serve the site on ``PORT`` (8000 by default) with uvicorn, of the ``agent`` or ``app`` extra."""
        try:
            import uvicorn
        except ImportError as e:
            raise ImportError("WebsiteEnvironment.serve needs uvicorn: pip install 'uvicorn>=0.30'") from e
        uvicorn.run(self.asgi or self.create_app(), host=os.environ.get("HOST", "0.0.0.0"),
                    port=int(os.environ.get("PORT", "8000")))

    def extension_params(self, uri: str) -> dict:
        if uri == WEBSITE_EXTENSION_URI:
            declaration = self.declaration()
            return {"domains": list(declaration.domains),
                    "routes": [r.advertised() for r in self.routes()], "session_cookie": SESSION_COOKIE,
                    "session_path": SESSION_PATH}
        return super().extension_params(uri)

    @extension(CLOCK_EXTENSION_URI, description="Sync this site to the gateway's virtual clock.")
    async def sync_time(self, env_get_time_url: str, timezone: Optional[str] = None) -> dict:
        """Follow the gateway's virtual time at ``env_get_time_url`` from now on, in ``timezone`` when the task names
        one; an empty ``env_get_time_url`` goes back to wall time."""
        virtual = await self._clock().sync(env_get_time_url, timezone.strip() if timezone else None)
        return {"env_get_time_url": env_get_time_url, "gateway_virtual_time": iso(virtual),
                **({"timezone": self.timezone} if self.timezone else {})}

    @extension(WEBSITE_EXTENSION_URI, op="openapi", method="GET", description=(
        "A site's server: domains and its @api routes by name; openapi returns the OpenAPI document "
        "of its routes, ready whether the site is usable now and why not, and session the storage state a browser adds "
        "to start signed in as user, or signed out without one: {cookies, local_storage}, each cookie's lifetime a "
        "max_age."))
    async def _website_openapi(self) -> dict:
        name = self._application.environment_card.name if self._application is not None else type(self).__name__
        return openapi(name, self.routes())

    @extension(WEBSITE_EXTENSION_URI, op=WEBSITE_READY_OP, method="GET")
    async def _website_ready(self) -> dict:
        """Whether the site is usable now, a ``WebsiteReadiness`` by its ``@ready_check`` methods: ``problems`` says
        what each not yet true returned, raised or how long it ran, and ``waited_s`` how long since the server started,
        against the ``ready_timeout_s`` it declared. A deploy reads it through the card."""
        problems = await self._not_ready()
        return WebsiteReadiness(ready=not problems, problems=problems, waited_s=round(time.monotonic() - self._started, 1),
                                timeout_s=self.declaration().ready_timeout_s).model_dump()

    @extension(WEBSITE_EXTENSION_URI, op="session")
    async def _website_session(self, user: Optional[str] = None) -> dict:
        return {"cookies": [self._session_signer().cookie(user)], "local_storage": []}

    async def _not_ready(self) -> list[str]:
        found = await asyncio.gather(*(self._checked(check) for check, _marker in _marked_members(self, _READY_ATTR)))
        return [problem for problem in found if problem is not None]

    async def _checked(self, check: Any) -> Optional[str]:
        """What ``check`` finds wrong, within ``CHECK_TIMEOUT_S``; one that runs past it or raises says so."""
        name = f"@ready_check {check.__name__}"
        try:
            passed = await asyncio.wait_for(self._run_check(check), CHECK_TIMEOUT_S)
        except asyncio.TimeoutError:
            return f"{name} ran past {CHECK_TIMEOUT_S:g} s"
        except Exception as e:
            return f"{name}: {type(e).__name__}: {e}"
        return None if passed else f"{name} returned {passed!r}"

    async def _run_check(self, check: Any) -> Any:
        """A plain ``def`` check runs in a worker thread, so it can't hold up the site's answers, and once at a time:
        a thread can't be stopped, so a poll while one runs past its bound waits on that same call."""
        if inspect.iscoroutinefunction(check):
            return await check()
        running = self._checking.get(check.__name__)
        if running is None or running.done():
            running = self._checking[check.__name__] = asyncio.ensure_future(asyncio.to_thread(check))
            running.add_done_callback(lambda done: done.cancelled() or done.exception())
        return await _maybe_await(await asyncio.shield(running))

    async def _session(self, request: Request) -> Response:
        context = self.context(request)
        if request.method == "DELETE":
            context.sign_out()
        elif context.session is None and context.user is not None:
            context.sign_in(context.user)
        return context.apply(JSONResponse({"user": context.user, "role": context.role,
                                           "signed_in": context.user is not None}))

    def _clock(self) -> UniverseClock:
        if not hasattr(self, "clock"):
            raise TypeError(f"{type(self).__qualname__}.__init__ must call super().__init__()")
        return self.clock

    def _session_signer(self) -> SessionSigner:
        if not hasattr(self, "_signer"):
            raise TypeError(f"{type(self).__qualname__}.__init__ must call super().__init__()")
        return self._signer

    @classmethod
    def _site_path(cls, path: str) -> Path:
        given = Path(path)
        return given if given.is_absolute() else Path(inspect.getfile(cls)).resolve().parent / given


class _Forwarded:
    """Puts the address the browser reached the site at in the scope, from ``X-Forwarded-Host``,
    ``X-Forwarded-Proto`` and ``X-Forwarded-Prefix``, before anything routes it, so redirects, ``request.url`` and a
    mounted app's ``url_for`` name that address and not the internal one the proxy reached the server at. A prefix
    becomes the scope's ``root_path`` and leads its ``path``, as ASGI has a mounted app's, so routing still sees the
    site's own path."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] in ("http", "websocket"):
            headers = dict(scope["headers"])
            host = _first(headers.get(b"x-forwarded-host"))
            proto = _first(headers.get(b"x-forwarded-proto")).lower()
            scope = dict(scope)
            if host:
                scope["headers"] = [(k, v) for k, v in scope["headers"] if k != b"host"] + [(b"host", host.encode())]
            if proto in ("http", "https"):
                scope["scheme"] = proto if scope["type"] == "http" else {"http": "ws", "https": "wss"}[proto]
            prefix = forwarded_prefix(_first(headers.get(b"x-forwarded-prefix")))
            if prefix:
                scope["root_path"] = unquote(prefix) + scope.get("root_path", "")
                scope["path"] = unquote(prefix) + scope["path"]
                if scope.get("raw_path") is not None:
                    scope["raw_path"] = prefix.encode("latin-1") + scope["raw_path"]
        await self.app(scope, receive, send)


def _first(value: bytes | None) -> str:
    return value.decode("latin-1").split(",")[0].strip() if value else ""


class _Universe:
    """Reads the universe's clock once per request, the instant ``now()`` reads while it's answered, and stamps it on
    the answer as ``X-Universe-Now``; with ``clock_patch`` it puts the clock script in the whole HTML (never a range of
    it), holding only the HTML up to its ``<head>`` (at most ``HEAD_SCAN_BYTES``), so a page streamed in chunks reaches
    the browser as it's sent. A request the clock can't be read for answers 503, but the data plane and the card,
    which need no time, answer without the header."""

    HEAD_SCAN_BYTES = 64 * 1024

    def __init__(self, app: Any, env: WebsiteEnvironment) -> None:
        self.app, self.env = app, env

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        clock = self.env._clock()
        try:
            now: Any = await clock.read()
        except ClockUnavailable as e:
            if not reserved_path(_route_path(scope)):
                return await error_response(ApiError(str(e), code="clock_unavailable", status=503))(scope, receive, send)
            now = None
        token = clock.answering_at(now)
        try:
            await self.app(scope, receive, self._stamping(scope, send, now))
        finally:
            clock.answered(token)

    def _stamping(self, scope: dict, send: Any, now: Any) -> Any:
        patch = self.env.declaration().clock_patch and scope["method"] != "HEAD" and now is not None
        held: dict = {}

        async def stamped(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                if now is not None:
                    headers[UNIVERSE_NOW_HEADER] = iso(now)
                if (patch and headers.get("content-type", "").startswith("text/html") and "content-encoding" not in headers
                        and message.get("status") != 206 and "content-range" not in headers):
                    held.update(start=message, body=b"", script=clock_patch_script(now))
                    return
            elif held and message["type"] == "http.response.body":
                held["body"] += message.get("body", b"")
                more = message.get("more_body", False)
                if more and not _HEAD.search(held["body"]) and len(held["body"]) < self.HEAD_SCAN_BYTES:
                    return
                start, script = held.pop("start"), held.pop("script")
                body = _with_script(held.pop("body"), script)
                held.clear()
                headers = MutableHeaders(scope=start)
                if "content-length" in headers:
                    headers["content-length"] = str(int(headers["content-length"]) + len(script))
                await send(start)
                message = {"type": "http.response.body", "body": body, "more_body": more}
            await send(message)

        return stamped


def _with_script(html: bytes, script: bytes) -> bytes:
    """``html`` with ``script`` after its ``<head>``, else after its doctype and ``<html>`` tag, so a page without a
    head keeps its doctype first."""
    head = _HEAD.search(html)
    at = head.end() if head else _PREAMBLE.match(html).end()
    return html[:at] + script + html[at:]


def _route_path(scope: dict) -> str:
    """The site's own path of a request, as routing reads it: its path without the ``root_path`` a forwarded prefix put
    before it (``_Forwarded``)."""
    root, path = scope.get("root_path", ""), scope["path"]
    return (path[len(root):] or "/") if root and path.startswith(root) else path


def _unreserved(handler: Any) -> Any:
    """``handler``, an ``@api`` route's endpoint, answering 404 at the data plane's and card's paths, so a broad route
    (``/{rest:path}``) never serves a page there; the routes agentenv registers at those paths answer before it."""
    async def answer(request: Request) -> Response:
        if reserved_path(_route_path(request.scope)):
            return error_response(NotFound("not found"))
        return await handler(request)
    return answer


class _Fallback:
    """What no route answers: a file of the static folder, then the mounted app, then for a page's path the folder's
    ``index.html``; the data plane's and card's paths answer 404 however they're spelled."""

    def __init__(self, root: Path | None, app: Any) -> None:
        self.root = root.resolve() if root is not None else None
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            if self.app is not None:
                await self.app(scope, receive, send)
            return
        path, reading = _route_path(scope), scope["method"] in ("GET", "HEAD")
        if reserved_path(path):
            return await error_response(NotFound("not found"))(scope, receive, send)
        if reading and (file := self._file(path)) is not None:
            return await FileResponse(file)(scope, receive, send)
        index = self._file("/") if reading and _page(path) else None
        if self.app is not None:
            return await self._app_or(index, scope, receive, send)
        if index is not None:
            return await FileResponse(index)(scope, receive, send)
        await error_response(NotFound("not found"))(scope, receive, send)

    def _file(self, path: str) -> Path | None:
        """The static folder's file at ``path``, a directory's ``index.html``; none under a dot-named segment, such as
        ``.env``, or for a path that can't be a file's."""
        if self.root is None or any(part.startswith(".") for part in path.split("/")):
            return None
        try:
            candidate = (self.root / path.lstrip("/")).resolve()
            if candidate.is_dir():
                candidate = (candidate / "index.html").resolve()
            found = candidate.is_relative_to(self.root) and candidate.is_file()
        except (OSError, ValueError):
            return None
        return candidate if found else None

    async def _app_or(self, index: Path | None, scope: dict, receive: Any, send: Any) -> None:
        """The mounted app's answer, or ``index`` instead of the app's 404 when there is one."""
        missed = False

        async def unless_missed(message: dict) -> None:
            nonlocal missed
            if message["type"] == "http.response.start" and message["status"] == 404 and index is not None:
                missed = True
            if not missed:
                await send(message)

        await self.app(scope, receive, unless_missed)
        if missed:
            await FileResponse(index)(scope, receive, send)


def _page(path: str) -> bool:
    """Whether ``path`` is a page a single-page app routes itself: not the API's, and not a file's."""
    return not (path == "/api" or path.startswith("/api/")) and "." not in path.rsplit("/", 1)[-1]
