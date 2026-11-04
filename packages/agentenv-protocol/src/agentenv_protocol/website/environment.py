"""WebsiteEnvironment: an AgentEnvEnvironment that is a site's server, meeting ``urn:agentenv:website/v1``. It serves
plain HTTP on ``PORT`` for any host name: its ``@api`` routes, its static folder at ``/`` with ``index.html`` for the
pages a single-page app routes itself, its health path, and the data plane and card the base class serves. Every answer
carries ``X-Universe-Now``, the universe's time, which ``self.now()`` reads and ``clock/v1``'s ``sync_time`` ties to
the gateway's virtual clock. Without data hooks of its own a site keeps its data in ``self.store``, which the data
plane loads, exports and resets: ``data/get`` is how agentenv reads what a task left a site with.

One route is built in, replaced by a site's own ``@api`` route at its very path and answering before one whose path
parameters would also match it: ``/api/session``, which a frontend calls to learn who is signed in, signing the acting
user in when nobody is (GET), or to sign out (DELETE)."""
from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, Optional

from starlette.applications import Starlette
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Route

from ..agent_env_environment import (
    _OP_ATTR,
    AgentEnvEnvironment,
    AgentEnvStarletteApplication,
    _marked_members,
    _maybe_await,
    add_data,
    extension,
    get_data,
    reset_data,
)
from ..types import DataPart
from .clock import CLOCK_EXTENSION_URI, UNIVERSE_NOW_HEADER, UniverseClock, clock_patch_script, iso
from .errors import NotFound
from .readiness import WEBSITE_EXTENSION_URI, WEBSITE_READY_OP, WebsiteReadiness
from .routes import ApiContext, ApiRoute, endpoint, error_response, openapi, openapi_path, request_role, reserved_path, routes_of
from .session import SESSION_COOKIE, SessionSigner, expires_at
from .store import Collections

DEFAULT_HEALTH_PATH = "/api/health"
SESSION_PATH = "/api/session"
_WEBSITE_ATTR = "_aee_website"
_READY_ATTR = "_aee_website_ready"
_DEFAULT_HOOK_ATTR = "_aee_default_store_hook"
_DEFAULT_HOOKS = {"reset_data": "_store_reset", "add_data": "_store_add", "get_data": "_store_get"}
_HEAD = re.compile(rb"<head[^>]*>", re.IGNORECASE)


class WebsiteDeclaration(NamedTuple):
    """What ``@website`` declares: the site's domains, its static folder and seed data file (relative to the class's
    file), its health path, and whether its HTML pages read the universe's time."""

    domains: tuple[str, ...] = ()
    static: Optional[str] = None
    health: str = DEFAULT_HEALTH_PATH
    clock_patch: bool = False
    data: Optional[str] = None


def website(_cls: type | None = None, *, domains: list[str] | tuple[str, ...] = (), static: str | None = None,
            health: str = DEFAULT_HEALTH_PATH, clock_patch: bool = False, data: str | None = None) -> Any:
    """Declare a WebsiteEnvironment's site. ``data`` names a JSON file the default store starts with."""
    if _cls is not None:
        raise TypeError("use @website(...) with parentheses")
    if not health.startswith("/") or health == "/" or reserved_path(health):
        raise ValueError(f"health is a path of its own under '/', not {health!r}")

    def deco(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, WebsiteEnvironment)):
            raise TypeError(f"@website declares a WebsiteEnvironment subclass, not {cls!r}")
        setattr(cls, _WEBSITE_ATTR, WebsiteDeclaration(tuple(domains), static, health, clock_patch, data))
        _check_health(cls)
        return cls

    return deco


def ready_check(fn: Any) -> Any:
    """Mark a method that says whether the site is usable yet: the health path answers 200 once every one returns a
    truthy value, and 503 while one returns a falsy one or raises."""
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
        own = {getattr(member, _OP_ATTR) for klass in cls.__mro__ for member in vars(klass).values()
               if getattr(member, _OP_ATTR, None) is not None and not getattr(member, _DEFAULT_HOOK_ATTR, False)}
        for op, name in _DEFAULT_HOOKS.items():
            if op in own:
                setattr(cls, name, None)

    def __init__(self) -> None:
        declaration = self.declaration()
        self.store = Collections()
        if declaration.data is not None:
            self.store.load(json.loads(self._site_path(declaration.data).read_text(encoding="utf-8")))
        self.clock = UniverseClock()
        self._signer = SessionSigner()

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

    @property
    def uses_store(self) -> bool:
        """Whether the site's data plane exports ``self.store``: it declares no ``@get_data`` of its own."""
        return type(self)._store_get is not None

    def now(self) -> Any:
        """The universe's time, an aware UTC datetime: the gateway's virtual clock once synced, else wall time."""
        return self._clock().now()

    @property
    def timezone(self) -> str | None:
        """The universe's time zone, an IANA name such as ``America/Chicago``, as the last ``sync_time`` named it;
        None when it named none. ``now()`` is UTC either way."""
        return self._clock().timezone

    def acting_user(self) -> str | None:
        """Who a browser with no session is signed in as: the store's ``current_user`` unless a site says else."""
        return self.store.current_user if hasattr(self, "store") else None

    def context(self, request: Request) -> ApiContext:
        """The ``ApiContext`` of ``request``, for a mounted ``app``'s routes as for ``@api`` ones."""
        session = self._session_signer().decode(request.cookies.get(SESSION_COOKIE))
        user = session["user"] if session is not None else self.acting_user()
        return ApiContext(request, user, request_role(request), session, self._session_signer())

    def create_app(self) -> Starlette:
        if self._mounted:
            raise RuntimeError(f"{type(self).__name__} is already mounted")
        declaration = self.declaration()
        _check_health(type(self))
        api_routes = self.routes()
        taken = {openapi_path(r.path) for r in api_routes}
        root = self._site_path(declaration.static) if declaration.static is not None else None
        asgi = Starlette(middleware=[Middleware(_Forwarded), Middleware(_Universe, env=self)])
        AgentEnvStarletteApplication(self._build_card(), self).add_routes_to_app(asgi)
        asgi.router.routes.append(Route(declaration.health, self._health, methods=["GET"]))
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
            return {"domains": list(declaration.domains), "health": declaration.health,
                    "routes": [r.advertised() for r in self.routes()], "session_cookie": SESSION_COOKIE,
                    "session_path": SESSION_PATH}
        return super().extension_params(uri)

    @extension(CLOCK_EXTENSION_URI, description="Sync this site to the gateway's virtual clock.")
    async def sync_time(self, env_get_time_url: str, timezone: Optional[str] = None) -> dict:
        """Follow the gateway's virtual time at ``env_get_time_url`` from now on, in ``timezone`` when the task names
        one."""
        virtual = await self._clock().sync(env_get_time_url, timezone.strip() if timezone else None)
        return {"env_get_time_url": env_get_time_url, "gateway_virtual_time": iso(virtual),
                **({"timezone": self.timezone} if self.timezone else {})}

    @extension(WEBSITE_EXTENSION_URI, op="openapi", method="GET", description=(
        "A site's server: domains, its health path and its @api routes by name; openapi returns the OpenAPI document "
        "of its routes, ready whether the site is usable now and why not, and session the cookies a browser adds to "
        "start signed in as user, or signed out without one."))
    async def _website_openapi(self) -> dict:
        name = self._application.environment_card.name if self._application is not None else type(self).__name__
        return openapi(name, self.routes())

    @extension(WEBSITE_EXTENSION_URI, op=WEBSITE_READY_OP, method="GET")
    async def _website_ready(self) -> dict:
        """Whether the site is usable now, a ``WebsiteReadiness`` by its ``@ready_check`` methods: ``problems`` names
        each not yet true. A deploy reads it through the card, as ``computer-app/v1``'s ``ready`` of an app."""
        problems = await self._not_ready()
        return WebsiteReadiness(ready=not problems, problems=problems).model_dump()

    @extension(WEBSITE_EXTENSION_URI, op="session")
    async def _website_session(self, user: Optional[str] = None) -> dict:
        cookie = self._session_signer().cookie(user)
        return {"user": user, "cookies": [cookie], "expires_at": expires_at(cookie["expires"])}

    @reset_data
    async def _store_reset(self) -> None:
        self.store.reset()

    @add_data
    async def _store_add(self, parts: list) -> None:
        self.store.load_parts(parts)

    @get_data
    async def _store_get(self) -> list:
        return [DataPart(data=self.store.export())]

    async def _not_ready(self) -> list[str]:
        return [check.__name__ for check, _marker in _marked_members(self, _READY_ATTR) if not await _ready(check)]

    async def _health(self, request: Request) -> Response:
        waiting = await self._not_ready()
        if waiting:
            return JSONResponse({"status": "starting", "waiting_for": waiting}, status_code=503)
        return JSONResponse({"status": "ok"})

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


for _hook in _DEFAULT_HOOKS.values():
    setattr(getattr(WebsiteEnvironment, _hook), _DEFAULT_HOOK_ATTR, True)


async def _ready(check: Any) -> bool:
    try:
        return bool(await _maybe_await(check()))
    except Exception:
        return False


def _check_health(cls: type) -> None:
    health = cls.declaration().health
    clash = [r.name for r in routes_of(cls) if r.method == "GET" and openapi_path(r.path) == health]
    if clash:
        raise ValueError(f"{cls.__name__}'s health path {health} is served by @api {clash[0]}; use @ready_check")


class _Forwarded:
    """Puts the address the browser reached the site at in the scope, from ``X-Forwarded-Host`` and
    ``X-Forwarded-Proto``, before anything routes it, so redirects, ``request.url`` and a mounted app's ``url_for``
    name that address and not the internal one the proxy reached the server at."""

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
        await self.app(scope, receive, send)


def _first(value: bytes | None) -> str:
    return value.decode("latin-1").split(",")[0].strip() if value else ""


class _Universe:
    """Stamps ``X-Universe-Now`` on every answer, and with ``clock_patch`` puts the clock script in its whole HTML (never a range of it), holding
    only the HTML up to its ``<head>`` (at most ``HEAD_SCAN_BYTES``), so a page streamed in chunks reaches the browser
    as it's sent."""

    HEAD_SCAN_BYTES = 64 * 1024

    def __init__(self, app: Any, env: WebsiteEnvironment) -> None:
        self.app, self.env = app, env

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        clock = self.env._clock()
        await clock.refresh()
        patch = self.env.declaration().clock_patch and scope["method"] != "HEAD"
        held: dict = {}

        async def stamped(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                now = clock.now()
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

        await self.app(scope, receive, stamped)


def _with_script(html: bytes, script: bytes) -> bytes:
    head = _HEAD.search(html)
    return html[:head.end()] + script + html[head.end():] if head else script + html


def _unreserved(handler: Any) -> Any:
    """``handler``, an ``@api`` route's endpoint, answering 404 at the data plane's and card's paths, so a broad route
    (``/{rest:path}``) never serves a page there; the routes agentenv registers at those paths answer before it."""
    async def answer(request: Request) -> Response:
        if reserved_path(request.scope["path"]):
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
        path, reading = scope["path"], scope["method"] in ("GET", "HEAD")
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
        if self.root is None:
            return None
        candidate = (self.root / path.lstrip("/")).resolve()
        if candidate.is_dir():
            candidate = (candidate / "index.html").resolve()
        if not candidate.is_relative_to(self.root):
            return None
        return candidate if candidate.is_file() else None

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
