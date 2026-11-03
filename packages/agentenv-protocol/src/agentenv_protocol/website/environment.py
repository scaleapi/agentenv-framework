"""WebsiteEnvironment: an AgentEnvEnvironment that is a site's server, meeting ``urn:agentenv:website/v1``. It serves
plain HTTP on ``PORT`` for any host name: its ``@api`` routes, its static folder at ``/`` with ``index.html`` for the
pages a single-page app routes itself, its health path, and the data plane and card the base class serves. Every answer
carries ``X-Universe-Now``, the universe's time, which ``self.now()`` reads and ``clock/v1``'s ``sync_time`` ties to
the gateway's virtual clock. Without data hooks of its own a site keeps its state in ``self.store``, or with
``@website(models=...)`` in an MCP server's database, which ``self.db`` opens sessions of (``website.database``).

A store-backed site keeps a state per session, the CUA-Gym convention the website state verifier reads: a request's
session is its ``?sid=``, else the one its browser last named, which the site keeps in the ``agentenv_sid_<name>``
cookie, else the default one. ``self.store`` is the request's session's store, each session started as a copy of the
initial state; the default session, ``sid=default``, is the state the data plane loads, exports and resets, and the
one a browser that names no session uses, so a site browsed without ``sid`` works as it did before sessions.

Two routes are built in, each replaced by a site's own ``@api`` route at its very path and answering before one whose
path parameters would also match it: ``/api/session``, which a frontend calls to learn who is signed in, signing the
acting user in when nobody is (GET), or to sign out (DELETE); and, for a store-backed site, ``/go?sid=``, that
session's ``{initial_state, current_state, state_diff}`` at the path ``website/v1``'s ``params.state.path`` names,
which a reader of a card without ``state`` falls back to, with the site's ``changed_sessions``, the named sessions whose
state changed, which data/reset would drop; a session no request has used yet answers the initial state, unchanged. A
store-backed site answers ``website/v1``'s ``state``, of the session its ``sid`` names, with the same ``WebsiteState``,
among agentenv's own routes, which is how the website state verifier reads it."""
from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, Optional
from urllib.parse import unquote

from starlette.applications import Starlette
from starlette.datastructures import MutableHeaders
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette._utils import get_route_path
from starlette.routing import Route

from ..agent_env_environment import (
    _CARD_CONFIG_ATTR,
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
from ..types import DataPart, EnvironmentExtension
from .clock import CLOCK_EXTENSION_URI, UNIVERSE_NOW_HEADER, UniverseClock, clock_patch_script, iso
from .database import ModelState, Sessions, Tables, is_hooks, metadata_of
from .errors import NotFound
from .readiness import WEBSITE_EXTENSION_URI, WEBSITE_READY_OP, WebsiteReadiness
from .routes import (ApiContext, ApiRoute, endpoint, error_response, forwarded_prefix, openapi, openapi_path, request_role,
                     reserved_path, routes_of)
from .session import SessionSigner, expires_at, session_cookie_name, sid_cookie_name
from .state import WEBSITE_STATE_OP, WebsiteState
from .store import DEFAULT_SID, Collections, active_sid, session_id

DEFAULT_HEALTH_PATH = "/api/health"
SESSION_PATH = "/api/session"
STATE_PATH = "/go"
_WEBSITE_ATTR = "_aee_website"
_READY_ATTR = "_aee_website_ready"
_DEFAULT_HOOK_ATTR = "_aee_default_store_hook"
_DEFAULT_HOOKS = {"reset_data": "_store_reset", "add_data": "_store_add", "get_data": "_store_get"}
_HEAD = re.compile(rb"<head[^>]*>", re.IGNORECASE)


class WebsiteDeclaration(NamedTuple):
    """What ``@website`` declares: the site's domains, its static folder and seed data file (relative to the class's
    file), its health path, whether its HTML pages read the universe's time, and the models whose database keeps its
    state instead of the default store."""

    domains: tuple[str, ...] = ()
    static: Optional[str] = None
    health: str = DEFAULT_HEALTH_PATH
    clock_patch: bool = False
    data: Optional[str] = None
    models: Any = None


def website(_cls: type | None = None, *, domains: list[str] | tuple[str, ...] = (), static: str | None = None,
            health: str = DEFAULT_HEALTH_PATH, clock_patch: bool = False, data: str | None = None,
            models: Any = None) -> Any:
    """Declare a WebsiteEnvironment's site. ``data`` names a JSON file the default store starts with; ``models``
    keeps the site's state in a database instead, an MCP server's (see ``website.database``), and the site's
    data plane and ``/go`` are that database's."""
    if _cls is not None:
        raise TypeError("use @website(...) with parentheses")
    if not health.startswith("/") or health == "/" or reserved_path(health):
        raise ValueError(f"health is a path of its own under '/', not {health!r}")
    if models is not None and data is not None:
        raise ValueError("data= seeds the default store; a site of models= is seeded through data/add")
    if models is not None and not is_hooks(models):
        metadata_of(models)

    def deco(cls: type) -> type:
        if not (isinstance(cls, type) and issubclass(cls, WebsiteEnvironment)):
            raise TypeError(f"@website declares a WebsiteEnvironment subclass, not {cls!r}")
        setattr(cls, _WEBSITE_ATTR, WebsiteDeclaration(tuple(domains), static, health, clock_patch, data, models))
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
    db: Any = None
    _store: Collections

    @property
    def store(self) -> Collections:
        """The state store of the request's session: the default session's outside a request."""
        return self._store.session(active_sid.get())

    @store.setter
    def store(self, value: Collections) -> None:
        self._store = value

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
            self.store.mark_initial()
        self.clock = UniverseClock()
        self._signer = SessionSigner(cookie_name=session_cookie_name(self.environment_name()))
        models = declaration.models
        self._tables = (None if models is None or is_hooks(models)
                        else Tables(metadata_of(models), lambda: self.database_url))
        self._models = None if models is None else ModelState(models if self._tables is None else self._tables)
        self.db = Sessions(self._tables) if self._tables is not None else None

    @classmethod
    def declaration(cls) -> WebsiteDeclaration:
        return getattr(cls, _WEBSITE_ATTR, None) or WebsiteDeclaration()

    @classmethod
    def routes(cls) -> list[ApiRoute]:
        return routes_of(cls)

    def environment_name(self) -> str:
        """The site's name, as its card names it: ``ENVIRONMENT_NAME``, else the card's declared name, else the
        class's."""
        declared = (getattr(self, _CARD_CONFIG_ATTR, None) or {}).get("name")
        return os.environ.get("ENVIRONMENT_NAME") or declared or type(self).__name__

    @property
    def session_cookie(self) -> str:
        """The name of the site's session cookie, ``agentenv_session_<environment_name>``."""
        return self._session_signer().cookie_name

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
        """Who a browser with no session is signed in as: the store's ``current_user``, or the one a models site's data
        named, unless a site says else."""
        if getattr(self, "_models", None) is not None:
            return self._models.current_user
        return self.store.current_user if hasattr(self, "store") else None

    def context(self, request: Request) -> ApiContext:
        """The ``ApiContext`` of ``request``, for a mounted ``app``'s routes as for ``@api`` ones."""
        session = self._session_signer().read(request.cookies)
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
        asgi = Starlette(middleware=[Middleware(_Forwarded), Middleware(_Universe, env=self),
                                     Middleware(_Session, cookie=sid_cookie_name(self.environment_name()))])
        AgentEnvStarletteApplication(self._build_card(), self).add_routes_to_app(asgi)
        asgi.router.routes.append(Route(declaration.health, self._health, methods=["GET"]))
        if SESSION_PATH not in taken:
            asgi.router.routes.append(Route(SESSION_PATH, self._session, methods=["GET", "DELETE"]))
        if self._serves_state_path():
            asgi.router.routes.append(Route(STATE_PATH, self._state, methods=["GET"]))
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
                    "routes": [r.advertised() for r in self.routes()], "session_cookie": self.session_cookie,
                    "session_path": SESSION_PATH, **({"state": {"path": STATE_PATH}} if self._serves_state_path() else {})}
        return super().extension_params(uri)

    def served_extensions(self, extensions: list[EnvironmentExtension]) -> list[EnvironmentExtension]:
        """The extensions the base class serves, ``website/v1``'s ``state`` only for a store-backed site: one with data
        hooks of its own keeps its state where ``self.store`` can't see it."""
        served = super().served_extensions(extensions)
        if self.uses_store:
            return served
        return [_without_method(e, WEBSITE_STATE_OP) if e.uri == WEBSITE_EXTENSION_URI else e for e in served]

    @extension(CLOCK_EXTENSION_URI, description="Sync this site to the gateway's virtual clock.")
    async def sync_time(self, env_get_time_url: str, timezone: Optional[str] = None) -> dict:
        """Follow the gateway's virtual time at ``env_get_time_url`` from now on, in ``timezone`` when the task names
        one."""
        virtual = await self._clock().sync(env_get_time_url, timezone.strip() if timezone else None)
        return {"env_get_time_url": env_get_time_url, "gateway_virtual_time": iso(virtual),
                **({"timezone": self.timezone} if self.timezone else {})}

    @extension(WEBSITE_EXTENSION_URI, op="openapi", method="GET", description=(
        "A site's server: domains, its health path and its @api routes by name; openapi returns the OpenAPI document "
        "of its routes, ready whether the site is usable now and why not, state the task state it is in (its "
        "initial and current state and their diff), and session the cookies a browser adds to start signed in as user, "
        "or signed out without one."))
    async def _website_openapi(self) -> dict:
        name = self._application.environment_card.name if self._application is not None else type(self).__name__
        return openapi(name, self.routes())

    @extension(WEBSITE_EXTENSION_URI, op=WEBSITE_READY_OP, method="GET")
    async def _website_ready(self) -> dict:
        """Whether the site is usable now, a ``WebsiteReadiness`` by its ``@ready_check`` methods: ``problems`` names
        each not yet true. A deploy reads it through the card, as ``computer-app/v1``'s ``ready`` of an app."""
        problems = await self._not_ready()
        return WebsiteReadiness(ready=not problems, problems=problems).model_dump()

    @extension(WEBSITE_EXTENSION_URI, op=WEBSITE_STATE_OP, method="GET")
    async def _website_state(self, sid: Optional[str] = None) -> dict:
        """The site's task state, a ``WebsiteState``: its state when the task started, now, and what changed between
        them, of session ``sid``'s store (the default session's without one), with the site's ``changed_sessions`` as
        ``/go`` names them, or of a models site's database, which keeps one."""
        if self._models is not None:
            return WebsiteState.model_validate(await self._models.state()).model_dump()
        return {**WebsiteState.model_validate(self._store.session(sid).state()).model_dump(),
                "changed_sessions": self._store.changed_sessions()}

    @extension(WEBSITE_EXTENSION_URI, op="session")
    async def _website_session(self, user: Optional[str] = None) -> dict:
        cookie = self._session_signer().cookie(user)
        return {"user": user, "cookies": [cookie], "expires_at": expires_at(cookie["expires"])}

    @reset_data
    async def _store_reset(self) -> None:
        if self._models is not None:
            return await self._models.reset()
        self._store.reset()

    @add_data
    async def _store_add(self, parts: list) -> None:
        if self._models is not None:
            return await self._models.load_parts(parts)
        self._store.load_parts(parts)
        self._store.mark_initial()

    @get_data
    async def _store_get(self) -> list:
        return [DataPart(data=await self._models.export() if self._models is not None else self._store.export())]

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

    def _serves_state_path(self) -> bool:
        """Whether the site serves its state at ``STATE_PATH``: it is store-backed and no ``@api`` route takes the path."""
        return self.uses_store and STATE_PATH not in {openapi_path(r.path) for r in self.routes()}

    async def _state(self, request: Request) -> Response:
        if self._models is not None:
            return JSONResponse(await self._models.state())
        return JSONResponse({**self.store.state(), "changed_sessions": self._store.changed_sessions()})

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


def _without_method(extension: EnvironmentExtension, method: str) -> EnvironmentExtension:
    params = dict(extension.params or {})
    params["methods"] = {name: entry for name, entry in (params.get("methods") or {}).items() if name != method}
    return extension.model_copy(update={"params": params})


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


class _Session:
    """Runs the request in its session (``active_sid``): its ``sid`` query parameter's, else its ``cookie``'s; an
    answer to a request naming a session other than its cookie's sets the cookie, so a page opened at ``?sid=`` keeps
    its session for the requests it makes after."""

    def __init__(self, app: Any, cookie: str) -> None:
        self.app, self.cookie = app, cookie

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        named, kept = request.query_params.get("sid"), request.cookies.get(self.cookie)
        sid = session_id(named if named is not None else kept)

        async def remembered(message: dict) -> None:
            if message["type"] == "http.response.start" and named is not None and session_id(kept) != sid:
                MutableHeaders(scope=message).append(
                    "set-cookie", f"{self.cookie}={sid or DEFAULT_SID}; Path=/; HttpOnly; SameSite=Lax")
            await send(message)

        token = active_sid.set(sid)
        try:
            await self.app(scope, receive, remembered)
        finally:
            active_sid.reset(token)


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
        path, reading = get_route_path(scope) or "/", scope["method"] in ("GET", "HEAD")
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
