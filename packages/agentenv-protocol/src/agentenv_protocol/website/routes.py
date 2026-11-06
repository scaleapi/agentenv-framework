"""A site's API: ``@api.get/post/put/patch/delete(path)`` methods, each bound from its type hints. A parameter named in
the path is a path parameter, one annotated with a pydantic model or ``dict`` is the JSON body, one annotated
``ApiContext`` receives the request's context, and every other one is a query parameter; path and query values are
coerced to their annotated types (int, float, bool, datetime, UUID, enum, Literal). What the method returns is the
JSON answer, a Starlette ``Response`` as it is; an ``ApiError`` it raises answers with its status, and a request its
parameters can't be bound from answers 400. A plain ``def`` method runs in a worker thread, as FastAPI runs one, so a
blocking call in it (a database query, say) holds up only its own request; an ``async def`` one runs on the event
loop."""
from __future__ import annotations

import dataclasses
import hmac
import inspect
import json
import logging
import os
import posixpath
import re
from typing import Any, Callable, Literal, NamedTuple, Optional, get_args, get_origin, get_type_hints

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..agent_env_environment import _coerce_scalar, _maybe_await, _schema_for_annotation, _unwrap_annotation
from ..tool_context import DEFAULT_ROLE, caller_from_request
from ..types import RPC_PATH, WELL_KNOWN_PATH
from .errors import ApiError, BadRequest, Forbidden
from .session import SESSION_TTL_S, SessionSigner

logger = logging.getLogger(__name__)

_ROUTE_ATTR = "_aee_api_route"
_PATH_PARAM = re.compile(r"{(\w+)(?::\w+)?}")
Kind = Literal["path", "query", "body", "context"]


def reserved_path(path: str) -> bool:
    """Whether ``path`` is the data plane's or the card's, however it's spelled, as the gateway decides it."""
    normalized = posixpath.normpath("/" + path.lstrip("/"))
    return normalized == RPC_PATH or normalized.startswith(RPC_PATH + "/") or normalized == WELL_KNOWN_PATH


class ApiRoute(NamedTuple):
    """One ``@api`` method: its verb, its Starlette path, its name on the card and in the OpenAPI document, the roles
    it answers (any when empty), the status it succeeds with, and its function."""

    method: str
    path: str
    name: str
    roles: tuple[str, ...]
    status: int
    fn: Callable

    def advertised(self) -> dict:
        """The route as the card names it: its path is the Starlette path it is served at, converters and all
        (``/files/{rest:path}``), so a gateway matches the requests it answers, nested paths included."""
        return {"method": self.method, "path": self.path, "name": self.name, "roles": list(self.roles) or None}


class _Api:
    """``api.get(path, name=..., roles=[...])`` and its siblings mark a method as one route of the site's API."""

    def get(self, path: str, **options: Any) -> Callable:
        return self.route("GET", path, **options)

    def post(self, path: str, **options: Any) -> Callable:
        return self.route("POST", path, **options)

    def put(self, path: str, **options: Any) -> Callable:
        return self.route("PUT", path, **options)

    def patch(self, path: str, **options: Any) -> Callable:
        return self.route("PATCH", path, **options)

    def delete(self, path: str, **options: Any) -> Callable:
        return self.route("DELETE", path, **options)

    def route(self, method: str, path: str, *, name: str | None = None, roles: list[str] | None = None,
              status: int = 200) -> Callable:
        if not path.startswith("/"):
            raise ValueError(f"an @api path starts with '/', not {path!r}")
        if reserved_path(path):
            raise ValueError(f"@api path {path!r} is under the data plane's {RPC_PATH} or the card's {WELL_KNOWN_PATH}")

        def mark(fn: Callable) -> Callable:
            missing = sorted(set(_PATH_PARAM.findall(path)) - set(inspect.signature(fn).parameters))
            if missing:
                raise TypeError(f"{fn.__qualname__} takes no parameter for the path's {missing}")
            setattr(fn, _ROUTE_ATTR, ApiRoute(method.upper(), path, name or fn.__name__, tuple(roles or ()), status, fn))
            return fn

        return mark


api = _Api()


def routes_of(cls: type) -> list[ApiRoute]:
    """The class's ``@api`` routes in the order they were first defined, each as the class resolves its name;
    a name or a verb and path served twice is refused."""
    names = dict.fromkeys(name for klass in reversed(cls.__mro__) for name in vars(klass))
    routes = [route for name in names if (route := getattr(getattr(cls, name, None), _ROUTE_ATTR, None)) is not None]
    for label, keys in (("name", [r.name for r in routes]), ("route", [f"{r.method} {r.path}" for r in routes])):
        if twice := sorted({k for k in keys if keys.count(k) > 1}):
            raise ValueError(f"{cls.__name__} serves the @api {label} {twice} twice")
    return routes


def openapi_path(path: str) -> str:
    return _PATH_PARAM.sub(r"{\1}", path)


@dataclasses.dataclass
class ApiContext:
    """The request an ``@api`` method answers: its headers, cookies and query, the user it's signed in as (the
    session cookie's, else the site's acting user), the caller's role (``request_role``), and the site's own address as the browser reached it. ``sign_in`` and ``sign_out`` set the
    session cookie on the answer."""

    request: Request
    user: Optional[str]
    role: str
    session: Optional[dict]
    _signer: SessionSigner = dataclasses.field(repr=False, default_factory=SessionSigner)
    _set_cookie: list = dataclasses.field(repr=False, default_factory=list)

    @property
    def headers(self) -> Any:
        return self.request.headers

    @property
    def cookies(self) -> dict[str, str]:
        return self.request.cookies

    @property
    def query(self) -> Any:
        return self.request.query_params

    @property
    def base_url(self) -> str:
        return base_url(self.request)

    def sign_in(self, user: str) -> None:
        self._set_session(user)

    def sign_out(self) -> None:
        self._set_session(None)

    def _set_session(self, user: str | None) -> None:
        self._set_cookie.append(self._signer.encode(user))
        self.user, self.session = user, {"user": user}

    def apply(self, response: Response) -> Response:
        for value in self._set_cookie:
            response.set_cookie(self._signer.cookie_name, value, max_age=SESSION_TTL_S, path="/", httponly=True,
                                samesite="lax")
        return response


def base_url(request: Request) -> str:
    """The site's address as the browser reached it: ``X-Forwarded-Proto`` and ``X-Forwarded-Host`` when a proxy
    sent them, the request's own scheme and ``Host`` otherwise, under the path ``X-Forwarded-Prefix`` names when the
    proxy serves the site below one (the gateway's ``/website/<name>``)."""
    proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip() or request.url.scheme
    host = (request.headers.get("x-forwarded-host", "").split(",")[0].strip() or request.headers.get("host")
            or request.url.netloc)
    return f"{proto}://{host}{forwarded_prefix(request.headers.get('x-forwarded-prefix', ''))}"


def forwarded_prefix(value: str) -> str:
    """The path an ``X-Forwarded-Prefix`` value names, without its trailing slash; empty for none, and for one that is
    not an absolute path or would make the site's URLs name another host (``//host``).

    >>> [forwarded_prefix(v) for v in ("/website/notes/", "/a, /b", "", "/", "website", "//evil.example", "/a//b")]
    ['/website/notes', '/a', '', '', '', '', '']
    """
    prefix = value.split(",")[0].strip().rstrip("/")
    return prefix if prefix.startswith("/") and "//" not in prefix and prefix.isprintable() else ""


ROLE_KEY_ENV = "WEBSITE_ROLE_KEY"
ROLE_KEY_HEADER = "AgentEnv-Role-Key"


def request_role(request: Request) -> str:
    """The role ``AgentEnv-Role`` names, believed only from a caller that also sends ``AgentEnv-Role-Key`` equal to
    ``WEBSITE_ROLE_KEY``; every other caller, and every caller of a site with no key set, is ``default``. The gateway
    forwards whatever role a browser sends, so without the key a page could claim any role."""
    claimed = caller_from_request(None, request.headers).role or DEFAULT_ROLE
    key = os.environ.get(ROLE_KEY_ENV, "")
    if claimed == DEFAULT_ROLE or not key:
        return DEFAULT_ROLE
    given = request.headers.get(ROLE_KEY_HEADER, "")
    return claimed if hmac.compare_digest(given.encode(), key.encode()) else DEFAULT_ROLE


class _Param(NamedTuple):
    name: str
    kind: Kind
    annotation: Any
    default: Any


def plan(route: ApiRoute) -> list[_Param]:
    """How each of the route's parameters is bound; at most one is the JSON body, and never on a GET."""
    hints = _hints(route.fn)
    path_params = set(_PATH_PARAM.findall(route.path))
    params = []
    for name, param in list(inspect.signature(route.fn).parameters.items())[1:]:
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        kind: Kind = ("context" if _is_context(annotation) else "path" if name in path_params
                      else "body" if _body_type(annotation) is not None else "query")
        if kind == "query" and _fixed_tuple(annotation):
            raise TypeError(f"{route.fn.__qualname__}: query parameter {name!r} is a fixed-length tuple, which repeated "
                            f"query values can't fill; take tuple[T, ...] or list[T]")
        params.append(_Param(name, kind, annotation, param.default))
    bodies = [p.name for p in params if p.kind == "body"]
    if len(bodies) > 1 or (bodies and route.method == "GET"):
        raise TypeError(f"{route.fn.__qualname__}: a route takes at most one JSON body and a GET route none, "
                        f"not {bodies}")
    return params


def _is_context(annotation: Any) -> bool:
    if isinstance(annotation, str):
        return annotation.strip("'\"").rsplit(".", 1)[-1] in ("ApiContext", "Optional[ApiContext]")
    return any(t is ApiContext for t in _unwrap_annotation(annotation))


def _body_type(annotation: Any) -> type | None:
    for t in _unwrap_annotation(annotation):
        base = get_origin(t) or t
        if base is dict or (isinstance(base, type) and issubclass(base, BaseModel)):
            return t
    return None


async def bind(params: list[_Param], request: Request, context: ApiContext) -> dict:
    """The method's arguments from ``request``; raises BadRequest for one that is missing or doesn't fit."""
    kwargs: dict = {}
    for p in params:
        if p.kind == "context":
            kwargs[p.name] = context
        elif p.kind == "body":
            kwargs[p.name] = await _body(p, request)
        else:
            source = request.path_params if p.kind == "path" else request.query_params
            if p.name not in source:
                if p.default is inspect.Parameter.empty:
                    raise BadRequest(f"missing {p.kind} parameter {p.name!r}", code="invalid_params")
                kwargs[p.name] = p.default
                continue
            many = _many_type(p.annotation) if p.kind == "query" else None
            try:
                kwargs[p.name] = (many[0](_coerce_scalar(v, many[1]) for v in source.getlist(p.name)) if many is not None
                                  else _coerce_scalar(source[p.name], p.annotation))
            except (TypeError, ValueError) as e:
                raise BadRequest(f"{p.kind} parameter {p.name!r}: {e}", code="invalid_params") from None
    return kwargs


_COLLECTIONS = (list, set, frozenset, tuple)


def _many_type(annotation: Any) -> tuple[type, Any] | None:
    """The collection a query parameter's annotation names, Optional or Annotated or not, and its items' type (None
    when it names none): ``list[T]``, ``set[T]``, ``frozenset[T]`` or ``tuple[T, ...]``; None for anything else."""
    inner = _unwrap_annotation(annotation)
    if len(inner) != 1:
        return None
    container = get_origin(inner[0]) or inner[0]
    if container not in _COLLECTIONS:
        return None
    args = get_args(inner[0])
    if container is tuple and args and not (len(args) == 2 and args[1] is Ellipsis):
        return None
    return container, args[0] if args else None


def _fixed_tuple(annotation: Any) -> bool:
    inner = _unwrap_annotation(annotation)
    return len(inner) == 1 and get_origin(inner[0]) is tuple and _many_type(annotation) is None


async def _body(p: _Param, request: Request) -> Any:
    raw = await request.body()
    if not raw:
        if p.default is not inspect.Parameter.empty:
            return p.default
        raise BadRequest("this route takes a JSON body", code="invalid_params")
    try:
        value = json.loads(raw)
    except ValueError:
        raise BadRequest("the request body is not JSON", code="invalid_params") from None
    try:
        return TypeAdapter(p.annotation).validate_python(value)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, err['loc'])) or 'body'}: {err['msg']}" for err in e.errors())
        raise BadRequest(f"invalid body: {problems}", code="invalid_params") from None


def endpoint(fn: Callable, route: ApiRoute, context_for: Callable[[Request], ApiContext]) -> Callable:
    """The Starlette endpoint of one route, calling the bound method ``fn``, a plain ``def`` one in a worker thread.
    The request's session cookies go on a successful answer only."""
    params = plan(route)
    threaded = not inspect.iscoroutinefunction(fn)

    async def handle(request: Request) -> Response:
        context = None
        try:
            context = context_for(request)
            if route.roles and context.role not in route.roles:
                raise Forbidden(f"role {context.role!r} may not call {route.name}")
            kwargs = await bind(params, request, context)
            result = await _maybe_await(await run_in_threadpool(fn, **kwargs) if threaded else fn(**kwargs))
            if isinstance(result, Response):
                response = result
            elif not _has_body(route.status):
                response = Response(status_code=route.status)
            else:
                response = JSONResponse(to_jsonable_python(result), status_code=route.status)
        except ApiError as e:
            response = error_response(e)
        except Exception:
            logger.exception("@api %s failed", route.name)
            response = error_response(ApiError("internal error", code="internal"))
        return context.apply(response) if context is not None and response.status_code < 400 else response

    return handle


def _has_body(status: int) -> bool:
    """Whether an answer with ``status`` may carry a body: not a 1xx, 204 or 304."""
    return status >= 200 and status not in (204, 304)


def error_response(error: ApiError) -> Response:
    return JSONResponse(error.body(), status_code=error.status)


def openapi(title: str, routes: list[ApiRoute]) -> dict:
    """The OpenAPI 3.1 document of ``routes``, each operation's id its route's name; the models their bodies and
    answers name, nested ones too, are its ``components.schemas``, which their schemas reference. Bodies are
    described as they are validated and answers as they are serialized; a model the two modes see differently is in
    ``components.schemas`` twice, as ``<Model>-Input`` and ``<Model>-Output``."""
    paths: dict = {}
    schemas: list = []
    for route in routes:
        doc = inspect.getdoc(route.fn) or ""
        operation: dict = {"operationId": route.name, "parameters": [],
                           "responses": {str(route.status): {"description": "Success"}}}
        if doc:
            operation["summary"], operation["description"] = doc.splitlines()[0], doc
        if route.roles:
            operation["x-agentenv-roles"] = list(route.roles)
        for p in plan(route):
            if p.kind in ("path", "query"):
                many = _many_type(p.annotation) if p.kind == "query" else None
                schema = _schema_for_annotation(p.annotation) if many is None else {
                    "type": "array", **({"items": _schema_for_annotation(many[1])} if many[1] is not None else {}),
                    **({"uniqueItems": True} if many[0] in (set, frozenset) else {})}
                if p.default is not inspect.Parameter.empty:
                    schema["default"] = to_jsonable_python(p.default)
                operation["parameters"].append({"name": p.name, "in": p.kind, "schema": schema,
                                                "required": p.kind == "path" or p.default is inspect.Parameter.empty})
            elif p.kind == "body":
                media: dict = {}
                operation["requestBody"] = {"required": p.default is inspect.Parameter.empty,
                                            "content": {"application/json": media}}
                schemas.append((media, p.annotation, "validation"))
        returns = _hints(route.fn).get("return")
        if _has_body(route.status) and _adapter(returns) is not None:
            media = {}
            operation["responses"][str(route.status)]["content"] = {"application/json": media}
            schemas.append((media, returns, "serialization"))
        paths.setdefault(openapi_path(route.path), {})[route.method.lower()] = operation
    document: dict = {"openapi": "3.1.0", "info": {"title": title, "version": "1.0"}, "paths": paths}
    adapters = [(i, mode, adapter) for i, (_media, annotation, mode) in enumerate(schemas)
                if (adapter := _adapter(annotation)) is not None]
    found, definitions = TypeAdapter.json_schemas(adapters, ref_template="#/components/schemas/{model}")
    for i, (media, _annotation, mode) in enumerate(schemas):
        media["schema"] = found.get((i, mode), {})
    if definitions.get("$defs"):
        document["components"] = {"schemas": definitions["$defs"]}
    return document


def _hints(fn: Callable) -> dict:
    """``fn``'s type hints; a ``TypeError`` naming the route when one can't be resolved, as one naming a type imported
    only under ``TYPE_CHECKING`` can't, rather than binding every parameter as an unresolved string."""
    try:
        return get_type_hints(fn)
    except Exception as e:
        raise TypeError(f"@api {fn.__qualname__}: its type hints can't be resolved ({type(e).__name__}: {e}); "
                        "every type its parameters and answer name must be importable when the site runs") from e


def _adapter(annotation: Any) -> TypeAdapter | None:
    """The adapter of a type a JSON schema can describe; None for no type, None itself, or one it can't describe."""
    if annotation is None or annotation is type(None) or annotation is inspect.Parameter.empty:
        return None
    try:
        adapter = TypeAdapter(annotation)
        adapter.json_schema()
    except Exception:
        return None
    return adapter
