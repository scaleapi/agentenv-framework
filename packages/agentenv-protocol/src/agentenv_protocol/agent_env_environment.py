"""Server SDK: decorate a handler's methods with @reset_data/@add_data/@get_data (JSON-RPC data plane at /agentenv; each optional, advertised as capabilities.operations), @extension (card-advertised REST routes), and @tool (MCP tools; FastMCP-backed apps only). Mount onto a FastMCP, Starlette, or FastAPI app via a per-framework AgentEnvApplication subclass — or subclass AgentEnvEnvironment, configure its card with @environment_card(...), and serve() it."""
from __future__ import annotations

import datetime
import decimal
import enum
import inspect
import json
import logging
import os
import types
import uuid
from abc import ABC, abstractmethod
from typing import Annotated, Any, Callable, ClassVar, Literal, NamedTuple, Union, get_args, get_origin, get_type_hints

from pydantic import ValidationError
from pydantic.fields import FieldInfo
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from .transfers import WriteNamespaceGrant
from .types import DATA_OBJECTS_EXTENSION_URI, MCP_TRANSPORT, METHOD_ADD, METHOD_GET, METHOD_RESET, RPC_PATH, WELL_KNOWN_PATH, AddDataRequest, AddDataResponse, EnvironmentCapabilities, EnvironmentCard, EnvironmentExtension, EnvironmentInterface, EnvironmentTool, GetDataResponse, ResetDataResponse, error_body

logger = logging.getLogger(__name__)

_OP_ATTR = "_aee_op"
_EXT_ATTR = "_aee_ext"
_EXT_ROUTE_ATTR = "_aee_ext_route"
_TOOL_ATTR = "_aee_tool"
_CARD_CONFIG_ATTR = "_aee_card_config"

OP_RESET_DATA = "reset_data"
OP_ADD_DATA = "add_data"
OP_GET_DATA = "get_data"

RPC_ERROR_CODE_PARSE_ERROR = -32700
RPC_ERROR_CODE_INVALID_REQUEST = -32600
RPC_ERROR_CODE_METHOD_NOT_FOUND = -32601
RPC_ERROR_CODE_INVALID_PARAMS = -32602
RPC_ERROR_CODE_SERVER_ERROR = -32000


def reset_data(fn: Callable) -> Callable:
    setattr(fn, _OP_ATTR, OP_RESET_DATA)
    return fn


def add_data(fn: Callable) -> Callable:
    setattr(fn, _OP_ATTR, OP_ADD_DATA)
    return fn


def get_data(fn: Callable) -> Callable:
    """Mark the ``data/get`` handler. One that takes a ``write_namespace`` parameter is advertised under
    ``DATA_OBJECTS_EXTENSION_URI`` and called with the grant a caller sends, else None; it may upload its
    export under the grant and return ``uploaded_file_part``."""
    setattr(fn, _OP_ATTR, OP_GET_DATA)
    return fn


def extension(uri: str, *, description: str | None = None, params: dict | None = None,
              required: bool | None = None, method: str = "POST", path: str | None = None) -> Callable:
    """Mark a handler method as the invocation handler for a card extension.

    The extension is served as its own REST route and auto-advertised on the EnvironmentCard,
    mirroring A2A's extension shape. By default the route is ``<RPC_PATH>/ext/<method-name>``
    (the handler's name, verbatim) with HTTP ``POST``; pass ``path`` / ``method`` to override.
    The call's params are bound to the handler's named parameters (``fn(**params)``): the JSON body
    for POST, or the query string for GET. Either way the values are coerced to the handler's
    annotated types (datetime/date/UUID/Decimal/enum/bool/int/float) so the handler receives the
    runtime types the advertised schema implies. The return value is the JSON result (None -> {}).

    The advertised ``params`` are A2A-shaped::

        {"endpoint": <path>, "methods": {<op>: {"method": <verb>, "request": <JSON-Schema>}}}

    where the request JSON-Schema is derived from the signature. Pass an explicit ``params`` to
    override the advertisement. A handler that wants the raw params dict can declare ``**params``.
    """
    # Normalize the HTTP verb once, here, so the advertised card method, the
    # route registration, and the handler's request-binding (which checks for
    # the exact string "GET") all agree. Without this, ``@extension(method="get")``
    # advertises/invokes GET while the handler parses the request body instead of
    # query params, so GET extensions with required query params return 400.
    http_method = method.upper()

    def deco(fn: Callable) -> Callable:
        op = fn.__name__ or "invoke"
        route_path = path or f"{RPC_PATH}/ext/{op}"
        advertised = params if params is not None else {
            "endpoint": route_path,
            "methods": {op: {"method": http_method, "request": _schema_from_signature(fn)}},
        }
        setattr(fn, _EXT_ATTR, EnvironmentExtension(uri=uri, description=description, params=advertised, required=required))
        setattr(fn, _EXT_ROUTE_ATTR, (route_path, http_method))
        return fn

    return deco


def tool(name: str | None = None, *, description: str | None = None) -> Callable:
    """Mark a handler method as an MCP tool.

    At mount the bound method is registered as a real MCP tool on the server app and advertised
    on the EnvironmentCard under ``capabilities.tools`` (name, description, signature-derived
    ``inputSchema`` including ``Annotated[..., Field]`` param descriptions). ``name`` defaults to
    the method name; a literal ``{environment_name}`` token in an explicit name is resolved to the
    environment card's name at mount (any other ``{...}`` token raises). ``description`` defaults to
    the docstring and is used verbatim on both
    the card and the registration. Requires a FastMCP-backed application (non-FastMCP apps raise
    at construction). Duplicate resolved names raise; a card-declared name wins the advertisement
    while the method is still registered.
    """
    if callable(name):
        raise TypeError("use @tool(...) with parentheses, not bare @tool")

    def deco(fn: Callable) -> Callable:
        setattr(fn, _TOOL_ATTR, EnvironmentTool(
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn),
            inputSchema=_schema_from_signature(fn),
        ))
        return fn

    return deco


def environment_card(_cls: type | None = None, **config: Any) -> Callable:
    """Configure the class's EnvironmentCard; kwargs are EnvironmentCard fields.

    The SDK assembles the card at mount: ENVIRONMENT_NAME wins, then a card-declared name,
    then the class name; decorator-discovered tools/extensions/operations are merged in.
    agent-env injects ENVIRONMENT_NAME as the env's registered name, so the card follows a
    registration that overrides the code's name. SERVICE_NAME is no longer consulted.
    """
    if _cls is not None:
        raise TypeError("use @environment_card(...) with parentheses")
    unknown = set(config) - set(EnvironmentCard.model_fields)
    if unknown:
        raise TypeError(f"unknown EnvironmentCard field(s): {sorted(unknown)}")

    def deco(cls: type) -> type:
        setattr(cls, _CARD_CONFIG_ATTR, config)
        return cls

    return deco


class AgentEnvApplication(ABC):
    """Holds the environment card + handler and builds the JSON-RPC data-plane routes. Subclasses mount them on a specific server framework."""

    supports_tools: ClassVar[bool] = False

    def __init__(self, environment_card: EnvironmentCard, handler: Any) -> None:
        self.handler = handler
        ext_handlers = _discover_extensions(handler)
        environment_card = _merge_extensions(environment_card, ext_handlers)
        tool_handlers = _discover_tools(handler, environment_card.name)
        if tool_handlers and not self.supports_tools:
            raise ValueError(f"@tool methods require a FastMCP-backed application; {type(self).__name__} cannot serve MCP tools")
        environment_card = _merge_tools(environment_card, tool_handlers)
        methods = _discover_methods(handler)
        environment_card = _merge_operations(environment_card, methods)
        environment_card = _merge_data_objects(environment_card, methods)
        self.environment_card = environment_card
        self._dispatch = _jsonrpc_handler(methods)
        self._ext_routes = [(path, method, _extension_handler(fn, method)) for _descriptor, fn, path, method in ext_handlers]
        # Register with the advertised descriptor (card-declared wins in _merge_tools) so
        # tools/list cannot drift from the card; falls back to the discovered descriptor.
        caps_tools = (environment_card.capabilities.tools if environment_card.capabilities else None) or []
        advertised = {t.name: t for t in caps_tools}
        self._tools = [(advertised.get(descriptor.name, descriptor), fn) for descriptor, fn in tool_handlers]

    def add_routes_to_app(self, app: Any, *, rpc_url: str = RPC_PATH, card_url: str = WELL_KNOWN_PATH) -> None:
        self.environment_card = self._card_for(app)
        self._add_route(app, rpc_url, ["POST"], self._dispatch)
        self._add_route(app, card_url, ["GET"], _card_handler(self.environment_card))
        for path, method, handler in self._ext_routes:
            self._add_route(app, path, [method], handler)
        for descriptor, fn in self._tools:
            self._register_tool(app, descriptor, fn)

    @abstractmethod
    def _add_route(self, app: Any, path: str, methods: list, handler: Callable) -> None:
        ...

    def _register_tool(self, app: Any, descriptor: EnvironmentTool, fn: Callable) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not serve MCP tools")

    def _card_for(self, app: Any) -> EnvironmentCard:
        return self.environment_card


class AgentEnvFastMCPApplication(AgentEnvApplication):
    supports_tools: ClassVar[bool] = True

    def _add_route(self, app: Any, path: str, methods: list, handler: Callable) -> None:
        app.custom_route(path, methods=methods)(handler)

    def _register_tool(self, app: Any, descriptor: EnvironmentTool, fn: Callable) -> None:
        app.tool(name=descriptor.name, description=descriptor.description)(fn)

    def _card_for(self, app: Any) -> EnvironmentCard:
        """Declare the MCP endpoint only when the app says where it serves it; FastMCP-shaped targets may not."""
        path = getattr(getattr(app, "settings", None), "streamable_http_path", None)
        return _merge_mcp_interface(self.environment_card, path) if path else self.environment_card


class AgentEnvStarletteApplication(AgentEnvApplication):
    def _add_route(self, app: Any, path: str, methods: list, handler: Callable) -> None:
        app.router.routes.append(Route(path, handler, methods=methods))


def create_fastmcp_app(handler: Any, *, card: EnvironmentCard, **fastmcp_kwargs: Any) -> Any:
    """FastMCP app wired for the agent-env deploy contract, with ``handler`` mounted under ``card``; returned un-served."""
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise ImportError("create_fastmcp_app requires the 'mcp' package: pip install 'mcp>=1.25,<2'") from e
    app = FastMCP(card.name, **fastmcp_kwargs)
    app.settings.host = os.environ.get("MCP_HOST", "0.0.0.0")
    app.settings.port = int(os.environ.get("MCP_PORT", "18765"))
    app.settings.transport_security.enable_dns_rebinding_protection = False
    AgentEnvFastMCPApplication(card, handler).add_routes_to_app(app)
    return app


class AgentEnvEnvironment:
    """MCP environment server base: decorate with ``@environment_card(...)``, decorate methods, ``serve()``.

    ``serve()`` builds the app via ``create_fastmcp_app`` unless the subclass set its own
    ``self.mcp`` — then it mounts if needed and runs it as-is, never altering caller settings.
    """

    mcp: Any = None
    _mounted: bool = False

    def _build_card(self) -> EnvironmentCard:
        config = dict(getattr(self, _CARD_CONFIG_ATTR, None) or {})
        name = os.environ.get("ENVIRONMENT_NAME") or config.get("name") or type(self).__name__
        return EnvironmentCard(**{**config, "name": name})

    def create_app(self) -> Any:
        if self._mounted:
            raise RuntimeError(f"{type(self).__name__} is already mounted")
        self.mcp = create_fastmcp_app(self, card=self._build_card())
        self._mounted = True
        return self.mcp

    def mount(self, app: Any) -> Any:
        if self._mounted:
            raise RuntimeError(f"{type(self).__name__} is already mounted")
        AgentEnvFastMCPApplication(self._build_card(), self).add_routes_to_app(app)
        self.mcp = app
        self._mounted = True
        return app

    def serve(self, transport: str = "streamable-http") -> None:
        if self.mcp is None:
            self.create_app()
        elif not self._mounted:
            self.mount(self.mcp)
        self.mcp.run(transport=transport)


_SCALAR_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean", list: "array", dict: "object"}
_STRING_FORMATS = {datetime.datetime: "date-time", datetime.date: "date", datetime.time: "time", uuid.UUID: "uuid"}


def _enum_schema(values: list) -> dict:
    """JSON-Schema for an enum/Literal: the allowed values, plus a `type` if they're homogeneous."""
    schema: dict = {"enum": values}
    value_types = {_SCALAR_TYPES.get(type(v)) for v in values}
    if len(value_types) == 1 and None not in value_types:
        schema["type"] = value_types.pop()
    return schema


def _schema_for_annotation(annotation: Any) -> dict:
    """Map a Python annotation to a JSON-Schema fragment (type/enum/format/description); {} = unconstrained."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}
    origin = get_origin(annotation)
    if origin is Annotated:
        base, *metadata = get_args(annotation)
        schema = _schema_for_annotation(base)
        for meta in metadata:
            if isinstance(meta, FieldInfo) and meta.description:
                schema["description"] = meta.description
        return schema
    if origin is Union or origin is getattr(types, "UnionType", None):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        return _schema_for_annotation(non_none[0]) if len(non_none) == 1 else {}
    if origin is Literal:
        return _enum_schema(list(get_args(annotation)))
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return _enum_schema([m.value for m in annotation])
    fmt = _STRING_FORMATS.get(annotation)
    if fmt is not None:
        return {"type": "string", "format": fmt}
    if annotation is decimal.Decimal:
        return {"type": "string"}
    if origin in (tuple, set, frozenset) or annotation in (tuple, set, frozenset):
        return {"type": "array"}
    json_type = _SCALAR_TYPES.get(origin or annotation)
    return {"type": json_type} if json_type is not None else {}


def _schema_from_signature(fn: Callable) -> dict:
    """Derive a JSON-Schema object for a handler's call params from its signature (Annotated[Field] descriptions included)."""
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn, include_extras=True)
    except Exception:
        hints = {}
    properties: dict = {}
    required: list = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls") or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        prop = _schema_for_annotation(hints.get(name, param.annotation))
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default.value if isinstance(param.default, enum.Enum) else param.default
        properties[name] = prop
    schema: dict = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def _marked_members(handler: Any, marker_attr: str):
    """Yield (bound_member, marker) for every handler member carrying ``marker_attr``."""
    for name in dir(handler):
        if name.startswith("__"):
            continue
        attr = getattr(handler, name)
        marker = getattr(attr, marker_attr, None)
        if marker is not None:
            yield attr, marker


def _discover_methods(handler: Any) -> dict:
    found: dict[str, Callable] = {}
    for attr, op in _marked_members(handler, _OP_ATTR):
        if op in found:
            raise ValueError(f"Multiple methods marked @{op}")
        found[op] = attr
    methods: dict = {}
    for op in _OPERATIONS:
        fn = found.get(op.name)
        if fn is not None:
            methods[op.rpc_method] = (op, fn)
    return methods


def _discover_extensions(handler: Any) -> list:
    """Collect (descriptor, bound_method, route_path, http_method) for every @extension method."""
    seen_uris: set = set()
    seen_paths: set = set()
    ordered: list = []
    for attr, descriptor in _marked_members(handler, _EXT_ATTR):
        route_path, http_method = getattr(attr, _EXT_ROUTE_ATTR)
        if descriptor.uri in seen_uris:
            raise ValueError(f"Multiple handlers for extension {descriptor.uri}")
        if route_path in seen_paths:
            raise ValueError(f"Multiple handlers for extension path {route_path}")
        seen_uris.add(descriptor.uri)
        seen_paths.add(route_path)
        ordered.append((descriptor, attr, route_path, http_method))
    return ordered


def _merge_extensions(card: EnvironmentCard, ext_handlers: list) -> EnvironmentCard:
    """Advertise decorator-declared extensions under card.capabilities; card-declared URIs win."""
    if not ext_handlers:
        return card
    caps = card.capabilities or EnvironmentCapabilities()
    advertised = list(caps.extensions or [])
    seen = {e.uri for e in advertised}
    for descriptor, *_ in ext_handlers:
        if descriptor.uri not in seen:
            advertised.append(descriptor)
            seen.add(descriptor.uri)
    return card.model_copy(update={"capabilities": caps.model_copy(update={"extensions": advertised or None})})


_ENV_NAME_TOKEN = "{environment_name}"


def _discover_tools(handler: Any, environment_name: str) -> list:
    """Collect (EnvironmentTool, bound_method) for every @tool method, resolving the
    ``{environment_name}`` placeholder in each name against the environment card name.
    Any other unresolved ``{...}`` token raises."""
    seen: set = set()
    ordered: list = []
    for attr, marker in _marked_members(handler, _TOOL_ATTR):
        name = marker.name.replace(_ENV_NAME_TOKEN, environment_name)
        if "{" in name or "}" in name:
            raise ValueError(
                f"@tool name {marker.name!r} has an unresolved placeholder; "
                f"the only supported token is {_ENV_NAME_TOKEN!r}"
            )
        descriptor = marker.model_copy(update={"name": name})
        if descriptor.name in seen:
            raise ValueError(f"Multiple handlers for tool {descriptor.name}")
        seen.add(descriptor.name)
        ordered.append((descriptor, attr))
    return ordered


def _merge_tools(card: EnvironmentCard, tool_handlers: list) -> EnvironmentCard:
    """Advertise decorator-declared tools under card.capabilities; card-declared names win."""
    if not tool_handlers:
        return card
    caps = card.capabilities or EnvironmentCapabilities()
    advertised = list(caps.tools or [])
    seen = {t.name for t in advertised}
    for descriptor, _fn in tool_handlers:
        if descriptor.name not in seen:
            advertised.append(descriptor)
            seen.add(descriptor.name)
    return card.model_copy(update={"capabilities": caps.model_copy(update={"tools": advertised or None})})


def _merge_operations(card: EnvironmentCard, methods: dict) -> EnvironmentCard:
    """Advertise the discovered operations; always overwritten — registration is the truth.
    [] is distinct from absent, which marks a pre-advertisement card (full trio required)."""
    caps = card.capabilities or EnvironmentCapabilities()
    return card.model_copy(update={"capabilities": caps.model_copy(update={"operations": list(methods)})})


_DATA_OBJECTS_EXTENSION = EnvironmentExtension(
    uri=DATA_OBJECTS_EXTENSION_URI,
    description="data/get takes a write_namespace grant and may upload its export under it.",
)


def _takes_write_namespace(fn: Callable) -> bool:
    return "write_namespace" in inspect.signature(fn).parameters


def _merge_data_objects(card: EnvironmentCard, methods: dict) -> EnvironmentCard:
    """Advertise the data-objects extension when the ``data/get`` handler takes a ``write_namespace``."""
    entry = methods.get(METHOD_GET)
    if entry is None or not _takes_write_namespace(entry[1]):
        return card
    return _merge_extensions(card, [(_DATA_OBJECTS_EXTENSION,)])


def _merge_mcp_interface(card: EnvironmentCard, path: str) -> EnvironmentCard:
    """Declare the MCP endpoint at the path the app serves it on; a card-declared `MCP_TRANSPORT` interface wins.

    Only that transport counts, because it is the only one `client.mcp_path` reads.
    """
    if any(i.transport == MCP_TRANSPORT for i in card.additionalInterfaces):
        return card
    interfaces = [*card.additionalInterfaces, EnvironmentInterface(url=path, transport=MCP_TRANSPORT)]
    return card.model_copy(update={"additionalInterfaces": interfaces})


class _InvalidParams(Exception):
    pass


class _OperationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


async def _invoke_reset(fn: Callable, params: dict) -> dict:
    try:
        await _maybe_await(fn())
    except Exception as e:
        logger.exception("data/reset failed")
        raise _OperationError("reset_failed", str(e))
    return ResetDataResponse().model_dump()


async def _invoke_add(fn: Callable, params: dict) -> dict:
    try:
        req = AddDataRequest(**params)
    except Exception as e:
        raise _InvalidParams(str(e))
    try:
        await _maybe_await(fn(req.parts))
    except Exception as e:
        logger.exception("data/add failed")
        raise _OperationError("add_failed", str(e))
    return AddDataResponse().model_dump()


async def _invoke_get(fn: Callable, params: dict) -> dict:
    kwargs = {}
    if _takes_write_namespace(fn):
        if not isinstance(params, dict):
            raise _InvalidParams("data/get takes its params by name")
        raw = params.get("write_namespace")
        try:
            kwargs["write_namespace"] = None if raw is None else WriteNamespaceGrant.model_validate(raw)
        except ValidationError as e:
            # Name the fields only: the values carry the grant's signature.
            fields = ", ".join(".".join(map(str, error["loc"])) or "write_namespace" for error in e.errors())
            raise _InvalidParams(f"write_namespace is not a valid namespace grant ({fields})")
    try:
        parts = await _maybe_await(fn(**kwargs))
        return GetDataResponse(parts=parts).model_dump(exclude_none=True)
    except Exception as e:
        logger.exception("data/get failed")
        raise _OperationError("get_failed", str(e))


def _coerce_literal(value: Any, target: Any) -> Any:
    """Validate (and coerce) a request value against a ``Literal[...]`` annotation.

    ``_schema_for_annotation`` advertises a ``Literal`` as an enum, so the request path must enforce
    it: when every allowed value shares one homogeneous scalar type (e.g. ``Literal[1, 2]``) a raw
    GET query string is coerced to that type so ``"1"`` matches ``1``. The value must then be one of
    the allowed literals — anything else raises ``ValueError`` (turned into a 400 ``invalid_params``)
    so a handler with ``mode: Literal["live", "fixed"]`` can't receive ``"delete"`` and run anyway."""
    allowed = get_args(target)
    value_types = {type(a) for a in allowed}
    if isinstance(value, str) and len(value_types) == 1:
        (lit_type,) = value_types
        if lit_type is not str:
            try:
                value = _coerce_scalar(value, lit_type)
            except Exception:
                pass  # leave as-is; the membership check below rejects it
    # `is`-aware membership so bool literals aren't matched by 0/1 (since 1 == True in Python).
    if not any(v is value or (type(v) is type(value) and v == value) for v in allowed):
        raise ValueError(f"invalid value {value!r}; expected one of {list(allowed)}")
    return value


def _coerce_scalar(value: Any, target: Any) -> Any:
    """Coerce a request value to the handler's annotated type so the handler receives the runtime
    type its signature (and the advertised JSON-Schema) implies — not a raw ``str``.

    Covers every type ``_schema_for_annotation`` advertises as a concrete input: bool/int/float,
    ``datetime``/``date``/``time`` (``date-time``/``date``/``time`` formats), ``UUID`` (``uuid``),
    ``Decimal`` (advertised as ``string``), and ``Enum`` subclasses. Without this, a client that
    sends a value matching the advertised schema (e.g. a UUID or ISO datetime string) hands the
    handler a plain ``str``, so ``ident.hex`` / datetime comparisons / ``Decimal`` arithmetic /
    enum checks fail with a 500 even though the request matched the card. ``Optional[T]`` is
    unwrapped; ``Literal[...]`` values are validated against (and coerced to) their advertised
    allowed values; unknown/unannotated targets pass through unchanged. Invalid inputs (e.g. a
    non-ISO datetime, or a value outside a ``Literal``) raise — the caller turns that into a 400
    ``invalid_params``."""
    origin = get_origin(target)
    if origin is Union or origin is getattr(types, "UnionType", None):
        non_none = [a for a in get_args(target) if a is not type(None)]
        target = non_none[0] if len(non_none) == 1 else None
        origin = get_origin(target)
    if origin is Literal:
        return _coerce_literal(value, target)
    if target is None or not isinstance(target, type):
        return value
    # Enum: accept the member's underlying value (str/int/…) or pass an already-built member through.
    if issubclass(target, enum.Enum):
        return value if isinstance(value, target) else target(value)
    # Everything below maps from a string; native JSON ints/floats/bools/None pass straight through.
    if not isinstance(value, str):
        return value
    if target is bool:
        low = value.strip().lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"invalid boolean: {value!r}")
    if target is int:
        return int(value)
    if target is float:
        return float(value)
    if target is decimal.Decimal:
        return decimal.Decimal(value)
    if issubclass(target, datetime.datetime):   # before datetime.date — datetime subclasses date
        return datetime.datetime.fromisoformat(value)
    if issubclass(target, datetime.date):
        return datetime.date.fromisoformat(value)
    if issubclass(target, datetime.time):
        return datetime.time.fromisoformat(value)
    if issubclass(target, uuid.UUID):
        return uuid.UUID(value)
    return value


def _coerce_params(fn: Callable, params: dict) -> dict:
    """Coerce request params (GET query strings OR a POST JSON body) to the handler's annotated
    types — see ``_coerce_scalar``. Keys with no annotation (or no hint) pass through untouched."""
    try:
        hints = get_type_hints(fn)
    except Exception:
        return params
    return {k: _coerce_scalar(v, hints[k]) if k in hints else v for k, v in params.items()}


def _extension_handler(fn: Callable, http_method: str) -> Callable:
    """REST handler for an @extension route: bind request params to the handler and return JSON."""
    async def handler(request: Request) -> Response:
        try:
            if http_method == "GET":
                params = _coerce_params(fn, dict(request.query_params))
            else:
                raw = await request.body()
                params = json.loads(raw) if raw else {}
                if not isinstance(params, dict):
                    raise ValueError("request body must be a JSON object")
                params = _coerce_params(fn, params)
        except Exception as e:
            return _json_response(error_body("invalid_params", str(e)), 400)
        try:
            inspect.signature(fn).bind(**params)
        except TypeError as e:
            return _json_response(error_body("invalid_params", str(e)), 400)
        try:
            result = await _maybe_await(fn(**params))
        except Exception as e:
            logger.exception("extension invocation failed")
            return _json_response(error_body("extension_failed", str(e)), 500)
        return _json_response(result if result is not None else {})
    return handler


class _Op(NamedTuple):
    name: str
    rpc_method: str
    invoke: Callable


_OPERATIONS = [
    _Op(OP_RESET_DATA, METHOD_RESET, _invoke_reset),
    _Op(OP_ADD_DATA, METHOD_ADD, _invoke_add),
    _Op(OP_GET_DATA, METHOD_GET, _invoke_get),
]


def _jsonrpc_handler(methods: dict) -> Callable:
    async def handler(request: Request) -> Response:
        raw = await request.body()
        try:
            req = json.loads(raw) if raw else None
        except Exception:
            return _rpc_error(None, RPC_ERROR_CODE_PARSE_ERROR, "parse error")
        if not isinstance(req, dict) or req.get("jsonrpc") != "2.0" or "method" not in req:
            rid = req.get("id") if isinstance(req, dict) else None
            return _rpc_error(rid, RPC_ERROR_CODE_INVALID_REQUEST, "invalid request")
        rid = req.get("id")
        entry = methods.get(req["method"])
        if entry is None:
            return _rpc_error(rid, RPC_ERROR_CODE_METHOD_NOT_FOUND, f"method not found: {req['method']}")
        op, fn = entry
        try:
            result = await op.invoke(fn, req.get("params") or {})
        except _InvalidParams as e:
            return _rpc_error(rid, RPC_ERROR_CODE_INVALID_PARAMS, "invalid_request", {"detail": str(e)})
        except _OperationError as e:
            return _rpc_error(rid, RPC_ERROR_CODE_SERVER_ERROR, e.message, {"code": e.code})
        return _json_response({"jsonrpc": "2.0", "id": rid, "result": result})
    return handler


def _card_handler(card: EnvironmentCard) -> Callable:
    async def handler(request: Request) -> Response:
        return _json_response(card.model_dump())
    return handler


def _rpc_error(rid: Any, code: int, message: str, data: Any = None) -> Response:
    error: dict = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return _json_response({"jsonrpc": "2.0", "id": rid, "error": error})


def _json_response(body: Any, status: int = 200) -> Response:
    return Response(json.dumps(body, default=str), status_code=status, media_type="application/json")


async def _maybe_await(result: Any) -> Any:
    if inspect.isawaitable(result):
        return await result
    return result
