"""Server SDK: decorate a handler's methods with @reset_data/@add_data/@get_data (JSON-RPC data plane at /agentenv; each optional, advertised as capabilities.operations), @extension (card-advertised REST routes), and @tool (MCP tools; FastMCP-backed apps only; a parameter annotated ToolContext receives the caller and is never advertised). Mount onto a FastMCP, Starlette, or FastAPI app via a per-framework AgentEnvApplication subclass — or subclass AgentEnvEnvironment, configure its card with @environment_card(...), and serve() it."""
from __future__ import annotations

import datetime
import decimal
import enum
import functools
import inspect
import json
import logging
import os
import sys
import types
import uuid
from abc import ABC, abstractmethod
from typing import Annotated, Any, Callable, ClassVar, Literal, NamedTuple, Union, get_args, get_origin, get_type_hints

from pydantic import ValidationError
from pydantic.fields import FieldInfo
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from .tool_context import EMPTY, ToolContext, bound, context_from_http, context_from_mcp
from .transfers import WriteNamespaceGrant
from .types import DATA_OBJECTS_EXTENSION_URI, MCP_TRANSPORT, METHOD_ADD, METHOD_GET, METHOD_RESET, RPC_PATH, WELL_KNOWN_PATH, AddDataRequest, AddDataResponse, EnvironmentCapabilities, EnvironmentCard, EnvironmentExtension, EnvironmentInterface, EnvironmentTool, GetDataResponse, ResetDataResponse, error_body

logger = logging.getLogger(__name__)

_OP_ATTR = "_aee_op"
_EXT_ATTR = "_aee_ext"
_EXT_ROUTE_ATTR = "_aee_ext_route"
_TOOL_ATTR = "_aee_tool"
_CARD_CONFIG_ATTR = "_aee_card_config"
#: Set on the tool dispatch installed by ``_bind_tool_calls``; a guard downstream checks it at boot.
_TOOL_CONTEXT_BOUND_ATTR = "__agentenv_tool_context__"

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

    A parameter annotated ``ToolContext`` (or ``Optional[ToolContext]``; any name) is filled by the
    SDK with the caller of the current call and is omitted from the advertised ``inputSchema`` and
    from tools/list; a FastMCP ``Context`` parameter is likewise omitted from the card while FastMCP
    keeps injecting it.
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
        self._check_tool_binding(app)
        self.environment_card = self._card_for(app)
        self._add_route(app, rpc_url, ["POST"], self._dispatch)
        self._add_route(app, card_url, ["GET"], _card_handler(self.environment_card))
        for path, method, handler in self._ext_routes:
            self._add_route(app, path, [method], handler)
        for descriptor, fn in self._tools:
            self._register_tool(app, descriptor, fn)
        # Last, so it is the outermost layer: a dispatch guard the handler installed before mounting
        # then runs under the bound ToolContext instead of outside it.
        self._bind_tool_calls(app)

    @abstractmethod
    def _add_route(self, app: Any, path: str, methods: list, handler: Callable) -> None:
        ...

    def _register_tool(self, app: Any, descriptor: EnvironmentTool, fn: Callable) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not serve MCP tools")

    def _check_tool_binding(self, app: Any) -> None:
        """Everything that can refuse the mount, before the app is touched; nothing to refuse without a tool dispatch."""
        return None

    def _bind_tool_calls(self, app: Any) -> None:
        """Frameworks with a tool dispatch chokepoint bind a ToolContext around every call there; the others serve no tools."""
        return None

    def _card_for(self, app: Any) -> EnvironmentCard:
        return self.environment_card


class AgentEnvFastMCPApplication(AgentEnvApplication):
    supports_tools: ClassVar[bool] = True
    _hook: Callable | None = None

    def _add_route(self, app: Any, path: str, methods: list, handler: Callable) -> None:
        app.custom_route(path, methods=methods)(handler)

    def _register_tool(self, app: Any, descriptor: EnvironmentTool, fn: Callable) -> None:
        app.tool(name=descriptor.name, description=descriptor.description)(injecting(fn))

    def _check_tool_binding(self, app: Any) -> None:
        """Refuse before any route or tool is added: an app without the dispatch the binding wraps (its tool
        calls would run unbound, which nothing reading ``ToolContext.current()`` can tell from an empty
        caller), a dispatch this protocol already wraps (one handler per app, or the second hook would answer
        the first handler's tools) and an ``on_tool_call`` of the wrong shape. The hook is kept for
        ``_bind_tool_calls``."""
        manager = getattr(app, "_tool_manager", None)
        if not (callable(getattr(manager, "call_tool", None)) and callable(getattr(manager, "get_tool", None))):
            raise RuntimeError(
                "this app has no FastMCP tool dispatch to bind a ToolContext around "
                "(mcp>=1.25,<2 keeps it at FastMCP._tool_manager); "
                "an app that serves no tools mounts through AgentEnvStarletteApplication"
            )
        if getattr(manager.call_tool, _TOOL_CONTEXT_BOUND_ATTR, False):
            raise RuntimeError("this FastMCP app already carries an AgentEnv tool dispatch; mount one handler per app")
        self._hook = _tool_call_hook(self.handler)

    def _bind_tool_calls(self, app: Any) -> None:
        """Wrap the tool manager's ``call_tool``, which every tools/call resolves at call time, so each call
        of a registered tool runs under its ToolContext and the handler's ``on_tool_call`` may answer
        it first. An unknown tool name goes straight to the dispatch, whose error answers it. FastMCP
        has no public hook for this (its low-level handler is ``FastMCP.call_tool``, bound at
        construction), so the manager is the one private seam, required by ``_check_tool_binding``.

        The installed callable carries ``__agentenv_tool_context__ = True`` and ``__wrapped__`` (the
        dispatch it wraps): the handles a downstream guard uses to assert the ordering contract.
        """
        manager = app._tool_manager
        inner = manager.call_tool
        hook = self._hook

        async def call_tool(name: str, arguments: dict, *args: Any, **kwargs: Any) -> Any:
            if manager.get_tool(name) is None:
                return await inner(name, arguments, *args, **kwargs)
            context = kwargs.get("context", args[0] if args else None)
            with bound(context_from_mcp(context, name, arguments)) as tool_context:
                if hook is not None:
                    answer = await _maybe_await(hook(tool_context))
                    if answer is not None:
                        _check_call_tool_result(answer)
                        return answer
                return await inner(name, arguments, *args, **kwargs)

        functools.update_wrapper(call_tool, inner)
        setattr(call_tool, _TOOL_CONTEXT_BOUND_ATTR, True)
        manager.call_tool = call_tool

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

    def on_tool_call(self, context: ToolContext) -> Any:
        """Runs before every tools/call on this environment's app, under the bound ``context``.

        Return None to let the call proceed, or an ``mcp.types.CallToolResult`` (``isError=True`` for
        a refusal) to answer the call without running the tool; it reaches the wire verbatim, before
        output validation. Anything else raises TypeError. May be ``async``. Not invoked for
        extension (REST) routes.
        """
        return None


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


def _unwrap_annotation(annotation: Any) -> list:
    """The concrete types an annotation names: Annotated metadata and None peeled off Optional/Union."""
    origin = get_origin(annotation)
    if origin is Annotated:
        return _unwrap_annotation(get_args(annotation)[0])
    if origin is Union or origin is getattr(types, "UnionType", None):
        return [t for arg in get_args(annotation) if arg is not type(None) for t in _unwrap_annotation(arg)]
    return [annotation]


def _split_top_level(text: str, sep: str) -> list:
    parts: list = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        elif ch == sep and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    parts.append(text[start:])
    return [p.strip() for p in parts]


def _string_annotation_names(text: str) -> list:
    """The type names a string annotation spells out, unwrapped like ``_unwrap_annotation``: the fallback
    under ``from __future__ import annotations`` when the function's hints cannot be resolved."""
    text = text.strip().strip("'\"")
    parts = _split_top_level(text, "|")
    if len(parts) > 1:
        return [n for part in parts for n in _string_annotation_names(part)]
    head, bracket, rest = text.partition("[")
    if bracket and text.endswith("]"):
        args = _split_top_level(rest[:-1], ",")
        wrapper = head.rsplit(".", 1)[-1]
        if wrapper == "Annotated":
            return _string_annotation_names(args[0])
        if wrapper in ("Optional", "Union"):
            return [n for arg in args for n in _string_annotation_names(arg)]
        return [text]
    return [] if text == "None" else [text.rsplit(".", 1)[-1]]


def _is_tool_context_annotation(annotation: Any) -> bool:
    if isinstance(annotation, str):
        return ToolContext.__name__ in _string_annotation_names(annotation)
    return any(t is ToolContext for t in _unwrap_annotation(annotation))


def _fastmcp_context_type() -> type | None:
    """FastMCP's Context class when mcp is loaded, else None. mcp is an optional dependency, so it is
    looked up rather than imported; a Context-annotated handler has necessarily imported it already."""
    return getattr(sys.modules.get("mcp.server.fastmcp.server"), "Context", None)


def _is_fastmcp_context_annotation(annotation: Any) -> bool:
    context_type = _fastmcp_context_type()
    if context_type is None:
        return False
    if isinstance(annotation, str):
        return context_type.__name__ in _string_annotation_names(annotation)
    return any(isinstance(t, type) and issubclass(t, context_type) for t in _unwrap_annotation(annotation))


def _resolved_hints(fn: Callable) -> dict:
    try:
        return get_type_hints(fn, include_extras=True)
    except Exception:
        return {}


def _callable_name(fn: Callable) -> str:
    return getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or repr(fn)


def _tool_context_slot(fn: Callable) -> str | None:
    """Name of the parameter annotated ``ToolContext`` / ``Optional[ToolContext]`` (any name), or None.
    The slot is filled by keyword, so a positional-only one is refused here rather than on every call."""
    hints = _resolved_hints(fn)
    slots = []
    for name, param in inspect.signature(fn).parameters.items():
        if name in ("self", "cls") or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if not _is_tool_context_annotation(hints.get(name, param.annotation)):
            continue
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(f"{_callable_name(fn)}: ToolContext parameter {name!r} must not be positional-only")
        slots.append(name)
    if len(slots) > 1:
        raise TypeError(f"{_callable_name(fn)} declares more than one ToolContext parameter: {slots}")
    return slots[0] if slots else None


def _tool_call_hook(handler: Any) -> Callable | None:
    """The handler's ``on_tool_call`` when it has one of its own (the base class's no-op does not count),
    checked at mount to take the ToolContext as its one argument so a mismatch fails here, not per call."""
    hook = getattr(handler, "on_tool_call", None)
    if hook is None or getattr(type(handler), "on_tool_call", None) is AgentEnvEnvironment.on_tool_call:
        return None
    try:
        inspect.signature(hook).bind(EMPTY)
    except TypeError as e:
        raise TypeError(f"{_callable_name(hook)} must take the ToolContext as its only argument") from e
    return hook


def _reduced_signature(fn: Callable, slot: str) -> tuple:
    """``fn``'s signature and annotations without ``slot``, every annotation resolved: the registry
    reads ``__signature__`` verbatim and cannot evaluate a string left in it."""
    sig = inspect.signature(fn)
    hints = _resolved_hints(fn)
    params = [p.replace(annotation=hints.get(p.name, p.annotation)) for p in sig.parameters.values() if p.name != slot]
    returns = hints.get("return", sig.return_annotation)
    unresolved = [p.name for p in params if isinstance(p.annotation, str)] + (["return"] if isinstance(returns, str) else [])
    if unresolved:
        raise TypeError(
            f"{_callable_name(fn)}: cannot resolve the annotations of {unresolved}; "
            "a function that declares a ToolContext parameter must have resolvable annotations"
        )
    annotations = {p.name: p.annotation for p in params if p.annotation is not inspect.Parameter.empty}
    if returns is not inspect.Signature.empty:
        annotations["return"] = returns
    return sig.replace(parameters=params, return_annotation=returns), annotations


def _inject_tool_context(fn: Callable, slot: str) -> Callable:
    # Two bodies so the registry's sync/async detection sees the same kind as ``fn``.
    if inspect.iscoroutinefunction(fn):
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs[slot] = ToolContext.current()
            return await fn(*args, **kwargs)
    else:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs[slot] = ToolContext.current()
            return fn(*args, **kwargs)
    functools.update_wrapper(wrapper, fn)
    wrapper.__signature__, wrapper.__annotations__ = _reduced_signature(fn, slot)
    return wrapper


def injecting(fn: Callable) -> Callable:
    """``fn`` as an MCP tool registry should see it: unchanged when it declares no ``ToolContext``
    parameter, else a twin of the same name, doc and sync/async nature whose signature and annotations
    omit that parameter and which fills it with ``ToolContext.current()`` on every call. ``@tool``
    methods get this at mount; use it for a slotted function registered any other way
    (``app.tool()(fn)``, a separate ``ToolManager``), which otherwise fails at registration."""
    slot = _tool_context_slot(fn)
    return _inject_tool_context(fn, slot) if slot else fn


def _check_call_tool_result(value: Any) -> None:
    # The value goes to the wire as-is, so anything else would be emitted as a successful result.
    result_type = getattr(sys.modules.get("mcp.types"), "CallToolResult", None)
    if result_type is None or not isinstance(value, result_type):
        raise TypeError(f"on_tool_call must return None or mcp.types.CallToolResult, got {type(value).__name__}")


def _schema_from_signature(fn: Callable) -> dict:
    """Derive a JSON-Schema object for a handler's call params from its signature (Annotated[Field] descriptions
    included). Injected parameters (``ToolContext``, FastMCP ``Context``) are not wire parameters and are left out."""
    sig = inspect.signature(fn)
    hints = _resolved_hints(fn)
    properties: dict = {}
    required: list = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls") or param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(name, param.annotation)
        if _is_tool_context_annotation(annotation) or _is_fastmcp_context_annotation(annotation):
            continue
        prop = _schema_for_annotation(annotation)
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
    """REST handler for an @extension route: bind request params to the handler and return JSON. A
    ``ToolContext`` parameter is filled from the request headers (transport ``"rest"``) and never bound
    from the params."""
    slot = _tool_context_slot(fn)
    signature = inspect.signature(fn)
    if slot:
        signature = signature.replace(parameters=[p for p in signature.parameters.values() if p.name != slot])
    op = fn.__name__ or "invoke"

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
            signature.bind(**params)
        except TypeError as e:
            return _json_response(error_body("invalid_params", str(e)), 400)
        with bound(context_from_http(request, op, params)) as tool_context:
            call_kwargs = {**params, slot: tool_context} if slot else params
            try:
                result = await _maybe_await(fn(**call_kwargs))
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
