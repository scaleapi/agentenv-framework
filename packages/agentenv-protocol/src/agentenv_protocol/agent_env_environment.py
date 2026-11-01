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
import re
import sys
import types
import uuid
from abc import ABC, abstractmethod
from typing import (Annotated, Any, Callable, ClassVar, Literal, NamedTuple, Optional, Union, get_args, get_origin,
                    get_type_hints)

from pydantic import TypeAdapter, ValidationError
from pydantic.fields import FieldInfo
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from .errors import AgentEnvError
from .tool_access import DISABLE as TOOL_DISABLE, ENABLE as TOOL_ENABLE, PATHS as TOOL_ACCESS_PATHS, ToolAccess, caller_role
from .tool_access import extensions as tool_access_extensions
from .tool_context import EMPTY, ToolContext, bound, context_from_http, context_from_mcp
from .transfers import WriteNamespaceGrant
from .types import DATA_OBJECTS_EXTENSION_URI, MCP_TRANSPORT, SKILL_EXTENSION_URI, METHOD_ADD, METHOD_GET, METHOD_RESET, RPC_PATH, WELL_KNOWN_PATH, AddDataRequest, AddDataResponse, EnvironmentCapabilities, EnvironmentCard, EnvironmentExtension, EnvironmentInterface, EnvironmentTool, GetDataResponse, ResetDataResponse, error_body

logger = logging.getLogger(__name__)

_OP_ATTR = "_aee_op"
_EXT_ATTR = "_aee_ext"
_EXT_ROUTE_ATTR = "_aee_ext_route"
_EXT_OP_ATTR = "_aee_ext_op"
_TOOL_ATTR = "_aee_tool"
_HIDDEN_BY_DEFAULT_ATTR = "_aee_hidden_by_default"
_TOOL_OPTIONS_ATTR = "_aee_tool_options"
_FORWARDED_ATTR = "_aee_forwarded"
_INSTRUCTIONS_ATTR = "_aee_instructions"
_RESOURCE_ATTR = "_aee_resource"
_CARD_CONFIG_ATTR = "_aee_card_config"
#: Set on the tool dispatch installed by ``_bind_tool_calls``; a guard downstream checks it at boot.
_TOOL_CONTEXT_BOUND_ATTR = "__agentenv_tool_context__"

#: MCP tools/list ``_meta`` key marking a tool a gateway hides from every role until a tool-access step names it.
HIDDEN_BY_DEFAULT_META_KEY = "agentenv.io/hiddenByDefault"

OP_RESET_DATA = "reset_data"
OP_ADD_DATA = "add_data"
OP_GET_DATA = "get_data"

RPC_ERROR_CODE_PARSE_ERROR = -32700
RPC_ERROR_CODE_INVALID_REQUEST = -32600
RPC_ERROR_CODE_METHOD_NOT_FOUND = -32601
RPC_ERROR_CODE_INVALID_PARAMS = -32602
RPC_ERROR_CODE_SERVER_ERROR = -32000
_SKILL_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


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


def extension(uri: str, *, op: str | None = None, description: str | None = None, params: dict | None = None,
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

    With ``op``, the handler is one method of an extension that has several: handlers sharing the URI
    merge into one card entry, each method under ``methods.<op>`` with its own verb and ``endpoint``,
    by default ``<RPC_PATH>/ext/<extension>/<op>`` where ``<extension>`` is the URN's name without
    ``urn:agentenv:`` and the version. Extension-level params come from the environment's
    ``extension_params(uri)``, merged beside ``methods``, so ``params`` is refused with ``op``.
    """
    if op is not None and params is not None:
        raise TypeError("an extension method's advertisement comes from extension_params(uri), not params=")
    # Normalize the HTTP verb once, here, so the advertised card method, the
    # route registration, and the handler's request-binding (which checks for
    # the exact string "GET") all agree. Without this, ``@extension(method="get")``
    # advertises/invokes GET while the handler parses the request body instead of
    # query params, so GET extensions with required query params return 400.
    http_method = method.upper()

    def deco(fn: Callable) -> Callable:
        if op is None:
            name = fn.__name__ or "invoke"
            route_path = path or f"{RPC_PATH}/ext/{name}"
            advertised = params if params is not None else {
                "endpoint": route_path,
                "methods": {name: {"method": http_method, "request": _schema_from_signature(fn)}},
            }
        else:
            route_path = path or f"{RPC_PATH}/ext/{_extension_slug(uri)}/{op}"
            advertised = {"methods": {op: {"method": http_method, "endpoint": route_path, "request": _schema_from_signature(fn)}}}
        setattr(fn, _EXT_ATTR, EnvironmentExtension(uri=uri, description=description, params=advertised, required=required))
        setattr(fn, _EXT_ROUTE_ATTR, (route_path, http_method))
        setattr(fn, _EXT_OP_ATTR, op)
        return fn

    return deco


def resource(uri: str, *, name: str | None = None, description: str | None = None, mime_type: str = "text/plain",
             listed: Callable[[Any], list[dict]] | None = None) -> Callable:
    """Mark a method that serves MCP resources, returning their text or bytes; an env may mark several, and each read
    calls the method. A fixed ``uri`` is one resource. A template such as ``orders://{order_id}`` is an MCP resource
    template, the method taking its parts, unless ``listed`` is given: it takes the env and returns the resources to
    list, each as MCP lists one (``uri``, ``name``, ``description``, ``mimeType``), each read through the method with
    its URI's parts. ``name`` defaults to the method's name and ``description`` to its docstring."""
    parts = set(re.findall(r"{(\w+)}", uri))

    def mark(fn: Callable) -> Callable:
        params = set(list(inspect.signature(fn).parameters)[1:])
        if params != parts:
            raise ValueError(f"@resource {uri!r} needs a method taking {sorted(parts) or 'nothing'} besides self, "
                             f"not {sorted(params)}")
        if listed is not None and not parts:
            raise ValueError(f"@resource {uri!r} lists resources only from a template URI")
        setattr(fn, _RESOURCE_ATTR, _ResourceMarker(uri, name, description, mime_type, listed))
        return fn

    return mark


def server_instructions(fn: Callable) -> Callable:
    """Mark the plain method that returns what this env's MCP server tells an agent when it connects, the
    ``instructions`` of its initialize result, which clients may put in the agent's context. It's read once, when the
    SDK builds the app (``create_app``, ``serve``, ``create_fastmcp_app``); an app passed to ``mount`` keeps its own."""
    if inspect.iscoroutinefunction(fn):
        raise TypeError(f"@server_instructions marks a plain method, not the coroutine {fn.__name__!r}")
    setattr(fn, _INSTRUCTIONS_ATTR, True)
    return fn


def agent_skill(fn: Callable) -> Callable:
    """Mark the method that writes this env's skill for agents, returning its description, its markdown body and,
    optionally, its other files by relative path; it is served under ``SKILL_EXTENSION_URI`` as ``get {name}``, which
    writes the SKILL.md under the name asked for and returns the files beside it."""

    async def get(self, name: str) -> dict:
        if len(name) > 64 or not _SKILL_NAME.fullmatch(name):
            raise AgentEnvError("invalid_argument",
                                f"a skill's name is lowercase letters, digits and single hyphens, up to 64, not {name!r}")
        description, body, *files = await fn(self) if inspect.iscoroutinefunction(fn) else fn(self)
        frontmatter = f"---\nname: {name}\ndescription: {json.dumps(description)}\n---\n"
        return {"name": name, "description": description, "skill_md": f"{frontmatter}\n{body}",
                **({"files": files[0]} if files and files[0] else {})}

    get.__name__ = fn.__name__
    return extension(SKILL_EXTENSION_URI, op="get", method="GET",
                     description="get returns a SKILL.md for agents under the name it's given, and the skill's other "
                                 "files when it has any.")(get)


def _extension_slug(uri: str) -> str:
    """``urn:agentenv:computer-display/v1`` -> ``computer-display``; another namespace keeps its name as a path."""
    name = uri.removeprefix("urn:").removeprefix("agentenv:")
    head, _, version = name.rpartition("/")
    if head and version[:1] == "v" and version[1:].isdigit():
        name = head
    return name.replace(":", "/")


def tool(name: str | None = None, *, description: str | None = None, hidden_by_default: bool = False,
         annotations: dict | None = None, structured_output: bool | None = None, meta: dict | None = None) -> Callable:
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

    ``hidden_by_default=True`` makes being hidden a fixed property of the tool, for plumbing a task
    drives but the agent under test must not see: it is listed with ``_meta``
    ``{HIDDEN_BY_DEFAULT_META_KEY: true}``, and a gateway hides it from every role until a tool-access
    step names it. Steps for every tool (``tools="*"``) never show it, though they still clear earlier
    named steps as they do for any tool.

    ``annotations`` (MCP ``ToolAnnotations`` fields, such as ``{"readOnlyHint": True}``) and ``meta`` are listed with
    the tool. ``structured_output`` follows mcp's FastMCP: None derives an output schema from the return annotation
    where it can, True requires one, False lists none; on a fastmcp app it is fastmcp's own inference, off with
    False. A tool returning ``Annotated[CallToolResult, Model]`` sends its result as is on either, listed with
    ``Model``'s schema.
    """
    if callable(name):
        raise TypeError("use @tool(...) with parentheses, not bare @tool")

    def deco(fn: Callable) -> Callable:
        setattr(fn, _TOOL_ATTR, EnvironmentTool(
            name=name or fn.__name__,
            description=description or inspect.getdoc(fn),
            inputSchema=_schema_from_signature(fn),
        ))
        if hidden_by_default:
            setattr(fn, _HIDDEN_BY_DEFAULT_ATTR, True)
        setattr(fn, _TOOL_OPTIONS_ATTR, _ToolOptions(annotations, structured_output, meta))
        return fn

    return deco


class ForwardedTool(NamedTuple):
    """A tool an environment serves on another server's behalf, as that server lists it: its descriptor, its own
    schema included, its MCP ``annotations`` and its ``output_schema``; ``call`` takes the call's arguments as keywords
    and returns the other server's ``CallToolResult``, which reaches the caller as it is."""

    tool: EnvironmentTool
    call: Callable[..., Any]
    annotations: dict | None = None
    output_schema: dict | None = None


class _ResourceMarker(NamedTuple):
    uri: str
    name: str | None
    description: str | None
    mime_type: str
    listed: Callable[[Any], list[dict]] | None


class _ServedResource(NamedTuple):
    """One resource, or a resource template when ``template``, as it is registered: ``read`` returns its content."""

    uri: str
    name: str
    description: str | None
    mime_type: str
    read: Callable
    template: bool


class _ToolOptions(NamedTuple):
    annotations: dict | None
    structured_output: bool | None
    meta: dict | None


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
        self._app: Any = None
        ext_handlers = _served_extensions(handler, _discover_extensions(handler))
        environment_card = _merge_extensions(environment_card, ext_handlers + _declared_extensions(handler, ext_handlers))
        tool_handlers = _served_tools(handler, _with_forwarded(_discover_tools(handler, environment_card.name), handler))
        if tool_handlers and not self.supports_tools:
            raise ValueError(f"@tool methods require a FastMCP-backed application; {type(self).__name__} cannot serve MCP tools")
        self._resources = _discover_resources(handler)
        if self._resources and not self.supports_tools:
            raise ValueError(f"@resource methods require a FastMCP-backed application; {type(self).__name__} cannot "
                             "serve MCP resources")
        environment_card = _merge_tools(environment_card, tool_handlers)
        self._tool_access = (
            ToolAccess(descriptor.name for descriptor, fn in tool_handlers if getattr(fn, _HIDDEN_BY_DEFAULT_ATTR, False))
            if self.supports_tools and getattr(handler, "tool_access", False) else None
        )
        if self._tool_access is not None:
            environment_card = _merge_extensions(environment_card, [(e,) for e in tool_access_extensions()])
        methods = _served_operations(handler, _discover_methods(handler))
        environment_card = _merge_operations(environment_card, methods)
        environment_card = _merge_data_objects(environment_card, methods)
        self.environment_card = environment_card
        self._dispatch = _jsonrpc_handler(methods)
        self._ext_routes = [(path, method, _extension_handler(fn, method)) for _descriptor, fn, path, method in ext_handlers]
        if self._tool_access is not None:
            self._ext_routes += [(TOOL_ACCESS_PATHS[action], "POST", _tool_access_handler(self._tool_access, action))
                                 for action in (TOOL_DISABLE, TOOL_ENABLE)]
        # Register with the advertised descriptor (card-declared wins in _merge_tools) so
        # tools/list cannot drift from the card; falls back to the discovered descriptor.
        caps_tools = (environment_card.capabilities.tools if environment_card.capabilities else None) or []
        advertised = {t.name: t for t in caps_tools}
        self._tools = [(advertised.get(descriptor.name, descriptor), fn) for descriptor, fn in tool_handlers]

    def add_routes_to_app(self, app: Any, *, rpc_url: str = RPC_PATH, card_url: str = WELL_KNOWN_PATH) -> None:
        self._check_tool_binding(app)
        self._app = app
        if isinstance(self.handler, AgentEnvEnvironment):
            self.handler._application = self
        self.environment_card = self._card_for(app)
        self._add_route(app, rpc_url, ["POST"], self._dispatch)
        self._add_route(app, card_url, ["GET"], self._serve_card)
        for path, method, handler in self._ext_routes:
            self._add_route(app, path, [method], handler)
        for descriptor, fn in self._tools:
            self._register_tool(app, descriptor, fn)
        for resource in self._resources:
            self._register_resource(app, resource)
        # Last, so it is the outermost layer: a dispatch guard the handler installed before mounting
        # then runs under the bound ToolContext instead of outside it.
        self._bind_tool_calls(app)

    @abstractmethod
    def _add_route(self, app: Any, path: str, methods: list, handler: Callable) -> None:
        ...

    def _register_tool(self, app: Any, descriptor: EnvironmentTool, fn: Callable) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not serve MCP tools")

    def _register_resource(self, app: Any, resource: _ServedResource) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not serve MCP resources")

    def _check_tool_binding(self, app: Any) -> None:
        """Everything that can refuse the mount, before the app is touched; nothing to refuse without a tool dispatch."""
        return None

    def _bind_tool_calls(self, app: Any) -> None:
        """Frameworks with a tool dispatch chokepoint bind a ToolContext around every call there; the others serve no tools."""
        return None

    def _card_for(self, app: Any) -> EnvironmentCard:
        return self.environment_card

    async def _serve_card(self, request: Request) -> Response:
        card = self.environment_card
        if isinstance(self.handler, AgentEnvEnvironment) and (children := await self.handler.children_environments()):
            card = card.model_copy(update={"children_environments": children})
        return _json_response(card.model_dump())

    async def update_tool(self, name: str, *, description: str) -> None:
        """Change a served tool's description after the mount: on the card and in tools/list alike."""
        if all(descriptor.name != name for descriptor, _fn in self._tools):
            raise ValueError(f"no served tool is named {name!r}")
        self._tools = [(d.model_copy(update={"description": description}) if d.name == name else d, fn)
                       for d, fn in self._tools]
        caps = self.environment_card.capabilities
        tools = [t.model_copy(update={"description": description}) if t.name == name else t for t in caps.tools or []]
        self.environment_card = self.environment_card.model_copy(
            update={"capabilities": caps.model_copy(update={"tools": tools})})
        await self._describe_registered_tool(name, description)

    async def _describe_registered_tool(self, name: str, description: str) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not serve MCP tools")


class AgentEnvFastMCPApplication(AgentEnvApplication):
    supports_tools: ClassVar[bool] = True
    _hook: Callable | None = None
    #: How this app's tool calls get their ToolContext, decided by ``_check_tool_binding``: ``"dispatch"`` wraps
    #: mcp FastMCP's tool manager, ``"middleware"`` joins a fastmcp app's middleware chain, None leaves them unbound.
    _binding: str | None = None
    _middleware: Any = None

    def _add_route(self, app: Any, path: str, methods: list, handler: Callable) -> None:
        app.custom_route(path, methods=methods)(handler)

    def _register_tool(self, app: Any, descriptor: EnvironmentTool, fn: Callable) -> None:
        if (forwarded := getattr(fn, _FORWARDED_ATTR, None)) is not None:
            return self._register_forwarded(app, descriptor, fn, forwarded)
        # passed only when set, so a FastMCP-shaped target without the parameters still mounts plain tools
        options = getattr(fn, _TOOL_OPTIONS_ATTR, None) or _ToolOptions(None, None, None)
        fastmcp = self._binding == "middleware"
        extra: dict = {}
        meta = {**(options.meta or {}), **({HIDDEN_BY_DEFAULT_META_KEY: True} if getattr(fn, _HIDDEN_BY_DEFAULT_ATTR, False) else {})}
        if meta:
            extra["meta"] = meta
        if options.annotations is not None:
            extra["annotations"] = options.annotations if fastmcp else _mcp_tool_annotations(options.annotations)
        if options.structured_output is not None and not fastmcp:
            extra["structured_output"] = options.structured_output
        elif options.structured_output is False:
            extra["output_schema"] = None
        registered = injecting(fn)
        if fastmcp and _returns_call_tool_result(fn):
            registered, schema = _fastmcp_returning(registered)
            extra["output_schema"] = None if options.structured_output is False else schema
        app.tool(name=descriptor.name, description=descriptor.description, **extra)(registered)

    def _register_forwarded(self, app: Any, descriptor: EnvironmentTool, fn: Callable, forwarded: ForwardedTool) -> None:
        """Put a forwarded tool into mcp's FastMCP tool manager as it is, with its own schema and its arguments passed
        through untouched, since FastMCP's ``tool`` builds a schema from a signature."""
        from mcp.server.fastmcp.tools.base import Tool
        from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata

        if self._binding != "dispatch":
            raise RuntimeError(f"the forwarded tool {descriptor.name!r} needs mcp's FastMCP, whose tool manager takes a "
                               "tool with its own schema")
        schema = forwarded.output_schema
        app._tool_manager._tools[descriptor.name] = Tool(
            fn=fn, name=descriptor.name, description=descriptor.description or "",
            parameters=descriptor.inputSchema or {"type": "object"},
            fn_metadata=FuncMetadata(arg_model=_passed_through(), output_schema=schema,
                                     output_model=_as_it_came() if schema is not None else None),
            is_async=True,
            annotations=_mcp_tool_annotations(forwarded.annotations) if forwarded.annotations is not None else None)

    def _register_resource(self, app: Any, resource: _ServedResource) -> None:
        """A resource as it is, its description absent when it has none, or a template through FastMCP's own
        decorator."""
        from mcp.server.fastmcp.resources import FunctionResource

        if self._binding != "dispatch":
            raise RuntimeError(f"the resource {resource.uri!r} needs mcp's FastMCP")
        if resource.template:
            app.resource(resource.uri, name=resource.name, description=resource.description,
                         mime_type=resource.mime_type)(resource.read)
            return
        app.add_resource(FunctionResource(uri=resource.uri, name=resource.name, description=resource.description,
                                          mime_type=resource.mime_type, fn=resource.read))

    def _check_tool_binding(self, app: Any) -> None:
        """Decide, before any route or tool is added, how every tools/call on ``app`` gets its ToolContext, and
        refuse what cannot be honoured. The standalone ``fastmcp`` package (a ``middleware`` list with
        ``add_middleware``) is joined through that public chain, checked first because its 2.x line also keeps a
        tool manager whose dispatch has another shape; mcp's own FastMCP (``mcp>=1.25,<2``) has no public hook,
        so its tool manager's ``call_tool`` is wrapped. An app with neither cannot bind: mounting it is refused
        when the handler declared that it reads the context (a ``ToolContext`` parameter or an ``on_tool_call``),
        since an empty caller would then be a lie nothing can detect, and goes ahead unbound, with a warning,
        otherwise. A dispatch this protocol already wraps is refused either way (one handler per app, or the
        second hook would answer the first handler's tools), an ``on_tool_call`` of the wrong shape fails here,
        not per call, and so does a fastmcp that cannot be imported: nothing has been added to the app yet."""
        self._hook = _tool_call_hook(self.handler)
        middleware = getattr(app, "middleware", None)
        if isinstance(middleware, list) and callable(getattr(app, "add_middleware", None)):
            if any(getattr(entry, _TOOL_CONTEXT_BOUND_ATTR, False) for entry in middleware):
                raise RuntimeError("this fastmcp app already carries an AgentEnv tool-context middleware; mount one handler per app")
            self._middleware = _tool_context_middleware(self._hook, self._tool_access)
            self._binding = "middleware"
            return
        manager = getattr(app, "_tool_manager", None)
        if callable(getattr(manager, "call_tool", None)) and callable(getattr(manager, "get_tool", None)):
            if getattr(manager.call_tool, _TOOL_CONTEXT_BOUND_ATTR, False):
                raise RuntimeError("this FastMCP app already carries an AgentEnv tool dispatch; mount one handler per app")
            self._binding = "dispatch"
            return
        readers = [descriptor.name for descriptor, fn in self._tools if _tool_context_slot(fn)]
        readers += ["on_tool_call"] if self._hook is not None else []
        readers += ["tool_access"] if self._tool_access is not None else []
        if readers:
            raise RuntimeError(
                "this app has no tool dispatch to bind a ToolContext around (mcp>=1.25,<2 keeps one at "
                f"FastMCP._tool_manager; fastmcp exposes a middleware list), and {', '.join(readers)} "
                "would read an empty caller"
            )
        self._binding = None
        logger.warning(
            "%s.%s (environment %r): no tool dispatch to bind a ToolContext around, so its tool calls run with an "
            "empty caller; build the app with create_fastmcp_app or on fastmcp to bind one",
            type(app).__module__, type(app).__qualname__, self.environment_card.name,
        )

    def _bind_tool_calls(self, app: Any) -> None:
        """Bind as ``_check_tool_binding`` decided. First in fastmcp's middleware list runs outermost, so a
        handler's own middleware and the tool both run under the binding, the same layering as the wrap."""
        if self._binding == "dispatch":
            self._bind_tool_dispatch(app)
        elif self._binding == "middleware":
            app.middleware.insert(0, self._middleware)

    def _bind_tool_dispatch(self, app: Any) -> None:
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
        access = self._tool_access

        async def call_tool(name: str, arguments: dict, *args: Any, **kwargs: Any) -> Any:
            if manager.get_tool(name) is None:
                return await inner(name, arguments, *args, **kwargs)
            context = kwargs.get("context", args[0] if args else None)
            with bound(context_from_mcp(context, name, arguments)) as tool_context:
                try:
                    if access is not None:
                        access.check_call(tool_context)
                    if hook is not None:
                        answer = await _maybe_await(hook(tool_context))
                        if answer is not None:
                            _check_call_tool_result(answer)
                            return answer
                    return await inner(name, arguments, *args, **kwargs)
                except Exception as e:
                    error = _agentenv_error(e)
                    if error is None:
                        raise
                    return _error_result(error)

        functools.update_wrapper(call_tool, inner)
        setattr(call_tool, _TOOL_CONTEXT_BOUND_ATTR, True)
        manager.call_tool = call_tool
        if access is not None:
            self._filter_tool_list(app, manager, access)

    @staticmethod
    def _filter_tool_list(app: Any, manager: Any, access: ToolAccess) -> None:
        """tools/list under the live request's caller, read through ``FastMCP.get_context``, as tools/call reads it."""
        inner = manager.list_tools

        def list_tools(*args: Any, **kwargs: Any) -> Any:
            role = caller_role(context_from_mcp(app.get_context(), "", {}))
            return [t for t in inner(*args, **kwargs) if not access.is_disabled(role, t.name)]

        functools.update_wrapper(list_tools, inner)
        manager.list_tools = list_tools

    async def _describe_registered_tool(self, name: str, description: str) -> None:
        """fastmcp replaces a tool through its public provider; mcp's FastMCP lists the tool its manager holds."""
        app = self._app
        if self._binding == "middleware":
            registered = await app.get_tool(name)
            app.local_provider.remove_tool(name)
            app.local_provider.add_tool(registered.model_copy(update={"description": description}))
            return
        manager = getattr(app, "_tool_manager", None)
        registered = manager.get_tool(name) if manager is not None else None
        if registered is not None:
            registered.description = description

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
    app = FastMCP(card.name, **{"instructions": _instructions(handler), **fastmcp_kwargs})
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
    #: Filter tools/list and refuse tools/call per caller role, and serve disable-tool/v1 and enable-tool/v1 at the
    #: gateway's paths and with its rules, for an env agents reach with no gateway in front.
    tool_access: ClassVar[bool] = False
    _mounted: bool = False
    #: The application mounted with this environment, which ``update_tool`` changes.
    _application: Any = None

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

    def mounted_card(self) -> EnvironmentCard:
        """The card the mounted application serves: what ``served_*`` kept, as ``update_tool`` left it."""
        return self._mounted_application().environment_card

    async def update_tool(self, name: str, *, description: str) -> None:
        """Change a served tool's description after the mount: on the card and in tools/list alike."""
        await self._mounted_application().update_tool(name, description=description)

    def _mounted_application(self) -> Any:
        if self._application is None:
            raise RuntimeError(f"{type(self).__name__} is not mounted")
        return self._application

    def serve(self, transport: str = "streamable-http") -> None:
        if self.mcp is None:
            self.create_app()
        elif not self._mounted:
            self.mount(self.mcp)
        self.mcp.run(transport=transport)

    def extension_params(self, uri: str) -> dict:
        """Extension-level params (features, limits) advertised beside the methods of an extension served with ``op``."""
        return {}

    def declared_extensions(self) -> list[EnvironmentExtension]:
        """Declaration-only extensions to advertise, like ``intake/v1``: a promise with params and no handler."""
        return []

    async def children_environments(self) -> Optional[list[EnvironmentCard]]:
        """The environments this one fronts, as each serves its card with its paths under this one's, listed on the card
        each time it's served; None for none."""
        return None

    def served_extensions(self, extensions: list[EnvironmentExtension]) -> list[EnvironmentExtension]:
        """The ``@extension`` handlers' extensions to serve, chosen at mount: one left out is not served, and of one served
        with ``op``, only the methods its returned descriptor still lists."""
        return extensions

    def served_operations(self, operations: list[str]) -> list[str]:
        """The data-plane methods to serve (``data/add`` and the others the handler marks), chosen at mount."""
        return operations

    def served_tools(self, tools: list[EnvironmentTool]) -> list[EnvironmentTool]:
        """The ``@tool`` methods to serve, chosen at mount; a returned descriptor's description replaces the marked one."""
        return tools

    def forwarded_tools(self) -> list[ForwardedTool]:
        """Tools served on another server's behalf, each with its own schema, beside the ``@tool`` methods: listed on
        the card, chosen by ``served_tools`` and under tool access like them. Needs mcp's FastMCP."""
        return []

    def on_tool_call(self, context: ToolContext) -> Any:
        """Runs before every tools/call on this environment's app, under the bound ``context``.

        Return None to let the call proceed, or an ``mcp.types.CallToolResult`` (``isError=True`` for
        a refusal) to answer the call without running the tool. On mcp's FastMCP it reaches the wire
        verbatim, before output validation; on a fastmcp app it passes through fastmcp's output validation,
        so a non-error answer there must fit the tool's declared output, while a refusal passes unchanged.
        Anything else raises TypeError. May be ``async``. Not invoked for extension (REST) routes.
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
    The slot is filled by name, so a positional-only one is refused here rather than on every call."""
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
    signature, annotations = _reduced_signature(fn, slot)
    full = inspect.signature(fn)

    def filled(args: tuple, kwargs: dict) -> inspect.BoundArguments:
        # Positional arguments mean what the twin advertises, then land where ``fn`` declares them.
        call = full.bind_partial()
        call.arguments.update(signature.bind(*args, **kwargs).arguments)
        call.arguments[slot] = ToolContext.current()
        return call

    # Two bodies so the registry's sync/async detection sees the same kind as ``fn``.
    if inspect.iscoroutinefunction(fn):
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            call = filled(args, kwargs)
            return await fn(*call.args, **call.kwargs)
    else:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call = filled(args, kwargs)
            return fn(*call.args, **call.kwargs)
    functools.update_wrapper(wrapper, fn)
    wrapper.__signature__, wrapper.__annotations__ = signature, annotations
    return wrapper


def injecting(fn: Callable) -> Callable:
    """``fn`` as an MCP tool registry should see it: unchanged when it declares no ``ToolContext``
    parameter, else a twin of the same name, doc and sync/async nature whose signature and annotations
    omit that parameter and which fills it with ``ToolContext.current()`` on every call; positional
    arguments follow the twin's signature, not ``fn``'s. ``@tool`` methods get this at mount; use it for a slotted function registered any other way
    (``app.tool()(fn)``, a separate ``ToolManager``), which otherwise fails at registration."""
    slot = _tool_context_slot(fn)
    return _inject_tool_context(fn, slot) if slot else fn


def _check_call_tool_result(value: Any) -> None:
    # The value goes to the wire as-is, so anything else would be emitted as a successful result.
    result_type = getattr(sys.modules.get("mcp.types"), "CallToolResult", None)
    if result_type is None or not isinstance(value, result_type):
        raise TypeError(f"on_tool_call must return None or mcp.types.CallToolResult, got {type(value).__name__}")


def _tool_context_middleware(hook: Callable | None, access: ToolAccess | None = None) -> Any:
    """A fastmcp ``Middleware`` that runs every tools/call under its ToolContext, the handler's ``on_tool_call``
    first, and with ``access`` filters tools/list and refuses tools/call per caller role. A ``Middleware`` subclass
    rather than a bare ``(context, call_next)`` callable because fastmcp runs the typed hooks exactly once per call,
    where a bare callable runs once per dispatch phase. fastmcp is not a dependency of this package: it is imported
    here, where an app built with it has just been handed over."""
    try:
        from fastmcp.server.middleware import Middleware
    except ImportError as e:
        raise RuntimeError("this app carries a fastmcp middleware list but fastmcp cannot be imported") from e

    class ToolContextMiddleware(Middleware):
        async def on_list_tools(self, context: Any, call_next: Callable) -> Any:
            tools = await call_next(context)
            if access is None:
                return tools
            role = caller_role(context_from_mcp(context.fastmcp_context, "", {}))
            return [t for t in tools if not access.is_disabled(role, t.name)]

        async def on_call_tool(self, context: Any, call_next: Callable) -> Any:
            message = context.message
            arguments = dict(getattr(message, "arguments", None) or {})
            with bound(context_from_mcp(context.fastmcp_context, message.name, arguments)) as tool_context:
                try:
                    if access is not None:
                        access.check_call(tool_context)
                    if hook is not None:
                        answer = await _maybe_await(hook(tool_context))
                        if answer is not None:
                            _check_call_tool_result(answer)
                            return _fastmcp_tool_result(answer)
                    return await call_next(context)
                except Exception as e:
                    error = _agentenv_error(e)
                    if error is None:
                        raise
                    return _fastmcp_tool_result(_error_result(error))

    middleware = ToolContextMiddleware()
    setattr(middleware, _TOOL_CONTEXT_BOUND_ATTR, True)
    return middleware


def _fastmcp_tool_result(answer: Any) -> Any:
    """A hook's ``mcp.types.CallToolResult`` as the ``ToolResult`` a fastmcp middleware chain returns, the error
    flag kept, under the field spellings of mcp 1 and 2."""
    from fastmcp.tools import ToolResult
    # mcp 2's spellings first: on mcp 1 they are plain misses, on mcp 2 the camel-case ones are deprecation shims.
    structured = getattr(answer, "structured_content", None)
    if structured is None:
        structured = getattr(answer, "structuredContent", None)
    is_error = getattr(answer, "is_error", None)
    if is_error is None:
        is_error = getattr(answer, "isError", False)
    return ToolResult(content=list(answer.content), structured_content=structured, is_error=bool(is_error))


def _mcp_tool_annotations(annotations: Any) -> Any:
    from mcp.types import ToolAnnotations
    return ToolAnnotations.model_validate(annotations)


def _returns_call_tool_result(fn: Callable) -> bool:
    returns = _resolved_hints(fn).get("return")
    if get_origin(returns) is Annotated:
        returns = get_args(returns)[0]
    result_type = getattr(sys.modules.get("mcp.types"), "CallToolResult", None)
    return result_type is not None and isinstance(returns, type) and issubclass(returns, result_type)


def _fastmcp_returning(fn: Callable) -> tuple:
    """A tool returning ``CallToolResult`` as fastmcp sends it as is: a twin returning fastmcp's ``ToolResult``, and the
    output schema of ``Annotated[CallToolResult, Model]``'s model (None for a bare ``CallToolResult``)."""
    from fastmcp.tools import ToolResult
    signature, annotations = _reduced_signature(fn, "")
    returns = annotations.get("return")
    schema = TypeAdapter(get_args(returns)[1]).json_schema() if get_origin(returns) is Annotated else None

    if inspect.iscoroutinefunction(fn):
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _fastmcp_tool_result(await fn(*args, **kwargs))
    else:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _fastmcp_tool_result(fn(*args, **kwargs))
    functools.update_wrapper(wrapper, fn)
    wrapper.__signature__ = signature.replace(return_annotation=ToolResult)
    wrapper.__annotations__ = {**annotations, "return": ToolResult}
    return wrapper, schema


def _agentenv_error(exc: BaseException) -> AgentEnvError | None:
    """The AgentEnvError a tool raised, found under the ToolError both FastMCPs wrap it in."""
    while exc is not None:
        if isinstance(exc, AgentEnvError):
            return exc
        exc = exc.__cause__
    return None


def _error_result(error: AgentEnvError) -> Any:
    from mcp.types import CallToolResult, TextContent
    return CallToolResult(
        isError=True,
        content=[TextContent(type="text", text=f"{error.code}: {error.message}")],
        structuredContent=error.fields(),
    )


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


def _discover_resources(handler: Any) -> list[_ServedResource]:
    """The resources and resource templates the handler's @resource methods serve; a URI served twice is refused."""
    served: list[_ServedResource] = []
    for method, marker in _marked_members(handler, _RESOURCE_ATTR):
        name, description = marker.name or method.__name__, marker.description or inspect.getdoc(method)
        if marker.listed is None:
            served.append(_ServedResource(marker.uri, name, description, marker.mime_type, method, "{" in marker.uri))
            continue
        for entry in marker.listed(handler):
            parts = _template_parts(marker.uri, entry["uri"])
            if parts is None:
                raise ValueError(f"the listed resource {entry['uri']!r} doesn't match {marker.uri!r}")
            served.append(_ServedResource(entry["uri"], entry["name"], entry.get("description"),
                                          entry.get("mimeType") or marker.mime_type, _with_parts(method, parts), False))
    uris = [resource.uri for resource in served]
    if duplicated := sorted({uri for uri in uris if uris.count(uri) > 1}):
        raise ValueError(f"resources served twice: {duplicated}")
    return served


def _template_parts(template: str, uri: str) -> dict[str, str] | None:
    """The parts of ``uri`` that ``template``'s ``{name}`` slots match, each within one path segment."""
    match = re.fullmatch(re.sub(r"\\\{(\w+)\\\}", r"(?P<\1>[^/]+)", re.escape(template)), uri)
    return match.groupdict() if match else None


def _with_parts(method: Callable, parts: dict[str, str]) -> Callable:
    def read() -> Any:
        return method(**parts)

    return read


def _instructions(handler: Any) -> str | None:
    """What the handler's @server_instructions method returns; None without one."""
    marked = [method for method, _marker in _marked_members(handler, _INSTRUCTIONS_ATTR)]
    if len(marked) > 1:
        raise ValueError(f"an environment has one @server_instructions method, not {sorted(m.__name__ for m in marked)}")
    return marked[0]() if marked else None


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
    """Collect (descriptor, bound_method, route_path, http_method) for every @extension method. The methods of an
    extension served with ``op`` share one descriptor: their ``methods`` merged beside ``extension_params(uri)``."""
    ops_by_uri: dict = {}
    seen_paths: set = set()
    ordered: list = []
    for attr, descriptor in _marked_members(handler, _EXT_ATTR):
        route_path, http_method = getattr(attr, _EXT_ROUTE_ATTR)
        op = getattr(attr, _EXT_OP_ATTR, None)
        ops = ops_by_uri.setdefault(descriptor.uri, set())
        if None in ops or (op is None and ops):
            raise ValueError(f"Multiple handlers for extension {descriptor.uri}")
        if op in ops:
            raise ValueError(f"Multiple handlers for extension {descriptor.uri} method {op}")
        if route_path in seen_paths:
            raise ValueError(f"Multiple handlers for extension path {route_path}")
        ops.add(op)
        seen_paths.add(route_path)
        ordered.append((descriptor, attr, route_path, http_method))
    merged = {
        uri: _merged_extension(handler, [d for d, *_ in ordered if d.uri == uri])
        for uri, ops in ops_by_uri.items() if None not in ops
    }
    return [(merged.get(d.uri, d), attr, path, method) for d, attr, path, method in ordered]


def _merged_extension(handler: Any, descriptors: list) -> EnvironmentExtension:
    uri = descriptors[0].uri
    methods = {name: entry for d in descriptors for name, entry in d.params["methods"].items()}
    hook = getattr(handler, "extension_params", None)
    params = {**(hook(uri) if hook is not None else {}), "methods": methods}
    return EnvironmentExtension(
        uri=uri,
        description=_one_value(uri, "description", [d.description for d in descriptors]),
        params=params,
        required=_one_value(uri, "required", [d.required for d in descriptors]),
    )


def _one_value(uri: str, field: str, values: list) -> Any:
    given = {v for v in values if v is not None}
    if len(given) > 1:
        raise ValueError(f"the methods of extension {uri} declare different {field} values: {sorted(map(str, given))}")
    return given.pop() if given else None


def _declared_extensions(handler: Any, ext_handlers: list) -> list:
    hook = getattr(handler, "declared_extensions", None)
    declared = list(hook()) if hook is not None else []
    served = {descriptor.uri for descriptor, *_ in ext_handlers}
    clash = sorted(d.uri for d in declared if d.uri in served)
    if clash:
        raise ValueError(f"extensions both declared and served by a handler: {clash}")
    return [(d,) for d in declared]


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


def _served_extensions(handler: Any, ext_handlers: list) -> list:
    """The extension handlers the handler's ``served_extensions`` keeps, with its descriptors."""
    hook = getattr(handler, "served_extensions", None)
    if hook is None:
        return ext_handlers
    offered = {descriptor.uri: descriptor for descriptor, *_ in ext_handlers}
    chosen = {descriptor.uri: descriptor for descriptor in hook(list(offered.values()))}
    unknown = sorted(set(chosen) - set(offered))
    if unknown:
        raise ValueError(f"served_extensions returned extensions with no @extension handler: {unknown}")
    ops = {}
    for descriptor, fn, *_ in ext_handlers:
        ops.setdefault(descriptor.uri, set()).add(getattr(fn, _EXT_OP_ATTR, None))
    for uri, served in chosen.items():
        if None not in ops[uri]:
            added = sorted(set((served.params or {}).get("methods") or {}) - ops[uri])
            if added:
                raise ValueError(f"served_extensions lists methods of {uri} with no @extension handler: {added}")
    kept = []
    for descriptor, fn, path, method in ext_handlers:
        served = chosen.get(descriptor.uri)
        op = getattr(fn, _EXT_OP_ATTR, None)
        if served is None or (op is not None and op not in (served.params or {}).get("methods", {})):
            continue
        kept.append((served, fn, path, method))
    return kept


def _served_operations(handler: Any, methods: dict) -> dict:
    """The data-plane methods the handler's ``served_operations`` keeps."""
    hook = getattr(handler, "served_operations", None)
    if hook is None:
        return methods
    chosen = set(hook(list(methods)))
    unknown = sorted(chosen - set(methods))
    if unknown:
        raise ValueError(f"served_operations returned operations with no handler: {unknown}")
    return {name: entry for name, entry in methods.items() if name in chosen}


def _with_forwarded(tool_handlers: list, handler: Any) -> list:
    """The ``@tool`` handlers and then the handler's forwarded tools, each as (descriptor, call), the call marked so it
    registers with its own schema; a forwarded tool named like another tool is refused."""
    hook = getattr(handler, "forwarded_tools", None)
    forwarded = [(entry.tool, _forwarding(entry)) for entry in (hook() if hook is not None else [])]
    names = [descriptor.name for descriptor, _fn in tool_handlers + forwarded]
    if len(set(names)) != len(names):
        raise ValueError(f"forwarded tools named like another tool: {sorted({n for n in names if names.count(n) > 1})}")
    return tool_handlers + forwarded


def _forwarding(forwarded: ForwardedTool) -> Callable:
    async def call(**arguments: Any) -> Any:
        return await forwarded.call(**arguments)

    setattr(call, _FORWARDED_ATTR, forwarded)
    return call


@functools.cache
def _as_it_came() -> type:
    """mcp's FastMCP output model for a forwarded tool with an output schema: the other server's result, unchecked."""
    from pydantic import RootModel

    return RootModel[Any]


@functools.cache
def _passed_through() -> type:
    """mcp's FastMCP argument model for a forwarded tool: any arguments, each passed to the call as it came."""
    from mcp.server.fastmcp.utilities.func_metadata import ArgModelBase
    from pydantic import ConfigDict

    class PassedThrough(ArgModelBase):
        model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

        def model_dump_one_level(self) -> dict[str, Any]:
            return dict(self.__pydantic_extra__ or {})

    return PassedThrough


def _served_tools(handler: Any, tool_handlers: list) -> list:
    """The tools the handler's ``served_tools`` keeps, in its order and with its descriptors; all of them without one."""
    hook = getattr(handler, "served_tools", None)
    if hook is None:
        return tool_handlers
    chosen = list(hook([descriptor for descriptor, _fn in tool_handlers]))
    by_name = {descriptor.name: fn for descriptor, fn in tool_handlers}
    names = [descriptor.name for descriptor in chosen]
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"served_tools returned tools with no @tool method: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError(f"served_tools returned a tool twice: {sorted({n for n in names if names.count(n) > 1})}")
    return [(descriptor, by_name[descriptor.name]) for descriptor in chosen]


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
    takes_paths = "paths" in inspect.signature(fn).parameters
    if isinstance(params, dict) and "paths" in params:
        if not takes_paths:
            raise _InvalidParams("data/get on this environment takes no paths: it returns all of its data")
        if not isinstance(params["paths"], list) or not all(isinstance(path, str) for path in params["paths"]):
            raise _InvalidParams("paths must be a list of strings")
    if takes_paths:
        kwargs["paths"] = params.get("paths") if isinstance(params, dict) else None
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
    ``ToolContext`` parameter is filled from the request headers and never bound
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
            return _error_response(AgentEnvError("invalid_params", str(e)))
        try:
            signature.bind(**params)
        except TypeError as e:
            return _error_response(AgentEnvError("invalid_params", str(e)))
        with bound(context_from_http(request, op, params)) as tool_context:
            call_kwargs = {**params, slot: tool_context} if slot else params
            try:
                result = await _maybe_await(fn(**call_kwargs))
            except AgentEnvError as e:
                return _error_response(e)
            except Exception as e:
                logger.exception("extension invocation failed")
                return _error_response(AgentEnvError("extension_failed", str(e), applied="unknown", status=500))
        return _json_response(result if result is not None else {})
    return handler


def _error_response(error: AgentEnvError) -> Response:
    return _json_response(error_body(**error.fields()), error.status)


def _tool_access_handler(access: ToolAccess, action: str) -> Callable:
    async def handler(request: Request) -> Response:
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            return _error_response(AgentEnvError("invalid_params", "request body must be JSON"))
        try:
            return _json_response(access.change(action, body))
        except AgentEnvError as e:
            return _error_response(e)
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
