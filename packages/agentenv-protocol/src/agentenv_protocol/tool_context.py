"""Who is calling a tool, handed to the tool.

A caller's role and session reach a server as MCP request ``_meta`` entries
(the reverse-DNS keys below) when a gateway in front of the server stamps
them on each proxied call, or as the ``AgentEnv-Role`` and ``mcp-session-id``
headers when the caller reaches the server directly; a call that carries
neither has an empty caller. An environment built on
:class:`~agentenv_protocol.AgentEnvEnvironment` never reads either itself.
The protocol binds one :class:`ToolContext` per call, in the request's task,
before the handler runs, and hands it to the handler in two ways that always
agree:

- a handler that declares a parameter annotated :class:`ToolContext` (any
  name) receives it as an argument, and that parameter never appears in the
  tool's advertised schema;
- any code running on behalf of the call (a database method, a shared helper)
  can read the same object through :meth:`ToolContext.current`.

The binding follows the request's task: work handed to ``asyncio.to_thread``
or a new task inherits it, a raw ``threading.Thread`` does not and must be
given the context explicitly. Outside a call :meth:`ToolContext.current`
returns :data:`EMPTY`, never raises.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import types
import uuid
from typing import Any, Iterator, Literal, Mapping, Optional

from pydantic_core import core_schema

__all__ = [
    "DEFAULT_ROLE",
    "EMPTY",
    "ROLE_HEADER",
    "ROLE_META_KEY",
    "SESSION_META_KEY",
    "Caller",
    "ToolContext",
    "Transport",
    "bound",
    "caller_from_request",
    "context_from_http",
    "context_from_mcp",
    "normalize_role",
]

#: MCP request ``_meta`` key a gateway stamps the caller's role under.
ROLE_META_KEY = "agentenv.io/role"
#: MCP request ``_meta`` key a gateway stamps the caller's session under, so
#: per-session state keeps one owner per agent when the gateway shares one
#: connection to a server across agents.
SESSION_META_KEY = "agentenv.io/session"
#: Header carrying the role when no ``_meta`` does: non-MCP paths (a gateway
#: sets it on proxied REST requests) and direct MCP callers.
ROLE_HEADER = "AgentEnv-Role"
#: A gateway's unforwarded role. Same meaning as no role at all.
DEFAULT_ROLE = "default"

#: How the call reached the server. ``"none"`` is :data:`EMPTY`, the context
#: outside any call; the other two name the dispatch that bound the context.
Transport = Literal["mcp", "rest", "none"]


def normalize_role(value: Any) -> Optional[str]:
    """None for absent, blank and the gateway's unforwarded role; else the stripped role."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return None if value in ("", DEFAULT_ROLE) else value


@dataclasses.dataclass(frozen=True)
class Caller:
    """The identity a call arrived with.

    ``role`` is the role the agent was deployed with, normalised: None when
    nothing was forwarded or the gateway's default role came through.
    ``raw_role`` is what was received before normalisation, for logs.
    ``session`` tells callers apart when the gateway shares one connection.
    """

    role: Optional[str] = None
    session: Optional[str] = None
    raw_role: Optional[str] = None

    @property
    def is_forwarded(self) -> bool:
        return self.role is not None


@dataclasses.dataclass(frozen=True)
class ToolContext:
    """What the protocol knows about one tool call.

    Hashable, so per-call state can be keyed on the context itself: ``call_id``
    identifies the call, and ``arguments`` (a read-only mapping) and ``mcp``
    stay out of the hash.
    """

    caller: Caller = Caller()
    tool: str = ""
    arguments: Mapping[str, Any] = dataclasses.field(default_factory=lambda: types.MappingProxyType({}), hash=False)
    call_id: str = ""
    transport: Transport = "none"
    #: FastMCP's own request context when the call came over MCP (progress, logging, session); else None.
    mcp: Any = dataclasses.field(default=None, hash=False, compare=False)

    @property
    def role(self) -> Optional[str]:
        return self.caller.role

    @staticmethod
    def current() -> "ToolContext":
        """The context of the call this code runs on behalf of; :data:`EMPTY` outside any call."""
        return _current.get()

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        # A pydantic model may carry a ToolContext (an audit record, an event): validate by instance.
        return core_schema.is_instance_schema(cls)

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: Any, handler: Any) -> Any:
        # Every field here is schema-able, so a slotted function registered on a tool registry
        # without the injecting twin would silently advertise this type as a wire parameter;
        # the registry renders its argument model to JSON Schema at registration, so failing
        # here fails the mount.
        raise TypeError(
            "ToolContext is injected by the SDK and is not a wire parameter; "
            "register the tool with @tool or wrap it with agentenv_protocol.injecting(fn)"
        )

    @classmethod
    def for_test(cls, *, role: Optional[str] = None, session: Optional[str] = None, tool: str = "test",
                 arguments: Optional[Mapping[str, Any]] = None, transport: Transport = "mcp") -> "ToolContext":
        """A context for a test that calls a handler directly or binds one with :func:`bound`."""
        return cls(caller=Caller(role=normalize_role(role), session=session, raw_role=role), tool=tool,
                   arguments=_frozen(arguments), call_id=uuid.uuid4().hex, transport=transport)


#: The context handed back outside any call.
EMPTY = ToolContext()


def _frozen(arguments: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return types.MappingProxyType(dict(arguments or {}))

_current: contextvars.ContextVar[ToolContext] = contextvars.ContextVar("agentenv_protocol.tool_context", default=EMPTY)


@contextlib.contextmanager
def bound(context: ToolContext) -> Iterator[ToolContext]:
    """Bind ``context`` as the current one for the block, restoring the previous binding after."""
    token = _current.set(context)
    try:
        yield context
    finally:
        _current.reset(token)


def caller_from_request(meta_extra: Optional[Mapping[str, Any]], headers: Any) -> Caller:
    """Build the caller from an MCP request's custom ``_meta`` entries and its HTTP headers.

    A ``_meta`` role entry, whatever its value, marks a call a gateway proxied from its own
    view of the caller: the headers then describe the gateway's connection, not the caller,
    and are not consulted, so a blank or ``default`` role stays "not forwarded" and the
    session is the ``_meta`` session or none. Without that entry the caller reached the
    server itself, and the role header and ``mcp-session-id`` are trusted exactly as the
    caller's other requests are.
    """
    meta = meta_extra or {}
    forwarded = ROLE_META_KEY in meta
    if forwarded:
        raw_role = meta.get(ROLE_META_KEY)
    else:
        raw_role = headers.get(ROLE_HEADER) if headers is not None else None
    session = meta.get(SESSION_META_KEY)
    if not isinstance(session, str) or not session:
        session = None if forwarded or headers is None else headers.get("mcp-session-id")
    raw_role = raw_role if isinstance(raw_role, str) else None
    return Caller(role=normalize_role(raw_role), session=session or None, raw_role=raw_role)


def context_from_mcp(context: Any, tool: str, arguments: Mapping[str, Any]) -> ToolContext:
    """The context for one MCP tool call, from FastMCP's ``Context`` (or any object with a
    ``request_context``). Tolerates every missing link: no context, no live request, no meta,
    no HTTP request (stdio transports)."""
    request_context = None
    if context is not None:
        try:
            request_context = context.request_context
        except (ValueError, AttributeError):
            request_context = None
    meta = getattr(request_context, "meta", None)
    extra = getattr(meta, "model_extra", None) if meta is not None else None
    request = getattr(request_context, "request", None)
    headers = getattr(request, "headers", None) if request is not None else None
    return ToolContext(caller=caller_from_request(extra, headers), tool=tool, arguments=_frozen(arguments),
                       call_id=uuid.uuid4().hex, transport="mcp", mcp=context)


def context_from_http(request: Any, tool: str, arguments: Mapping[str, Any]) -> ToolContext:
    """The context for a REST or extension call: the caller from the request headers only."""
    headers = getattr(request, "headers", None) if request is not None else None
    return ToolContext(caller=caller_from_request(None, headers), tool=tool, arguments=_frozen(arguments),
                       call_id=uuid.uuid4().hex, transport="rest")
