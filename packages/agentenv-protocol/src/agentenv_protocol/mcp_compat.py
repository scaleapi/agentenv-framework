"""mcp 1 and mcp 2 behind one surface: the high-level server an app is built on, a client session on a URL, a
low-level server's tool handlers, a crash's text in a tool's error, and an mcp type's field under either spelling.
mcp is an optional dependency of this package, so each function imports it when called."""

from __future__ import annotations

import contextlib
import functools
import importlib
import importlib.metadata
import logging
import re
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

_logger = logging.getLogger(__name__)

#: The serving options mcp 2 takes at ``streamable_http_app()`` and ``run()``, which mcp 1's FastMCP took as settings.
_SERVING = ("host", "port", "streamable_http_path", "json_response", "stateless_http", "transport_security")


@functools.cache
def mcp_2() -> bool:
    return int(importlib.metadata.version("mcp").split(".")[0]) >= 2


def server(name: str, **kwargs: Any) -> Any:
    """mcp's high-level server named ``name``, ``kwargs`` taking mcp 1's FastMCP settings on either version. mcp 2
    takes the serving ones (path, host, port, transport security) at ``streamable_http_app()`` and ``run()``, so its
    server keeps them for those."""
    if not mcp_2():
        from mcp.server.fastmcp import FastMCP
        return FastMCP(name, **kwargs)
    serving = {key: kwargs.pop(key) for key in _SERVING if key in kwargs}
    return _serving_server_class()(name, {"streamable_http_path": "/mcp", **serving}, **kwargs)


def deployed_server(name: str, *, host: str, port: int, **kwargs: Any) -> Any:
    """``server`` serving streamable HTTP at ``host`` and ``port`` without DNS-rebinding protection."""
    if not mcp_2():
        app = server(name, **kwargs)
        app.settings.host, app.settings.port = host, port
        app.settings.transport_security.enable_dns_rebinding_protection = False
        return app
    from mcp.server.transport_security import TransportSecuritySettings
    return server(name, host=host, port=port,
                  transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False), **kwargs)


def streamable_http_path(app: Any) -> str | None:
    """Where ``app`` serves streamable HTTP, when it says: mcp 1's FastMCP in its settings, ``server``'s on mcp 2 in
    what it serves with."""
    serving = getattr(app, "serving", None)
    if isinstance(serving, dict):
        return serving.get("streamable_http_path")
    return getattr(getattr(app, "settings", None), "streamable_http_path", None)


def context_class() -> type | None:
    """The high-level server's tool Context class, when mcp is loaded; None otherwise. Looked up, not imported: a
    handler annotated with it has necessarily imported it already."""
    for module, name in (("mcp.server.fastmcp.server", "Context"), ("mcp.server.mcpserver.context", "Context")):
        found = getattr(sys.modules.get(module), name, None)
        if found is not None:
            return found
    return None


def low_level_server(name: str, list_tools: Callable[[], Awaitable[list[Any]]],
                     call_tool: Callable[[str, dict[str, Any]], Awaitable[list[Any]]]) -> Any:
    """mcp's low-level ``Server`` serving ``list_tools`` (the tools) and ``call_tool`` (a call's content). A call that
    raises answers with its message as an error result, as mcp 1's decorators answer it."""
    from mcp import types
    from mcp.server.lowlevel import Server
    if not mcp_2():
        server = Server(name)
        server.list_tools()(list_tools)
        server.call_tool()(call_tool)
        return server

    async def on_list_tools(ctx: Any, params: Any) -> Any:
        return types.ListToolsResult(tools=await list_tools())

    async def on_call_tool(ctx: Any, params: Any) -> Any:
        try:
            content = await call_tool(params.name, params.arguments or {})
        except Exception as e:
            return types.CallToolResult(content=[types.TextContent(type="text", text=str(e))], isError=True)
        return types.CallToolResult(content=content)

    return Server(name, on_list_tools=on_list_tools, on_call_tool=on_call_tool)


@contextlib.asynccontextmanager
async def client_session(url: str, *, headers: dict[str, str] | None = None, timeout: float | None = None,
                         read_timeout: float | None = None) -> AsyncIterator[Any]:
    """An initialized ``ClientSession`` on ``url`` over streamable HTTP, on the client mcp builds for itself: its
    timeouts (30 s, 300 s to read a stream) unless ``timeout`` is given. mcp 2's client is ``httpx2`` where mcp 1's is
    ``httpx``, and yields two streams where mcp 1's yields three."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared._httpx_utils import create_mcp_http_client
    http = importlib.import_module("httpx2" if mcp_2() else "httpx")
    limits = None if timeout is None else http.Timeout(timeout, read=read_timeout or timeout)
    async with create_mcp_http_client(headers=headers, timeout=limits) as client:
        async with streamable_http_client(url, http_client=client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield session


def with_crash_text(error: Exception) -> Exception:
    """The error a tool's dispatch raised, with a crash's own text as mcp 1 gives it ("Error executing tool x: ...").
    mcp 2 keeps a crash's text on the server, logging it there; this logs it the same way and hands it on."""
    if not mcp_2():
        return error
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
    if not isinstance(error, UnexpectedToolError) or error.__cause__ is None:
        return error
    _logger.error("%s", error, exc_info=error.__cause__)
    exposed = ToolError(f"{error}: {error.__cause__}")
    exposed.__cause__ = error.__cause__
    return exposed


def field(model: Any, name: str) -> Any:
    """An mcp type's field by its wire name (``isError``): mcp 2 spells the attribute in snake case, mcp 1 as on the
    wire."""
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return getattr(model, snake) if hasattr(model, snake) else getattr(model, name)


_SERVING_SERVER: type | None = None


def _serving_server_class() -> type:
    """``MCPServer`` holding the serving options it was built with, which ``streamable_http_app()`` and ``run()``
    default to, as mcp 1's FastMCP read them from its settings. Built once mcp 2 is known to be installed."""
    global _SERVING_SERVER
    if _SERVING_SERVER is None:
        from mcp.server.mcpserver import MCPServer

        class ServingServer(MCPServer):
            def __init__(self, name: str, serving: dict[str, Any], **kwargs: Any) -> None:
                super().__init__(name, **kwargs)
                self.serving = serving

            def streamable_http_app(self, **kwargs: Any) -> Any:
                defaults = {key: value for key, value in self.serving.items() if key != "port"}
                return super().streamable_http_app(**{**defaults, **kwargs})

            async def run_streamable_http_async(self, **kwargs: Any) -> None:
                await super().run_streamable_http_async(**{**self.serving, **kwargs})

        _SERVING_SERVER = ServingServer
    return _SERVING_SERVER
