"""The tests' view of mcp 1 and mcp 2 alike: the high-level server (``FastMCP``, built as ``mcp_compat.server`` builds
it), its tool classes, and an in-memory session on a server whose results read by their wire names (``isError``,
``structuredContent``), as mcp 1 spells the attributes."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel

from agentenv_protocol.mcp_compat import mcp_2
from agentenv_protocol.mcp_compat import server as FastMCP

MCP_2 = mcp_2()
if MCP_2:
    import httpx2 as http
    from mcp import Client
    from mcp.server.mcpserver import Context
    from mcp.server.mcpserver.tools.base import Tool
else:
    import httpx as http
    from mcp.server.fastmcp import Context
    from mcp.server.fastmcp.tools.base import Tool
    from mcp.shared.memory import create_connected_server_and_client_session

__all__ = ["MCP_2", "Context", "FastMCP", "Tool", "http_session", "memory_session", "wire"]

BASE = "http://127.0.0.1:8000"


@contextlib.asynccontextmanager
async def memory_session(app: Any) -> AsyncIterator[Any]:
    """An initialized client session on ``app``, in memory."""
    if MCP_2:
        async with Client(app) as client:
            yield _Session(client)
    else:
        async with create_connected_server_and_client_session(app) as session:
            yield _Session(session)


@contextlib.asynccontextmanager
async def http_session(asgi: Any, *, headers: dict[str, str] | None = None) -> AsyncIterator[tuple[Any, Any]]:
    """An initialized client session on ``asgi``'s ``/mcp`` over streamable HTTP, in process, sending ``headers``, and
    a callable answering the id of the session the server opened for it (read from its response headers: mcp 2's
    client doesn't hand it over)."""
    opened: list[str] = []

    async def record(response: Any) -> None:
        if "mcp-session-id" in response.headers:
            opened.append(response.headers["mcp-session-id"])

    async with http.AsyncClient(transport=http.ASGITransport(app=asgi), base_url=BASE, headers=headers or {},
                                event_hooks={"response": [record]}) as client:
        async with streamable_http_client(f"{BASE}/mcp", http_client=client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                yield _Session(session), lambda: opened[0] if opened else None


def wire(value: Any) -> Any:
    """An mcp result with its fields named as mcp 1 names them (``isError``, ``meta``): as it is on mcp 1, and on mcp 2
    a model's fields as attributes under their camel-case wire names, a mapping kept a mapping."""
    if not MCP_2:
        return value
    if isinstance(value, BaseModel):
        fields = {_mcp_1_name(name, info.alias): wire(getattr(value, name))
                  for name, info in type(value).model_fields.items()}
        return SimpleNamespace(**fields, **(value.model_extra or {}))
    if isinstance(value, list):
        return [wire(item) for item in value]
    return value


def _mcp_1_name(name: str, alias: str | None) -> str:
    return alias if alias and not alias.startswith("_") else name


class _Session:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return wire(await self._session.call_tool(name, arguments, **kwargs))

    async def list_tools(self, **kwargs: Any) -> Any:
        return wire(await self._session.list_tools(**kwargs))
