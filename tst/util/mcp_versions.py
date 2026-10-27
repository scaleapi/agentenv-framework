"""An in-memory client session on an mcp server, on mcp 1 and mcp 2 alike, whose results read by mcp 1's attribute
names (``isError``, ``structuredContent``); the protocol package's tests keep the same helper."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

from agentenv_protocol.mcp_compat import mcp_2
from pydantic import BaseModel

MCP_2 = mcp_2()
if MCP_2:
    from mcp import Client
else:
    from mcp.shared.memory import create_connected_server_and_client_session


@contextlib.asynccontextmanager
async def memory_session(app: Any) -> AsyncIterator[Any]:
    """An initialized client session on ``app``, in memory."""
    if MCP_2:
        async with Client(app) as client:
            yield _Session(client)
    else:
        async with create_connected_server_and_client_session(app) as session:
            yield _Session(session)


def wire(value: Any) -> Any:
    """An mcp result as mcp 1 names its fields: as it is on mcp 1, and on mcp 2 a model's fields as attributes under
    their camel-case wire names, a mapping kept a mapping."""
    if not MCP_2:
        return value
    if isinstance(value, BaseModel):
        fields = {(info.alias if info.alias and not info.alias.startswith("_") else name): wire(getattr(value, name))
                  for name, info in type(value).model_fields.items()}
        return SimpleNamespace(**fields, **(value.model_extra or {}))
    if isinstance(value, list):
        return [wire(item) for item in value]
    return value


class _Session:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return wire(await self._session.call_tool(name, arguments, **kwargs))

    async def list_tools(self, **kwargs: Any) -> Any:
        return wire(await self._session.list_tools(**kwargs))
