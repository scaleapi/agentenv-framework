"""The probe's view of one running driver: its card, MCP sessions on its tools as a role, its framework extensions,
the data plane, and commands through computer-exec."""
from __future__ import annotations

import base64
import contextlib
import io
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult
from PIL import Image as PILImage

from ... import client as protocol
from ...tool_context import ROLE_HEADER
from ..card import COMPUTER_EXEC

#: How long an MCP session's event stream may stay quiet.
STREAM_READ_S = 300


class Driver:
    def __init__(self, url: str, card: dict, timeout_s: float, token: str | None = None) -> None:
        self.url = url.rstrip("/")
        self.card = card
        self.timeout_s = timeout_s
        self.headers = protocol.control_headers(token) if token else None

    @classmethod
    async def connect(cls, url: str, timeout_s: float, token: str | None = None) -> Driver:
        return cls(url, await protocol.get_card(url.rstrip("/"), timeout=int(timeout_s)), timeout_s, token)

    @property
    def name(self) -> str:
        return self.card.get("name") or ""

    @property
    def extensions(self) -> list[str]:
        return [e["uri"] for e in (self.card.get("capabilities") or {}).get("extensions") or []]

    def declares(self, uri: str) -> bool:
        return uri in self.extensions

    def params(self, uri: str) -> dict:
        return protocol.extension_params(self.card, uri)

    @contextlib.asynccontextmanager
    async def session(self, role: str | None = None) -> AsyncIterator[ClientSession]:
        headers = {ROLE_HEADER: role} if role else {}
        timeout = httpx.Timeout(self.timeout_s, read=max(self.timeout_s, STREAM_READ_S))
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as http:
            async with streamable_http_client(self.url + protocol.mcp_path(self.card), http_client=http) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    yield session

    async def invoke(self, uri: str, method: str, params: dict | None = None) -> Any:
        return await protocol.invoke_extension(self.url, self.card, uri, params or {}, method=method,
                                               timeout=int(self.timeout_s), headers=self.headers)

    async def rpc(self, method: str, params: dict | None = None, card: dict | None = None) -> dict:
        """``method``'s answer from the data plane of ``card``, the driver's own or a child's it fronts."""
        async with httpx.AsyncClient(timeout=self.timeout_s) as http:
            url = self.url + ((card or self.card).get("url") or "/agentenv")
            response = await http.post(url, headers=protocol.request_headers(url, self.headers),
                                       json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})
        return response.json()

    async def run(self, *argv: str, timeout_s: float = 30) -> dict:
        return await self.invoke(COMPUTER_EXEC, "run", {"argv": list(argv), "request_id": uuid.uuid4().hex,
                                                        "timeout_s": timeout_s})

    async def add_file(self, path: str, data: bytes) -> dict:
        part = {"kind": "file", "file": {"bytes": base64.b64encode(data).decode("ascii")},
                "metadata": {"urn:agentenv:data-objects/v1": {"path": path}}}
        return await self.rpc("data/add", {"parts": [part]})


def image(result: CallToolResult) -> PILImage.Image:
    block = next(c for c in result.content if c.type == "image")
    return PILImage.open(io.BytesIO(base64.b64decode(block.data))).convert("RGB")
