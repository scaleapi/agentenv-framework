"""MCP client for an environment's tools, over the streamable-HTTP transport on httpx alone: a trainer's or eval
harness's own loop lists and calls an env's tools with it, whatever `mcp` version that loop's environment pins."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from .tool_context import ROLE_HEADER
from .types import PROTOCOL_VERSION, EnvironmentTool

_MCP_PROTOCOL_VERSION = "2025-11-25"
_SESSION_HEADER = "Mcp-Session-Id"
_VERSION_HEADER = "MCP-Protocol-Version"
_ACCEPT = "application/json, text/event-stream"
_CLOSE_TIMEOUT_S = 5.0


class ToolSessionError(RuntimeError):
    """An MCP-level failure: a JSON-RPC error, an HTTP error status, a reply that can't be read, or a session the server
    no longer knows. A tool that fails is not one; its result carries ``isError``."""


class ToolResult(BaseModel):
    content: list[dict] = Field(default_factory=list)
    isError: bool = False

    @field_validator("content")
    @classmethod
    def _text_blocks_carry_text(cls, content: list[dict]) -> list[dict]:
        if any(block.get("type") == "text" and not isinstance(block.get("text"), str) for block in content):
            raise ValueError("a text block has no string text")
        return content

    @property
    def text(self) -> str:
        return "\n".join(block["text"] for block in self.content if block.get("type") == "text")


class ToolSession:
    """One MCP session on an env's ``mcp_url``. Use it on the event loop that opened it, and after any failed call treat
    the env's state as unknown: nothing is retried, and a call that raised may still run on the server."""

    def __init__(self, *, mcp_url: str, role: str | None = None, headers: dict[str, str] | None = None,
                 timeout: float = 600.0) -> None:
        self.mcp_url = mcp_url
        self.timeout = timeout
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self._headers = {**(headers or {}), **({ROLE_HEADER: role} if role else {}), "Accept": _ACCEPT}
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ids = itertools.count(1)

    async def __aenter__(self) -> ToolSession:
        return await self.open()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def open(self) -> ToolSession:
        """Start a new MCP session, ending the one this object holds, if any."""
        await self.close()
        self.session_id = self.protocol_version = None
        self._loop = asyncio.get_running_loop()
        self._client = httpx.AsyncClient(headers=self._headers, timeout=None, follow_redirects=True)
        try:
            result = await self._request("initialize", {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agentenv-protocol", "version": PROTOCOL_VERSION},
            })
            self.protocol_version = result.get("protocolVersion") or _MCP_PROTOCOL_VERSION
            await self._deadline("notifications/initialized",
                                 self._exchange({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        except BaseException:
            await self.close()
            raise
        return self

    async def close(self) -> None:
        """End the session on the server, waiting at most 5 s (or ``timeout``, if shorter), then close the connection; a
        server that is already gone is not an error."""
        if self._client is None:
            return
        client, self._client = self._open_client(), None
        try:
            if self.session_id:
                with contextlib.suppress(httpx.HTTPError, asyncio.TimeoutError):
                    await asyncio.wait_for(client.delete(self.mcp_url, headers=self._session_headers()),
                                           min(self.timeout, _CLOSE_TIMEOUT_S))
        finally:
            await client.aclose()

    async def list_tools(self) -> list[EnvironmentTool]:
        tools: list[EnvironmentTool] = []
        params: dict[str, Any] = {}
        while True:
            result = await self._request("tools/list", params)
            if not isinstance(result.get("tools"), list):
                raise ToolSessionError("tools/list: unreadable result: tools is not a list")
            tools += [_validated(EnvironmentTool, tool, "tools/list") for tool in result["tools"]]
            if not result.get("nextCursor"):
                return tools
            params = {"cursor": result["nextCursor"]}

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        result = await self._request("tools/call", {"name": name, "arguments": arguments or {}})
        return _validated(ToolResult, result, "tools/call")

    async def _request(self, method: str, params: dict[str, Any]) -> dict:
        message = {"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}
        reply = await self._deadline(method, self._exchange(message))
        if "error" in reply:
            error = reply["error"] if isinstance(reply["error"], dict) else {"message": reply["error"]}
            raise ToolSessionError(f"{method} failed: {error.get('message')} (code {error.get('code')})")
        if not isinstance(reply.get("result"), dict):
            raise ToolSessionError(f"{method}: the reply has no result object")
        return reply["result"]

    async def _deadline(self, method: str, exchange: Any) -> Any:
        try:
            return await asyncio.wait_for(exchange, self.timeout)
        except asyncio.TimeoutError as e:
            raise TimeoutError(f"{method} did not finish within {self.timeout}s") from e

    async def _exchange(self, message: dict) -> dict:
        async with self._open_client().stream("POST", self.mcp_url, json=message,
                                              headers=self._session_headers()) as response:
            await self._raise_for_status(response, message["method"])
            if "id" not in message:
                return {}
            if message["method"] == "initialize":
                self.session_id = response.headers.get(_SESSION_HEADER)
            content_type = response.headers.get("content-type", "")
            try:
                if content_type.startswith("text/event-stream"):
                    async for data in _sse_data(response):
                        reply = json.loads(data)
                        if isinstance(reply, dict) and reply.get("id") == message["id"] and (
                                "result" in reply or "error" in reply):
                            return reply
                    raise ToolSessionError(f"{message['method']}: the event stream ended before the reply")
                if content_type.startswith("application/json"):
                    reply = json.loads(await response.aread())
                    if isinstance(reply, dict):
                        return reply
                    raise ToolSessionError(f"{message['method']}: the reply is not a JSON-RPC object")
            except ValueError as e:
                raise ToolSessionError(f"{message['method']}: unreadable reply: {e}") from e
            raise ToolSessionError(f"{message['method']}: unexpected content type {content_type!r}")

    async def _raise_for_status(self, response: httpx.Response, method: str) -> None:
        if response.status_code == 404 and self.session_id:
            raise ToolSessionError(f"{method}: the server no longer knows session {self.session_id}; it restarted "
                                   "or ended the session, and the env's state may be gone with it")
        if response.status_code >= 400:
            body = (await response.aread()).decode(errors="replace")[:1000]
            raise ToolSessionError(f"{method}: HTTP {response.status_code} from {self.mcp_url}: {body}")

    def _session_headers(self) -> dict[str, str]:
        headers = {_SESSION_HEADER: self.session_id} if self.session_id else {}
        if self.protocol_version:
            headers[_VERSION_HEADER] = self.protocol_version
        return headers

    def _open_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("the ToolSession is not open: use `async with ToolSession(...)` or `await open()`")
        if asyncio.get_running_loop() is not self._loop:
            raise RuntimeError("the ToolSession was opened on another event loop; use and close it on that one")
        return self._client


def _validated(model: type[BaseModel], data: Any, method: str) -> Any:
    try:
        return model.model_validate(data)
    except ValueError as e:
        raise ToolSessionError(f"{method}: unreadable result: {e}") from e


async def _sse_data(response: httpx.Response) -> AsyncIterator[str]:
    """The data of each ``message`` event in a server-sent event stream; an event the stream ends inside is dropped,
    as the SSE spec has it, and so are events with no data, such as a resumable server's priming event."""
    event, data = "message", []
    async for line in response.aiter_lines():
        if not line:
            payload = "\n".join(data)
            if payload.strip() and event == "message":
                yield payload
            event, data = "message", []
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "data":
            data.append(value)
        elif field == "event":
            event = value
