"""MCP client for an environment's tools, over the streamable-HTTP transport on httpx alone: a trainer's or eval
harness's own loop lists and calls an env's tools with it, whatever `mcp` version that loop's environment pins."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import itertools
import json
from collections.abc import AsyncIterator, Iterable
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field, field_validator

from .types import PROTOCOL_VERSION, ROLE_HEADER, EnvironmentTool

MCP_PROTOCOL_VERSION = "2025-11-25"

_SESSION_HEADER = "Mcp-Session-Id"
_VERSION_HEADER = "MCP-Protocol-Version"
_ACCEPT = "application/json, text/event-stream"
_CLOSE_TIMEOUT_S = 5.0
_EMPTY_SCHEMA = {"type": "object", "properties": {}}

ToolApi = Literal["openai_chat", "openai_responses", "anthropic"]


class ToolSessionError(RuntimeError):
    """The server failed a request at the MCP level: a JSON-RPC error, an HTTP error status, a session it no longer
    knows, or a response that can't be read. A tool that fails is not one; its result carries ``isError``."""


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
    """One MCP session on an env's streamable-HTTP endpoint (``DeployedEnv.mcp_url``, or a card's address joined with
    ``client.mcp_path(card)``); ``role`` is sent as ``AgentEnv-Role``, which a gateway filters tools by. A request that
    outlives ``timeout`` (by default the gateway's own tool-call limit) raises TimeoutError, a failed connection
    httpx.TransportError. Nothing is retried, and a call that raised may still have run, or may yet run once its stalled
    request gets through, so after any failure the env's state is unknown."""

    def __init__(self, url: str, *, role: str | None = None, headers: dict[str, str] | None = None,
                 timeout: float = 600.0, verify: bool = True) -> None:
        self.url = url
        self.timeout = timeout
        self.session_id: str | None = None
        self.protocol_version: str | None = None
        self._headers = {**(headers or {}), **({ROLE_HEADER: role} if role else {}), "Accept": _ACCEPT}
        self._verify = verify
        self._client: httpx.AsyncClient | None = None
        self._ids = itertools.count(1)

    async def __aenter__(self) -> ToolSession:
        return await self.open()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def open(self) -> ToolSession:
        """Start a new MCP session, ending the one this object holds, if any."""
        await self.close()
        self.session_id = self.protocol_version = None
        self._client = httpx.AsyncClient(headers=self._headers, timeout=self.timeout, verify=self._verify,
                                         follow_redirects=True)
        try:
            result = await self._request("initialize", {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "agentenv-protocol", "version": PROTOCOL_VERSION},
            })
            self.protocol_version = result.get("protocolVersion") or MCP_PROTOCOL_VERSION
            await self._deadline("notifications/initialized", self._notify("notifications/initialized"))
        except BaseException:
            await self.close()
            raise
        return self

    async def close(self) -> None:
        """End the session on the server, waiting at most 5 s (or ``timeout``, if shorter), then close the connection; a
        server that is already gone is not an error."""
        client, self._client = self._client, None
        if client is None:
            return
        try:
            if self.session_id:
                with contextlib.suppress(httpx.HTTPError, asyncio.TimeoutError, TimeoutError):
                    await asyncio.wait_for(client.delete(self.url, headers=self._session_headers()),
                                           min(self.timeout, _CLOSE_TIMEOUT_S))
        finally:
            await client.aclose()

    async def list_tools(self) -> list[EnvironmentTool]:
        tools: list[EnvironmentTool] = []
        params: dict[str, Any] = {}
        while True:
            result = await self._request("tools/list", params)
            tools += [_validated(EnvironmentTool, tool, "tools/list") for tool in result.get("tools") or []]
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
        except (asyncio.TimeoutError, httpx.TimeoutException) as e:
            raise TimeoutError(f"{method} did not finish within {self.timeout}s") from e

    async def _exchange(self, message: dict) -> dict:
        async with self._open_client().stream("POST", self.url, json=message,
                                              headers=self._session_headers()) as response:
            await self._raise_for_status(response, message["method"])
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

    async def _notify(self, method: str) -> None:
        response = await self._open_client().post(self.url, json={"jsonrpc": "2.0", "method": method},
                                                  headers=self._session_headers())
        await self._raise_for_status(response, method)

    async def _raise_for_status(self, response: httpx.Response, method: str) -> None:
        if response.status_code == 404 and self.session_id:
            raise ToolSessionError(f"{method}: the server no longer knows session {self.session_id}; it restarted "
                                   "or ended the session, and the env's state may be gone with it")
        if response.status_code >= 400:
            body = (await response.aread()).decode(errors="replace")[:1000]
            raise ToolSessionError(f"{method}: HTTP {response.status_code} from {self.url}: {body}")

    def _session_headers(self) -> dict[str, str]:
        headers = {_SESSION_HEADER: self.session_id} if self.session_id else {}
        if self.protocol_version:
            headers[_VERSION_HEADER] = self.protocol_version
        return headers

    def _open_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("the ToolSession is not open: use `async with ToolSession(...)` or `await open()`")
        return self._client


_SHAPES = {
    "openai_chat": lambda name, description, schema: {
        "type": "function", "function": {"name": name, "description": description, "parameters": schema}},
    "openai_responses": lambda name, description, schema: {
        "type": "function", "name": name, "description": description, "parameters": schema},
    "anthropic": lambda name, description, schema: {"name": name, "description": description, "input_schema": schema},
}


def tool_definitions(tools: Iterable[EnvironmentTool], api: ToolApi) -> list[dict]:
    """The tools as a model API declares them: OpenAI Chat Completions, OpenAI Responses or Anthropic Messages.

    ``tools`` comes from ``ToolSession.list_tools()`` or a card's ``capabilities.tools``; each schema is a copy."""
    if api not in _SHAPES:
        raise ValueError(f"api must be one of {', '.join(sorted(_SHAPES))}, not {api!r}")
    shape = _SHAPES[api]
    return [shape(t.name, t.description or "", copy.deepcopy(t.inputSchema or _EMPTY_SCHEMA)) for t in tools]


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
