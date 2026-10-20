"""Peer A2A agents exposed to pi as MCP tools, served by the agent itself at ``/mcp`` over loopback.

A minimal streamable-HTTP MCP server: JSON responses to POSTed requests, 202 for notifications, and no
server-to-client stream (405 on GET), which is all pi's MCP client needs for tool calls. It answers loopback
clients only, and only for a running task: each run gets a random token, which pi sends as a header whose value
it reads from its environment, and the token names the run's A2A context, so every context holds its own
conversation with each peer and a task cannot speak in another's.
"""

from __future__ import annotations

import json
import secrets
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from agentenv_protocol.a2a_agent import PeerAgent
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
PEER_TIMEOUT_SECONDS = 900
RUN_HEADER = "x-pi-a2a-run"
_LOOPBACK = frozenset({"127.0.0.1", "localhost"})
_LOOPBACK_CLIENTS = frozenset({"127.0.0.1", "::1"})
# Seen from inside a container, a peer published on the host's loopback is at the host gateway.
_HOST_FROM_CONTAINER = "host.docker.internal"

TOOLS = [
    {
        "name": "peer_list",
        "description": "List the peer agents you can send messages to.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "peer_send_message",
        "description": (
            "Send a message to a peer agent and return its reply. Messages to the same peer within this task's "
            "conversation continue one peer conversation unless new_conversation is true."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "peer_name": {"type": "string", "description": "The peer's name, as listed by peer_list."},
                "message": {"type": "string", "description": "The message to send."},
                "new_conversation": {"type": "boolean", "description": "Start a fresh conversation with the peer."},
            },
            "required": ["peer_name", "message"],
        },
    },
]


def _rpc_url(peer: PeerAgent) -> str:
    card_url = peer.card.get("url") or "/a2a"
    return urljoin(peer.url.rstrip("/") + "/", card_url)


def _via_host_gateway(url: str) -> str | None:
    parts = urlsplit(url)
    if parts.hostname not in _LOOPBACK:
        return None
    netloc = _HOST_FROM_CONTAINER + (f":{parts.port}" if parts.port else "")
    return urlunsplit(parts._replace(netloc=netloc))


def _reply_text(result: Mapping[str, Any]) -> str:
    message = result if result.get("kind") == "message" else (result.get("status") or {}).get("message") or {}
    texts = [part.get("text", "") for part in message.get("parts") or () if part.get("kind") == "text"]
    state = (result.get("status") or {}).get("state")
    text = "\n".join(texts)
    return text if state in (None, "completed") else f"[peer task {state}] {text}"


class Peers:
    def __init__(self) -> None:
        self.agents: dict[str, PeerAgent] = {}
        self._contexts: dict[tuple[str, str], str] = {}
        self._runs: dict[str, str] = {}

    @contextmanager
    def run(self, context_id: str) -> Iterator[str]:
        """A token that speaks for ``context_id`` until the run ends."""
        token = secrets.token_urlsafe(32)
        self._runs[token] = context_id
        try:
            yield token
        finally:
            del self._runs[token]

    def set(self, peers: list[PeerAgent]) -> None:
        self.agents = {peer.name: peer for peer in peers}
        self._contexts = {key: context for key, context in self._contexts.items() if key[1] in self.agents}

    def listing(self) -> list[dict[str, Any]]:
        return [{"name": peer.name, "url": peer.url, "description": peer.description} for peer in self.agents.values()]

    async def send(self, origin: str, peer_name: str, message: str, *, new_conversation: bool = False) -> str:
        """Message ``peer_name`` in the conversation the A2A context ``origin`` holds with it."""
        peer = self.agents.get(peer_name)
        if peer is None:
            raise ValueError(f"unknown peer {peer_name!r}; known peers: {sorted(self.agents)}")
        key = (origin, peer_name)
        if new_conversation:
            self._contexts.pop(key, None)
        context_id = self._contexts.setdefault(key, uuid.uuid4().hex)
        payload = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex,
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "messageId": uuid.uuid4().hex,
                    "role": "user",
                    "contextId": context_id,
                    "parts": [{"kind": "text", "text": message}],
                },
                "configuration": {"blocking": True},
            },
        }
        url = _rpc_url(peer)
        async with httpx.AsyncClient(timeout=PEER_TIMEOUT_SECONDS) as client:
            try:
                response = await client.post(url, json=payload)
            except httpx.ConnectError:
                fallback = _via_host_gateway(url)
                if fallback is None:
                    raise
                response = await client.post(fallback, json=payload)
        response.raise_for_status()
        body = response.json()
        if "error" in body:
            raise RuntimeError(f"peer {peer_name!r} rejected the message: {body['error'].get('message')}")
        return _reply_text(body["result"])

    async def mcp(self, request: Request) -> Response:
        if request.client is None or request.client.host not in _LOOPBACK_CLIENTS:
            return Response(status_code=403)
        origin = self._runs.get(request.headers.get(RUN_HEADER, ""))
        if origin is None:
            return Response(status_code=403)
        if request.method != "POST":
            return Response(status_code=405)
        body = await request.json()
        messages = body if isinstance(body, list) else [body]
        replies = [reply for message in messages if (reply := await self._answer(message, origin)) is not None]
        if not replies:
            return Response(status_code=202)
        return JSONResponse(replies if isinstance(body, list) else replies[0])

    async def _answer(self, message: Mapping[str, Any], origin: str) -> dict[str, Any] | None:
        if "id" not in message:
            return None
        method, params = message.get("method"), message.get("params") or {}
        if method == "initialize":
            requested = params.get("protocolVersion")
            result: dict[str, Any] = {
                "protocolVersion": requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "peers", "version": "1.0.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            result = await self._call(params.get("name"), params.get("arguments") or {}, origin)
        else:
            return {"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": f"unknown method {method}"}}
        return {"jsonrpc": "2.0", "id": message["id"], "result": result}

    async def _call(self, name: str | None, arguments: Mapping[str, Any], origin: str) -> dict[str, Any]:
        try:
            if name == "peer_list":
                text = json.dumps(self.listing())
            elif name == "peer_send_message":
                text = await self.send(
                    origin,
                    str(arguments["peer_name"]),
                    str(arguments["message"]),
                    new_conversation=bool(arguments.get("new_conversation")),
                )
            else:
                raise ValueError(f"unknown tool {name!r}")
        except (KeyError, ValueError, RuntimeError, httpx.HTTPError) as exc:
            return {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
        return {"content": [{"type": "text", "text": text}], "isError": False}
