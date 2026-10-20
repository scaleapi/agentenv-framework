"""peer-agents/v1 as the SDK serves it. A task sets the routing table, the agents this one may delegate to, and the
agent's model reaches them through six MCP tools on the agent's own port, which the SDK registers with the agent as
the ``peer_agents`` MCP server. The tools wrap A2A's own methods: ``peer_send_message`` sends and waits,
``peer_start_task`` and ``peer_get_task`` send and poll, ``peer_cancel_task`` cancels, and ``peer_list`` and
``peer_get_card`` describe the peers."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import PlainTextResponse
from starlette.types import Receive, Scope, Send

from ..transfers import serving_port
from .extensions import PeerAgent

logger = logging.getLogger(__name__)

PEER_MCP_SERVER_NAME = "peer_agents"
PEER_MCP_PATH = "/ext/peer-agents/mcp"

_TERMINAL_STATES = frozenset({"completed", "failed", "canceled", "rejected"})
# A2A states a task doesn't leave without its caller: finished, or waiting on the caller.
_SETTLED_STATES = _TERMINAL_STATES | {"input-required", "auth-required"}
_DEFAULT_RPC_PATH = "/a2a"
# Many MCP clients, Claude Code's among them, give an HTTP request 60 s: a tool call answers within the budget,
# and a wait, counting its send, ends early enough for its last poll to fit too.
_CALL_BUDGET_SECONDS = 58
_DEFAULT_WAIT_SECONDS = 45
_MAX_WAIT_SECONDS = 50
_SEND_TIMEOUT_SECONDS = 60
_REQUEST_TIMEOUT_SECONDS = 30
# The most a wait's last poll may run past the wait's own limit.
_LAST_POLL_TIMEOUT_SECONDS = 5.0
_POLL_INTERVAL_SECONDS = 5.0
_POLL_BACKOFF_STEP_SECONDS = 5.0
_POLL_BACKOFF_MAX_SECONDS = 60.0
_CONNECT_ATTEMPTS = 5
_CONNECT_BACKOFF_SECONDS = 2.0
_CONNECT_BACKOFF_MAX_SECONDS = 15.0
_MAX_TRACKED_TASKS = 1_024
_LOOPBACK_CLIENTS = frozenset({"127.0.0.1", "::1", "::ffff:127.0.0.1"})
_LOOPBACK_SECURITY = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
    allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"],
)


class _Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _NoArguments(_Arguments):
    pass


class _PeerName(_Arguments):
    name: str = Field(description="The peer's name, as peer_list gives it.")


class _Message(_PeerName):
    prompt: str | None = Field(None, description="The text to send. Give this or parts.")
    parts: list[dict[str, Any]] | None = Field(None, description="A2A message parts to send instead of prompt.")
    context_id: str | None = Field(
        None, description="A context_id an earlier call returned, to continue that conversation with the peer.")


class _MessageAndWait(_Message):
    # No schema maximum, which the MCP server would refuse past: a longer wait is capped instead.
    timeout_seconds: int = Field(_DEFAULT_WAIT_SECONDS, ge=1, description=(
        f"How long to wait for the peer's result; a longer wait is capped at {_MAX_WAIT_SECONDS} seconds."))


class _TaskId(_Arguments):
    task_id: str = Field(description="The task_id peer_send_message or peer_start_task returned.")


class _TaskQuery(_TaskId):
    history_length: int | None = Field(None, ge=0, description="How many of the task's messages to return.")


@dataclass(frozen=True)
class _Tool:
    arguments: type[_Arguments]
    description: str
    run: Callable[[Any], Awaitable[Any]]


@dataclass
class _StartedTask:
    task_id: str
    peer_name: str
    rpc_url: str
    context_id: str | None
    state: str  # as last seen


class PeerAgents:
    """One agent application's routing table, and the MCP tools that reach the agents in it."""

    def __init__(self) -> None:
        self._peers: dict[str, PeerAgent] = {}
        self._tasks: OrderedDict[str, _StartedTask] = OrderedDict()
        self._manager: StreamableHTTPSessionManager | None = None
        self._tools = {
            "peer_list": _Tool(_NoArguments, (
                "Return the routing table of available peer A2A agents. Each entry has name, description, skills, "
                "default input/output modes, and supported capabilities. Call this before peer_send_message / "
                "peer_start_task to choose the right peer."), self._peer_list),
            "peer_get_card": _Tool(_PeerName, (
                "Return the full A2A AgentCard for a peer (skills with examples, extension parameters, etc.). Use "
                "when peer_list's summary isn't enough."), self._peer_get_card),
            "peer_send_message": _Tool(_MessageAndWait, (
                f"Send a message to a peer A2A agent and wait up to timeout_seconds ({_DEFAULT_WAIT_SECONDS} by "
                f"default, {_MAX_WAIT_SECONDS} at most) for the result.\n"
                "Returns {task_id, context_id, state, response?, parts?, artifacts?}. If state is 'working', the "
                "peer is still on it: keep polling with peer_get_task(task_id). If state is 'input-required' or "
                "'auth-required', the peer is waiting on you: answer it with peer_send_message in the same "
                "context_id.\nFor work expected to take longer, prefer peer_start_task + peer_get_task from the "
                "start so you stay responsive."), self._peer_send_message),
            "peer_start_task": _Tool(_Message, (
                "Start a peer A2A task and return immediately without waiting for it to finish. Returns "
                "{task_id, context_id, state}. You must then call peer_get_task(task_id) repeatedly until state is "
                "one of 'completed' | 'failed' | 'canceled' | 'rejected'. If state becomes 'input-required' or "
                "'auth-required', the peer is waiting on you: stop polling and answer it with peer_send_message in "
                "the task's context_id."), self._peer_start_task),
            "peer_get_task": _Tool(_TaskQuery, (
                "Fetch the current state of a peer task by id. Returns {state, response?, parts?, artifacts?, "
                "history?, error?}. Call repeatedly (with a small delay between calls) until state is terminal "
                "('completed', 'failed', 'canceled', or 'rejected'). 'input-required' or 'auth-required' means the "
                "peer is waiting on you: stop polling and answer it with peer_send_message in the task's "
                "context_id."), self._peer_get_task),
            "peer_cancel_task": _Tool(_TaskId, (
                "Request cancellation of an in-flight peer task. Returns the task's final state. Fails if the task "
                "is already in a terminal state."), self._peer_cancel_task),
        }
        self._server = self._mcp_server()
        self.app = _PeerToolsApp(self)

    def set(self, peers: list[PeerAgent]) -> None:
        """Replace the routing table: an empty list clears it."""
        self._peers = {peer.name: peer for peer in peers}

    def listing(self) -> list[dict[str, Any]]:
        return [peer.model_dump(include={"name", "url", "description"}, exclude_none=True)
                for peer in self._peers.values()]

    def registration(self) -> dict[str, Any] | None:
        """The ``peer_agents`` MCP server as the agent's model reaches it, at the port this agent serves on; None
        while that port is unknown."""
        port = serving_port()
        return None if port is None else {"url": f"http://127.0.0.1:{port}{PEER_MCP_PATH}", "headers": None}

    @asynccontextmanager
    async def serving(self) -> AsyncIterator[None]:
        """Serve the tools for the length of the application's lifespan."""
        manager = StreamableHTTPSessionManager(
            app=self._server, json_response=True, stateless=True, security_settings=_LOOPBACK_SECURITY)
        async with manager.run():
            self._manager = manager
            try:
                yield
            finally:
                self._manager = None

    async def handle(self, scope: Scope, receive: Receive, send: Send) -> None:
        client = scope.get("client")
        if scope["type"] != "http" or not client or client[0] not in _LOOPBACK_CLIENTS:
            refusal = PlainTextResponse("the peer tools serve this agent's own model, over loopback", status_code=403)
            await refusal(scope, receive, send)
        elif self._manager is None:
            idle = PlainTextResponse("the peer tools serve while the application's lifespan runs", status_code=503)
            await idle(scope, receive, send)
        else:
            await self._manager.handle_request(scope, receive, send)

    def _mcp_server(self) -> Server:
        server: Server = Server(PEER_MCP_SERVER_NAME)

        @server.list_tools()
        async def list_tools() -> list[types.Tool]:
            return [types.Tool(name=name, description=tool.description, inputSchema=_schema(tool.arguments))
                    for name, tool in self._tools.items()]

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
            tool = self._tools.get(name)
            if tool is None:
                raise ValueError(f"unknown tool '{name}'")
            try:
                result = await asyncio.wait_for(tool.run(tool.arguments.model_validate(arguments or {})),
                                                _CALL_BUDGET_SECONDS)
            except asyncio.TimeoutError:
                raise TimeoutError(f"{name} got no answer from the peer within {_CALL_BUDGET_SECONDS}s") from None
            return [types.TextContent(type="text", text=json.dumps(result))]

        return server

    async def _peer_list(self, _: _NoArguments) -> list[dict[str, Any]]:
        return [_summary(peer) for peer in self._peers.values()]

    async def _peer_get_card(self, arguments: _PeerName) -> dict[str, Any]:
        peer = self._resolve(arguments.name)
        if peer.card:
            return peer.card
        async with _http_client() as client:
            response = await client.get(f"{peer.url.rstrip('/')}/.well-known/agent.json",
                                        timeout=_REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()

    async def _peer_send_message(self, arguments: _MessageAndWait) -> dict[str, Any]:
        deadline = time.monotonic() + min(arguments.timeout_seconds, _MAX_WAIT_SECONDS)
        task = await self._start(arguments)
        if task.state in _SETTLED_STATES:
            result = await _call(task.rpc_url, "tasks/get", {"id": task.task_id})
        else:
            try:
                result = await _wait(task.rpc_url, task.task_id, deadline - time.monotonic())
            except TimeoutError:
                return {"task_id": task.task_id, "context_id": task.context_id, "state": "working"}
        return _outcome(task, result)

    async def _peer_start_task(self, arguments: _Message) -> dict[str, Any]:
        task = await self._start(arguments)
        return {"task_id": task.task_id, "context_id": task.context_id, "state": task.state}

    async def _peer_get_task(self, arguments: _TaskQuery) -> dict[str, Any]:
        task = self._tracked(arguments.task_id)
        params: dict[str, Any] = {"id": task.task_id}
        if arguments.history_length is not None:
            params["historyLength"] = arguments.history_length
        return _outcome(task, await _call(task.rpc_url, "tasks/get", params),
                        history=arguments.history_length is not None)

    async def _peer_cancel_task(self, arguments: _TaskId) -> dict[str, Any]:
        task = self._tracked(arguments.task_id)
        result = await _call(task.rpc_url, "tasks/cancel", {"id": task.task_id})
        task.state = _state(result)
        return {"task_id": task.task_id, "state": task.state}

    async def _start(self, arguments: _Message) -> _StartedTask:
        peer = self._resolve(arguments.name)
        rpc_url = _rpc_url(peer)
        message: dict[str, Any] = {"messageId": uuid.uuid4().hex, "role": "user", "parts": _parts(arguments)}
        if arguments.context_id:
            message["contextId"] = arguments.context_id
        result = await _send(rpc_url, message)
        task_id = result.get("id")
        if not task_id:
            raise RuntimeError(f"peer '{peer.name}' answered message/send without a task: {result}")
        task = _StartedTask(task_id=task_id, peer_name=peer.name, rpc_url=rpc_url,
                            context_id=arguments.context_id or result.get("contextId"), state=_state(result))
        self._tasks[task_id] = task
        while len(self._tasks) > _MAX_TRACKED_TASKS:
            # The oldest task known finished goes first, so a running one stays reachable.
            evicted = next((tracked for tracked, seen in self._tasks.items() if seen.state in _TERMINAL_STATES),
                           next(iter(self._tasks)))
            del self._tasks[evicted]
        return task

    def _resolve(self, name: str) -> PeerAgent:
        peer = self._peers.get(name)
        if peer is None:
            available = ", ".join(sorted(self._peers)) or "<none>"
            raise ValueError(f"unknown peer '{name}'. Available peers: {available}")
        return peer

    def _tracked(self, task_id: str) -> _StartedTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise ValueError(f"unknown task_id '{task_id}': start it with peer_send_message or peer_start_task")
        return task


class _PeerToolsApp:
    """The tools' ASGI endpoint: an object rather than a function, so Starlette routes it as an ASGI app."""

    def __init__(self, peers: PeerAgents) -> None:
        self._peers = peers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._peers.handle(scope, receive, send)


def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient()


def _schema(arguments: type[_Arguments]) -> dict[str, Any]:
    return {key: value for key, value in arguments.model_json_schema().items() if key != "title"}


def _rpc_url(peer: PeerAgent) -> str:
    """Where the peer takes A2A JSON-RPC: its card URL's path, the SDK's default without one, below its own URL,
    which may already end in it."""
    card_url = peer.card.get("url")
    path = (urlsplit(str(card_url)).path or "/") if card_url else _DEFAULT_RPC_PATH
    base = peer.url.rstrip("/")
    return base if urlsplit(base).path.endswith(path) else base + path


def _parts(arguments: _Message) -> list[dict[str, Any]]:
    if arguments.parts:
        return list(arguments.parts)
    if arguments.prompt is None:
        raise ValueError("either 'prompt' or 'parts' must be provided")
    return [{"kind": "text", "text": arguments.prompt}]


def _state(result: dict[str, Any]) -> str:
    return (result.get("status") or {}).get("state") or "submitted"


async def _call(rpc_url: str, method: str, params: dict[str, Any], *,
                timeout: float = _REQUEST_TIMEOUT_SECONDS) -> dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}
    async with _http_client() as client:
        response = await client.post(rpc_url, json=payload, timeout=timeout)
    response.raise_for_status()
    body = response.json()
    if "error" in body:
        raise RuntimeError(f"peer {method} failed: {body['error']}")
    return body.get("result") or {}


async def _send(rpc_url: str, message: dict[str, Any]) -> dict[str, Any]:
    """``message/send`` without waiting, retried while the peer can't be reached."""
    params = {"message": message, "configuration": {"blocking": False}}
    delay = _CONNECT_BACKOFF_SECONDS
    for attempt in range(1, _CONNECT_ATTEMPTS + 1):
        try:
            return await _call(rpc_url, "message/send", params, timeout=_SEND_TIMEOUT_SECONDS)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if attempt == _CONNECT_ATTEMPTS:
                raise
            logger.warning("peer message/send to %s didn't connect (attempt %d of %d): %s",
                           rpc_url, attempt, _CONNECT_ATTEMPTS, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2, _CONNECT_BACKOFF_MAX_SECONDS)
    raise AssertionError("unreachable")


async def _wait(rpc_url: str, task_id: str, timeout_seconds: float) -> dict[str, Any]:
    """Poll ``tasks/get`` until the task settles, backing off while polls fail; TimeoutError after
    ``timeout_seconds``, which a poll overruns by at most ``_LAST_POLL_TIMEOUT_SECONDS``. A client error is the
    task's own answer, so it is raised at once."""
    deadline = time.monotonic() + timeout_seconds
    failures = 0
    while (remaining := deadline - time.monotonic()) > 0:
        pause = _POLL_INTERVAL_SECONDS + failures * _POLL_BACKOFF_STEP_SECONDS
        await asyncio.sleep(min(pause, _POLL_BACKOFF_MAX_SECONDS, remaining))
        budget = max(deadline - time.monotonic(), _LAST_POLL_TIMEOUT_SECONDS)
        try:
            result = await _call(rpc_url, "tasks/get", {"id": task_id}, timeout=min(budget, _REQUEST_TIMEOUT_SECONDS))
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code < 500:
                raise
            failures += 1
            logger.warning("peer tasks/get for %s answered %d", task_id, exc.response.status_code)
            continue
        except (httpx.HTTPError, RuntimeError) as exc:
            failures += 1
            logger.warning("peer tasks/get for %s failed: %s", task_id, exc)
            continue
        failures = 0
        if _state(result) in _SETTLED_STATES:
            return result
    raise TimeoutError(f"peer task {task_id} didn't settle within {timeout_seconds}s")


def _summary(peer: PeerAgent) -> dict[str, Any]:
    card = peer.card or {}
    description = peer.description or card.get("description") or f"Peer A2A agent at {peer.url}"
    return {
        "name": peer.name,
        "description": description,
        "url": peer.url,
        "skills": [{"id": skill.get("id"), "name": skill.get("name"), "description": skill.get("description")}
                   for skill in card.get("skills") or []],
        "default_input_modes": card.get("default_input_modes") or card.get("defaultInputModes") or ["text"],
        "default_output_modes": card.get("default_output_modes") or card.get("defaultOutputModes") or ["text"],
        "capabilities": card.get("capabilities") if isinstance(card.get("capabilities"), dict) else {},
    }


def _outcome(task: _StartedTask, result: dict[str, Any], *, history: bool = False) -> dict[str, Any]:
    """A settled or running task as the tools return it, recording its state: the reply's text and its other
    parts, any artifacts, and its messages when ``history`` asks for them."""
    task.state = _state(result)
    payload: dict[str, Any] = {"task_id": task.task_id, "context_id": task.context_id, "state": task.state}
    message = (result.get("status") or {}).get("message") or {}
    parts = message.get("parts") or []
    texts = [part.get("text", "") for part in parts if part.get("kind") == "text"]
    if texts:
        payload["response"] = "\n".join(texts)
    if others := [part for part in parts if part.get("kind") != "text"]:
        payload["parts"] = others
    if result.get("artifacts"):
        payload["artifacts"] = list(result["artifacts"])
    if history and result.get("history"):
        payload["history"] = list(result["history"])
    if payload["state"] == "failed":
        payload["error"] = {"raw_message": message}
    return payload
