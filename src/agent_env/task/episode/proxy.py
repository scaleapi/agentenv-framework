"""The episode's end of a proxy agent: the driver calls of ``urn:agentenv:proxy/v1``, and a proxy served in this process
for an outside agent that works in no sandbox."""

from __future__ import annotations

import asyncio
import secrets
import threading
from typing import Any

import httpx
import uvicorn
from agentenv_protocol.a2a_agent import PROXY_V1
from agentenv_protocol.a2a_agent.proxy import ProxyAgent

from agent_env.a2a_agent.a2a_agent import A2AAgent

_CALL_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_SERVE_TIMEOUT_SECONDS = 30.0
# Short, so a long-poll in flight when the episode ends finishes within the server's shutdown.
_IN_PROCESS_LONG_POLL_SECONDS = 1.0
_ATTEMPTS = 3
_RETRY_SECONDS = 1.0


class ProxyRefused(Exception):
    """The proxy refused a call, a 400 or 409, saying why."""


class ProxyClient:
    """One proxy, claimed with a token of this episode's, at the endpoints its card names. A call that doesn't reach
    the proxy, over a connection a tunnel dropped say, is tried again, except ``record``'s, which isn't idempotent."""

    def __init__(self, a2a_url: str, card: dict[str, Any]):
        proxy = A2AAgent.find_extension(card, PROXY_V1.uri)
        if proxy is None:
            raise RuntimeError(f"the agent at {a2a_url} isn't a proxy: its card has no {PROXY_V1.uri}")
        self._endpoints = {operation: a2a_url + A2AAgent.operation(proxy, operation)[1]
                           for operation in PROXY_V1.operations}
        self._token = secrets.token_urlsafe(24)
        self._client = httpx.AsyncClient(timeout=_CALL_TIMEOUT)

    async def claim(self) -> None:
        await self._call("claim")

    async def next(self, after: int) -> dict[str, Any]:
        return await self._call("next", after=after)

    async def tools(self, task_id: str) -> list[dict[str, Any]]:
        return (await self._call("tools", task_id=task_id))["tools"]

    async def act(self, task_id: str, action: dict[str, Any]) -> dict[str, Any]:
        """``action``'s result, once it has run in the turn. A call that didn't reach the proxy is tried again: the
        action's id makes a retry return the action it already started."""
        action_id = secrets.token_hex(16)
        state = await self._call("act", task_id=task_id, action_id=action_id, action=action)
        while not state["done"]:
            state = await self._call("result", action_id=action_id)
        return state["result"]

    async def record(self, task_id: str, message: str | dict[str, Any], model: str | None,
                     usage: dict[str, Any] | None) -> None:
        _raise_for_status(await self._post("record", {"task_id": task_id, "message": message, "model": model,
                                                      "usage": usage}, attempts=1))

    async def reply(self, task_id: str, **reply: Any) -> bool:
        """Whether the reply reached an open turn: a turn its sender stopped waiting for is gone, as is one a reply that
        reached it but whose answer was lost ended."""
        response = await self._post("reply", {"task_id": task_id, **reply})
        if response.status_code == 404:
            return False
        _raise_for_status(response)
        return True

    async def close(self) -> None:
        await self._client.aclose()

    async def _call(self, operation: str, **body: Any) -> dict[str, Any]:
        response = await self._post(operation, body)
        _raise_for_status(response)
        return response.json() or {}

    async def _post(self, operation: str, body: dict[str, Any], *, attempts: int = _ATTEMPTS) -> httpx.Response:
        for attempt in range(1, attempts + 1):
            try:
                return await self._client.post(self._endpoints[operation], json={"token": self._token, **body})
            except httpx.TransportError:
                if attempt == attempts:
                    raise
                await asyncio.sleep(_RETRY_SECONDS)
        raise AssertionError("unreachable")


class InProcessProxy:
    """A proxy served on loopback in a thread of its own, with no shell: one for an outside agent that works in no
    sandbox, which only this process's steps message."""

    def __init__(self) -> None:
        agent = ProxyAgent(shell=False, long_poll_seconds=_IN_PROCESS_LONG_POLL_SECONDS)
        self._server = uvicorn.Server(uvicorn.Config(agent.create_app(), host="127.0.0.1", port=0, log_level="warning",
                                                     timeout_graceful_shutdown=2 * _IN_PROCESS_LONG_POLL_SECONDS))
        self._thread = threading.Thread(target=self._server.run, name="agentenv-proxy", daemon=True)

    async def start(self) -> str:
        """Serve it, and return its URL once it answers."""
        self._thread.start()
        async with asyncio.timeout(_SERVE_TIMEOUT_SECONDS):
            while not self._server.started:
                if not self._thread.is_alive():
                    raise RuntimeError("the in-process proxy stopped as it started")
                await asyncio.sleep(0.01)
        port = self._server.servers[0].sockets[0].getsockname()[1]
        return f"http://127.0.0.1:{port}"

    def stop(self) -> None:
        """Have it stop; its thread ends once the requests in flight have."""
        self._server.should_exit = True


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code in (400, 409):
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        raise ProxyRefused(str(detail))
    response.raise_for_status()
