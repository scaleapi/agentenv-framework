"""The task env's clock reaches every AgentEnvAgent: while a registered MCP server lists get_time (the gateway does only
while the env's clock is armed), each turn's request ends with the env's time and carries ``TaskRequest.env_clock``
for the agent's harness to read again between steps. Unarmed, nothing changes. Read here from real MCP servers over
streamable HTTP on this machine's loopback."""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    AgentEnvAgent,
    AgentIdentity,
    TaskRequest,
    TaskResult,
    TextPart,
    a2a_agent,
)
from agentenv_protocol.a2a_agent import env_clock, framework

NOW = "2026-05-01T09:00:00Z"


def _serve(server: FastMCP) -> tuple[str, uvicorn.Server]:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    running = uvicorn.Server(uvicorn.Config(server.streamable_http_app(), log_level="warning"))
    threading.Thread(target=lambda: asyncio.run(running.serve(sockets=[sock])), daemon=True).start()
    deadline = time.monotonic() + 10
    while not running.started:
        assert time.monotonic() < deadline, "the MCP server didn't start"
        time.sleep(0.02)
    return f"http://127.0.0.1:{port}/mcp", running


@pytest.fixture(scope="module")
def servers():
    """A gateway-like server listing get_time, and one without it."""
    gateway = FastMCP("env", stateless_http=True, json_response=True)

    @gateway.tool()
    def get_time() -> dict:
        return {"current_time": NOW}

    @gateway.tool()
    def list_notes() -> list[str]:
        return []

    other = FastMCP("peers", stateless_http=True, json_response=True)

    @other.tool()
    def peer_send_message(text: str) -> str:
        return text

    started = [_serve(gateway), _serve(other)]
    yield {"env": started[0][0], "peers": started[1][0]}
    for _, running in started:
        running.should_exit = True


def _servers(**urls: str) -> dict[str, env_clock.ClockServer]:
    return {name: env_clock.ClockServer(url=url) for name, url in urls.items()}


def test_the_server_listing_get_time_is_found_and_read(servers):
    found = asyncio.run(env_clock.find_env_clock(_servers(peers=servers["peers"], env=servers["env"])))

    assert found == env_clock.EnvClock(server=env_clock.ClockServer(url=servers["env"]), time=NOW)
    assert asyncio.run(found.read()) == NOW


def test_unarmed_no_server_lists_get_time_and_nothing_is_found(servers):
    assert asyncio.run(env_clock.find_env_clock(_servers(peers=servers["peers"]))) is None


def test_a_server_that_fails_is_skipped_for_the_next(servers):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    closed = f"http://127.0.0.1:{sock.getsockname()[1]}/mcp"
    sock.close()

    found = asyncio.run(env_clock.find_env_clock(_servers(gone=closed, env=servers["env"])))

    assert found is not None and found.time == NOW


def test_a_slow_clock_leaves_the_turn_without_it(monkeypatch):
    async def _hangs(server, *, find):
        await asyncio.sleep(60)

    monkeypatch.setattr(env_clock, "_read", _hangs)

    assert asyncio.run(env_clock.find_env_clock(_servers(env="http://unused"), timeout_s=0.05)) is None


def test_the_servers_carry_the_tasks_role_over_a_registrations():
    servers = env_clock.clock_servers({"env": {"url": "http://gw/mcp", "headers": {"AgentEnv-Role": "x", "X": "1"}},
                                       "bare": {"url": "http://b/mcp", "headers": None}}, "agent")

    assert servers == {"env": env_clock.ClockServer(url="http://gw/mcp", headers={"AgentEnv-Role": "agent", "X": "1"}),
                       "bare": env_clock.ClockServer(url="http://b/mcp", headers={"AgentEnv-Role": "agent"})}
    assert env_clock.clock_servers({"bare": {"url": "http://b/mcp"}}, None)["bare"].headers == {}


def test_a_server_travels_to_a_subprocess_through_its_environment():
    server = env_clock.ClockServer(url="http://gw/mcp", headers={"AgentEnv-Role": "agent"})

    assert env_clock.ClockServer.from_env(server.to_env()) == server


def test_a_failure_inside_the_mcp_clients_task_groups_is_named():
    class Group(Exception):
        def __init__(self, *inner):
            super().__init__("1 sub-exception")
            self.exceptions = inner

    assert env_clock.failure_reason(Group(Group(ConnectionError("refused")))) == "ConnectionError: refused"


def _message() -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": "r", "method": "message/send", "params": {
        "message": {"kind": "message", "messageId": "m", "role": "user", "contextId": "c",
                    "parts": [{"kind": "text", "text": "book it"}]},
        "configuration": {"blocking": True}}}


def _turn(monkeypatch, found, *, note: bool = True, register: bool = True) -> tuple[TaskRequest, list]:
    """One turn with ``find_env_clock`` answering ``found``; the request the agent got and what was searched."""
    seen: list[TaskRequest] = []
    searched: list = []

    async def _find(servers, timeout_s=env_clock.FIND_TIMEOUT_S):
        searched.append(servers)
        return found

    monkeypatch.setattr(framework, "find_env_clock", _find)

    @a2a_agent(identity=AgentIdentity(name="clocked", description="test", version="1"), extensions=(MCP_CONFIG_V1,))
    class Agent(AgentEnvAgent):
        env_clock_prompt_note = note

        async def run(self, request: TaskRequest) -> TaskResult:
            seen.append(request)
            return TaskResult.text("ok")

    with TestClient(Agent().create_app()) as client:
        if register:
            card = client.get("/.well-known/agent.json").json()
            [extension] = [e for e in card["capabilities"]["extensions"] if e["uri"] == MCP_CONFIG_V1.uri]
            add = extension["params"]["methods"]["add"]
            path = add.get("endpoint", extension["params"].get("endpoint"))
            assert client.request(add["method"], path, json={"name": "env", "url": "http://gw/mcp"}).is_success
        assert client.post("/a2a", json=_message()).json()["result"]["status"]["state"] == "completed"
    return seen[0], searched


def test_an_armed_clock_ends_every_agents_request_with_the_envs_time(monkeypatch):
    clock = env_clock.EnvClock(server=env_clock.ClockServer(url="http://gw/mcp"), time=NOW)
    request, searched = _turn(monkeypatch, clock)

    assert searched == [{"env": env_clock.ClockServer(url="http://gw/mcp")}]
    assert request.parts == (TextPart(text="book it"), TextPart(text=env_clock.prompt_note(NOW)))
    assert request.env_clock == clock


def test_an_agent_that_places_the_note_itself_gets_the_clock_alone(monkeypatch):
    clock = env_clock.EnvClock(server=env_clock.ClockServer(url="http://gw/mcp"), time=NOW)
    request, _ = _turn(monkeypatch, clock, note=False)

    assert request.parts == (TextPart(text="book it"),) and request.env_clock == clock


def test_an_unarmed_clock_leaves_the_request_alone(monkeypatch):
    request, _ = _turn(monkeypatch, None)

    assert request.parts == (TextPart(text="book it"),) and request.env_clock is None


def test_with_no_mcp_server_registered_there_is_no_clock_to_ask(monkeypatch):
    request, searched = _turn(monkeypatch, None, register=False)

    assert searched == [] and request.env_clock is None
