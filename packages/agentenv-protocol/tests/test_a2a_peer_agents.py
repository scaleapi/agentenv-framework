"""peer-agents/v1 as the SDK serves it: the routing table, the ``peer_agents`` MCP server registered on the agent's
own port, and the six tools reaching a real peer agent's A2A server."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1,
    PEER_AGENTS_V1,
    AgentEnvAgent,
    AgentIdentity,
    DataPart,
    PeerAgentsSetRequest,
    TaskRequest,
    TaskResult,
    TextPart,
    a2a_agent,
    extension,
)
from agentenv_protocol.a2a_agent import peer_agents as peer_agents_module
from agentenv_protocol.a2a_agent.peer_agents import PEER_MCP_PATH, PEER_MCP_SERVER_NAME

PORT = 8000
LOOPBACK = {"base_url": f"http://127.0.0.1:{PORT}", "client": ("127.0.0.1", 50000)}
PEER_TOOLS_URL = f"http://127.0.0.1:{PORT}{PEER_MCP_PATH}"
REVIEWER_URL = "http://reviewer.test"
TOOLS = ["peer_list", "peer_get_card", "peer_send_message", "peer_start_task", "peer_get_task", "peer_cancel_task"]


@a2a_agent(identity=AgentIdentity(name="lead", description="Delegates to its peers", version="1"),
           extensions=(MCP_CONFIG_V1, PEER_AGENTS_V1))
class Lead(AgentEnvAgent):
    def __init__(self) -> None:
        self.requests: list[TaskRequest] = []

    async def run(self, request: TaskRequest) -> TaskResult:
        self.requests.append(request)
        return TaskResult.text("ok")


@a2a_agent(identity=AgentIdentity(name="reviewer", description="Reviews what it is sent", version="1"))
class Reviewer(AgentEnvAgent):
    """Answers "reviewed: <prompt>"; holds a "slow" prompt until released, and answers "report" in three parts."""

    def __init__(self) -> None:
        self.contexts: list[str] = []
        self.release = asyncio.Event()

    async def run(self, request: TaskRequest) -> TaskResult:
        self.contexts.append(request.context_id)
        text = " ".join(part.text for part in request.parts if isinstance(part, TextPart))
        if text == "slow":
            await self.release.wait()
        if text == "report":
            return TaskResult.success(parts=(TextPart(text="first"), DataPart(data={"score": 3}),
                                             TextPart(text="second")))
        return TaskResult.text(f"reviewed: {text}")


@pytest.fixture
def reviewer(monkeypatch) -> Reviewer:
    """A peer agent, its A2A server reached in-process wherever the lead's tools send."""
    agent = Reviewer()
    app = agent.create_app()
    monkeypatch.setattr(peer_agents_module, "_http_client",
                        lambda: httpx.AsyncClient(transport=httpx.ASGITransport(app=app)))
    monkeypatch.setattr(peer_agents_module, "_POLL_INTERVAL_SECONDS", 0.01)
    return agent


@pytest.fixture
def card(reviewer) -> dict[str, Any]:
    with TestClient(reviewer.create_app()) as client:
        return client.get("/.well-known/agent.json").json()


def _mcp(client: TestClient, method: str, params: dict[str, Any] | None = None) -> httpx.Response:
    return client.post(PEER_MCP_PATH, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}},
                       headers={"Accept": "application/json, text/event-stream"})


def _tool(client: TestClient, tool: str, /, **arguments: Any) -> tuple[bool, Any]:
    """A tool's error flag and its JSON answer, or its error text."""
    result = _mcp(client, "tools/call", {"name": tool, "arguments": arguments}).json()["result"]
    (content,) = result["content"]
    return result.get("isError", False), (content["text"] if result.get("isError") else json.loads(content["text"]))


def _set(client: TestClient, peers: list[dict[str, Any]]) -> httpx.Response:
    return client.post(PEER_AGENTS_V1.endpoint, json={"peers": peers})


def test_set_replaces_the_routing_table_and_an_empty_one_clears_it() -> None:
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        table = [{"name": "reviewer", "url": REVIEWER_URL, "description": "Reviews"}]
        assert _set(client, table).json() == {"status": "updated", "peers": 1}
        assert _set(client, table).status_code == 200  # setting the same table again is no conflict
        assert client.get(PEER_AGENTS_V1.endpoint).json() == {"peers": table}
        assert _set(client, []).json() == {"status": "updated", "peers": 0}
        assert client.get(PEER_AGENTS_V1.endpoint).json() == {"peers": []}


def test_peer_names_must_be_unique() -> None:
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        response = _set(client, [{"name": "a", "url": "http://a.test"}, {"name": "a", "url": "http://b.test"}])
        assert (response.status_code, response.text) == (400, "peer names must be unique")


def test_the_tools_are_registered_on_the_agents_own_port_and_stay_registered_once_the_table_is_cleared() -> None:
    lead = Lead()
    with TestClient(lead.create_app(), **LOOPBACK) as client:
        registered = {PEER_MCP_SERVER_NAME: {"url": PEER_TOOLS_URL, "has_headers": False}}
        assert client.get(MCP_CONFIG_V1.endpoint).json() == {"mcp_servers": registered}
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL}])
        _set(client, [])
        assert client.get(MCP_CONFIG_V1.endpoint).json() == {"mcp_servers": registered}
        client.post("/a2a", json={"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
            "message": {"messageId": "m-1", "role": "user", "parts": [{"kind": "text", "text": "hi"}]}}})

    (request,) = lead.requests
    assert request.mcp_servers == {PEER_MCP_SERVER_NAME: {"url": PEER_TOOLS_URL, "headers": None}}


def test_the_peer_servers_name_is_the_sdks() -> None:
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        response = client.post(MCP_CONFIG_V1.endpoint, json={"url": "http://other/mcp", "name": PEER_MCP_SERVER_NAME})
        assert response.status_code == 409
        assert client.post(MCP_CONFIG_V1.endpoint, json={"url": "http://other/mcp", "name": "env"}).status_code == 200


def test_the_server_lists_the_six_tools() -> None:
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        tools = _mcp(client, "tools/list").json()["result"]["tools"]

    assert [tool["name"] for tool in tools] == TOOLS
    # A peer waiting on its caller can't progress, so the model is told to stop polling and answer.
    assert all("input-required" in tool["description"] for tool in tools
               if tool["name"] in {"peer_send_message", "peer_start_task", "peer_get_task"})
    send = next(tool for tool in tools if tool["name"] == "peer_send_message")
    assert send["inputSchema"]["required"] == ["name"]
    assert set(send["inputSchema"]["properties"]) == {"name", "prompt", "parts", "context_id", "timeout_seconds"}
    wait = send["inputSchema"]["properties"]["timeout_seconds"]
    assert wait["default"] == 45 and "capped at 50 seconds" in wait["description"]


def test_peer_list_and_peer_get_card_describe_the_peers(card) -> None:
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL, "card": card}])
        failed, peers = _tool(client, "peer_list")
        _, got = _tool(client, "peer_get_card", name="reviewer")

    assert not failed and [(peer["name"], peer["description"], peer["url"]) for peer in peers] == [
        ("reviewer", "Reviews what it is sent", REVIEWER_URL)]
    assert got == card


def test_peer_send_message_waits_for_the_peers_reply_and_a_context_carries_the_conversation(reviewer, card) -> None:
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL, "card": card}])
        failed, first = _tool(client, "peer_send_message", name="reviewer", prompt="the diff")
        _, second = _tool(client, "peer_send_message", name="reviewer", prompt="again",
                          context_id=first["context_id"])

    assert not failed and (first["state"], first["response"]) == ("completed", "reviewed: the diff")
    assert "history" not in first
    assert second["response"] == "reviewed: again" and reviewer.contexts == [first["context_id"]] * 2


def test_peer_start_task_returns_at_once_and_peer_get_task_reads_the_result(reviewer, card) -> None:
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL, "card": card}])
        _, started = _tool(client, "peer_start_task", name="reviewer", prompt="later")
        for _ in range(100):
            _, task = _tool(client, "peer_get_task", task_id=started["task_id"])
            if task["state"] == "completed":
                break
        _, with_history = _tool(client, "peer_get_task", task_id=started["task_id"], history_length=1)

    assert set(started) == {"task_id", "context_id", "state"}
    assert (task["task_id"], task["state"], task["response"]) == (started["task_id"], "completed", "reviewed: later")
    assert "history" not in task and with_history["history"][0]["parts"] == [{"kind": "text", "text": "later"}]


def test_peer_cancel_task_cancels_a_running_task(reviewer, card) -> None:
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL, "card": card}])
        _, started = _tool(client, "peer_start_task", name="reviewer", prompt="slow")
        failed, cancelled = _tool(client, "peer_cancel_task", task_id=started["task_id"])

    assert not failed and cancelled == {"task_id": started["task_id"], "state": "canceled"}


def test_a_peer_that_runs_past_the_wait_leaves_the_task_working(reviewer, card, monkeypatch) -> None:
    timeouts: list[float] = []
    call = peer_agents_module._call

    async def recording(rpc_url: str, method: str, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if method == "tasks/get":
            timeouts.append(kwargs.get("timeout", peer_agents_module._REQUEST_TIMEOUT_SECONDS))
        return await call(rpc_url, method, params, **kwargs)

    monkeypatch.setattr(peer_agents_module, "_call", recording)
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL, "card": card}])
        _, waited = _tool(client, "peer_send_message", name="reviewer", prompt="slow", timeout_seconds=1)
        reviewer.release.set()

    assert waited["state"] == "working" and set(waited) == {"task_id", "context_id", "state"}
    # A poll may outlast the wait by the last poll's budget, not by a whole request timeout.
    assert timeouts and max(timeouts) <= peer_agents_module._LAST_POLL_TIMEOUT_SECONDS


def test_a_wait_past_the_maximum_is_capped_to_it(reviewer, card, monkeypatch) -> None:
    monkeypatch.setattr(peer_agents_module, "_MAX_WAIT_SECONDS", 1)
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL, "card": card}])
        started = time.monotonic()
        failed, waited = _tool(client, "peer_send_message", name="reviewer", prompt="slow", timeout_seconds=600)
        took = time.monotonic() - started
        reviewer.release.set()

    assert not failed and waited["state"] == "working" and took < 10


def test_the_send_counts_toward_the_wait(reviewer, card, monkeypatch) -> None:
    polls: list[str] = []
    call, send = peer_agents_module._call, peer_agents_module._send

    async def recording(rpc_url: str, method: str, params: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        polls.extend([method] if method == "tasks/get" else [])
        return await call(rpc_url, method, params, **kwargs)

    async def slow_send(rpc_url: str, message: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(1.2)
        return await send(rpc_url, message)

    monkeypatch.setattr(peer_agents_module, "_call", recording)
    monkeypatch.setattr(peer_agents_module, "_send", slow_send)
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL, "card": card}])
        _, waited = _tool(client, "peer_send_message", name="reviewer", prompt="slow", timeout_seconds=1)
        reviewer.release.set()

    assert waited["state"] == "working" and polls == []


def test_a_call_the_peer_doesnt_answer_in_time_is_a_tool_error(reviewer, card, monkeypatch) -> None:
    async def stalled_send(rpc_url: str, message: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(5)
        raise AssertionError("the budget should have ended the call")

    monkeypatch.setattr(peer_agents_module, "_CALL_BUDGET_SECONDS", 0.2)
    monkeypatch.setattr(peer_agents_module, "_send", stalled_send)
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL, "card": card}])
        failed, error = _tool(client, "peer_start_task", name="reviewer", prompt="hi")

    assert failed and "peer_start_task got no answer from the peer within 0.2s" in error


def test_a_reply_keeps_every_part_the_peer_answered_with(reviewer, card) -> None:
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL, "card": card}])
        _, reply = _tool(client, "peer_send_message", name="reviewer", prompt="report")

    assert reply["response"] == "first\nsecond"
    assert [(part["kind"], part["data"]) for part in reply["parts"]] == [("data", {"score": 3})]


def test_a_running_task_outlives_finished_ones_at_the_tracking_limit(reviewer, card, monkeypatch) -> None:
    monkeypatch.setattr(peer_agents_module, "_MAX_TRACKED_TASKS", 2)
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL, "card": card}])
        _, running = _tool(client, "peer_start_task", name="reviewer", prompt="slow")
        for prompt in ("one", "two"):
            _tool(client, "peer_send_message", name="reviewer", prompt=prompt)
        failed, still = _tool(client, "peer_get_task", task_id=running["task_id"])
        reviewer.release.set()

    assert not failed and still["task_id"] == running["task_id"]


def test_an_unknown_peer_or_task_is_a_tool_error_naming_what_there_is(reviewer) -> None:
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": REVIEWER_URL}])
        failed, peer_error = _tool(client, "peer_send_message", name="writer", prompt="hi")
        task_failed, task_error = _tool(client, "peer_get_task", task_id="nope")

    assert failed and "unknown peer 'writer'. Available peers: reviewer" in peer_error
    assert task_failed and "unknown task_id 'nope'" in task_error


def test_a_peer_url_ending_in_its_rpc_path_is_not_extended(reviewer, card) -> None:
    with TestClient(Lead().create_app(), **LOOPBACK) as client:
        _set(client, [{"name": "reviewer", "url": f"{REVIEWER_URL}/a2a", "card": card}])
        _, answer = _tool(client, "peer_send_message", name="reviewer", prompt="x")

    assert answer["response"] == "reviewed: x"


def test_the_tools_serve_loopback_callers_only() -> None:
    with TestClient(Lead().create_app(), base_url=LOOPBACK["base_url"], client=("10.0.0.5", 50000)) as client:
        response = _mcp(client, "tools/list")

    assert response.status_code == 403


def test_the_tools_answer_only_while_the_application_lifespan_runs() -> None:
    client = TestClient(Lead().create_app(), **LOOPBACK)  # no `with`, so the lifespan never starts

    assert _mcp(client, "tools/list").status_code == 503


def test_an_agent_with_its_own_peer_handlers_gets_no_sdk_tools() -> None:
    @a2a_agent(identity=AgentIdentity(name="own-peers", description="test", version="1"), extensions=(MCP_CONFIG_V1,))
    class OwnPeers(AgentEnvAgent):
        async def run(self, request: TaskRequest) -> TaskResult:
            return TaskResult.text("ok")

        @extension(PEER_AGENTS_V1.set)
        async def set_peers(self, request: PeerAgentsSetRequest):
            return {"status": "updated"}

        @extension(PEER_AGENTS_V1.list)
        async def list_peers(self):
            return {"peers": []}

    with TestClient(OwnPeers().create_app(), **LOOPBACK) as client:
        assert client.get(MCP_CONFIG_V1.endpoint).json() == {"mcp_servers": {}}
        assert _mcp(client, "tools/list").status_code == 404
