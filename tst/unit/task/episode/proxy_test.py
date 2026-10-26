"""The episode's client of a proxy: which calls it sends again when one doesn't reach the proxy, and how long it
waits for each."""

import httpx
import pytest
from agentenv_protocol.a2a_agent import ProxyTextPart, ProxyTool
from agentenv_protocol.a2a_agent.proxy import LONG_POLL_SECONDS, TOOL_TIMEOUT_SECONDS, ProxyAgent

from agent_env.task.episode import proxy as client_module
from agent_env.task.episode.proxy import ProxyClient

CARD = ProxyAgent().create_app().state.agentenv_a2a.card.model_dump(mode="json", by_alias=True, exclude_none=True)
PARTS = [ProxyTextPart(text="Added a crate.")]


def _client(monkeypatch, answer):
    """A client whose transport drops the first request to each endpoint, as a tunnel drops an idle connection, and
    answers the next with ``answer``."""
    monkeypatch.setattr(client_module, "_RETRY_SECONDS", 0)
    sent: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.path)
        if sent.count(request.url.path) == 1:
            raise httpx.RemoteProtocolError("Server disconnected without sending a response.")
        return answer

    client = ProxyClient("http://proxy", CARD)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    return client, sent


@pytest.mark.asyncio
async def test_a_reply_that_doesnt_reach_the_proxy_is_sent_again(monkeypatch):
    client, sent = _client(monkeypatch, httpx.Response(200, json={}))

    assert await client.reply("task-1", parts=PARTS) is True
    assert sent == ["/ext/proxy/reply", "/ext/proxy/reply"]


@pytest.mark.asyncio
async def test_a_reply_whose_answer_was_lost_finds_its_turn_ended(monkeypatch):
    client, _ = _client(monkeypatch, httpx.Response(404, json={"detail": "no open turn 'task-1'"}))

    assert await client.reply("task-1", parts=PARTS) is False


@pytest.mark.asyncio
async def test_a_listing_that_doesnt_reach_the_proxy_is_asked_again(monkeypatch):
    client, sent = _client(monkeypatch, httpx.Response(200, json={"tools": [{"name": "bash"}]}))

    assert await client.tools("task-1") == [ProxyTool(name="bash")]
    assert sent == ["/ext/proxy/tools", "/ext/proxy/tools"]


@pytest.mark.asyncio
async def test_a_recorded_model_call_is_never_sent_twice(monkeypatch):
    client, sent = _client(monkeypatch, httpx.Response(200, json={}))

    with pytest.raises(httpx.RemoteProtocolError):
        await client.record("task-1", "Adding one.", None, None)
    assert sent == ["/ext/proxy/record"]


@pytest.mark.asyncio
async def test_fields_a_newer_proxy_adds_are_ignored():
    turn = {"task_id": "task-1", "context_id": "c", "parts": [{"kind": "text", "text": "Add one.", "lang": "en"}],
            "priority": 1}
    client = ProxyClient("http://proxy", CARD)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"version": 2, "turns": [turn], "paused": False})))

    answer = await client.next(0)

    assert (answer.version, [t.parts[0].text for t in answer.turns]) == (2, ["Add one."])


@pytest.mark.asyncio
async def test_a_call_that_waits_for_the_turns_listing_waits_as_long_as_the_proxy_may_take_to_list():
    waits: dict[str, float] = {}

    answers = {"tools": {"tools": []}, "next": {"version": 1, "turns": []}}

    def handle(request: httpx.Request) -> httpx.Response:
        operation = request.url.path.rsplit("/", 1)[-1]
        waits[operation] = request.extensions["timeout"]["read"]
        return httpx.Response(200, json=answers.get(operation, {}))

    client = ProxyClient("http://proxy", CARD)
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    await client.tools("task-1")
    await client.record("task-1", "Adding one.", None, None)
    await client.next(0)
    await client.reply("task-1", parts=PARTS)

    assert waits["tools"] > TOOL_TIMEOUT_SECONDS and waits["record"] > TOOL_TIMEOUT_SECONDS
    assert LONG_POLL_SECONDS < waits["next"] == waits["reply"] < TOOL_TIMEOUT_SECONDS
