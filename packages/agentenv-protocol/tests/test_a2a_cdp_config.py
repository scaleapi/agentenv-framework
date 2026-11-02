"""cdp-config/v1 as the SDK serves it: the card entry, add, list and remove, the endpoints every turn gets, and
``request.cdp_endpoint()`` failing the turn closed."""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from starlette.testclient import TestClient

from agentenv_protocol.a2a_agent import (
    CDP_CONFIG_V1,
    AgentEnvAgent,
    AgentIdentity,
    TaskProgress,
    TaskRequest,
    TaskResult,
    TextPart,
    a2a_agent,
)
from agentenv_protocol.a2a_agent.tasks.v1 import AgentConfig

CHROMIUM = {
    "name": "chromium",
    "http_url": "http://browser.test:8000/cdp/token-1/",
    "ws_url": "ws://browser.test:8000/cdp/token-1/",
    "metadata": {"browser": "chromium", "context": "default"},
}


@a2a_agent(
    identity=AgentIdentity(name="cdp-agent", description="test", version="1"),
    extensions=(CDP_CONFIG_V1,),
)
class CdpAgent(AgentEnvAgent):
    """Answers with the endpoint its prompt names, or the only one when the prompt is ``only``."""

    async def run(self, request: TaskRequest) -> TaskResult:
        prompt = request.parts[0].text
        endpoint = request.cdp_endpoint(None if prompt == "only" else prompt)
        return TaskResult.text(f"{endpoint['http_url']} {endpoint['ws_url']} {sorted(endpoint['headers'] or {})}")


def _turn(client: TestClient, prompt: str) -> dict[str, Any]:
    message = {
        "kind": "message",
        "messageId": uuid.uuid4().hex,
        "role": "user",
        "contextId": "context-1",
        "parts": [{"kind": "text", "text": prompt}],
    }
    request = {
        "jsonrpc": "2.0",
        "id": uuid.uuid4().hex,
        "method": "message/send",
        "params": {"message": message, "configuration": {"blocking": True}},
    }
    return client.post("/a2a", json=request).json()["result"]


def _reply(task: dict[str, Any]) -> str:
    assert task["status"]["state"] == "completed", task["status"]
    return task["status"]["message"]["parts"][0]["text"]


def _error(task: dict[str, Any]) -> dict[str, Any]:
    assert task["status"]["state"] == "failed"
    return task["status"]["message"]["parts"][-1]["data"]


def test_the_card_lists_add_list_and_remove() -> None:
    with TestClient(CdpAgent().create_app()) as client:
        card = client.get("/.well-known/agent-card.json").json()

    [extension] = [item for item in card["capabilities"]["extensions"] if item["uri"] == "urn:agentenv:cdp-config/v1"]
    assert extension["params"]["endpoint"] == "/ext/cdp-config"
    methods = extension["params"]["methods"]
    assert {name: method["method"] for name, method in methods.items()} == {
        "add": "POST",
        "list": "GET",
        "remove": "DELETE",
    }
    assert all("endpoint" not in method for method in methods.values())


def test_adding_a_name_again_replaces_it_and_list_hides_urls_and_header_values() -> None:
    with TestClient(CdpAgent().create_app()) as client:
        assert client.post("/ext/cdp-config", json=CHROMIUM).json() == {"status": "added", "name": "chromium"}
        replaced = {**CHROMIUM, "http_url": "http://browser.test:8000/cdp/token-2/", "headers": {"X-Secret": "s3"}}
        client.post("/ext/cdp-config", json=replaced)
        listing = client.get("/ext/cdp-config")

    assert listing.json() == {
        "cdp_endpoints": {"chromium": {"has_headers": True, "metadata": CHROMIUM["metadata"]}}
    }
    assert "s3" not in listing.text
    assert "token-" not in listing.text


def test_remove_takes_a_name_and_a_name_that_is_not_there_is_not_an_error() -> None:
    with TestClient(CdpAgent().create_app()) as client:
        client.post("/ext/cdp-config", json=CHROMIUM)
        removed = client.request("DELETE", "/ext/cdp-config", json={"name": "chromium"})
        again = client.request("DELETE", "/ext/cdp-config", json={"name": "chromium"})
        listing = client.get("/ext/cdp-config").json()

    assert removed.json() == {"status": "removed", "name": "chromium"}
    assert again.status_code == 200
    assert again.json() == {"status": "absent", "name": "chromium"}
    assert listing == {"cdp_endpoints": {}}


@pytest.mark.parametrize(
    "change",
    [
        {"name": ""},
        {"name": "-chromium"},
        {"name": "chromium buyer"},
        {"http_url": "ws://browser.test/cdp/"},
        {"http_url": "/cdp/token-1/"},
        {"http_url": "http://:8080/"},
        {"ws_url": "http://browser.test/cdp/"},
        {"ws_url": "ws:///cdp/"},
        {"ws_url": "ws://browser.test:abc/"},
        {"ws_url": "ws://browser.test:70000/"},
        {"extra": "field"},
    ],
)
def test_add_refuses_a_bad_name_url_or_field(change: dict[str, str]) -> None:
    with TestClient(CdpAgent().create_app()) as client:
        response = client.post("/ext/cdp-config", json={**CHROMIUM, **change})
        listing = client.get("/ext/cdp-config").json()

    assert response.status_code == 400
    assert listing == {"cdp_endpoints": {}}


def test_each_turn_gets_the_endpoints_added_before_it_with_their_headers() -> None:
    with TestClient(CdpAgent().create_app()) as client:
        before = _turn(client, "only")
        client.post("/ext/cdp-config", json={**CHROMIUM, "headers": {"Authorization": "Bearer t"}})
        with_one = _turn(client, "only")
        client.request("DELETE", "/ext/cdp-config", json={"name": "chromium"})
        after = _turn(client, "only")

    assert _error(before) == {
        "error_type": "infra_error",
        "error_code": "cdp.no_endpoint",
        "error_message": "the task gave no browser",
    }
    assert _reply(with_one) == f"{CHROMIUM['http_url']} {CHROMIUM['ws_url']} ['Authorization']"
    assert _error(after)["error_code"] == "cdp.no_endpoint"


def test_several_endpoints_need_a_name() -> None:
    buyer = {**CHROMIUM, "name": "chromium-buyer", "http_url": "http://browser.test:8000/cdp/token-b/"}
    with TestClient(CdpAgent().create_app()) as client:
        client.post("/ext/cdp-config", json=CHROMIUM)
        client.post("/ext/cdp-config", json=buyer)
        unnamed = _turn(client, "only")
        named = _turn(client, "chromium-buyer")
        unknown = _turn(client, "chromium-seller")

    assert _error(unnamed) == {
        "error_type": "infra_error",
        "error_code": "cdp.ambiguous",
        "error_message": "the task gave several browsers; name one of chromium, chromium-buyer",
    }
    assert _reply(named).startswith("http://browser.test:8000/cdp/token-b/ ")
    assert _error(unknown) == {
        "error_type": "infra_error",
        "error_code": "cdp.no_endpoint",
        "error_message": "the task gave no browser named 'chromium-seller'",
    }


def test_a_streaming_run_fails_closed_the_same_way() -> None:
    @a2a_agent(
        identity=AgentIdentity(name="cdp-streaming", description="test", version="1"),
        extensions=(CDP_CONFIG_V1,),
    )
    class StreamingAgent(AgentEnvAgent):
        async def run(self, request: TaskRequest):
            yield TaskProgress.text("connecting")
            yield TaskResult.text(request.cdp_endpoint()["ws_url"])

    with TestClient(StreamingAgent().create_app()) as client:
        task = _turn(client, "only")

    assert _error(task)["error_code"] == "cdp.no_endpoint"


def test_a_turn_holds_a_copy_of_the_endpoints() -> None:
    endpoints = {"chromium": {**CHROMIUM, "headers": None}}
    request = TaskRequest(
        task_id="task-1",
        context_id="context-1",
        parts=(TextPart(text="only"),),
        config=AgentConfig(),
        cdp_endpoints=endpoints,
    )
    endpoints["chromium"]["ws_url"] = "ws://elsewhere.test/"

    assert request.cdp_endpoint()["ws_url"] == CHROMIUM["ws_url"]
    assert request.cdp_endpoint("chromium") == request.cdp_endpoints["chromium"]

