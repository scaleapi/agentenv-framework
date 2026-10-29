"""The /litellm-gemini proxy the CLI's model calls go through: attribution headers from the
run's own config, which its route names, the verbatim /gemini passthrough, the chat-completions
bridge for provider-prefixed models, and its errors."""

from __future__ import annotations

import json

import httpx
import pytest
from agentenv_protocol.a2a_agent import AGENT_CONFIG_V1

from tst.unit.agents.gemini.agent import a2a_server, gemini_bridge
from tst.unit.agents.gemini.support import send

AGENT_CONFIG_PATH = AGENT_CONFIG_V1.endpoint
PROBE_PATH = "/ext/attribution-probe"
BASE = "https://litellm.example"
TOKEN = "t1"
PROXY = f"/litellm-gemini/{TOKEN}"
GENERATE = f"{PROXY}/v1beta/models/gemini-3.1-pro-preview:generateContent"
STREAM = f"{PROXY}/v1beta/models/gemini-3.1-pro-preview:streamGenerateContent"


def _route(agent, model=None, project_id=None, task_id=None) -> None:
    """A running turn's route, as run() registers it for its CLI."""
    agent._routes[TOKEN] = a2a_server._Route(model, project_id, task_id)


class Upstream:
    """LiteLLM behind the proxy: records each request and its client's options, and answers
    with ``response`` (an exception is raised instead)."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.client_options: list[dict] = []
        self.response: httpx.Response | Exception = httpx.Response(200, json={"ok": True})

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


@pytest.fixture
def upstream(monkeypatch) -> Upstream:
    fake = Upstream()
    real_async_client = httpx.AsyncClient

    def _client(**kwargs):
        fake.client_options.append(kwargs)
        return real_async_client(transport=httpx.MockTransport(fake), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _client)
    monkeypatch.setenv("LITELLM_BASE_URL", BASE + "/")
    monkeypatch.setenv("LITELLM_API_KEY", "sk-FAKEfake0000fake0000fak")
    monkeypatch.setattr(gemini_bridge, "_last_attribution", {"project_id": None, "task_id": None, "last_seen_at_utc": None})
    return fake


def test_the_passthrough_forwards_verbatim_with_attribution(upstream, agent, client):
    upstream.response = httpx.Response(
        207, content=b'{"candidates":[]}',
        headers={"content-type": "application/json", "x-upstream": "1", "connection": "keep-alive"},
    )
    _route(agent, model="gemini-2.5-flash", project_id="p-1", task_id="t-1")

    response = client.post(
        GENERATE + "?alt=json&key=k",
        content=b'{"contents":[]}',
        headers={"x-goog-api-key": "g", "keep-alive": "timeout=5", "content-type": "application/json"},
    )

    (request,) = upstream.requests
    assert (request.method, str(request.url)) == (
        "POST", f"{BASE}/gemini/v1beta/models/gemini-3.1-pro-preview:generateContent?alt=json&key=k")
    assert request.content == b'{"contents":[]}'
    assert request.headers["x-goog-api-key"] == "g"
    assert request.headers["x-litellm-customer-id"] == "p-1"
    assert request.headers["x-litellm-tags"] == "projectId:p-1,taskId:t-1"
    assert request.headers["host"] == "litellm.example" and "keep-alive" not in request.headers
    assert (response.status_code, response.content) == (207, b'{"candidates":[]}')
    assert response.headers["x-upstream"] == "1" and "connection" not in response.headers
    assert client.post(PROBE_PATH).json()["last_seen_attribution"] == {"project_id": "p-1", "task_id": "t-1"}


def test_the_endpoints_model_headers_are_forwarded_under_the_runs_attribution(monkeypatch, upstream, agent, client):
    headers = {"X-Route": "eu", "X-LiteLLM-Tags": "useCase:evals,taskId:own"}
    monkeypatch.setenv("LITELLM_EXTRA_HEADERS", json.dumps(headers))
    _route(agent, project_id="p-1", task_id="t-1")

    client.post(GENERATE, content=b"{}", headers={"x-route": "cli", "x-litellm-tags": "from-cli"})

    (request,) = upstream.requests
    assert request.headers.get_list("x-route") == ["eu"]
    assert request.headers.get_list("x-litellm-tags") == ["useCase:evals,taskId:own,projectId:p-1"]


@pytest.mark.parametrize("raw", ["not json", '["x-route", "eu"]'])
def test_model_headers_that_are_not_a_json_object_are_ignored(monkeypatch, upstream, agent, client, raw):
    monkeypatch.setenv("LITELLM_EXTRA_HEADERS", raw)
    _route(agent)

    assert client.post(GENERATE, content=b"{}").status_code == 200
    assert "x-route" not in upstream.requests[0].headers


@pytest.mark.parametrize("method", ["GET", "PUT", "DELETE", "PATCH"])
def test_every_method_is_forwarded(upstream, agent, client, method):
    _route(agent)
    response = client.request(method, f"{PROXY}/v1beta/models")

    assert response.status_code == 200 and upstream.requests[0].method == method


def test_nothing_to_attribute_sends_no_attribution_headers(upstream, agent, client):
    _route(agent)
    client.get(f"{PROXY}/v1beta/models")

    (request,) = upstream.requests
    assert "x-litellm-customer-id" not in request.headers and "x-litellm-tags" not in request.headers
    assert client.post(PROBE_PATH).json() == {"last_seen_attribution": {}}


def test_the_upstream_hop_verifies_tls(upstream, agent, client):
    _route(agent)
    client.get(f"{PROXY}/v1beta/models")

    assert upstream.client_options == [{"timeout": 600}]


def test_a_call_no_running_turn_named_is_a_404(upstream, agent, client):
    _route(agent)
    response = client.get("/litellm-gemini/other/v1beta/models")

    assert (response.status_code, response.json()) == (404, {"detail": "no running turn has this proxy route"})
    assert upstream.requests == []


def test_each_run_routes_its_model_calls_with_its_own_config(monkeypatch, agent, client):
    """A turn's CLI names its own route, which carries that turn's config, so turns running at once each attribute
    their own calls; the route goes once the turn ends."""
    monkeypatch.setenv("LITELLM_BASE_URL", BASE)
    seen = []

    async def fake_run_gemini(**kwargs):
        token = kwargs["env"]["GOOGLE_GEMINI_BASE_URL"].rsplit("/", 1)[1]
        seen.append((kwargs["env"]["GOOGLE_GEMINI_BASE_URL"], agent._routes[token]))
        return gemini_bridge.GeminiResult(response="ok", tool_call_count=0, session_id=None)

    monkeypatch.setattr(a2a_server, "run_gemini", fake_run_gemini)
    client.post(AGENT_CONFIG_PATH, json={"project_id": "p-1", "task_id": "t-1", "model": "vertex_ai/global/g"})
    send(client, "go")

    ((url, route),) = seen
    assert url.startswith("http://127.0.0.1:") and "/litellm-gemini/" in url
    assert route == a2a_server._Route("vertex_ai/global/g", "p-1", "t-1")
    assert agent._routes == {}


def test_an_unset_base_url_is_a_503(monkeypatch, upstream, agent, client):
    _route(agent)
    monkeypatch.delenv("LITELLM_BASE_URL")

    response = client.post(GENERATE, content=b"{}")

    assert (response.status_code, response.json()) == (
        503, {"detail": "LITELLM_BASE_URL not set; cannot forward gemini model calls"})
    assert upstream.requests == []


def test_an_unreachable_upstream_is_a_502(upstream, agent, client):
    _route(agent)
    upstream.response = httpx.ConnectError("All connection attempts failed")

    response = client.post(GENERATE, content=b"{}")

    assert (response.status_code, response.json()) == (
        502, {"detail": "gemini upstream request failed: All connection attempts failed"})


# ── provider-prefixed models: the chat-completions bridge ────────────────────────


@pytest.fixture
def vertex(upstream, agent):
    _route(agent, model="vertex_ai/global/gemini-3.1-pro-preview", project_id="p-1")
    upstream.response = httpx.Response(200, json={
        "model": "gemini-3.1-pro-preview",
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    })
    return upstream


_GREQ = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}], "generationConfig": {"temperature": 0.1}}


def test_a_generate_call_is_bridged_to_chat_completions(vertex, client):
    response = client.post(GENERATE, json=_GREQ)

    (request,) = vertex.requests
    assert (request.method, str(request.url)) == ("POST", f"{BASE}/v1/chat/completions")
    assert request.headers["authorization"] == "Bearer sk-FAKEfake0000fake0000fak"
    assert request.headers["x-litellm-customer-id"] == "p-1" and request.headers["x-litellm-tags"] == "projectId:p-1"
    assert json.loads(request.content) == gemini_bridge.gemini_request_to_openai(_GREQ, "vertex_ai/global/gemini-3.1-pro-preview")
    assert response.headers["content-type"] == "application/json"
    assert response.json() == gemini_bridge.openai_response_to_gemini(json.loads(vertex.response.content))


def test_the_bridge_forwards_the_endpoints_model_headers_under_the_runs_attribution(monkeypatch, vertex, client):
    monkeypatch.setenv("LITELLM_EXTRA_HEADERS", json.dumps(
        {"x-route": "eu", "authorization": "Bearer gateway", "X-LiteLLM-Customer-Id": "spoofed"}))

    client.post(GENERATE, json=_GREQ)

    (request,) = vertex.requests
    assert request.headers.get_list("x-route") == ["eu"]
    assert request.headers.get_list("authorization") == ["Bearer gateway"]
    assert request.headers.get_list("x-litellm-customer-id") == ["p-1"]


def test_a_stream_call_is_answered_as_one_sse_event(vertex, client):
    response = client.post(STREAM + "?alt=sse", json=_GREQ)

    assert response.headers["content-type"].startswith("text/event-stream")
    gresp = gemini_bridge.openai_response_to_gemini(json.loads(vertex.response.content))
    assert response.content == f"data: {json.dumps(gresp)}\n\n".encode()


def test_count_tokens_keeps_the_passthrough(vertex, client):
    client.post(f"{PROXY}/v1beta/models/gemini-3.1-pro-preview:countTokens", json=_GREQ)

    assert str(vertex.requests[0].url) == f"{BASE}/gemini/v1beta/models/gemini-3.1-pro-preview:countTokens"


def test_bridge_upstream_errors(vertex, client):
    vertex.response = httpx.Response(429, content=b'{"error":"slow down"}', headers={"content-type": "application/json"})
    limited = client.post(GENERATE, json=_GREQ)
    vertex.response = httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})
    garbled = client.post(GENERATE, json=_GREQ)
    vertex.response = httpx.ConnectError("refused")
    unreachable = client.post(GENERATE, json=_GREQ)

    assert (limited.status_code, limited.content) == (429, b'{"error":"slow down"}')
    assert (garbled.status_code, garbled.json()) == (502, {"detail": "vertex bridge: non-JSON upstream response"})
    assert (unreachable.status_code, unreachable.json()) == (502, {"detail": "vertex bridge upstream failed: refused"})
