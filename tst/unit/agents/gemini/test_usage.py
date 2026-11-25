"""A Gemini turn's usage is what its proxy forwarded: every generate call the CLI made through its base URL, each
recorded with its counts, and the turn's usage their sum. A call the endpoint refused counts nothing; one answered
without counts, or still running when the turn ended, counts as a call whose usage is unknown."""

from __future__ import annotations

import json

import httpx
import pytest
from agentenv_protocol.a2a_agent import Usage

from tst.unit.agents.gemini.agent import a2a_server, gemini_bridge
from tst.unit.agents.gemini.support import send, terminal
from tst.unit.agents.gemini.test_litellm_proxy import GENERATE, PROXY, STREAM, TOKEN, _GREQ, _route

TRAJECTORY_PATH = "/ext/trajectory"


def _sse(*metas: dict) -> bytes:
    return b"".join(f"data: {json.dumps({'candidates': [{}], 'usageMetadata': meta})}\r\n\r\n".encode()
                    for meta in metas)


# gemini-baseline on gemini-2.5-flash, recorded on 2026-11-21 through Scale's LiteLLM proxy: each call's last
# usageMetadata as the proxy streamed it, Google leaving its zero counts out.
BASELINE = [
    {"promptTokenCount": 7311, "candidatesTokenCount": 22, "thoughtsTokenCount": 95, "totalTokenCount": 7428},
    {"promptTokenCount": 7350, "candidatesTokenCount": 18, "cachedContentTokenCount": 6332, "totalTokenCount": 7368},
    {"promptTokenCount": 7424, "candidatesTokenCount": 2, "thoughtsTokenCount": 44, "cachedContentTokenCount": 6338,
     "totalTokenCount": 7470},
]


def test_the_recorded_turns_calls_sum_to_what_the_proxy_billed(native, agent, client):
    """22,085 in (12,670 cached) and 181 out, thoughts included: the proxy's own usage for the turn's three calls."""
    for meta in BASELINE:
        native.response = httpx.Response(200, content=_sse({"promptTokenCount": meta["promptTokenCount"]}, meta),
                                         headers={"content-type": "text/event-stream"})
        client.post(STREAM + "?alt=sse", json=_GREQ)

    usage = a2a_server._turn_usage(agent._routes[TOKEN].calls, 2, [])

    assert (usage.input_tokens, usage.output_tokens, usage.cache_read_tokens, usage.total_tokens) == (
        22085, 181, 12670, 22266)
    assert usage.provider_details["reasoning_tokens"] == 139


def test_a_streamed_answers_counts_are_its_last_usage():
    body = _sse({"promptTokenCount": 100}, {"promptTokenCount": 100, "candidatesTokenCount": 7,
                                             "thoughtsTokenCount": 3, "cachedContentTokenCount": 64,
                                             "totalTokenCount": 110})

    assert gemini_bridge.native_call_usage(body) == {
        "input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 64, "reasoning_tokens": 3}


def test_an_answers_missing_counts_are_zero_and_a_tool_use_prompt_is_input():
    one = json.dumps({"usageMetadata": {"promptTokenCount": 9, "toolUsePromptTokenCount": 4,
                                        "candidatesTokenCount": 2}}).encode()
    chunks = json.dumps([{"usageMetadata": {"promptTokenCount": 5}},
                         {"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2}}]).encode()

    assert gemini_bridge.native_call_usage(one) == {
        "input_tokens": 13, "output_tokens": 2, "cache_read_tokens": 0, "reasoning_tokens": 0}
    assert gemini_bridge.native_call_usage(chunks)["output_tokens"] == 2


def test_candidates_that_hold_the_thoughts_already_are_not_added_to_twice():
    body = _sse({"promptTokenCount": 10, "candidatesTokenCount": 8, "thoughtsTokenCount": 5, "totalTokenCount": 18})

    assert gemini_bridge.native_call_usage(body)["output_tokens"] == 8


def test_an_answer_without_usage_has_no_counts():
    assert gemini_bridge.native_call_usage(b'data: {"candidates": [{}]}\n\n') is None
    assert gemini_bridge.native_call_usage(b"<html>") is None


def test_the_model_a_call_names_is_read_from_its_path():
    assert gemini_bridge.call_model("v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse") == \
        "gemini-2.5-flash"
    assert gemini_bridge.call_model("v1beta/files") is None


@pytest.fixture
def native(upstream, agent):
    _route(agent)
    return upstream


def _records(agent) -> list[dict]:
    return agent._routes[TOKEN].calls.records


def test_the_passthrough_records_each_generate_call_and_what_it_used(native, agent, client):
    native.response = httpx.Response(200, content=_sse({"promptTokenCount": 9, "candidatesTokenCount": 2}),
                                     headers={"content-type": "text/event-stream"})
    client.post(STREAM + "?alt=sse", json=_GREQ)
    client.post(f"{PROXY}/v1beta/models/gemini-3.1-pro-preview:countTokens", json=_GREQ)

    assert _records(agent) == [{
        "type": "agentenv.model_call", "via": "passthrough", "model": "gemini-3.1-pro-preview", "status": 200,
        "outcome": "answered",
        "usage": {"input_tokens": 9, "output_tokens": 2, "cache_read_tokens": 0, "reasoning_tokens": 0}}]


def test_a_refused_call_counts_nothing_and_one_answered_without_counts_is_unknown(native, agent, client):
    native.response = httpx.Response(429, json={"error": "slow down"})
    client.post(GENERATE, json=_GREQ)
    native.response = httpx.ConnectError("refused")
    client.post(GENERATE, json=_GREQ)
    native.response = httpx.Response(200, json={"candidates": [{}]})
    client.post(GENERATE, json=_GREQ)

    assert [(record["status"], record["outcome"]) for record in _records(agent)] == [
        (429, "refused"), (None, "refused"), (200, "unreported")]
    assert agent._routes[TOKEN].calls.running == 0


def test_the_bridge_records_the_cache_and_reasoning_its_chat_completion_carried(vertex, agent, client):
    vertex.response = httpx.Response(200, json={
        "model": "gemini-3.1-pro-preview", "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150,
                  "prompt_tokens_details": {"cached_tokens": 100}, "completion_tokens_details": {"reasoning_tokens": 20}}})
    client.post(GENERATE, json=_GREQ)
    vertex.response = httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})
    client.post(GENERATE, json=_GREQ)

    answered, garbled = _records(agent)
    assert (answered["via"], answered["model"], answered["usage"]) == (
        "bridge", "vertex_ai/global/gemini-3.1-pro-preview",
        {"input_tokens": 120, "output_tokens": 30, "cache_read_tokens": 100, "cache_write_tokens": None,
         "reasoning_tokens": 20})
    assert garbled["outcome"] == "unreported"


@pytest.fixture
def vertex(upstream, agent):
    _route(agent, model="vertex_ai/global/gemini-3.1-pro-preview", project_id="p-1")
    return upstream


def _calls(*records: dict, running: int = 0) -> a2a_server._Calls:
    calls = a2a_server._Calls(running=running)
    for record in records:
        calls.record(**record)
    return calls


def _answered(model: str, inputs: int, output: int, cached: int = 0, thoughts: int = 0) -> dict:
    return {"via": "passthrough", "model": model, "status": 200,
            "counts": {"input_tokens": inputs, "output_tokens": output, "cache_read_tokens": cached,
                       "reasoning_tokens": thoughts}}


def test_a_turns_usage_sums_its_calls_with_each_model_kept_apart():
    calls = _calls(_answered("gemini-2.5-flash", 9000, 40, cached=8000, thoughts=12),
                   _answered("gemini-2.5-flash-lite", 300, 5), _answered("gemini-2.5-flash", 9100, 20, cached=8900),
                   {"via": "passthrough", "model": "gemini-2.5-flash", "status": 503, "counts": None})

    usage = a2a_server._turn_usage(calls, 2, [])

    assert usage.to_dict() == {
        "tool_call_count": 2, "input_tokens": 18400, "output_tokens": 65, "total_tokens": 18465,
        "cache_read_tokens": 16900, "input_includes_cache": True, "models": [
            {"model": "gemini-2.5-flash", "calls": 2, "input_tokens": 18100, "output_tokens": 60,
             "cache_read_tokens": 16900, "input_includes_cache": True},
            {"model": "gemini-2.5-flash-lite", "calls": 1, "input_tokens": 300, "output_tokens": 5,
             "cache_read_tokens": 0, "input_includes_cache": True}],
        "provider_details": {"model_calls": 3, "reasoning_tokens": 12}}


def test_a_call_that_named_no_model_counts_as_the_configured_models():
    calls = _calls(_answered("gemini-2.5-flash-lite", 300, 5), {**_answered("unused", 900, 9), "model": None})

    usage = a2a_server._turn_usage(calls, 0, [], "gemini-2.5-pro")

    assert [(row.model, row.calls, row.input_tokens) for row in usage.models] == [
        ("gemini-2.5-flash-lite", 1, 300), ("gemini-2.5-pro", 1, 900)]
    assert a2a_server._turn_usage(calls, 0, []).models == ()


def test_calls_without_counts_or_still_running_are_unreported():
    partly = _calls(_answered("gemini-2.5-flash", 900, 4),
                    {"via": "passthrough", "model": "gemini-2.5-flash", "status": 200, "counts": None}, running=1)
    unknown = _calls({"via": "passthrough", "model": "gemini-2.5-flash", "status": 200, "counts": None})

    assert a2a_server._turn_usage(partly, 0, []).provider_details["unreported_calls"] == 2
    assert a2a_server._turn_usage(unknown, 0, []).to_dict() == {
        "tool_call_count": 0, "provider_details": {"model_calls": 1, "unreported_calls": 1}}
    assert a2a_server._turn_usage(_calls(), 0, []).to_dict() == {"tool_call_count": 0}


def test_a_cli_that_went_around_the_proxy_reports_its_own_stats_as_a_lower_bound():
    stats = {"type": "result", "stats": {"input_tokens": 9000, "output_tokens": 30, "cached": 8000, "tool_calls": 1}}

    usage = a2a_server._turn_usage(_calls(), 1, [{"type": "init"}, stats])

    assert usage.to_dict() == {
        "tool_call_count": 1, "input_tokens": 9000, "output_tokens": 30, "total_tokens": 9030,
        "cache_read_tokens": 8000, "input_includes_cache": True,
        "provider_details": {"source": "cli_stats", "unreported": True}}


def test_a_turn_reports_its_proxys_calls_and_records_each_in_its_trajectory(monkeypatch, agent, client):
    async def fake_run_gemini(**kwargs):
        (route,) = agent._routes.values()
        route.calls.record(**_answered("gemini-2.5-flash", 9000, 40, cached=8000))
        route.calls.record(**_answered("gemini-2.5-flash", 9100, 20, cached=8900))
        return gemini_bridge.GeminiResult(response="42", tool_call_count=1, session_id=None,
                                          trajectory=[{"type": "result", "stats": {"tool_calls": 1}}])

    monkeypatch.setattr(a2a_server, "run_gemini", fake_run_gemini)

    task = send(client, "hello")
    *_, data = terminal(task)
    trajectory = client.post(TRAJECTORY_PATH, json={"task_id": task["id"]}).json()["trajectory"]
    reported = Usage.from_dict(next(part["usage"] for part in data if "usage" in part))
    records = [event for event in trajectory if event["type"] == "agentenv.model_call"]

    assert (reported.input_tokens, reported.output_tokens, reported.cache_read_tokens) == (18100, 60, 16900)
    assert [record["usage"]["input_tokens"] for record in records] == [9000, 9100]
    assert reported.input_tokens == sum(record["usage"]["input_tokens"] for record in records)
