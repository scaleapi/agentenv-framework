"""A Codex turn reports the CLI's own token counts for that turn, with ``input_tokens`` counting the cache as OpenAI
counts it. The counts below were recorded on 2026-11-20 through Scale's LiteLLM proxy, with the proxy's own usage for
each Responses call beside them: a smoke turn on gpt-4.1-mini made three calls."""

import asyncio

from agentenv_protocol.a2a_agent import Usage

from tst.unit.agents.codex.agent import a2a_server, codex_bridge

# The three calls' usage as the proxy answered them.
CALLS = [
    {"input_tokens": 9270, "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
     "output_tokens": 22, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 9292},
    {"input_tokens": 9333, "input_tokens_details": {"cached_tokens": 9088, "cache_write_tokens": 0},
     "output_tokens": 16, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 9349},
    {"input_tokens": 9431, "input_tokens_details": {"cached_tokens": 9216, "cache_write_tokens": 0},
     "output_tokens": 3, "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 9434},
]
# The turn's counts, as the bridge worked them out from the CLI's turn.completed.
TURN = {"input_tokens": 28034, "cached_input_tokens": 18304, "cache_write_input_tokens": 0, "output_tokens": 41,
        "reasoning_output_tokens": 0}


def _result(**kw) -> codex_bridge.CodexResult:
    return codex_bridge.CodexResult(response="42", tool_call_count=2, session_id="t-1", **kw)


def test_a_turn_reports_what_its_model_calls_used_with_the_cache_in_its_input():
    usage = a2a_server._usage(_result(token_usage=TURN, model_requests=3))
    calls = [Usage.from_model_response(call) for call in CALLS]

    assert usage.to_dict() == {
        "tool_call_count": 2, "input_tokens": 28034, "output_tokens": 41, "total_tokens": 28075,
        "cache_read_tokens": 18304, "cache_write_tokens": 0, "input_includes_cache": True,
        "provider_details": {"model_calls": 3}}
    for field in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "total_tokens"):
        assert getattr(usage, field) == sum(getattr(call, field) for call in calls)


def test_reasoning_is_counted_in_the_output_and_kept_beside_it():
    usage = a2a_server._usage(_result(token_usage={**TURN, "output_tokens": 541, "reasoning_output_tokens": 500}))

    assert (usage.output_tokens, usage.provider_details["reasoning_tokens"]) == (541, 500)


def test_a_turn_without_counts_is_unknown_not_nothing():
    """A turn stopped before turn.completed, or whose counters went back, has no counts; its model requests are counted
    as calls whose usage is unknown."""
    assert a2a_server._usage(_result(model_requests=4)).to_dict() == {
        "tool_call_count": 2, "provider_details": {"model_calls": 4, "unreported_calls": 4}}
    assert a2a_server._usage(_result()).to_dict() == {"tool_call_count": 2}


def test_the_turns_usage_carries_the_counts_its_trajectory_records(codex, monkeypatch):
    async def fake_run_codex(**kwargs):
        return _result(token_usage=TURN, model_requests=3)

    monkeypatch.setattr(a2a_server, "run_codex", fake_run_codex)

    async def exercise():
        async with codex.client() as client:
            task = await codex.send(client, "go")
            trajectory = (await client.post("/ext/trajectory", json={"task_id": task["id"]})).json()["trajectory"]
            return task, trajectory

    task, trajectory = asyncio.run(exercise())
    *_, data = codex.terminal(task)
    (event,) = [event for event in trajectory if event["type"] == "agentenv.usage"]
    reported = next(part["usage"] for part in data if "usage" in part)

    assert (reported["input_tokens"], reported["cache_read_tokens"], reported["cache_write_tokens"],
            reported["output_tokens"]) == (event["usage"]["input_tokens"], event["usage"]["cached_input_tokens"],
                                           event["usage"]["cache_write_input_tokens"], event["usage"]["output_tokens"])
    assert reported["provider_details"]["model_calls"] == event["model_requests"]
