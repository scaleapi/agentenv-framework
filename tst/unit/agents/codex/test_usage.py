"""A Codex turn reports the CLI's own token counts for that turn, with ``input_tokens`` counting the cache as OpenAI
counts it. The counts below were recorded on 2026-11-20 through Scale's LiteLLM proxy, with the proxy's own usage for
each Responses call beside them: a smoke turn on gpt-4.1-mini made three calls."""

import asyncio
import json

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


def test_a_turn_the_bridge_stopped_is_a_lower_bound():
    """Recorded in codex-max-turns-stop: stopped at its tool-call limit, the turn's rollout held the first of its two
    calls (9,297 in, 28 out); the second, already answered by the proxy (9,369 in, 9,088 cached, 28 out), was in
    flight when the bridge stopped the CLI."""
    first = {"input_tokens": 9297, "cached_input_tokens": 0, "cache_write_input_tokens": 0, "output_tokens": 28}

    usage = a2a_server._usage(_result(token_usage=first, model_requests=1, unreported_calls=1))
    unknown = a2a_server._usage(_result(model_requests=1, unreported_calls=1))

    assert (usage.input_tokens, usage.output_tokens, usage.provider_details["unreported_calls"]) == (9297, 28, 1)
    assert unknown.to_dict() == {"tool_call_count": 2, "provider_details": {"model_calls": 1, "unreported_calls": 2}}


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


PARENT, CHILD = "01a1fb01-2718-7003-9a69-04d8a681cdf8", "01a1fb01-2fd1-7923-92f5-681485f2abf5"
# Recorded in codex-subagent on gpt-5.3-codex: the turn spawned one subagent; the proxy answered the parent's three calls
# and the subagent's one with these counts, in this order.
SUBAGENT_CALLS = [(PARENT, 11873, 0, 62), (CHILD, 11491, 2432, 11), (PARENT, 11957, 11776, 21), (PARENT, 12040, 11904, 5)]


def _write_rollout(sessions, thread, calls, parent=None):
    """A thread's rollout as Codex writes it: its session_meta, then the thread's running total after each call."""
    meta = {"id": thread, "source": "exec"}
    if parent:
        meta |= {"parent_thread_id": parent, "source": {"subagent": {"thread_spawn": {"parent_thread_id": parent}}}}
    records, total = [{"type": "session_meta", "payload": meta}], dict.fromkeys(
        ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens", "total_tokens"), 0)
    for inputs, cached, outputs in calls:
        total = {**total, "input_tokens": total["input_tokens"] + inputs,
                 "cached_input_tokens": total["cached_input_tokens"] + cached,
                 "output_tokens": total["output_tokens"] + outputs,
                 "total_tokens": total["total_tokens"] + inputs + outputs}
        records.append({"type": "event_msg", "payload": {"type": "token_count", "info": {"total_token_usage": total}}})
    path = sessions / "2026" / "11" / "20" / f"rollout-2026-11-20T08-39-01-{thread}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _turn(monkeypatch, tmp_path, rollouts, thread_id=None):
    """Runs a turn whose CLI leaves ``rollouts`` (thread, calls, parent) behind."""
    sessions = tmp_path / "sessions"
    monkeypatch.setattr(codex_bridge, "SESSIONS_DIR", sessions)

    async def cli(**kwargs):
        for thread, calls, parent in rollouts:
            _write_rollout(sessions, thread, calls, parent)
        return codex_bridge.CodexResult(response="42", tool_call_count=1, session_id=PARENT)

    monkeypatch.setattr(codex_bridge, "_run_codex_once", cli)
    return asyncio.run(codex_bridge.run_codex(prompt="go", thread_id=thread_id))


def test_a_turn_counts_its_subagents_calls(monkeypatch, tmp_path):
    """Codex counts a subagent's calls only in the subagent's own rollout, never in its parent's turn."""
    calls = {PARENT: [], CHILD: []}
    for thread, *counts in SUBAGENT_CALLS:
        calls[thread].append(counts)

    result = _turn(monkeypatch, tmp_path, [(PARENT, calls[PARENT], None), (CHILD, calls[CHILD], PARENT)])
    usage = a2a_server._usage(result)

    assert result.model_requests == len(SUBAGENT_CALLS)
    assert (usage.input_tokens, usage.cache_read_tokens, usage.output_tokens) == (
        sum(call[1] for call in SUBAGENT_CALLS), sum(call[2] for call in SUBAGENT_CALLS),
        sum(call[3] for call in SUBAGENT_CALLS))


def test_a_later_turn_counts_only_what_its_subagents_used_in_it(monkeypatch, tmp_path):
    """A subagent from an earlier turn that works again counts only its new calls; one it spawns counts in full, and
    another thread's subagent not at all."""
    sessions = tmp_path / "sessions"
    _write_rollout(sessions, PARENT, [(100, 0, 1)])
    _write_rollout(sessions, CHILD, [(50, 0, 1)], PARENT)
    _write_rollout(sessions, "other-child", [(70, 0, 1)], "other-thread")

    result = _turn(monkeypatch, tmp_path, [
        (PARENT, [(100, 0, 1), (200, 100, 2)], None),
        (CHILD, [(50, 0, 1), (60, 40, 3)], PARENT),
        ("grandchild", [(30, 0, 4)], CHILD),
        ("other-child", [(70, 0, 1), (80, 0, 5)], "other-thread"),
    ], thread_id=PARENT)

    assert result.model_requests == 3
    assert {key: result.token_usage[key] for key in ("input_tokens", "cached_input_tokens", "output_tokens")} == {
        "input_tokens": 200 + 60 + 30, "cached_input_tokens": 100 + 40, "output_tokens": 2 + 3 + 4}
