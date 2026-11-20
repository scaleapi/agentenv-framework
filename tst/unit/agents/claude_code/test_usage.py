"""A Claude Code turn reports the tokens its runs of the CLI used, as each run's `result` event counts them, with
``input_tokens`` counting the cache. The counts below were recorded on 2026-11-20 through Scale's LiteLLM proxy, with
the proxy's own usage for each model call beside them: a smoke turn on Claude Haiku 4.5 made two calls."""

import asyncio
import json

from agentenv_protocol.a2a_agent import Usage

from tst.unit.agents.claude_code.agent import a2a_server, claude_bridge
from tst.unit.agents.claude_code.support import FakeCLI
from tst.unit.agents.claude_code.test_claude_bridge_retry import _event, _write_session, scripted  # noqa: F401

# The two calls' usage as the proxy answered them.
CALLS = [
    {"input_tokens": 10, "cache_creation_input_tokens": 33811, "cache_read_input_tokens": 0, "output_tokens": 216,
     "cache_creation": {"ephemeral_5m_input_tokens": 33811, "ephemeral_1h_input_tokens": 0}},
    {"input_tokens": 8, "cache_creation_input_tokens": 357, "cache_read_input_tokens": 33811, "output_tokens": 47,
     "cache_creation": {"ephemeral_5m_input_tokens": 357, "ephemeral_1h_input_tokens": 0}},
]
# The run's `result` event's usage, as the CLI wrote it.
RUN = {"input_tokens": 18, "cache_creation_input_tokens": 34168, "cache_read_input_tokens": 33811,
       "output_tokens": 263, "output_tokens_details": {"thinking_tokens": 114},
       "server_tool_use": {"web_search_requests": 0, "web_fetch_requests": 0}, "service_tier": "standard",
       "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 34168}, "speed": "standard"}


def _result(**kw) -> claude_bridge.ClaudeResult:
    return claude_bridge.ClaudeResult(response="42", tool_call_count=2, session_id="s", **kw)


def test_a_run_reports_what_its_model_calls_used_with_the_cache_in_its_input():
    usage = a2a_server._usage(_result(run_usage=[RUN]))
    calls = [Usage.from_model_response(call) for call in CALLS]

    assert usage.to_dict() == {
        "tool_call_count": 2, "input_tokens": 67997, "output_tokens": 263, "total_tokens": 68260,
        "cache_read_tokens": 33811, "cache_write_tokens": 34168, "input_includes_cache": True,
        "provider_details": {"cache_creation": {"ephemeral_5m_input_tokens": 34168, "ephemeral_1h_input_tokens": 0},
                             "reasoning_tokens": 114}}
    for field in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
        assert getattr(usage, field) == sum(getattr(call, field) for call in calls)
    assert usage.cost_usd is None


def test_a_resumed_turn_adds_up_its_runs():
    """Each run's `result` counts that run alone; a resumed session's `total_cost_usd` and `modelUsage` don't, so the
    turn reports no cost of its own."""
    first, second = ({"input_tokens": call["input_tokens"], "output_tokens": call["output_tokens"],
                      "cache_read_input_tokens": call["cache_read_input_tokens"],
                      "cache_creation_input_tokens": call["cache_creation_input_tokens"],
                      "cache_creation": call["cache_creation"]} for call in CALLS)

    assert a2a_server._usage(_result(run_usage=[first, second])).to_dict() == \
        a2a_server._usage(_result(run_usage=[RUN])).to_dict() | {"provider_details": {
            "cache_creation": {"ephemeral_5m_input_tokens": 34168, "ephemeral_1h_input_tokens": 0}}}


def test_runs_cut_off_before_their_result_leave_their_calls_unreported():
    cut_off = a2a_server._usage(_result(unreported_calls=2))
    partly = a2a_server._usage(_result(run_usage=[RUN], unreported_calls=1))

    assert cut_off.to_dict() == {"tool_call_count": 2, "provider_details": {"unreported_calls": 2}}
    assert partly.input_tokens == 67997 and partly.provider_details["unreported_calls"] == 1


def test_a_run_without_a_result_counts_its_model_calls_by_message_id():
    """An assistant event is written as its message streams, so its usage is no account of the call; each message id
    is one call whose usage is unknown."""
    streamed = [{"type": "system", "subtype": "init"},
                *({"type": "assistant", "message": {"id": "m1", "usage": {"output_tokens": 3}}} for _ in range(3)),
                {"type": "user"}, {"type": "assistant", "message": {"id": "m2", "usage": {"output_tokens": 1}}}]

    assert claude_bridge._usage_of_run(streamed) == ([], 2)
    assert claude_bridge._usage_of_run([*streamed, {"type": "result", "usage": RUN}]) == ([RUN], 0)


def test_run_claude_keeps_each_runs_usage_across_a_resume(scripted):  # noqa: F811
    """A run cut short by an upstream failure still writes its `result` with what it used; the resumed run adds its
    own, and the turn keeps both."""
    _write_session(scripted)
    first, second = ({"input_tokens": call["input_tokens"], "output_tokens": call["output_tokens"],
                      "cache_read_input_tokens": call["cache_read_input_tokens"],
                      "cache_creation_input_tokens": call["cache_creation_input_tokens"]} for call in CALLS)
    scripted["queue"].append((
        (_event(type="system", subtype="init", session_id="sess-1")
         + _event(type="result", subtype="success", is_error=True, result="API Error: The operation timed out.",
                  session_id="sess-1", usage=first)).encode(), 1))
    scripted["queue"].append((_event(type="result", subtype="success", result="42", session_id="sess-1",
                                     usage=second).encode(), 0))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False))

    assert result.attempts == 2 and result.run_usage == [first, second] and result.unreported_calls == 0
    assert json.loads(json.dumps(a2a_server._usage(result).to_dict()))["input_tokens"] == 67997


def test_a_run_that_hangs_after_its_result_keeps_the_usage_it_reported(monkeypatch):
    """A CLI that wrote its `result` and then never closed its output times out, but what it reported still counts;
    a run that hung before its `result` counts its calls as unreported."""
    reported = (_event(type="system", subtype="init", session_id="s")
                + _event(type="result", subtype="success", result="42", session_id="s", usage=RUN)).encode()
    streamed = _event(type="assistant", message={"id": "m1", "content": [], "usage": {"output_tokens": 3}}).encode()
    results = []
    for printed in (reported, streamed):
        async def hung(*cmd, printed=printed, **kwargs):
            return FakeCLI(printed, 0, hang=True)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", hung)
        results.append(asyncio.run(claude_bridge._spawn_claude(prompt="p", session_id="s", resume=False,
                                                               timeout_s=0.2)))

    after, before = results
    assert after.error and (after.run_usage, after.unreported_calls) == ([RUN], 0)
    assert before.error and (before.run_usage, before.unreported_calls) == ([], 1)
