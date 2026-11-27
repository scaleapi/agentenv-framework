"""A Claude Code turn reports what all of its model calls used, its subagents' included, with ``input_tokens`` counting
the cache. The figures below were recorded on 2026-11-20 through Scale's LiteLLM proxy, with the proxy's own usage for
every call beside the CLI's events: a three-turn chat on Claude Haiku 4.5 whose first turn handed work to a subagent.

What the recording showed: a `result` event's ``usage`` counts the main agent's calls only, and a run that hands work to
a subagent writes two `result` events; ``modelUsage`` counts every call, the subagent's too, and a resumed session's
``modelUsage`` starts from the session's tally so far, which the CLI restores, so a turn's own tally is the difference."""

import asyncio
from dataclasses import replace

import pytest
from agentenv_protocol.a2a_agent import ModelUsage, Usage

from tst.unit.agents.claude_code.agent import a2a_server, claude_bridge
from tst.unit.agents.claude_code.support import FakeCLI
from tst.unit.agents.claude_code.test_claude_bridge_retry import _event, _write_session, scripted  # noqa: F401

HAIKU = "claude-haiku-4-5-20251001"


def _call(input_tokens, read, write, output):
    return {"input_tokens": input_tokens, "cache_read_input_tokens": read, "cache_creation_input_tokens": write,
            "output_tokens": output, "cache_creation": {"ephemeral_5m_input_tokens": write, "ephemeral_1h_input_tokens": 0}}


# The turns' calls as the proxy answered them, the subagent's two marked.
TURN_1 = [_call(10, 30799, 3031, 215), _call(10, 0, 19938, 116), _call(8, 33830, 545, 74), _call(8, 19938, 620, 32),
          _call(10, 34375, 486, 77)]
SUBAGENT = [TURN_1[1], TURN_1[3]]
TURN_2 = [_call(10, 34861, 161, 156)]
# The first turn's two `result` events: the main agent's calls before and after the subagent answered.
RESULTS_1 = [{"input_tokens": 18, "cache_read_input_tokens": 64629, "cache_creation_input_tokens": 3576,
              "output_tokens": 289}, {"input_tokens": 10, "cache_read_input_tokens": 34375,
                                      "cache_creation_input_tokens": 486, "output_tokens": 77}]
RESULTS_2 = [{"input_tokens": 10, "cache_read_input_tokens": 34861, "cache_creation_input_tokens": 161,
              "output_tokens": 156}]


def _tally(input_tokens, output, read, write, cost):
    return {HAIKU: {"inputTokens": input_tokens, "outputTokens": output, "cacheReadInputTokens": read,
                    "cacheCreationInputTokens": write, "cacheCreationEphemeral5mInputTokens": write,
                    "cacheCreationEphemeral1hInputTokens": 0, "costUSD": cost}}


# The CLI's `modelUsage` at each turn's end: the session's tally so far.
TALLY_1 = _tally(46, 514, 118942, 24620, 0.0452852)
TALLY_2 = _tally(56, 670, 153803, 24781, 0.04976255)


def _truth(calls):
    usages = [Usage.from_model_response(call) for call in calls]
    return {field: sum(getattr(u, field) for u in usages)
            for field in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")}


def _counts(usage: Usage):
    return {field: getattr(usage, field) for field in ("input_tokens", "output_tokens", "cache_read_tokens",
                                                        "cache_write_tokens")}


def _result(**kw) -> claude_bridge.ClaudeResult:
    return claude_bridge.ClaudeResult(response="42", tool_call_count=1, session_id="s", **kw)


def test_a_new_sessions_turn_counts_every_call_its_subagents_included():
    usage = a2a_server._usage(_result(run_usage=RESULTS_1, model_usage=TALLY_1, session_base={}, subagent_calls=2))

    assert _counts(usage) == _truth(TURN_1)
    assert usage.input_includes_cache is True and usage.cost_usd == 0.0452852
    assert "unreported_calls" not in usage.provider_details
    assert usage.models == (ModelUsage(model=HAIKU, **_truth(TURN_1), input_includes_cache=True, cost_usd=0.0452852),)
    assert "models" not in usage.provider_details


def test_a_turn_whose_subagent_ran_on_another_model_gives_each_models_counts_and_cost():
    sonnet = "claude-sonnet-4-6"
    tally = {sonnet: {"inputTokens": 20, "outputTokens": 300, "cacheReadInputTokens": 60000,
                      "cacheCreationInputTokens": 4000, "costUSD": 0.05},
             **_tally(10, 90, 20000, 2000, 0.004)}

    usage = a2a_server._usage(_result(run_usage=[], model_usage=tally, session_base={}))

    assert usage.models == (
        ModelUsage(model=sonnet, input_tokens=64020, output_tokens=300, cache_read_tokens=60000,
                   cache_write_tokens=4000, input_includes_cache=True, cost_usd=0.05),
        ModelUsage(model=HAIKU, input_tokens=22010, output_tokens=90, cache_read_tokens=20000,
                   cache_write_tokens=2000, input_includes_cache=True, cost_usd=0.004))
    assert usage.cost_usd == pytest.approx(0.054) and usage.input_tokens == 64020 + 22010


def test_a_resumed_turn_counts_its_own_calls_not_the_sessions():
    usage = a2a_server._usage(_result(run_usage=RESULTS_2, model_usage=TALLY_2, session_base=TALLY_1))

    assert _counts(usage) == _truth(TURN_2)
    assert abs(usage.cost_usd - (0.04976255 - 0.0452852)) < 1e-12


def test_without_the_sessions_tally_the_turn_is_its_main_agents_calls_and_a_marked_lower_bound():
    """A resumed session this process hasn't run before: its tally so far is unknown, so the turn counts its `result`
    events, which leave the subagents out, and counts the subagents' calls as unreported."""
    usage = a2a_server._usage(_result(run_usage=RESULTS_1, model_usage=TALLY_1, session_base=None, subagent_calls=2))
    main = [call for call in TURN_1 if call not in SUBAGENT]

    assert _counts(usage) == _truth(main)
    assert usage.cost_usd is None and usage.provider_details["unreported_calls"] == 2


def test_a_tally_short_of_the_main_agents_calls_isnt_used():
    """A base that isn't what the CLI restored would give a difference smaller than the main agent's own calls."""
    usage = a2a_server._usage(_result(run_usage=RESULTS_2, model_usage=TALLY_2, session_base=TALLY_2))

    assert _counts(usage) == _truth(TURN_2) and usage.cost_usd is None


def test_runs_cut_off_before_their_result_leave_their_calls_unreported():
    cut_off = a2a_server._usage(_result(unreported_calls=2))
    partly = a2a_server._usage(_result(run_usage=RESULTS_2, unreported_calls=1))

    assert cut_off.to_dict() == {"tool_call_count": 1, "provider_details": {"unreported_calls": 2}}
    assert _counts(partly) == _truth(TURN_2) and partly.provider_details["unreported_calls"] == 1


def test_a_run_counts_every_result_its_subagents_calls_and_its_calls_without_one():
    """An assistant event is written as its message streams, so its usage is no account of the call; calls are counted
    by message id, a subagent's by its parent tool call."""
    def assistant(message_id, parent=None):
        return {"type": "assistant", "parent_tool_use_id": parent,
                "message": {"id": message_id, "usage": {"output_tokens": 1}}}

    streamed = [{"type": "system", "subtype": "init"}, assistant("m1"), assistant("m1"), assistant("s1", "t1"),
                assistant("s2", "t1"), assistant("m2")]
    ran = [*streamed[:3], {"type": "result", "usage": RESULTS_1[0], "modelUsage": TALLY_2}, *streamed[3:],
           {"type": "result", "usage": RESULTS_1[1], "modelUsage": TALLY_1}]

    assert claude_bridge._usage_of_run(streamed) == claude_bridge.RunUsage([], None, 0, 4)
    assert claude_bridge._usage_of_run(ran) == claude_bridge.RunUsage(RESULTS_1, TALLY_1, 2, 0)


def test_calls_after_a_runs_last_result_are_unreported():
    """A run that wrote a `result` as it handed work to a subagent, kept calling the model and was then killed: no
    `result` counts the calls after it, so the turn is a lower bound (recorded in claude-code-host-timeout)."""
    def assistant(message_id, parent=None):
        return {"type": "assistant", "parent_tool_use_id": parent, "message": {"id": message_id}}

    killed = [assistant("m1"), {"type": "result", "usage": RESULTS_1[0], "modelUsage": TALLY_1}, assistant("s1", "t1"),
              assistant("s1", "t1"), assistant("m2")]
    used = claude_bridge._usage_of_run(killed)
    usage = a2a_server._usage(_result(run_usage=used.results, model_usage=used.model_usage, session_base={},
                                      subagent_calls=used.subagent_calls, unreported_calls=used.unreported_calls))

    assert used == claude_bridge.RunUsage([RESULTS_1[0]], TALLY_1, 0, 2)
    assert usage.provider_details["unreported_calls"] == 2


def test_run_claude_starts_a_resumed_turn_from_the_sessions_last_tally(scripted, monkeypatch):  # noqa: F811
    """A new session starts from nothing; the session's next turn starts from the tally its last turn ended with."""
    monkeypatch.setattr(claude_bridge, "_SESSION_TALLIES", {})
    _write_session(scripted)
    for session, results, tally in (("sess-1", RESULTS_1, TALLY_1), ("sess-1", RESULTS_2, TALLY_2),
                                    ("sess-2", RESULTS_2, TALLY_2)):
        scripted["queue"].append(("".join(_event(type="result", subtype="success", result="42", session_id=session,
                                                 usage=usage, modelUsage=tally) for usage in results).encode(), 0))

    first = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False, web_tools=True))
    second = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=True, web_tools=True))
    elsewhere = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-2", resume=True, web_tools=True))

    assert (first.run_usage, first.model_usage, first.session_base) == (RESULTS_1, TALLY_1, {})
    # Its transcript has model calls but no cost-state, so the base is the process's own last tally.
    assert (second.run_usage, second.model_usage, second.session_base) == (RESULTS_2, TALLY_2, TALLY_1)
    assert _counts(a2a_server._usage(second)) == _truth(TURN_2)
    assert elsewhere.session_base == {}


def _transcript(cli, *records, session_id="sess-1"):
    (cli["session_dir"] / f"{session_id}.jsonl").write_text("".join(_event(**record) for record in records))


def test_the_base_is_the_last_cost_state_the_transcript_holds(scripted):  # noqa: F811
    """The CLI writes a `cost-state` record as a run exits and restores the last one on a resume."""
    saved = claude_bridge.saved_session_tally
    _transcript(scripted, {"type": "assistant"}, {"type": "cost-state", "sessionId": "sess-1", "modelUsage": TALLY_1},
                {"type": "assistant"}, {"type": "cost-state", "sessionId": "sess-1", "modelUsage": TALLY_2})
    _transcript(scripted, {"type": "assistant"}, {"type": "user"}, session_id="sess-2")
    _transcript(scripted, {"type": "user"}, session_id="sess-3")

    assert saved("sess-1") == TALLY_2
    assert saved("sess-2") is None
    assert saved("sess-3") == {} and saved("sess-4") == {}


def test_a_run_killed_before_it_exited_isnt_in_the_next_turns_base(scripted, monkeypatch):  # noqa: F811
    """A run that wrote its `result` but was killed before exiting wrote no cost-state, so the CLI restores the one
    before it; the base is that one, not the killed run's tally the process saw."""
    monkeypatch.setattr(claude_bridge, "_SESSION_TALLIES", {"sess-1": TALLY_2})
    _transcript(scripted, {"type": "assistant"}, {"type": "cost-state", "sessionId": "sess-1", "modelUsage": TALLY_1})
    scripted["queue"].append((_event(type="result", subtype="success", result="42", session_id="sess-1",
                                     usage=RESULTS_2[0], modelUsage=TALLY_2).encode(), 0))

    resumed = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=True, web_tools=True))

    assert resumed.session_base == TALLY_1


def test_run_claude_keeps_each_runs_usage_across_a_resume(scripted):  # noqa: F811
    """A run cut short by an upstream failure still writes its `result` with what it used; the resumed run adds its
    own, and the turn keeps both, its tally being the resumed run's."""
    _write_session(scripted)
    scripted["queue"].append((
        (_event(type="system", subtype="init", session_id="sess-1")
         + _event(type="result", subtype="success", is_error=True, result="API Error: The operation timed out.",
                  session_id="sess-1", usage=RESULTS_1[0])).encode(), 1))
    scripted["queue"].append((_event(type="result", subtype="success", result="42", session_id="sess-1",
                                     usage=RESULTS_1[1], modelUsage=TALLY_1).encode(), 0))

    result = asyncio.run(claude_bridge.run_claude(prompt="p", session_id="sess-1", resume=False, web_tools=True))

    assert result.attempts == 2 and result.run_usage == RESULTS_1 and result.model_usage == TALLY_1


def test_a_run_that_hangs_after_its_result_keeps_the_usage_it_reported(monkeypatch):
    """A CLI that wrote its `result` and then never closed its output times out, but what it reported still counts;
    a run that hung before its `result` counts its calls as unreported."""
    reported = (_event(type="system", subtype="init", session_id="s")
                + _event(type="result", subtype="success", result="42", session_id="s", usage=RESULTS_2[0],
                         modelUsage=TALLY_2)).encode()
    streamed = _event(type="assistant", message={"id": "m1", "content": [], "usage": {"output_tokens": 3}}).encode()
    results = []
    for printed in (reported, streamed):
        async def hung(*cmd, printed=printed, **kwargs):
            return FakeCLI(printed, 0, hang=True)

        monkeypatch.setattr(asyncio, "create_subprocess_exec", hung)
        results.append(asyncio.run(claude_bridge._spawn_claude(prompt="p", session_id="s", resume=False,
                                                               timeout_s=0.2, web_tools=True)))

    after, before = results
    assert after.error and (after.run_usage, after.model_usage, after.unreported_calls) == (RESULTS_2, TALLY_2, 0)
    assert before.error and (before.run_usage, before.unreported_calls) == ([], 1)


def test_the_atif_turn_record_counts_the_turn_apart_from_its_cache_and_not_at_all_when_its_a_lower_bound():
    """What the atif/1 adapter gives a compaction's step from: the turn's counts as a session record counts a call's."""
    counted = Usage(input_tokens=1000, output_tokens=50, cache_read_tokens=600, cache_write_tokens=100,
                    input_includes_cache=True, cost_usd=0.2, models=(ModelUsage(model=HAIKU),))
    assert a2a_server._turn_record(counted) == {
        "type": "turn", "cost_usd": 0.2, "input_tokens": 300, "output_tokens": 50, "cache_read_tokens": 600,
        "cache_write_tokens": 100, "model": HAIKU}
    two_models = replace(counted, models=(ModelUsage(model=HAIKU), ModelUsage(model="claude-sonnet-5-5")))
    assert "model" not in a2a_server._turn_record(two_models)
    lower_bound = replace(counted, provider_details={"unreported_calls": 2})
    assert a2a_server._turn_record(lower_bound) == {"type": "turn", "cost_usd": 0.2}

