"""A wake whose turn reaches its time or tool-call limit, with continue_after_wake_limit on: only that wake ends, its
work stands, the agent is told at its next wake, the run goes on, and the record and the judge's files say so. Off,
and for any other failure, the run ends as an error, as it always has."""

from __future__ import annotations

import json

import pytest

from agent_env.a2a_agent.protocol import AgentUnreachableError
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.verifiers.always_on_staging import INDEX_FILE, stage_run, wake_file
from tst.unit.task_step.always_on_harness import Fail, Rig, Slow

pytestmark = pytest.mark.usefixtures("local_stores")

TIME_NOTE = ao.CUT_OFF_TEXT.format(limit="time limit of 1 second")
TOOL_CALL_NOTE = ao.CUT_OFF_TEXT.format(limit="limit of 150 tool calls")
MAX_TURNS_ERROR = "claude CLI exited with code 1: Reached maximum number of turns (150)"


@pytest.fixture(autouse=True)
def _quick(monkeypatch):
    monkeypatch.setattr(ao, "TURN_GRACE_SECONDS", 0)  # the step gives up on a turn at its time limit itself
    monkeypatch.setattr(ao, "CUT_OFF_GRACE_SECONDS", 0)  # unless a test waits for a late tool call


def _alert_later(rig: Rig) -> None:
    """Something that wakes the agent again after its first wake."""
    rig.triggers(rig.notify_at("live-oak", "2026-11-09T16:00:00Z", "Live Oak: we can come Friday", channel="Gmail"))


def _slow_first_turn(n, texts, rig):
    """The first turn sends one update, then keeps working past its time limit and would send another; later turns
    stay quiet."""
    if n == 1:
        rig.message("on it: comparing the three houses")
        return Slow(5, then=lambda rig: (rig.message("booked the blue house"), "done")[1])
    rig.quiet()
    return ""


def _tool_call_limit_first_turn(n, texts, rig):
    if n == 1:
        rig.message("found two houses under $900, checking the third")
        return Fail(MAX_TURNS_ERROR)
    rig.quiet()
    return ""


def _sent_texts(rig: Rig, ctx) -> list[str]:
    return [e["text"] for e in rig.events(ctx, "message_to_user")]


async def _run(rig: Rig, **settings):
    await rig.arm()
    _alert_later(rig)
    try:
        return await rig.run(rig.step(**settings))
    finally:
        await rig.close()


# ---- the time limit --------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_turn_past_its_time_limit_is_canceled_and_the_run_goes_on_to_the_next_event(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_slow_first_turn)
    ctx = await _run(rig, timeout_seconds=1, continue_after_wake_limit=True)

    assert rig.cancels == ["task-1"]  # the step canceled the agent's task
    work, _ = rig.agent_tasks["task-1"]
    assert work.cancelled()  # and its work stopped: its late message never lands
    assert _sent_texts(rig, ctx) == ["on it: comparing the three houses"]
    assert len(rig.sent) == 2 and "Live Oak: we can come Friday" in " ".join(rig.sent[1])
    record = rig.record(ctx)
    assert record["end_reason"] == "finished_nothing_pending" and record["error_type"] is None
    [response] = ctx.prompt_responses
    assert response.error_type is None  # graded, not skipped
    assert response.response == "[Mon Nov 9, 9:00 AM EST] on it: comparing the three houses"  # the wake's work stands
    [turn, _] = rig.events(ctx, "agent_turn_end")
    assert (turn["state"], turn["cut_off"], turn["messages_to_user"]) == ("timeout", "time_limit", 1)


@pytest.mark.asyncio
async def test_what_a_canceled_turn_still_does_in_the_grace_counts_in_the_cut_off_wake(monkeypatch, tmp_path):
    """The agent reads the turn canceled before its work has stopped, and a tool call already under way lands 3 s
    later: the step waits out its grace, the task's clock frozen, so that call is the cut-off wake's, at the time the
    wake was cut off, and the next wake is clean."""
    monkeypatch.setattr(ao, "CUT_OFF_GRACE_SECONDS", 4)

    def late_booking(rig):
        rig.app_write("bookings", "blue house held, $855 a night", tool="stayncook_request_booking")
        rig.message("held the blue house, $855 a night")

    def agent(n, texts, rig):
        if n == 1:
            rig.message("on it: comparing the three houses")
            return Slow(5, stopping_seconds=3, while_stopping=late_booking)
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, rate=60)
    rig.keep_changelog()
    ctx = await _run(rig, timeout_seconds=1, continue_after_wake_limit=True)

    [first, second] = rig.events(ctx, "agent_turn_end")
    assert (first["messages_to_user"], second["messages_to_user"]) == (2, 0)
    late = [e for e in rig.events(ctx, "message_to_user") if e["text"] == "held the blue house, $855 a night"]
    assert [(e["wake"], e["virtual_time"]) for e in late] == [(1, first["virtual_time"])]  # the clock stood still
    changes = [c for e in rig.events(ctx, "app_changes") for c in e["changes"]]
    assert [(c["table"], c["by"], c["wake"], c["phase"]) for c in changes] == [("public.bookings", "agent", 1, "turn")]
    assert rig.record(ctx)["cut_off_wakes"][0]["stopped"] == "canceled"

    # Staged for the judge with each change in full, the late change is the cut-off wake's, beside its cut-off.
    [change] = changes
    held = {"status": [None, "held"], "nightly_usd": [None, 855]}
    staged = stage_run(rig.timeline(ctx), {}, change_values={(change["env_id"], change["id"]): held})
    lines = [json.loads(line) for line in staged.files[wake_file(1)].splitlines()]
    assert [line["changed_fields"] for line in lines if line["event"] == "app_change"] == [held]
    assert [line["cut_off"] for line in lines if line["event"] == "turn_end"] == ["time_limit"]
    assert [line["limit"] for line in lines if line["event"] == "wake_cut_off"] == ["time_limit"]
    assert staged.record["app_change_values"] == {"found": 1, "missing": 0}


@pytest.mark.asyncio
async def test_the_next_wake_opens_by_telling_the_agent_its_last_turn_was_cut_off(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_slow_first_turn)
    ctx = await _run(rig, timeout_seconds=1, continue_after_wake_limit=True)

    assert rig.sent[1][0] == TIME_NOTE
    assert TIME_NOTE not in rig.sent[0]
    assert rig.parts[1][0]["metadata"]["agentenv"]["source"] == "cut_off_notice"
    [_, second] = rig.events(ctx, "wake")
    assert second["cut_off_notice"] == {"wake": 1, "limit": "time_limit"}
    assert second["woke_by"] == ["notification:Gmail"]  # the note says what happened; it isn't what woke the agent


@pytest.mark.asyncio
@pytest.mark.parametrize("timed_out", [
    Fail("claude CLI timed out after 1s", error_code="cli_error"),  # the CLI harnesses say so first in the error
    Fail("pi did not finish within 1s", error_code="pi.timeout"),  # the others give a .timeout code
    Fail("Pydantic AI did not finish within 1s", error_code="pydantic_ai.timeout"),
    Fail("OpenHands did not finish within 1s", error_code="openhands.timeout"),
], ids=["claude-code", "pi", "pydantic-ai", "openhands"])
async def test_a_turn_the_harness_ended_at_its_own_time_limit_is_cut_off_without_a_cancel(monkeypatch, tmp_path,
                                                                                         timed_out):
    def agent(n, texts, rig):
        if n == 1:
            rig.message("on it")
            return timed_out
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    ctx = await _run(rig, timeout_seconds=1, continue_after_wake_limit=True)

    assert rig.cancels == []  # the harness already stopped it
    assert rig.sent[1][0] == TIME_NOTE
    [cut] = rig.record(ctx)["cut_off_wakes"]
    assert (cut["limit"], cut["error"]) == ("time_limit", timed_out.message)


@pytest.mark.asyncio
async def test_a_turn_cut_off_at_its_time_limit_counts_in_the_usage_as_a_turn_with_none_reported(monkeypatch,
                                                                                                 tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_slow_first_turn)
    rig.solver_usage = {"input_tokens": 1200, "output_tokens": 300}  # what the agent's later turn reports
    ctx = await _run(rig, timeout_seconds=1, continue_after_wake_limit=True)

    [solver] = [e for e in ctx.prompt_responses[0].usage if e["role"] == "solver"]
    assert (solver["turns"], solver["input_tokens"]) == (2, 1200)
    assert solver["partial"] is True  # the cut-off turn's usage is unknown, so the sum is a lower bound


@pytest.mark.asyncio
async def test_a_turn_the_agent_will_not_stop_still_ends_the_run(monkeypatch, tmp_path):
    """Its tool calls could land in the next wake, so the run ends as an error, as it always has."""
    monkeypatch.setattr(ao, "STOP_TURN_SECONDS", 0.3)

    def agent(n, texts, rig):
        rig.message("on it")
        return Slow(5, then=lambda rig: "done", stops_on_cancel=False)

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    ctx = await _run(rig, timeout_seconds=1, continue_after_wake_limit=True)

    assert rig.cancels == ["task-1"] and len(rig.sent) == 1
    record = rig.record(ctx)
    assert (record["end_reason"], record["error_type"], record["cut_off_wakes"]) == (
        "agent_turn_timeout", "agent_turn_timeout", [])
    [turn] = rig.events(ctx, "agent_turn_end")
    assert turn["stop"] == "unconfirmed"


# ---- the tool-call limit ---------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_turn_at_its_tool_call_limit_ends_only_its_wake(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_tool_call_limit_first_turn)
    ctx = await _run(rig, max_turns=150, continue_after_wake_limit=True)

    assert rig.cancels == []  # the harness ended the turn at its limit
    assert rig.sent[1][0] == TOOL_CALL_NOTE
    assert _sent_texts(rig, ctx) == ["found two houses under $900, checking the third"]
    record = rig.record(ctx)
    assert record["end_reason"] == "finished_nothing_pending" and ctx.prompt_responses[0].error_type is None
    [turn, _] = rig.events(ctx, "agent_turn_end")
    assert (turn["state"], turn["cut_off"]) == ("failed", "tool_call_limit")


@pytest.mark.asyncio
async def test_a_nudge_turn_at_its_tool_call_limit_ends_only_its_wake(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n == 1:
            return "Booked the house."  # closing text, no message_user: a nudge follows
        if n == 2:
            return Fail(MAX_TURNS_ERROR)
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    ctx = await _run(rig, max_turns=150, continue_after_wake_limit=True)

    assert ao.NUDGE_TEXT in rig.sent[1] and rig.sent[2][0] == TOOL_CALL_NOTE
    [cut] = rig.record(ctx)["cut_off_wakes"]
    assert (cut["wake"], cut["turn"], cut["limit"], cut["told_in_wake"]) == (1, "nudge", "tool_call_limit", 2)
    [nudge] = rig.events(ctx, "nudge")
    assert nudge["cut_off"] == "tool_call_limit"


@pytest.mark.asyncio
async def test_a_cut_off_last_wake_ends_the_run_as_its_last_wake_would(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_tool_call_limit_first_turn,
              answers=[{"outcome": "reply", "message": "great, thanks"}])
    ctx = await _run(rig, max_turns=150, max_wakes=1, continue_after_wake_limit=True)

    record = rig.record(ctx)
    assert (record["end_reason"], record["error_type"]) == ("max_wakes", None)
    [cut] = record["cut_off_wakes"]
    assert "told_in_wake" not in cut  # no wake came after it
    [read] = rig.events(ctx, "user_sim_read")
    assert read["last_read"] is True  # she still reads what the wake sent


# ---- the record and what the judge reads -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_record_lists_each_cut_off_wake(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_slow_first_turn)
    ctx = await _run(rig, timeout_seconds=1, continue_after_wake_limit=True)

    record = rig.record(ctx)
    [cut] = record["cut_off_wakes"]
    assert {k: cut[k] for k in ("wake", "turn", "limit", "stopped", "told_in_wake")} == {
        "wake": 1, "turn": "wake", "limit": "time_limit", "stopped": "canceled", "told_in_wake": 2}
    assert cut["error"].startswith("agent turn timed out after 1s")
    assert record["counts"]["cut_off_wakes"] == 1 and record["settings"]["continue_after_wake_limit"] is True
    [event] = rig.events(ctx, "wake_cut_off")
    assert (event["wake"], event["limit"]) == (1, "time_limit")


@pytest.mark.asyncio
async def test_the_judge_s_files_and_prompt_show_the_cut_off(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_slow_first_turn)
    ctx = await _run(rig, timeout_seconds=1, continue_after_wake_limit=True)

    staged = stage_run(rig.timeline(ctx), {})
    index = staged.files[INDEX_FILE]
    assert ("- cut off: its turn reached its time limit; the wake ended there and the run went on "
            f"({wake_file(1)}:") in index
    assert f"- opened by telling the assistant that wake 1's turn was cut off at its time limit ({wake_file(2)}:1)" \
        in index
    first = [line for line in staged.files[wake_file(1)].splitlines()]
    assert any('"event": "turn_end"' in line and '"cut_off": "time_limit"' in line for line in first)
    assert any('"event": "wake_cut_off"' in line and '"limit": "time_limit"' in line for line in first)
    assert TIME_NOTE in staged.files[wake_file(2)]  # the wake's input, as the agent got it
    rendered = ao.render_for_judge(rig.record(ctx))
    assert "- Wake 1 · Mon Nov 9, 9:00 AM EST · woken by the user's request · sent the user 1 message · its turn " \
           "was cut off at its time limit and the run went on" in rendered


# ---- what still ends the run -----------------------------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    Fail("claude CLI exited with code 1: API Error: 500 Internal server error"),
    Fail("A2A protocol error: invalid task state", error_type="infra_error"),
])
async def test_a_turn_that_fails_for_another_reason_still_ends_the_run_as_an_error(monkeypatch, tmp_path, failure):
    def agent(n, texts, rig):
        rig.message("on it")
        return failure

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    ctx = await _run(rig, max_turns=150, continue_after_wake_limit=True)

    assert len(rig.sent) == 1
    record = rig.record(ctx)
    assert record["end_reason"] == "agent_turn_failed" and record["cut_off_wakes"] == []
    assert ctx.prompt_responses[0].error_type == (failure.error_type or "agent_turn_failed")


@pytest.mark.asyncio
async def test_a_turn_that_fails_for_another_reason_just_as_its_time_runs_out_still_ends_the_run(monkeypatch,
                                                                                                 tmp_path):
    """The step cuts the turn off at its time limit, but by the time the agent answers the cancel, the turn had
    already failed on its own, for a reason that is no limit: that failure is what ended it, and the run ends."""
    crash = Fail("claude CLI exited with code 1: API Error: 500 Internal server error")

    def agent(n, texts, rig):
        rig.message("on it")
        return Slow(5, ends_as=crash)

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    ctx = await _run(rig, timeout_seconds=1, continue_after_wake_limit=True)

    assert rig.cancels == ["task-1"] and len(rig.sent) == 1
    record = rig.record(ctx)
    assert (record["end_reason"], record["cut_off_wakes"]) == ("agent_turn_failed", [])
    assert ctx.prompt_responses[0].error_message == crash.message
    [turn] = rig.events(ctx, "agent_turn_end")
    assert turn["state"] == "failed" and "cut_off" not in turn


@pytest.mark.asyncio
async def test_a_turn_that_reached_its_tool_call_limit_just_as_its_time_runs_out_is_cut_off(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n == 1:
            rig.message("on it")
            return Slow(5, ends_as=Fail(MAX_TURNS_ERROR))
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    ctx = await _run(rig, timeout_seconds=1, max_turns=150, continue_after_wake_limit=True)

    assert rig.cancels == ["task-1"] and rig.sent[1][0] == TOOL_CALL_NOTE
    [cut] = rig.record(ctx)["cut_off_wakes"]
    assert (cut["limit"], cut["error"]) == ("tool_call_limit", MAX_TURNS_ERROR)
    assert rig.record(ctx)["end_reason"] == "finished_nothing_pending"


@pytest.mark.asyncio
async def test_an_agent_that_stops_answering_still_ends_the_run(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        raise AgentUnreachableError("The agent on sandbox sb stopped answering")

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    ctx = await _run(rig, continue_after_wake_limit=True)

    assert rig.cancels == [] and len(rig.sent) == 1
    assert rig.record(ctx)["end_reason"] == "agent_turn_timeout"


# ---- off: as before --------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_off_a_turn_past_its_time_limit_ends_the_run_as_before(monkeypatch, tmp_path):
    monkeypatch.setattr(ao, "CUT_OFF_GRACE_SECONDS", 30)  # off, the step never waits it out
    rig = Rig(monkeypatch, tmp_path, agent=_slow_first_turn)
    ctx = await _run(rig, timeout_seconds=1)

    assert rig.cancels == [] and len(rig.sent) == 1
    assert rig.record(ctx)["real_seconds"] < 10
    record = rig.record(ctx)
    assert (record["end_reason"], record["error_type"]) == ("agent_turn_timeout", "agent_turn_timeout")
    assert "cut_off_wakes" not in record and "cut_off_wakes" not in record["counts"]
    assert "continue_after_wake_limit" not in record["settings"]
    [turn] = rig.events(ctx, "agent_turn_end")
    assert turn["state"] == "timeout" and "cut_off" not in turn and "stop" not in turn
    assert not rig.events(ctx, "wake_cut_off")


@pytest.mark.asyncio
async def test_off_a_turn_at_its_tool_call_limit_ends_the_run_as_before(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_tool_call_limit_first_turn)
    ctx = await _run(rig, max_turns=150)

    assert len(rig.sent) == 1
    record = rig.record(ctx)
    assert record["end_reason"] == "agent_turn_failed" and ctx.prompt_responses[0].error_type == "agent_turn_failed"
    [turn] = rig.events(ctx, "agent_turn_end")
    assert "cut_off" not in turn


def test_the_setting_is_off_unless_set_and_survives_a_round_trip(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    off, on = rig.step(), rig.step(continue_after_wake_limit=True)
    assert off.continue_after_wake_limit is False and off.to_dict()["continue_after_wake_limit"] is False
    assert ao.AlwaysOnTaskStep.from_dict(on.to_dict()).continue_after_wake_limit is True
    # On, one wake's worst case also holds the time it takes to stop a turn cut off at its time limit.
    # The turn and its nudge each get the time to stop and the grace after.
    stopping = ao.STOP_TURN_SECONDS + ao.CUT_OFF_GRACE_SECONDS
    assert on.wake_reserve_seconds() - off.wake_reserve_seconds() == stopping * 2
