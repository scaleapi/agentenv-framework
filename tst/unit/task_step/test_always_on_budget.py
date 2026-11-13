"""The always_on step's real-time budget, against the hub's 8-hour job: once one more wake might end past the budget,
the agent gets its closing wake (the last-turn notice) and the run ends cleanly (time_budget) instead of being cut off
and retried. Settings whose worst case can't fit the job are refused."""

from __future__ import annotations

import asyncio

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

NOTICE = ao.FINAL_REMINDER.format(first="Linda")


def _step(**settings) -> ao.AlwaysOnTaskStep:
    return ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt_id="p", agent_name="a", **settings)


@pytest.mark.asyncio
async def test_once_another_wake_might_end_past_the_budget_the_agent_gets_its_closing_wake(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if NOTICE in texts:
            rig.message("RESULT: wrapping up early, La Ola still open")
            return ""
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, answers=[{"outcome": "reply", "message": "ok thanks"}])
    monkeypatch.setattr(ao._Run, "spent", lambda run: 3600.0 * len(rig.sent))  # each turn takes an hour
    await rig.arm()
    for day in range(10, 20):
        rig.triggers(rig.notify_at(f"alert-{day}", f"2026-11-{day}T14:00:00Z", "Sam: news?", channel="noor 30"))
    step = rig.step(time_budget_seconds=12120)  # no contacts, so a reserve of 3780 s: closes once 8340 s are spent
    ctx = await rig.run(step)

    record = rig.record(ctx)
    assert record["settings"]["contacts"]["longest_timeout_seconds"] == 0
    assert record["settings"]["wake_reserve_seconds"] == step.wake_reserve_seconds(0) == 3780
    assert record["settings"]["time_budget_seconds"] == 12120
    wakes = rig.events(ctx, "wake")
    assert len(wakes) == 4 and wakes[-1]["woke_by"] == ["step:final"] and wakes[-1]["last"] is True
    assert wakes[-1]["virtual_time"] == "2026-11-11T14:00:00Z"  # where the run was; the clock doesn't jump
    assert ao.FINAL_DEFAULT_TEXT.format(first="Linda") in rig.sent[-1] and NOTICE in rig.sent[-1]
    [reached] = rig.events(ctx, "time_budget_reached")
    assert reached["real_seconds"] == 10800
    assert record["end_reason"] == "time_budget" and record["error_type"] is None
    assert rig.events(ctx, "user_sim_read")[-1]["last_read"] is True
    assert rig.chat(ctx)[-1]["texts"][-1].startswith("[End of run · Wed Nov 11, 2:00 PM UTC] The run was cut off by "
                                                     "its real-time budget.")


@pytest.mark.parametrize("timeout,hours", [(600, 6.72), (1800, 6.05), (3600, 5.05)])
def test_the_default_budget_is_7_h_lowered_so_the_closing_wake_fits_the_8_h_job(timeout, hours):
    step = _step(timeout_seconds=timeout)
    budget = step.effective_time_budget()
    assert round(budget / 3600, 2) == hours
    assert budget + step.wake_reserve_seconds() + ao.JOB_HEADROOM_SECONDS <= ao.JOB_LIMIT_SECONDS


def test_a_budget_whose_closing_can_t_fit_the_job_is_refused():
    with pytest.raises(ValueError, match="time_budget_seconds=27000 can't fit"):
        _step(time_budget_seconds=7.5 * 3600)
    with pytest.raises(ValueError, match="time_budget_seconds=0 can't fit"):
        _step(time_budget_seconds=0)
    assert _step(time_budget_seconds=6 * 3600).effective_time_budget() == 6 * 3600


def test_a_wake_whose_worst_case_leaves_no_room_for_a_run_is_refused():
    with pytest.raises(ValueError, match="no run fits the hub's 8 h job"):
        _step(timeout_seconds=4 * 3600)


def test_the_budget_survives_a_round_trip():
    again = ao.AlwaysOnTaskStep.from_dict(_step(time_budget_seconds=5 * 3600).to_dict())
    assert again.time_budget_seconds == 5 * 3600
    assert ao.AlwaysOnTaskStep.from_dict(_step().to_dict()).time_budget_seconds is None


# ---- the reserve is a real bound -------------------------------------------------------------------------------------

@pytest.mark.parametrize("contact", [ao.CONTACT_TIMEOUT_DEFAULT_SECONDS, 0, 900], ids=["default", "none", "900 s"])
@pytest.mark.parametrize("settings", [{}, {"nudge": False}, {"timeout_seconds": 1800, "user_timeout_seconds": 900},
                                      {"settle_timeout_seconds": 600}, {"time_budget_seconds": 5 * 3600}])
def test_the_reserve_adds_up_every_capped_part_and_the_budget_plus_one_turn_fits_the_job(settings, contact):
    step = _step(**settings)
    settle = contact + ao.CONTACT_GRACE_SECONDS + step.settle_timeout_seconds
    turn = step.timeout_seconds + ao.TURN_GRACE_SECONDS + ao.TRAJECTORY_FETCH_SECONDS
    read = ao.READ_ASKS * (step.user_timeout_seconds + ao.TURN_GRACE_SECONDS + ao.TRAJECTORY_FETCH_SECONDS)
    wake = turn * (2 if step.nudge else 1) + settle + ao.GATEWAY_ALLOWANCE_SECONDS
    assert step.settle_cap_seconds(contact) == settle and step.read_cap_seconds() == read
    # a jump's settle, then a wake (its turns, its settle, the step's calls), then a read and its calls
    reserve = step.wake_reserve_seconds(contact)
    assert reserve == settle + wake + read + ao.GATEWAY_ALLOWANCE_SECONDS
    assert step.effective_time_budget(contact) + reserve + ao.JOB_HEADROOM_SECONDS <= ao.JOB_LIMIT_SECONDS
    if contact == ao.CONTACT_TIMEOUT_DEFAULT_SECONDS:  # what the task is checked with when it loads
        assert (step.settle_cap_seconds(), step.wake_reserve_seconds()) == (settle, reserve)


def _estimate(rig, n):
    rig.alert(f"Live Oak: estimate {n}", channel="Gmail · Live Oak Arbor")


def _ask_live_oak(rig, contact: dict) -> None:
    rig.triggers({"id": "estimate", "when": {"type": "time", "at": "2026-11-09T15:00:00Z"},
                  "actions": [{"type": "nl", "executor": "liveoak", "instruction": "send your estimate"}]},
                 executors={"liveoak": contact})


@pytest.mark.asyncio
async def test_a_settle_is_capped_by_the_limit_the_contact_was_registered_with(monkeypatch, tmp_path):
    """The contact was registered with a 0.3 s limit and overruns it: the settle stops at that limit, its grace and
    settle_timeout_seconds (0.5 s), not at the gateway's default."""
    monkeypatch.setattr(ao, "CONTACT_GRACE_SECONDS", 0.1)
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    _ask_live_oak(rig, rig.contact("liveoak", seconds=5, reply=_estimate, timeout_seconds=0.3,
                                   keeps_to_its_limit=False))
    step = rig.step(settle_timeout_seconds=0.1)
    try:
        ctx = await rig.run(step)
    finally:
        await rig.close()

    assert rig.record(ctx)["settings"]["contacts"]["longest_timeout_seconds"] == pytest.approx(0.3)
    assert step.settle_cap_seconds(0.3) == pytest.approx(0.5)
    [jump] = [j for j in rig.events(ctx, "clock_jump") if j["to"] == "2026-11-09T15:00:00Z"]
    assert not jump["settled"] and jump["settle"]["seconds"] < 1.5


@pytest.mark.asyncio
async def test_the_run_s_budget_is_worked_out_again_from_its_contacts_limits(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    _ask_live_oak(rig, rig.contact("liveoak", seconds=0, reply=_estimate, timeout_seconds=1800))
    step = rig.step(timeout_seconds=1800)
    ctx = await rig.run(step)

    settings = rig.record(ctx)["settings"]
    assert settings["contacts"]["longest_timeout_seconds"] == 1800
    assert settings["wake_reserve_seconds"] == step.wake_reserve_seconds(1800)
    assert settings["time_budget_seconds"] == step.effective_time_budget(1800) < step.effective_time_budget()
    assert settings["time_budget_seconds"] + settings["wake_reserve_seconds"] + ao.JOB_HEADROOM_SECONDS \
        <= ao.JOB_LIMIT_SECONDS


@pytest.mark.asyncio
async def test_a_contact_whose_limit_leaves_no_room_for_a_run_is_refused_when_the_run_starts(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    _ask_live_oak(rig, rig.contact("liveoak", seconds=0, reply=_estimate, timeout_seconds=2 * 3600))
    with pytest.raises(ValueError, match="no run fits the hub's 8 h job"):
        await rig.run(rig.step())
    assert rig.sent == []  # before the agent's first wake


@pytest.mark.asyncio
async def test_a_gateway_that_doesn_t_report_its_contacts_limits_gets_the_gateway_s_default(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    engine = rig.gw._trigger_engine
    real_state = engine.state

    def without_limits():
        state = real_state()
        return {**state, "config": {k: v for k, v in state["config"].items() if k != "executor_timeouts"}}

    monkeypatch.setattr(engine, "state", without_limits)
    ctx = await rig.run(rig.step())

    contacts = rig.record(ctx)["settings"]["contacts"]
    assert contacts["longest_timeout_seconds"] == ao.CONTACT_TIMEOUT_DEFAULT_SECONDS
    assert contacts["envs"][0]["contact_timeouts"] is None


@pytest.mark.asyncio
async def test_an_event_due_that_won_t_go_gets_the_settle_timeout_not_forever(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    engine = rig.gw._trigger_engine
    real_state = engine.state
    monkeypatch.setattr(engine, "state", lambda: {**real_state(), "next_event_at": rig.now()})  # due, never going
    ctx = await asyncio.wait_for(rig.run(rig.step(settle_timeout_seconds=0.2)), 10)

    [timeout] = rig.events(ctx, "settle_timeout")
    [env] = timeout["still_at_work"]
    assert env["due_not_going"] == "2026-11-09T14:00:00Z" and timeout["settle"]["seconds"] < 1.5


@pytest.mark.asyncio
async def test_an_agent_turn_is_cut_off_at_its_limit_and_grace_whatever_send_and_wait_does(monkeypatch, tmp_path):
    monkeypatch.setattr(ao, "TURN_GRACE_SECONDS", 0.2)
    rig = Rig(monkeypatch, tmp_path)

    async def retries_forever(url, parts, **kw):
        await asyncio.sleep(60)

    monkeypatch.setattr(ao, "send_and_wait", retries_forever)
    await rig.arm()
    ctx = await asyncio.wait_for(rig.run(rig.step(timeout_seconds=1)), 10)

    record = rig.record(ctx)
    assert (record["end_reason"], record["error_type"]) == ("agent_turn_timeout", "agent_turn_timeout")
    assert rig.events(ctx, "agent_turn_end")[0]["real_seconds"] < 2


@pytest.mark.asyncio
async def test_a_read_is_cut_off_at_its_cap_as_a_user_sim_error(monkeypatch, tmp_path):
    monkeypatch.setattr(ao, "TURN_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(ao, "TRAJECTORY_FETCH_SECONDS", 0.1)
    monkeypatch.setattr(ao, "READ_ASKS", 1)

    class Stalls:
        @staticmethod
        def setup(version, person):
            return {"system_prompt": "you are Linda", "output_format": None}

        @staticmethod
        async def answer(version, person, history, read, ask):
            await asyncio.sleep(60)

    def agent(n, texts, rig):
        rig.message("blue or green?")
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=Stalls)
    await rig.arm()
    step = rig.step(user_timeout_seconds=1)
    ctx = await asyncio.wait_for(rig.run(step), 10)

    assert step.read_cap_seconds() == pytest.approx(1.2)
    [read] = rig.events(ctx, "user_sim_read")
    assert read["user_sim_error"] == "the read ran past its cap (1s)" and read["real_seconds"] < 2
    assert rig.record(ctx)["end_reason"] == "user_sim_error"


@pytest.mark.asyncio
async def test_a_turn_of_the_loop_past_the_reserve_ends_the_run_on_a_step_error(monkeypatch, tmp_path):
    """Every part is capped on its own; only calls hanging past their allowance reach this last cut-off."""
    monkeypatch.setattr(ao.AlwaysOnTaskStep, "wake_reserve_seconds", lambda step, contact_seconds=0: 0.5)
    rig = Rig(monkeypatch, tmp_path)

    real_take = ao._Run._take_messages
    hung: list[int] = []

    async def hangs_once(run):
        if run.wakes and not hung:  # the opening wake's take, not the one before the run
            hung.append(1)
            await asyncio.sleep(60)
        return await real_take(run)

    monkeypatch.setattr(ao._Run, "_take_messages", hangs_once)
    await rig.arm()
    ctx = rig.context()
    with pytest.raises(RuntimeError, match="the opening wake ran past its worst case"):
        await asyncio.wait_for(rig.run(rig.step(), ctx), 10)
    assert ctx.metadata[ao.ALWAYS_ON_KEY]["week"]["end_reason"] == "step_error"
