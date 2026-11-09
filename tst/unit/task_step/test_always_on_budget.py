"""The always_on step's real-time budget, against the hub's 8-hour job: once one more wake might end past the budget,
the agent gets its closing wake (the last-turn notice) and the run ends cleanly (time_budget) instead of being cut off
and retried. Settings whose worst case can't fit the job are refused."""

from __future__ import annotations

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

NOTICE = ao.FINAL_REMINDER.format(first="Linda")


def _step(**settings) -> ao.AlwaysOnTaskStep:
    return ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt="hi", prompt_id="p", agent_name="a", **settings)


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
    step = rig.step(time_budget_seconds=3 * 3600)  # reserve 3060 s: closes once 7740 s are spent
    ctx = await rig.run(step)

    record = rig.record(ctx)
    assert step.wake_reserve_seconds() == 3060 and record["settings"]["time_budget_seconds"] == 10800
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


@pytest.mark.parametrize("timeout,hours", [(600, 6.98), (1800, 6.32), (3600, 5.32)])
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
