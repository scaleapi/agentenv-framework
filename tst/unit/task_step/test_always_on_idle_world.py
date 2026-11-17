"""A world that only repeats: a recurring trigger that never wakes the agent (an hourly alert to the user's phone, say)
must not move the clock forever. With no end_time the run ends once that is all the world has ahead; a known event ahead
keeps it going; with an end_time it goes on to it, and the timeline stays small."""

from __future__ import annotations

import asyncio

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

NEWS = {"id": "news", "when": {"type": "time", "every": "PT1H"},
        "actions": [{"type": "notify", "channel": "News", "text": "headline", "to": ao.USER_PHONE}]}


async def _run(rig, step, ctx=None, seconds=30):
    return await asyncio.wait_for(rig.run(step, ctx), seconds)  # the old loop never ended


@pytest.mark.asyncio
async def test_with_no_end_time_a_world_that_only_repeats_ends_the_run(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    rig.triggers(NEWS)
    ctx = await _run(rig, rig.step())

    record = rig.record(ctx)
    assert record["end_reason"] == "finished_nothing_pending"
    assert record["counts"]["clock_jumps"] == ao.IDLE_JUMPS_MAX and record["counts"]["wakes"] == 1
    [idle] = rig.events(ctx, "world_only_repeats")
    assert (idle["jumps"], idle["since"], idle["triggers"]) == (ao.IDLE_JUMPS_MAX, "2026-11-09T14:00:00Z", ["news"])
    assert rig.now() == "2026-11-10T14:00:00Z"  # a day of hourly ticks, not a month


@pytest.mark.asyncio
async def test_an_event_still_to_come_keeps_the_run_going_past_the_ticks(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    rig.triggers(NEWS, rig.notify_at("sam", "2026-11-12T14:00:00Z", "Sam: flights?", channel="noor 30"))
    ctx = await _run(rig, rig.step())

    woke = [(w["virtual_time"], w["woke_by"]) for w in rig.events(ctx, "wake")]
    assert woke == [("2026-11-09T14:00:00Z", ["user_request"]), ("2026-11-12T14:00:00Z", ["notification:noor 30"])]
    assert rig.record(ctx)["end_reason"] == "finished_nothing_pending"
    assert rig.events(ctx, "world_only_repeats")[0]["since"] == "2026-11-12T14:00:00Z"


@pytest.mark.asyncio
async def test_a_known_reply_ahead_isn_t_cut_short_by_the_ticks(monkeypatch, tmp_path):
    monkeypatch.setattr(ao, "IDLE_JUMPS_MAX", 3)

    def agent(n, texts, rig):
        if n == 1:
            rig.message("blue house or green?")
        else:
            rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, answers=[{"outcome": "reply", "message": "blue"}])
    await rig.arm()
    rig.triggers(NEWS)
    user = {"schedule": {"wake": "07:30", "sleep": "23:30", "work": ["09:00", "18:00"]},
            "response_times": {"work": 600, "free": 600, "morning": 600}}  # her reply reaches it ten hours later
    ctx = await _run(rig, rig.step(), ctx=rig.context(user=user))

    woke = [(w["virtual_time"], w["woke_by"]) for w in rig.events(ctx, "wake")]
    assert woke == [("2026-11-09T14:00:00Z", ["user_request"]), ("2026-11-10T00:00:00Z", ["user_reply"])]
    assert rig.record(ctx)["end_reason"] == "finished_nothing_pending"
    assert rig.now() == "2026-11-10T03:00:00Z"  # then three ticks with nothing ahead


@pytest.mark.asyncio
async def test_with_an_end_time_the_run_goes_on_to_it_and_folds_the_jumps(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    rig.triggers(NEWS)
    ctx = await _run(rig, rig.step(end_time="P3D"))

    record = rig.record(ctx)
    assert record["end_reason"] == "end_time" and rig.now() == "2026-11-12T14:00:00Z"
    assert record["counts"]["clock_jumps"] == 72
    jumps = rig.events(ctx, "clock_jump")
    [folded] = rig.events(ctx, "clock_jumps_folded")
    assert len(jumps) == ao.JUMPS_KEPT_PER_GAP and folded["count"] == 72 - ao.JUMPS_KEPT_PER_GAP
    assert folded["to"] == "2026-11-12T14:00:00Z" and sum(folded["reasons"].values()) == folded["count"]
    assert rig.events(ctx, "world_only_repeats") == []
