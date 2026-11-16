"""The always_on step's loop on an in-process gateway: what wakes the agent, how the clock moves, when the user sim reads
and replies, what each read holds, and how done ends a run."""

from __future__ import annotations

import pytest

from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import DeployUserSimAgentTaskStep
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

LINDA_DAY = {"schedule": {"wake": "07:30", "sleep": "23:30", "work": ["09:00", "18:00"]},
             "response_times": {"work": 150, "free": 30, "morning": 15}}


def _agent(**plays):
    """An agent that plays ``plays["t<n>"]`` on its nth turn and otherwise stays quiet."""
    def agent(n, texts, rig):
        play = plays.get(f"t{n}")
        if play is None:
            rig.quiet()
            return "nothing new"
        return play(rig)
    return agent


def _message(text):
    return lambda rig: (rig.message(text), "sent")[1]


def _woke(ctx, rig):
    return [(w["virtual_time"], w["woke_by"]) for w in rig.events(ctx, "wake")]


def _without_step_lines(texts):
    return [t for t in texts if t not in (ao.WAKE_REMINDER_TEXT, f"{ao.MESSAGE_RULES} {ao.QUIET_RULE}")]


# ---- what wakes the agent -------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_notify_alert_wakes_the_agent_labelled_with_its_time_and_channel(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    rig.triggers(rig.notify_at("sam", "2026-11-10T14:00:00Z", "Sam: she suspects nothing", channel="noor 30"))
    ctx = await rig.run(rig.step())

    assert _woke(ctx, rig) == [("2026-11-09T14:00:00Z", ["user_request"]),
                               ("2026-11-10T14:00:00Z", ["notification:noor 30"])]
    assert "[Tue Nov 10, 2:00 PM UTC · noor 30] Sam: she suspects nothing" in rig.sent[1]
    assert rig.record(ctx)["end_reason"] == "finished_nothing_pending"


@pytest.mark.asyncio
async def test_a_scripted_user_line_reaches_the_agent_as_the_user_and_counts_as_hers(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t2=_message("on it: three dinner spots tonight")))
    await rig.arm()
    rig.triggers(rig.notify_at("ask-dinner", "2026-11-10T15:00:00Z", "also find a dinner spot", to=ao.USER_SAYS))
    ctx = await rig.run(rig.step())

    assert _woke(ctx, rig)[1] == ("2026-11-10T15:00:00Z", ["user_says"])
    assert "[Tue Nov 10, 3:00 PM UTC] Linda: also find a dinner spot" in rig.sent[1]
    [read] = rig.reads
    assert [s["text"] for s in read["you_sent"]] == ["sort the cdmx weekend", "also find a dinner spot"]
    assert [e["text"] for e in rig.events(ctx, "scripted_user_line")] == ["also find a dinner spot"]


@pytest.mark.asyncio
async def test_the_agent_s_scheduled_check_moves_the_clock_and_wakes_it(monkeypatch, tmp_path):
    def book(rig):
        rig.schedule("book La Ola when it opens", at="2026-11-12T13:00:00Z")
        rig.quiet()
        return "check set"

    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=book))
    await rig.arm()
    rig.gw._notifications.configure({"roles": ["default"]})
    ctx = await rig.run(rig.step())

    assert _woke(ctx, rig)[1] == ("2026-11-12T13:00:00Z", ["notification:your scheduled check"])
    assert "[Thu Nov 12, 1:00 PM UTC · your scheduled check] book La Ola when it opens" in rig.sent[1]
    assert [j["to"] for j in rig.events(ctx, "clock_jump")] == ["2026-11-12T13:00:00Z"]


@pytest.mark.asyncio
async def test_a_reply_is_held_for_the_reply_time_then_wakes_the_agent(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("blue house or green?")),
              answers=[{"outcome": "reply", "message": "the blue one"}])
    await rig.arm()
    ctx = await rig.run(rig.step(), rig.context(user=LINDA_DAY))

    [read] = rig.events(ctx, "user_sim_read")
    assert read["virtual_time"] == "2026-11-09T14:00:00Z" and read["mode"] == "work"
    assert read["deliver_at"] == "2026-11-09T16:30:00Z"
    assert _woke(ctx, rig)[1] == ("2026-11-09T16:30:00Z", ["user_reply"])
    assert "[Mon Nov 9, 4:30 PM UTC] Linda: the blue one" in rig.sent[1]


@pytest.mark.asyncio
async def test_the_user_reads_once_awake_and_not_busy(monkeypatch, tmp_path):
    busy = {**LINDA_DAY["schedule"], "busy": [{"start": "2026-11-10T07:00", "end": "2026-11-10T10:00"}]}
    rig = Rig(monkeypatch, tmp_path, t0="2026-11-10T05:00:00Z",  # midnight in New York
              agent=_agent(t1=_message("found two houses"), t2=_message("and a third")),
              answers=[{"outcome": "reply", "message": "send all three"}])
    await rig.arm()
    ctx = await rig.run(rig.step(), rig.context(user={**LINDA_DAY, "schedule": busy}))

    [scheduled, *_] = rig.events(ctx, "user_read_scheduled")
    assert scheduled["read_at"] == "2026-11-10T15:00:00Z"  # past the night and the 7-10 AM block
    [first, second] = rig.events(ctx, "user_sim_read")
    assert first["virtual_time"] == "2026-11-10T15:00:00Z"
    assert first["deliver_at"] == "2026-11-10T15:15:00Z"


@pytest.mark.asyncio
async def test_messages_sent_before_a_read_arrive_in_that_one_read(monkeypatch, tmp_path):
    def two(rig):
        rig.message("house: the blue one, $855")
        rig.message("dinner: La Ola, Saturday 8pm")
        return "sent both"

    rig = Rig(monkeypatch, tmp_path, t0="2026-11-10T05:00:00Z", agent=_agent(t1=two))
    await rig.arm()
    await rig.run(rig.step(), rig.context(user=LINDA_DAY))

    [read] = rig.reads
    assert read["time"] == "2026-11-10T12:30:00Z"
    assert [m["text"] for m in read["assistant_messages"]] == ["house: the blue one, $855",
                                                               "dinner: La Ola, Saturday 8pm"]


# ---- the clock ------------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_clock_jumps_to_each_next_event_in_turn_never_past_one(monkeypatch, tmp_path):
    def check(rig):
        rig.schedule("chase the host", at="2026-11-10T11:00:00Z")
        rig.quiet()
        return "ok"

    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=check))
    await rig.arm()
    rig.gw._notifications.configure({"roles": ["default"]})
    rig.triggers(rig.notify_at("landlord", "2026-11-10T09:30:00Z", "rent", to="landlord"),
                 rig.notify_at("noon", "2026-11-10T12:00:00Z", "Sam: flights?", channel="noor 30"),
                 rig.notify_at("ten", "2026-11-10T10:00:00Z", "Pri: I'm in", channel="noor 30"))
    ctx = await rig.run(rig.step())

    assert [j["to"] for j in rig.events(ctx, "clock_jump")] == [
        "2026-11-10T09:30:00Z", "2026-11-10T10:00:00Z", "2026-11-10T11:00:00Z", "2026-11-10T12:00:00Z"]
    assert [t for t, _ in _woke(ctx, rig)[1:]] == ["2026-11-10T10:00:00Z", "2026-11-10T11:00:00Z",
                                                   "2026-11-10T12:00:00Z"]


@pytest.mark.asyncio
async def test_what_arrives_in_the_same_minute_comes_in_one_wake(monkeypatch, tmp_path):
    def check(rig):
        rig.schedule("look at the airline change", at="2026-11-10T10:00:40Z")
        rig.quiet()
        return "ok"

    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=check))
    await rig.arm()
    rig.gw._notifications.configure({"roles": ["default"]})
    rig.triggers(rig.notify_at("airline", "2026-11-10T10:00:00Z", "Wingloft: your flight was retimed",
                               channel="Gmail"),
                 rig.notify_at("ask", "2026-11-10T10:00:20Z", "did the flight change?", to=ao.USER_SAYS),
                 rig.notify_at("next", "2026-11-10T10:01:00Z", "Dev: thx!", channel="Messages"))
    ctx = await rig.run(rig.step())

    woke = _woke(ctx, rig)
    assert woke[1] == ("2026-11-10T10:00:40Z", ["notification:Gmail", "notification:your scheduled check",
                                                "user_says"])
    assert woke[2] == ("2026-11-10T10:01:00Z", ["notification:Messages"])
    assert len(woke) == 3


# ---- what the user sim sees, and done ----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_each_read_holds_the_time_the_agent_s_messages_her_phone_and_only_her_own_lines(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n == 1:
            return "Linda should know the house is booked."
        if any("Sam: shh" in t for t in texts):
            rig.message("house booked; Sam says she suspects nothing")
            return "sent"
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    await rig.arm()
    rig.gw._notifications.configure({"roles": ["default"]})
    rig.schedule("chase the host", at="2026-11-09T18:00:00Z")
    rig.triggers(rig.notify_at("phone", "2026-11-09T19:00:00Z", "New email from Sam", channel="Gmail",
                               to=ao.USER_PHONE),
                 rig.notify_at("ask", "2026-11-09T19:30:00Z", "is the house sorted?", to=ao.USER_SAYS),
                 rig.notify_at("alert", "2026-11-09T20:00:00Z", "Sam: shh", channel="noor 30"))
    ctx = await rig.run(rig.step(end_time="2026-11-10T02:00:00Z", final="send me the rundown"))

    [read] = rig.reads
    assert read["time"] == "2026-11-09T20:00:00Z"
    assert [m["text"] for m in read["assistant_messages"]] == ["house booked; Sam says she suspects nothing"]
    assert [(p["channel"], p["text"]) for p in read["phone"]] == [("Gmail", "New email from Sam")]
    assert [s["text"] for s in read["you_sent"]] == ["sort the cdmx weekend", "is the house sorted?"]
    seen = repr(rig.reads)
    for meant_for_the_agent in ("Sam: shh", "chase the host", ao.NUDGE_TEXT, ao.WAKE_REMINDER_TEXT,
                                ao.MESSAGE_RULES):
        assert meant_for_the_agent not in seen
    assert rig.events(ctx, "nudge") and rig.record(ctx)["end_reason"] == "end_time"


@pytest.mark.asyncio
async def test_done_ends_a_one_sitting_run_after_the_agent_s_turn_on_her_sign_off(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("all set: shelves restocked"),
                                                  t2=_message("glad to help!")),
              answers=[{"outcome": "done", "message": "thanks!"}])
    await rig.arm()
    rig.triggers(rig.notify_at("later", "2026-11-10T14:00:00Z", "a new order", channel="Craftweb"))
    ctx = await rig.run(rig.step())

    assert rig.record(ctx)["end_reason"] == "user_done" and len(rig.sent) == 2
    assert "[Mon Nov 9, 2:00 PM UTC] Linda: thanks!" in rig.sent[1]
    assert len(rig.reads) == 1 and rig.events(ctx, "user_sim_read")[0]["ends_run"] is True


@pytest.mark.asyncio
async def test_a_blank_done_ends_a_one_sitting_run_at_once(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("all set")),
              answers=[{"outcome": "done", "message": "  "}])
    await rig.arm()
    ctx = await rig.run(rig.step())

    assert rig.record(ctx)["end_reason"] == "user_done" and len(rig.sent) == 1


@pytest.mark.asyncio
async def test_done_in_a_multi_day_run_is_recorded_and_the_run_goes_on_to_end_time(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("house booked"), t2=_message("airline moved Dev")),
              answers=[{"outcome": "done", "message": "perfect ty"}, {"outcome": "reply", "message": "ok!"}])
    await rig.arm()
    rig.triggers(rig.notify_at("airline", "2026-11-11T14:00:00Z", "Wingloft: flight retimed", channel="Gmail"))
    ctx = await rig.run(rig.step(end_time="P3D"))

    record = rig.record(ctx)
    assert record["multi_day"] is True and record["end_reason"] == "end_time"
    assert [r["outcome"] for r in rig.events(ctx, "user_sim_read")] == ["done", "reply"]
    assert rig.events(ctx, "user_sim_read")[0]["ends_run"] is False
    assert [m["text"] for m in rig.reads[1]["assistant_messages"]] == ["airline moved Dev"]
    assert record["counts"]["user_dones"] == 1
    assert any("Linda: perfect ty" in t for texts in rig.sent for t in texts)


@pytest.mark.parametrize("end_time,ends", [(None, True), ("P3D", False)], ids=["one sitting", "multi-day"])
@pytest.mark.asyncio
async def test_each_read_says_whether_done_would_end_the_run(monkeypatch, tmp_path, end_time, ends):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("house booked")))
    await rig.arm()
    await rig.run(rig.step(**({"end_time": end_time} if end_time else {})))

    [read] = rig.reads
    assert read["done_ends_run"] is ends


@pytest.mark.parametrize("end_time,multi_day", [(None, False), ("PT6H", False), ("P1D", True)])
@pytest.mark.asyncio
async def test_a_run_is_multi_day_when_its_end_falls_on_a_later_day_for_the_user(monkeypatch, tmp_path, end_time,
                                                                                  multi_day):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    ctx = await rig.run(rig.step(**({"end_time": end_time} if end_time else {})))
    assert rig.record(ctx)["multi_day"] is multi_day


# ---- the time zone --------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_world_s_time_zone_is_the_user_s_when_the_user_sim_sets_none(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    ctx = rig.context(user={"time_zone": None})
    ctx.metadata["clock_configurations"] = [{"step_id": "clock", "timezone": "America/Los_Angeles"}]
    ctx = await rig.run(rig.step(), ctx)

    assert rig.record(ctx)["time_zone"] == "America/Los_Angeles"
    assert rig.setups[0]["time_zone"] == "America/Los_Angeles"


@pytest.mark.asyncio
async def test_a_user_and_a_world_in_different_time_zones_are_refused(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    ctx = rig.context()
    ctx.metadata["clock_configurations"] = [{"step_id": "clock", "timezone": "America/Los_Angeles"}]
    with pytest.raises(ValueError, match="America/New_York but sync_env_clock set"):
        await rig.run(rig.step(), ctx)


# ---- the steps ------------------------------------------------------------------------------------------------------

def test_both_steps_are_registered():
    registry = get_task_step_registry()
    assert registry["always_on"] is ao.AlwaysOnTaskStep
    assert registry["deploy_user_sim_agent"] is DeployUserSimAgentTaskStep and "user_sim" not in registry


# ---- the user's schedule --------------------------------------------------------------------------------------------

def _available(monkeypatch, tmp_path, schedule: dict, at: str):
    rig = Rig(monkeypatch, tmp_path)
    run = ao._Run(rig.step(), rig.context(user={"schedule": schedule}))
    run.us = {"schedule": schedule}
    run.tz = ao.ZoneInfo("America/New_York")
    got, mode = run._available(ao._utc(at))
    return ao._iso(got), mode


@pytest.mark.parametrize("at,read_at,mode", [
    ("2026-11-10T14:00:00Z", "2026-11-10T14:00:00Z", "work"),     # Tue 9:00 AM: awake
    ("2026-11-11T04:00:00Z", "2026-11-11T04:00:00Z", "free"),     # Tue 11:00 PM: still up
    ("2026-11-11T05:15:00Z", "2026-11-11T05:15:00Z", "free"),     # Wed 12:15 AM: still up
    ("2026-11-11T05:45:00Z", "2026-11-11T13:00:00Z", "morning"),  # Wed 12:45 AM: asleep until 8:00
    ("2026-11-11T10:00:00Z", "2026-11-11T13:00:00Z", "morning"),  # Wed 5:00 AM: asleep until 8:00
])
def test_a_night_that_starts_after_midnight_keeps_the_evening_awake(monkeypatch, tmp_path, at, read_at, mode):
    night_owl = {"wake": "08:00", "sleep": "00:30", "work": ["09:00", "18:00"]}
    assert _available(monkeypatch, tmp_path, night_owl, at) == (read_at, mode)


def test_back_to_back_busy_blocks_and_nights_never_leave_a_read_inside_one(monkeypatch, tmp_path):
    blocks = [("2026-11-10T08:00", "2026-11-10T09:00"), ("2026-11-10T09:00", "2026-11-10T10:00"),
              ("2026-11-10T10:00", "2026-11-10T23:45"),  # into the night: she reads next morning
              ("2026-11-11T07:30", "2026-11-11T08:30"), ("2026-11-11T08:30", "2026-11-11T09:30"),
              ("2026-11-11T09:30", "2026-11-11T10:00")]
    schedule = {"wake": "07:30", "sleep": "23:30", "work": ["09:00", "18:00"],
                "busy": [{"start": a, "end": b} for a, b in reversed(blocks)]}  # latest first: one block a pass
    # Tue 8:00 AM in New York: six blocks and a night in a row, more than the old four passes
    assert _available(monkeypatch, tmp_path, schedule, "2026-11-10T13:00:00Z") == ("2026-11-11T15:00:00Z", "morning")
