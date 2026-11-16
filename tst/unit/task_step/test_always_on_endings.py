"""How an always_on run ends: a final check-in at end_time whether or not the task sets a `final` line, a last-turn
notice at end_time and before the wake cap, the user sim's last read of the agent's wrap-up, what was still pending,
and end reasons that tell a finished run from a stuck one."""

from __future__ import annotations

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

DAY = {"schedule": {"wake": "07:30", "sleep": "23:30", "work": ["09:00", "18:00"]},
       "response_times": {"work": 150, "free": 30, "morning": 15}}
NOTICE = ao.FINAL_REMINDER.format(first="Linda")
DEFAULT_FINAL = ao.FINAL_DEFAULT_TEXT.format(first="Linda")


def _agent(**plays):
    def agent(n, texts, rig):
        play = plays.get(f"t{n}")
        if play is None:
            rig.quiet()
            return ""
        return play(rig)
    return agent


def _message(text):
    return lambda rig: (rig.message(text), "sent")[1]


def _recaps(n, texts, rig):
    """Messages Linda on a wake that carries the last-turn notice, and stays quiet otherwise."""
    if NOTICE in texts:
        rig.message("RESULT: house booked, flights held; La Ola still open")
        return "sent"
    rig.quiet()
    return ""


# ---- 1. a final check-in at end_time, with or without a `final` line -------------------------------------------------

@pytest.mark.asyncio
async def test_without_a_final_line_the_end_time_still_brings_a_check_in_with_the_step_s_own_line(monkeypatch,
                                                                                                tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_recaps, answers=[{"outcome": "reply", "message": "perfect, thanks"}])
    await rig.arm()
    ctx = await rig.run(rig.step(end_time="P2D"))

    last = rig.parts[-1]
    assert [p["text"] for p in last if p["metadata"]["agentenv"]["source"] == "final"] == [DEFAULT_FINAL]
    assert rig.events(ctx, "wake")[-1]["virtual_time"] == "2026-11-11T14:00:00Z"
    assert rig.events(ctx, "wake")[-1]["woke_by"] == ["step:final"]
    assert DEFAULT_FINAL not in repr([r["you_sent"] for r in rig.reads])  # the step's line, not hers
    assert rig.record(ctx)["end_reason"] == "end_time"


@pytest.mark.asyncio
async def test_a_turn_that_runs_past_end_time_is_followed_by_the_final_check_in_not_more_wakes(monkeypatch, tmp_path):
    """The clock runs during the agent's turns, so one can end past end_time: what is due then (her reply, an alert)
    comes with the final check-in, rather than in ordinary wakes that use up max_wakes without one."""
    def chatty(n, texts, rig):
        rig.message(f"update {n}: anything else?")
        if n == 1:
            rig.alert("Live Oak: we can come Friday", channel="Gmail · Live Oak Arbor")
            rig.gw._clock.advance("2026-11-10T15:00:00Z")  # a long turn: the task's clock runs past end_time
        return "sent"

    rig = Rig(monkeypatch, tmp_path, agent=chatty,
              answers=[{"outcome": "reply", "message": f"reply {i}"} for i in range(1, 6)])
    await rig.arm()
    ctx = await rig.run(rig.step(end_time="P1D", max_wakes=4))

    assert len(rig.sent) == 2  # the opening wake, then the final check-in
    final = " ".join(rig.sent[1])
    assert DEFAULT_FINAL in final and NOTICE in final
    assert "Linda: reply 1" in final and "Live Oak: we can come Friday" in final
    assert rig.events(ctx, "wake")[-1]["virtual_time"] == "2026-11-10T15:00:00Z"
    assert rig.record(ctx)["end_reason"] == "end_time"


@pytest.mark.asyncio
async def test_a_task_s_final_line_is_hers_and_comes_with_the_notice(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_recaps)
    await rig.arm()
    await rig.run(rig.step(end_time="P2D", final="send me the rundown"))

    texts = rig.sent[-1]
    assert "[Wed Nov 11, 2:00 PM UTC] Linda: send me the rundown" in texts and DEFAULT_FINAL not in texts
    assert texts[-1] == NOTICE and ao.WAKE_REMINDER_TEXT not in texts
    assert [s["text"] for s in rig.reads[-1]["you_sent"]] == ["sort the cdmx weekend", "send me the rundown"]


# ---- 2. the last-turn notice ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_wake_the_cap_makes_the_last_carries_the_notice_and_no_earlier_one_does(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_recaps)
    await rig.arm()
    for day in range(10, 16):
        rig.triggers(rig.notify_at(f"alert-{day}", f"2026-11-{day}T14:00:00Z", "Sam: any news?", channel="noor 30"))
    ctx = await rig.run(rig.step(max_wakes=4))

    assert [NOTICE in texts for texts in rig.sent] == [False, False, False, True]
    assert [w["last"] for w in rig.events(ctx, "wake")] == [False, False, False, True]
    assert rig.record(ctx)["end_reason"] == "max_wakes"
    assert [m["text"] for m in rig.reads[-1]["assistant_messages"]] == [
        "RESULT: house booked, flights held; La Ola still open"]


# ---- 3. final -> the agent's turn -> her last read -> her reaction recorded -> end ----------------------------------

@pytest.mark.asyncio
async def test_the_user_sim_reads_the_wrap_up_and_her_reaction_and_held_reply_are_recorded(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_recaps,
              answers=[{"outcome": "reply", "message": "amazing, book La Ola too"}])
    await rig.arm()
    ctx = await rig.run(rig.step(end_time="P2D", final="send me the rundown"), rig.context(user=DAY))

    order = [e["event"] for e in rig.timeline(ctx)]
    tail = order[order.index("scripted_user_line"):]
    assert tail[:2] == ["scripted_user_line", "wake"] and tail[-2:] == ["user_sim_read", "end"]
    assert tail.index("message_to_user") < tail.index("user_sim_read")
    [read] = rig.events(ctx, "user_sim_read")
    assert read["last_read"] is True and read["outcome"] == "reply"
    assert read["virtual_time"] == "2026-11-11T14:00:00Z" and read["deliver_at"] == "2026-11-11T16:30:00Z"
    held = rig.record(ctx)["pending_at_end"]["held_replies"]
    assert held == {"count": 1, "items": [{"at": "2026-11-11T16:30:00Z", "text": "amazing, book La Ola too",
                                           "done": False}]}
    assert rig.record(ctx)["end_reason"] == "end_time" and len(rig.sent) == 2


@pytest.mark.asyncio
async def test_the_last_read_comes_after_every_message_of_a_last_turn_that_ran_on(monkeypatch, tmp_path):
    """Her last read is no earlier than the task's time when the run ended, so it never shows a message before it was
    sent, nor holds a reply that comes before the question it answers."""
    def wraps_up_slowly(n, texts, rig):
        rig.message("booked the house")
        rig.gw._clock.advance("2026-11-09T16:00:00Z")  # the turn goes on two hours, the clock running
        rig.message("and the flights: is Friday ok?")
        return "sent"

    rig = Rig(monkeypatch, tmp_path, agent=wraps_up_slowly,
              answers=[{"outcome": "reply", "message": "friday is fine"}])
    await rig.arm()
    ctx = await rig.run(rig.step(max_wakes=1))

    [read] = rig.reads
    assert [m["at"] for m in read["assistant_messages"]] == ["2026-11-09T14:00:00Z", "2026-11-09T16:00:00Z"]
    assert read["time"] == "2026-11-09T16:00:00Z"
    [event] = rig.events(ctx, "user_sim_read")
    assert event["last_read"] is True and event["deliver_at"] == "2026-11-09T16:00:00Z"


@pytest.mark.asyncio
async def test_a_last_read_that_would_fall_in_her_night_waits_for_the_morning(monkeypatch, tmp_path):
    def wraps_up_late(n, texts, rig):
        rig.message("booked the house")
        rig.gw._clock.advance("2026-11-10T05:00:00Z")  # midnight in New York: past her bedtime
        rig.message("and the flights: is Friday ok?")
        return "sent"

    rig = Rig(monkeypatch, tmp_path, agent=wraps_up_late, answers=[{"outcome": "reply", "message": "friday is fine"}])
    await rig.arm()
    ctx = await rig.run(rig.step(max_wakes=1), rig.context(user=DAY))

    [read] = rig.reads
    assert read["time"] == "2026-11-10T12:30:00Z"  # 7:30 AM, when she wakes
    [event] = rig.events(ctx, "user_sim_read")
    assert (event["mode"], event["deliver_at"]) == ("morning", "2026-11-10T12:45:00Z")


@pytest.mark.asyncio
async def test_a_read_whose_time_passed_during_a_turn_that_ran_into_her_night_waits_for_the_morning(monkeypatch,
                                                                                                    tmp_path):
    def runs_late(n, texts, rig):
        if n == 1:
            rig.message("booked the house")
            rig.gw._clock.advance("2026-11-10T05:00:00Z")  # midnight in New York: past her bedtime
            return "sent"
        rig.quiet()
        return "nothing new"

    rig = Rig(monkeypatch, tmp_path, agent=runs_late, answers=[{"outcome": "reply", "message": "great"}])
    await rig.arm()
    ctx = await rig.run(rig.step(max_wakes=3), rig.context(user=DAY))

    [read, *_] = rig.reads
    assert read["time"] == "2026-11-10T12:30:00Z"  # 7:30 AM, when she wakes, not at midnight
    [event, *_] = rig.events(ctx, "user_sim_read")
    assert (event["mode"], event["deliver_at"]) == ("morning", "2026-11-10T12:45:00Z")


@pytest.mark.asyncio
async def test_a_done_on_the_last_read_is_recorded_and_changes_nothing(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_recaps, answers=[{"outcome": "done", "message": "thanks!!"}])
    await rig.arm()
    ctx = await rig.run(rig.step(max_wakes=1))

    [read] = rig.events(ctx, "user_sim_read")
    assert read["last_read"] is True and read["outcome"] == "done" and "ends_run" not in read
    assert rig.record(ctx)["end_reason"] == "max_wakes"
    assert rig.record(ctx)["pending_at_end"]["held_replies"]["items"][0]["done"] is True


@pytest.mark.asyncio
async def test_a_reply_still_held_at_end_time_is_kept_not_dropped(monkeypatch, tmp_path):
    """FD-5332 test 8: Linda's last reply was lost at end_time, and the record said nothing was held."""
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t2=_message("flight retimed, Dev is fine")),
              answers=[{"outcome": "reply", "message": "ok good, tell Sam"}])
    await rig.arm()
    rig.triggers(rig.notify_at("airline", "2026-11-10T13:50:00Z", "Wingloft: flight retimed", channel="Gmail"))
    ctx = await rig.run(rig.step(end_time="P1D"), rig.context(user=DAY))

    [read] = rig.events(ctx, "user_sim_read")
    assert read["deliver_at"] == "2026-11-10T14:20:00Z"  # after end_time
    assert [i["text"] for i in rig.record(ctx)["pending_at_end"]["held_replies"]["items"]] == ["ok good, tell Sam"]
    assert rig.events(ctx, "end")[0]["held_at_end"] == 1


# ---- 5. what was still pending ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_agent_s_checks_not_yet_due_are_recorded_and_events_after_the_end_never_happened(monkeypatch,
                                                                                                    tmp_path):
    def sets_a_check(n, texts, rig):
        if n == 1:
            rig.schedule("chase Live Oak", at="2026-11-20T14:00:00Z")
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=sets_a_check)
    await rig.arm()
    rig.triggers(rig.notify_at("late", "2026-11-12T14:00:00Z", "Live Oak: invoice", channel="Gmail"),
                 rig.notify_at("ask", "2026-11-12T15:00:00Z", "did they invoice?", to=ao.USER_SAYS))
    ctx = await rig.run(rig.step(end_time="P2D"))

    pending = rig.record(ctx)["pending_at_end"]
    assert pending["scheduled_checks_not_due"]["items"][0]["note"] == "chase Live Oak"
    end = rig.events(ctx, "end")[0]["pending"]
    assert end["alerts_never_delivered"] == end["user_lines_not_delivered"] == []  # never came due: not triggered


@pytest.mark.asyncio
async def test_an_alert_that_arrives_during_the_last_turn_is_recorded_as_never_delivered(monkeypatch, tmp_path):
    def last_turn_brings_mail(n, texts, rig):
        if NOTICE in texts:
            rig.alert("Live Oak: we can come Friday", channel="Gmail · Live Oak Arbor")
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=last_turn_brings_mail)
    await rig.arm()
    ctx = await rig.run(rig.step(max_wakes=1))

    [alert] = rig.record(ctx)["pending_at_end"]["alerts_never_delivered"]["items"]
    assert (alert["channel"], alert["text"]) == ("Gmail · Live Oak Arbor", "Live Oak: we can come Friday")


# ---- 7. finished vs stuck --------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_with_nothing_left_to_wake_the_agent_and_nothing_pending_the_run_finished(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("report sent")),
              answers=[{"outcome": "no_reply", "message": ""}])
    await rig.arm()
    ctx = await rig.run(rig.step())

    record = rig.record(ctx)
    assert record["end_reason"] == "finished_nothing_pending"
    assert all(v["count"] == 0 for v in record["pending_at_end"].values())


@pytest.mark.asyncio
async def test_with_nothing_left_to_wake_the_agent_but_a_message_she_will_never_read_the_run_is_stuck(monkeypatch,
                                                                                                      tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("blue or green?"), t2=_message("and the dates?")),
              answers=[{"outcome": "reply", "message": "blue"}])
    await rig.arm()
    ctx = await rig.run(rig.step(), rig.context(user={"max_replies": 1}))

    record = rig.record(ctx)
    assert record["end_reason"] == "stuck"
    assert [i["text"] for i in record["pending_at_end"]["unread_by_user"]["items"]] == ["and the dates?"]
    assert record["pending_at_end"]["unread_by_user"]["items"][0]["wake"] == 2
    assert record["end_note"] == ao.END_NOTES["stuck"]


@pytest.mark.asyncio
async def test_a_contact_still_replying_when_nothing_else_can_wake_the_agent_makes_the_run_stuck(monkeypatch,
                                                                                                 tmp_path):
    monkeypatch.setattr(ao, "CONTACT_GRACE_SECONDS", 0)
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    slow = rig.contact("liveoak", seconds=5, reply=lambda rig, n: rig.alert("Live Oak: estimate"),
                       timeout_seconds=0.2, keeps_to_its_limit=False)
    rig.triggers({"id": "liveoak-estimate", "when": {"type": "time", "at": "2026-11-09T15:00:00Z"},
                  "actions": [{"type": "nl", "executor": "liveoak", "instruction": "send your estimate"}]},
                 executors={"liveoak": slow})
    try:
        ctx = await rig.run(rig.step(settle_timeout_seconds=0.2))
    finally:
        await rig.close()

    record = rig.record(ctx)
    assert record["end_reason"] == "stuck"
    [env] = record["pending_at_end"]["contact_replies_in_progress"]["items"]
    assert env["triggers"][0]["id"] == "liveoak-estimate"


@pytest.mark.asyncio
async def test_messages_a_turn_sent_before_it_timed_out_are_in_the_graded_response_and_pending(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        rig.message("found it: the blue house, $855")
        raise TimeoutError("the agent's turn ran out")

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    await rig.arm()
    ctx = await rig.run(rig.step())

    [response] = ctx.prompt_responses
    assert response.response == "[Mon Nov 9, 2:00 PM UTC] found it: the blue house, $855"
    assert response.error_type == "agent_turn_timeout"
    [sent] = rig.events(ctx, "message_to_user")
    assert sent["leftover"] is True
    [unread] = rig.record(ctx)["pending_at_end"]["unread_by_user"]["items"]
    assert unread["text"] == "found it: the blue house, $855" and unread["at"] == "2026-11-09T14:00:00Z"
    assert "found it: the blue house" in " ".join(t for m in rig.chat(ctx) for t in m["texts"])
