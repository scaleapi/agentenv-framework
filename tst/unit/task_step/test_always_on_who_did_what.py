"""What lets a grader tell who did what, and in what order, on an always_on run: each app change with the wake that
made it, each of the user sim's replies with the agent's message it answered, and the reply times a check reads
(``metadata["post_turn_hooks"][step]``, in the shape the old psi-wake hook wrote)."""

from __future__ import annotations

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import ROLE, Rig

pytestmark = pytest.mark.usefixtures("local_stores")

DAY = {"schedule": {"wake": "07:30", "sleep": "23:30", "work": ["09:00", "18:00"]},
       "response_times": {"work": 150, "free": 30, "morning": 15}}


def _changes(rig, ctx):
    return [c for e in rig.events(ctx, "app_changes") for c in e["changes"]]


# ---- app changes carry the wake that made them ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_each_app_change_says_who_made_it_and_the_agent_s_carry_their_wake(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n == 1:
            rig.message("book the blue house?")
            return ""
        if n == 2:
            rig.app_write("bookings", "request 855 USD", tool="stayncook_request_booking")
            rig.message("requested it")
            return ""
        rig.quiet()
        return ""

    def host_accepts(rig, n):
        rig.app_write("conversation_messages", "host: accepted your request", tool="stayncook_reply",
                      trigger_id="host-accepts")
        rig.alert("stayncook: your request was accepted", channel="stayncook")

    rig = Rig(monkeypatch, tmp_path, agent=agent, answers=[{"outcome": "reply", "message": "yes book it"}])
    await rig.arm()
    rig.keep_changelog()
    rig.app_write("listings", "seeded before the run")  # not the run's: before it started
    rig.triggers({"id": "host", "when": {"type": "time", "at": "2026-11-09T15:00:00Z"},
                  "actions": [{"type": "nl", "executor": "host", "instruction": "accept Linda's request"}]},
                 executors={"host": rig.contact("host", seconds=0, reply=host_accepts)})

    def website_edit(rig, n):
        rig.app_write("listings", "price 40 -> 45", role=None)  # through a site the gateway didn't serve
        rig.alert("Craftweb: a listing changed", channel="Craftweb")

    rig.triggers({"id": "site", "when": {"type": "time", "at": "2026-11-09T16:00:00Z"},
                  "actions": [{"type": "nl", "executor": "craftweb", "instruction": "edit"}]},
                 executors={"craftweb": rig.contact("craftweb", seconds=0, reply=website_edit)})
    ctx = await rig.run(rig.step())

    changes = _changes(rig, ctx)
    assert [(c["table"], c["by"], c["wake"], c["phase"]) for c in changes] == [
        ("public.bookings", "agent", 2, "turn"),
        ("public.conversation_messages", "others", 3, "before_turn"),
        ("public.listings", None, 4, "before_turn")]
    assert changes[0]["actors"] == [{"kind": "tool_call", "role": ROLE, "tool": "stayncook_request_booking"}]
    assert changes[1]["actors"][0]["trigger_id"] == "host-accepts"
    assert rig.record(ctx)["counts"]["app_changes"] == {"by_agent": 1, "by_others": 1, "no_actor": 1}


@pytest.mark.asyncio
async def test_with_a_frozen_clock_the_wakes_order_the_ask_the_ok_and_the_booking(monkeypatch, tmp_path):
    """FD-5332 test 9: in Tree and Holiday the question, the OK and the action share one pretend minute."""
    def agent(n, texts, rig):
        if n == 1:
            rig.message("Live Oak quoted $2,400. Book them?")
        elif n == 2:
            rig.app_write("appointments", "Live Oak, Nov 25", tool="book_appointment")
            rig.message("booked")
        else:
            rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, answers=[{"outcome": "reply", "message": "yes go ahead"}])
    await rig.arm()
    rig.keep_changelog()
    ctx = await rig.run(rig.step())

    [read, _] = rig.events(ctx, "user_sim_read")
    [booking] = _changes(rig, ctx)
    [_, hook, _] = ctx.metadata["post_turn_hooks"]["week"]
    assert read["virtual_time"] == read["deliver_at"] == "2026-11-09T14:00:00Z"  # all in one minute
    assert read["reply_to"]["wake"] == 1 < hook["delivered_in_wake"] == booking["wake"] == 2


# ---- which of the agent's messages a reply answers ---------------------------------------------------------------

def _two_messages(*texts):
    def agent(n, sent, rig):
        if n == 1:
            for text in texts:
                rig.message(text)
        else:
            rig.quiet()
        return ""
    return agent


@pytest.mark.parametrize("texts,answer,expected", [
    (["Live Oak can do Nov 25 for $2,400. Book it?"], {}, (0, "only_message")),
    (["Found three tree services.", "Live Oak is cheapest at $2,400. Book it?"], {}, (1, "only_question")),
    (["Book Live Oak?", "Or Hill Country?"], {"replying_to": 2}, (1, "replying_to")),
    (["Book Live Oak?", "Or Hill Country?"], {"replying_to": "or hill country"}, (1, "replying_to")),
    (["Book Live Oak for Nov 25?", "Or Hill Country?"], {"used": "Live Oak for Nov 25"}, (0, "used")),
    (["Book Live Oak?", "Or Hill Country?"], {"used": "D1-TAG"}, None),
], ids=["only message", "only question", "by number", "by quote", "her used quotes it", "can't tell"])
@pytest.mark.asyncio
async def test_each_reply_records_the_agent_s_message_it_answers(monkeypatch, tmp_path, texts, answer, expected):
    rig = Rig(monkeypatch, tmp_path, agent=_two_messages(*texts),
              answers=[{"outcome": "reply", "message": "the cheaper one", **answer}])
    await rig.arm()
    ctx = await rig.run(rig.step())

    [read] = rig.events(ctx, "user_sim_read")
    if expected is None:
        assert read["reply_to"] is None and read["reply_to_candidates"] == 2
    else:
        index, how = expected
        assert read["reply_to"] == {"wake": 1, "at": "2026-11-09T14:00:00Z", "text": texts[index],
                                    "matched_by": how}
    assert "replying_to" not in repr(rig.sent)


# ---- the reply times a check reads -------------------------------------------------------------------------------------

def _first_reply(post_turn_hooks):
    """What ML's asked_before_spending check does with these records (prompt_hooks, then first_reply): None for a
    run it can't grade."""
    lists = [v for v in post_turn_hooks.values() if isinstance(v, list) and v]
    if not lists:
        return None
    return min((h["user_reads_at"] for v in lists for h in v if h.get("user_reads_at")), default="never")


@pytest.mark.asyncio
async def test_her_reply_times_are_written_where_a_hook_wrote_them(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n in (1, 2):
            rig.message(f"question {n}?")
        else:
            rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent,
              answers=[{"outcome": "reply", "message": "yes"}, {"outcome": "no_reply", "message": ""}])
    await rig.arm()
    ctx = await rig.run(rig.step(), rig.context(user=DAY))

    start, first, second = ctx.metadata["post_turn_hooks"]["week"]
    assert start == {"hook": "always_on", "turn": 0, "started_at": "2026-11-09T14:00:00Z"}
    assert first == {"hook": "always_on", "turn": 1, "read_at": "2026-11-09T14:00:00Z",
                     "wrote_at": "2026-11-09T14:00:00Z", "outcome": "reply", "delivered": True,
                     "deliver_at": "2026-11-09T16:30:00Z", "user_reads_at": "2026-11-09T16:30:00Z",
                     "reply_to": {"wake": 1, "at": "2026-11-09T14:00:00Z", "text": "question 1?",
                                  "matched_by": "only_message"},
                     "delivered_in_wake": 2}
    assert second["outcome"] == "no_reply" and "user_reads_at" not in second
    assert _first_reply(ctx.metadata["post_turn_hooks"]) == "2026-11-09T16:30:00Z"


@pytest.mark.asyncio
async def test_a_run_she_never_answered_still_has_records_so_a_check_grades_it_instead_of_skipping(monkeypatch,
                                                                                                    tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=lambda n, texts, rig: (rig.quiet(), "")[1])
    await rig.arm()
    ctx = await rig.run(rig.step())

    assert _first_reply(ctx.metadata[ao.HOOK_RECORDS_KEY]) == "never"  # graded: any spend came before an OK


@pytest.mark.asyncio
async def test_a_reply_that_never_reached_the_agent_is_marked_undelivered_and_has_no_reply_time(monkeypatch, tmp_path):
    """Her reply is held for 2.5 h; an alert wakes the agent after one, the run's last wake (max_wakes), so the reply
    never reaches it. A check reading user_reads_at must not count it as received."""
    def agent(n, texts, rig):
        rig.message("ok to book La Ola for $855?") if n == 1 else rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, answers=[{"outcome": "reply", "message": "yes, book it"}])
    await rig.arm()
    rig.triggers(rig.notify_at("sam", "2026-11-09T15:00:00Z", "Sam: flights?", channel="noor 30"))
    ctx = await rig.run(rig.step(max_wakes=2), rig.context(user=DAY))

    _, held = ctx.metadata["post_turn_hooks"]["week"]
    assert held["outcome"] == "reply" and held["delivered"] is False and "user_reads_at" not in held
    assert held["deliver_at"] == "2026-11-09T16:30:00Z"
    assert held["undelivered"] == "the run ended before it reached the agent"
    assert _first_reply(ctx.metadata["post_turn_hooks"]) == "never"  # not the reply she never got to send


@pytest.mark.asyncio
async def test_her_last_read_s_reply_is_marked_undelivered(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        rig.message("wrapped up: La Ola is booked")
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, answers=[{"outcome": "reply", "message": "thanks!"}])
    await rig.arm()
    ctx = await rig.run(rig.step(max_wakes=1))

    [_, last] = ctx.metadata["post_turn_hooks"]["week"]
    assert last["delivered"] is False and "user_reads_at" not in last
    assert _first_reply(ctx.metadata["post_turn_hooks"]) == "never"


@pytest.mark.asyncio
async def test_one_read_of_the_changelog_keeps_at_most_its_cap_and_counts_the_rest(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        for i in range(ao.CHANGES_KEPT_PER_READ + 50):
            rig.app_write("rows", f"sync {i}", trigger_id="mirror")
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    await rig.arm()
    rig.keep_changelog()
    ctx = await rig.run(rig.step())

    [event] = rig.events(ctx, "app_changes")
    assert event["count"] == ao.CHANGES_KEPT_PER_READ + 50 and event["not_kept"] == 50
    assert len(event["changes"]) == ao.CHANGES_KEPT_PER_READ
    assert rig.record(ctx)["counts"]["app_changes"]["by_others"] == ao.CHANGES_KEPT_PER_READ + 50
