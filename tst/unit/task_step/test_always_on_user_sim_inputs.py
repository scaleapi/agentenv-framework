"""What the always_on loop hands the user sim, and what it does with her answers: one test per item the loop owns on
the user sim's checklist of what she sees and must never see (the ids in each test name)."""

from __future__ import annotations

import json

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

DAY = {"schedule": {"wake": "07:30", "sleep": "23:30", "work": ["09:00", "18:00"]},
       "response_times": {"work": 150, "free": 30, "morning": 15}}
MIDNIGHT = "2026-11-10T05:00:00Z"  # Tue Nov 10, 12:00 AM in New York


def _agent(**plays):
    """An agent that plays ``plays["t<n>"]`` on its nth turn and otherwise stays quiet."""
    def agent(n, texts, rig):
        play = plays.get(f"t{n}")
        if play is None:
            rig.quiet()
            return ""
        return play(rig)
    return agent


def _message(*texts):
    def play(rig):
        for text in texts:
            rig.message(text)
        return "sent"
    return play


def _texts(items):
    return [i["text"] for i in items]


def _woken_by_user(rig):
    return [t for texts in rig.sent for t in texts if "] Linda: " in t]


@pytest.mark.asyncio
async def test_a1_she_reads_only_after_the_agent_uses_message_user(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=lambda rig: "planning", t3=_message("house booked")))
    await rig.arm()
    rig.triggers(rig.notify_at("alert", "2026-11-10T14:00:00Z", "Sam: shh", channel="noor 30"))
    ctx = await rig.run(rig.step())

    [read] = rig.reads
    assert read["time"] == "2026-11-10T14:00:00Z" and _texts(read["assistant_messages"]) == ["house booked"]
    assert len(rig.events(ctx, "user_sim_read")) == 1


@pytest.mark.asyncio
async def test_a4_a5_her_phone_shows_only_what_is_new_with_its_time_and_full_text(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t2=_message("first"), t3=_message("second")))
    await rig.arm()
    rig.triggers(rig.notify_at("mom", "2026-11-09T15:00:00Z", "Are you coming Sunday?\nBring the pie.",
                               channel="Gmail · Carol Reyes (Mom)", to=ao.USER_PHONE),
                 rig.notify_at("alert-1", "2026-11-09T16:00:00Z", "Sam: shh", channel="noor 30"),
                 rig.notify_at("bank", "2026-11-10T13:00:00Z", "Your card was charged $855.",
                               channel="Messages · Chase", to=ao.USER_PHONE),
                 rig.notify_at("alert-2", "2026-11-10T14:00:00Z", "Dev: thx", channel="noor 30"))
    await rig.run(rig.step())

    first, second = rig.reads
    assert first["phone"] == [{"at": "2026-11-09T15:00:00Z", "channel": "Gmail · Carol Reyes (Mom)",
                               "text": "Are you coming Sunday?\nBring the pie."}]
    assert second["phone"] == [{"at": "2026-11-10T13:00:00Z", "channel": "Messages · Chase",
                                "text": "Your card was charged $855."}]


@pytest.mark.asyncio
async def test_a8_b3_you_sent_holds_only_her_own_lines_each_at_its_time(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if any("Linda: also find a dinner spot" in t for t in texts):
            rig.message("three dinner spots")
            return "sent"
        if n == 1:
            rig.message("on it")
            return "sent"
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, answers=[{"outcome": "reply", "message": "great"}])
    await rig.arm()
    rig.triggers(rig.notify_at("ask", "2026-11-10T15:00:00Z", "also find a dinner spot", to=ao.USER_SAYS),
                 rig.notify_at("alert", "2026-11-10T16:00:00Z", "Sam: shh", channel="noor 30"))
    ctx = await rig.run(rig.step(end_time="2026-11-11T02:00:00Z", final="send me the rundown"))

    first, second = rig.reads
    assert first["you_sent"] == [{"at": "2026-11-09T14:00:00Z", "text": "sort the cdmx weekend"}]
    assert second["you_sent"] == [{"at": "2026-11-10T15:00:00Z", "text": "also find a dinner spot"}]
    [final] = [e for e in rig.events(ctx, "scripted_user_line") if e["kind"] == "final"]
    assert (final["virtual_time"], final["text"]) == ("2026-11-11T02:00:00Z", "send me the rundown")
    assert "great" not in repr([r["you_sent"] for r in rig.reads])


@pytest.mark.asyncio
async def test_a9_every_message_since_her_last_read_arrives_in_one_read_oldest_first(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, t0=MIDNIGHT,
              agent=_agent(t1=_message("house: blue, $855", "dinner: La Ola"), t2=_message("flights held")))
    await rig.arm()
    rig.triggers(rig.notify_at("alert", "2026-11-10T08:00:00Z", "Wingloft: fare drop", channel="Gmail"))
    await rig.run(rig.step(), rig.context(user=DAY))

    [read] = rig.reads
    assert read["time"] == "2026-11-10T12:30:00Z"
    assert [(m["at"], m["text"]) for m in read["assistant_messages"]] == [
        ("2026-11-10T05:00:00Z", "house: blue, $855"), ("2026-11-10T05:00:00Z", "dinner: La Ola"),
        ("2026-11-10T08:00:00Z", "flights held")]


@pytest.mark.asyncio
async def test_a13_the_opening_request_is_in_person_and_in_her_first_you_sent(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("on it")))
    await rig.arm()
    await rig.run(rig.step())

    opening = {"at": "2026-11-09T14:00:00Z", "text": "sort the cdmx weekend"}
    assert rig.setups[0]["opening"] == opening
    assert rig.reads[0]["you_sent"] == [opening]


def _asking_plugin(text_of_read=lambda read: f"read at {read['time']}"):
    class Plugin:
        @staticmethod
        def setup(version, person):
            return {"system_prompt": "you are Linda", "output_format": None}

        @staticmethod
        async def answer(version, person, history, read, ask):
            got = await ask(text_of_read(read))
            return {**(got["structured"] or {}), "raw": got["text"]}

    return Plugin


@pytest.mark.asyncio
async def test_a14_one_user_sim_session_for_the_whole_run(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("house?"), t2=_message("dinner?"), t3=_message("ok")),
              plugin=_asking_plugin(),
              user_texts=['{"outcome": "reply", "message": "blue", "used": "W1"}',
                          '{"outcome": "reply", "message": "La Ola", "used": "W2"}'])
    await rig.arm()
    await rig.run(rig.step())

    contexts = {c["context_id"] for c in rig.user_calls}
    assert len(rig.user_calls) == 3 and len(contexts) == 1 and None not in contexts


@pytest.mark.asyncio
async def test_a15_history_is_never_trimmed(monkeypatch, tmp_path):
    seen = []

    class Plugin:
        @staticmethod
        def setup(version, person):
            return {"system_prompt": None, "output_format": None}

        @staticmethod
        async def answer(version, person, history, read, ask):
            seen.append(json.loads(json.dumps(history)))
            return {"outcome": "reply", "message": f"answer {len(seen)}"}

    plays = {f"t{n}": _message(f"update {n}") for n in range(1, 6)}
    rig = Rig(monkeypatch, tmp_path, agent=_agent(**plays), plugin=Plugin)
    await rig.arm()
    await rig.run(rig.step(max_wakes=6))

    assert [len(h) for h in seen] == [0, 1, 2, 3, 4]
    assert seen[-1][:3] == seen[3][:3] and [h["answer"]["message"] for h in seen[-1]] == [
        "answer 1", "answer 2", "answer 3", "answer 4"]


@pytest.mark.asyncio
async def test_b1_b2_b4_nothing_meant_for_the_agent_reaches_her_at_any_depth(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n == 1:
            rig.schedule("chase the host about the deposit", at="2026-11-09T18:00:00Z")
            return "PRIVATE CLOSING NOTE: Linda should hear about the house."
        if any("Sam: SECRET ALERT" in t for t in texts):
            rig.message("house booked")
            return "PRIVATE second note"
        rig.quiet()
        return "PRIVATE quiet note"

    rig = Rig(monkeypatch, tmp_path, agent=agent, answers=[{"outcome": "reply", "message": "yay"}])
    await rig.arm()
    rig.gw._notifications.configure({"roles": ["default"]})
    rig.triggers(rig.notify_at("alert", "2026-11-09T20:00:00Z", "Sam: SECRET ALERT", channel="noor 30"))
    ctx = await rig.run(rig.step(end_time="2026-11-10T02:00:00Z", final="send me the rundown"))

    assert rig.reads and rig.events(ctx, "nudge")
    seen = json.dumps(rig.reads)
    for meant_for_the_agent in ("SECRET ALERT", "noor 30", ao.NUDGE_TEXT, ao.WAKE_REMINDER_TEXT, ao.MESSAGE_RULES,
                                ao.QUIET_RULE, "last time you'll be woken", "your scheduled check",
                                "chase the host", "PRIVATE", "stay_quiet", "Any other date you see"):
        assert meant_for_the_agent not in seen, meant_for_the_agent


@pytest.mark.asyncio
async def test_b8_ask_sends_exactly_the_text_it_is_given(monkeypatch, tmp_path):
    text = "  It's Mon, Nov 9, 9:00 AM.\n\nNEW FROM YOUR AI ASSISTANT:\n    house?  \n"
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("house?")), plugin=_asking_plugin(lambda read: text))
    await rig.arm()
    await rig.run(rig.step())

    [call] = rig.user_calls
    assert call["parts"] == [{"kind": "text", "text": text}]


@pytest.mark.parametrize("outcome", ["reply", "done"])
@pytest.mark.asyncio
async def test_c2_c4_a_reply_or_a_done_reaches_the_agent_after_her_reply_time(monkeypatch, tmp_path, outcome):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("blue house or green?")),
              answers=[{"outcome": outcome, "message": "blue, thanks!"}])
    await rig.arm()
    ctx = await rig.run(rig.step(end_time="P3D"), rig.context(user=DAY))

    [read] = rig.events(ctx, "user_sim_read")
    assert read["virtual_time"] == "2026-11-09T14:00:00Z" and read["deliver_at"] == "2026-11-09T16:30:00Z"
    assert _woken_by_user(rig)[-1] == "[Mon Nov 9, 4:30 PM UTC] Linda: blue, thanks!"
    assert rig.events(ctx, "wake")[1]["virtual_time"] == "2026-11-09T16:30:00Z"


@pytest.mark.asyncio
async def test_c5_used_and_score_are_recorded_and_never_sent_to_the_agent(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("house?")),
              answers=[{"outcome": "reply", "message": "the blue one", "used": "D1-TAG", "score": 4}])
    await rig.arm()
    ctx = await rig.run(rig.step())

    [read] = rig.events(ctx, "user_sim_read")
    assert (read["used"], read["score"]) == ("D1-TAG", 4)
    sent = json.dumps(rig.sent)
    assert "the blue one" in sent and "D1-TAG" not in sent


@pytest.mark.parametrize("end_time,ends", [(None, True), ("P3D", False)], ids=["one sitting", "multi-day"])
@pytest.mark.asyncio
async def test_done_ends_run_says_whether_done_would_end_the_run(monkeypatch, tmp_path, end_time, ends):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("house booked")))
    await rig.arm()
    await rig.run(rig.step(**({"end_time": end_time} if end_time else {})))

    assert [r["done_ends_run"] for r in rig.reads] == [ends]


@pytest.mark.parametrize("end_time,ends", [(None, True), ("P3D", False)], ids=["one sitting", "multi-day"])
@pytest.mark.asyncio
async def test_setup_s_person_says_whether_done_ends_the_run(monkeypatch, tmp_path, end_time, ends):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    await rig.run(rig.step(**({"end_time": end_time} if end_time else {})))

    [person] = rig.setups
    assert person["done_ends_run"] is ends
