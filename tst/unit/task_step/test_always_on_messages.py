"""The always_on step's message rules: the agent reaches the user only through message_user, says "nothing to tell"
with stay_quiet, gets one nudge when a wake ends with closing text and neither, and every wake ends with a reminder.
All on unless a task turns them off; none of it reaches the user sim."""

from __future__ import annotations

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import ROLE, Rig

pytestmark = pytest.mark.usefixtures("local_stores")


def _turns(*plays):
    """An agent that plays ``plays[n-1]`` on its nth turn, then says "ok"."""
    def agent(n, texts, rig):
        return plays[n - 1](rig) if n <= len(plays) else "ok"
    return agent


def _send(text, closing="Sent."):
    return lambda rig: (rig.message(text), closing)[1]


def _quiet(closing="Nothing new."):
    return lambda rig: (rig.quiet(), closing)[1]


def _says(closing):
    return lambda rig: closing


@pytest.mark.asyncio
async def test_stay_quiet_and_the_nudge_are_on_and_the_wake_reminder_off_by_default(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_turns(_quiet()))
    await rig.arm()
    step = rig.step()
    ctx = await rig.run(step)

    assert (step.stay_quiet, step.nudge, step.wake_reminder) == (True, True, False)
    assert rig.record(ctx)["settings"].items() >= {"stay_quiet": True, "nudge": True, "wake_reminder": False}.items()
    assert not any(ao.WAKE_REMINDER_TEXT in texts for texts in rig.sent)
    assert rig.during[0]["stay_quiet"] and ROLE in rig.during[0]["user_chat_roles"]


@pytest.mark.asyncio
async def test_a_task_that_turns_the_wake_reminder_on_ends_each_wake_with_it(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_turns(_send("two houses, which one?"), _quiet()),
              answers=[{"outcome": "reply", "message": "the blue one"}])
    await rig.arm()
    await rig.run(rig.step(wake_reminder=True))

    assert len(rig.sent) == 2
    assert all(texts[-1] == ao.WAKE_REMINDER_TEXT for texts in rig.sent)
    assert rig.parts[0][-1]["metadata"] == {"agentenv": {"source": "reminder"}}


@pytest.mark.asyncio
async def test_closing_text_with_neither_tool_gets_one_nudge_recorded_and_counted(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_turns(_says("Found the house; Linda should pick."), _send("pick one?")))
    await rig.arm()
    ctx = await rig.run(rig.step())

    assert rig.sent[1] == [ao.NUDGE_TEXT]
    [nudge] = rig.events(ctx, "nudge")
    assert nudge["wake"] == 1 and nudge["closing_text"] == "Found the house; Linda should pick."
    assert nudge["outcome"] == "message_user" and nudge["plain_reply_after"] == "Sent."
    counts = rig.record(ctx)["counts"]
    assert counts["nudges"] == 1 and counts["nudge_outcomes"] == {"message_user": 1}
    assert [m["text"] for m in rig.events(ctx, "message_to_user")] == ["pick one?"]


@pytest.mark.asyncio
async def test_a_nudge_the_agent_ignores_is_recorded_and_not_repeated(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_turns(_says("All done for tonight."), _says("Okay.")))
    await rig.arm()
    ctx = await rig.run(rig.step())

    assert len(rig.sent) == 2
    [nudge] = rig.events(ctx, "nudge")
    assert nudge["outcome"] == "neither" and nudge["plain_reply_after"] == "Okay."


@pytest.mark.parametrize("play", [_send("booked the house"), _quiet()], ids=["message_user", "stay_quiet"])
@pytest.mark.asyncio
async def test_no_nudge_after_either_tool(monkeypatch, tmp_path, play):
    rig = Rig(monkeypatch, tmp_path, agent=_turns(play))
    await rig.arm()
    ctx = await rig.run(rig.step())

    assert ao.NUDGE_TEXT not in [t for texts in rig.sent for t in texts]
    assert rig.events(ctx, "nudge") == []


@pytest.mark.parametrize("closing", ["", "   ", "(The assistant's reply was empty.)"],
                         ids=["none", "blank", "harness placeholder"])
@pytest.mark.asyncio
async def test_no_nudge_on_an_empty_reply(monkeypatch, tmp_path, closing):
    rig = Rig(monkeypatch, tmp_path, agent=_turns(_says(closing)))
    await rig.arm()
    ctx = await rig.run(rig.step())

    assert len(rig.sent) == 1 and rig.events(ctx, "nudge") == []


@pytest.mark.asyncio
async def test_a_task_can_turn_each_off(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_turns(_says("Updated all quantities.")))
    await rig.arm()
    ctx = await rig.run(rig.step(stay_quiet=False, nudge=False, wake_reminder=False))

    assert len(rig.sent) == 1 and ao.WAKE_REMINDER_TEXT not in rig.sent[0]
    assert rig.events(ctx, "nudge") == [] and "nudges" not in rig.record(ctx)["counts"]
    assert not rig.gw._user_chat.stay_quiet and "stay_quiet" not in await rig.tools()


def test_off_survives_a_round_trip():
    step = ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt_id="p", agent_name="a",
                               stay_quiet=False, nudge=False, wake_reminder=False)
    again = ao.AlwaysOnTaskStep.from_dict(step.to_dict())
    assert (again.stay_quiet, again.nudge, again.wake_reminder) == (False, False, False)
    assert ao.AlwaysOnTaskStep.from_dict({**step.to_dict(), "stay_quiet": None}).stay_quiet is True


@pytest.mark.asyncio
async def test_the_rules_follow_the_task_system_prompt(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_turns(_quiet()))
    await rig.arm()
    run = ao._Run(rig.step(system_prompt="You are Linda's assistant."), rig.context())
    assert run._system_prompt() == f"You are Linda's assistant.\n\n{ao.MESSAGE_RULES} {ao.QUIET_RULE}"
    run = ao._Run(rig.step(system_prompt="You are Linda's assistant.", stay_quiet=False), rig.context())
    assert run._system_prompt() == f"You are Linda's assistant.\n\n{ao.MESSAGE_RULES}"


@pytest.mark.asyncio
async def test_without_a_system_prompt_the_first_wake_opens_with_the_rules(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_turns(_quiet()))
    await rig.arm()
    await rig.run(rig.step())

    assert rig.sent[0][0] == f"{ao.MESSAGE_RULES} {ao.QUIET_RULE}"


@pytest.mark.asyncio
async def test_neither_the_nudge_nor_the_reminder_reaches_the_user_sim(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path,
              agent=_turns(_says("Linda should hear about the house."), _send("blue or green house?"), _quiet()),
              answers=[{"outcome": "reply", "message": "green"}])
    await rig.arm()
    await rig.run(rig.step())

    [read] = rig.reads
    assert [m["text"] for m in read["assistant_messages"]] == ["blue or green house?"]
    assert [s["text"] for s in read["you_sent"]] == ["sort the cdmx weekend"]
    seen = repr(rig.reads)
    assert ao.NUDGE_TEXT not in seen and ao.WAKE_REMINDER_TEXT not in seen and ao.MESSAGE_RULES not in seen
