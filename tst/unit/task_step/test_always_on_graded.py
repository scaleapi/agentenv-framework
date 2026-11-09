"""What graders read of an always_on run: the graded response (the user's whole chat from the agent, with times), the
conversation (what the user saw: her messages and the agent's message_user texts, her last word, an end marker), and
error_type on a run whose agent turn failed or timed out, so verifiers skip it."""

from __future__ import annotations

import pytest

from agent_env.a2a_agent import conversation_store
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.verifiers.agent_prompt_response_verifier import (
    AgentPromptResponseVerifierTaskStep,
)
from agent_env.task_step.task_steps.verifiers.run_context import load_run_context
from tst.unit.task_step.always_on_harness import ROLE, Fail, Rig

pytestmark = pytest.mark.usefixtures("local_stores")


def _agent(**plays):
    def agent(n, texts, rig):
        play = plays.get(f"t{n}")
        if play is None:
            rig.quiet()
            return "PRIVATE nothing to report"
        return play(rig)
    return agent


def _message(*texts, closing="PRIVATE closing note"):
    def play(rig):
        for text in texts:
            rig.message(text)
        return closing
    return play


async def _turns(ctx):
    run = await load_run_context(ctx, ctx.prompt_responses[0])
    return [(t.message.text, t.reply.text if t.reply else None) for t in run.turns]


# ---- the graded response -------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_graded_response_is_every_message_user_text_in_order_with_its_time(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("blue house or green?", "flights: two options"),
                                                  t2=_message("booked the blue one")),
              answers=[{"outcome": "reply", "message": "blue"}])
    await rig.arm()
    rig.triggers(rig.notify_at("sam", "2026-11-10T14:00:00Z", "Sam: shh", channel="noor 30"))
    ctx = await rig.run(rig.step())

    [response] = ctx.prompt_responses
    assert response.response == ("[Mon Nov 9, 2:00 PM UTC] blue house or green?\n\n"
                                 "[Mon Nov 9, 2:00 PM UTC] flights: two options\n\n"
                                 "[Mon Nov 9, 2:00 PM UTC] booked the blue one")
    assert "PRIVATE" not in response.response and response.error_type is None
    verifier = AgentPromptResponseVerifierTaskStep("check", None, prompt_id="ask", verifier_id="words", criteria=[
        {"type": "response_contains", "needles": ["blue house or green?", "booked the blue one"]}])
    await verifier.execute(ctx)
    assert ctx.metadata["verifications"]["words"]["score"] == 1


# ---- the conversation graders read ---------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_conversation_is_what_she_saw_her_messages_and_the_agent_s_message_user_texts(monkeypatch,
                                                                                                tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("blue house or green?"), t2=_message("booked it"),
                                                  t4=_message("Sam says she suspects nothing")),
              answers=[{"outcome": "reply", "message": "blue"}, {"outcome": "no_reply", "message": ""},
                       {"outcome": "reply", "message": "great"}])
    await rig.arm()
    rig.triggers(rig.notify_at("sam", "2026-11-10T14:00:00Z", "Sam: shh", channel="noor 30"),
                 rig.notify_at("ask", "2026-11-10T15:00:00Z", "any news from Sam?", to=ao.USER_SAYS))
    ctx = await rig.run(rig.step())

    turns = await _turns(ctx)
    assert turns[:3] == [
        ("[Mon Nov 9, 2:00 PM UTC] Linda: sort the cdmx weekend", "[Mon Nov 9, 2:00 PM UTC] blue house or green?"),
        ("[Mon Nov 9, 2:00 PM UTC] Linda: blue", "[Mon Nov 9, 2:00 PM UTC] booked it"),
        ("[Tue Nov 10, 3:00 PM UTC] Linda: any news from Sam?",
         "[Tue Nov 10, 3:00 PM UTC] Sam says she suspects nothing"),
    ]
    seen = repr(rig.chat(ctx))
    for never_shown in ("PRIVATE", "Sam: shh", "noor 30", ao.WAKE_REMINDER_TEXT, ao.MESSAGE_RULES,
                        "Any other date you see"):
        assert never_shown not in seen, never_shown


@pytest.mark.asyncio
async def test_messages_sent_on_several_wakes_before_she_answers_are_one_reply_in_order(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("on it"), t2=_message("Sam is in")),
              answers=[{"outcome": "no_reply", "message": ""}, {"outcome": "no_reply", "message": ""}])
    await rig.arm()
    rig.triggers(rig.notify_at("sam", "2026-11-10T14:00:00Z", "Sam: I'm in", channel="noor 30"))
    ctx = await rig.run(rig.step())

    [first, *_] = await _turns(ctx)
    assert first[1] == "[Mon Nov 9, 2:00 PM UTC] on it\n[Tue Nov 10, 2:00 PM UTC] Sam is in"
    assert [p["metadata"]["agentenv"]["wake"] for p in rig.chat(ctx)[1]["parts"]] == [1, 2]


# ---- her last word and the end marker (FD-5369 4 and 6) ------------------------------------------------------------

@pytest.mark.asyncio
async def test_her_done_reaches_the_conversation(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("all set"), t2=_message("glad to help!")),
              answers=[{"outcome": "done", "message": "thanks!"}])
    await rig.arm()
    ctx = await rig.run(rig.step())

    turns = await _turns(ctx)
    assert turns[1] == ("[Mon Nov 9, 2:00 PM UTC] Linda: thanks!", "[Mon Nov 9, 2:00 PM UTC] glad to help!")
    assert rig.chat(ctx)[2]["parts"][0]["metadata"]["agentenv"]["done"] is True
    assert turns[-1][0].startswith("[End of run · Mon Nov 9, 2:00 PM UTC] Linda signed off.")


@pytest.mark.asyncio
async def test_a_blank_done_is_shown_as_her_reaction(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("all set")),
              answers=[{"outcome": "done", "message": ""}])
    await rig.arm()
    ctx = await rig.run(rig.step())

    [last] = rig.chat(ctx)[-1:]
    assert last["role"] == "user"
    assert last["texts"][0] == "(Linda read it and signed off without a message.)"


@pytest.mark.asyncio
async def test_her_reply_to_the_wrap_up_and_the_end_marker_close_the_conversation(monkeypatch, tmp_path):
    def recap(n, texts, rig):
        if ao.FINAL_REMINDER.format(first="Linda") in texts:
            rig.message("RESULT: all booked")
        rig.quiet()
        return "PRIVATE"

    rig = Rig(monkeypatch, tmp_path, agent=recap, answers=[{"outcome": "reply", "message": "you're the best"}])
    await rig.arm()
    ctx = await rig.run(rig.step(end_time="P2D", final="send me the rundown"))

    turns = await _turns(ctx)
    assert turns[-2] == ("[Wed Nov 11, 2:00 PM UTC] Linda: send me the rundown",
                         "[Wed Nov 11, 2:00 PM UTC] RESULT: all booked")
    last = rig.chat(ctx)[-1]
    assert last["texts"] == [
        "[Wed Nov 11, 2:00 PM UTC] Linda: you're the best (sent after the assistant's last turn; it never saw this)",
        "[End of run · Wed Nov 11, 2:00 PM UTC] The run reached its end time. (A note from the run, not a message "
        "from Linda.)"]
    assert last["parts"][-1]["metadata"]["agentenv"] == {"source": "end_marker", "end_reason": "end_time",
                                                         "at": "2026-11-11T14:00:00Z"}


@pytest.mark.asyncio
async def test_a_run_cut_off_by_its_wake_cap_says_so_in_the_conversation(monkeypatch, tmp_path):
    def chatty(n, texts, rig):
        rig.message(f"update {n}")
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=chatty, answers=[{"outcome": "reply", "message": "ok"}] * 5)
    await rig.arm()
    ctx = await rig.run(rig.step(max_wakes=2))

    last = rig.chat(ctx)[-1]
    assert last["texts"][-1].startswith("[End of run · Mon Nov 9, 2:00 PM UTC] The run was cut off at its cap of 2 "
                                        "wakes.")
    assert "Linda: ok (sent after the assistant's last turn" in last["texts"][0]


@pytest.mark.asyncio
async def test_messages_queued_before_the_run_are_not_the_run_s(monkeypatch, tmp_path):
    """Another role's messages stay queued for whoever takes them; what the agent's own role left before the run is
    recorded as such; neither reaches her or the graded chat."""
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("on it")))
    await rig.arm()
    chat = rig.gw._user_chat
    chat.configure({"roles": ["other", ROLE], "stay_quiet": True})
    chat.send("from another assistant", role="other")
    chat.send("left from an earlier step", role=ROLE)
    chat.quiet(role=ROLE)
    ctx = await rig.run(rig.step())

    [response] = ctx.prompt_responses
    assert response.response == "[Mon Nov 9, 2:00 PM UTC] on it"
    assert [m["text"] for m in rig.reads[0]["assistant_messages"]] == ["on it"]
    assert [(m["from"], m["text"]) for m in chat.state()["pending"]] == [("other", "from another assistant")]
    [before] = rig.events(ctx, "user_chat_before_run")
    assert [m["text"] for m in before["messages"]] == ["left from an earlier step"]
    assert before["stay_quiet_calls"] == 1


@pytest.mark.asyncio
async def test_a_take_refused_before_the_run_still_closes_the_conversation_and_records_the_run(monkeypatch, tmp_path):
    """The take before the run comes after the graded conversation is made: when the gateway refuses it, the run still
    leaves its record and prompt response, and the conversation is closed, not left active."""
    rig = Rig(monkeypatch, tmp_path)
    real_take = ao._Run._take_messages
    refused: list[int] = []

    async def refused_once(run):
        if not refused:  # the take before the run, not finish()'s
            refused.append(1)
            raise RuntimeError("the gateway refused the take")
        return await real_take(run)

    monkeypatch.setattr(ao._Run, "_take_messages", refused_once)
    await rig.arm()
    ctx = rig.context()
    with pytest.raises(RuntimeError, match="the gateway refused the take"):
        await rig.run(rig.step(), ctx)

    assert rig.sent == []
    assert rig.record(ctx)["end_reason"] == "step_error"
    [response] = ctx.prompt_responses
    assert response.error_type == "step_error"
    assert conversation_store.get_conversation(ctx.metadata["a2a_conversations"]["week"])["status"] == "closed"


# ---- failed turns are errors -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("play,error_type,end_reason", [
    (lambda rig: Fail("the CLI crashed"), "agent_turn_failed", "agent_turn_failed"),
    (lambda rig: Fail("sandbox gone", error_type="infra_error"), "infra_error", "agent_turn_failed"),
    (lambda rig: (_ for _ in ()).throw(TimeoutError("no answer")), "agent_turn_timeout", "agent_turn_timeout"),
    (lambda rig: Fail("stopped", state="canceled"), "agent_turn_canceled", "agent_turn_failed"),
    (lambda rig: Fail("won't do that", state="rejected"), "agent_turn_rejected", "agent_turn_failed"),
], ids=["failed", "failed with the agent's error type", "timed out", "canceled", "rejected"])
@pytest.mark.asyncio
async def test_a_failed_or_timed_out_turn_sets_error_type_and_the_verifier_skips_the_run(monkeypatch, tmp_path, play,
                                                                                         error_type, end_reason):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=_message("on it"), t2=play),
              answers=[{"outcome": "reply", "message": "great"}])
    await rig.arm()
    ctx = await rig.run(rig.step())

    [response] = ctx.prompt_responses
    assert (response.error_type, rig.record(ctx)["end_reason"]) == (error_type, end_reason)
    assert rig.record(ctx)["error_type"] == error_type
    verifier = AgentPromptResponseVerifierTaskStep("check", None, prompt_id="ask", verifier_id="words",
                                                   criteria=[{"type": "response_contains", "needles": ["on it"]}])
    await verifier.execute(ctx)
    assert ctx.metadata["verifications"]["words"]["results"][0]["id"] == "prompt_error"


@pytest.mark.parametrize("state", ["failed", "canceled", "rejected"])
@pytest.mark.asyncio
async def test_a_nudge_turn_that_doesn_t_complete_sets_error_type_too(monkeypatch, tmp_path, state):
    rig = Rig(monkeypatch, tmp_path, agent=_agent(t1=lambda rig: "PRIVATE all done", t2=lambda rig: Fail(state=state)))
    await rig.arm()
    ctx = await rig.run(rig.step())

    [nudge] = rig.events(ctx, "nudge")
    assert nudge["state"] == state and len(rig.sent) == 2
    [response] = ctx.prompt_responses
    assert (response.error_type, rig.record(ctx)["end_reason"]) == (f"agent_turn_{state}", "agent_turn_failed")
