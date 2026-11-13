"""The user sim's rating of the run (FD-5491): a description's satisfaction lines go in the prompt, tagged S1...; once
the run is over the user sim is asked once to rate it 1-5 with a reason, in always_on after the last read and in
prompt_agent once the chat has ended. The rating is recorded, never delivered, and never fails the run. Without
satisfaction lines nothing changes: the same prompt, the same reads, no rating."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from agent_env.store import get_config
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent import user_sim_versions as us
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import DeployUserSimAgentTaskStep, persona_problems
from agent_env.task_step.task_steps.agent.user_sim_common import SATISFACTION_KEY
from tst.unit.task_step.always_on_harness import USER_SIM, Rig
from tst.unit.task_step.test_deploy_user_sim_agent import DESCRIPTION, HANNAH, _always_on, _rig_context
from tst.unit.task_step.test_deploy_user_sim_agent import _step as _deploy_step
from tst.unit.task_step.test_prompt_agent_built_in_user_sim import AGENT, USER, Peers, _context, _step, said, text
from tst.unit.task_step.test_prompt_agent_user_side import _stored

pytestmark = pytest.mark.usefixtures("local_stores")

DATA = Path(__file__).parents[2] / "data" / "user_sim"
PERSON = json.loads((DATA / "hannah_person.json").read_text())
PROMPT = (DATA / "hannah_system_prompt.txt").read_text()
SATISFIED = ["5, perfect: flights, a step-free place a walk from you, and both dinners booked, with nothing you had "
             "to chase.",
             "3, fine: all booked, but you had to answer a question your rules already settled.",
             "1, you'd have done it yourself: a dinner or the place is still not booked at the end.",
             "Being asked for things that are in Mom's email bothers you."]
NOTICE = ao.FINAL_REMINDER.format(first="Linda")


def _satisfied(person: dict | None = None) -> dict:
    person = copy.deepcopy(person or PERSON)
    person["description"]["satisfaction"] = list(SATISFIED)
    return person


def rated(score, message="all booked, no chasing", used="S1", outcome="done") -> dict:
    return {"outcome": outcome, "score": score, "message": message, "used": used}


def _asker(*answers):
    sent = []

    async def ask(text):
        sent.append(text)
        got = answers[len(sent) - 1]
        if isinstance(got, Exception):
            raise got
        if isinstance(got, dict) and "error" in got:
            return got
        return {"text": got if isinstance(got, str) else json.dumps(got),
                "structured": got if isinstance(got, dict) else None, "trajectory_url": None}

    return ask, sent


# ---- the prompt -------------------------------------------------------------------------------------------------------
def test_without_satisfaction_lines_the_prompt_is_the_tested_one_and_there_is_no_rating():
    assert us.setup("ours", PERSON)["system_prompt"] == PROMPT
    assert us.rates(PERSON) is False and us.rates({"description": "Linda."}) is False
    assert "HOW YOU'D RATE" not in PROMPT and "  S1" not in PROMPT


def test_satisfaction_lines_come_last_tagged_s1_and_nothing_else_in_the_prompt_changes():
    prompt = us.setup("ours", _satisfied())["system_prompt"]
    assert prompt == PROMPT + "\nHOW YOU'D RATE THE JOB (only when asked, once it's over)\n" + "".join(
        f"  S{i:<3}{line}\n" for i, line in enumerate(SATISFIED, 1))
    assert us.setup("ours", _satisfied())["output_format"] == us.OUTPUT_FORMAT  # the same answer format
    assert us.rates(_satisfied()) is True


def test_satisfaction_lines_do_not_change_any_read():
    read = {"time": "2026-11-17T17:05:00Z", "assistant_messages": [{"at": "2026-11-17T17:00:00Z", "text": "Booked."}],
            "phone": [], "you_sent": [], "done_ends_run": False}
    assert us.render(_satisfied(), [], read) == us.render(PERSON, [], read)


# ---- the load check ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("satisfaction, problem", [
    ("5: all booked", "description.satisfaction must be a list of non-empty lines"),
    (["5: all booked", " "], "description.satisfaction must be a list of non-empty lines"),
    ([5, 3, 1], "description.satisfaction must be a list of non-empty lines"),
])
def test_a_bad_satisfaction_section_is_named(satisfaction, problem):
    person = copy.deepcopy(PERSON)
    person["description"]["satisfaction"] = satisfaction
    assert problem in us.problems(person)
    with pytest.raises(ValueError, match="user sim"):
        us.setup("ours", person)
    persona = {**HANNAH, "description": person["description"]}
    assert problem in persona_problems(persona)
    with pytest.raises(ValueError, match="satisfaction must be a list"):
        _deploy_step(persona=persona)


def test_satisfaction_lines_alone_are_not_a_description():
    assert "description is empty: write at least who they are" in us.problems(
        {**PERSON, "description": {"satisfaction": SATISFIED}})


def test_the_deploy_step_takes_a_satisfaction_section_and_records_it_for_the_reads():
    step = _deploy_step(persona={**HANNAH, "description": _satisfied()["description"]})
    assert step.person(_FakeContext())["description"]["satisfaction"] == SATISFIED
    assert DeployUserSimAgentTaskStep.from_dict(step.to_dict()).persona == step.persona


class _FakeContext:
    metadata: dict = {}


# ---- the rating ask ---------------------------------------------------------------------------------------------------
def test_the_rating_ask_word_for_word():
    read = {"time": "2026-11-24T05:00:00Z", "assistant_messages": [
        {"at": "2026-11-24T04:55:00Z", "text": "All set: flights, the Castro Street flat, both dinners.\n\nEnjoy!"}]}
    assert us.render_rating(_satisfied(), read) == (
        "It's Mon, Nov 23, 9:00 PM. The job is over: your AI assistant won't see this, or anything you write from now "
        "on.\n"
        "\nNEW FROM YOUR AI ASSISTANT (you haven't read this yet):\n"
        "  [8:55 PM]\n"
        "    All set: flights, the Castro Street flat, both dinners.\n"
        "\n"
        "    Enjoy!\n"
        "\nAs Hannah, rate how the job went, from everything you've seen: 5 if it was done perfectly, 3 if it wasn't "
        "perfect but was fine, 1 if you'd have done it yourself; 2 and 4 are in between. HOW YOU'D RATE THE JOB says "
        "what those mean for this job. Something you were never told was done doesn't count as done.\n"
        'In "message", give your reason in one line. In "used", write the tag of the line it rests on, or "none". '
        'Answer in JSON: {"outcome": "done", "score": 1 | 2 | 3 | 4 | 5, "message": "...", "used": "..."}')
    no_unread = us.render_rating(_satisfied(), {"time": "2026-11-24T05:00:00Z", "assistant_messages": []})
    assert "NEW FROM" not in no_unread and no_unread.startswith("It's Mon, Nov 23, 9:00 PM. The job is over")


@pytest.mark.asyncio
async def test_a_rating_is_asked_once_and_parsed():
    ask, sent = _asker(rated(4, "  all booked, but it asked about the dates  ", "S2"))
    got = await us.rate("ours", _satisfied(), {"time": "2026-11-24T05:00:00Z", "assistant_messages": []}, ask)
    assert len(sent) == 1 and sent[0] == got["input"]
    assert (got["score"], got["reason"], got["used"], got["schema_errors"], got["retried"], got["turns"]) == (
        4, "all booked, but it asked about the dates", "S2", [], False, 1)


@pytest.mark.asyncio
async def test_without_satisfaction_lines_nothing_is_asked():
    ask, sent = _asker()
    assert await us.rate("ours", PERSON, {"time": "2026-11-24T05:00:00Z", "assistant_messages": []}, ask) is None
    assert sent == []


@pytest.mark.parametrize("raw, error", [
    (rated(7), "score 7 is not one of [1, 2, 3, 4, 5]"),
    (rated(0), "score 0 is not one of [1, 2, 3, 4, 5]"),
    (rated(True), "score must be a whole number"),
    ({"outcome": "done", "message": "fine"}, "missing score"),
    (rated(3, "   "), "message must give the reason"),
    ("a solid 4", "the answer is not a JSON object"),
])
def test_a_malformed_rating_has_no_score_and_says_why(raw, error):
    got = us.parse_rating(raw)
    assert got["score"] is None and error in got["schema_errors"], got


@pytest.mark.asyncio
async def test_a_malformed_rating_gets_one_retry():
    ask, sent = _asker("a solid 4", rated(4))
    got = await us.rate("ours", _satisfied(), {"time": "2026-11-24T05:00:00Z", "assistant_messages": []}, ask)
    assert sent[1] == us.RATING_RETRY and got["score"] == 4 and got["retried"] is True and got["turns"] == 2
    assert got["first_answer"] == {"raw": "a solid 4", "schema_errors": ["the answer is not a JSON object"]}

    ask, sent = _asker("a solid 4", rated(9))
    got = await us.rate("ours", _satisfied(), {"time": "2026-11-24T05:00:00Z", "assistant_messages": []}, ask)
    assert len(sent) == 2 and got["score"] is None
    assert got["schema_errors"] == ["first answer: the answer is not a JSON object",
                                    "answer after the retry: score 9 is not one of [1, 2, 3, 4, 5]"]


@pytest.mark.asyncio
async def test_a_user_sim_error_on_the_rating_is_not_retried():
    ask, sent = _asker({"text": None, "structured": None, "error": "timeout"})
    got = await us.rate("ours", _satisfied(), {"time": "2026-11-24T05:00:00Z", "assistant_messages": []}, ask)
    assert len(sent) == 1 and got["score"] is None and got["retried"] is False


# ---- always_on --------------------------------------------------------------------------------------------------------
LINDA_SATISFIED = {**DESCRIPTION, "satisfaction": ["5: the house and the dinner booked.", "1: nothing booked."]}


def _wraps_up(n, texts, rig):
    if n == 1:
        rig.message("Blue house or green?")
        return "sent"
    if NOTICE in texts:
        rig.message("RESULT: blue house booked, dinner at La Ola")
        return "sent"
    rig.quiet()
    return ""


async def _always_on_run(monkeypatch, tmp_path, *user_texts, description=LINDA_SATISFIED, agent=_wraps_up,
                         persona=None, **settings):
    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=us, user_texts=user_texts)
    await rig.arm()
    persona = {"display_name": "Linda Murphy", "description": description, **(persona or {})}
    ctx = await _rig_context(monkeypatch, rig, persona=persona)
    ctx = await rig.run(_always_on(**{"end_time": "P3D", **settings}), ctx)
    return rig, ctx


def _reply(message: str) -> str:
    return json.dumps({"outcome": "reply", "message": message, "used": "W1"})


@pytest.mark.asyncio
async def test_always_on_rates_once_after_the_last_read_and_records_it_without_delivering_it(monkeypatch, tmp_path):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), _reply("amazing, thanks"),
                                    json.dumps(rated(5, "house and dinner booked", "S1")))

    asks = [c["parts"][0]["text"] for c in rig.user_calls]
    assert len(asks) == 3 and asks[2] == rig.events(ctx, "user_sim_rating")[0]["input"]
    assert asks[2].startswith("It's ") and us.RATING_ASK.format(first="Linda") in asks[2]
    assert "NEW FROM YOUR AI ASSISTANT (you haven't read" not in asks[2]  # the last read already showed the wrap-up
    assert len({c["context_id"] for c in rig.user_calls}) == 1  # the run's one user sim session
    order = [e["event"] for e in rig.timeline(ctx)]
    assert order[-3:] == ["user_sim_read", "user_sim_rating", "end"]
    assert rig.events(ctx, "user_sim_read")[-1]["last_read"] is True

    rating = ctx.metadata[SATISFACTION_KEY]["week"]
    assert (rating["user_sim"], rating["score"], rating["reason"], rating["used"]) == (
        "user-sim", 5, "house and dinner booked", "S1")
    assert rating["at"] == rig.events(ctx, "user_sim_read")[-1]["virtual_time"]
    assert (rating["schema_errors"], rating["retried"], rating["user_sim_error"], rating["user_sim_turns"]) == (
        [], False, None, 1)

    # Never delivered: the agent's wakes, the graded chat, the response and the hook records don't have it.
    everything_sent = json.dumps(rig.sent) + json.dumps(rig.chat(ctx)) + ctx.prompt_responses[-1].response
    assert "house and dinner booked" not in everything_sent
    assert "house and dinner booked" not in json.dumps(ctx.metadata[ao.HOOK_RECORDS_KEY])
    record = rig.record(ctx)
    assert record["counts"]["user_sim_reads"] == 2 and record["counts"]["user_replies"] == 2
    assert record["end_reason"] == "end_time" and record["error_type"] is None
    assert record["settings"]["rating_cap_seconds"] == _always_on().rating_cap_seconds()


@pytest.mark.asyncio
async def test_always_on_without_satisfaction_lines_asks_and_records_nothing_more(monkeypatch, tmp_path):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), _reply("amazing, thanks"),
                                    description=DESCRIPTION)
    assert len(rig.user_calls) == 2 and SATISFACTION_KEY not in ctx.metadata
    assert not rig.events(ctx, "user_sim_rating") and "rating_cap_seconds" not in rig.record(ctx)["settings"]


@pytest.mark.asyncio
async def test_always_on_rates_even_with_no_wrap_up_to_read(monkeypatch, tmp_path):
    def asks_once(n, texts, rig):
        if n == 1:
            rig.message("Blue house or green?")
            return "sent"
        rig.quiet()
        return ""

    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), json.dumps(rated(1, "nothing booked")),
                                    agent=asks_once)
    assert [e["event"] for e in rig.timeline(ctx)][-2:] == ["user_sim_rating", "end"]
    assert len(rig.events(ctx, "user_sim_read")) == 1 and ctx.metadata[SATISFACTION_KEY]["week"]["score"] == 1


@pytest.mark.asyncio
async def test_messages_she_never_read_come_with_the_rating_and_it_is_no_reply(monkeypatch, tmp_path):
    """After max_replies she stops reading, so the wrap-up is unread: it comes with the rating, and the rating doesn't
    count as a reply."""
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), json.dumps(rated(4, "booked, mostly")),
                                    persona={"max_replies": 1})
    assert len(rig.events(ctx, "user_sim_read")) == 1
    ask = rig.user_calls[-1]["parts"][0]["text"]
    assert "NEW FROM YOUR AI ASSISTANT (you haven't read this yet):" in ask
    assert "RESULT: blue house booked, dinner at La Ola" in ask
    rating = ctx.metadata[SATISFACTION_KEY]["week"]
    assert rating["score"] == 4 and rating["unread_shown"] == 1
    assert rig.record(ctx)["counts"]["user_replies"] == 1
    assert rig.record(ctx)["pending_at_end"]["unread_by_user"]["count"] == 1  # a rating isn't a read


@pytest.mark.asyncio
@pytest.mark.parametrize("answer, error", [
    ("not json", None),
    (None, "timeout"),
])
async def test_a_rating_that_fails_is_recorded_and_the_run_ends_as_it_would(monkeypatch, tmp_path, answer, error):
    rig = Rig(monkeypatch, tmp_path, agent=_wraps_up, plugin=us,
              user_texts=(_reply("blue!"), _reply("amazing, thanks"), *([answer, answer] if answer else [])))
    if error:  # the user sim times out on the rating
        real = rig._send_and_wait

        async def times_out(url, parts, **kw):
            if url == USER_SIM and len(rig.user_calls) >= 2:
                raise TimeoutError("the user sim took too long")
            return await real(url, parts, **kw)

        monkeypatch.setattr(ao, "send_and_wait", times_out)
    await rig.arm()
    ctx = await _rig_context(monkeypatch, rig, persona={"display_name": "Linda Murphy", "description": LINDA_SATISFIED})
    ctx = await rig.run(_always_on(end_time="P3D"), ctx)

    rating = ctx.metadata[SATISFACTION_KEY]["week"]
    assert rating["score"] is None
    if error:
        assert rating["user_sim_error"].startswith("timeout")
    else:
        assert rating["retried"] is True and rating["schema_errors"]
    record = rig.record(ctx)
    assert (record["end_reason"], record["error_type"], ctx.prompt_responses[-1].error_type) == (
        "end_time", None, None)


@pytest.mark.asyncio
@pytest.mark.parametrize("breaks, error", [("raise", "RuntimeError: the plug-in broke"), ("hang", "TimeoutError")])
async def test_a_rating_that_raises_or_runs_past_its_cap_is_recorded_and_the_run_ends_as_it_would(
        monkeypatch, tmp_path, breaks, error):
    class Breaks:
        setup, answer, rates = us.setup, us.answer, us.rates

        @staticmethod
        async def rate(version, person, read, ask):
            if breaks == "raise":
                raise RuntimeError("the plug-in broke")
            await asyncio.sleep(5)

    monkeypatch.setattr(ao.AlwaysOnTaskStep, "rating_cap_seconds", lambda step: 0.05)
    rig = Rig(monkeypatch, tmp_path, agent=_wraps_up, plugin=Breaks,
              user_texts=(_reply("blue!"), _reply("amazing, thanks")))
    await rig.arm()
    ctx = await _rig_context(monkeypatch, rig, persona={"display_name": "Linda Murphy", "description": LINDA_SATISFIED})
    ctx = await rig.run(_always_on(end_time="P3D"), ctx)
    assert ctx.metadata[SATISFACTION_KEY]["week"]["user_sim_error"] == error
    assert rig.record(ctx)["end_reason"] == "end_time" and rig.record(ctx)["error_type"] is None


@pytest.mark.asyncio
async def test_a_run_that_ended_on_an_error_is_not_rated(monkeypatch, tmp_path):
    from tst.unit.task_step.always_on_harness import Fail

    rig, ctx = await _always_on_run(monkeypatch, tmp_path, agent=lambda n, texts, rig: Fail())
    assert rig.user_calls == []
    assert ctx.metadata[SATISFACTION_KEY]["week"] == {"user_sim": "user-sim", "score": None, "reason": None,
                                                      "skipped": "the run ended on agent_turn_failed"}


@pytest.mark.asyncio
async def test_the_time_budget_keeps_room_for_the_rating(monkeypatch, tmp_path):
    step = _always_on(end_time="P3D")
    # The default budget is lowered to fit the closing, so with a rating it closes that much sooner.
    assert step.effective_time_budget(rating=True) == step.effective_time_budget() - step.rating_cap_seconds()
    tight = ao.JOB_LIMIT_SECONDS - ao.JOB_HEADROOM_SECONDS - step.wake_reserve_seconds()
    step = _always_on(end_time="P3D", time_budget_seconds=tight)
    with pytest.raises(ValueError, match="can't fit"):
        step.effective_time_budget(rating=True)
    satisfied = _deploy_step(agent_name="user-sim", persona={**HANNAH, "description": _satisfied()["description"]})
    plain = _deploy_step(agent_name="user-sim")
    [problem] = step.task_problems([satisfied, step])
    assert "can't fit" in problem and "with the user sim's rating after the run" in problem
    assert step.task_problems([plain, step]) == []


# ---- prompt_agent -----------------------------------------------------------------------------------------------------
def _rated_text(score, message="booked, no chasing") -> list[dict]:
    return text(json.dumps(rated(score, message)))


def _rating(ctx) -> dict:
    return ctx.metadata[SATISFACTION_KEY]["talk"]


@pytest.mark.asyncio
async def test_prompt_agent_rates_once_the_chat_has_ended_with_the_reply_she_never_read(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("glad to help")],
                                USER: [said("done", "perfect, thanks!"), _rated_text(5)]})
    ctx = await _context(monkeypatch, persona={**HANNAH, "description": _satisfied()["description"]})

    result = await _step().execute(ctx)

    user_asks = [r[2]["params"]["message"]["parts"][0]["text"] for r in peers.requests(ctx)
                 if r[1] == f"{USER}/a2a" and r[2]["method"] == "message/send"]
    assert len(user_asks) == 2 and us.RATING_ASK.format(first="Hannah") in user_asks[1]
    assert "NEW FROM YOUR AI ASSISTANT (you haven't read this yet):\n  [" in user_asks[1]
    assert "glad to help" in user_asks[1]
    rating = _rating(ctx)
    assert (rating["user_sim"], rating["score"], rating["reason"], rating["unread_shown"]) == (
        "hannah", 5, "booked, no chasing", 1)
    stored = json.loads(get_config().get_object_store().get(rating["object_url"]))
    assert stored["input"] == user_asks[1] and stored["score"] == 5
    # Never delivered: not sent to the agent, not in the graded conversation, not a read.
    agent_parts = json.dumps([r for r in peers.requests(ctx) if r[1] == f"{AGENT}/a2a"])
    assert "booked, no chasing" not in agent_parts
    assert "booked, no chasing" not in json.dumps(_stored(ctx, peers)["messages"])
    assert len(ctx.metadata["user_sim_reads"]["talk"]) == 1
    assert (result.prompt_responses[-1].response, result.prompt_responses[-1].error_type) == ("glad to help", None)


@pytest.mark.asyncio
async def test_prompt_agent_shows_nothing_unread_when_she_read_the_last_reply(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("no_reply"), _rated_text(3)]})
    ctx = await _context(monkeypatch, persona={**HANNAH, "description": _satisfied()["description"]})
    await _step().execute(ctx)
    asks = [r for r in peers.requests(ctx) if r[1] == f"{USER}/a2a" and r[2]["method"] == "message/send"]
    assert "NEW FROM" not in asks[1][2]["params"]["message"]["parts"][0]["text"]
    assert _rating(ctx)["score"] == 3 and _rating(ctx)["unread_shown"] == 0


@pytest.mark.asyncio
async def test_prompt_agent_without_satisfaction_lines_asks_nothing_more(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("no_reply")]})
    ctx = await _context(monkeypatch)
    await _step().execute(ctx)
    assert [r[1] for r in peers.requests(ctx)] == [f"{AGENT}/a2a"] * 2 + [f"{USER}/a2a"] * 2
    assert SATISFACTION_KEY not in ctx.metadata


@pytest.mark.asyncio
async def test_prompt_agent_a_rating_that_fails_is_recorded_and_the_run_is_unchanged(monkeypatch):
    Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("no_reply"), ("failed", _rated_text(5))]})
    ctx = await _context(monkeypatch, persona={**HANNAH, "description": _satisfied()["description"]})
    result = await _step().execute(ctx)
    assert _rating(ctx)["score"] is None and _rating(ctx)["user_sim_error"] == "state failed"
    assert (result.prompt_responses[-1].response, result.prompt_responses[-1].error_type) == ("draft one", None)


@pytest.mark.asyncio
async def test_prompt_agent_a_rating_that_raises_is_recorded_and_the_run_is_unchanged(monkeypatch):
    Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("no_reply")]})  # nothing left to answer the rating
    ctx = await _context(monkeypatch, persona={**HANNAH, "description": _satisfied()["description"]})
    result = await _step().execute(ctx)
    assert _rating(ctx)["score"] is None and _rating(ctx)["user_sim_error"]
    assert result.prompt_responses[-1].error_type is None


@pytest.mark.asyncio
async def test_prompt_agent_does_not_rate_a_chat_the_user_sim_broke(monkeypatch):
    Peers(monkeypatch, {AGENT: [text("draft one")], USER: [("failed", said("reply", "x"))]})
    ctx = await _context(monkeypatch, persona={**HANNAH, "description": _satisfied()["description"]})
    await _step().execute(ctx)
    assert _rating(ctx)["skipped"] == "the user sim failed: state failed" and _rating(ctx)["score"] is None
