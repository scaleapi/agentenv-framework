"""The user sim's rating of the run (FD-5491), a step of its own: rate_user_satisfaction runs after an always_on or
prompt_agent run and asks the user sim once, in the session the run used, to rate the job 1-5 with a reason, by the
step's satisfaction lines. The lines are never in the persona or the user sim's prompt, the run steps are as they were
without a rating, and the rating is recorded, never delivered, and never fails the task."""

from __future__ import annotations

import asyncio
import copy
import json
from pathlib import Path

import pytest

from agent_env.a2a_agent import conversation_store
from agent_env.store import get_config
from agent_env.task.task import Task
from agent_env.task_step.registry import _builtin_registry
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent import rate_user_satisfaction as rus
from agent_env.task_step.task_steps.agent import user_sim_versions as us
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import persona_problems
from agent_env.task_step.task_steps.agent.rate_user_satisfaction import RateUserSatisfactionTaskStep
from agent_env.task_step.task_steps.agent.user_sim_common import SATISFACTION_KEY
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from tst.unit.task_step.always_on_harness import Fail, Rig
from tst.unit.task_step.test_deploy_user_sim_agent import DESCRIPTION, HANNAH, _always_on, _Deploy, _rig_context
from tst.unit.task_step.test_deploy_user_sim_agent import _step as _deploy_step
from tst.unit.task_step.test_prompt_agent_built_in_user_sim import AGENT, USER, Peers, _context, _step, said, text

pytestmark = pytest.mark.usefixtures("local_stores")

DATA = Path(__file__).parents[2] / "data" / "user_sim"
PERSON = json.loads((DATA / "hannah_person.json").read_text())
PROMPT = (DATA / "hannah_system_prompt.txt").read_text()
SATISFIED = ["5, perfect: flights, a step-free place a walk from you, and both dinners booked, with nothing you had "
             "to chase.",
             "3, fine: all booked, but you had to answer a question your rules already settled.",
             "1, you'd have done it yourself: a dinner or the place is still not booked at the end.",
             "Being asked for things that are in Mom's email bothers you."]
LINDA_LINES = ["5: the house and the dinner booked.", "1: nothing booked."]
NOTICE = ao.FINAL_REMINDER.format(first="Linda")


def rated(score, message="all booked, no chasing", used="S1", outcome="done") -> dict:
    return {"outcome": outcome, "score": score, "message": message, "used": used}


def _rater(**fields) -> RateUserSatisfactionTaskStep:
    return RateUserSatisfactionTaskStep.from_dict({"id": "rate", "type": "rate_user_satisfaction",
                                                   "satisfaction": LINDA_LINES, "poll_interval_seconds": 0, **fields})


def _asker(*answers):
    sent = []

    async def ask(text):
        sent.append(text)
        got = answers[len(sent) - 1]
        if isinstance(got, dict) and "error" in got:
            return got
        return {"text": got if isinstance(got, str) else json.dumps(got),
                "structured": got if isinstance(got, dict) else None, "trajectory_url": None}

    return ask, sent


NO_UNREAD = {"time": "2026-11-24T05:00:00Z", "assistant_messages": []}


# ---- the user sim's prompt and persona: no satisfaction lines, ever ----------------------------------------------------
def test_the_user_sims_prompt_is_the_tested_one_and_never_has_s_lines():
    prompt = us.setup("ours", PERSON)["system_prompt"]
    assert prompt == PROMPT and "HOW YOU'D RATE" not in prompt and "  S1" not in prompt
    assert all(key != "satisfaction" for key, _, _ in us.SECTIONS)


@pytest.mark.asyncio
async def test_with_a_rating_step_the_deployed_prompt_is_still_the_tested_one(monkeypatch):
    deploy = _Deploy(monkeypatch)
    step = _deploy_step()
    Task("t", None, [step, PromptAgentTaskStep(id="talk", version=None, prompt=PERSON["opening"]["text"],
                                               agent_name="solver", user_agent_name="hannah"),
                     _rater(satisfaction=SATISFIED)])
    await deploy.run(step)
    assert deploy.fields("system_prompt") == [PROMPT]


def test_a_persona_with_satisfaction_lines_is_refused_and_pointed_at_the_step():
    person = copy.deepcopy(PERSON)
    person["description"]["satisfaction"] = SATISFIED
    [problem] = [p for p in us.problems(person) if "unknown sections" in p]
    assert "satisfaction goes on the rate_user_satisfaction step" in problem
    with pytest.raises(ValueError, match="rate_user_satisfaction"):
        _deploy_step(persona={**HANNAH, "description": person["description"]})
    assert any("satisfaction" in p for p in persona_problems({**HANNAH, "description": person["description"]}))


# ---- the ask ------------------------------------------------------------------------------------------------------------
def test_the_rating_ask_word_for_word():
    read = {"time": "2026-11-24T05:00:00Z", "assistant_messages": [
        {"at": "2026-11-24T04:55:00Z", "text": "All set: flights, the Castro Street flat, both dinners.\n\nEnjoy!"}]}
    assert us.render_rating(PERSON, SATISFIED, read) == (
        "It's Mon, Nov 23, 9:00 PM. The job is over: your AI assistant won't see this, or anything you write from now "
        "on.\n"
        "\nNEW FROM YOUR AI ASSISTANT (you haven't read this yet):\n"
        "  [8:55 PM]\n"
        "    All set: flights, the Castro Street flat, both dinners.\n"
        "\n"
        "    Enjoy!\n"
        "\nHOW YOU'D RATE THE JOB\n"
        "  S1  5, perfect: flights, a step-free place a walk from you, and both dinners booked, with nothing you had to "
        "chase.\n"
        "  S2  3, fine: all booked, but you had to answer a question your rules already settled.\n"
        "  S3  1, you'd have done it yourself: a dinner or the place is still not booked at the end.\n"
        "  S4  Being asked for things that are in Mom's email bothers you.\n"
        "\nAs Hannah, rate how the job went, from everything you've seen: 5 if it was done perfectly, 3 if it wasn't "
        "perfect but was fine, 1 if you'd have done it yourself; 2 and 4 are in between. HOW YOU'D RATE THE JOB says "
        "what those mean for this job. Something you were never told was done doesn't count as done.\n"
        'In "message", give your reason in one line. In "used", write the tag of the line it rests on, or "none". '
        'Answer in JSON: {"outcome": "done", "score": 1 | 2 | 3 | 4 | 5, "message": "...", "used": "..."}')
    no_unread = us.render_rating(PERSON, SATISFIED, NO_UNREAD)
    assert "NEW FROM" not in no_unread and no_unread.startswith("It's Mon, Nov 23, 9:00 PM. The job is over")
    assert "\n\nHOW YOU'D RATE THE JOB\n  S1  5, perfect" in no_unread


@pytest.mark.asyncio
async def test_a_rating_is_asked_once_and_parsed():
    ask, sent = _asker(rated(4, "  all booked, but it asked about the dates  ", "S2"))
    got = await us.rate("ours", PERSON, SATISFIED, NO_UNREAD, ask)
    assert len(sent) == 1 and sent[0] == got["input"]
    assert (got["score"], got["reason"], got["used"], got["schema_errors"], got["retried"], got["turns"]) == (
        4, "all booked, but it asked about the dates", "S2", [], False, 1)


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
    got = await us.rate("ours", PERSON, SATISFIED, NO_UNREAD, ask)
    assert sent[1] == us.RATING_RETRY and got["score"] == 4 and got["retried"] is True and got["turns"] == 2
    assert got["first_answer"] == {"raw": "a solid 4", "schema_errors": ["the answer is not a JSON object"]}

    ask, sent = _asker("a solid 4", rated(9))
    got = await us.rate("ours", PERSON, SATISFIED, NO_UNREAD, ask)
    assert len(sent) == 2 and got["score"] is None
    assert got["schema_errors"] == ["first answer: the answer is not a JSON object",
                                    "answer after the retry: score 9 is not one of [1, 2, 3, 4, 5]"]


@pytest.mark.asyncio
async def test_a_user_sim_error_on_the_rating_is_not_retried():
    ask, sent = _asker({"text": None, "structured": None, "error": "timeout"})
    got = await us.rate("ours", PERSON, SATISFIED, NO_UNREAD, ask)
    assert len(sent) == 1 and got["score"] is None and got["retried"] is False


# ---- the step's settings and the load check -----------------------------------------------------------------------------
@pytest.mark.parametrize("lines", ["5: all booked", [], ["5: all booked", " "], [5, 3, 1], None])
def test_bad_satisfaction_lines_are_refused(lines):
    with pytest.raises(ValueError, match="satisfaction must be a list of non-empty lines"):
        _rater(satisfaction=lines)


@pytest.mark.parametrize("fields, problem", [
    ({"timeout_seconds": 0}, "timeout_seconds must be a number of seconds, more than 0"),
    ({"timeout_seconds": True}, "timeout_seconds must be a number of seconds"),
    ({"run_step_id": " "}, "run_step_id must name"),
])
def test_bad_settings_are_refused(fields, problem):
    with pytest.raises(ValueError, match=problem):
        _rater(**fields)


def test_the_step_round_trips_is_registered_and_never_fails_the_task_by_default():
    step = _rater(run_step_id="week", depends_on=["week"])
    again = RateUserSatisfactionTaskStep.from_dict(step.to_dict())
    assert again.to_dict() == step.to_dict()
    assert (again.satisfaction, again.run_step_id, again.fail_task_on_error) == (LINDA_LINES, "week", False)
    assert again.timeout_seconds == ao.AlwaysOnTaskStep("w", None, user_sim="u", prompt_id="p").read_cap_seconds()
    assert _builtin_registry()["rate_user_satisfaction"] is RateUserSatisfactionTaskStep


def _sim():
    return _deploy_step(agent_name="user-sim")


def _talk(**kw):
    return PromptAgentTaskStep(id="talk", version=None, prompt="plan it", agent_name="solver",
                               **{"user_agent_name": "user-sim", **kw})


def test_the_only_run_step_is_rated_by_default_and_a_named_one_is_checked():
    Task("t", None, [_sim(), _always_on(end_time="P3D"), _rater()])
    Task("t", None, [_sim(), _always_on(end_time="P3D"), _rater(run_step_id="week", depends_on=["week"])])
    Task("t", None, [_sim(), _talk(), _rater()])


@pytest.mark.parametrize("steps, problem", [
    (lambda: [_sim(), _rater()], "the task has no always_on or prompt_agent step to rate"),
    (lambda: [_sim(), _always_on(end_time="P3D"), _talk(), _rater()],
     "several always_on and prompt_agent steps (week, talk); set run_step_id"),
    (lambda: [_sim(), _always_on(end_time="P3D"), _rater(run_step_id="later")],
     "run_step_id 'later' names no step of the task"),
    (lambda: [_sim(), _always_on(end_time="P3D"), _rater(run_step_id="user")],
     "run_step_id 'user' is a deploy_user_sim_agent step"),
    (lambda: [_sim(), _rater(), _always_on(end_time="P3D")], "comes before step 'week'"),
    (lambda: [_sim(), _always_on(end_time="P3D"), _rater(depends_on=["user"])], "its depends_on must name 'week'"),
    (lambda: [_sim(), _talk(user_agent_name="someone-else"), _rater()],
     "the user of prompt_agent step 'talk' isn't a user sim a deploy_user_sim_agent step deploys"),
    (lambda: [_sim(), _talk(user_a2a_url="http://human.test"), _rater()],
     "the user of prompt_agent step 'talk' isn't a user sim"),
    (lambda: [_sim(), _always_on(end_time="P3D"), _rater(), _rater(id="again")],
     "step 'rate' already rates 'week'; a run is rated once"),
])
def test_a_task_the_step_cant_rate_is_refused_when_it_loads(steps, problem):
    with pytest.raises(ValueError, match="rate_user_satisfaction") as e:
        Task("t", None, steps())
    assert problem in str(e.value)


# ---- after an always_on run ---------------------------------------------------------------------------------------------
def _wraps_up(n, texts, rig):
    if n == 1:
        rig.message("Blue house or green?")
        return "sent"
    if NOTICE in texts:
        rig.message("RESULT: blue house booked, dinner at La Ola")
        return "sent"
    rig.quiet()
    return ""


def _reply(message: str) -> str:
    return json.dumps({"outcome": "reply", "message": message, "used": "W1"})


async def _always_on_run(monkeypatch, tmp_path, *user_texts, agent=_wraps_up, persona=None, **settings):
    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=us, user_texts=user_texts)
    monkeypatch.setattr(rus, "send_and_wait", rig._send_and_wait)
    await rig.arm()
    persona = {"display_name": "Linda Murphy", "description": DESCRIPTION, **(persona or {})}
    ctx = await _rig_context(monkeypatch, rig, persona=persona)
    ctx = await rig.run(_always_on(**{"end_time": "P3D", **settings}), ctx)
    return rig, ctx


def _delivered(rig, ctx) -> str:
    """Everything the agent was sent, the graded chat, the responses and the verifications."""
    return (json.dumps(rig.sent) + json.dumps(rig.chat(ctx)) + json.dumps([r.response for r in ctx.prompt_responses])
            + json.dumps(ctx.metadata.get(ao.HOOK_RECORDS_KEY)) + json.dumps(ctx.metadata.get("verifications")))


@pytest.mark.asyncio
async def test_an_always_on_task_without_the_step_asks_and_records_nothing_more(monkeypatch, tmp_path):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), _reply("amazing, thanks"))
    assert len(rig.user_calls) == 2 and SATISFACTION_KEY not in ctx.metadata
    assert "HOW YOU'D RATE" not in json.dumps(rig.user_calls)


@pytest.mark.asyncio
async def test_after_always_on_one_ask_in_the_runs_session_recorded_and_never_delivered(monkeypatch, tmp_path):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), _reply("amazing, thanks"),
                                    json.dumps(rated(5, "house and dinner booked", "S1")))
    assert len(rig.user_calls) == 2  # the run asks nothing of a rating
    before = _delivered(rig, ctx), len(ctx.prompt_responses), len(rig.timeline(ctx))

    await _rater().execute(ctx)

    assert len(rig.user_calls) == 3
    ask = rig.user_calls[2]["parts"][0]["text"]
    record = rig.record(ctx)
    assert record["user_sim"]["context_id"] == rig.user_calls[0]["context_id"]
    assert len({c["context_id"] for c in rig.user_calls}) == 1  # the run's one user sim session
    assert us.RATING_ASK.format(first="Linda") in ask
    assert "\nHOW YOU'D RATE THE JOB\n  S1  5: the house and the dinner booked.\n  S2  1: nothing booked.\n" in ask
    assert "NEW FROM YOUR AI ASSISTANT (you haven't read" not in ask  # the last read already showed the wrap-up

    rating = ctx.metadata[SATISFACTION_KEY]["week"]
    assert (rating["user_sim"], rating["score"], rating["reason"], rating["used"], rating["rated_by"]) == (
        "user-sim", 5, "house and dinner booked", "S1", "rate")
    assert rating["at"] == rig.events(ctx, "user_sim_read")[-1]["virtual_time"]
    assert (rating["schema_errors"], rating["retried"], rating["user_sim_error"], rating["user_sim_turns"]) == (
        [], False, None, 1)
    assert (rating["context_id"], rating["session_from"], rating["unread_shown"]) == (
        record["user_sim"]["context_id"], "always_on record", 0)
    stored = json.loads(get_config().get_object_store().get(rating["object_url"]))
    assert stored["input"] == ask and stored["satisfaction"] == LINDA_LINES

    # Never delivered: the agent's wakes, the graded chat, the responses, the hook records and the verifications are as
    # the run left them.
    assert (_delivered(rig, ctx), len(ctx.prompt_responses), len(rig.timeline(ctx))) == before
    assert "house and dinner booked" not in _delivered(rig, ctx)
    assert record["counts"]["user_sim_reads"] == 2 and record["counts"]["user_replies"] == 2


@pytest.mark.asyncio
async def test_after_always_on_the_messages_she_never_read_come_with_the_ask(monkeypatch, tmp_path):
    """After max_replies she stops reading, so the wrap-up is unread: it comes with the rating."""
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), json.dumps(rated(4, "booked, mostly")),
                                    persona={"max_replies": 1})
    assert len(rig.user_calls) == 1
    await _rater().execute(ctx)
    ask = rig.user_calls[-1]["parts"][0]["text"]
    assert "NEW FROM YOUR AI ASSISTANT (you haven't read this yet):" in ask
    assert "RESULT: blue house booked, dinner at La Ola" in ask
    assert ask.index("RESULT: blue house") < ask.index("HOW YOU'D RATE THE JOB") < ask.index("As Linda, rate")
    rating = ctx.metadata[SATISFACTION_KEY]["week"]
    assert rating["score"] == 4 and rating["unread_shown"] == 1
    assert rig.record(ctx)["counts"]["user_replies"] == 1


@pytest.mark.asyncio
async def test_after_always_on_the_unread_messages_come_from_the_record_when_its_timeline_cant_be_read(
        monkeypatch, tmp_path):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), json.dumps(rated(4, "booked")),
                                    persona={"max_replies": 1})
    rig.record(ctx)["object_url"] = "file:///nowhere/timeline.json"
    await _rater().execute(ctx)
    assert "RESULT: blue house booked" in rig.user_calls[-1]["parts"][0]["text"]
    assert ctx.metadata[SATISFACTION_KEY]["week"]["score"] == 4


@pytest.mark.asyncio
async def test_a_malformed_answer_gets_one_retry_in_the_same_session(monkeypatch, tmp_path):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), _reply("amazing, thanks"),
                                    "a solid 4", json.dumps(rated(4)))
    await _rater().execute(ctx)
    assert [c["parts"][0]["text"] for c in rig.user_calls[3:]] == [us.RATING_RETRY]
    assert len({c["context_id"] for c in rig.user_calls}) == 1
    rating = ctx.metadata[SATISFACTION_KEY]["week"]
    assert (rating["score"], rating["retried"], rating["user_sim_turns"], rating["schema_errors"]) == (4, True, 2, [])
    stored = json.loads(get_config().get_object_store().get(rating["object_url"]))
    assert stored["first_answer"]["raw"] == "a solid 4"


@pytest.mark.asyncio
@pytest.mark.parametrize("answers, error", [(("not json", "still not json"), None), ((), "timeout")])
async def test_a_rating_that_fails_is_recorded_and_the_step_goes_on(monkeypatch, tmp_path, answers, error):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), _reply("amazing, thanks"), *answers)
    if error:
        async def times_out(url, parts, **kw):
            raise TimeoutError("the user sim took too long")

        monkeypatch.setattr(rus, "send_and_wait", times_out)
    await _rater().execute(ctx)
    rating = ctx.metadata[SATISFACTION_KEY]["week"]
    assert rating["score"] is None
    if error:
        assert rating["user_sim_error"].startswith("timeout")
    else:
        assert rating["retried"] is True and len(rating["schema_errors"]) == 2


@pytest.mark.asyncio
async def test_a_rating_past_its_timeout_is_recorded(monkeypatch, tmp_path):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), _reply("amazing, thanks"))

    async def hangs(url, parts, **kw):
        await asyncio.sleep(5)

    monkeypatch.setattr(rus, "send_and_wait", hangs)
    await _rater(timeout_seconds=0.05).execute(ctx)
    rating = ctx.metadata[SATISFACTION_KEY]["week"]
    assert rating["score"] is None and rating["user_sim_error"].lower().startswith("timeout")


@pytest.mark.asyncio
async def test_an_always_on_run_that_ended_on_an_error_is_not_rated(monkeypatch, tmp_path):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, agent=lambda n, texts, rig: Fail())
    await _rater().execute(ctx)
    assert rig.user_calls == []
    assert ctx.metadata[SATISFACTION_KEY]["week"] == {"user_sim": "user-sim", "score": None, "reason": None,
                                                      "rated_by": "rate",
                                                      "skipped": "the run ended on agent_turn_failed"}


@pytest.mark.asyncio
async def test_a_run_whose_user_sim_agent_is_gone_is_not_rated(monkeypatch, tmp_path):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), _reply("amazing, thanks"))
    ctx.deployed_agents = [a for a in ctx.deployed_agents if a.agent_name != "user-sim"]
    await _rater().execute(ctx)
    assert len(rig.user_calls) == 2
    assert ctx.metadata[SATISFACTION_KEY]["week"]["skipped"] == "the user sim's agent 'user-sim' is no longer deployed"


@pytest.mark.asyncio
async def test_a_run_step_that_failed_and_let_the_task_go_on_is_not_rated(monkeypatch, tmp_path):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), _reply("amazing, thanks"))
    ctx.metadata["failed_steps"] = [{"step_id": "week", "step_type": "always_on", "error": "it broke"}]
    await _rater().execute(ctx)
    assert ctx.metadata[SATISFACTION_KEY]["week"]["skipped"] == "the run step failed: it broke"


@pytest.mark.asyncio
async def test_a_run_step_that_failed_once_and_succeeded_on_its_retry_is_rated(monkeypatch, tmp_path):
    rig, ctx = await _always_on_run(monkeypatch, tmp_path, _reply("blue!"), _reply("amazing, thanks"),
                                    json.dumps(rated(5, "house and dinner booked", "S1")))
    ctx.metadata["failed_steps"] = [{"step_id": "week", "step_type": "always_on", "error": "it broke",
                                     "is_fatal": True, "retried": True}]
    await _rater().execute(ctx)
    rating = ctx.metadata[SATISFACTION_KEY]["week"]
    assert "skipped" not in rating and rating["score"] == 5 and len(rig.user_calls) == 3


@pytest.mark.asyncio
async def test_a_step_with_no_run_before_it_fails():
    from agent_env.task_step.context import StepContextError, TaskStepContext

    with pytest.raises(StepContextError, match="no always_on or prompt_agent step ran before it"):
        await _rater().execute(TaskStepContext())


# ---- after a prompt_agent chat ------------------------------------------------------------------------------------------
def _user_asks(peers) -> list[tuple[str, str]]:
    """Each message the user sim got: its text and its session."""
    return [(r[2]["params"]["message"]["parts"][0]["text"], r[2]["params"]["message"]["contextId"])
            for r in peers.raw if r[1] == f"{USER}/a2a" and r[2]["method"] == "message/send"]


def _hannah_rater(**fields):
    return _rater(run_step_id="talk", satisfaction=SATISFIED, **fields)


@pytest.mark.asyncio
async def test_a_prompt_agent_task_without_the_step_asks_nothing_more(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("no_reply")]})
    ctx = await _context(monkeypatch)
    await _step().execute(ctx)
    assert [r[1] for r in peers.requests(ctx)] == [f"{AGENT}/a2a"] * 2 + [f"{USER}/a2a"] * 2
    assert SATISFACTION_KEY not in ctx.metadata


@pytest.mark.asyncio
async def test_after_a_prompt_agent_chat_one_ask_in_its_session_with_the_reply_she_never_read(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one"), text("glad to help")],
                                USER: [said("done", "perfect, thanks!"), text(json.dumps(rated(5, "booked")))]})
    ctx = await _context(monkeypatch)
    result = await _step().execute(ctx)
    conversation = copy.deepcopy(conversation_store.get_conversation(ctx.metadata["a2a_conversations"]["talk"]))
    responses = [r.to_dict() for r in result.prompt_responses]

    await _hannah_rater().execute(ctx)

    asks = _user_asks(peers)
    assert len(asks) == 2
    (read, read_session), (ask, ask_session) = asks
    assert ask_session == read_session == ctx.metadata["a2a_conversations"]["talk"]
    assert us.RATING_ASK.format(first="Hannah") in ask and "\nHOW YOU'D RATE THE JOB\n  S1  5, perfect" in ask
    assert "NEW FROM YOUR AI ASSISTANT (you haven't read this yet):\n  [" in ask and "glad to help" in ask
    assert "draft one" not in ask  # she read it
    rating = ctx.metadata[SATISFACTION_KEY]["talk"]
    assert (rating["user_sim"], rating["score"], rating["reason"], rating["unread_shown"]) == ("hannah", 5, "booked", 1)
    assert (rating["context_id"], rating["session_from"]) == (read_session, "prompt_agent conversation")
    stored = json.loads(get_config().get_object_store().get(rating["object_url"]))
    assert stored["input"] == ask and stored["score"] == 5
    # Never delivered: nothing more went to the agent, and the conversation, the reads, the responses and the
    # verifications are as the chat left them.
    assert [r for r in peers.raw if r[1] == f"{AGENT}/a2a"][-1][2]["method"] == "tasks/get"
    assert len([r for r in peers.raw if r[1] == f"{AGENT}/a2a"]) == 4
    assert conversation_store.get_conversation(ctx.metadata["a2a_conversations"]["talk"]) == conversation
    assert len(ctx.metadata["user_sim_reads"]["talk"]) == 1
    assert [r.to_dict() for r in ctx.prompt_responses] == responses and "verifications" not in ctx.metadata


@pytest.mark.asyncio
async def test_after_a_prompt_agent_chat_she_read_to_the_end_nothing_is_unread(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [said("no_reply"), text(json.dumps(rated(3)))]})
    ctx = await _context(monkeypatch)
    await _step().execute(ctx)
    await _hannah_rater().execute(ctx)
    assert "NEW FROM" not in _user_asks(peers)[1][0]
    assert ctx.metadata[SATISFACTION_KEY]["talk"]["score"] == 3
    assert ctx.metadata[SATISFACTION_KEY]["talk"]["unread_shown"] == 0


@pytest.mark.asyncio
async def test_after_a_prompt_agent_chat_a_malformed_answer_gets_one_retry(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")],
                                USER: [said("no_reply"), text("a solid 4"), text(json.dumps(rated(4)))]})
    ctx = await _context(monkeypatch)
    await _step().execute(ctx)
    await _hannah_rater().execute(ctx)
    asks = _user_asks(peers)
    assert asks[2][0] == us.RATING_RETRY and len({session for _, session in asks}) == 1
    assert ctx.metadata[SATISFACTION_KEY]["talk"]["score"] == 4 and ctx.metadata[SATISFACTION_KEY]["talk"]["retried"]


@pytest.mark.asyncio
async def test_a_prompt_agent_chat_the_user_sim_broke_is_not_rated(monkeypatch):
    peers = Peers(monkeypatch, {AGENT: [text("draft one")], USER: [("failed", said("reply", "x"))]})
    ctx = await _context(monkeypatch)
    await _step().execute(ctx)
    await _hannah_rater().execute(ctx)
    assert len(_user_asks(peers)) == 1
    rating = ctx.metadata[SATISFACTION_KEY]["talk"]
    assert rating["skipped"] == "the user sim failed: state failed" and rating["score"] is None


@pytest.mark.asyncio
async def test_a_prompt_agent_chat_whose_agent_turn_failed_is_not_rated(monkeypatch):
    Peers(monkeypatch, {AGENT: [("failed", text("it broke"))], USER: []})
    ctx = await _context(monkeypatch)
    with pytest.raises(RuntimeError, match="A2A task failed"):
        await _step().execute(ctx)
    await _hannah_rater().execute(ctx)
    assert ctx.metadata[SATISFACTION_KEY]["talk"]["skipped"] == "the agent's turn failed"
