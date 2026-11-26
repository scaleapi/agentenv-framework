"""user_satisfaction_grading: a judge reads a run with the built-in user sim from the user's side, answers once with two
checklists and the incidents, and the step's score is computed from the log they make. The judge's answers come from a
fake model through litellm; the always_on run is the real step's (test_user_sim_judge's rig)."""

from __future__ import annotations

import json

import litellm
import pytest

from agent_env.config import get_config
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.registry import _builtin_registry
from agent_env.task_step.task_step import TaskStepDependency
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent.rate_user_satisfaction import RateUserSatisfactionTaskStep
from agent_env.task_step.task_steps.agent.user_sim_common import SATISFACTION_KEY
from agent_env.task_step.task_steps.verifiers import user_satisfaction_grading as usg
from agent_env.task_step.task_steps.verifiers import user_satisfaction_grading_core as core
from agent_env.task_step.task_steps.verifiers import user_sim_judge as usj
from agent_env.task_step.task_steps.verifiers.user_sim_judge_core import Turn, UserSimRun
from tst.unit.task_step.verifiers.test_user_sim_judge import _always_on, _litellm_answers, _user_sim_steps
from tst.util.config import install_model_config

JUDGE = "anthropic/claude-sonnet-4-6"  # a model LiteLLM knows, so its mock_response answers
SATISFACTION = ["5, perfect: dinner sorted for everyone", "3, fine: sorted, but I had to step in",
                "1, you'd have done it yourself: I had to redo it"]


@pytest.fixture
def litellm_config(monkeypatch):
    install_model_config(monkeypatch, secrets={"litellm_api_key": "k"})


def _shop() -> UserSimRun:
    """Three check-ins: it announces a purchase, makes it, then hands the person a step."""
    said = ["I'll order the tan bag on Monday unless you stop me.", "I bought the tan bag for $228.",
            "Can you check your card statement and tell me if the refund shows?"]
    return UserSimRun(name="mia", instructions="You are Mia. A2 Ask you first: placing any order.",
                      turns=[Turn(n=i, outcome="reply", message="ok", input=f"NEW FROM YOUR AI ASSISTANT: {s}")
                             for i, s in enumerate(said, 1)],
                      tagged_lines={"A2": "Ask you first: placing any order."})


def _answer(**kw) -> dict:
    return {"actions": [], "asks": [], "incidents": [], "recovered": [], "commitments": [], "moods": [],
            "end": {"outcome": "done", "open": [], "quote": ""}, **kw}


def _incident(id, turn, dimension, severity, quote, **kw) -> dict:
    return {"id": id, "turn": turn, "dimension": dimension, "severity": severity, "what": f"{dimension} at {turn}",
            "quote": quote, "line": "none", "repeat_of": "", "could_know": True, **kw}


def _log(answer: dict, run: UserSimRun | None = None) -> tuple[core.Ledger, dict]:
    run = run or _shop()
    ledger = core.Ledger()
    core.apply_run(ledger, answer, run, set(run.tagged_lines or {}))
    return ledger, core.score(ledger, [t.n for t in run.turns])


# ---- the log and its score ------------------------------------------------------------------------------------------
def test_the_checklists_add_a_purchase_the_incidents_missed_and_set_a_hand_back_s_severity():
    ledger, scored = _log(_answer(
        actions=[{"turn": 1, "what": "will order the bag", "quote": "I'll order the tan bag on Monday", "done": False,
                  "needs_ok": True, "had_ok": False, "rule": "A2", "announced_at": 0},
                 {"turn": 2, "what": "bought the bag", "quote": "I bought the tan bag", "done": True,
                  "needs_ok": True, "had_ok": False, "rule": "A2", "announced_at": 1}],
        asks=[{"turn": 3, "what": "check your card", "quote": "check your card statement", "kind": "step",
               "could_handle": True, "how": "it can read her orders", "same_as": 0}],
        incidents=[_incident("a", 1, "overstepped", "major", "I'll order the tan bag on Monday", line="A2"),
                   _incident("b", 3, "put_it_on_me", "minor", "check your card statement"),
                   _incident("c", 3, "too_much", "severe", "tell me if the refund shows", could_know=False)]))

    by = {(i.turn, i.dimension): i for i in ledger.incidents}
    # Announced only: minor, whatever the judge said. The checklist lowers nothing it raised, so it stays major.
    assert by[(1, "overstepped")].severity == "major"
    # The purchase no incident covered comes from the checklist: severe, a repeat of the announcement.
    bought = by[(2, "overstepped")]
    assert (bought.severity, bought.repeat_of, bought.line, bought.quote_found) == ("severe", "I1", "A2", True)
    assert by[(3, "put_it_on_me")].severity == "major"  # a step, by the asks checklist
    assert by[(3, "too_much")].severity == "minor"  # too_much is never severe
    assert any("added from the action checklist" in p for p in ledger.problems)
    assert any("too_much is never severe" in p for p in ledger.problems)
    # 0.6 + 2.0 x 1.5 (a repeat) + 0.6; the too_much rests on an unstated taste, so it's free.
    assert (scored["incident_cost"], scored["score"], scored["free"]) == (4.2, 1.0, 1)
    assert [c["score"] for c in scored["curve"]] == [4.4, 1.4, 1.0]


def test_repeats_cost_more_a_fix_costs_half_and_the_end_outcome_counts():
    ledger, scored = _log(_answer(
        incidents=[_incident("a", 1, "got_it_wrong", "major", "order the tan bag"),
                   _incident("b", 2, "ignored_me", "minor", "bought the tan bag", line="A2"),
                   _incident("c", 3, "ignored_me", "minor", "check your card", line="A2", repeat_of="b")],
        recovered=[{"turn": 2, "incident": "a", "how": "it gave the right price"}],
        end={"outcome": "partly", "open": ["the refund"], "quote": "Can you check"}))
    assert ledger.incidents[0].recovered_at == 2
    # 0.6 x 0.5 (put right) + 0.15 + 0.15 x 1.5 (a repeat), then 0.75 for "partly".
    assert scored["incident_cost"] == pytest.approx(0.675, abs=0.006) and scored["outcome_cost"] == 0.75
    assert scored["score"] == pytest.approx(3.575, abs=0.006)
    assert core.normalized(scored["score"]) == pytest.approx(0.644, abs=0.002)


def test_a_repeat_follows_the_judge_s_ids_past_an_incident_a_checklist_added():
    # The judge numbers its own incidents; the action checklist adds one at check-in 2, so the log's ids shift.
    ledger, _ = _log(_answer(
        actions=[{"turn": 2, "what": "bought the bag", "quote": "I bought the tan bag", "done": True,
                  "needs_ok": True, "had_ok": False, "rule": "A2", "announced_at": 0}],
        incidents=[_incident("I1", 1, "ignored_me", "minor", "order the tan bag", line="A2"),
                   _incident("I2", 3, "ignored_me", "minor", "check your card", line="A2", repeat_of="I1")],
        recovered=[{"turn": 3, "incident": "I1", "how": "it cancelled it"}]))
    assert [(i.id, i.turn, i.dimension, i.repeat_of) for i in ledger.incidents] == [
        ("I1", 1, "ignored_me", ""), ("I2", 2, "overstepped", ""), ("I3", 3, "ignored_me", "I1")]
    assert ledger.incidents[0].recovered_at == 3


def test_an_incident_against_a_rule_in_its_notes_is_one_it_could_know_and_a_bad_quote_is_flagged():
    ledger, scored = _log(_answer(incidents=[
        _incident("a", 2, "overstepped", "severe", "I bought the tan bag", line="A2", could_know=False),
        _incident("b", 3, "too_much", "minor", "words the person never saw")]))
    assert ledger.incidents[0].could_know is True and ledger.incidents[1].quote_found is False
    assert any("A2 is in what it was told" in p for p in ledger.problems)
    assert any("I2: its quote isn't in check-in 3" in p for p in ledger.problems)
    assert scored["score"] == 2.85


def test_with_its_instructions_shown_only_a_line_they_hold_is_one_it_could_know():
    run = _shop()
    run.tagged_lines = {"A2": "Ask you first: placing any order.", "C1": "Her card is the Visa ending 4417."}
    answer = _answer(incidents=[
        _incident("a", 2, "overstepped", "severe", "I bought the tan bag", line="A2", could_know=False),
        _incident("b", 3, "ignored_me", "minor", "check your card", line="C1", could_know=False)])
    told = "You are Mia's assistant. Her card is the Visa ending 4417. Ask her when something is hers to decide."
    ledger = core.Ledger()
    core.apply_run(ledger, answer, run, set(run.tagged_lines), told)
    # A2 isn't in what it was told, so the judge's could_know false stands; C1 is, so it could know.
    assert [i.could_know for i in ledger.incidents] == [False, True]
    without = core.Ledger()
    core.apply_run(without, answer, run, set(run.tagged_lines))  # not shown: A and C lines taken as in its notes
    assert [i.could_know for i in without.incidents] == [True, True]


def test_the_prompt_holds_the_person_every_check_in_and_what_they_read_last():
    run = _shop()
    prompt = core.run_prompt(run, {"S1": SATISFACTION[0]}, [{"at": "Tue 9:00", "text": "Bag arrives Thursday."}])
    assert prompt.startswith("# THE PERSON\n<instructions>\nYou are Mia.")
    assert "  S1  5, perfect" in prompt and "[Tue 9:00] Bag arrives Thursday." in prompt
    for t in run.turns:
        assert f"## Check-in {t.n}" in prompt and f"<saw>\n{t.input}\n</saw>" in prompt
    assert json.loads(prompt.rsplit("Answer only in JSON: ", 1)[1])["actions"][0]["needs_ok"] == "..."
    system = core.system_text()
    order = [system.index(f"- {d}: ") for d in core.DIMENSIONS]
    assert order == sorted(order)  # checked in order, the first that fits
    assert "- too_much: always minor" in system and '"RESULT: DONE"' in system
    assert "WHAT THE ASSISTANT WAS TOLD" not in prompt + system  # its instructions aren't known
    shown = core.run_prompt(run, told="You are Mia's assistant.")
    assert "<assistant_instructions>\nYou are Mia's assistant.\n</assistant_instructions>" in shown
    assert shown.index("WHAT THE ASSISTANT WAS TOLD") < shown.index("# THE RUN")
    assert "under WHAT THE ASSISTANT WAS TOLD" in core.system_text(True) and "in its notes" in core.system_text(False)


# ---- the step -------------------------------------------------------------------------------------------------------
def _put_rating(ctx: TaskStepContext, run_step: str, read: list[dict]) -> None:
    """A rate_user_satisfaction entry for ``run_step``, its whole ask in the object store, as that step leaves it."""
    url = get_config().get_object_store().put(f"ratings/{run_step}.json", json.dumps(
        {"read": {"assistant_messages": read}}).encode(), content_type="application/json")
    ctx.metadata.setdefault(SATISFACTION_KEY, {})[run_step] = {
        "score": 4, "object_url": url, "tagged_lines": {f"S{n}": line for n, line in enumerate(SATISFACTION, 1)}}


def _step(**kw) -> usg.UserSatisfactionGrading:
    return usg.UserSatisfactionGrading("grading", None, **{"run_step_id": "week", "judge_model": JUDGE,
                                                   "verifier_id": "psi_satisfaction_grading", **kw})


@pytest.mark.usefixtures("local_stores", "litellm_config")
@pytest.mark.asyncio
async def test_on_an_always_on_run_the_judge_reads_what_the_user_saw_and_the_log_scores_it(monkeypatch, tmp_path):
    rig, ctx, prompt = await _always_on(monkeypatch, tmp_path)
    _put_rating(ctx, "week", [{"at": "2026-11-12T10:00:00Z", "text": "Table is at 7."}])
    answer = _answer(
        asks=[{"turn": 1, "what": "asked whether to keep a seafood place", "quote": "Keep it?", "kind": "question",
               "could_handle": True, "how": "dad's allergy is in its notes", "same_as": 0}],
        incidents=[_incident("I1", 1, "put_it_on_me", "minor", "Keep it?")],
        moods=[{"turn": 1, "mood": "a little annoyed"}, {"turn": 2, "mood": "happy"}],
        end={"outcome": "done", "open": [], "quote": "Booked Pell & Finch instead."})
    calls = _litellm_answers(monkeypatch, lambda kwargs: json.dumps(answer))
    ctx.metadata["seed"] = {"guest": "her dad"}
    step = _step()
    step._assistant_prompt = "You are Hannah's assistant. Dinner is for <guest>."  # as task_problems reads it

    await step.execute(ctx)

    [call] = calls
    system, user = call["messages"]
    assert system["content"] == core.system_text(True) and call["response_format"] == usg.RESPONSE_FORMAT
    assert "Dinner is for her dad." in user["content"]  # the run's seed filled in
    sent = user["content"]
    assert "<instructions>\n" + prompt.strip() + "\n</instructions>" in sent  # the very prompt the user sim ran with
    for c in rig.user_calls:  # and every check-in as the user sim got it
        assert "<saw>\n" + c["parts"][0]["text"].strip() + "\n</saw>" in sent
    assert '"dad is allergic to shellfish, swap it pls"' in sent
    assert "  S2  3, fine" in sent and "Table is at 7." in sent

    entry = ctx.metadata["verifications"]["psi_satisfaction_grading"]
    assert (entry["run_step_id"], entry["run_kind"], entry["user_sim"], entry["judge_model"]) == (
        "week", "always_on", "user-sim", JUDGE)
    assert entry["instructions"]["matches"] is True and entry["turns"] == 2
    assert entry["assistant_instructions"] == "shown"
    assert (entry["score"], entry["outcome"], entry["incidents"], entry["calls"]) == (4.85, "done", 1, 1)
    log = usj._load(entry["log_object_url"])
    assert log["incidents"][0]["quote_found"] is True and log["asks"][0]["could_handle"] is True
    grade = ctx.metadata["grades"]["grading"]
    assert grade["verifier_status"] == "graded" and grade["score"] == core.normalized(4.85) and grade["passed"]
    assert {r["id"] for r in grade["verifier_criteria"]} == {*core.DIMENSIONS, "outcome"}
    assert all(r["weight"] == 0.0 for r in grade["verifier_criteria"])  # nothing pooling them moves
    assert [c["outcome"] for c in grade["judge_calls"]] == ["answered"]
    assert grade["verifier_settings"]["judge_model"] == JUDGE
    assert ctx.metadata["models"]["judge"]["grading"] == JUDGE


def _chat(monkeypatch, run: UserSimRun) -> TaskStepContext:
    """A chat run's context, its reading stubbed as test_user_sim_judge stubs it."""
    monkeypatch.setattr(usg, "_read_run", lambda kind, step_id, record, context: (
        run, {"user_sim": "mia", "instructions": {"matches": True}}))
    ctx = TaskStepContext()
    ctx.prompt_responses.append(PromptResponse(prompt_id="ask", response="all set", step_id="week"))
    ctx.metadata["user_sim_reads"] = {"week": [{}]}
    return ctx


@pytest.mark.usefixtures("local_stores", "litellm_config")
@pytest.mark.asyncio
async def test_an_answer_that_isn_t_json_is_asked_again_and_without_a_rating_the_log_says_so(monkeypatch):
    ctx = _chat(monkeypatch, _shop())
    answers = iter(["I think it went fine.", json.dumps(_answer())])
    calls = _litellm_answers(monkeypatch, lambda kwargs: next(answers))

    await _step().execute(ctx)

    entry = ctx.metadata["verifications"]["psi_satisfaction_grading"]
    assert len(calls) == 2 and entry["calls"] == 2 and entry["score"] == 5.0
    assert entry["run_kind"] == "prompt_agent"
    assert entry["notes"] == ["the run has no rate_user_satisfaction entry: no rating lines, and nothing read after "
                              "the last check-in"]
    assert entry["assistant_instructions"].startswith("unknown")  # built outside a task
    assert calls[0]["messages"][0]["content"] == core.system_text(False)
    assert [c["attempt"] for c in ctx.metadata["grades"]["grading"]["judge_calls"]] == [1, 2]


@pytest.mark.usefixtures("local_stores", "litellm_config")
@pytest.mark.asyncio
async def test_it_sends_no_temperature_and_drops_a_setting_the_model_refuses(monkeypatch):
    ctx = _chat(monkeypatch, _shop())
    refused = litellm.UnsupportedParamsError(message="max_tokens isn't supported", llm_provider="anthropic", model=JUDGE)
    answers = iter([refused, json.dumps(_answer())])
    calls = _litellm_answers(monkeypatch, lambda kwargs: next(answers))

    await _step().execute(ctx)

    assert "temperature" not in calls[0] and calls[0]["max_tokens"] == usg.MAX_TOKENS
    assert "max_tokens" not in calls[1]  # dropped after the refusal
    assert ctx.metadata["grades"]["grading"]["verifier_status"] == "graded"


@pytest.mark.usefixtures("local_stores", "litellm_config")
@pytest.mark.asyncio
async def test_a_judge_that_never_answers_fails_the_grade_with_no_score_and_never_the_task(monkeypatch):
    ctx = _chat(monkeypatch, _shop())
    _litellm_answers(monkeypatch, lambda kwargs: RuntimeError("the endpoint is down"))
    step = _step()
    assert step.fail_task_on_error is False

    with pytest.raises(RuntimeError, match="gave no log after 3 calls"):
        await step.execute(ctx)

    grade = ctx.metadata["grades"]["grading"]
    assert (grade["verifier_status"], grade["score"]) == ("failed", None)
    assert [c["outcome"] for c in grade["judge_calls"]] == ["refused"] * 3
    assert "psi_satisfaction_grading" not in ctx.metadata.get("verifications", {})


@pytest.mark.usefixtures("litellm_config")
@pytest.mark.asyncio
async def test_a_run_that_ended_on_an_error_or_never_reached_the_user_sim_is_skipped():
    errored = TaskStepContext()
    errored.metadata["always_on"] = {"week": {"error_type": "agent_unreachable"}}
    await _step().execute(errored)
    assert errored.metadata["grades"]["grading"]["verifier_skip_reasoning"] == "the run ended with error_type=" \
                                                                           "agent_unreachable"
    unread = TaskStepContext()
    unread.prompt_responses.append(PromptResponse(prompt_id="ask", response="done", step_id="week"))
    await _step().execute(unread)
    assert unread.metadata["grades"]["grading"]["verifier_status"] == "skipped"


# ---- the task it's in ---------------------------------------------------------------------------------------------
def test_it_logs_a_built_in_user_sim_s_run_after_the_user_sim_judge_and_the_rating():
    deploy, week = _user_sim_steps()
    after_week = [TaskStepDependency("week")]
    judge = usj.UserSimJudge("judge", None, run_step_id="week", depends_on=after_week)
    rate = RateUserSatisfactionTaskStep("rate", None, satisfaction=SATISFACTION, run_step_id="week",
                                        depends_on=after_week)
    def problems(*deps: str, before=(deploy, week, judge, rate)) -> list[str]:
        log = _step(depends_on=[TaskStepDependency(d) for d in deps])
        return log.task_problems([*before, log])

    assert problems("judge", "rate") == []
    told = ao.AlwaysOnTaskStep("week", None, user_sim="u", prompt_id="ask", agent_name="solver",
                               system_prompt="You are Pat's assistant.")
    log = _step(depends_on=[TaskStepDependency("judge"), TaskStepDependency("rate")])
    assert log.task_problems([deploy, told, judge, rate, log]) == []
    assert log._assistant_prompt == "You are Pat's assistant."  # what the assistant was told, read at load
    seeded = _step()
    seeded.seeded_from(log)
    assert seeded._assistant_prompt == "You are Pat's assistant."
    [no_judge] = problems("rate")
    assert "it must depend on 'judge' (user_sim_judge), so a run the user sim spoiled is never graded" in no_judge
    [no_rating] = problems("judge")
    assert "it must depend on 'rate' (rate_user_satisfaction)" in no_rating
    early = _step(depends_on=after_week)
    assert any("comes before step 'week'" in p for p in early.task_problems([deploy, early, week]))
    assert any("isn't a user sim" in p for p in problems("week", before=(week,)))
    nope = _step(run_step_id="nope")
    [nowhere] = nope.task_problems([deploy, week, nope])
    assert "names no step of the task" in nowhere


def test_it_is_registered_and_its_settings_round_trip():
    assert _builtin_registry()[usg.UserSatisfactionGrading.type] is usg.UserSatisfactionGrading
    step = _step(judge_timeout_seconds=120, depends_on=[TaskStepDependency("judge")])
    data = step.to_dict()
    assert data["fail_task_on_error"] is False and "min_passing_score" not in data  # 0.5, a 3, by default
    assert usg.UserSatisfactionGrading.from_dict(data).to_dict() == data
    assert step.min_passing_score == 0.5 and step.criterion_rows() == len(core.DIMENSIONS) + 1
    bad = usg.UserSatisfactionGrading("grading", None, run_step_id=" ", judge_timeout_seconds=0)
    assert len(bad.config_problems()) == 2
