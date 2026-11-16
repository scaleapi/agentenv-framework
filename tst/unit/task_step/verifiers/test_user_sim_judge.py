"""user_sim_judge: a rubrics_verifier that grades the user sim of an always_on run or a prompt_agent chat run, one call
per check with the task's items riding along, with a fake model standing in for the judge. A grade-changing slip fails
the task (fail_task_when_not_passing, on by default); an undecided judge never does."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace

import httpx
import litellm
import pytest

from agent_env.config.model import ModelConfig
from agent_env.openenv.reward import reward_of
from agent_env.store import LocalSecretStore
from agent_env.task import Task
from agent_env.task_step.context import PromptResponse, StepContextError, TaskStepContext, VerifierNotPassingError
from agent_env.task_step.registry import _builtin_registry
from agent_env.task_step.task_step import TaskStepDependency
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent import user_sim_versions
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import USER_SIMS_KEY
from agent_env.task_step.task_steps.verifiers import user_sim_judge as usj
from agent_env.task_step.task_steps.verifiers import user_sim_judge_core as core
from agent_env.task_step.task_steps.verifiers.agent_prompt_response_verifier import AgentPromptResponseVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import (
    JudgeOutputFormat,
    diagnose_user_sim_checks_response,
    get_judge_output_format_spec,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep, is_rubrics_verifier
from tst.unit.task_step.always_on_harness import Rig
from tst.unit.task_step.test_deploy_user_sim_agent import HANNAH, _Deploy
from tst.unit.task_step.test_deploy_user_sim_agent import _step as _deploy_step
from tst.unit.task_step.test_prompt_agent_built_in_user_sim import _context as _chat_context
from tst.unit.task_step.test_prompt_agent_built_in_user_sim import _step as _chat_step
from tst.unit.task_step.test_prompt_agent_built_in_user_sim import said
from tst.unit.task_step.test_prompt_agent_user_side import AGENT, USER, Peers, text

JUDGE = "fake/judge"
OTHER_PERSONA = "0" * 64  # a persona fingerprint no run here has
TITLES = {core.check_title(name): name for name in core.CHECK_NAMES}
ALL_ERROR_CHECKS = list(core.DEFAULT_ERROR_CHECKS)


def PASS(**kw) -> dict:
    return {"reason": "fine", "turns": [], "verdict": "pass", "could_change_grade": False, "assistant_asked": False,
            **kw}


def FAIL(turns=(1,), change=True, **kw) -> dict:
    return {"reason": "it broke the rule", "turns": list(turns), "verdict": "fail", "could_change_grade": change,
            "assistant_asked": False, **kw}


def NA(**kw) -> dict:
    return {**PASS(reason="it never came up"), "verdict": "n/a", **kw}


def _criteria_of(prompt: str) -> list[dict]:
    """The criteria a judge prompt asks about, as the user_sim_checks template lays them out."""
    block = prompt.split("\n# THE CRITERIA\n", 1)[1]
    return json.loads(block.split("\n\nReturn only the JSON object.", 1)[0])


class FakeModel:
    """Stands in for rubrics_verifier's model call. Each criterion is answered from ``script[name]`` (a check's name or
    an item's id): a row's fields, a whole answer's text, an exception, or a list of them used in turn; anything
    unscripted passes. Records every call."""

    def __init__(self, script=None):
        self.script = {k: list(v) if isinstance(v, list) else v for k, v in (script or {}).items()}
        self.calls: list[dict] = []
        self.in_flight = 0

    async def __call__(self, eval_prompt, *, model, context, image_blocks=None, system=None, cached_prefix=None):
        criteria = _criteria_of(eval_prompt)
        names = [c.get("item") or TITLES[c["check"]] for c in criteria]
        self.calls.append({"model": model, "names": names, "criteria": criteria, "prompt": eval_prompt,
                           "system": system, "cached_prefix": cached_prefix, "alongside": self.in_flight})
        self.in_flight += 1
        try:
            await asyncio.sleep(0)  # let the other calls start, if they may
            rows = []
            for name, criterion in zip(names, criteria):
                got = self.script.get(name, PASS())
                if isinstance(got, list):
                    got = got.pop(0)
                if isinstance(got, Exception):
                    raise got
                if isinstance(got, str):
                    return self._reply(got)
                rows.append({"id": criterion["id"], **got})
            return self._reply(json.dumps({"results": rows}))
        finally:
            self.in_flight -= 1

    @staticmethod
    def _reply(text: str) -> dict:
        return {"response": text, "trajectory_object_url": None,
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}, "cost": 0.001}

    def asked(self, name: str) -> list[dict]:
        return [c for c in self.calls if name in c["names"]]

    def run_text(self) -> str:
        """The prompt of the judge's first call, which holds the run."""
        return self.calls[0]["prompt"]


def _step(fake: FakeModel, monkeypatch, **kw) -> usj.UserSimJudge:
    monkeypatch.setattr(RubricsVerifierTaskStep, "_prompt_llm_judge", fake)
    return usj.UserSimJudge("judge_user_sim", None, **{"run_step_id": "talk", "judge_model": JUDGE, "verifier_id": "usj",
                                                       **kw})


def _entry(ctx: TaskStepContext) -> dict:
    return ctx.metadata["verifications"]["usj"]


def _rows(entry: dict) -> dict[str, dict]:
    return {r["id"]: r for r in entry["results"]}


def _verdicts(entry: dict, *ids: str) -> dict[str, str]:
    """Each row's verdict, or only those of ``ids``."""
    return {r["id"]: r["verdict"] for r in entry["results"] if not ids or r["id"] in ids}


def _run(*answers: tuple[str, str], graded=None) -> core.UserSimRun:
    turns = [core.Turn(n=i, outcome=o, message=m, input=f"It's Mon. NEW FROM YOUR AI ASSISTANT: question {i}?",
                       graded=graded is None or i in graded) for i, (o, m) in enumerate(answers, 1)]
    return core.UserSimRun(name="hannah", instructions="You are Hannah Reyes. K1 Dad is allergic to shellfish.",
                           turns=turns)


async def _judge(run: core.UserSimRun, fake: FakeModel, monkeypatch, *, matches: bool = True, **kw) -> dict:
    """The step's entry for ``run``, read as a chat run's (its reading stubbed), the gate off unless ``kw`` says."""
    instructions = {"source": "rebuilt", "prompt_sha256": "a" * 64, "recorded_prompt_sha256": "a" * 64,
                    "matches": matches}
    monkeypatch.setattr(usj, "_read_run", lambda kind, step_id, record, context: (
        run, {"user_sim": "hannah", "instructions": instructions}))
    ctx = TaskStepContext()
    ctx.prompt_responses.append(PromptResponse(prompt_id="ask", response="all set", step_id="talk"))
    ctx.metadata["user_sim_reads"] = {"talk": [{}]}
    ctx.metadata[USER_SIMS_KEY] = {"hannah": {"persona_sha256": OTHER_PERSONA}}
    await _step(fake, monkeypatch, **{"fail_task_when_not_passing": False, **kw}).execute(ctx)
    return _entry(ctx)


def _plan(run: core.UserSimRun, items=(), checks=None, error_checks=None, **kw) -> core.Plan:
    return core.plan_run(run, checks=list(core.CHECK_NAMES) if checks is None else checks, items=list(items),
                         error_checks=ALL_ERROR_CHECKS if error_checks is None else error_checks, **kw)


# ---- what the judge is asked --------------------------------------------------------------------------------------
def test_each_check_is_one_narrow_question_naming_its_families():
    for name, check in core.CHECKS.items():
        q = core.check_question(name)
        assert q.startswith(check.question) and all(f in q for f in check.fails + check.not_fails)
        assert core.check_title(name) == f"{check.title} ({', '.join(check.families)})"
    assert "could_change_grade: Usually no." in core.check_question("style")
    assert sorted(core.FAMILY_OWNER) == ["A1", "A2", "A3", "A4", "F1", "F2", "F3", "F4", "G1", "G2", "G3", "G4", "G5"]


def test_self_contradiction_is_asked_under_made_up_not_guidance():
    line = "It contradicted something it said earlier in the run."
    assert line in core.check_question("made_up") and "contradict" not in core.check_question("guidance")


def test_the_run_is_split_before_the_last_graded_answer_so_repeats_share_a_cached_prefix():
    run = _run(("reply", "yes book it"), ("no_reply", ""), ("reply", "no seafood pls"), graded={3})
    head, tail = core.render_run(run)
    assert "<instructions>\nYou are Hannah Reyes." in head
    assert "## Turn 1 · CONTEXT ONLY" in head and 'message: "yes book it"' in head
    assert "## Turn 3 · GRADED" in head and "question 3?" in head and "no seafood pls" not in head
    assert tail == '### Its answer\noutcome: reply\nmessage: "no seafood pls"'
    assert "message:" not in head.split("## Turn 2")[1].split("## Turn 3")[0]  # a no_reply shows no empty message
    assert core.run_text(run) == head + "\n" + tail


def test_task_items_go_with_the_check_that_owns_their_family_and_the_rest_get_a_call_of_their_own():
    items = [core.RubricItem("leak", "G1", "never mentions the engagement"),
             core.RubricItem("ends", "G4", "signs off only after the recap"),
             core.RubricItem("tone", "F2", "stays lowercase")]
    calls = core.plan_calls(["oversharing", "made_up"], items)
    assert [i.id for i in calls["oversharing"]] == ["leak"] and calls["made_up"] == []
    assert [i.id for i in calls[core.TASK_ITEMS]] == ["ends", "tone"]  # style isn't running, so "tone" goes there too
    plan = _plan(_run(("reply", "x")), items, checks=["oversharing", "made_up"])
    assert [c["id"] for c in plan.criteria] == ["oversharing", "leak", "made_up", "ends", "tone"]
    assert plan.batches == [2, 1, 2]
    assert plan.criteria[1] == {"id": "leak", "item": "leak", "family": "G1 (Leaking and privacy)", "kind": "general",
                                "question": "never mentions the engagement", "graded_turns": [1], "weight": 1.0}
    assert plan.about["ends"]["asked_with"] == plan.about["tone"]["asked_with"] == core.TASK_ITEMS


def test_rubric_items_are_checked_when_the_task_loads():
    bad = [{"id": "a", "family": "Z9", "text": "x"}, {"id": "a", "family": "G1", "text": "y", "weight": 2}, {"id": ""}]
    problems = core.item_problems(bad)
    assert any("family 'Z9'" in p for p in problems) and any("used twice" in p for p in problems)
    assert any("unknown fields ['weight']" in p for p in problems) and any("needs an id" in p for p in problems)


# ---- reading an answer (the user_sim_checks format, on this step's criteria) ----------------------------------------
def test_an_answer_is_checked_and_its_turns_kept_to_the_graded_ones():
    criteria = _plan(_run(("reply", "a"), ("reply", "b"), ("reply", "c"), graded={3}), checks=["made_up"]).criteria
    rows, problem = diagnose_user_sim_checks_response(json.dumps({"results": [
        {"id": "made_up", **FAIL(turns=[2, 3, "x"])}]}), criteria)
    assert problem is None and (rows[0]["verdict"], rows[0]["turns"], rows[0]["turns_outside_graded"]) == (
        "fail", [3], [2])
    rows, _ = diagnose_user_sim_checks_response(json.dumps({"results": [
        {"id": "made_up", **PASS(could_change_grade=True)}]}), criteria)
    assert rows[0]["could_change_grade"] is False  # a pass can't change the grade


def test_a_fail_that_cites_no_graded_turn_is_not_usable_and_a_pass_needs_none():
    criteria = _plan(_run(("reply", "a")), checks=["made_up"]).criteria
    for turns in ([99], []):
        rows, problem = diagnose_user_sim_checks_response(
            json.dumps({"results": [{"id": "made_up", **FAIL(turns=turns)}]}), criteria)
        assert rows is None and problem.row_problems == [
            f"made_up: a fail must cite at least one graded turn; cited {turns}, graded [1]"]
    rows, problem = diagnose_user_sim_checks_response(
        json.dumps({"results": [{"id": "made_up", **PASS(turns=[99])}]}), criteria)
    assert problem is None and rows[0]["turns"] == []


def test_a_fail_that_clears_one_turn_and_fails_another_is_kept():
    criteria = _plan(_run(("reply", "a")), checks=["oversharing"]).criteria
    reason = "No clear failure at turn 1; at turn 2 she revealed the private password."
    rows, problem = diagnose_user_sim_checks_response(
        json.dumps({"results": [{"id": "oversharing", **FAIL(reason=reason)}]}), criteria)
    assert problem is None and rows[0]["verdict"] == "fail" and rows[0]["could_change_grade"] is True


@pytest.mark.asyncio
async def test_the_judge_writes_its_reason_before_its_verdict_within_the_word_limit(monkeypatch):
    fake = FakeModel()
    await _judge(_run(("reply", "x")), fake, monkeypatch)
    prompt = fake.calls[0]["prompt"]
    assert "Reason first and decide last." in prompt and "at most 40 words" in prompt
    assert prompt.index("# THE USER SIM'S INSTRUCTIONS") < prompt.index("# THE CRITERIA")  # the run first
    row = get_judge_output_format_spec(JudgeOutputFormat.USER_SIM_CHECKS).output_format["schema"]["properties"][
        "results"]["items"]
    assert list(row["properties"]) == ["id", "reason", "turns", "verdict", "could_change_grade", "assistant_asked"]


# ---- one run, through the step ----------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_each_check_s_verdict_is_recorded_with_its_reason_and_turns(monkeypatch):
    fake = FakeModel({"made_up": FAIL(), "style": FAIL(change=False), "delegation": NA(reason="no rules")})
    entry = await _judge(_run(("reply", "dad's allergic to shellfish fyi")), fake, monkeypatch)
    assert _verdicts(entry) == {
        "guidance": "pass", "oversharing": "pass", "delegation": "n/a", "style": "fail", "rogue": "pass",
        "made_up": "fail", "ending": "pass"}
    assert _rows(entry)["made_up"] == {
        "id": "made_up", "title": "Made-up facts", "families": ["A2"], "weight": 1.0, "verdict": "fail",
        "reason": "it broke the rule", "turns": [1], "could_change_grade": True, "assistant_asked": False,
        "score": 0.0, "result": False, "justification": "it broke the rule"}
    assert _rows(entry)["delegation"]["reason"] == "no rules"
    # Style failed, but its row has weight 0; made_up failed in a way that could change the grade.
    assert _rows(entry)["style"]["weight"] == 0.0 and _rows(entry)["style"]["result"] is True
    assert entry["user_sim_error"] == {"value": "yes", "because": ["made_up"], "unknown": [], "fails_task": False,
                                       "not_failing": usj.NOT_FAILING_OFF}
    assert entry["format"] == "user_sim_checks" and entry["score"] == 0.0
    assert len(fake.calls) == 7 == entry["calls"] and {c["model"] for c in fake.calls} == {JUDGE}
    assert entry["usage"]["prompt_tokens"] == 700 and entry["cost"] == pytest.approx(0.007)


@pytest.mark.asyncio
async def test_each_call_shares_the_run_and_asks_one_check_and_the_first_call_runs_alone(monkeypatch):
    fake = FakeModel()
    run = _run(("reply", "ok"))
    entry = await _judge(run, fake, monkeypatch)
    prefixes = {c["cached_prefix"] for c in fake.calls}
    assert len(prefixes) == 1 and next(iter(prefixes)).endswith(core.run_text(run))  # one shared, cacheable prefix
    assert all(c["system"] == usj.SYSTEM and "never the assistant" in c["system"] for c in fake.calls)
    assert sorted(n for c in fake.calls for n in c["names"]) == sorted(core.CHECK_NAMES)
    assert fake.calls[0]["names"] == ["guidance"] and all(len(c["names"]) == 1 for c in fake.calls)
    assert fake.calls[0]["alongside"] == 0 and max(c["alongside"] for c in fake.calls[1:]) > 0  # alone, then the rest
    assert entry["score"] == 1.0 and entry["user_sim_error"]["value"] == "no"


@pytest.mark.asyncio
async def test_a_failure_that_can_t_change_the_grade_doesn_t_make_a_user_sim_error(monkeypatch):
    entry = await _judge(_run(("reply", "Here's a summary of what I booked:")),
                         FakeModel({"rogue": FAIL(change=False)}), monkeypatch)
    assert _verdicts(entry)["rogue"] == "fail" and entry["user_sim_error"]["value"] == "no" and entry["score"] == 1.0


@pytest.mark.asyncio
async def test_a_bad_answer_is_asked_again_and_a_check_whose_call_fails_is_undecided(monkeypatch):
    fake = FakeModel({"guidance": ["not json", FAIL()], "rogue": [RuntimeError("503")]})
    entry = await _judge(_run(("reply", "you pick")), fake, monkeypatch)
    assert _verdicts(entry, "guidance", "rogue") == {"guidance": "fail", "rogue": "error"}
    assert entry["judge_output_retries"] == 1
    retry = fake.asked("guidance")[1]["prompt"]
    assert "## Correction Required" in retry and "the answer is not a JSON object" in retry
    assert _rows(entry)["rogue"]["error"] == "no usable answer after 4 tries: RuntimeError: 503"
    assert (_rows(entry)["rogue"]["score"], _rows(entry)["rogue"]["assistant_asked"]) == (1.0, None)
    assert entry["user_sim_error"]["value"] == "yes"
    assert (entry["user_sim_error"]["because"], entry["user_sim_error"]["unknown"]) == (["guidance"], ["rogue"])
    alone = await _judge(_run(("reply", "x")), FakeModel({"rogue": [RuntimeError("503")]}), monkeypatch)
    assert (alone["user_sim_error"]["value"], alone["user_sim_error"]["unknown"]) == ("unknown", ["rogue"])
    assert alone["flag"] == "user sim judge undecided" and alone["score"] == 1.0


@pytest.mark.asyncio
async def test_a_broken_answer_is_asked_again_up_to_3_times_then_the_run_is_flagged_and_the_task_not_failed(
        monkeypatch):
    assert usj.UserSimJudge.DEFAULT_MAX_RETRIES == 4  # the first ask and 3 corrections
    fake = FakeModel({"made_up": ["not json"] * 5})
    entry = await _judge(_run(("reply", "x")), fake, monkeypatch, fail_task_when_not_passing=True)
    assert len(fake.asked("made_up")) == 4
    assert all("## Correction Required" in c["prompt"] for c in fake.asked("made_up")[1:])
    assert _verdicts(entry, "made_up") == {"made_up": "error"} and _rows(entry)["made_up"]["error"].startswith(
        "no usable answer after 4 tries: ValueError: ")
    assert entry["user_sim_error"] == {"value": "unknown", "because": [], "unknown": ["made_up"], "fails_task": False}
    assert entry["flag"] == "user sim judge undecided"  # and execute didn't raise


@pytest.mark.asyncio
async def test_a_fail_citing_only_ungraded_turns_is_asked_again(monkeypatch):
    fake = FakeModel({"made_up": [FAIL(turns=[99]), FAIL(turns=[1])]})
    entry = await _judge(_run(("reply", "dad's allergic to shellfish")), fake, monkeypatch)
    assert _rows(entry)["made_up"]["verdict"] == "fail" and _rows(entry)["made_up"]["turns"] == [1]
    assert entry["judge_output_retries"] == 1 and entry["user_sim_error"]["value"] == "yes"
    assert "a fail must cite at least one graded turn; cited [99], graded [1]" in fake.asked("made_up")[1]["prompt"]
    never = await _judge(_run(("reply", "x")), FakeModel({"made_up": [FAIL(turns=[99])] * 4}), monkeypatch)
    assert _verdicts(never, "made_up") == {"made_up": "error"} and "cited [99]" in _rows(never)["made_up"]["error"]
    assert never["user_sim_error"]["value"] == "unknown"


SAYS_NO_FAILURE = ("Hannah flagged it anyway which is fine. No clear failure found.",
                   "This is not a delegation failure by Linda. No clear delegation failures found.",
                   "Every answer fits her rules; nothing fails.")


@pytest.mark.asyncio
async def test_a_fail_whose_reason_says_no_clear_failure_found_is_asked_again(monkeypatch):
    fake = FakeModel({"guidance": [FAIL(reason=SAYS_NO_FAILURE[0]), PASS(reason="No clear failure found.")],
                      "oversharing": FAIL()})
    entry = await _judge(_run(("reply", "dad's allergic to shellfish fyi")), fake, monkeypatch)
    assert (_rows(entry)["guidance"]["verdict"], _rows(entry)["guidance"]["reason"]) == (
        "pass", "No clear failure found.")
    retry = fake.asked("guidance")[1]["prompt"]
    assert "a fail whose reason says 'No clear failure found' contradicts itself" in retry
    # The ordinary fail is answered once and alone makes the run a user sim error.
    assert len(fake.asked("oversharing")) == 1 and entry["user_sim_error"]["because"] == ["oversharing"]
    for reason in SAYS_NO_FAILURE:
        never = await _judge(_run(("reply", "x")), FakeModel({"delegation": [FAIL(reason=reason)] * 4}), monkeypatch)
        assert _verdicts(never, "delegation") == {"delegation": "error"} and never["user_sim_error"]["value"] == "unknown"


@pytest.mark.asyncio
async def test_a_task_item_fail_citing_no_graded_turn_is_asked_again(monkeypatch):
    items = [{"id": "rating", "family": "A5", "check": "rates the trip 2 or lower", "can_error": True}]
    fake = FakeModel({"rating": [FAIL(turns=[99]), FAIL(turns=[1])]})
    entry = await _judge(_run(("done", "5 stars")), fake, monkeypatch, rubric_items=items)
    assert _rows(entry)["rating"]["verdict"] == "fail" and _rows(entry)["rating"]["turns"] == [1]
    assert entry["user_sim_error"]["because"] == ["rating"] and len(fake.asked("rating")) == 2
    never = await _judge(_run(("done", "5 stars")), FakeModel({"rating": [FAIL(turns=[])] * 4}), monkeypatch,
                         rubric_items=items)
    assert _verdicts(never, "rating") == {"rating": "error"} and "a fail must cite" in _rows(never)["rating"]["error"]
    assert (never["user_sim_error"]["value"], never["user_sim_error"]["unknown"]) == ("unknown", ["rating"])


@pytest.mark.asyncio
async def test_task_items_get_their_own_rows_and_one_rule_says_which_make_a_user_sim_error(monkeypatch):
    items = [{"id": "leak", "family": "G1", "check": "never hints at the engagement"},
             {"id": "rating", "family": "A5", "check": "rates the trip 2 or lower", "can_error": True},
             {"id": "tone", "family": "A5", "check": "stays upbeat"}]
    fake = FakeModel({"rating": FAIL(reason="rated it 5"), "leak": PASS(reason="-"), "tone": FAIL()})
    entry = await _judge(_run(("done", "thx bye")), fake, monkeypatch, rubric_items=items)
    assert _rows(entry)["leak"] == {
        "id": "leak", "family": "G1", "kind": "general", "check": "never hints at the engagement",
        "asked_with": "oversharing", "weight": 1.0, "verdict": "pass", "reason": "-", "turns": [],
        "could_change_grade": False, "assistant_asked": False, "score": 1.0, "result": True, "justification": "-"}
    assert (_rows(entry)["rating"]["asked_with"], _rows(entry)["rating"]["weight"]) == ("task_items", 1.0)
    # An A5 item has no check of its own, so unless it can_error it is recorded only.
    assert _rows(entry)["tone"]["verdict"] == "fail" and _rows(entry)["tone"]["weight"] == 0.0
    assert entry["user_sim_error"]["because"] == ["rating"] and len(fake.calls) == 8
    assert [r["id"] for r in entry["results"]] == [*core.CHECK_NAMES, "leak", "rating", "tone"]
    # The rule is the checks' rule: an item's fail that couldn't change the grade is no user sim error.
    mild = await _judge(_run(("done", "thx bye")), FakeModel({"rating": FAIL(change=False)}), monkeypatch,
                        rubric_items=items[1:2])
    assert _verdicts(mild, "rating") == {"rating": "fail"} and mild["user_sim_error"]["value"] == "no"


@pytest.mark.asyncio
async def test_the_ending_check_asks_when_it_signed_off_and_a_grade_changing_fail_makes_a_user_sim_error(monkeypatch):
    q = core.check_question("ending")
    assert "while the assistant was waiting on its answer" in q and "WHEN YOU'RE DONE" in q
    assert "A short sign-off when its reply limit is reached." in q and "never said \"done\"" in q
    assert "ending" in core.DEFAULT_ERROR_CHECKS
    for answer, verdict, error in ((FAIL(turns=[1]), "fail", "yes"), (PASS(), "pass", "no"),
                                   (NA(reason="never said done"), "n/a", "no")):
        entry = await _judge(_run(("done", "ok thx bye")), FakeModel({"ending": answer}), monkeypatch)
        assert _verdicts(entry, "ending") == {"ending": verdict} and _rows(entry)["ending"]["families"] == ["G4"]
        assert entry["user_sim_error"]["value"] == error


def test_the_judge_is_told_when_the_run_ended_at_the_user_sim_s_reply_limit():
    turns = _run(("reply", "swap it"), ("no_reply", ""), ("reply", "ok"), ("reply", "thx")).turns
    [note] = usj._reply_limit_note(turns, 2)
    assert note.startswith("the run ended at the user sim's reply limit: its reply at turn 3 was the last of its 2")
    assert usj._reply_limit_note(turns, 4) == usj._reply_limit_note(turns, None) == []


@pytest.mark.asyncio
async def test_a_run_with_no_graded_turn_is_not_sent_to_the_judge(monkeypatch):
    fake = FakeModel()
    entry = await _judge(_run(("reply", "ok"), graded=set()), fake, monkeypatch, fail_task_when_not_passing=True)
    assert entry["skipped"] == "no graded turns" and fake.calls == []
    assert (entry["results"], entry["score"], entry["user_sim_error"]["value"]) == ([], None, "no")


# ---- the gate -------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_grade_changing_slip_with_matching_instructions_fails_the_task_naming_the_rows(monkeypatch):
    with pytest.raises(VerifierNotPassingError, match=r"user_sim_judge 'judge_user_sim': the user sim slipped .* "
                                                      r"\(guidance, a1\)"):
        await _judge(_run(("reply", "you pick")), FakeModel({"guidance": FAIL(), "a1": FAIL()}), monkeypatch,
                     fail_task_when_not_passing=True,
                     rubric_items=[{"id": "a1", "family": "A1", "check": "gives the date"}])


@pytest.mark.asyncio
async def test_a_slip_isn_t_held_against_the_run_when_the_instructions_don_t_match(monkeypatch):
    entry = await _judge(_run(("reply", "you pick")), FakeModel({"guidance": FAIL()}), monkeypatch, matches=False,
                         fail_task_when_not_passing=True)
    assert entry["user_sim_error"] == {"value": "yes", "because": ["guidance"], "unknown": [], "fails_task": False,
                                       "not_failing": usj.NOT_FAILING_UNCONFIRMED}


@pytest.mark.asyncio
async def test_style_never_gates(monkeypatch):
    entry = await _judge(_run(("reply", "Dear assistant, I hope this finds you well.")),
                         FakeModel({"style": FAIL(change=True)}), monkeypatch, fail_task_when_not_passing=True)
    assert _rows(entry)["style"]["verdict"] == "fail" and _rows(entry)["style"]["could_change_grade"] is True
    assert entry["user_sim_error"]["value"] == "no" and entry["score"] == 1.0
    assert _rows(entry)["style"]["weight"] == 0.0 and "style" not in core.DEFAULT_ERROR_CHECKS


@pytest.mark.asyncio
async def test_assistant_asked_is_recorded_on_each_row_and_never_changes_the_gate(monkeypatch):
    fake = FakeModel({"rogue": FAIL(assistant_asked=True, reason="drafted the email to Ben the assistant asked for"),
                      "guidance": PASS(assistant_asked=True)})
    with pytest.raises(VerifierNotPassingError, match=r"\(rogue\)"):
        await _judge(_run(("reply", "Hi Ben, ...")), fake, monkeypatch, fail_task_when_not_passing=True)
    entry = await _judge(_run(("reply", "Hi Ben, ...")), FakeModel(fake.script | {
        "rogue": FAIL(assistant_asked=True)}), monkeypatch)
    assert {r["id"]: r["assistant_asked"] for r in entry["results"]} == {
        **dict.fromkeys(core.CHECK_NAMES, False), "guidance": True, "rogue": True}
    assert entry["user_sim_error"]["because"] == ["rogue"]


def test_its_score_checks_the_run_so_the_task_s_grader_is_its_reward():
    judge = usj.UserSimJudge("judge", None, run_step_id="week", depends_on=[TaskStepDependency("week")])
    assert usj.UserSimJudge.can_be_reward is False and is_rubrics_verifier(judge)
    task = Task("t", None, [*_user_sim_steps(), judge, _grader(depends_on=[TaskStepDependency("judge")])])
    pick, why = reward_of(task, None)
    assert why is None and pick({"grade": 0.75, "judge": 1.0}) == 0.75
    _, why = reward_of(task, "judge")
    assert "records a score that checks the run and can't be its reward" in why


# ---- the step on an always_on run ---------------------------------------------------------------------------------
ALWAYS_ON_ANSWERS = ['{"outcome": "reply", "message": "dad is allergic to shellfish, swap it pls", "used": "K3"}',
                     '{"outcome": "reply", "message": "all good, thx", "used": "none"}']


async def _always_on(monkeypatch, tmp_path, answers=ALWAYS_ON_ANSWERS):
    """An always_on run whose user the real deploy_user_sim_agent step deployed and recorded; the prompt it set on the
    user sim's agent; the Rig."""
    def agent(n, texts, rig):
        if n == 1:
            rig.message("Saturday is Saltwater Shack, a seafood place. Keep it?")
            return "sent"
        if n == 2:
            rig.message("Booked Pell & Finch instead. Anything else?")
            return "sent"
        rig.quiet()
        return ""

    deploy = _Deploy(monkeypatch)
    deployed = await deploy.run(_deploy_step(agent_name="user-sim", persona=HANNAH))
    [prompt] = deploy.fields("system_prompt")
    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=user_sim_versions, user_texts=list(answers))
    await rig.arm()
    ctx = rig.context()
    ctx.metadata[USER_SIMS_KEY] = {"user-sim": deployed.metadata[USER_SIMS_KEY]["user-sim"]}
    ctx.metadata["clock_configurations"] = [{"step_id": "clock", "timezone": "America/New_York"}]
    await ao.AlwaysOnTaskStep("week", None, user_sim="user-sim", prompt_id="ask", agent_name="solver",
                              end_time="P3D").execute(ctx)
    return rig, ctx, prompt


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_on_an_always_on_run_the_judge_reads_what_the_user_sim_got_and_said_and_a_slip_fails_the_task(
        monkeypatch, tmp_path):
    rig, ctx, prompt = await _always_on(monkeypatch, tmp_path)
    fake = FakeModel({"guidance": FAIL(turns=[2])})

    with pytest.raises(VerifierNotPassingError, match=r"\(guidance\)"):
        await _step(fake, monkeypatch, run_step_id="week").execute(ctx)

    run_text = fake.run_text()
    assert "<instructions>\n" + prompt.strip() + "\n</instructions>" in run_text  # the very prompt it ran with
    assert len(rig.user_calls) == 2
    for call in rig.user_calls:  # and every check-in word for word
        assert "<sent>\n" + call["parts"][0]["text"].strip() + "\n</sent>" in run_text
    assert '"dad is allergic to shellfish, swap it pls"' in run_text and "K3" not in run_text.split("</instructions>")[1]
    entry = _entry(ctx)
    assert (entry["run_step_id"], entry["run_kind"], entry["user_sim"], entry["judge_model"]) == (
        "week", "always_on", "user-sim", JUDGE)
    assert entry["instructions"]["matches"] is True and entry["notes"] == []
    assert _verdicts(entry) == {**dict.fromkeys(core.CHECK_NAMES, "pass"), "guidance": "fail"}
    assert _rows(entry)["guidance"]["turns"] == [2]
    assert entry["user_sim_error"] == {"value": "yes", "because": ["guidance"], "unknown": [], "fails_task": True}
    # The run itself is left as it was: the task fails, so nothing scores it.
    assert ctx.prompt_responses[-1].error_type is None and "user_sim_judge" not in ctx.metadata
    assert ctx.metadata["models"]["judge"]["judge_user_sim"] == JUDGE


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_without_the_stored_timeline_the_judge_reads_the_tail_and_is_told_what_s_missing(monkeypatch, tmp_path):
    rig, ctx, _ = await _always_on(monkeypatch, tmp_path)
    record, timeline = rig.record(ctx), rig.timeline(ctx)
    record.pop("object_url")
    record["timeline_tail"] = [{k: v for k, v in e.items() if k not in ("input", "read")} for e in timeline]
    fake = FakeModel()

    await _step(fake, monkeypatch, run_step_id="week").execute(ctx)

    head = fake.calls[0]["cached_prefix"]
    assert "Note: the run's full record wasn't stored" in head and "### What it was sent\n(not recorded)" in head
    assert "## Turn 1 · 2026-11-09T14:00:00Z UTC · GRADED" in head
    assert _entry(ctx)["notes"][0].startswith("the run's full record wasn't stored")


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_from_the_tail_an_answer_always_on_shortened_is_marked_cut(monkeypatch, tmp_path):
    from agent_env.task_step.task_steps.always_on import PENDING_TEXT_CHARS, _small

    record = {"timeline_tail": [_small({"event": "user_sim_read", "outcome": "reply", "message": "ok " * 200}),
                                _small({"event": "user_sim_read", "outcome": "reply", "message": "sounds good"})]}
    reads, notes, missing = usj._always_on_reads(record)
    assert [r["message_cut"] for r in reads] == [True, False] and len(reads[0]["message"]) == PENDING_TEXT_CHARS + 1
    assert missing == 0  # the record has no counts to compare with
    assert usj._turn(1, reads[0]).cut and not usj._turn(2, reads[1]).cut


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_run_that_ended_in_an_error_is_not_judged(monkeypatch, tmp_path):
    rig, ctx, _ = await _always_on(monkeypatch, tmp_path, answers=[])
    rig.record(ctx)["error_type"] = "agent_turn_failed"
    fake = FakeModel()

    await _step(fake, monkeypatch, run_step_id="week").execute(ctx)

    assert fake.calls == [] and _entry(ctx)["skipped"] == "the run ended with error_type=agent_turn_failed"
    assert _entry(ctx)["user_sim_error"] == {"value": "no", "because": [], "unknown": [], "fails_task": False}
    assert (_entry(ctx)["score"], _entry(ctx)["results"]) == (None, [])


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_run_s_judge_model_override_chooses_the_model_as_for_every_judge(monkeypatch, tmp_path):
    _, ctx, _ = await _always_on(monkeypatch, tmp_path)
    ctx.metadata["user_overrides"] = {"judge_model": "other"}
    fake = FakeModel()

    await _step(fake, monkeypatch, run_step_id="week").execute(ctx)

    assert {c["model"] for c in fake.calls} == {"other"} and len(fake.calls) == 7
    assert _entry(ctx)["judge_model"] == "other" and ctx.metadata["models"]["judge"]["judge_user_sim"] == "other"


# ---- the step on a prompt_agent chat run --------------------------------------------------------------------------
async def _chat(monkeypatch):
    Peers(monkeypatch, {AGENT: [text("draft one"), text("draft two"), text("glad to help")],
                        USER: [said("reply", "make it shorter", "D1"), said("done", "perfect, thanks!")]})
    ctx = await _chat_context(monkeypatch)
    await _chat_step().execute(ctx)
    return ctx


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_on_a_chat_run_the_judge_reads_each_stored_read_and_the_prompt_it_was_deployed_with(monkeypatch):
    ctx = await _chat(monkeypatch)
    us = ctx.metadata[USER_SIMS_KEY]["hannah"]
    fake = FakeModel({"style": FAIL(turns=[2], change=False)})

    await _step(fake, monkeypatch).execute(ctx)

    run_text = fake.run_text()
    shown = run_text.split("<instructions>\n")[1].split("\n</instructions>")[0] + "\n"
    assert hashlib.sha256(shown.encode()).hexdigest() == us["prompt_sha256"]  # the prompt deploy_user_sim_agent set
    for summary in ctx.metadata["user_sim_reads"]["talk"]:
        full = usj._load(summary["object_url"])
        assert "<sent>\n" + full["input"].strip() + "\n</sent>" in run_text
    assert 'outcome: reply\nmessage: "make it shorter"' in run_text and "D1" not in run_text.split("</instructions>")[1]
    assert 'outcome: done\nmessage: "perfect, thanks!"' in run_text
    entry = _entry(ctx)
    assert (entry["run_kind"], entry["user_sim"], entry["instructions"]["matches"]) == ("prompt_agent", "hannah", True)
    assert _verdicts(entry)["style"] == "fail" and entry["user_sim_error"]["value"] == "no"
    assert entry["graded_turns"] == [1, 2] and ctx.prompt_responses[-1].error_type is None


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_with_the_gate_on_a_slip_fails_the_step_and_graders_never_see_the_run_marked(monkeypatch):
    ctx = await _chat(monkeypatch)

    with pytest.raises(VerifierNotPassingError, match=r"\(oversharing\)"):
        await _step(FakeModel({"oversharing": FAIL()}), monkeypatch).execute(ctx)

    assert _entry(ctx)["user_sim_error"]["fails_task"] is True and ctx.prompt_responses[-1].error_type is None


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_with_the_gate_off_a_slip_is_recorded_only(monkeypatch):
    ctx = await _chat(monkeypatch)

    await _step(FakeModel({"rogue": FAIL()}), monkeypatch, fail_task_when_not_passing=False).execute(ctx)

    assert _entry(ctx)["user_sim_error"] == {"value": "yes", "because": ["rogue"], "unknown": [], "fails_task": False,
                                             "not_failing": usj.NOT_FAILING_OFF}


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_instructions_that_don_t_match_the_recorded_hash_are_flagged_and_never_fail_the_task(monkeypatch):
    ctx = await _chat(monkeypatch)
    us = ctx.metadata[USER_SIMS_KEY]["hannah"]
    us["description"] = {**us["description"], "who_you_are": [*us["description"]["who_you_are"], "A changed line."]}
    fake = FakeModel({"guidance": FAIL()})

    await _step(fake, monkeypatch).execute(ctx)

    entry = _entry(ctx)
    assert entry["instructions"]["matches"] is False
    assert entry["instructions"]["recorded_prompt_sha256"] == us["prompt_sha256"] != entry["instructions"]["prompt_sha256"]
    assert "Note: these instructions were rebuilt from the user sim's settings and differ from the ones it got" in (
        fake.calls[0]["prompt"])
    assert entry["user_sim_error"]["value"] == "yes" and entry["user_sim_error"]["fails_task"] is False
    assert entry["user_sim_error"]["not_failing"] == usj.NOT_FAILING_UNCONFIRMED


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_chat_run_that_ended_in_an_error_or_where_the_user_sim_never_read_is_not_judged(monkeypatch):
    ctx = await _chat(monkeypatch)
    ctx.prompt_responses[-1].error_type = "user_sim_error"
    fake = FakeModel()
    await _step(fake, monkeypatch).execute(ctx)
    assert fake.calls == [] and _entry(ctx)["skipped"] == "the run ended with error_type=user_sim_error"

    ctx.prompt_responses[-1].error_type = None
    ctx.metadata.pop("user_sim_reads")
    await _step(fake, monkeypatch).execute(ctx)
    assert fake.calls == [] and _entry(ctx)["skipped"] == "the user sim never read: the run has no user_sim_reads"


def test_the_run_is_the_one_run_step_id_names():
    ctx = TaskStepContext()
    ctx.metadata = {"always_on": {"week": {}}, "user_sim_reads": {"talk": [{}]}}
    with pytest.raises(StepContextError, match="step 'nope' left no run to judge"):
        usj.UserSimJudge("j", None, run_step_id="nope")._find_run(ctx)
    assert usj.UserSimJudge("j", None, run_step_id="talk")._find_run(ctx) == ("prompt_agent", "talk")
    assert usj.UserSimJudge("j", None, run_step_id="week")._find_run(ctx) == ("always_on", "week")
    # A one-turn chat ends before its user sim reads: it leaves a response and no reads, and is skipped, not refused.
    ctx.prompt_responses.append(PromptResponse(prompt_id="ask1", response="done", step_id="one-turn"))
    assert usj.UserSimJudge("j", None, run_step_id="one-turn")._find_run(ctx) == ("prompt_agent", "one-turn")


# ---- the judge's calls, through litellm -----------------------------------------------------------------------------
@pytest.fixture
def litellm_config(monkeypatch):
    import agent_env.config as cfg_mod

    config = cfg_mod.Config()
    config.set_secret_store(LocalSecretStore(values={"litellm_api_key": "k"}, use_env=False))
    monkeypatch.setattr(config, "_model_cfg", ModelConfig())
    monkeypatch.setattr(cfg_mod, "get_config", lambda: config)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)


def _litellm_answers(monkeypatch, answer):
    """Each call answered by the real acompletion with ``answer(kwargs)`` as its mock_response (an exception raises)."""
    real, calls = litellm.acompletion, []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        got = answer(kwargs)
        if isinstance(got, Exception):
            raise got
        return await real(**kwargs, mock_response=got)

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    return calls


def _all_pass(kwargs) -> str:
    content = kwargs["messages"][-1]["content"]
    prompt = "".join(b["text"] for b in content) if isinstance(content, list) else content
    return json.dumps({"results": [{"id": c["id"], **PASS()} for c in _criteria_of(prompt)]})


async def _judge_through_litellm(monkeypatch, run, **kw) -> dict:
    monkeypatch.setattr(usj, "_read_run", lambda kind, step_id, record, context: (
        run, {"user_sim": "hannah", "instructions": {"matches": True}}))
    ctx = TaskStepContext()
    ctx.prompt_responses.append(PromptResponse(prompt_id="ask", response="all set", step_id="talk"))
    ctx.metadata["user_sim_reads"] = {"talk": [{}]}
    await usj.UserSimJudge("judge_user_sim", None, run_step_id="talk", verifier_id="usj", **kw).execute(ctx)
    return _entry(ctx)


@pytest.mark.asyncio
async def test_a_claude_judge_gets_the_system_message_and_the_run_marked_for_the_prompt_cache(monkeypatch,
                                                                                               litellm_config):
    calls = _litellm_answers(monkeypatch, _all_pass)
    run = _run(("reply", "ok"))

    entry = await _judge_through_litellm(monkeypatch, run, judge_model="anthropic/claude-sonnet-4-6")

    assert len(calls) == 7 and entry["user_sim_error"]["value"] == "no" and entry["calls"] == 7
    for call in calls:
        system, user = call["messages"]
        assert system == {"role": "system", "content": usj.SYSTEM}
        cached, rest = user["content"]
        assert cached["cache_control"] == {"type": "ephemeral"} and cached["text"].endswith(core.run_text(run))
        assert cached["text"].startswith('You check a user simulator ("the user sim")')
        assert rest["text"].startswith("\n\n# THE CRITERIA\n") and "cache_control" not in rest
    assert calls[0]["messages"][1]["content"][0] == calls[1]["messages"][1]["content"][0]
    assert entry["usage"]["prompt_tokens"] > 0 and entry["cost"] > 0


_GPT = dict(llm_provider="openai", model="openai/gpt-4o")
_RESPONSE = httpx.Response(400, request=httpx.Request("POST", "https://judge.invalid/v1/chat/completions"))
FLAKY = {
    "timeout": litellm.Timeout("timed out", **_GPT),
    "rate limit": litellm.RateLimitError("slow down", **_GPT),
    "service unavailable": litellm.ServiceUnavailableError("503", **_GPT),
    "internal server error": litellm.InternalServerError("500", **_GPT),
    "bad gateway": litellm.BadGatewayError("502", **_GPT),
    "connection": litellm.APIConnectionError("connection reset", **_GPT),
}
SETUP = {
    "auth": litellm.AuthenticationError("invalid api key", **_GPT),
    "permission": litellm.PermissionDeniedError("no access to this model", response=_RESPONSE, **_GPT),
    "unknown model": litellm.NotFoundError("model not found", **_GPT),
    "bad request": litellm.BadRequestError("LLM Provider NOT provided", **_GPT),
    "unprocessable": litellm.UnprocessableEntityError("unprocessable", response=_RESPONSE, **_GPT),
    "budget": litellm.BudgetExceededError(current_cost=10.0, max_budget=5.0),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("error", list(FLAKY.values()), ids=list(FLAKY))
async def test_a_flaky_model_call_is_tried_as_rubrics_verifier_tries_it_and_left_undecided(monkeypatch, litellm_config,
                                                                                            error):
    calls = _litellm_answers(monkeypatch, lambda kwargs: error)

    entry = await _judge_through_litellm(monkeypatch, _run(("reply", "ok")), judge_model="openai/gpt-4o")

    assert len(calls) == 7 * usj.UserSimJudge.DEFAULT_MAX_RETRIES  # every check, each call tried 4 times
    assert set(_verdicts(entry).values()) == {"error"}
    assert _rows(entry)["rogue"]["error"].startswith(f"no usable answer after 4 tries: {type(error).__name__}: ")
    # Nothing decided at all: the flag says to look at the judge's setup, and still nothing fails.
    assert entry["flag"] == "the judge couldn't decide any check \u2014 check its setup"
    assert entry["user_sim_error"]["value"] == "unknown" and entry["user_sim_error"]["fails_task"] is False


@pytest.mark.asyncio
async def test_one_undecided_check_among_decided_ones_is_flagged_undecided(monkeypatch, litellm_config):
    def answer(kwargs):
        content = kwargs["messages"][-1]["content"]
        prompt = "".join(b["text"] for b in content) if isinstance(content, list) else content
        rogue = any(c.get("check") == core.check_title("rogue") for c in _criteria_of(prompt))
        return litellm.RateLimitError("slow down", **_GPT) if rogue else _all_pass(kwargs)

    _litellm_answers(monkeypatch, answer)
    entry = await _judge_through_litellm(monkeypatch, _run(("reply", "ok")), judge_model="openai/gpt-4o")
    assert _verdicts(entry, "rogue", "guidance") == {"rogue": "error", "guidance": "pass"}
    assert entry["flag"] == "user sim judge undecided"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", list(SETUP.values()), ids=list(SETUP))
async def test_a_judge_that_can_t_reach_its_model_fails_the_run_at_once(monkeypatch, litellm_config, error):
    calls = _litellm_answers(monkeypatch, lambda kwargs: error)

    with pytest.raises(usj.JudgeSetupError, match=f"^the user sim judge can't reach its model: {type(error).__name__}: "):
        await _judge_through_litellm(monkeypatch, _run(("reply", "ok")), judge_model="openai/gpt-4o")

    assert len(calls) == usj.UserSimJudge.DEFAULT_MAX_RETRIES  # only the run's first call was made


@pytest.mark.asyncio
async def test_a_judge_with_no_key_or_config_fails_the_run_at_once(monkeypatch, litellm_config):
    from agent_env.config import get_config
    from agent_env.config.errors import ConfigError

    def missing(*args, **kwargs):
        raise ConfigError("no LiteLLM key is configured for the judge")

    monkeypatch.setattr(get_config(), "resolve_model_call", missing)
    with pytest.raises(usj.JudgeSetupError, match="can't reach its model: ConfigError: no LiteLLM key is configured"):
        await _judge_through_litellm(monkeypatch, _run(("reply", "ok")), judge_model="openai/gpt-4o")


# ---- the instructions, rebuilt as deploy_user_sim_agent wrote them ---------------------------------------------------
SEED = {"trip": "nov-19"}
TEXT = "ok mom and dad are officially coming thu-mon (nov 19-23), can you handle it?"
ITINERARY = {"kind": "file", "file": {"uri": "s3://reyes-family/<trip>/itinerary.pdf", "name": "itinerary.pdf",
                                      "mimeType": "application/pdf"},
             "metadata": {"description": "Mom and Dad's flight itinerary"}}
SHORTLIST = {"kind": "file", "file": {"uri": "s3://reyes-family/<trip>/<trip>-shortlist.xlsx"}}
TOLD_AND_KNOWN = {**HANNAH["description"], "already_told": ["Dad can't do stairs since his knee surgery."],
                  "should_already_know": ["You never book anything before 9am for your parents."]}


@pytest.mark.parametrize("fields, shows", [
    pytest.param({"opening_prompt": [{"kind": "text", "text": TEXT}, ITINERARY]},
                 ["(you attached itinerary.pdf: Mom and Dad's flight itinerary)", user_sim_versions.ATTACHED_KNOWN],
                 id="a-described-file"),
    pytest.param({"opening_prompt": [{"kind": "text", "text": TEXT}, SHORTLIST]},
                 ["(you attached nov-19-shortlist.xlsx)", user_sim_versions.ATTACHED_KNOWN], id="an-undescribed-file"),
    pytest.param({"persona": {**HANNAH, "description": TOLD_AND_KNOWN}},
                 ["  T1  Dad can't do stairs", "  C1  You never book anything before 9am"], id="told-and-known"),
    pytest.param({"opening_prompt": TEXT, "multi_day": False}, [], id="text-only"),
])
@pytest.mark.asyncio
async def test_the_rebuilt_instructions_are_the_ones_deployed(monkeypatch, fields, shows):
    deploy = _Deploy(monkeypatch)
    ctx = TaskStepContext()
    ctx.metadata["seed"] = dict(SEED)
    await deploy.run(_deploy_step(**fields), ctx)
    us = ctx.metadata[USER_SIMS_KEY]["hannah"]

    prompt, check, notes = usj._instructions(us, SEED, None)

    assert prompt == deploy.fields("system_prompt")[0] and all(line in prompt for line in shows)
    assert check["matches"] is True and check["prompt_sha256"] == us["prompt_sha256"] and notes == []


@pytest.mark.asyncio
async def test_a_text_only_opening_rebuilds_as_before(monkeypatch):
    deploy = _Deploy(monkeypatch)
    ctx = TaskStepContext()
    await deploy.run(_deploy_step(), ctx)
    us = ctx.metadata[USER_SIMS_KEY]["hannah"]
    assert "opening_parts" not in us
    person = {"key": "hannah", "name": HANNAH["display_name"], "first": HANNAH["first_name"],
              "description": HANNAH["description"], "permissions": HANNAH["permissions"],
              "keep_private": HANNAH["keep_private"], "opening": {"text": us["opening"]}, "done_ends_run": False}
    prompt, check, _ = usj._instructions(us, {}, None)
    assert prompt == user_sim_versions.setup("ours", person)["system_prompt"] == deploy.fields("system_prompt")[0]
    assert check["matches"] is True


# ---- settings -----------------------------------------------------------------------------------------------------
def test_settings_are_checked_and_round_trip():
    assert _builtin_registry()["user_sim_judge"] is usj.UserSimJudge
    step = usj.UserSimJudge("s", None, run_step_id="week", judge_model=JUDGE, fail_task_when_not_passing=False,
                            judge_timeout_seconds=300,
                            rubric_items=[{"id": "leak", "family": "G1", "text": "never mentions the ring"}])
    assert step.config_problems() == []
    again = usj.UserSimJudge.from_dict(json.loads(json.dumps(step.to_dict())))
    assert again.to_dict() == step.to_dict()
    own = set(step.to_dict()) - {"id", "type", "version", "fail_task_on_error"}
    assert own == set(usj.SETTINGS) == {"run_step_id", "rubric_items", "rubric_persona_sha256", "judge_model",
                                        "judge_timeout_seconds", "fail_task_when_not_passing", "verifier_id"}
    problems = usj.UserSimJudge("s", None, run_step_id="week", judge_model=" ", fail_task_when_not_passing="yes",
                                rubric_items=[{"id": "x", "family": "Q1", "text": "y"},
                                              {"id": "rogue", "family": "G3", "check": "stays the user"}]
                                ).config_problems()
    assert any("judge_model must be a model name" in p for p in problems) and any("family 'Q1'" in p for p in problems)
    assert any("fail_task_when_not_passing must be true or false" in p for p in problems)
    assert "rubric_items[1] (rogue): id 'rogue' is a check's name; give the item another id" in problems


@pytest.mark.asyncio
async def test_settings_with_problems_stop_the_step_before_it_judges(monkeypatch):
    with pytest.raises(ValueError, match=r"user_sim_judge 'judge_user_sim': judge_model must be a model name"):
        await _judge(_run(("reply", "x")), FakeModel(), monkeypatch, judge_model=" ")


STORED = {"id": "s", "type": "user_sim_judge", "run_step_id": "week"}


def test_a_task_without_run_step_id_is_refused_when_it_loads():
    for missing in ({"id": "s", "type": "user_sim_judge"}, {**STORED, "run_step_id": None}, {**STORED, "run_step_id": " "}):
        with pytest.raises(ValueError, match="user_sim_judge 's': run_step_id is required: name the always_on or "
                                             "prompt_agent step whose user sim it judges"):
            usj.UserSimJudge.from_dict(missing)


def test_checks_and_error_checks_are_refused_when_the_task_loads():
    for given, named in (({"checks": ["guidance"]}, "checks"), ({"error_checks": []}, "error_checks"),
                         ({"checks": list(core.CHECK_NAMES), "error_checks": ["rogue"]}, "checks and error_checks")):
        with pytest.raises(ValueError, match=f"user_sim_judge 's': {named} can't be set: the judge always asks all "
                                             "seven checks, and every check but style can fail a run"):
            usj.UserSimJudge.from_dict({**STORED, **given})


def test_a_rubrics_verifier_setting_the_judge_fixes_is_refused_unless_it_has_the_fixed_value():
    for key, value in (("use_trajectory", True), ("output_format", "rubric_binary"), ("judge_batch_size", 3),
                       ("use_agent_judge", True), ("criteria", [{"id": "c1"}]), ("prompt_id", "ask"),
                       ("default_model", "x"), ("score_aggregator", "weighted_average"), ("grade_up_to_turn", 2)):
        with pytest.raises(ValueError, match=f"user_sim_judge 's': {key} can't be set on the user sim judge, which "
                                             f"fixes it \\({key}="):
            usj.UserSimJudge.from_dict({**STORED, key: value})
    with pytest.raises(ValueError, match="use_trajectory, judge_batch_size can't be set .* fixes them .*Remove them; "
                                         "its settings are run_step_id, rubric_items"):
        usj.UserSimJudge.from_dict({**STORED, "use_trajectory": True, "judge_batch_size": 3})
    same = usj.UserSimJudge.from_dict({**STORED, **{k: usj.PINNED[k] for k in (
        "use_trajectory", "use_agent_judge", "output_format", "judge_batch_size", "criteria")}})
    assert same.config_problems() == []  # the fixed value itself is fine


def test_by_default_it_fails_the_task_on_a_slip_and_judges_with_the_verifiers_default_model(monkeypatch):
    step = usj.UserSimJudge.from_dict(STORED)
    assert (step.fail_task_when_not_passing, step.fail_task_on_error, step.judge_model) == (True, True, None)
    assert (step.use_trajectory, step.use_agent_judge, step.output_format) == (
        False, False, JudgeOutputFormat.USER_SIM_CHECKS)
    assert step.to_dict()["judge_model"] is None  # the task doesn't pin it

    class Config:
        role = None

        def get_model_for_role(self, role):
            assert role == "judge"
            return self.role

    fake = Config()
    monkeypatch.setattr("agent_env.task_step.task_steps.verifiers.rubrics_verifier.get_config", lambda: fake)
    assert usj.UserSimJudge("s", None, run_step_id="week").default_model == "claude-sonnet-4-6"
    fake.role = "anthropic/claude-opus-4-8"
    assert usj.UserSimJudge("s", None, run_step_id="week").default_model == "anthropic/claude-opus-4-8"


# ---- the load checks ----------------------------------------------------------------------------------------------
def _user_sim_steps():
    """A task's deploy_user_sim_agent step and the always_on run its user sim plays."""
    return [_deploy_step(id="deploy-u", agent_name="u"),
            ao.AlwaysOnTaskStep("week", None, user_sim="u", prompt_id="ask", agent_name="solver")]


def _grader(**kw) -> RubricsVerifierTaskStep:
    from agent_env.task_step.task_steps.verifiers.always_on_agent_verifier import AlwaysOnAgentVerifier

    return AlwaysOnAgentVerifier("grade", None, prompt_id="ask", verifier_id="grade", use_agent_judge=False,
                                 criteria=[{"id": "c1", "description": "booked it"}], **kw)


def test_its_run_must_be_a_run_step_the_judge_follows_and_its_user_a_built_in_user_sim():
    deploy, week = _user_sim_steps()
    judge = usj.UserSimJudge("j", None, run_step_id="week")
    assert judge.task_problems([deploy, week, judge]) == []
    assert judge.task_problems([deploy, week, _chat_step(), judge]) == []  # it names its run, so another doesn't matter
    [nowhere] = usj.UserSimJudge("j", None, run_step_id="nope").task_problems([deploy, week])
    assert nowhere == "user_sim_judge 'j': run_step_id 'nope' names no step of the task"
    other = AgentPromptResponseVerifierTaskStep("grade", None, prompt_id="ask")
    [named] = usj.UserSimJudge("j", None, run_step_id="grade").task_problems([deploy, week, other])
    assert "run_step_id 'grade' is a agent_prompt_response_verifier step; name the always_on or prompt_agent step to " \
           "judge" in named
    [outside] = judge.task_problems([week, judge])
    assert "the user of always_on step 'week' isn't a user sim a deploy_user_sim_agent step deploys" in outside
    [early] = judge.task_problems([deploy, judge, week])
    assert early == "user_sim_judge 'j' comes before step 'week', the run it judges; put it after"
    loose = usj.UserSimJudge("j", None, run_step_id="week", depends_on=[TaskStepDependency("deploy-u")])
    assert "its depends_on must name 'week'" in loose.task_problems([deploy, week, loose])[0]


def test_with_the_gate_on_every_step_that_can_be_the_reward_must_depend_on_the_judge():
    deploy, week = _user_sim_steps()
    judge = usj.UserSimJudge("judge", None, run_step_id="week", depends_on=[TaskStepDependency("week")])
    after = [TaskStepDependency("judge")]
    assert judge.task_problems([deploy, week, judge, _grader(depends_on=after)]) == []
    assert judge.task_problems([deploy, week, judge, _grader()]) == []  # no depends_on: every step before it
    [skips] = judge.task_problems([deploy, week, judge, _grader(depends_on=[TaskStepDependency("week")])])
    assert skips == ("step 'grade' (always_on_agent_verifier) can be the task's reward but doesn't depend on "
                     "user_sim_judge 'judge', so a run the user sim spoiled could still be scored; make it depend on "
                     "'judge', or set fail_task_when_not_passing to false to record the verdicts only")
    assert judge.task_problems([deploy, week, _grader(), judge]) == [skips]  # before it, so not after it
    off = usj.UserSimJudge("judge", None, run_step_id="week", fail_task_when_not_passing=False)
    assert off.task_problems([deploy, week, _grader(), off]) == []
    with pytest.raises(ValueError, match="can be the task's reward but doesn't depend on user_sim_judge 'judge'"):
        Task("t", None, [deploy, week, _grader(), judge])
    assert Task("t", None, [deploy, week, judge, _grader(depends_on=after)])  # an always_on task takes it


def test_a_task_that_sets_mark_user_sim_error_or_turns_off_fail_task_on_error_is_refused():
    with pytest.raises(ValueError, match="mark_user_sim_error is gone; fail_task_when_not_passing .* takes its place"):
        usj.UserSimJudge.from_dict({**STORED, "mark_user_sim_error": False})
    for gate in (True, False):  # with the gate off too: a judge that fails has to fail the task
        with pytest.raises(ValueError, match="user_sim_judge 's': fail_task_on_error can't be false"):
            usj.UserSimJudge.from_dict({**STORED, "fail_task_on_error": False, "fail_task_when_not_passing": gate})
    quiet = usj.UserSimJudge.from_dict({**STORED, "fail_task_on_error": True, "fail_task_when_not_passing": False})
    assert (quiet.fail_task_on_error, quiet.fail_task_when_not_passing) == (True, False)


# ---- task rubric items in usersim.yaml's shape ----------------------------------------------------------------------
# From psi-ts-craftweb-holiday-rush's usersim.yaml, as psi-benchmark-tasks turns it into this step's settings.
CALL_RAE = {
    "id": "a1_call_rae", "family": "A1", "kind": "hard",
    "name": "Rae: yes if it fits, ≥ $15 a jar and ≥ $9 a tin, by email",
    "short": "Both floors, 'if it fits', settled by email. Missing a floor, a different floor, or 'text her' fails.",
    "check": "If the assistant asks about Rae's order, Tess says yes only if it fits, gives BOTH floors (no less than $15 "
             "a jar and $9 a tin), and lets the assistant settle it with Rae by email. A different floor, a missing "
             "floor, a flat yes or no, or \"text her\" fails.",
    "na_when": "The assistant didn't raise Rae's order."}
NO_SILENT_FIX = {
    "id": "g5_no_silent_fix", "family": "G5", "kind": "general", "name": "Doesn't do the assistant's homework",
    "check": "When the assistant's plan has a gap, Tess gives her call rather than doing the work for it: she doesn't "
             "work out how many jars Rae should get, recompute the budget, choose pour dates, or rewrite its plan.",
    "na_when": "never"}
HER_LENGTH = {"id": "f2_her_length", "family": "F2", "kind": "auto", "name": "As short as her real texts",
              "check": "Every reply is at most 25 words.", "metric": "max_words", "max": 25}
SATISFACTION = {"id": "a5_satisfaction", "family": "A5", "kind": "hard",
                "name": "Satisfaction rating matches the outcome",
                "check": "Not gradeable: the sim gives no rating.", "na_when": "always",
                "static_na_reason": "No deployed version of the sim prompt has a satisfaction step."}
# Tess's real texts, from the same file's style_reference.
TESS = ("Never again. I ran out of oil and refunded 22 people at Christmas. Two 1-star reviews already 😩",
        "Saw it! Let me look at my numbers this weekend.")


def _items(*dicts) -> list[core.RubricItem]:
    assert core.item_problems(list(dicts)) == []
    return [core.RubricItem.from_dict(d) for d in dicts]


def _unjudged(plan: core.Plan) -> dict[str, dict]:
    return {r["id"]: r for r in core.finish_rows(plan.unjudged, plan)}


def test_the_craftweb_item_round_trips_and_is_asked_as_written():
    [item] = _items(CALL_RAE)
    assert item.to_dict() == CALL_RAE and core.RubricItem.from_dict(item.to_dict()) == item
    plan = _plan(_run(("reply", "x")), [item], checks=["guidance"])
    asked = plan.criteria[1]
    assert asked == {"id": "a1_call_rae", "item": "a1_call_rae", "family": "A1 (Requirement provision)",
                     "kind": "hard", "question": CALL_RAE["check"], "na_when": CALL_RAE["na_when"],
                     "graded_turns": [1], "weight": 1.0}
    assert CALL_RAE["name"] not in json.dumps(plan.criteria) and CALL_RAE["short"] not in json.dumps(plan.criteria)
    assert (plan.about["a1_call_rae"]["name"], plan.about["a1_call_rae"]["short"]) == (CALL_RAE["name"],
                                                                                     CALL_RAE["short"])
    step = usj.UserSimJudge("s", None, run_step_id="week", rubric_items=[CALL_RAE, HER_LENGTH, SATISFACTION],
                            rubric_persona_sha256=OTHER_PERSONA)
    again = usj.UserSimJudge.from_dict(json.loads(json.dumps(step.to_dict())))
    assert step.config_problems() == [] and again.to_dict() == step.to_dict()
    assert (again.rubric_items, again.rubric_persona_sha256) == ([CALL_RAE, HER_LENGTH, SATISFACTION], OTHER_PERSONA)


def test_the_converter_s_step_loads_with_its_fields_as_written():
    # psi-benchmark-tasks' tools/usersim_judge_step.py writes these fields; the gate needs fail_task_on_error.
    written = {"id": "psi-ts-x-usersim-judge", "type": "user_sim_judge", "version": None, "fail_task_on_error": True,
               "depends_on": [{"task_step_id": "psi-ts-x-prompt-step"}], "run_step_id": "psi-ts-x-prompt-step",
               "verifier_id": "psi_usersim", "rubric_persona_sha256": OTHER_PERSONA,
               "rubric_items": [CALL_RAE, HER_LENGTH, SATISFACTION, NO_SILENT_FIX]}
    step = usj.UserSimJudge.from_dict(written)
    assert step.config_problems() == [] and {k: step.to_dict()[k] for k in written} == written


def test_the_old_text_field_still_loads_as_the_check_and_kind_defaults_to_general():
    old = {"id": "leak", "family": "g1", "text": "never mentions the ring"}
    [item] = _items(old)
    assert (item.family, item.kind, item.check) == ("G1", "general", "never mentions the ring")
    assert item.to_dict() == {"id": "leak", "family": "G1", "kind": "general", "check": "never mentions the ring"}
    assert usj.UserSimJudge("s", None, run_step_id="week", rubric_items=[old]).config_problems() == []
    [both] = core.item_problems([{**old, "check": "never mentions the ring"}])
    assert both == "rubric_items[0] (leak): give check or text, not both (text is the old name of check)"


def test_hard_and_general_items_are_told_to_the_judge_by_kind():
    hard, general = _items(CALL_RAE, {**NO_SILENT_FIX, "family": "A3"})
    plan = _plan(_run(("reply", "x")), [hard, general], checks=["guidance"])
    assert [(c.get("kind"), c.get("na_when")) for c in plan.criteria] == [
        (None, None), ("hard", CALL_RAE["na_when"]), ("general", "never")]
    framing = get_judge_output_format_spec(JudgeOutputFormat.USER_SIM_CHECKS).prompt_template_inline.template
    assert 'A "hard" item is an exact rule' in framing and 'A "general" item is a judgement' in framing


@pytest.mark.asyncio
async def test_an_auto_item_is_counted_on_the_graded_answers_with_no_model_call(monkeypatch):
    [length] = _items(HER_LENGTH)
    # Too long for Tess: the length a model playing her drifts to.
    over = ("Yes to Rae if it fits, but no less than $15 a jar and $9 a tin. Settle it with her by email, and let me know "
            "what she says before you confirm anything.")
    run = _run(("reply", TESS[0]), ("no_reply", ""), ("reply", over), ("reply", TESS[1]))
    fake = FakeModel()
    entry = await _judge(run, fake, monkeypatch, rubric_items=[HER_LENGTH])
    assert all("f2_her_length" not in c["names"] for c in fake.calls) and entry["calls"] == len(fake.calls) == 7
    assert _rows(entry)["f2_her_length"] == {
        "id": "f2_her_length", "family": "F2", "kind": "auto", "name": "As short as her real texts",
        "check": "Every reply is at most 25 words.", "metric": "max_words", "max": 25, "asked_with": "computed",
        "weight": 0.0, "verdict": "fail", "reason": "over the 25 words max at turn 3: 34", "turns": [3],
        "could_change_grade": True, "assistant_asked": None, "score": 1.0, "result": True,
        "justification": "over the 25 words max at turn 3: 34"}
    assert entry["user_sim_error"]["value"] == "no"  # F2's check is style, so it is recorded only
    passed = _unjudged(_plan(_run(("reply", TESS[0]), ("reply", TESS[1])), [length], checks=[]))
    assert passed["f2_her_length"]["verdict"] == "pass"
    assert passed["f2_her_length"]["reason"] == "every answer at most 25 words; the longest, turn 1, has 18"
    # A context-only turn is never failed; a run with no graded message is n/a.
    context_only = _unjudged(_plan(_run(("reply", over), ("reply", TESS[1]), graded={2}), [length], checks=[]))
    assert context_only["f2_her_length"]["verdict"] == "pass"
    assert _unjudged(_plan(_run(("no_reply", "")), [length], checks=[]))["f2_her_length"]["verdict"] == "n/a"
    # One that can_error makes the run a user sim error.
    fake = FakeModel()
    entry = await _judge(_run(("reply", over)), fake, monkeypatch, rubric_items=[{**HER_LENGTH, "can_error": True}])
    assert all("f2_her_length" not in c["names"] for c in fake.calls) and len(fake.calls) == 7
    assert entry["user_sim_error"]["because"] == ["f2_her_length"]


def test_an_auto_item_can_t_pass_an_answer_the_record_kept_only_the_start_of():
    [length] = _items(HER_LENGTH)
    short = _run(("reply", TESS[0]), ("reply", TESS[1]))
    short.turns[1] = replace(short.turns[1], cut=True)  # under 25 words as kept, but the whole answer may not be
    row = _unjudged(_plan(short, [length], checks=[]))["f2_her_length"]
    assert (row["verdict"], row["turns"], row["asked_with"], row["error"]) == (
        "error", [2], "computed", "only the start of the answer at turn 2 was kept, so its full length can't be counted")
    # Over the max as kept is over it whole: still a fail.
    over = _run(("reply", ("word " * 30).strip()), ("reply", TESS[1]))
    over.turns[0] = replace(over.turns[0], cut=True)
    assert _unjudged(_plan(over, [length], checks=[]))["f2_her_length"]["verdict"] == "fail"


def test_a_reply_still_waiting_when_the_run_ended_is_flagged_as_never_delivered():
    timeline = [
        {"event": "user_sim_read", "outcome": "reply", "message": "fri at 7 works", "virtual_time": "2026-11-19T14:00:00Z",
         "deliver_at": "2026-11-19T14:20:00Z"},
        {"event": "user_reply_delivered", "virtual_time": "2026-11-19T14:20:00Z", "text": "fri at 7 works"},
        {"event": "user_sim_read", "outcome": "reply", "message": "dad can't do stairs",
         "virtual_time": "2026-11-19T21:00:00Z", "deliver_at": "2026-11-20T08:05:00Z"},
        {"event": "user_sim_read", "outcome": "no_reply", "message": "", "virtual_time": "2026-11-20T07:00:00Z"},
    ]
    reads = usj._with_delivery(timeline, timeline)
    assert [r["held"] for r in reads] == [False, True, False]
    assert usj.FLAG_HELD in usj._turn(2, reads[1]).flags and usj.FLAG_HELD not in usj._turn(1, reads[0]).flags
    # The last read's reply is flagged as such, not twice.
    last = {**timeline[2], "last_read": True}
    flags = usj._turn(2, usj._with_delivery([last], [last])[0]).flags
    assert usj.FLAG_LAST in flags and usj.FLAG_HELD not in flags


def test_from_the_tail_the_check_ins_it_dropped_are_counted():
    tail = [{"event": "user_sim_read", "outcome": "reply", "message": "sounds good"}] * 2
    reads, notes, missing = usj._always_on_reads({"timeline_tail": tail, "counts": {"user_sim_reads": 5}})
    assert len(reads) == 2 and missing == 3
    assert notes[-1] == "the first 3 of the run's 5 check-ins aren't kept, so they aren't shown"
    assert usj._always_on_reads({"timeline_tail": tail, "counts": {"user_sim_reads": 2}})[2] == 0


def test_an_auto_item_can_t_pass_when_some_answers_weren_t_kept():
    [length] = _items(HER_LENGTH)
    run = _run(("reply", TESS[0]), ("reply", TESS[1]))
    run.missing_answers = 3
    row = _unjudged(_plan(run, [length], checks=[]))["f2_her_length"]
    assert (row["verdict"], row["turns"], row["error"]) == (
        "error", [], "3 of the run's answers weren't kept, so they can't be counted")
    long = _run(("reply", "word " * 30), ("reply", TESS[1]))
    long.missing_answers = 3  # a kept answer over the max is a fail either way
    assert _unjudged(_plan(long, [length], checks=[]))["f2_her_length"]["verdict"] == "fail"


def test_an_auto_item_counts_sentences_simply():
    assert [core.count_sentences(t) for t in TESS] == [3, 2]
    assert core.count_sentences("no less than $15.50 a jar") == 1  # a decimal point ends nothing
    assert core.count_sentences("ok\nsounds good") == 2 and core.count_sentences("😩") == 0
    assert core.count_words(TESS[0]) == 18  # the emoji is a word
    [two] = _items({"id": "f2_short", "family": "F2", "kind": "auto", "metric": "max_sentences", "max": 2})
    row = _unjudged(_plan(_run(("reply", TESS[1]), ("reply", TESS[0])), [two], checks=[]))["f2_short"]
    assert (row["verdict"], row["turns"], row["reason"]) == ("fail", [2], "over the 2 sentences max at turn 2: 3")


@pytest.mark.asyncio
async def test_na_when_always_is_n_a_with_its_reason_and_no_call(monkeypatch):
    fake = FakeModel()
    entry = await _judge(_run(("done", "thx")), fake, monkeypatch, rubric_items=[SATISFACTION])
    row = _rows(entry)["a5_satisfaction"]
    assert len(fake.calls) == 7 and not fake.asked("a5_satisfaction") and (row["verdict"], row["reason"], row["asked_with"]) == (
        "n/a", SATISFACTION["static_na_reason"], "static_na")


@pytest.mark.asyncio
async def test_na_when_never_doesn_t_allow_an_n_a_so_one_is_asked_again(monkeypatch):
    fake = FakeModel({"g5_no_silent_fix": [NA(), PASS(reason="she gave her call")]})
    entry = await _judge(_run(("reply", "your call on the dates, just not oct 31")), fake, monkeypatch,
                         rubric_items=[NO_SILENT_FIX])
    asked = fake.asked("g5_no_silent_fix")
    assert _rows(entry)["g5_no_silent_fix"]["verdict"] == "pass" and len(asked) == 2
    assert "may not be n/a: it always applies, so answer pass or fail" in asked[1]["prompt"]
    never = await _judge(_run(("reply", "x")), FakeModel({"g5_no_silent_fix": [NA()] * 4}), monkeypatch,
                         rubric_items=[NO_SILENT_FIX])
    assert _verdicts(never, "rogue", "g5_no_silent_fix") == {"rogue": "error", "g5_no_silent_fix": "error"}
    # Where n/a is allowed, it stands.
    got = await _judge(_run(("reply", "x")), FakeModel({"a1_call_rae": NA(reason="Rae never came up")}), monkeypatch,
                       rubric_items=[CALL_RAE])
    assert _rows(got)["a1_call_rae"]["verdict"] == "n/a"


def test_a_big_rubric_is_split_over_calls_of_at_most_the_cap():
    assert core.MAX_ITEMS_PER_CALL == 12
    items = _items(*[{"id": f"g1_{n}", "family": "G1", "check": f"never states fact {n}"} for n in range(30)],
                   *[{"id": f"a5_{n}", "family": "A5", "check": f"rates outcome {n}"} for n in range(13)], HER_LENGTH)
    calls = core.plan_calls(["oversharing", "rogue"], items)
    assert {k: len(v) for k, v in calls.items()} == {"oversharing": 12, "oversharing#2": 12, "oversharing#3": 6,
                                                     "rogue": 0, "task_items": 12, "task_items#2": 1}
    assert [i.id for i in calls["oversharing#3"]] == [f"g1_{n}" for n in range(24, 30)]
    plan = _plan(_run(("reply", "x")), items, checks=["oversharing", "rogue"])
    assert plan.batches == [13, 12, 6, 1, 12, 1]  # each check's own call holds the check and its first 12 items
    assert [c["id"] for c in plan.criteria if "check" in c] == ["oversharing", "rogue"]  # each check is asked once


@pytest.mark.asyncio
async def test_split_items_get_their_rows_and_the_check_is_asked_once(monkeypatch):
    items = [{"id": f"g1_{n}", "family": "G1", "check": f"never states fact {n}"} for n in range(14)]
    fake = FakeModel({"g1_13": FAIL()})
    entry = await _judge(_run(("reply", "x")), fake, monkeypatch, rubric_items=items)
    split = [c for c in fake.calls if any(n.startswith("g1_") for n in c["names"])]
    assert [len(c["names"]) for c in split] == [13, 2] and split[0]["names"][0] == "oversharing"
    assert len(fake.asked("oversharing")) == 1 and "These are criteria 15–16 of 21" in split[1]["prompt"]
    assert _rows(entry)["g1_13"]["asked_with"] == "oversharing#2" and _rows(entry)["g1_13"]["verdict"] == "fail"
    assert [r["id"] for r in entry["results"]] == [*core.CHECK_NAMES, *[f"g1_{n}" for n in range(14)]]
    assert entry["user_sim_error"]["because"] == ["g1_13"]  # G1's owner, oversharing, is an error check


def test_each_bad_item_is_refused_when_the_task_loads_with_a_clear_error():
    cases = [
        ({"family": "A1", "check": "x"}, "rubric_items[0] needs an id"),
        ({"id": "a", "check": "x"}, "rubric_items[0] (a) needs a family"),
        ({"id": "a", "family": "B1", "check": "x"}, "rubric_items[0] (a): family 'B1' is not one of"),
        ({"id": "a", "family": "A1", "kind": "strict", "check": "x"}, "kind 'strict' is not one of ['hard', 'general', "
                                                                      "'auto']"),
        ({"id": "a", "family": "F2", "kind": "auto", "max": 25}, "an auto item's metric must be one of"),
        ({"id": "a", "family": "F2", "kind": "auto", "metric": "max_chars", "max": 25}, "not 'max_chars'"),
        ({"id": "a", "family": "F2", "kind": "auto", "metric": "max_words"}, "an auto item needs max"),
        ({"id": "a", "family": "F2", "kind": "auto", "metric": "max_words", "max": 0}, "not 0"),
        ({"id": "a", "family": "F2", "kind": "auto", "metric": "max_words", "max": "25"}, "not '25'"),
        ({"id": "a", "family": "F2", "kind": "general", "check": "x", "metric": "max_words", "max": 25},
         "metric and max are for kind auto only"),
        ({"id": "a", "family": "A5", "check": "x", "na_when": "always"}, "na_when always needs a static_na_reason"),
        ({"id": "a", "family": "A5", "check": "x", "static_na_reason": "y"}, "static_na_reason is only for na_when "
                                                                            "always"),
        ({"id": "a", "family": "A1", "kind": "hard", "check": "  "}, "a hard item needs a check"),
        ({"id": "a", "family": "A1"}, "a general item needs a check"),
        ({"id": "a", "family": "A1", "check": "x", "na-when": "never"}, "unknown fields ['na-when']"),
        ({"id": "a", "family": "F2", "kind": "auto", "metric": "max_words", "max": 25, "na_when": "never"},
         "auto items are counted, so na_when can only be 'always' (with a static_na_reason), not 'never'"),
        ({"id": "a", "family": "F2", "kind": "auto", "metric": "max_words", "max": 25,
          "na_when": "The assistant never asked her anything."},
         "auto items are counted, so na_when can only be 'always'"),
    ]
    for item, error in cases:
        problems = core.item_problems([item])
        assert any(error in p for p in problems), (item, problems)
        assert any(error in p for p in usj.UserSimJudge("s", None, run_step_id="week",
                                                        rubric_items=[item]).config_problems())
    dup = core.item_problems([CALL_RAE, {**CALL_RAE, "family": "A3"}])
    assert dup == ["rubric_items[1] (a1_call_rae): id 'a1_call_rae' is used twice"]
    for bad in (7, "3f1c", "A" * 64):
        assert any("rubric_persona_sha256 must be a persona fingerprint" in p for p in usj.UserSimJudge(
            "s", None, run_step_id="week", rubric_items=[CALL_RAE], rubric_persona_sha256=bad).config_problems())


def test_an_auto_item_loads_with_no_na_when_or_with_always():
    auto = {"id": "a", "family": "F2", "kind": "auto", "metric": "max_words", "max": 25}
    assert core.item_problems([auto]) == []
    assert core.item_problems([{**auto, "na_when": "always", "static_na_reason": "She never texts here."}]) == []


# ---- the rubric's persona ------------------------------------------------------------------------------------------
@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_items_written_for_another_persona_aren_t_graded_and_the_checks_still_run(monkeypatch):
    ctx = await _chat(monkeypatch)
    run = ctx.metadata[USER_SIMS_KEY]["hannah"]["persona_sha256"]
    fake = FakeModel({"guidance": FAIL()})

    with pytest.raises(VerifierNotPassingError, match=r"\(guidance\)"):
        await _step(fake, monkeypatch, rubric_items=[CALL_RAE, HER_LENGTH],
                    rubric_persona_sha256=OTHER_PERSONA).execute(ctx)

    entry = _entry(ctx)
    assert entry["rubric_persona"] == {
        "rubric": OTHER_PERSONA, "run": run, "status": "mismatch",
        "note": "the task items were written for a different persona or opening than this run's user sim had, so they "
                "weren't graded; the general checks were"}
    assert list(_rows(entry)) == list(core.CHECK_NAMES) and len(fake.calls) == 7
    assert all("a1_call_rae" not in c["names"] for c in fake.calls)
    assert entry["user_sim_error"]["because"] == ["guidance"]


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
@pytest.mark.parametrize("recorded, status", [(True, "match"), (False, "unknown")])
async def test_items_for_this_persona_or_a_run_with_no_fingerprint_are_graded(monkeypatch, recorded, status):
    ctx = await _chat(monkeypatch)
    us = ctx.metadata[USER_SIMS_KEY]["hannah"]
    fingerprint = us["persona_sha256"] if recorded else OTHER_PERSONA
    if not recorded:
        del us["persona_sha256"]  # a run from before the step recorded it

    await _step(FakeModel(), monkeypatch, rubric_items=[HER_LENGTH],
                rubric_persona_sha256=fingerprint).execute(ctx)

    entry = _entry(ctx)
    assert entry["rubric_persona"]["status"] == status and _rows(entry)["f2_her_length"]["verdict"] == "pass"
    assert entry["rubric_persona"]["rubric"] == fingerprint
    if not recorded:
        assert entry["rubric_persona"]["run"] is None and "graded as written" in entry["rubric_persona"]["note"]
    # Without a rubric_persona_sha256 nothing is compared.
    await _step(FakeModel(), monkeypatch, rubric_items=[HER_LENGTH]).execute(ctx)
    assert "rubric_persona" not in _entry(ctx) and _rows(_entry(ctx))["f2_her_length"]["verdict"] == "pass"


# ---- items that point at a persona line ------------------------------------------------------------------------------
ENGAGEMENT = {"id": "g1_engagement", "family": "G1", "kind": "hard", "line": "P1"}
RING = "Ben proposed last month. Never mention or hint at it."


def _lined_run(tagged_lines=None) -> core.UserSimRun:
    return replace(_run(("reply", "just a family dinner")), tagged_lines=tagged_lines)


def test_the_guidance_question_holds_it_to_its_already_told_and_should_already_know_lines():
    q = core.check_question("guidance")
    assert "THINGS YOU ALREADY TOLD THE ASSISTANT lines (T)" in q and "remind it, a little annoyed" in q
    assert "THINGS THE ASSISTANT SHOULD ALREADY KNOW lines (C) wrong, and it didn't correct it briefly" in q
    assert "even where its other lines say to let that kind of mistake pass" in q
    assert "Correcting a THINGS THE ASSISTANT SHOULD ALREADY KNOW line (C)" in q.split("Not a failure:")[1]


@pytest.mark.asyncio
async def test_a_line_item_is_graded_by_that_line_s_text_from_the_run_and_records_it(monkeypatch):
    [item] = _items(ENGAGEMENT)
    assert item.to_dict() == ENGAGEMENT and item.check == ""
    fake = FakeModel({"g1_engagement": FAIL()})
    entry = await _judge(_lined_run({"W1": "Hannah.", "P1": RING}), fake, monkeypatch,
                         rubric_items=[ENGAGEMENT], rubric_persona_sha256=OTHER_PERSONA)
    assert fake.asked("g1_engagement")[0]["criteria"][1]["question"] == f"Follows its instruction P1: {RING}"
    row = _rows(entry)["g1_engagement"]
    assert (row["line"], row["line_text"], row["asked_with"], row["verdict"], row["kind"]) == (
        "P1", RING, "oversharing", "fail", "hard")
    assert entry["user_sim_error"]["because"] == ["g1_engagement"]


def test_a_line_s_later_lines_stay_under_the_item():
    [item] = _items({**ENGAGEMENT, "line": "W2"})
    plan = _plan(_lined_run({"W2": 'Earlier you texted the assistant:\n"sort the weekend"'}), [item],
                 checks=["oversharing"])
    assert plan.criteria[1]["question"] == 'Follows its instruction W2: Earlier you texted the assistant:\n  "sort the ' \
                                           'weekend"'


@pytest.mark.parametrize("tagged_lines, error", [
    ({"P1": RING}, "the user sim's instructions have no line P2"),
    (None, "the run recorded no tagged_lines for the user sim, so its line P2 can't be read"),
])
def test_a_line_the_run_doesn_t_have_is_an_error_never_a_guess(tagged_lines, error):
    [item] = _items({**ENGAGEMENT, "line": "P2"})
    plan = _plan(_lined_run(tagged_lines), [item], checks=["oversharing"])
    row = _unjudged(plan)["g1_engagement"]
    assert (row["verdict"], row["turns"], row["line"], row["error"], row["asked_with"]) == (
        "error", [], "P2", error, core.NO_LINE)
    assert [c["id"] for c in plan.criteria] == ["oversharing"]  # it makes no call
    assert core.user_sim_error(core.finish_rows(plan.unjudged, plan)) == {
        "value": "unknown", "because": [], "unknown": ["g1_engagement"]}


def test_a_line_item_is_refused_with_a_check_on_an_auto_item_or_without_the_persona_fingerprint():
    cases = [
        ({**ENGAGEMENT, "check": "Never mentions the ring."}, "give line or check, not both"),
        ({**ENGAGEMENT, "text": "Never mentions the ring."}, "give line or check, not both"),
        ({**ENGAGEMENT, "kind": "auto", "metric": "max_words", "max": 25},
         "line is for hard and general items: an auto item is counted, not judged"),
        ({**ENGAGEMENT, "line": "p1"}, "line must be a tag of the user sim's instructions, like P1 or D3, not 'p1'"),
        ({**ENGAGEMENT, "line": "P"}, "not 'P'"),
        ({**ENGAGEMENT, "line": 1}, "line must be text"),
    ]
    for item, error in cases:
        problems = core.item_problems([item])
        assert any(error in p for p in problems), (item, problems)
    [no_check] = core.item_problems([{"id": "a", "family": "A1"}])
    assert no_check == ("rubric_items[0] (a): a general item needs a check (the text the judge applies) or a line (a "
                        "tag of the user sim's instructions, like P1)")
    assert usj.UserSimJudge("s", None, run_step_id="week", rubric_items=[ENGAGEMENT],
                            rubric_persona_sha256=OTHER_PERSONA).config_problems() == []
    assert usj.UserSimJudge("s", None, run_step_id="week", rubric_items=[ENGAGEMENT]).config_problems() == [
        "rubric_items[0] (g1_engagement): a line item needs rubric_persona_sha256, so its tag is graded only on runs "
        "of the persona it was written for"]


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_on_a_run_a_line_item_reads_the_deploy_record_s_tagged_lines(monkeypatch):
    ctx = await _chat(monkeypatch)
    us = ctx.metadata[USER_SIMS_KEY]["hannah"]
    fake = FakeModel()

    await _step(fake, monkeypatch, rubric_items=[ENGAGEMENT],
                rubric_persona_sha256=us["persona_sha256"]).execute(ctx)

    [call] = fake.asked("g1_engagement")
    assert call["criteria"][1]["question"] == f"Follows its instruction P1: {us['tagged_lines']['P1']}"
    row = _rows(_entry(ctx))["g1_engagement"]
    assert (row["line"], row["line_text"], row["asked_with"]) == ("P1", us["tagged_lines"]["P1"], "oversharing")


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_on_a_run_with_no_persona_fingerprint_a_line_item_is_an_error(monkeypatch):
    ctx = await _chat(monkeypatch)
    del ctx.metadata[USER_SIMS_KEY]["hannah"]["persona_sha256"]
    fake = FakeModel()

    await _step(fake, monkeypatch, rubric_items=[ENGAGEMENT, HER_LENGTH],
                rubric_persona_sha256=OTHER_PERSONA).execute(ctx)

    rows = _rows(_entry(ctx))
    assert rows["g1_engagement"]["verdict"] == "error" and rows["f2_her_length"]["verdict"] == "pass"
    assert rows["g1_engagement"]["error"] == "line P1 can't be graded: " + usj.LINE_UNCHECKED
    assert not fake.asked("g1_engagement") and len(fake.calls) == 7
    assert _entry(ctx)["user_sim_error"]["value"] == "unknown" and _entry(ctx)["flag"] == "user sim judge undecided"
