"""user_sim_judge: grades the user sim of an always_on run or a prompt_agent chat run, one narrow question per check,
by one judge model, with a fake model standing in for it. It records only, unless marking is turned on."""

from __future__ import annotations

import hashlib
import json

import pytest

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import _builtin_registry
from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.agent import user_sim_versions
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import USER_SIMS_KEY
from agent_env.task_step.task_steps.verifiers import user_sim_judge as usj
from agent_env.task_step.task_steps.verifiers import user_sim_judge_core as core
from agent_env.task_step.task_steps.verifiers.agent_prompt_response_verifier import AgentPromptResponseVerifierTaskStep
from tst.unit.task_step.always_on_harness import Rig
from tst.unit.task_step.test_deploy_user_sim_agent import HANNAH, _Deploy
from tst.unit.task_step.test_deploy_user_sim_agent import _step as _deploy_step
from tst.unit.task_step.test_prompt_agent_built_in_user_sim import _context as _chat_context
from tst.unit.task_step.test_prompt_agent_built_in_user_sim import _step as _chat_step
from tst.unit.task_step.test_prompt_agent_built_in_user_sim import said
from tst.unit.task_step.test_prompt_agent_user_side import AGENT, USER, Peers, text

JUDGE = "fake/judge"
TASK_ITEMS_Q = "task's own rubric items"


def _pass(**kw) -> str:
    return json.dumps({"verdict": "pass", "could_change_grade": False, "reason": "fine", "turns": [], "items": [],
                       **kw})


def _fail(turns=(1,), change=True, **kw) -> str:
    return json.dumps({"verdict": "fail", "could_change_grade": change, "reason": "it broke the rule",
                       "turns": list(turns), "items": [], **kw})


class FakeModel:
    """Answers each call from ``script[check]``, a text, an exception, or a list of them used in turn; anything
    unscripted passes. Records every call."""

    def __init__(self, script=None):
        self.script = dict(script or {})
        self.calls: list[dict] = []

    async def __call__(self, model, system, parts, schema):
        check = next(c for c in [*core.CHECK_NAMES, TASK_ITEMS_Q] if self._asks(parts[-1], c))
        self.calls.append({"model": model, "check": check, "system": system, "parts": parts, "schema": schema})
        got = self.script.get(check, _pass())
        if isinstance(got, list):
            got = got.pop(0)
        if isinstance(got, Exception):
            raise got
        return {"text": got, "usage": {"prompt_tokens": 100, "completion_tokens": 20}, "cost": 0.001}

    @staticmethod
    def _asks(question: str, check: str) -> bool:
        if check in core.CHECKS:
            return question.startswith(f"# YOUR QUESTION: {core.CHECKS[check].title} (")
        return question.startswith(f"# YOUR QUESTION: the {check}")

    def run_text(self) -> str:
        """The run as the judge's first call read it."""
        return self.calls[0]["parts"][0] + "\n" + self.calls[0]["parts"][1]


def _run(*answers: tuple[str, str], graded=None) -> core.UserSimRun:
    turns = [core.Turn(n=i, outcome=o, message=m, input=f"It's Mon. NEW FROM YOUR AI ASSISTANT: question {i}?",
                       graded=graded is None or i in graded) for i, (o, m) in enumerate(answers, 1)]
    return core.UserSimRun(name="hannah", instructions="You are Hannah Reyes. K1 Dad is allergic to shellfish.",
                           turns=turns)


async def _judge(run, fake, **kw):
    kw.setdefault("model", JUDGE)
    kw.setdefault("checks", list(core.CHECK_NAMES))
    return await core.judge_run(run, complete=fake, **kw)


# ---- what the judge is asked --------------------------------------------------------------------------------------
def test_each_check_gets_one_narrow_question_naming_its_families():
    for name, check in core.CHECKS.items():
        q = core.question_text(name, [])
        assert q.startswith(f"# YOUR QUESTION: {check.title} ({', '.join(check.families)})\n{check.question}")
        assert all(f in q for f in check.fails) and "Return only the JSON object." in q
    assert sorted(core.FAMILY_OWNER) == ["A1", "A2", "A3", "A4", "F1", "F2", "F3", "G1", "G2", "G3", "G5"]


def test_the_run_is_split_before_the_last_graded_answer_so_repeats_share_a_cached_prefix():
    run = _run(("reply", "yes book it"), ("no_reply", ""), ("reply", "no seafood pls"), graded={3})
    head, tail = core.render_run(run)
    assert "<instructions>\nYou are Hannah Reyes." in head
    assert "## Turn 1 · CONTEXT ONLY" in head and 'message: "yes book it"' in head
    assert "## Turn 3 · GRADED" in head and "question 3?" in head and "no seafood pls" not in head
    assert tail == '### Its answer\noutcome: reply\nmessage: "no seafood pls"'
    assert "message:" not in head.split("## Turn 2")[1].split("## Turn 3")[0]  # a no_reply shows no empty message


def test_task_items_go_with_the_check_that_owns_their_family_and_the_rest_get_a_call_of_their_own():
    items = [core.RubricItem("leak", "G1", "never mentions the engagement"),
             core.RubricItem("ends", "G4", "signs off only after the recap"),
             core.RubricItem("tone", "F2", "stays lowercase")]
    calls = core.plan_calls(["oversharing", "made_up"], items)
    assert [i.id for i in calls["oversharing"]] == ["leak"] and calls["made_up"] == []
    assert [i.id for i in calls[core.TASK_ITEMS]] == ["ends", "tone"]  # style isn't running, so "tone" goes there too
    assert "- leak (G1: Leaking and privacy): never mentions the engagement" in core.question_text(
        "oversharing", calls["oversharing"])


def test_rubric_items_are_checked_when_the_task_loads():
    bad = [{"id": "a", "family": "Z9", "text": "x"}, {"id": "a", "family": "G1", "text": "y", "weight": 2}, {"id": ""}]
    problems = core.item_problems(bad)
    assert any("family 'Z9'" in p for p in problems) and any("used twice" in p for p in problems)
    assert any("unknown fields ['weight']" in p for p in problems) and any("needs an id" in p for p in problems)


# ---- reading an answer --------------------------------------------------------------------------------------------
def test_an_answer_is_checked_and_its_turns_kept_to_the_graded_ones():
    got = core.parse_answer(_fail(turns=[2, 3, "x"]), graded={3}, item_ids=[])
    assert (got["verdict"], got["turns"], got["turns_outside_graded"]) == ("fail", [3], [2])
    assert core.parse_answer(_pass(could_change_grade=True), {1}, [])["could_change_grade"] is False
    for bad, why in ((json.dumps({"verdict": "maybe"}), "verdict"), ("no json here", "not a JSON object"),
                     (_pass(), "items \\['i1'\\] have no answer")):
        with pytest.raises(ValueError, match=why):
            core.parse_answer(bad, {1}, ["i1"] if "items" in why else [])


# ---- one judge ----------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_each_check_s_verdict_is_recorded_with_its_reason_and_turns():
    na = json.dumps({"verdict": "n/a", "could_change_grade": False, "reason": "no rules", "turns": [], "items": []})
    fake = FakeModel({"made_up": _fail(), "style": _fail(change=False), "delegation": na})
    result = await _judge(_run(("reply", "dad's allergic to shellfish fyi")), fake)
    assert {k: v["verdict"] for k, v in result["checks"].items()} == {
        "guidance": "pass", "oversharing": "pass", "delegation": "n/a", "style": "fail", "rogue": "pass",
        "made_up": "fail"}
    assert result["checks"]["made_up"] == {"verdict": "fail", "could_change_grade": True, "reason": "it broke the rule",
                                           "turns": [1], "families": ["A2"]}
    assert result["checks"]["delegation"]["reason"] == "no rules"
    # Style failed, but it isn't an error check; made_up failed in a way that could change the grade.
    assert result["user_sim_error"] == {"value": "yes", "because": ["made_up"], "unknown": []}
    assert len(fake.calls) == 6 == result["calls"] and {c["model"] for c in fake.calls} == {JUDGE}
    assert result["usage"]["prompt_tokens"] == 600 and result["cost"] == pytest.approx(0.006)


@pytest.mark.asyncio
async def test_each_call_shares_the_run_and_asks_one_question_and_the_first_call_runs_alone():
    fake = FakeModel()
    await _judge(_run(("reply", "ok")), fake)
    assert {c["parts"][0] for c in fake.calls} == {fake.calls[0]["parts"][0]}  # one shared, cacheable prefix
    assert all(c["system"] == core.INSTRUCTIONS and c["schema"] == core.OUTPUT_SCHEMA for c in fake.calls)
    assert [c["check"] for c in fake.calls] == list(core.CHECK_NAMES)  # the first alone, then the rest in order


@pytest.mark.asyncio
async def test_a_failure_that_can_t_change_the_grade_doesn_t_make_a_user_sim_error():
    result = await _judge(_run(("reply", "Here's a summary of what I booked:")), FakeModel({"rogue": _fail(change=False)}))
    assert result["checks"]["rogue"]["verdict"] == "fail" and result["user_sim_error"]["value"] == "no"


@pytest.mark.asyncio
async def test_a_bad_answer_is_asked_again_once_and_a_check_whose_calls_keep_failing_is_an_error():
    fake = FakeModel({"guidance": ["not json", _fail()], "rogue": [RuntimeError("503"), RuntimeError("503")]})
    result = await _judge(_run(("reply", "you pick")), fake, checks=["guidance", "rogue"])
    assert result["checks"]["guidance"]["verdict"] == "fail" and result["checks"]["guidance"]["attempts"] == 2
    retry = [c for c in fake.calls if c["check"] == "guidance"][1]["parts"][-1]
    assert "Your last answer could not be used (answer not usable: the answer is not a JSON object)" in retry
    assert result["checks"]["rogue"] == {"verdict": "error", "error": "call failed: RuntimeError: 503",
                                         "attempts": 2, "families": ["G3", "G5"]}
    assert result["errors"] == ["rogue: call failed: RuntimeError: 503"]
    assert result["user_sim_error"] == {"value": "yes", "because": ["guidance"], "unknown": ["rogue"]}
    alone = await _judge(_run(("reply", "x")), FakeModel({"rogue": [RuntimeError("503")] * 2}), checks=["rogue"])
    assert alone["user_sim_error"] == {"value": "unknown", "because": [], "unknown": ["rogue"]}


@pytest.mark.asyncio
async def test_task_items_get_their_own_verdicts_and_one_that_can_error_makes_a_user_sim_error():
    items = [core.RubricItem("leak", "G1", "never hints at the engagement"),
             core.RubricItem("ends", "G4", "signs off only after the recap", can_error=True)]
    item_fail = json.dumps({"verdict": "fail", "could_change_grade": False, "reason": "signed off early",
                            "turns": [1], "items": [{"id": "ends", "verdict": "fail", "reason": "early",
                                                     "turns": [1]}]})
    leak_pass = _pass(items=[{"id": "leak", "verdict": "pass", "reason": "-", "turns": []}])
    fake = FakeModel({TASK_ITEMS_Q: item_fail, "oversharing": leak_pass})
    result = await _judge(_run(("done", "thx bye")), fake, items=items)
    assert result["items"]["leak"] == {"verdict": "pass", "reason": "-", "turns": [], "family": "G1",
                                       "asked_with": "oversharing"}
    assert result["items"]["ends"] == {"verdict": "fail", "reason": "early", "turns": [1], "family": "G4",
                                       "asked_with": "task_items"}
    assert "task_items" not in result["checks"] and result["user_sim_error"]["because"] == ["item:ends"]
    assert len(fake.calls) == 7


@pytest.mark.asyncio
async def test_a_run_with_no_graded_turn_is_not_sent_to_the_judge():
    fake = FakeModel()
    result = await _judge(_run(("reply", "ok"), graded=set()), fake)
    assert result["skipped"] == "no graded turns" and fake.calls == []


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


def _step(fake: FakeModel, monkeypatch, **kw) -> usj.UserSimJudgeTaskStep:
    step = usj.UserSimJudgeTaskStep("judge_user_sim", None, **{"judge_model": JUDGE, "verifier_id": "usj", **kw})
    monkeypatch.setattr(step, "_complete", lambda context: fake)
    return step


def _entry(ctx: TaskStepContext) -> dict:
    return ctx.metadata["user_sim_judge"]["usj"]


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_on_an_always_on_run_the_judge_reads_what_the_user_sim_got_and_said_and_only_records(monkeypatch,
                                                                                                tmp_path):
    rig, ctx, prompt = await _always_on(monkeypatch, tmp_path)
    fake = FakeModel({"guidance": _fail(turns=[2])})

    await _step(fake, monkeypatch).execute(ctx)

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
    assert {k: v["verdict"] for k, v in entry["checks"].items()} == {**dict.fromkeys(core.CHECK_NAMES, "pass"),
                                                                     "guidance": "fail"}
    assert entry["checks"]["guidance"]["turns"] == [2]
    # A grade-changing failure, recorded only: nothing a grader reads changes.
    assert entry["user_sim_error"] == {"value": "yes", "because": ["guidance"], "unknown": [], "marked": False,
                                       "not_marked": "mark_user_sim_error is off: verdicts are recorded only"}
    assert ctx.prompt_responses[-1].error_type is None and "user_sim_judge" not in rig.record(ctx)
    assert "verifications" not in ctx.metadata  # never mixed into the assistant's scores
    assert ctx.metadata["models"]["user sim judge"]["judge_user_sim"] == JUDGE


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_without_the_stored_timeline_the_judge_reads_the_tail_and_is_told_what_s_missing(monkeypatch, tmp_path):
    rig, ctx, _ = await _always_on(monkeypatch, tmp_path)
    record, timeline = rig.record(ctx), rig.timeline(ctx)
    record.pop("object_url")
    record["timeline_tail"] = [{k: v for k, v in e.items() if k not in ("input", "read")} for e in timeline]
    fake = FakeModel()

    await _step(fake, monkeypatch, checks=["made_up"]).execute(ctx)

    head = fake.calls[0]["parts"][0]
    assert "Note: the run's full record wasn't stored" in head and "### What it was sent\n(not recorded)" in head
    assert "## Turn 1 · 2026-11-09T14:00:00Z UTC · GRADED" in head


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_run_that_ended_in_an_error_is_not_judged(monkeypatch, tmp_path):
    rig, ctx, _ = await _always_on(monkeypatch, tmp_path, answers=[])
    rig.record(ctx)["error_type"] = "agent_turn_failed"
    fake = FakeModel()

    await _step(fake, monkeypatch, mark_user_sim_error=True).execute(ctx)

    assert fake.calls == [] and _entry(ctx)["skipped"] == "the run ended with error_type=agent_turn_failed"
    assert _entry(ctx)["user_sim_error"] == {"value": "no", "because": [], "unknown": [], "marked": False}


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_run_s_step_params_choose_the_checks_and_the_model(monkeypatch, tmp_path):
    _, ctx, _ = await _always_on(monkeypatch, tmp_path)
    ctx.metadata["user_overrides"] = {"step_params": {"judge_user_sim": {"checks": ["rogue"], "judge_model": "other"}}}
    fake = FakeModel()

    await _step(fake, monkeypatch).execute(ctx)

    assert {(c["check"], c["model"]) for c in fake.calls} == {("rogue", "other")}
    assert _entry(ctx)["error_checks"] == ["rogue"] and _entry(ctx)["judge_model"] == "other"


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
    fake = FakeModel({"style": _fail(turns=[2], change=False)})

    await _step(fake, monkeypatch, run_step_id="talk").execute(ctx)

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
    assert entry["checks"]["style"]["verdict"] == "fail" and entry["user_sim_error"]["value"] == "no"
    assert entry["graded_turns"] == [1, 2] and ctx.prompt_responses[-1].error_type is None


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_with_marking_turned_on_a_user_sim_error_marks_the_prompt_so_later_graders_skip_it(monkeypatch):
    ctx = await _chat(monkeypatch)

    await _step(FakeModel({"oversharing": _fail()}), monkeypatch, mark_user_sim_error=True).execute(ctx)

    assert _entry(ctx)["user_sim_error"] == {"value": "yes", "because": ["oversharing"], "unknown": [],
                                             "marked": True}
    response = ctx.prompt_responses[-1]
    assert response.error_type == "user_sim_error"
    assert response.error_message == "The user sim judge found a user sim error: oversharing"
    grader = AgentPromptResponseVerifierTaskStep("grade", None, prompt_id=response.prompt_id, verifier_id="grade",
                                                 criteria=[{"type": "response_contains", "needles": ["glad"]}])
    await grader.execute(ctx)
    assert ctx.metadata["verifications"]["grade"]["results"][0]["message"] == "Skipped: prompt had error_type=user_sim_error"


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_marking_can_be_turned_on_for_one_run_by_its_step_params(monkeypatch):
    ctx = await _chat(monkeypatch)
    ctx.metadata["user_overrides"] = {"step_params": {"judge_user_sim": {"mark_user_sim_error": True}}}

    await _step(FakeModel({"rogue": _fail()}), monkeypatch).execute(ctx)

    assert _entry(ctx)["user_sim_error"]["marked"] is True and ctx.prompt_responses[-1].error_type == "user_sim_error"


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_instructions_that_don_t_match_the_recorded_hash_are_flagged_and_never_mark_the_run(monkeypatch):
    ctx = await _chat(monkeypatch)
    us = ctx.metadata[USER_SIMS_KEY]["hannah"]
    us["description"] = {**us["description"], "who_you_are": [*us["description"]["who_you_are"], "A changed line."]}
    fake = FakeModel({"guidance": _fail()})

    await _step(fake, monkeypatch, mark_user_sim_error=True).execute(ctx)

    entry = _entry(ctx)
    assert entry["instructions"]["matches"] is False
    assert entry["instructions"]["recorded_prompt_sha256"] == us["prompt_sha256"] != entry["instructions"]["prompt_sha256"]
    assert "Note: these instructions were rebuilt from the user sim's settings and differ from the ones it got" in (
        fake.calls[0]["parts"][0])
    assert entry["user_sim_error"]["value"] == "yes" and entry["user_sim_error"]["marked"] is False
    assert "isn't confirmed" in entry["user_sim_error"]["not_marked"] and ctx.prompt_responses[-1].error_type is None


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
    await _step(fake, monkeypatch, run_step_id="talk").execute(ctx)
    assert fake.calls == [] and _entry(ctx)["skipped"] == "the user sim never read: the run has no user_sim_reads"


def test_without_run_step_id_the_run_must_have_one_run_to_judge():
    ctx = TaskStepContext()
    ctx.metadata = {"always_on": {"week": {}}, "user_sim_reads": {"talk": [{}]}}
    with pytest.raises(RuntimeError, match=r"set run_step_id; this run has 2 runs .*\['talk', 'week'\]"):
        usj.UserSimJudgeTaskStep("j", None)._find_run(ctx)
    with pytest.raises(RuntimeError, match="step 'nope' left no run to judge"):
        usj.UserSimJudgeTaskStep("j", None, run_step_id="nope")._find_run(ctx)
    assert usj.UserSimJudgeTaskStep("j", None, run_step_id="talk")._find_run(ctx) == ("prompt_agent", "talk")


# ---- settings -----------------------------------------------------------------------------------------------------
def test_settings_are_checked_and_round_trip(monkeypatch):
    assert "user_sim_judge" in _builtin_registry()
    step = usj.UserSimJudgeTaskStep("s", None, run_step_id="week", judge_model=JUDGE, checks=["guidance", "style"],
                                    error_checks=["guidance"], mark_user_sim_error=True,
                                    rubric_items=[{"id": "leak", "family": "G1", "text": "never mentions the ring"}])
    assert step.config_problems() == []
    again = usj.UserSimJudgeTaskStep.from_dict(json.loads(json.dumps(step.to_dict())))
    assert again.to_dict() == step.to_dict()
    problems = usj.UserSimJudgeTaskStep("s", None, checks=["guidance", "vibes"], error_checks=["rogue"], judge_model=" ",
                                        mark_user_sim_error="yes",
                                        rubric_items=[{"id": "x", "family": "Q1", "text": "y"}]).config_problems()
    assert any("unknown checks ['vibes']" in p for p in problems)
    assert any("error_checks names checks that don't run: ['rogue']" in p for p in problems)
    assert any("judge_model must be a model name" in p for p in problems) and any("family 'Q1'" in p for p in problems)
    assert any("mark_user_sim_error must be true or false" in p for p in problems)
    assert usj.UserSimJudgeTaskStep("s", None, checks=["style"]).error_checks == []  # the error checks that run


def test_by_default_it_records_only_never_fails_the_task_and_judges_with_the_verifiers_default_model(monkeypatch):
    step = usj.UserSimJudgeTaskStep.from_dict({"id": "s", "type": "user_sim_judge"})
    assert (step.mark_user_sim_error, step.fail_task_on_error, step.judge_model) == (False, False, None)
    assert step.to_dict()["judge_model"] is None  # resolved when it runs, so the task doesn't pin it
    class Config:
        role = None

        def get_model_for_role(self, role):
            assert role == "judge"
            return self.role

    fake = Config()
    monkeypatch.setattr("agent_env.config.get_config", lambda: fake)
    assert step._settings(TaskStepContext())["model"] == "claude-sonnet-4-6"
    fake.role = "anthropic/claude-opus-4-8"
    assert step._settings(TaskStepContext())["model"] == "anthropic/claude-opus-4-8"


def test_the_task_must_say_which_run_when_it_has_more_than_one_and_name_a_run_step():
    week = ao.AlwaysOnTaskStep("week", None, user_sim="u", prompt_id="ask", agent_name="solver")
    talk = _chat_step()
    other = AgentPromptResponseVerifierTaskStep("grade", None, prompt_id="ask")
    assert usj.UserSimJudgeTaskStep("j", None).task_problems([week]) == []
    assert "set run_step_id" in usj.UserSimJudgeTaskStep("j", None).task_problems([week, talk])[0]
    assert usj.UserSimJudgeTaskStep("j", None, run_step_id="talk").task_problems([week, talk]) == []
    assert "of step 'grade' (agent_prompt_response_verifier)" in usj.UserSimJudgeTaskStep(
        "j", None, run_step_id="grade").task_problems([week, other])[0]


def test_the_step_asks_anthropic_models_to_cache_the_run():
    assert usj._anthropic("anthropic/claude-opus-4-8") and usj._anthropic("claude-opus-4-8")
    assert not usj._anthropic("openai/gpt-5.5")
