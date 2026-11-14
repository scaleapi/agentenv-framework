"""user_sim_judge: grades the user sim of an always_on run or a prompt_agent chat run, one narrow question per check,
by one judge model, with a fake model standing in for it. It records only, unless marking is turned on."""

from __future__ import annotations

import hashlib
import json
import re

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
OTHER_PERSONA = "0" * 64  # a persona fingerprint no run here has
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
    assert sorted(core.FAMILY_OWNER) == ["A1", "A2", "A3", "A4", "F1", "F2", "F3", "F4", "G1", "G2", "G3", "G4", "G5"]


def test_self_contradiction_is_asked_under_made_up_not_guidance():
    line = "It contradicted something it said earlier in the run."
    assert line in core.question_text("made_up", []) and "contradict" not in core.question_text("guidance", [])


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
    assert "- leak (G1: Leaking and privacy; general): never mentions the engagement" in core.question_text(
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


def test_a_fail_that_cites_no_graded_turn_is_not_usable_and_a_pass_needs_none():
    with pytest.raises(ValueError, match=r"a fail must cite at least one graded turn; cited \[99\], graded \[1\]"):
        core.parse_answer(_fail(turns=[99]), {1}, [])
    with pytest.raises(ValueError, match=r"a fail must cite at least one graded turn; cited \[\], graded \[1\]"):
        core.parse_answer(_fail(turns=[]), {1}, [])
    item = {"id": "i1", "verdict": "fail", "reason": "rated it 5", "turns": [99]}
    with pytest.raises(ValueError, match=r"item i1: a fail must cite at least one graded turn; cited \[99\]"):
        core.parse_answer(_fail(items=[item]), {1}, ["i1"])
    na = {"verdict": "n/a", "could_change_grade": False, "reason": "-", "turns": [99], "items": []}
    for ok in (_pass(turns=[99]), json.dumps(na), _pass(items=[{**item, "verdict": "pass"}])):
        assert core.parse_answer(ok, {1}, ["i1"] if "i1" in ok else [])["turns"] == []
    got = core.parse_answer(_fail(turns=[1, 99], items=[{**item, "turns": [1, 99]}]), {1}, ["i1"])
    assert (got["turns"], got["turns_outside_graded"], got["items"]["i1"]["turns"]) == ([1], [99], [1])


def test_the_judge_writes_its_reason_before_its_verdict_within_the_word_limit():
    order = ["reason", "turns", "verdict", "could_change_grade", "items"]
    assert list(core.OUTPUT_SCHEMA["properties"]) == order == core.OUTPUT_SCHEMA["required"]
    assert list(core.OUTPUT_SCHEMA["properties"]["items"]["items"]["properties"]) == ["id", "reason", "turns", "verdict"]
    shape = core.INSTRUCTIONS.split("Return only this JSON object")[1]
    assert shape.index('"reason"') < shape.index('"turns"') < shape.index('"verdict"') < shape.index(
        '"could_change_grade"')
    item = shape.split('"items": [')[1]
    assert item.index('"reason"') < item.index('"turns"') < item.index('"verdict"')
    assert "Reason first and decide last." in core.INSTRUCTIONS and "at most 40 words" in core.INSTRUCTIONS
    # The parser doesn't care which order the keys come in.
    flipped = json.dumps({"items": [], "could_change_grade": True, "verdict": "fail", "turns": [1], "reason": "leaked"})
    assert core.parse_answer(flipped, {1}, [])["verdict"] == "fail"


# The endings of the two contradictory reasons seen in real runs, and a plain one.
SAYS_NO_FAILURE = ("Hannah flagged it anyway which is fine. No clear failure found.",
                   "This is not a delegation failure by Linda. No clear delegation failures found.",
                   "Every answer fits her rules; nothing fails.")


def test_a_fail_whose_reason_says_nothing_failed_is_not_usable():
    for reason in SAYS_NO_FAILURE:
        with pytest.raises(ValueError, match="a fail whose reason says .* contradicts itself"):
            core.parse_answer(_fail(reason=reason), {1}, [])
        item = {"id": "i1", "verdict": "fail", "reason": reason, "turns": [1]}
        with pytest.raises(ValueError, match="item i1: a fail whose reason says"):
            core.parse_answer(_pass(items=[item]), {1}, ["i1"])
        # A pass may say it; so may an item that passes.
        assert core.parse_answer(_pass(reason=reason, items=[{**item, "verdict": "pass"}]), {1}, ["i1"])["reason"]
    # An ordinary fail, even one that clears some turns, is kept.
    ok = "Turn 1 is fine, not a failure. At turn 2 she volunteered the shellfish allergy unasked; M3 says only when asked."
    assert core.parse_answer(_fail(turns=[2], reason=ok), {1, 2}, [])["reason"] == ok


# ---- one judge ----------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_each_check_s_verdict_is_recorded_with_its_reason_and_turns():
    na = json.dumps({"verdict": "n/a", "could_change_grade": False, "reason": "no rules", "turns": [], "items": []})
    fake = FakeModel({"made_up": _fail(), "style": _fail(change=False), "delegation": na})
    result = await _judge(_run(("reply", "dad's allergic to shellfish fyi")), fake)
    assert {k: v["verdict"] for k, v in result["checks"].items()} == {
        "guidance": "pass", "oversharing": "pass", "delegation": "n/a", "style": "fail", "rogue": "pass",
        "made_up": "fail", "ending": "pass"}
    assert result["checks"]["made_up"] == {"verdict": "fail", "could_change_grade": True, "reason": "it broke the rule",
                                           "turns": [1], "families": ["A2"]}
    assert result["checks"]["delegation"]["reason"] == "no rules"
    # Style failed, but it isn't an error check; made_up failed in a way that could change the grade.
    assert result["user_sim_error"] == {"value": "yes", "because": ["made_up"], "unknown": []}
    assert len(fake.calls) == 7 == result["calls"] and {c["model"] for c in fake.calls} == {JUDGE}
    assert result["usage"]["prompt_tokens"] == 700 and result["cost"] == pytest.approx(0.007)


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
async def test_a_fail_citing_only_ungraded_turns_is_asked_again_and_twice_is_an_error():
    fake = FakeModel({"made_up": [_fail(turns=[99]), _fail(turns=[1])]})
    result = await _judge(_run(("reply", "dad's allergic to shellfish")), fake, checks=["made_up"])
    assert result["checks"]["made_up"]["verdict"] == "fail" and result["checks"]["made_up"]["turns"] == [1]
    assert result["checks"]["made_up"]["attempts"] == 2 and result["user_sim_error"]["value"] == "yes"
    assert "a fail must cite at least one graded turn; cited [99], graded [1]" in fake.calls[1]["parts"][-1]
    twice = await _judge(_run(("reply", "x")), FakeModel({"made_up": [_fail(turns=[99])] * 2}), checks=["made_up"])
    assert twice["checks"]["made_up"]["verdict"] == "error" and "cited [99]" in twice["checks"]["made_up"]["error"]
    assert twice["user_sim_error"] == {"value": "unknown", "because": [], "unknown": ["made_up"]}


@pytest.mark.asyncio
async def test_a_fail_whose_reason_says_no_clear_failure_found_is_asked_again():
    contradicts = _fail(reason=SAYS_NO_FAILURE[0])
    fake = FakeModel({"guidance": [contradicts, _pass(reason="No clear failure found.")], "oversharing": _fail()})
    result = await _judge(_run(("reply", "dad's allergic to shellfish fyi")), fake, checks=["guidance", "oversharing"])
    assert result["checks"]["guidance"] == {"verdict": "pass", "could_change_grade": False,
                                            "reason": "No clear failure found.", "turns": [], "attempts": 2,
                                            "families": ["A1", "A3"]}
    retry = [c for c in fake.calls if c["check"] == "guidance"][1]["parts"][-1]
    assert "a fail whose reason says 'No clear failure found' contradicts itself" in retry
    # The ordinary fail is answered once and alone makes the run a user sim error.
    assert "attempts" not in result["checks"]["oversharing"] and result["checks"]["oversharing"]["verdict"] == "fail"
    assert result["user_sim_error"] == {"value": "yes", "because": ["oversharing"], "unknown": []}
    twice = await _judge(_run(("reply", "x")), FakeModel({"delegation": [contradicts] * 2}), checks=["delegation"])
    assert twice["checks"]["delegation"]["verdict"] == "error"
    assert twice["user_sim_error"] == {"value": "unknown", "because": [], "unknown": ["delegation"]}


@pytest.mark.asyncio
async def test_a_task_item_fail_citing_no_graded_turn_is_asked_again_and_twice_is_an_error():
    items = [core.RubricItem("rating", "A5", "rates the trip 2 or lower", can_error=True)]

    def answer(turns):
        return json.dumps({"verdict": "fail", "could_change_grade": False, "reason": "rated it 5", "turns": [1],
                           "items": [{"id": "rating", "verdict": "fail", "reason": "rated it 5", "turns": turns}]})

    fake = FakeModel({TASK_ITEMS_Q: [answer([99]), answer([1])]})
    result = await _judge(_run(("done", "5 stars")), fake, checks=[], items=items)
    assert result["items"]["rating"]["verdict"] == "fail" and result["items"]["rating"]["turns"] == [1]
    assert result["user_sim_error"]["because"] == ["item:rating"] and len(fake.calls) == 2
    twice = await _judge(_run(("done", "5 stars")), FakeModel({TASK_ITEMS_Q: [answer([])] * 2}), checks=[],
                         items=items)
    assert twice["items"]["rating"]["verdict"] == "error" and "item rating: a fail" in twice["items"]["rating"]["error"]
    assert twice["user_sim_error"] == {"value": "unknown", "because": [], "unknown": ["item:rating"]}


@pytest.mark.asyncio
async def test_task_items_get_their_own_verdicts_and_one_that_can_error_makes_a_user_sim_error():
    items = [core.RubricItem("leak", "G1", "never hints at the engagement"),
             core.RubricItem("rating", "A5", "rates the trip 2 or lower", can_error=True)]
    item_fail = json.dumps({"verdict": "fail", "could_change_grade": False, "reason": "rated it 5",
                            "turns": [1], "items": [{"id": "rating", "verdict": "fail", "reason": "rated it 5",
                                                     "turns": [1]}]})
    leak_pass = _pass(items=[{"id": "leak", "verdict": "pass", "reason": "-", "turns": []}])
    fake = FakeModel({TASK_ITEMS_Q: item_fail, "oversharing": leak_pass})
    result = await _judge(_run(("done", "thx bye")), fake, items=items)
    assert result["items"]["leak"] == {"verdict": "pass", "reason": "-", "turns": [], "family": "G1",
                                       "kind": "general", "asked_with": "oversharing"}
    assert result["items"]["rating"] == {"verdict": "fail", "reason": "rated it 5", "turns": [1], "family": "A5",
                                         "kind": "general", "asked_with": "task_items"}
    assert "task_items" not in result["checks"] and result["user_sim_error"]["because"] == ["item:rating"]
    assert len(fake.calls) == 8


@pytest.mark.asyncio
async def test_the_ending_check_asks_when_it_signed_off_and_a_grade_changing_fail_makes_a_user_sim_error():
    q = core.question_text("ending", [])
    assert "while the assistant was waiting on its answer" in q and "WHEN YOU'RE DONE" in q
    assert "A short sign-off when its reply limit is reached." in q and "never said \"done\"" in q
    assert "ending" in core.DEFAULT_ERROR_CHECKS
    na = json.dumps({"verdict": "n/a", "could_change_grade": False, "reason": "never said done", "turns": [],
                     "items": []})
    for answer, verdict, error in ((_fail(turns=[1]), "fail", "yes"), (_pass(), "pass", "no"), (na, "n/a", "no")):
        result = await _judge(_run(("done", "ok thx bye")), FakeModel({"ending": answer}), checks=["ending"])
        assert result["checks"]["ending"]["verdict"] == verdict and result["checks"]["ending"]["families"] == ["G4"]
        assert result["user_sim_error"]["value"] == error


def test_the_judge_is_told_when_the_run_ended_at_the_user_sim_s_reply_limit():
    turns = _run(("reply", "swap it"), ("no_reply", ""), ("reply", "ok"), ("reply", "thx")).turns
    [note] = usj._reply_limit_note(turns, 2)
    assert note.startswith("the run ended at the user sim's reply limit: its reply at turn 3 was the last of its 2")
    assert usj._reply_limit_note(turns, 4) == usj._reply_limit_note(turns, None) == []


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


def test_a_fail_that_clears_one_turn_and_fails_another_is_kept():
    text = json.dumps({"reason": "No clear failure at turn 1; at turn 2 she revealed the private password.",
                       "turns": [1], "verdict": "fail", "could_change_grade": True, "items": []})
    parsed = core.parse_answer(text, {1}, [])
    assert parsed["verdict"] == "fail" and parsed["could_change_grade"] is True


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


def _answer_items(verdicts: dict[str, str], turns=(1,)):
    """A complete() that answers every item it's asked, by id, with ``verdicts`` (else pass)."""
    calls = []

    async def complete(model, system, parts, schema):
        asked = re.findall(r"^- (\S+) \([AFG]\d: ", parts[-1], re.M)
        calls.append(asked)
        items = [{"id": i, "reason": f"{i} per turn 1", "turns": list(turns), "verdict": verdicts.get(i, "pass")}
                 for i in asked]
        return {"text": _pass(items=items), "usage": None, "cost": 0.0}

    complete.calls = calls
    return complete


def test_the_craftweb_item_round_trips_and_is_asked_as_written():
    [item] = _items(CALL_RAE)
    assert item.to_dict() == CALL_RAE and core.RubricItem.from_dict(item.to_dict()) == item
    q = core.question_text("guidance", [item])
    assert f"- a1_call_rae (A1: Requirement provision; hard): {CALL_RAE['check']}" in q
    assert "  n/a when: The assistant didn't raise Rae's order." in q
    assert CALL_RAE["name"] not in q and CALL_RAE["short"] not in q  # for reviewers, not the judge
    step = usj.UserSimJudgeTaskStep("s", None, rubric_items=[CALL_RAE, HER_LENGTH, SATISFACTION],
                                    rubric_persona_sha256=OTHER_PERSONA)
    again = usj.UserSimJudgeTaskStep.from_dict(json.loads(json.dumps(step.to_dict())))
    assert step.config_problems() == [] and again.to_dict() == step.to_dict()
    assert (again.rubric_items, again.rubric_persona_sha256) == ([CALL_RAE, HER_LENGTH, SATISFACTION], OTHER_PERSONA)


def test_the_old_text_field_still_loads_as_the_check_and_kind_defaults_to_general():
    old = {"id": "leak", "family": "g1", "text": "never mentions the ring"}
    [item] = _items(old)
    assert (item.family, item.kind, item.check) == ("G1", "general", "never mentions the ring")
    assert item.to_dict() == {"id": "leak", "family": "G1", "kind": "general", "check": "never mentions the ring"}
    assert usj.UserSimJudgeTaskStep("s", None, rubric_items=[old]).config_problems() == []
    [both] = core.item_problems([{**old, "check": "never mentions the ring"}])
    assert both == "rubric_items[0] (leak): give check or text, not both (text is the old name of check)"


def test_hard_and_general_items_are_told_to_the_judge_by_kind():
    hard, general = _items(CALL_RAE, {**NO_SILENT_FIX, "family": "A3"})
    q = core.question_text("guidance", [hard, general])
    assert core.KIND_RULES["hard"] in q and "fails on any breach" in q
    assert core.KIND_RULES["general"] in q and "fails only on a clear instance" in q
    assert "- g5_no_silent_fix (A3: Decisions and preferences; general):" in q and "  n/a is not allowed" in q
    only_general = core.question_text("guidance", [general])
    assert core.KIND_RULES["hard"] not in only_general and core.KIND_RULES["general"] in only_general


@pytest.mark.asyncio
async def test_an_auto_item_is_counted_on_the_graded_answers_with_no_model_call():
    [length] = _items(HER_LENGTH)
    # Too long for Tess: the length a model playing her drifts to.
    over = ("Yes to Rae if it fits, but no less than $15 a jar and $9 a tin. Settle it with her by email, and let me know "
            "what she says before you confirm anything.")
    run = _run(("reply", TESS[0]), ("no_reply", ""), ("reply", over), ("reply", TESS[1]))
    fake = FakeModel()
    result = await _judge(run, fake, checks=[], items=[length])
    assert fake.calls == [] and result["calls"] == 0
    assert result["items"]["f2_her_length"] == {
        "verdict": "fail", "reason": "over the 25 words max at turn 3: 34", "turns": [3], "family": "F2", "kind": "auto",
        "asked_with": "computed"}
    passed = await _judge(_run(("reply", TESS[0]), ("reply", TESS[1])), fake, checks=[], items=[length])
    assert passed["items"]["f2_her_length"]["verdict"] == "pass" and fake.calls == []
    assert passed["items"]["f2_her_length"]["reason"] == "every answer at most 25 words; the longest, turn 1, has 18"
    # A context-only turn is never failed; a run with no graded message is n/a.
    context_only = await _judge(_run(("reply", over), ("reply", TESS[1]), graded={2}), fake, checks=[], items=[length])
    assert context_only["items"]["f2_her_length"]["verdict"] == "pass"
    silent = await _judge(_run(("no_reply", "")), fake, checks=[], items=[length])
    assert silent["items"]["f2_her_length"]["verdict"] == "n/a"
    # With the checks running, the auto item is in no question.
    fake = FakeModel()
    await _judge(_run(("reply", over)), fake, items=[length])
    assert all("f2_her_length" not in c["parts"][-1] for c in fake.calls) and len(fake.calls) == 7


@pytest.mark.asyncio
async def test_an_auto_item_counts_sentences_simply():
    assert [core.count_sentences(t) for t in TESS] == [3, 2]
    assert core.count_sentences("no less than $15.50 a jar") == 1  # a decimal point ends nothing
    assert core.count_sentences("ok\nsounds good") == 2 and core.count_sentences("😩") == 0
    assert core.count_words(TESS[0]) == 18  # the emoji is a word
    [two] = _items({"id": "f2_short", "family": "F2", "kind": "auto", "metric": "max_sentences", "max": 2})
    result = await _judge(_run(("reply", TESS[1]), ("reply", TESS[0])), FakeModel(), checks=[], items=[two])
    assert result["items"]["f2_short"]["verdict"] == "fail" and result["items"]["f2_short"]["turns"] == [2]
    assert result["items"]["f2_short"]["reason"] == "over the 2 sentences max at turn 2: 3"


@pytest.mark.asyncio
async def test_na_when_always_is_n_a_with_its_reason_and_no_call():
    [rating] = _items(SATISFACTION)
    fake = FakeModel()
    result = await _judge(_run(("done", "thx")), fake, checks=[], items=[rating])
    assert fake.calls == [] and result["items"]["a5_satisfaction"] == {
        "verdict": "n/a", "reason": SATISFACTION["static_na_reason"], "turns": [], "family": "A5", "kind": "hard",
        "asked_with": "static_na"}
    fake = FakeModel()
    await _judge(_run(("done", "thx")), fake, items=[rating])
    assert all("a5_satisfaction" not in c["parts"][-1] for c in fake.calls)


@pytest.mark.asyncio
async def test_na_when_never_doesn_t_allow_an_n_a_so_one_is_asked_again():
    [fix] = _items(NO_SILENT_FIX)

    def answer(verdict):
        item = {"id": "g5_no_silent_fix", "reason": "she gave her call", "turns": [1], "verdict": verdict}
        return _pass(items=[item])

    fake = FakeModel({"rogue": [answer("n/a"), answer("pass")]})
    result = await _judge(_run(("reply", "your call on the dates, just not oct 31")), fake, checks=["rogue"],
                          items=[fix])
    assert result["items"]["g5_no_silent_fix"]["verdict"] == "pass" and len(fake.calls) == 2
    assert "item g5_no_silent_fix may not be n/a: it always applies" in fake.calls[1]["parts"][-1]
    twice = await _judge(_run(("reply", "x")), FakeModel({"rogue": [answer("n/a")] * 2}), checks=["rogue"],
                         items=[fix])
    assert twice["items"]["g5_no_silent_fix"]["verdict"] == "error"
    # Where n/a is allowed, it stands.
    [rae] = _items(CALL_RAE)
    na = _pass(items=[{"id": "a1_call_rae", "reason": "Rae never came up", "turns": [], "verdict": "n/a"}])
    got = await _judge(_run(("reply", "x")), FakeModel({"guidance": na}), checks=["guidance"], items=[rae])
    assert got["items"]["a1_call_rae"]["verdict"] == "n/a"


def test_a_big_rubric_is_split_over_calls_of_at_most_the_cap():
    assert core.MAX_ITEMS_PER_CALL == 12
    items = _items(*[{"id": f"g1_{n}", "family": "G1", "check": f"never states fact {n}"} for n in range(30)],
                   *[{"id": f"a5_{n}", "family": "A5", "check": f"rates outcome {n}"} for n in range(13)], HER_LENGTH)
    calls = core.plan_calls(["oversharing", "rogue"], items)
    assert {k: len(v) for k, v in calls.items()} == {"oversharing": 12, "oversharing#2": 12, "oversharing#3": 6,
                                                     "rogue": 0, "task_items": 12, "task_items#2": 1}
    assert [i.id for i in calls["oversharing#3"]] == [f"g1_{n}" for n in range(24, 30)]
    more = core.question_text("oversharing#2", calls["oversharing#2"])
    assert more.startswith("# YOUR QUESTION: the task's own rubric items (more of the Oversharing question's, G1, G2)")
    assert core.CHECKS["oversharing"].question not in more  # the check itself is asked once


@pytest.mark.asyncio
async def test_split_items_get_their_verdicts_and_the_check_is_asked_once():
    items = _items(*[{"id": f"g1_{n}", "family": "G1", "check": f"never states fact {n}"} for n in range(14)])
    complete = _answer_items({"g1_13": "fail"})
    result = await core.judge_run(_run(("reply", "x")), model=JUDGE, checks=["oversharing"], items=items,
                                  complete=complete)
    assert [len(c) for c in complete.calls] == [12, 2] and list(result["checks"]) == ["oversharing"]
    assert result["items"]["g1_13"]["asked_with"] == "oversharing#2" and result["items"]["g1_13"]["verdict"] == "fail"
    assert list(result["items"]) == [f"g1_{n}" for n in range(14)]  # in the task's order
    assert result["user_sim_error"]["because"] == ["item:g1_13"]  # G1's owner, oversharing, is an error check


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
        with pytest.raises(ValueError, match="user_sim_judge 's': "):
            usj.UserSimJudgeTaskStep("s", None, rubric_items=[item])._settings(TaskStepContext())
    dup = core.item_problems([CALL_RAE, {**CALL_RAE, "family": "A3"}])
    assert dup == ["rubric_items[1] (a1_call_rae): id 'a1_call_rae' is used twice"]
    for bad in (7, "3f1c", "A" * 64):
        assert any("rubric_persona_sha256 must be a persona fingerprint" in p for p in usj.UserSimJudgeTaskStep(
            "s", None, rubric_items=[CALL_RAE], rubric_persona_sha256=bad).config_problems())


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
    fake = FakeModel({"guidance": _fail()})

    await _step(fake, monkeypatch, rubric_items=[CALL_RAE, HER_LENGTH], rubric_persona_sha256=OTHER_PERSONA).execute(ctx)

    entry = _entry(ctx)
    assert entry["rubric_persona"] == {
        "rubric": OTHER_PERSONA, "run": run, "status": "mismatch",
        "note": "the task items were written for a different persona or opening than this run's user sim had, so they "
                "weren't graded; the general checks were"}
    assert entry["items"] == {} and len(fake.calls) == 7
    assert all("a1_call_rae" not in c["parts"][-1] for c in fake.calls)
    assert entry["checks"]["guidance"]["verdict"] == "fail" and entry["user_sim_error"]["because"] == ["guidance"]


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
@pytest.mark.parametrize("recorded, status", [(True, "match"), (False, "unknown")])
async def test_items_for_this_persona_or_a_run_with_no_fingerprint_are_graded(monkeypatch, recorded, status):
    ctx = await _chat(monkeypatch)
    us = ctx.metadata[USER_SIMS_KEY]["hannah"]
    fingerprint = us["persona_sha256"] if recorded else OTHER_PERSONA
    if not recorded:
        del us["persona_sha256"]  # a run from before the step recorded it

    await _step(FakeModel(), monkeypatch, checks=["style"], rubric_items=[HER_LENGTH],
                rubric_persona_sha256=fingerprint).execute(ctx)

    entry = _entry(ctx)
    assert entry["rubric_persona"]["status"] == status and entry["items"]["f2_her_length"]["verdict"] == "pass"
    assert entry["rubric_persona"]["rubric"] == fingerprint
    if not recorded:
        assert entry["rubric_persona"]["run"] is None and "graded as written" in entry["rubric_persona"]["note"]
    # Without a rubric_persona_sha256 nothing is compared.
    await _step(FakeModel(), monkeypatch, checks=["style"], rubric_items=[HER_LENGTH]).execute(ctx)
    assert "rubric_persona" not in _entry(ctx) and _entry(ctx)["items"]["f2_her_length"]["verdict"] == "pass"
