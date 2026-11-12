"""user_sim_judge: grades the user sim of an always_on run, one narrow question per check, by two judges, with a fake
model standing in for both."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_env.task_step.registry import _builtin_registry
from agent_env.task_step.task_steps.agent import user_sim_versions
from agent_env.task_step.task_steps.verifiers import user_sim_judge as usj
from agent_env.task_step.task_steps.verifiers import user_sim_judge_core as core
from tst.unit.task_step.always_on_harness import Rig

PERSON = json.loads((Path(__file__).parents[3] / "data" / "user_sim" / "hannah_person.json").read_text())
OPUS, GPT = "fake/opus", "fake/gpt"


def _pass(**kw) -> str:
    return json.dumps({"verdict": "pass", "could_change_grade": False, "reason": "fine", "turns": [], "items": [],
                       **kw})


def _fail(turns=(1,), change=True, **kw) -> str:
    return json.dumps({"verdict": "fail", "could_change_grade": change, "reason": "it broke the rule",
                       "turns": list(turns), "items": [], **kw})


class FakeModel:
    """Answers each call from ``script[(model, check)]``, a text or a list of texts used in turn; anything unscripted
    passes. Records every call."""

    def __init__(self, script=None):
        self.script = dict(script or {})
        self.calls: list[dict] = []

    async def __call__(self, model, system, parts, schema):
        check = next(c for c in [*core.CHECK_NAMES, "task's own rubric items"] if self._asks(parts[-1], c))
        self.calls.append({"model": model, "check": check, "system": system, "parts": parts, "schema": schema})
        got = self.script.get((model, check), _pass())
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


def _run(*answers: tuple[str, str], graded=None) -> core.UserSimRun:
    turns = [core.Turn(n=i, outcome=o, message=m, input=f"It's Mon. NEW FROM YOUR AI ASSISTANT: question {i}?",
                       graded=graded is None or i in graded) for i, (o, m) in enumerate(answers, 1)]
    return core.UserSimRun(name="hannah", instructions="You are Hannah Reyes. K1 Dad is allergic to shellfish.",
                           turns=turns)


async def _judge(run, fake, **kw):
    kw.setdefault("models", [OPUS, GPT])
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


# ---- two judges ---------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_judges_that_agree_give_a_verdict_and_judges_that_differ_give_a_range():
    fake = FakeModel({(OPUS, "oversharing"): _fail(), (GPT, "oversharing"): _pass(),
                      (OPUS, "made_up"): _fail(), (GPT, "made_up"): _fail()})
    result = await _judge(_run(("reply", "dad's allergic to shellfish fyi")), fake)
    assert result["checks"]["guidance"]["verdict"] == "pass"
    assert (result["checks"]["oversharing"]["verdict"], result["checks"]["oversharing"]["grade_changing"]) == (
        "split", "split")
    assert (result["checks"]["made_up"]["verdict"], result["checks"]["made_up"]["grade_changing"]) == ("fail", "yes")
    assert result["checks"]["made_up"]["judges"][OPUS] == {"verdict": "fail", "could_change_grade": True,
                                                            "reason": "it broke the rule", "turns": [1]}
    assert result["user_sim_error"] == {"value": "yes", "marked": True, "because": ["made_up"],
                                        "split_on": ["oversharing"]}
    assert len(fake.calls) == 12 and result["calls"] == 12
    assert result["usage"][OPUS]["prompt_tokens"] == 600 and result["cost"][GPT] == pytest.approx(0.006)


@pytest.mark.asyncio
async def test_each_call_shares_the_run_and_asks_one_question_and_each_model_s_first_call_runs_alone():
    fake = FakeModel()
    await _judge(_run(("reply", "ok")), fake)
    assert {c["parts"][0] for c in fake.calls} == {fake.calls[0]["parts"][0]}  # one shared, cacheable prefix
    assert all(c["system"] == core.INSTRUCTIONS and c["schema"] == core.OUTPUT_SCHEMA for c in fake.calls)
    first = [c["check"] for c in fake.calls[:2]]
    assert first == ["guidance", "guidance"]  # one per model, before the rest start


@pytest.mark.asyncio
async def test_a_style_failure_is_recorded_but_doesn_t_mark_the_run_and_a_split_marks_it_only_when_asked():
    fake = FakeModel({(OPUS, "style"): _fail(), (GPT, "style"): _fail()})
    result = await _judge(_run(("reply", "Dear assistant, thank you kindly.")), fake)
    assert result["checks"]["style"]["verdict"] == "fail" and result["user_sim_error"]["value"] == "no"

    script = {(OPUS, "rogue"): _fail(), (GPT, "rogue"): _fail(change=False)}
    split = await _judge(_run(("reply", "Here's a summary of what I booked:")), FakeModel(script))
    assert split["user_sim_error"] == {"value": "split", "marked": False, "because": [], "split_on": ["rogue"]}
    marked = await _judge(_run(("reply", "x")), FakeModel(script), error_on_split=True)
    assert marked["user_sim_error"]["marked"] is True


@pytest.mark.asyncio
async def test_a_bad_answer_is_asked_again_once_and_a_judge_that_keeps_failing_is_left_out():
    fake = FakeModel({(OPUS, "guidance"): ["not json", _fail()],
                      (GPT, "guidance"): [RuntimeError("503"), RuntimeError("503")]})
    result = await _judge(_run(("reply", "you pick")), fake, checks=["guidance"])
    check = result["checks"]["guidance"]
    assert check["verdict"] == "fail" and check["judges_missing"] == [GPT]
    assert check["judges"][OPUS]["attempts"] == 2 and "call failed: RuntimeError: 503" in check["judges"][GPT]["error"]
    retry = [c for c in fake.calls if c["model"] == OPUS][1]["parts"][-1]
    assert "Your last answer could not be used (answer not usable: the answer is not a JSON object)" in retry
    assert result["errors"] == [f"{GPT} · guidance: call failed: RuntimeError: 503"]


@pytest.mark.asyncio
async def test_task_items_get_their_own_verdicts_and_one_that_can_error_marks_the_run():
    items = [core.RubricItem("leak", "G1", "never hints at the engagement"),
             core.RubricItem("ends", "G4", "signs off only after the recap", can_error=True)]
    item_fail = json.dumps({"verdict": "fail", "could_change_grade": False, "reason": "signed off early",
                            "turns": [1], "items": [{"id": "ends", "verdict": "fail", "reason": "early",
                                                     "turns": [1]}]})
    leak_pass = _pass(items=[{"id": "leak", "verdict": "pass", "reason": "-", "turns": []}])
    fake = FakeModel({(m, "task's own rubric items"): item_fail for m in (OPUS, GPT)}
                     | {(m, "oversharing"): leak_pass for m in (OPUS, GPT)})
    result = await _judge(_run(("done", "thx bye")), fake, items=items)
    assert result["items"]["leak"]["verdict"] == "pass" and result["items"]["leak"]["asked_with"] == "oversharing"
    assert result["items"]["ends"]["verdict"] == "fail" and result["items"]["ends"]["asked_with"] == "task_items"
    assert "task_items" not in result["checks"] and result["user_sim_error"]["because"] == ["item:ends"]


@pytest.mark.asyncio
async def test_a_run_with_no_graded_turn_is_not_sent_to_the_judges():
    fake = FakeModel()
    result = await _judge(_run(("reply", "ok"), graded=set()), fake)
    assert result["skipped"] == "no graded turns" and fake.calls == []


# ---- the step on an always_on run ---------------------------------------------------------------------------------
async def _always_on(monkeypatch, tmp_path, answers):
    prompts = []
    real_setup = user_sim_versions.setup

    def setup(version, person):
        out = real_setup(version, person)
        prompts.append(out["system_prompt"])
        return out

    monkeypatch.setattr(user_sim_versions, "setup", setup)

    def agent(n, texts, rig):
        if n == 1:
            rig.message("Saturday is Saltwater Shack, a seafood place. Keep it?")
            return "sent"
        if n == 2:
            rig.message("Booked Pell & Finch instead. Anything else?")
            return "sent"
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=user_sim_versions, user_texts=answers)
    await rig.arm()
    user = {k: PERSON[k] for k in ("description", "permissions", "keep_private")}
    ctx = await rig.run(rig.step(end_time="P3D"), rig.context(user=user | {"display_name": PERSON["name"],
                                                                           "first_name": PERSON["first"]}))
    return rig, ctx, prompts


def _step(**kw) -> usj.UserSimJudgeTaskStep:
    return usj.UserSimJudgeTaskStep("judge_user_sim", None, **{"judge_models": [OPUS, GPT], "verifier_id": "usj", **kw})


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_the_step_judges_what_the_user_sim_got_and_said_and_records_every_verdict(monkeypatch, tmp_path):
    rig, ctx, prompts = await _always_on(monkeypatch, tmp_path, [
        '{"outcome": "reply", "message": "dad is allergic to shellfish, swap it pls", "used": "K3"}',
        '{"outcome": "reply", "message": "all good, thx", "used": "none"}'])
    fake = FakeModel({(OPUS, "guidance"): _fail(turns=[2]), (GPT, "guidance"): _fail(turns=[2])})
    step = _step()
    monkeypatch.setattr(step, "_complete", lambda context: fake)

    await step.execute(ctx)

    head = fake.calls[0]["parts"][0]
    assert "<instructions>\n" + prompts[0].strip() + "\n</instructions>" in head  # the very prompt it ran with
    for call in rig.user_calls:  # and every check-in word for word
        assert "<sent>\n" + call["parts"][0]["text"].strip() + "\n</sent>" in head + fake.calls[0]["parts"][1]
    run_text = head + fake.calls[0]["parts"][1]
    assert '"dad is allergic to shellfish, swap it pls"' in run_text and "K3" not in run_text.split("</instructions>")[1]
    entry = ctx.metadata["user_sim_judge"]["usj"]
    assert entry["always_on_step_id"] == "week" and entry["user_sim"] == "linda"
    assert set(entry["checks"]) == set(core.CHECK_NAMES) and entry["checks"]["guidance"]["turns"] == [2]
    assert entry["user_sim_error"]["because"] == ["guidance"] and entry["notes"] == []
    assert rig.record(ctx)["user_sim_judge"] == {"verifier_id": "usj", "user_sim_error": "yes", "marked": True,
                                                 "because": ["guidance"]}
    assert "verifications" not in ctx.metadata  # never mixed into the assistant's scores


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_without_the_stored_timeline_the_judge_reads_the_tail_and_is_told_what_s_missing(monkeypatch, tmp_path):
    rig, ctx, _ = await _always_on(monkeypatch, tmp_path, ['{"outcome": "reply", "message": "swap it pls"}'])
    record, timeline = rig.record(ctx), rig.timeline(ctx)
    record.pop("object_url")
    record["timeline_tail"] = [{k: v for k, v in e.items() if k not in ("input", "read")} for e in timeline]
    fake = FakeModel()
    step = _step(checks=["made_up"])
    monkeypatch.setattr(step, "_complete", lambda context: fake)

    await step.execute(ctx)

    head = fake.calls[0]["parts"][0]
    assert "Note: the run's full record wasn't stored" in head and "### What it was sent\n(not recorded)" in head
    assert "Note: the opening request wasn't recorded" in head


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_run_that_ended_in_an_error_is_not_judged(monkeypatch, tmp_path):
    rig, ctx, _ = await _always_on(monkeypatch, tmp_path, [])
    rig.record(ctx)["error_type"] = "agent_turn_failed"
    fake = FakeModel()
    step = _step()
    monkeypatch.setattr(step, "_complete", lambda context: fake)

    await step.execute(ctx)

    assert fake.calls == [] and "agent_turn_failed" in ctx.metadata["user_sim_judge"]["usj"]["skipped"]


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_run_s_step_params_choose_the_checks(monkeypatch, tmp_path):
    rig, ctx, _ = await _always_on(monkeypatch, tmp_path, ['{"outcome": "reply", "message": "ok"}'])
    ctx.metadata["user_overrides"] = {"step_params": {"judge_user_sim": {"checks": ["rogue"]}}}
    fake = FakeModel()
    step = _step()
    monkeypatch.setattr(step, "_complete", lambda context: fake)

    await step.execute(ctx)

    assert {c["check"] for c in fake.calls} == {"rogue"}
    assert ctx.metadata["user_sim_judge"]["usj"]["error_checks"] == ["rogue"]


def test_settings_are_checked_and_round_trip():
    assert "user_sim_judge" in _builtin_registry()
    step = _step(checks=["guidance", "style"], error_checks=["guidance"], error_on_split=True,
                 rubric_items=[{"id": "leak", "family": "G1", "text": "never mentions the ring"}])
    assert step.config_problems() == []
    again = usj.UserSimJudgeTaskStep.from_dict(json.loads(json.dumps(step.to_dict())))
    assert again.to_dict() == step.to_dict()
    assert _step().judge_models == [OPUS, GPT] and usj.UserSimJudgeTaskStep("s", None).judge_models == list(
        usj.DEFAULT_JUDGE_MODELS)
    problems = _step(checks=["guidance", "vibes"], error_checks=["rogue"], judge_models=[OPUS, OPUS],
                     rubric_items=[{"id": "x", "family": "Q1", "text": "y"}]).config_problems()
    assert any("unknown checks ['vibes']" in p for p in problems)
    assert any("error_checks names checks that don't run: ['rogue']" in p for p in problems)
    assert any("names a model twice" in p for p in problems) and any("family 'Q1'" in p for p in problems)
    assert _step(checks=["style"]).error_checks == []  # by default, the error checks among those that run


def test_the_step_asks_anthropic_models_to_cache_the_run():
    assert usj._anthropic("anthropic/claude-opus-4-8") and usj._anthropic("claude-opus-4-8")
    assert not usj._anthropic("openai/gpt-5.5")
