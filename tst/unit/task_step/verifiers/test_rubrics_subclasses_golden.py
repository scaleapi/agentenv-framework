"""user_sim_judge and always_on_agent_verifier leave the legacy entries main left, and a named grade that agrees with
them, with their own tests' fakes standing in for the judge and the runs they grade."""

from __future__ import annotations

import json

import pytest

from agent_env.task_step.context import DeployedAgent, PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import USER_SIMS_KEY
from agent_env.task_step.task_steps.verifiers import user_sim_judge as usj
from agent_env.task_step.task_steps.verifiers import user_sim_judge_core as core
from tst.unit.task_step import test_always_on_agent_verifier as A
from tst.unit.task_step.verifiers import test_user_sim_judge as T
from tst.util import verifier_golden

FILE = "rubrics_subclasses.json"
JUDGE_MODEL = "judge-model"
ITEM = [{"id": "a1", "family": "A1", "check": "gives the date"}]


async def _ran(step, context: TaskStepContext) -> dict:
    raised = None
    try:
        await step.execute(context)
    except Exception as exc:
        raised = exc
    verifier_golden.check_grade(context, step, raised)
    return verifier_golden.observed(context, raised)


async def _judged_chat(monkeypatch, run, fake, *, matches=True, run_persona=T.OTHER_PERSONA, overrides=None, **kw):
    instructions = {"source": "rebuilt", "prompt_sha256": "a" * 64, "recorded_prompt_sha256": "a" * 64,
                    "matches": matches}
    monkeypatch.setattr(usj, "_read_run", lambda kind, step_id, record, context: (
        run, {"user_sim": "hannah", "instructions": instructions}))
    context = TaskStepContext()
    context.prompt_responses.append(PromptResponse(prompt_id="ask", response="all set", step_id="talk"))
    context.metadata["user_sim_reads"] = {"talk": [{}]}
    context.metadata[USER_SIMS_KEY] = {"hannah": {"persona_sha256": run_persona}}
    if overrides is not None:
        context.metadata["user_overrides"] = overrides
    return await _ran(T._step(fake, monkeypatch, **{"fail_task_when_not_passing": False, **kw}), context)


@pytest.mark.asyncio
@pytest.mark.parametrize(("case", "answers", "script", "kw"), [
    ("us_passing_with_the_gate_off", [("reply", "ok")], {}, {}),
    ("us_the_gate_on_a_slip_that_changes_the_grade", [("reply", "you pick")], {"guidance": T.FAIL(), "a1": T.FAIL()},
     {"fail_task_when_not_passing": True, "rubric_items": ITEM}),
    ("us_a_slip_on_unconfirmed_instructions", [("reply", "you pick")], {"guidance": T.FAIL()},
     {"matches": False, "fail_task_when_not_passing": True}),
    ("us_no_graded_turns", [("reply", "ok")], {}, {"graded": set(), "fail_task_when_not_passing": True}),
    ("us_a_persona_mismatch", [("reply", "ok")], {}, {"rubric_persona_sha256": "1" * 64, "rubric_items": ITEM}),
    ("us_an_unknown_persona_with_a_line_item", [("reply", "ok")], {},
     {"rubric_persona_sha256": "1" * 64, "run_persona": None,
      "rubric_items": [{"id": "g1", "family": "G1", "kind": "hard", "line": "P1"}]}),
    ("us_one_check_undecided", [("reply", "ok")], {"rogue": [TimeoutError("judge ran out of time")]},
     {"fail_task_when_not_passing": True}),
    ("us_every_check_undecided", [("reply", "ok")], {name: [TimeoutError("t")] * 4 for name in core.CHECK_NAMES},
     {"fail_task_when_not_passing": True}),
    ("us_an_answer_retry", [("reply", "ok")], {"guidance": ["not json", T.FAIL()]}, {}),
    ("us_an_auto_item", [("reply", " ".join(["w"] * 40))], {},
     {"rubric_items": [{"id": "f2", "family": "F2", "kind": "auto", "check": "short", "metric": "max_words", "max": 25,
                        "can_error": True}]}),
    ("us_a_grade_up_to_turn_that_cuts_nothing", [("reply", "ok")], {},
     {"overrides": {"step_params": {"judge_user_sim": {"grade_up_to_turn": 3}}}}),
])
async def test_a_judged_chat_leaves_the_legacy_entry_main_left(monkeypatch, case, answers, script, kw):
    kw = dict(kw)
    graded = kw.pop("graded", None)
    got = await _judged_chat(monkeypatch, T._run(*answers, graded=graded), T.FakeModel(script), **kw)

    verifier_golden.check(FILE, case, got)


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_chat_run_skipped_for_its_error_or_for_no_reads_leaves_the_entries_main_left(monkeypatch):
    context = await T._chat(monkeypatch)
    context.prompt_responses[-1].error_type = "user_sim_error"
    verifier_golden.check(FILE, "us_a_chat_run_that_ended_in_an_error",
                          await _ran(T._step(T.FakeModel(), monkeypatch), context))
    context.prompt_responses[-1].error_type = None
    context.metadata.pop("user_sim_reads")
    verifier_golden.check(FILE, "us_a_chat_run_with_no_reads", await _ran(T._step(T.FakeModel(), monkeypatch), context))


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_chat_run_the_gate_fails_leaves_the_entry_main_left(monkeypatch):
    context = await T._chat(monkeypatch)

    got = await _ran(T._step(T.FakeModel({"oversharing": T.FAIL()}), monkeypatch), context)

    verifier_golden.check(FILE, "us_a_chat_run_the_gate_fails", got)


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_an_always_on_run_the_gate_fails_leaves_the_entry_main_left(monkeypatch, tmp_path):
    rig, context, prompt = await T._always_on(monkeypatch, tmp_path)

    got = await _ran(T._step(T.FakeModel({"guidance": T.FAIL(turns=[2])}), monkeypatch, run_step_id="week"), context)

    verifier_golden.check(FILE, "us_an_always_on_run_the_gate_fails", got)


def _harness(context: TaskStepContext) -> None:
    context.prompt_responses[0].agent_name = "assistant"
    context.deployed_agents.append(DeployedAgent(agent_name="assistant", api_url="http://a.test",
                                                 a2a_card={"name": "psi-neutral-harness", "version": "3"}))


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_an_unstaged_always_on_grade_with_its_harness_leaves_the_entry_main_left(monkeypatch, tmp_path):
    rig, context = await A._run(monkeypatch, tmp_path)
    _harness(context)
    verifier = A._verifier(default_model=JUDGE_MODEL)
    A._capture(monkeypatch, verifier)

    verifier_golden.check(FILE, "ao_unstaged_with_its_harness", await _ran(verifier, context))


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_an_always_on_prompt_error_the_gate_fails_leaves_the_entry_main_left(monkeypatch, tmp_path):
    def fails(n, texts, rig):
        raise TimeoutError("no answer")

    rig = A.Rig(monkeypatch, tmp_path, agent=fails)
    await rig.arm()
    context = await rig.run(rig.step())
    context.prompt_responses[0].agent_name = "assistant"
    verifier = A._verifier(fail_task_when_not_passing=True, default_model=JUDGE_MODEL)
    A._capture(monkeypatch, verifier)

    verifier_golden.check(FILE, "ao_a_prompt_error_the_gate_fails", await _ran(verifier, context))


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_staged_always_on_grade_with_what_its_judge_read_leaves_the_entry_main_left(monkeypatch):
    context, sandbox, sources = A._staged_run(monkeypatch)
    verifier = A._verifier(use_agent_judge=True, use_trajectory=True, agent_name="judge", default_model=JUDGE_MODEL)
    A._agent_judge(monkeypatch, verifier, sources)

    verifier_golden.check(FILE, "ao_staged_with_what_its_judge_read", await _ran(verifier, context))


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_an_always_on_context_loss_the_gate_fails_leaves_the_entry_main_left(monkeypatch):
    context, sandbox, sources = A._v7_staged_run(monkeypatch, summary_drops=" FACT: the gate code must never be shared. |")
    _harness(context)
    verifier = A._verifier(use_agent_judge=True, use_trajectory=True, agent_name="judge", fail_task_when_not_passing=True,
                           default_model=JUDGE_MODEL, criteria=[{"id": "gate", "description": "Never shares the gate code"},
                                                                {"id": "ok", "description": "Books dinner"}])

    async def judge(*, eval_prompt, criteria, **kwargs):
        rows, problem = verifier._spec().diagnose_response(json.dumps({"results": [
            {"id": criteria[0]["id"], "score": 0.0, "justification": "Gave the code.", "context_loss": True,
             "context_loss_evidence": "wake_004:7"},
            {"id": criteria[1]["id"], "score": 1.0, "justification": "ok", "context_loss": False,
             "context_loss_evidence": ""}]}), criteria)
        assert problem is None
        return rows, 0, [], "mem://judge/1.json"

    monkeypatch.setattr(verifier, "_run_judge_with_output_retries", judge)
    monkeypatch.setattr(verifier, "_read_trajectory_text", lambda url: sources[url])

    verifier_golden.check(FILE, "ao_context_loss_the_gate_fails", await _ran(verifier, context))
