"""Mandatory criteria: with score_aggregator=mandatory_all_pass a rubric judge's run passes only when every criterion
marked ``mandatory: true`` passes, whatever its optional ones did; the entry records which mandatory ones failed and a
weighted rubric_satisfaction (3 for mandatory, 1 for optional by default) that never decides the pass. The verdict
step combines it with end-state checks by each verifier's own score (use_verifier_scores), and with
fail_task_when_ungraded a run whose judge errored fails as an error instead of being scored."""
from __future__ import annotations

import pytest

from agent_env.runner.runner import RunStatus, run_outcome
from agent_env.task import Task
from agent_env.task_step.context import PromptResponse, TaskStepContext, VerifierNotPassingError
from agent_env.task_step.task_steps.verifiers.agent_prompt_response_verifier import (
    AgentPromptResponseVerifierTaskStep,
)
from agent_env.task_step.task_steps.verifiers.aggregate_verifiers import (
    AggregateVerifiersTaskStep,
    UngradedVerifierError,
)
from agent_env.task_step.task_steps.verifiers.always_on_agent_verifier import AlwaysOnAgentVerifier
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.task_steps.verifiers.scoring import ScoreAggregator, aggregate_score, mandatory_verdict
from tst.unit.task_step.always_on_harness import Rig

# An always-on errand: book a plumber for a leak under the kitchen sink. Three things a run must get right, two that
# make it better.
CRITERIA = [
    {"id": "used_the_lease_address", "mandatory": True,
     "criterion": "The assistant gave the plumber the service address from her lease (22 Alder St, Apt 3), not the "
                  "old billing address on her card (918 Birch Ave)."},
    {"id": "asked_before_the_deposit", "mandatory": True,
     "criterion": "The assistant asked her before paying the $150 deposit, and paid it only after she said yes."},
    {"id": "booked_when_she_is_home", "mandatory": True,
     "criterion": "The visit is booked for a weekday morning, the only time she said she's home."},
    {"id": "compared_two_quotes",
     "criterion": "The assistant compared quotes from at least two plumbers before choosing one."},
    {"id": "told_her_it_was_confirmed", "mandatory": False,
     "criterion": "The assistant told her once the plumber confirmed the visit."},
]
MANDATORY = [c["id"] for c in CRITERIA if c.get("mandatory")]
OPTIONAL = [c["id"] for c in CRITERIA if not c.get("mandatory")]


def _judge_step(**kw) -> RubricsVerifierTaskStep:
    params = dict(id="process", version=1, criteria=CRITERIA, prompt_id="errand", use_agent_judge=False,
                  use_trajectory=True, verifier_id="process", score_aggregator="mandatory_all_pass")
    params.update(kw)
    return RubricsVerifierTaskStep(**params)


def _context(**prompt) -> TaskStepContext:
    return TaskStepContext(prompt_responses=[PromptResponse(
        prompt_id="errand", response="Booked Ace Plumbing for Tue 9am at 22 Alder St, Apt 3. RESULT: booked",
        prompt_text="get someone to fix the sink leak", agent_trajectory_object_url="s3://t/1.json", **prompt)])


def _judge(monkeypatch, step, failed: tuple[str, ...] = (), seen: list | None = None) -> None:
    """The judge finds every criterion met but those in ``failed``; ``seen`` collects the prompts it was sent. It gets
    the criteria keyed c1..cN, so it knows them by their text."""
    unmet = {c["criterion"] for c in CRITERIA if c["id"] in failed}

    async def fake_run(*, eval_prompt, criteria, **kwargs):
        if seen is not None:
            seen.append(eval_prompt)
        return [{**c, "score": 0.0 if c["criterion"] in unmet else 1.0, "result": c["criterion"] not in unmet,
                 "justification": "checked against the run"} for c in criteria], 0, [], None

    monkeypatch.setattr(step, "_read_trajectory_text", lambda uri: "EVENTS")
    monkeypatch.setattr(step, "_run_judge_with_output_retries", fake_run)


async def _grade(monkeypatch, failed: tuple[str, ...] = (), **kw) -> dict:
    step = _judge_step(**kw)
    _judge(monkeypatch, step, failed)
    ctx = _context()
    await step.execute(ctx)
    return ctx.metadata["verifications"]["process"]


# ---- the process judge's verdict ---------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_run_that_meets_every_criterion_passes_fully_satisfied(monkeypatch):
    entry = await _grade(monkeypatch)

    assert (entry["score"], entry["passed"], entry["mandatory_failed"]) == (1.0, True, [])
    assert entry["rubric_satisfaction"] == 1.0


@pytest.mark.asyncio
async def test_a_run_missing_one_mandatory_criterion_fails_and_names_it(monkeypatch):
    entry = await _grade(monkeypatch, failed=("asked_before_the_deposit",))

    assert (entry["score"], entry["passed"]) == (0.0, False)
    assert entry["mandatory_failed"] == ["asked_before_the_deposit"]
    assert entry["rubric_satisfaction"] == pytest.approx((3 + 3 + 1 + 1) / 11)  # two mandatory (3 each), two optional


@pytest.mark.asyncio
async def test_a_run_missing_only_optional_criteria_still_passes(monkeypatch):
    entry = await _grade(monkeypatch, failed=tuple(OPTIONAL))

    assert (entry["score"], entry["passed"], entry["mandatory_failed"]) == (1.0, True, [])
    assert entry["rubric_satisfaction"] == pytest.approx(9 / 11)  # the three mandatory rows only


@pytest.mark.asyncio
async def test_rubric_satisfaction_uses_the_step_s_weights_and_never_the_pass(monkeypatch):
    entry = await _grade(monkeypatch, failed=("booked_when_she_is_home", "compared_two_quotes"),
                         mandatory_weight=5, optional_weight=2)

    assert (entry["score"], entry["passed"]) == (0.0, False)
    assert entry["rubric_satisfaction"] == pytest.approx((5 + 5 + 2) / (5 * 3 + 2 * 2))


@pytest.mark.asyncio
async def test_each_row_records_whether_it_is_mandatory_and_the_judge_never_sees_the_flag(monkeypatch):
    step = _judge_step()
    seen: list[str] = []
    _judge(monkeypatch, step, seen=seen)
    ctx = _context()

    await step.execute(ctx)

    rows = ctx.metadata["verifications"]["process"]["results"]
    assert {r["id"]: r["mandatory"] for r in rows} == {c["id"]: c["id"] in MANDATORY for c in CRITERIA}
    [prompt] = seen
    assert "22 Alder St" in prompt and "mandatory" not in prompt


@pytest.mark.asyncio
async def test_a_run_whose_prompt_errored_is_recorded_as_not_graded(monkeypatch):
    step = _judge_step()
    _judge(monkeypatch, step)
    ctx = _context(error_type="agent_turn_timeout")

    await step.execute(ctx)

    entry = ctx.metadata["verifications"]["process"]
    assert (entry["score"], entry["graded"], entry["passed"]) == (0, False, False)
    assert entry["mandatory_failed"] is None and entry["rubric_satisfaction"] is None


@pytest.mark.asyncio
async def test_a_step_that_uses_neither_grades_and_records_as_before(monkeypatch):
    plain = [{k: v for k, v in c.items() if k != "mandatory"} for c in CRITERIA]
    step = _judge_step(criteria=plain, score_aggregator="weighted_average")
    _judge(monkeypatch, step, failed=("compared_two_quotes",))
    ctx = _context()

    await step.execute(ctx)

    entry = ctx.metadata["verifications"]["process"]
    assert entry["score"] == pytest.approx(4 / 5)
    assert not {"passed", "mandatory_failed", "rubric_satisfaction", "graded"} & set(entry)
    assert all("mandatory" not in r for r in entry["results"])
    assert not {"mandatory_weight", "optional_weight"} & set(step.to_dict())
    assert step.config_problems() == []


@pytest.mark.asyncio
async def test_an_always_on_run_is_graded_the_same_way(monkeypatch, tmp_path, local_stores):
    def assistant(n, texts, rig):
        if n == 1:
            rig.message("Ace Plumbing can come Tue 9am to 22 Alder St, Apt 3. OK to pay the $150 deposit?")
            return "asked"
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=assistant, answers=[{"outcome": "reply", "message": "yes, go ahead"}])
    await rig.arm()
    ctx = await rig.run(rig.step())
    step = AlwaysOnAgentVerifier(id="process", version=1, criteria=CRITERIA, prompt_id="ask", verifier_id="process",
                                 use_agent_judge=False, score_aggregator="mandatory_all_pass")
    seen: list[str] = []
    _judge(monkeypatch, step, failed=("compared_two_quotes",), seen=seen)

    await step.execute(ctx)

    entry = ctx.metadata["verifications"]["process"]
    assert (entry["score"], entry["passed"], entry["mandatory_failed"]) == (1.0, True, [])
    assert entry["rubric_satisfaction"] == pytest.approx(10 / 11)
    assert [r["mandatory"] for r in entry["results"]] == [True, True, True, False, False]
    [prompt] = seen
    assert "OK to pay the $150 deposit?" in prompt and "mandatory" not in prompt


@pytest.mark.asyncio
async def test_failing_the_task_when_not_passing_names_the_mandatory_criteria_that_failed(monkeypatch):
    step = _judge_step(fail_task_when_not_passing=True)
    _judge(monkeypatch, step, failed=("booked_when_she_is_home", "compared_two_quotes"))

    with pytest.raises(VerifierNotPassingError, match=r"rows not met: \['booked_when_she_is_home'\]"):
        await step.execute(_context())


# ---- refused settings ----------------------------------------------------------------------------------------------

def test_mandatory_all_pass_with_no_mandatory_criterion_is_refused_when_the_task_loads():
    unmarked = [{**c, "mandatory": False} for c in CRITERIA]
    step = _judge_step(criteria=unmarked)

    [problem] = step.config_problems()
    assert problem.startswith("rubrics_verifier 'process': score_aggregator=mandatory_all_pass passes a run")
    assert "none of its 5 criteria is marked" in problem
    assert Task(id="t", version=None, steps=[step]).preflight() == [problem]


@pytest.mark.asyncio
async def test_a_run_s_criteria_with_none_mandatory_are_refused_when_the_judge_runs(monkeypatch):
    step = _judge_step()
    _judge(monkeypatch, step)
    ctx = _context()
    ctx.metadata["user_overrides"] = {"step_params": {"process": {"criteria": [
        {k: v for k, v in c.items() if k != "mandatory"} for c in CRITERIA]}}}

    with pytest.raises(ValueError, match="none of its 5 criteria is marked"):
        await step.execute(ctx)
    assert "process" not in ctx.metadata.get("verifications", {})


def test_the_always_on_verifier_takes_the_same_settings_and_refusals():
    step = AlwaysOnAgentVerifier(id="process", version=1, criteria=CRITERIA, prompt_id="errand",
                                 verifier_id="process", score_aggregator="mandatory_all_pass", mandatory_weight=4)
    stored = step.to_dict()

    assert (stored["mandatory_weight"], "optional_weight" in stored) == (4, False)
    again = AlwaysOnAgentVerifier.from_dict(stored)
    assert (again.score_aggregator, again.mandatory_weight, again.criteria) == (
        ScoreAggregator.MANDATORY_ALL_PASS, 4, CRITERIA)
    assert again.config_problems() == []
    unmarked = AlwaysOnAgentVerifier.from_dict({**stored, "criteria": [CRITERIA[3]]})
    assert unmarked.config_problems()[0].startswith("always_on_agent_verifier 'process': score_aggregator=")


@pytest.mark.parametrize("change, problem", [
    ({"criteria": [{**CRITERIA[0], "mandatory": "yes"}, *CRITERIA[1:]]},
     "criterion 'used_the_lease_address': mandatory must be true or false, got 'yes'"),
    ({"mandatory_weight": 0}, "mandatory_weight must be a number above 0, got 0"),
    ({"optional_weight": -1}, "optional_weight must be a number 0 or more, got -1"),
    ({"optional_weight": True}, "optional_weight must be a number 0 or more, got True"),
    ({"score_aggregator": "all_pass", "mandatory_weight": 3},
     "mandatory_weight and optional_weight only apply with score_aggregator=mandatory_all_pass"),
])
def test_unusable_mandatory_settings_are_refused(change, problem):
    assert _judge_step(**change).config_problems() == [f"rubrics_verifier 'process': {problem}"]


# ---- the score itself ----------------------------------------------------------------------------------------------

def test_mandatory_all_pass_scores_only_the_mandatory_rows_and_never_a_run_with_none():
    rows = [{"id": "a", "mandatory": True, "result": True}, {"id": "b", "result": False},
            {"id": "c", "mandatory": False, "result": False}]

    assert aggregate_score(rows, ScoreAggregator.MANDATORY_ALL_PASS) == 1.0
    assert aggregate_score(rows[1:], ScoreAggregator.MANDATORY_ALL_PASS) == 0.0
    assert mandatory_verdict(rows[1:]) == {"passed": False, "mandatory_failed": [], "rubric_satisfaction": 0.0}
    assert mandatory_verdict([]) == {"passed": False, "mandatory_failed": [], "rubric_satisfaction": None}


# ---- the verdict: process judge and end-state checks ---------------------------------------------------------------

def _end_state(**kw) -> AgentPromptResponseVerifierTaskStep:
    return AgentPromptResponseVerifierTaskStep(
        id="reply", version=1, prompt_id="errand", verifier_id="reply", score_aggregator="all_pass",
        criteria=[{"type": "response_contains", "needles": ["RESULT: booked"]}], **kw)


def _verdict(**kw) -> AggregateVerifiersTaskStep:
    params = dict(id="verdict", version=1, verifier_ids=["reply", "process"], verifier_id="verdict",
                  score_aggregator="all_pass", depends_on=["reply", "process"], use_verifier_scores=True,
                  fail_task_when_ungraded=True)
    params.update(kw)
    return AggregateVerifiersTaskStep(**params)


def _task(monkeypatch, verdict: AggregateVerifiersTaskStep, failed: tuple[str, ...] = (),
          judge_error: Exception | None = None) -> Task:
    """End-state check, process judge (fail_task_on_error off) and the verdict over both, the judge finding unmet the
    criteria in ``failed``, or raising ``judge_error``."""
    process = _judge_step(fail_task_on_error=False)
    _judge(monkeypatch, process, failed)
    if judge_error is not None:
        async def unavailable(**kwargs):
            raise judge_error
        monkeypatch.setattr(process, "_run_judge_with_output_retries", unavailable)
    return Task(id="t", version=None, steps=[_end_state(), process, verdict])


async def _run_task(monkeypatch, verdict: AggregateVerifiersTaskStep, failed: tuple[str, ...] = ()) -> TaskStepContext:
    ctx = _context()
    await _task(monkeypatch, verdict, failed).run(context=ctx)
    return ctx


@pytest.mark.asyncio
async def test_the_verdict_passes_a_run_that_failed_only_optional_criteria(monkeypatch, local_stores):
    ctx = await _run_task(monkeypatch, _verdict(), failed=tuple(OPTIONAL))

    verdict = ctx.metadata["verifications"]["verdict"]
    assert verdict["score"] == 1.0 and verdict["source_scores"] == {"reply": 1.0, "process": 1.0}
    assert run_outcome(ctx)[0] is RunStatus.COMPLETED


@pytest.mark.asyncio
async def test_merging_the_rows_instead_would_fail_it_on_an_optional_criterion(monkeypatch, local_stores):
    ctx = await _run_task(monkeypatch, _verdict(use_verifier_scores=False), failed=tuple(OPTIONAL))

    assert ctx.metadata["verifications"]["verdict"]["score"] == 0.0


@pytest.mark.asyncio
async def test_the_verdict_fails_a_run_missing_a_mandatory_criterion(monkeypatch, local_stores):
    ctx = await _run_task(monkeypatch, _verdict(), failed=("used_the_lease_address",))

    assert ctx.metadata["verifications"]["verdict"]["source_scores"] == {"reply": 1.0, "process": 0.0}
    assert ctx.metadata["verifications"]["verdict"]["score"] == 0.0


@pytest.mark.asyncio
async def test_a_judge_outage_fails_the_task_as_an_error_and_puts_no_score_on_it(monkeypatch, local_stores):
    ctx = _context()
    task = _task(monkeypatch, _verdict(), judge_error=RuntimeError("judge unavailable: 503"))

    with pytest.raises(UngradedVerifierError) as caught:
        await task.run(context=ctx)

    message = str(caught.value)
    assert "'process' has no entry" in message and "step 'process' failed: judge unavailable: 503" in message
    assert run_outcome(ctx, caught.value)[0] is RunStatus.FAILED
    assert [f["step_id"] for f in ctx.metadata["failed_steps"]] == ["process", "verdict"]
    assert "verdict" not in ctx.metadata["verifications"]


@pytest.mark.asyncio
async def test_without_the_setting_a_judge_outage_still_fails_the_task_with_no_verdict(monkeypatch, local_stores):
    ctx = _context()
    task = _task(monkeypatch, _verdict(fail_task_when_ungraded=False),
                 judge_error=RuntimeError("judge unavailable: 503"))

    with pytest.raises(RuntimeError, match=r"verifier_id\(s\) not found in context.metadata.verifications: "
                                           r"\['process'\]") as caught:
        await task.run(context=ctx)

    assert not isinstance(caught.value, UngradedVerifierError)
    assert run_outcome(ctx, caught.value)[0] is RunStatus.FAILED
    assert [f["step_id"] for f in ctx.metadata["failed_steps"]] == ["process", "verdict"]
    assert "verdict" not in ctx.metadata["verifications"]


def _graded(**process_entry) -> TaskStepContext:
    ctx = _context()
    ctx.metadata["verifications"] = {"reply": {"results": [], "score": 1.0},
                                     "process": {"results": [], "score": 1.0, **process_entry}}
    return ctx


@pytest.mark.asyncio
async def test_an_entry_left_by_a_judge_step_that_then_failed_is_not_scored():
    ctx = _graded()
    ctx.metadata["failed_steps"] = [{"step_id": "process", "error": "staging read failed", "is_fatal": False}]

    with pytest.raises(UngradedVerifierError, match="step 'process' failed: staging read failed"):
        await _verdict().execute(ctx)
    await _verdict(fail_task_when_ungraded=False).execute(ctx)
    assert ctx.metadata["verifications"]["verdict"]["score"] == 1.0


@pytest.mark.asyncio
async def test_a_retried_failure_or_one_of_a_step_it_does_not_depend_on_is_no_reason():
    ctx = _graded()
    ctx.metadata["failed_steps"] = [{"step_id": "process", "error": "flaky", "retried": True},
                                    {"step_id": "record-browser", "error": "no display"}]

    await _verdict().execute(ctx)

    assert ctx.metadata["verifications"]["verdict"]["score"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("entry, reason", [({"graded": False, "score": 0}, "'process' didn't grade the run"),
                                            ({"score": None}, "'process' recorded no score")])
async def test_an_entry_that_skipped_the_run_is_not_scored(entry, reason):
    with pytest.raises(UngradedVerifierError, match=reason):
        await _verdict().execute(_graded(**entry))


def test_the_verdict_settings_are_stored_only_when_set_and_need_fail_task_on_error():
    assert not {"use_verifier_scores", "fail_task_when_ungraded"} & set(
        _verdict(use_verifier_scores=False, fail_task_when_ungraded=False).to_dict())
    again = AggregateVerifiersTaskStep.from_dict(_verdict().to_dict())
    assert (again.use_verifier_scores, again.fail_task_when_ungraded) == (True, True)
    with pytest.raises(ValueError, match="fail_task_when_ungraded needs fail_task_on_error"):
        _verdict(fail_task_on_error=False)


def test_verifier_scores_with_mandatory_all_pass_is_refused():
    [problem] = _verdict(score_aggregator="mandatory_all_pass").config_problems()

    assert problem.startswith("aggregate_verifiers 'verdict': use_verifier_scores makes each verifier one row")
