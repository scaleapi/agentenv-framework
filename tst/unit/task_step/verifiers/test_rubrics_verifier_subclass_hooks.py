"""The override points a subclass judge builds on: ``_criteria_batches`` (which criteria share a judge call),
``unjudged_rows`` (rows no judge grades, scored with the judge's), ``judge_trajectory_content`` (the run, where the
template puts the trajectory) and the format spec's ``batchable`` / ``checks_bulk_verdicts``. Unoverridden, each
leaves the step as it was."""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext, VerifierNotPassingError
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import (
    JUDGE_OUTPUT_FORMAT_REGISTRY,
    JudgeOutputFormat,
    JudgeOutputFormatSpec,
    ResultRows,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from tst.unit.task_step.verifiers.test_rubrics_verifier_judge_batches import (
    _context,
    _fake_judge,
    _graded,
    _prompted_ids,
)
from tst.unit.task_step.verifiers.test_rubrics_verifier_per_criterion import (
    _NO_SCREENSHOTS,
    CRITERIA,
    _capture_judge,
    _ctx,
    _filter,
    _passing_rows,
)

CHECKS = [{"id": "asked", "title": "Asked for the booking", "check": "asked"},
          {"id": "asked-item", "title": "Named the restaurant", "check": "asked"},
          {"id": "replied", "title": "Replied to the agent", "check": "replied"},
          {"id": "replied-item", "title": "Confirmed the time", "check": "replied"}]


class _ByCheck(RubricsVerifierTaskStep):
    """Each check's criteria in one call, as a user sim judge puts a check's task items in that check's call."""

    def _criteria_batches(self, criteria: list[dict], size: Optional[int]) -> list[list[dict]]:
        groups: dict[str, list[dict]] = {}
        for criterion in criteria:
            groups.setdefault(criterion["check"], []).append(criterion)
        return list(groups.values())


class _WithRows(RubricsVerifierTaskStep):
    rows: list[dict] = []

    async def unjudged_rows(self, context: TaskStepContext, prompt_response: PromptResponse) -> list[dict]:
        return self.rows


def _step(cls: type[RubricsVerifierTaskStep] = RubricsVerifierTaskStep, **kw) -> RubricsVerifierTaskStep:
    params = dict(id="verify", version=1, criteria=CHECKS, prompt_id="p1", use_agent_judge=False,
                  use_trajectory=False, verifier_id="vt")
    params.update(kw)
    return cls(**params)


@pytest.mark.asyncio
async def test_a_subclass_groups_the_criteria_into_its_own_calls():
    v = _step(_ByCheck)
    prompts: list[str] = []
    _fake_judge(v, prompts)
    ctx = _context()

    await v.execute(ctx)

    assert sorted(_prompted_ids(p) for p in prompts) == [["c1", "c2"], ["c3", "c4"]]
    entry = ctx.metadata["verifications"]["vt"]
    assert [r["id"] for r in entry["results"]] == ["asked", "asked-item", "replied", "replied-item"]
    assert entry["judge_batches"] == 2


@pytest.mark.asyncio
async def test_batches_that_drop_a_criterion_are_refused():
    class Dropping(RubricsVerifierTaskStep):
        def _criteria_batches(self, criteria: list[dict], size: Optional[int]) -> list[list[dict]]:
            return [criteria[:1], criteria[2:]]

    v = _step(Dropping)
    _fake_judge(v, [])

    with pytest.raises(RuntimeError, match="must put every criterion in exactly one batch"):
        await v.execute(_context())


@pytest.mark.asyncio
async def test_unjudged_rows_join_the_entry_and_its_score_but_never_reach_the_judge():
    v = _step(_WithRows, score_aggregator="weighted_average")
    v.rows = [{"id": "auto-asked", "result": True, "score": 1.0},
              {"id": "always-na", "result": False, "score": 0.0, "weight": 0},
              {"id": "missing-line", "result": False, "score": 0.0}]
    prompts: list[str] = []
    _fake_judge(v, prompts)
    ctx = _context()

    await v.execute(ctx)

    assert [_prompted_ids(p) for p in prompts] == [["c1", "c2", "c3", "c4"]]
    entry = ctx.metadata["verifications"]["vt"]
    assert [r["id"] for r in entry["results"]] == [
        "asked", "asked-item", "replied", "replied-item", "auto-asked", "always-na", "missing-line"]
    assert entry["score"] == pytest.approx(5 / 6)  # the weight-0 row is recorded and never counts


@pytest.mark.asyncio
async def test_an_unmet_unjudged_row_fails_a_step_that_must_pass():
    v = _step(_WithRows, fail_task_when_not_passing=True)
    v.rows = [{"id": "missing-line", "result": False, "score": 0.0}]
    _fake_judge(v, [])

    with pytest.raises(VerifierNotPassingError, match=r"rows not met: \['missing-line'\]"):
        await v.execute(_context())


@pytest.mark.asyncio
async def test_an_unjudged_row_with_a_criterions_id_is_refused_before_the_judge_runs():
    v = _step(_WithRows)
    v.rows = [{"id": "replied", "result": True}]
    prompts: list[str] = []
    _fake_judge(v, prompts)

    with pytest.raises(RuntimeError, match=r"unjudged_rows gave rows with a criterion's id: \['replied'\]"):
        await v.execute(_context())
    assert prompts == []


@pytest.mark.asyncio
async def test_a_skipped_run_holds_no_unjudged_rows():
    v = _step(_WithRows)
    v.rows = [{"id": "auto-asked", "result": True, "score": 1.0}]
    ctx = TaskStepContext(prompt_responses=[PromptResponse(prompt_id="p1", response="", error_type="timeout")])

    await v.execute(ctx)

    assert [r["id"] for r in ctx.metadata["verifications"]["vt"]["results"]] == ["prompt_error"]


class _HooksCalled(RubricsVerifierTaskStep):
    calls: list[str] = []

    async def judge_trajectory_content(self, context: TaskStepContext, prompt_response: PromptResponse) -> None:
        self.calls.append("judge_trajectory_content")

    async def unjudged_rows(self, context: TaskStepContext, prompt_response: PromptResponse) -> list[dict]:
        self.calls.append("unjudged_rows")
        return [{"id": "auto-asked", "result": True, "score": 1.0}]


def _evidence_step(cls: type[RubricsVerifierTaskStep], **kw) -> RubricsVerifierTaskStep:
    params = dict(id="verify", version=1, criteria=CRITERIA, prompt_id="p1", use_agent_judge=False,
                  use_trajectory=True, trajectory_filter=_filter(), verifier_id="vid",
                  output_format=JudgeOutputFormat.RUBRIC_EVIDENCE,
                  grading_policy_prompt="Type every value in full; never accept an autocomplete suggestion.")
    params.update(kw)
    return cls(**params)


@pytest.mark.asyncio
async def test_a_run_skipped_for_want_of_screenshots_never_calls_the_hooks(monkeypatch):
    v = _evidence_step(_HooksCalled)
    v.calls = []
    _capture_judge(monkeypatch, v, lambda criteria: pytest.fail("the judge must not be called"))
    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: _NO_SCREENSHOTS)
    ctx = _ctx()

    await v.execute(ctx)

    assert v.calls == []
    assert [r["id"] for r in ctx.metadata["verifications"]["vid"]["results"]] == ["no_screenshots"]


class _OnePerCriterion(RubricsVerifierTaskStep):
    def _criteria_batches(self, criteria: list[dict], size: Optional[int]) -> list[list[dict]]:
        return [[c] for c in criteria]


@pytest.mark.asyncio
@pytest.mark.parametrize("fmt", [JudgeOutputFormat.RUBRIC_EVIDENCE, JudgeOutputFormat.RUBRIC_EVIDENCE_WITH_MISTAKES])
async def test_a_subclass_cannot_split_a_format_graded_in_one_call(monkeypatch, fmt):
    v = _evidence_step(_OnePerCriterion, output_format=fmt)
    captured = _capture_judge(monkeypatch, v, _passing_rows)
    ctx = _ctx()

    with pytest.raises(RuntimeError, match=rf"_criteria_batches gave 2 batches, but output_format={fmt.value} is "
                                           "graded in one judge call"):
        await v.execute(ctx)
    assert "eval_prompt" not in captured and "verifications" not in ctx.metadata


@pytest.mark.asyncio
async def test_a_subclass_keeping_one_batch_grades_a_format_graded_in_one_call(monkeypatch):
    class OneBatch(RubricsVerifierTaskStep):
        def _criteria_batches(self, criteria: list[dict], size: Optional[int]) -> list[list[dict]]:
            return [list(reversed(criteria))]

    v = _evidence_step(OneBatch)
    _capture_judge(monkeypatch, v, lambda criteria: ResultRows(
        _passing_rows(criteria), reasoning="Both shown on screen.", checks=[{"id": "typed", "passed": True}]))
    ctx = _ctx()

    await v.execute(ctx)

    entry = ctx.metadata["verifications"]["vid"]
    assert entry["score"] == 1.0 and entry["reasoning"] == "Both shown on screen."
    assert entry["checks"] == [{"id": "typed", "passed": True}]


class _RunInline(RubricsVerifierTaskStep):
    async def judge_trajectory_content(self, context: TaskStepContext, prompt_response: PromptResponse) -> str:
        return "THE RUN: the user sim asked for a table at 7."


@pytest.mark.asyncio
async def test_trajectory_content_from_a_subclass_comes_before_the_criteria_in_every_batch():
    v = _step(_RunInline, judge_batch_size=2)
    prompts: list[str] = []
    _fake_judge(v, prompts)

    await v.execute(_context())

    assert len(prompts) == 2
    assert all(p.index("THE RUN") < p.index('"id": "c') for p in prompts)


@pytest.mark.asyncio
async def test_without_trajectory_content_the_prompt_has_no_trajectory_section():
    with_run, without = _step(_RunInline), _step()
    prompts: list[str] = []
    bare: list[str] = []
    _fake_judge(with_run, prompts)
    _fake_judge(without, bare)

    await with_run.execute(_context())
    await without.execute(_context())

    assert "THE RUN" not in bare[0] and len(prompts[0]) > len(bare[0])


@pytest.mark.asyncio
async def test_trajectory_content_beside_a_recorded_trajectory_is_refused():
    v = _step(_RunInline, use_trajectory=True)
    ctx = TaskStepContext(prompt_responses=[PromptResponse(
        prompt_id="p1", response="done", prompt_text="do it", agent_trajectory_object_url="s3://t/1.json")])

    with pytest.raises(RuntimeError, match="judge_trajectory_content stands in for the recorded trajectory"):
        await v.execute(ctx)


def test_the_rubric_binary_and_partial_formats_alone_are_batchable_and_all_check_bulk_verdicts():
    assert {f for f, spec in JUDGE_OUTPUT_FORMAT_REGISTRY.items() if spec.batchable} == {
        JudgeOutputFormat.RUBRIC_BINARY, JudgeOutputFormat.RUBRIC_PARTIAL}
    assert all(spec.checks_bulk_verdicts for spec in JUDGE_OUTPUT_FORMAT_REGISTRY.values())


def _with_spec(**fields) -> type[RubricsVerifierTaskStep]:
    class WithSpec(RubricsVerifierTaskStep):
        def _spec(self) -> JudgeOutputFormatSpec:
            return replace(super()._spec(), **fields)

    return WithSpec


def test_a_spec_that_isnt_batchable_refuses_a_batch_size():
    with pytest.raises(ValueError, match="judge_batch_size does not apply to output_format=rubric_binary"):
        _step(_with_spec(batchable=False), judge_batch_size=2)


@pytest.mark.asyncio
@pytest.mark.parametrize(("checks", "calls"), [(True, 2), (False, 1)])
async def test_a_spec_can_turn_off_the_bulk_verdict_check(checks, calls):
    v = _step(_with_spec(checks_bulk_verdicts=checks), judge_batch_size=4)
    prompts: list[str] = []
    _fake_judge(v, prompts, respond=lambda ids: _graded(ids, justification=lambda cid: "Pass." if len(prompts) == 1
                                                         else f"Checked record {cid} with the sheet tool"))
    ctx = _context()

    await v.execute(ctx)

    assert len(prompts) == calls
    assert ctx.metadata["verifications"]["vt"]["score"] == 1.0
