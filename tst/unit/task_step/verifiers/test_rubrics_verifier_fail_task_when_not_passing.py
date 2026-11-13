"""fail_task_when_not_passing: a verifier whose verdict isn't passing (score below 1.0) fails its step, and so its
task, once the verdict is recorded; the score itself is computed as before."""
from __future__ import annotations

import pytest

from agent_env.runner.runner import RunStatus, run_outcome
from agent_env.task import Task
from agent_env.task_step.context import PromptResponse, TaskStepContext, VerifierNotPassingError
from agent_env.task_step.task_step import RetryConfig
from agent_env.task_step.task_steps.verifiers.aggregate_verifiers import AggregateVerifiersTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

CRITERIA = [{"id": "world", "description": "the events fired"}, {"id": "replies", "description": "everyone replied"}]


def _verifier(**kw) -> RubricsVerifierTaskStep:
    params = dict(
        id="validity", version=1, criteria=CRITERIA, prompt_id="p1", use_agent_judge=False, use_trajectory=True,
        verifier_id="validity", fail_task_when_not_passing=True,
    )
    params.update(kw)
    return RubricsVerifierTaskStep(**params)


def _context(**prompt) -> TaskStepContext:
    return TaskStepContext(prompt_responses=[PromptResponse(
        prompt_id="p1", response="done", prompt_text="do it", agent_trajectory_object_url="s3://t/1.json", **prompt)])


def _judge(monkeypatch, verifier, met: dict[str, bool]) -> None:
    async def fake_run(**kwargs):
        rows = [{**c, "score": 1.0 if met[c["id"]] else 0.0, "result": met[c["id"]]} for c in verifier.criteria]
        return rows, 0, [], None

    monkeypatch.setattr(verifier, "_read_trajectory_text", lambda uri: "EVENTS")
    monkeypatch.setattr(verifier, "_run_judge_with_output_retries", fake_run)


@pytest.mark.asyncio
async def test_a_passing_verdict_leaves_the_step_successful(monkeypatch):
    v = _verifier()
    _judge(monkeypatch, v, {"world": True, "replies": True})
    ctx = _context()

    await v.execute(ctx)

    assert ctx.metadata["verifications"]["validity"]["score"] == 1.0


@pytest.mark.asyncio
async def test_an_unmet_row_fails_the_step_naming_it_and_keeps_the_verdict(monkeypatch):
    v = _verifier()
    _judge(monkeypatch, v, {"world": False, "replies": True})
    ctx = _context()

    not_passing = r"Verifier 'validity' isn't passing \(score 0.0\); rows not met: \['world'\]"
    with pytest.raises(VerifierNotPassingError, match=not_passing):
        await v.execute(ctx)

    assert [r["result"] for r in ctx.metadata["verifications"]["validity"]["results"]] == [False, True]


@pytest.mark.asyncio
async def test_a_partial_weighted_score_is_not_passing(monkeypatch):
    v = _verifier(score_aggregator="weighted_average")
    _judge(monkeypatch, v, {"world": True, "replies": False})

    with pytest.raises(VerifierNotPassingError, match=r"\(score 0.5\); rows not met: \['replies'\]"):
        await v.execute(_context())


@pytest.mark.asyncio
async def test_a_penalized_negative_row_is_not_passing(monkeypatch):
    v = _verifier(score_aggregator="weighted_average",
                  criteria=[CRITERIA[0], {"id": "penalty", "description": "a negative-weight row", "weight": -1}])
    _judge(monkeypatch, v, {"world": True, "penalty": False})

    with pytest.raises(VerifierNotPassingError, match=r"rows not met: \['penalty'\]"):
        await v.execute(_context())


@pytest.mark.asyncio
async def test_a_run_the_judge_could_not_grade_is_not_passing(monkeypatch):
    v = _verifier()
    _judge(monkeypatch, v, {"world": True, "replies": True})
    ctx = _context(error_type="timeout")

    with pytest.raises(VerifierNotPassingError, match=r"rows not met: \['prompt_error'\]"):
        await v.execute(ctx)

    assert ctx.metadata["verifications"]["validity"]["score"] == 0


@pytest.mark.asyncio
async def test_without_the_option_a_failing_verdict_is_only_recorded(monkeypatch):
    v = _verifier(fail_task_when_not_passing=False)
    _judge(monkeypatch, v, {"world": False, "replies": False})
    ctx = _context()

    await v.execute(ctx)

    assert ctx.metadata["verifications"]["validity"]["score"] == 0.0


def test_the_option_needs_fail_task_on_error():
    problem = "fail_task_when_not_passing needs fail_task_on_error"
    with pytest.raises(ValueError, match=problem):
        _verifier(fail_task_on_error=False)
    with pytest.raises(ValueError, match=problem):
        RubricsVerifierTaskStep.from_dict(
            {**_verifier(fail_task_when_not_passing=False).to_dict(), "fail_task_on_error": False,
             "fail_task_when_not_passing": True})


def test_the_option_is_stored_only_when_set():
    assert RubricsVerifierTaskStep.from_dict(_verifier().to_dict()).fail_task_when_not_passing is True
    assert "fail_task_when_not_passing" not in _verifier(fail_task_when_not_passing=False).to_dict()


@pytest.mark.asyncio
async def test_a_task_fails_at_the_verifier_and_never_scores(monkeypatch, local_stores):
    v = _verifier()
    _judge(monkeypatch, v, {"world": False, "replies": True})
    final = AggregateVerifiersTaskStep(id="final-score", version=None, verifier_ids=["validity"],
                                       verifier_id="final-score", depends_on=["validity"])
    ctx = _context()

    with pytest.raises(VerifierNotPassingError) as caught:
        await Task(id="t", version=None, steps=[v, final]).run(context=ctx)

    assert run_outcome(ctx, caught.value)[0] is RunStatus.FAILED
    assert [f["step_id"] for f in ctx.metadata["failed_steps"]] == ["validity"]
    assert "final-score" not in ctx.metadata["verifications"]


@pytest.mark.asyncio
async def test_a_retry_config_does_not_grade_the_run_again(monkeypatch, local_stores):
    v = _verifier()
    v.retry_config = RetryConfig(retry_from_step_id="validity", max_retries=2)
    _judge(monkeypatch, v, {"world": False, "replies": True})
    graded = []
    grade = v._grade

    async def counting_grade(context):
        graded.append(context)
        return await grade(context)

    monkeypatch.setattr(v, "_grade", counting_grade)

    with pytest.raises(VerifierNotPassingError):
        await Task(id="t", version=None, steps=[v]).run(context=_context())

    assert len(graded) == 1
