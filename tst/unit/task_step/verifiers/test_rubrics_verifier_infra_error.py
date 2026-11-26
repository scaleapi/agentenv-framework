"""A run whose agent turn failed for an infrastructure reason (``error_type`` infra_error) says nothing about the agent:
a rubrics verifier skips it, unscored, where any other failed turn is scored 0 unjudged, and an aggregate of such
sources is skipped too, so the run is unscored rather than scored below 1."""
from __future__ import annotations

import pytest

from agent_env.bundle.run import Outcome, TaskRun
from agent_env.task import Task
from agent_env.task.task import recorded_scores
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.aggregate_verifiers import AggregateVerifiersTaskStep
from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import (
    CompactionType,
    FrameStrategy,
    TrajectoryFilter,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep, trajectory_mistakes_key
from agent_env.task_step.task_steps.verifiers.verifier import GRADES_KEY, VerifierStatus

CRITERIA = [{"id": "done", "description": "it is done"}]
REASON = "the run's turn failed for an infrastructure reason (error_code=psi.gateway), not the assistant's"


def _verifier(monkeypatch, step_id: str, **kw) -> RubricsVerifierTaskStep:
    step = RubricsVerifierTaskStep(**{
        "id": step_id, "version": 1, "criteria": CRITERIA, "prompt_id": "p1", "use_agent_judge": False,
        "use_trajectory": True, "verifier_id": step_id, **kw})

    async def judge(**kwargs):
        raise AssertionError("a run that failed for an infrastructure reason is never judged")

    monkeypatch.setattr(step, "_read_trajectory_text", lambda uri: "EVENTS")
    monkeypatch.setattr(step, "_run_judge_with_output_retries", judge)
    return step


def _context(error_type: str = "infra_error") -> TaskStepContext:
    return TaskStepContext(prompt_responses=[PromptResponse(
        prompt_id="p1", response="[Mon] on it", prompt_text="plan the trip", agent_trajectory_object_url="s3://t/1.json",
        error_type=error_type, error_code="psi.gateway", error_message="APIStatusError: 502 Bad Gateway")])


@pytest.mark.asyncio
async def test_an_infra_error_run_is_skipped_unscored_not_scored_0(monkeypatch):
    ctx = _context()

    await _verifier(monkeypatch, "psi-trust").execute(ctx)

    grade = ctx.metadata[GRADES_KEY]["psi-trust"]
    assert (grade["verifier_status"], grade["score"]) == (VerifierStatus.SKIPPED, None)
    entry = ctx.metadata["verifications"]["psi-trust"]
    assert (entry["results"], entry["score"], entry["skipped"]) == ([], None, REASON)


@pytest.mark.asyncio
async def test_an_agent_error_run_is_still_scored_0_unjudged(monkeypatch):
    ctx = _context(error_type="agent_error")

    await _verifier(monkeypatch, "psi-trust").execute(ctx)

    [row] = ctx.metadata["verifications"]["psi-trust"]["results"]
    assert (row["id"], ctx.metadata["verifications"]["psi-trust"]["score"]) == ("prompt_error", 0)
    assert ctx.metadata[GRADES_KEY]["psi-trust"]["verifier_status"] == VerifierStatus.GRADED


@pytest.mark.asyncio
async def test_a_skip_does_not_fail_a_verifier_that_fails_its_task_when_not_passing(monkeypatch):
    ctx = _context()

    await _verifier(monkeypatch, "psi-trust", fail_task_when_not_passing=True).execute(ctx)

    assert ctx.metadata[GRADES_KEY]["psi-trust"]["verifier_status"] == VerifierStatus.SKIPPED


@pytest.mark.asyncio
async def test_the_trajectory_mistakes_entry_is_skipped_too(monkeypatch):
    ctx = _context()

    frames = TrajectoryFilter(compaction_type=CompactionType.SCREENSHOT, frame_strategy=FrameStrategy.PER_CRITERION)
    await _verifier(monkeypatch, "psi-process", output_format="rubric_evidence_with_mistakes",
                    trajectory_filter=frames).execute(ctx)

    mistakes = ctx.metadata["verifications"][trajectory_mistakes_key("psi-process")]
    assert (mistakes["results"], mistakes["score"], mistakes["skipped"]) == ([], None, REASON)


@pytest.mark.asyncio
async def test_a_verdict_over_sources_all_skipped_is_skipped_and_the_run_unscored(monkeypatch, local_stores):
    trust, process = _verifier(monkeypatch, "psi-trust"), _verifier(monkeypatch, "psi-process")
    verdict = AggregateVerifiersTaskStep(id="psi-verdict", version=None, sources=["psi-trust", "psi-process"],
                                         score_aggregator="all_pass", depends_on=["psi-trust", "psi-process"])
    task = Task(id="t", version=None, steps=[trust, process, verdict])
    ctx = _context()

    await task.run(context=ctx)

    grade = ctx.metadata[GRADES_KEY]["psi-verdict"]
    assert (grade["verifier_status"], grade["verifier_skip_reasoning"]) == (VerifierStatus.SKIPPED,
                                                                             "every source it weighs was skipped")
    assert recorded_scores(task.steps, ctx) == {}
    assert TaskRun(None, task, ctx, None, 0.0).outcome is Outcome.UNSCORED
