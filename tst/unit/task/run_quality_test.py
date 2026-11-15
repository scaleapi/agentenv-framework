"""A completed run carries ``quality``: degraded when it tolerated a step failure that nothing repaired."""

from __future__ import annotations

import pytest

from agent_env.config import set_document_store
from agent_env.store import LocalSqliteDocumentStore
from agent_env.store.document_store import Filter
from agent_env.task.outcome import FailedStep, RunQuality, StepRole, run_quality
from agent_env.task.store import TASK_INSTANCES_COLLECTION, TaskInstance, TaskInstanceStore, set_task_instance_store
from agent_env.task.task import Task
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import RetryConfig
from agent_env.task_step.task_steps.verifiers import AgentPromptResponseVerifierTaskStep


def _failed(role: StepRole, **fields) -> FailedStep:
    return FailedStep(**{
        "step_id": role.value, "step_type": "t", "error": "boom", "error_type": "RuntimeError",
        "started_at_utc": "2026-01-01T00:00:00+00:00", "duration_seconds": 1.0, "is_fatal": False,
        "step_role": role, **fields,
    })


def test_a_run_with_no_failures_is_clean():
    assert run_quality([]) == RunQuality()
    assert RunQuality().to_dict() == {"degraded": False, "reasons": []}


def test_open_tolerated_failures_degrade_the_run_most_severe_first():
    quality = run_quality([_failed(StepRole.EVIDENCE), _failed(StepRole.GRADER), _failed(StepRole.EVIDENCE)])
    assert quality.to_dict() == {"degraded": True, "reasons": ["grader", "evidence"]}


@pytest.mark.parametrize("fields", [{"is_fatal": True}, {"retried": True}, {"resolved_by": "regrade:2026-01-02"}])
def test_a_fatal_retried_or_resolved_failure_does_not_degrade_the_run(fields):
    assert not run_quality([_failed(StepRole.GRADER, **fields)]).degraded


def test_an_entry_written_before_roles_existed_degrades_as_other():
    legacy = {"step_id": "s", "step_type": "run_code", "error": "boom", "error_type": "RuntimeError",
              "started_at_utc": "2026-01-01T00:00:00+00:00", "duration_seconds": 1.0, "is_fatal": False}
    assert run_quality([FailedStep.from_dict(legacy)]).to_dict() == {"degraded": True, "reasons": ["other"]}


def test_a_run_completed_before_quality_existed_reads_as_clean():
    doc = {"instance_id": "i", "task_id": "t", "task_version": 1, "status": "completed",
           "current_step": 1, "total_steps": 1}
    assert TaskInstance.from_dict(doc).quality == RunQuality()


class _FailingVerifier(AgentPromptResponseVerifierTaskStep):
    def __init__(self, *args, fail_times: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.fail_times = fail_times

    async def execute(self, context):
        if self.fail_times:
            self.fail_times -= 1
            raise RuntimeError("Judge A2A task failed: API Error: 503 Service Unavailable")
        return context


@pytest.fixture
def store(tmp_path):
    doc = LocalSqliteDocumentStore(str(tmp_path / "instances.db"))
    doc.ensure_index(TASK_INSTANCES_COLLECTION, ["instance_id"], unique=True)
    set_document_store(doc)
    s = TaskInstanceStore()
    set_task_instance_store(s)
    try:
        yield s
    finally:
        set_task_instance_store(None)


def _stored_quality(store, instance_id):
    return store._doc_store.find_one(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))["quality"]


@pytest.mark.asyncio
async def test_completing_with_a_tolerated_grader_failure_stamps_a_degraded_quality(store):
    grader = _FailingVerifier("grade", None, prompt_id="p", fail_task_on_error=False, fail_times=1)
    ctx = TaskStepContext()
    await Task(id="t", version=1, steps=[grader]).run(context=ctx)

    assert _stored_quality(store, ctx.instance_id) == {"degraded": True, "reasons": ["grader"]}
    instance = store.get(ctx.instance_id)
    assert instance.status == "completed"
    assert instance.quality == RunQuality((StepRole.GRADER,))


@pytest.mark.asyncio
async def test_completing_after_a_successful_retry_stamps_a_clean_quality(store):
    grader = _FailingVerifier("grade", None, prompt_id="p", fail_task_on_error=False, fail_times=1)
    grader.retry_config = RetryConfig(retry_from_step_id="grade", max_retries=1)
    ctx = TaskStepContext()
    result = await Task(id="t", version=1, steps=[grader]).run(context=ctx)

    assert [f.get("retried") for f in result.metadata["failed_steps"]] == [True]
    assert _stored_quality(store, ctx.instance_id) == {"degraded": False, "reasons": []}
