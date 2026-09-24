"""Integration coverage for deterministic task instance retry resume."""

import uuid

import pytest

from agent_env.config import get_config
from agent_env.store import Filter
from agent_env.task import Task
from agent_env.task.step_journal import commit_ordered, replay_context
from agent_env.task.store import TASK_INSTANCES_COLLECTION, get_task_instance_store
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep
from tst.task import journal_invariants as journal


def _instances():
    """Document-store handle for setup/verify/teardown in these integration tests."""
    return get_config().get_document_store()


class _MetadataStep(TaskStep):
    type = "_synthetic_metadata_step"

    def __init__(self, id: str):
        super().__init__(id=id, version=None)

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        context.metadata[self.id] = True
        return context


class _FailingStep(TaskStep):
    type = "_synthetic_failing_step"

    def __init__(self, id: str):
        super().__init__(id=id, version=None)

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        raise RuntimeError(f"{self.id} failed")


class _PartialWriteFailingStep(TaskStep):
    """Fails, but only after it has already written to the context."""

    type = "_synthetic_partial_write_failing_step"

    def __init__(self, id: str):
        super().__init__(id=id, version=None)

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        context.metadata[f"{self.id}_partial"] = True
        raise RuntimeError(f"{self.id} failed")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_resume_happy_path_completes_one_row():
    task_id = f"retry-resume-happy-test-{uuid.uuid4().hex[:8]}"
    instance_id = f"{task_id}-deterministic"
    store = get_task_instance_store()

    task = Task(
        id=task_id,
        version=1,
        steps=[
            _MetadataStep("s1"),
            _MetadataStep("s2"),
            _MetadataStep("s3"),
        ],
    )

    try:
        await task.run(instance_id=instance_id, end_step=2)

        partial = store.get(instance_id)
        assert partial.status == "running"
        assert partial.current_step == 2
        assert {s.step_id for s in partial.completed_steps} == {"s1", "s2"}
        partial_doc = journal.assert_journal_replays(instance_id, expect_steps={"s1", "s2"})
        assert partial_doc["run_generation"] == 0

        await task.run(
            instance_id=instance_id,
            start_step=2,
            context=TaskStepContext(metadata={"from_heartbeat": True}),
        )

        docs = _instances().query(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))
        assert len(docs) == 1

        doc = docs[0]
        assert doc["status"] == "completed"
        assert doc["current_step"] == 3
        assert doc["completed_at_utc"] is not None
        assert doc["error"] is None
        assert {
            (step["step_id"], step["status"])
            for step in doc["completed_steps"]
        } == {
            ("s1", "success"),
            ("s2", "success"),
            ("s3", "success"),
        }
        assert doc["context"]["metadata"]["from_heartbeat"] is True
        assert doc["context"]["metadata"]["s3"] is True

        # The resume re-registers and journals every completion it carries in, with an empty
        # diff: the seed already holds what s1/s2 wrote. So a completion with no entry now
        # means exactly one thing, its journal write failed, and no marker field is needed.
        assert doc["run_generation"] == 1
        journal.assert_journal_replays(instance_id, expect_steps={"s1", "s2", "s3"})
        rows = {r["step_id"]: r for r in journal.all_journal_rows(instance_id)}
        assert set(rows) == {journal.SEED, "s1", "s2", "s3"}
        assert {r["generation"] for r in rows.values()} == {1}, "a row from the old run was left behind"
        for sid in ("s1", "s2"):
            assert not any(rows[sid]["ops"].values()), f"{sid} was seeded complete; its diff must be empty"
    finally:
        _instances().delete(TASK_INSTANCES_COLLECTION, Filter.of(task_id=task_id))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_retry_after_failure_clears_failed_state_and_completes_one_row():
    task_id = f"retry-resume-failure-test-{uuid.uuid4().hex[:8]}"
    instance_id = f"{task_id}-deterministic"
    store = get_task_instance_store()

    failing_task = Task(
        id=task_id,
        version=1,
        steps=[
            _MetadataStep("s1"),
            _FailingStep("s2"),
        ],
    )
    fixed_task = Task(
        id=task_id,
        version=1,
        steps=[
            _MetadataStep("s1"),
            _MetadataStep("s2"),
        ],
    )

    try:
        with pytest.raises(RuntimeError, match="s2 failed"):
            await failing_task.run(instance_id=instance_id)

        failed = store.get(instance_id)
        assert failed.status == "failed"
        assert failed.error == "s2 failed"
        assert failed.completed_at_utc is not None
        assert {
            (step.step_id, step.status)
            for step in failed.completed_steps
        } == {
            ("s1", "success"),
            ("s2", "failure"),
        }
        # s2 raised before journaling. A failure completion is deliberately exempt from the
        # "completed with no entry" refusal, which is only safe because it wrote nothing first.
        assert {r["step_id"] for r in journal.all_journal_rows(instance_id)} == {journal.SEED, "s1"}

        await fixed_task.run(instance_id=instance_id)

        docs = _instances().query(TASK_INSTANCES_COLLECTION, Filter.of(instance_id=instance_id))
        assert len(docs) == 1

        doc = docs[0]
        assert doc["status"] == "completed"
        assert doc["current_step"] == 2
        assert doc["completed_at_utc"] is not None
        assert doc["error"] is None
        assert {
            (step["step_id"], step["status"])
            for step in doc["completed_steps"]
        } == {
            ("s1", "success"),
            ("s2", "success"),
        }
        assert doc["context"]["metadata"]["s1"] is True
        assert doc["context"]["metadata"]["s2"] is True
        assert doc["run_generation"] == 1
        journal.assert_journal_replays(instance_id, expect_steps={"s1", "s2"})
    finally:
        _instances().delete(TASK_INSTANCES_COLLECTION, Filter.of(task_id=task_id))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_failed_steps_partial_writes_reach_the_doc_but_not_the_journal():
    """Phase-1 limitation, pinned so a change is deliberate.

    A step that writes and then fails gets its writes onto the instance through
    ``record_task_failure``'s wholesale ``context`` write, which is not journaled. Replay
    therefore omits them and the journal is *not* a complete record of a failed run. The
    stacked retry PR is expected to journal the failing step with ``status=failure`` first;
    when it does, this test should start failing and be tightened to assert equality.
    """
    task_id = f"retry-resume-partial-test-{uuid.uuid4().hex[:8]}"
    instance_id = f"{task_id}-deterministic"

    failing_task = Task(
        id=task_id, version=1,
        steps=[_MetadataStep("s1"), _PartialWriteFailingStep("s2")],
    )
    fixed_task = Task(
        id=task_id, version=1,
        steps=[_MetadataStep("s1"), _MetadataStep("s2")],
    )

    try:
        with pytest.raises(RuntimeError, match="s2 failed"):
            await failing_task.run(instance_id=instance_id)

        doc = journal.instance_doc(instance_id)
        assert doc["context"]["metadata"]["s2_partial"] is True   # the failure write landed
        assert {r["step_id"] for r in journal.all_journal_rows(instance_id)} == {journal.SEED, "s1"}

        store = get_task_instance_store()
        entries = store.journal_entries_sync(instance_id)
        replayed = replay_context(
            commit_ordered(entries, doc["completed_steps"]), doc.get("rerecorded_steps")
        )
        assert replayed != doc["context"], "a failing step's partial writes are journaled now"
        assert "s2_partial" not in replayed["metadata"]

        await fixed_task.run(instance_id=instance_id)
        after = journal.assert_journal_replays(instance_id, expect_steps={"s1", "s2"})
        assert after["status"] == "completed"
        assert "s2_partial" not in after["context"]["metadata"]
    finally:
        _instances().delete(TASK_INSTANCES_COLLECTION, Filter.of(task_id=task_id))
