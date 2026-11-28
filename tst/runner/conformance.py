"""Backend-neutral Runner conformance: what ``task run``, ``eval run`` and the explorer rely on from any ``[runner]``.

Each case is ``await case(runner)`` on a runner the caller never started, as the CLI holds one, and stops after.
The cases store their tasks with ``Task.put`` and read the run index and the task instances, so the runner's worker
must share the caller's stores and have ``Step`` in its task-step registry. Every runner must pass every case in
``CASES``.
"""

import asyncio
import time
import uuid

from agent_env.config import get_config
from agent_env.runner import store as run_store
from agent_env.runner.runner import RunStatus
from agent_env.store import Filter
from agent_env.task import Task
from agent_env.task.store import find_task_instance
from agent_env.task_step.task_step import TaskStep

ENDED_WITHIN_SECONDS = 60
POLL_SECONDS = 0.05
# How long the step a submitter stops under naps, and so how long a run that outlives it takes to end.
OUTLIVED_NAP_SECONDS = 1
ATTEMPTS_COLLECTION = "runner_conformance_attempts"


class Step(TaskStep):
    """Naps ``nap`` seconds, fails unless the run's metadata sets each key in ``needs``, and records that it ran
    under its id. With ``flaky`` set it records each attempt under that marker, and its first attempt fails as an
    infrastructure failure does, with a ConnectionError."""

    type = "runner_conformance"
    entity_refs = ()

    def __init__(self, nap=0, needs=(), flaky=None, **base):
        super().__init__(**base)
        self.nap, self.needs, self.flaky = nap, list(needs), flaky

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), nap=data.get("nap", 0), needs=data.get("needs", []),
                   flaky=data.get("flaky"))

    def to_dict(self):
        return {**super().to_dict(), "nap": self.nap, "needs": self.needs, "flaky": self.flaky}

    async def execute(self, context):
        if self.flaky:
            get_config().get_document_store().insert(ATTEMPTS_COLLECTION, {"flaky": self.flaky})
            if _attempts(self.flaky) == 1:
                raise ConnectionError("the conformance step's first attempt lost its connection")
        await asyncio.sleep(self.nap)
        if missing := [key for key in self.needs if not context.metadata.get(key)]:
            raise RuntimeError(f"conformance step needs {missing}")
        context.metadata[self.id] = True
        return context


async def a_run_completes_and_is_recorded_under_its_handle(runner):
    task = _put(Step(id="a", version=None))
    handle = await runner.submit(task.id, task.version, metadata={"carried": "yes"})
    record = await _until_ended(runner, handle.run_id)
    saved = run_store.get_run(handle.run_id)
    for record in (record, saved):
        assert (record.status, record.run_id, record.instance_id, record.task_id) == (
            RunStatus.COMPLETED, handle.run_id, handle.instance_id, task.id)
    metadata = find_task_instance(handle.instance_id).context["metadata"]
    assert (metadata["workflow_id"], metadata["carried"], metadata["a"]) == (handle.run_id, "yes", True)


async def a_run_whose_step_fails_fails_with_its_error(runner):
    task = _put(Step(id="a", version=None, needs=["never"]))
    handle = await runner.submit(task.id, task.version)
    record = await _until_ended(runner, handle.run_id)
    saved = run_store.get_run(handle.run_id)
    for record in (record, saved):
        assert record.status is RunStatus.FAILED and "conformance step needs ['never']" in record.error


async def a_cancelled_run_ends_canceled(runner):
    task = _put(Step(id="a", version=None, nap=10 * ENDED_WITHIN_SECONDS))
    handle = await runner.submit(task.id, task.version)
    await _until(runner, handle.run_id, lambda record: record.status is not RunStatus.QUEUED)
    assert await runner.cancel(handle.run_id) is True
    assert (await _until_ended(runner, handle.run_id)).status is RunStatus.CANCELED
    assert run_store.get_run(handle.run_id).status is RunStatus.CANCELED


async def a_run_given_an_instance_continues_it_from_start_step(runner):
    task = _put(Step(id="a", version=None, needs=["first"]), Step(id="b", version=None, needs=["a", "resumed"]))
    first = await runner.submit(task.id, task.version, metadata={"first": True})
    assert (await _until_ended(runner, first.run_id)).status is RunStatus.FAILED
    handle = await runner.submit(task.id, task.version, instance_id=first.instance_id,
                                 metadata={"start_step": 1, "first": False, "resumed": True})
    assert handle.instance_id == first.instance_id
    assert (await _until_ended(runner, handle.run_id)).status is RunStatus.COMPLETED


async def a_run_it_does_not_have_or_that_ended_cannot_be_cancelled(runner):
    task = _put(Step(id="a", version=None))
    handle = await runner.submit(task.id, task.version)
    await _until_ended(runner, handle.run_id)
    assert await runner.cancel(handle.run_id) is False
    assert await runner.cancel(f"no-such-run-{uuid.uuid4().hex}") is False
    assert await runner.status(f"no-such-run-{uuid.uuid4().hex}") is None


async def a_run_outlives_its_submitter_only_when_the_runner_says_so(runner):
    """A runner whose runs outlive their submitter finishes one after ``stop``, so an eval resumed later reattaches to
    it; any other leaves it unfinished, so the eval submits it again."""
    task = _put(Step(id="a", version=None, nap=OUTLIVED_NAP_SECONDS))
    handle = await runner.submit(task.id, task.version)
    await _until(runner, handle.run_id, lambda record: record.status is not RunStatus.QUEUED)
    await runner.stop()
    if runner.runs_outlive_submitter:
        assert (await _until_recorded(handle.run_id)).status is RunStatus.COMPLETED
    else:
        await asyncio.sleep(2 * OUTLIVED_NAP_SECONDS)
        assert run_store.get_run(handle.run_id).status is not RunStatus.COMPLETED


async def a_run_is_retried_after_an_infrastructure_failure_only_when_the_runner_says_so(runner):
    """A runner that retries infrastructure failures runs a run again after one, so the eval engine adds no attempt
    of its own; any other ends the run failed after one attempt, so the engine's attempt is the only one."""
    marker = uuid.uuid4().hex
    task = _put(Step(id="a", version=None, flaky=marker))
    handle = await runner.submit(task.id, task.version)
    record = await _until_ended(runner, handle.run_id)
    if runner.retries_infra:
        assert record.status is RunStatus.COMPLETED and _attempts(marker) >= 2
    else:
        assert (record.status, _attempts(marker)) == (RunStatus.FAILED, 1)
        assert "ConnectionError" in record.error


CASES = [
    a_run_completes_and_is_recorded_under_its_handle,
    a_run_whose_step_fails_fails_with_its_error,
    a_cancelled_run_ends_canceled,
    a_run_given_an_instance_continues_it_from_start_step,
    a_run_it_does_not_have_or_that_ended_cannot_be_cancelled,
    a_run_outlives_its_submitter_only_when_the_runner_says_so,
    a_run_is_retried_after_an_infrastructure_failure_only_when_the_runner_says_so,
]


def _put(*steps) -> Task:
    return Task.put(id=f"runner-conformance-{uuid.uuid4().hex[:12]}", steps=list(steps))


async def _until_ended(runner, run_id):
    return await _until(runner, run_id, lambda record: record.status.is_terminal)


async def _until_recorded(run_id):
    """The run's index record once it ends, read from the store rather than a runner that has stopped."""
    deadline = time.monotonic() + ENDED_WITHIN_SECONDS
    while not (record := run_store.get_run(run_id)).status.is_terminal:
        assert time.monotonic() < deadline, f"run {run_id} is still {record.status}"
        await asyncio.sleep(POLL_SECONDS)
    return record


def _attempts(marker: str) -> int:
    return get_config().get_document_store().count(ATTEMPTS_COLLECTION, Filter.of(flaky=marker))


async def _until(runner, run_id, done):
    deadline = time.monotonic() + ENDED_WITHIN_SECONDS
    while (record := await runner.status(run_id)) is None or not done(record):
        assert time.monotonic() < deadline, f"run {run_id} is still {record and record.status}"
        await asyncio.sleep(POLL_SECONDS)
    return record
