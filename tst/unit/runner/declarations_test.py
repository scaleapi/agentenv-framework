"""What a runner declares to the eval engine: neither declaration by default, and the conformance kit holds a runner
to each either way, so a stub whose runs carry on after ``stop`` and that runs a run again after an infrastructure
failure passes the cases LocalRunner, which declares neither, passes too."""

import asyncio
import uuid

import pytest

from agent_env.config.runtime import Config
from agent_env.runner import store as run_store
from agent_env.runner.local_runner import LocalRunner
from agent_env.runner.runner import RunHandle, Runner, RunRecord, RunStatus, run_outcome
from agent_env.task import Task
from agent_env.task.store import new_instance_id
from tst.runner import conformance

DECLARATION_CASES = [
    conformance.a_run_outlives_its_submitter_only_when_the_runner_says_so,
    conformance.a_run_is_retried_after_an_infrastructure_failure_only_when_the_runner_says_so,
]


class _DeclaringRunner(Runner):
    """Runs each run in this loop like a remote worker would: ``stop`` leaves it going, and a ConnectionError starts
    a fresh attempt on a new instance."""

    type = "declaring"
    runs_outlive_submitter = True
    retries_infra = True
    ATTEMPTS = 3

    def __init__(self) -> None:
        self.runs: set[asyncio.Task] = set()

    async def submit(self, task_id, task_version=None, *, metadata=None, **_) -> RunHandle:
        record = run_store.insert_run(RunRecord(
            run_id=f"declaring-{uuid.uuid4().hex}", runner=self.type, task_id=task_id, task_version=task_version,
            instance_id=new_instance_id(task_id), status=RunStatus.RUNNING))
        run = asyncio.create_task(self._run(record))
        self.runs.add(run)
        run.add_done_callback(self.runs.discard)
        return RunHandle(run_id=record.run_id, instance_id=record.instance_id)

    async def _run(self, record: RunRecord) -> None:
        task = Task.get(record.task_id, record.task_version)
        for _ in range(self.ATTEMPTS):
            try:
                context = await task.run(instance_id=new_instance_id(record.task_id))
            except ConnectionError:
                continue
            run_store.mark_terminal(record.run_id, *run_outcome(context))
            return
        run_store.mark_terminal(record.run_id, RunStatus.FAILED, error=f"failed {self.ATTEMPTS} attempts")

    async def status(self, run_id):
        return run_store.get_run(run_id)

    async def cancel(self, run_id):
        return False


def test_a_runner_declares_neither_unless_it_says_so_and_the_local_runner_says_neither():
    assert (Runner.runs_outlive_submitter, Runner.retries_infra) == (False, False)
    assert (LocalRunner.runs_outlive_submitter, LocalRunner.retries_infra) == (False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", DECLARATION_CASES, ids=lambda case: case.__name__)
async def test_a_runner_declaring_both_passes_the_cases_that_hold_it_to_them(case, local_stores, monkeypatch):
    steps = {**Config().task_step_registry(), conformance.Step.type: conformance.Step}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    runner = _DeclaringRunner()
    try:
        await case(runner)
    finally:
        await asyncio.gather(*runner.runs)
