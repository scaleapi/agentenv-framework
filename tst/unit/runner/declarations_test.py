"""What a runner declares to the eval engine: neither declaration by default, and the conformance kit holds a runner to
each either way, so a runner whose runs run in processes of their own, outliving the process that submitted them, and
that runs a run again after an infrastructure failure passes the cases LocalRunner, which declares neither, passes."""

import asyncio
import os
import subprocess
import sys
import uuid

import pytest

from agent_env.config import get_config, reset_config
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
ATTEMPTS = 3


class DetachedRunner(Runner):
    """Runs each run in a process of its own that outlives the submitter, as a remote worker would, and starts a fresh
    attempt on a new instance after a ConnectionError."""

    type = "detached"
    runs_outlive_submitter = True
    retries_infra = True

    def __init__(self) -> None:
        self.workers: list[subprocess.Popen] = []

    async def submit(self, task_id, task_version=None, *, metadata=None, **_) -> RunHandle:
        record = run_store.insert_run(RunRecord(
            run_id=f"detached-{uuid.uuid4().hex}", runner=self.type, task_id=task_id, task_version=task_version,
            instance_id=new_instance_id(task_id)))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, (str(conformance._REPO),
                                                                           os.environ.get("PYTHONPATH"))))}
        self.workers.append(subprocess.Popen(
            [sys.executable, "-m", "tst.unit.runner.declarations_test", record.run_id], env=env,
            start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        return RunHandle(run_id=record.run_id, instance_id=record.instance_id)

    async def status(self, run_id):
        return run_store.get_run(run_id)

    async def cancel(self, run_id):
        return False


def _work(run_id: str) -> None:
    conformance.register_step()
    record = run_store.get_run(run_id)
    run_store.set_running(run_id)
    task = Task.get(record.task_id, record.task_version)
    for _ in range(ATTEMPTS):
        try:
            context = asyncio.run(task.run(instance_id=new_instance_id(record.task_id)))
        except ConnectionError:
            continue
        run_store.mark_terminal(run_id, *run_outcome(context))
        return
    run_store.mark_terminal(run_id, RunStatus.FAILED, error=f"failed {ATTEMPTS} attempts")


@pytest.fixture
def detached_runner(local_stores, tmp_path, monkeypatch):
    """The detached runner as the configured ``[runner]``, which the kit's submitting process builds too."""
    config = tmp_path / "config.toml"
    config.write_text(f'[runner]\nimpl = "{__name__}:{DetachedRunner.__name__}"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
    reset_config()
    steps = {**Config().task_step_registry(), conformance.Step.type: conformance.Step}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    runner = get_config().get_runner()
    assert isinstance(runner, DetachedRunner)
    yield runner
    for worker in runner.workers:
        try:
            worker.wait(timeout=30)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
    reset_config()


def test_a_runner_declares_neither_unless_it_says_so_and_the_local_runner_says_neither():
    assert (Runner.runs_outlive_submitter, Runner.retries_infra) == (False, False)
    assert (LocalRunner.runs_outlive_submitter, LocalRunner.retries_infra) == (False, False)


@pytest.mark.asyncio
@pytest.mark.parametrize("case", DECLARATION_CASES, ids=lambda case: case.__name__)
async def test_a_runner_declaring_both_passes_the_cases_that_hold_it_to_them(case, detached_runner):
    await case(detached_runner)


if __name__ == "__main__":
    _work(sys.argv[1])
