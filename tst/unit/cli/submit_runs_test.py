"""``task run --submit`` and ``eval run --submit`` hand their runs to the configured ``[runner]`` and follow them
from the run index: here ``LocalRunner``, which runs them in this process and tears them down itself."""

import asyncio
import logging
import os
import signal
import threading
import types

import pytest
from click.testing import CliRunner

from agent_env.cli import cli, submitted_runs
from agent_env.eval import Eval
from agent_env.runner import local_runner
from agent_env.runner import store as run_store
from agent_env.runner.runner import RunHandle, Runner, RunRecord, RunStatus
from agent_env.task import Task
from agent_env.task.teardown import TeardownReport

pytestmark = pytest.mark.usefixtures("sigint_handled", "local_stores")


class _Task:
    id, version, steps = "t", 1, []

    def __init__(self, nap=0.0, failed_steps=None):
        self.nap, self.failed_steps, self.contexts = nap, failed_steps, []

    async def run(self, context, **_):
        self.contexts.append(context)
        await asyncio.sleep(self.nap)
        if self.failed_steps:
            context.metadata["failed_steps"] = self.failed_steps
        return context


@pytest.fixture
def torn_down(monkeypatch):
    torn = []

    async def teardown_run(context):
        torn.append(context)
        return TeardownReport()

    monkeypatch.setattr(local_runner, "teardown_run", teardown_run)
    monkeypatch.setattr(submitted_runs, "POLL_SECONDS", 0.02)
    return torn


def _invoke(monkeypatch, task, *args, signal_after=None):
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: task))
    timer = threading.Timer(signal_after, lambda: os.kill(os.getpid(), signal.SIGINT)) if signal_after else None
    if timer:
        timer.start()
    logging.disable(logging.CRITICAL)
    try:
        return CliRunner().invoke(cli, list(args))
    finally:
        if timer:
            timer.cancel()
        logging.disable(logging.NOTSET)


def test_a_submitted_run_goes_through_the_runner_and_lands_in_the_run_index(monkeypatch, tmp_path, torn_down):
    task = _Task()

    result = _invoke(monkeypatch, task, "task", "run", "--id", "t", "--submit", "--output-dir", str(tmp_path))

    assert result.exit_code == 0, result.output
    assert "Submitted run local-" in result.output and "Run COMPLETED" in result.output
    assert "Task completed!" in result.output
    [record] = run_store.list_runs(task_id="t")
    assert record.status is RunStatus.COMPLETED
    assert task.contexts[0].metadata["workflow_id"] == record.run_id
    assert torn_down == task.contexts


def test_a_failed_submitted_run_fails_the_command(monkeypatch, tmp_path, torn_down):
    task = _Task(failed_steps=[{"step_id": "s1"}])

    result = _invoke(monkeypatch, task, "task", "run", "--id", "t", "--submit", "--k", "2",
                     "--output-dir", str(tmp_path))

    assert result.exit_code == 1, result.output
    assert "0/2 runs completed, 2/2 failed." in result.output
    assert {r.status for r in run_store.list_runs(task_id="t")} == {RunStatus.FAILED}


def test_ctrl_c_cancels_the_submitted_runs_through_the_runner_and_exits_130(monkeypatch, tmp_path, torn_down):
    task = _Task(nap=30)

    result = _invoke(monkeypatch, task, "task", "run", "--id", "t", "--submit", "--k", "2",
                     "--output-dir", str(tmp_path), signal_after=1.0)

    assert result.exit_code == 130, result.output
    assert "Cancelling the runs (Ctrl-C again to stop waiting for them)" in result.output
    assert {r.status for r in run_store.list_runs(task_id="t")} == {RunStatus.CANCELED}
    assert len(torn_down) == 2


@pytest.mark.parametrize("flag", [["--keep"], ["--gateway-env-id", "g"]])
def test_submit_refuses_what_acts_on_this_process_only(monkeypatch, flag):
    result = _invoke(monkeypatch, _Task(), "task", "run", "--id", "t", "--submit", *flag)

    assert result.exit_code == 2
    assert "act on this process only" in result.output


def test_an_eval_submits_every_task_k_times(monkeypatch, tmp_path, torn_down):
    task = _Task()
    monkeypatch.setattr(Eval, "get", classmethod(lambda cls, id, version=None: types.SimpleNamespace(
        id="e", version=1, tasks=[types.SimpleNamespace(task_id="t", task_version=1)])))

    result = _invoke(monkeypatch, task, "eval", "run", "--id", "e", "--submit", "--k", "2")

    assert result.exit_code == 0, result.output
    assert result.output.count("PASSED") == 2 and "All 2 runs completed!" in result.output
    assert len(run_store.list_runs(task_id="t")) == 2


def test_an_eval_submit_leaves_concurrency_to_the_runner(monkeypatch):
    result = _invoke(monkeypatch, _Task(), "eval", "run", "--id", "e", "--submit", "--max-concurrency", "2")

    assert result.exit_code == 2
    assert "drop --max-concurrency" in result.output


class _Runner(Runner):
    """Gives every run ``run_status`` at submit; ``stop()`` naps, then records that it finished."""

    type = "fake"

    def __init__(self, status_error=None, cancel_error=None, stop_seconds=0.0, run_status=RunStatus.COMPLETED):
        self.status_error, self.cancel_error, self.stop_seconds = status_error, cancel_error, stop_seconds
        self.run_status, self.stopped, self.records = run_status, False, {}

    async def submit(self, task_id, task_version=None, **_):
        record = RunRecord(run_id=f"r{len(self.records)}", runner=self.type, task_id=task_id,
                           task_version=task_version, instance_id=f"i{len(self.records)}", status=self.run_status)
        self.records[record.run_id] = record
        return RunHandle(record.run_id, record.instance_id)

    async def status(self, run_id):
        if self.status_error:
            raise self.status_error
        return self.records[run_id]

    async def cancel(self, run_id):
        if self.cancel_error:
            raise self.cancel_error
        return False

    async def stop(self):
        await asyncio.sleep(self.stop_seconds)
        self.stopped = True


def test_an_error_following_the_runs_still_stops_the_runner(monkeypatch, local_stores):
    runner = _Runner(status_error=RuntimeError("run index unreachable"))
    local_stores.set_runner(runner)

    result = _invoke(monkeypatch, _Task(), "task", "run", "--id", "t", "--submit")

    assert isinstance(result.exception, RuntimeError)
    assert runner.stopped


def test_a_first_ctrl_c_lets_the_runner_finish_stopping(monkeypatch, local_stores):
    runner = _Runner(stop_seconds=2.0)
    local_stores.set_runner(runner)

    result = _invoke(monkeypatch, _Task(), "task", "run", "--id", "t", "--submit", signal_after=0.5)

    assert result.exit_code == 130, result.output
    assert runner.stopped


def test_an_error_cancelling_after_ctrl_c_still_stops_the_runner(monkeypatch, local_stores):
    runner = _Runner(cancel_error=RuntimeError("runner unreachable"), run_status=RunStatus.RUNNING)
    local_stores.set_runner(runner)

    result = _invoke(monkeypatch, _Task(), "task", "run", "--id", "t", "--submit", signal_after=0.5)

    assert isinstance(result.exception, RuntimeError)
    assert runner.stopped
