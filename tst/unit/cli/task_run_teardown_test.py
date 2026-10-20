"""``agent-env task run`` and ``task run-batch`` tear down what each run deployed, however it ends; ``--keep`` holds
it up until Ctrl-C, and Ctrl-C mid-run tears it down and exits 130. ``task run`` goes through the configured runner,
here ``LocalRunner``: it tears a run down, and the CLI tears down what ``--keep`` held, from the stored instance."""

import asyncio
import importlib
import logging
import os
import signal
import threading
import types

import pytest
from click.testing import CliRunner

from agent_env.cli import cli, submitted_runs
from agent_env.runner import local_runner
from agent_env.runner import store as run_store
from agent_env.runner.runner import RunStatus
from agent_env.task import Task
from agent_env.task.store import register_task_instance, seed_task_instance_context
from agent_env.task.teardown import TORN_DOWN_KEY, RecordedSandbox, TeardownReport
from agent_env.task_step.context import DeployedSandbox

task_run = importlib.import_module("agent_env.cli.task.run")  # the package's ``run`` is the click command
pytestmark = pytest.mark.usefixtures("sigint_handled", "local_stores")


class _Deploys:
    """Records a sandbox on a fake backend, then naps, then ends as told."""

    id, version, steps = "t", 1, []

    def __init__(self, nap=0.0, ending=None):
        self.nap, self.ending, self.contexts = nap, ending, []

    async def run(self, context, instance_id=None, **_):
        self.contexts.append(context)
        context.deployed_sandboxes.append(DeployedSandbox(
            sandbox_name="box", sandbox_id=f"sb-{len(self.contexts)}", sandbox_mode="vm", sandbox_type="fake"))
        if instance_id:
            register_task_instance("t", 1, 1, instance_id=instance_id)
            seed_task_instance_context(instance_id, context)
        await asyncio.sleep(self.nap)
        if self.ending:
            raise self.ending
        return context


@pytest.fixture
def torn_down(monkeypatch):
    torn = []

    async def teardown_run(context):  # records what it took down, as the real one does
        torn.append(context)
        context.metadata.setdefault(TORN_DOWN_KEY, []).extend(s.sandbox_id for s in context.deployed_sandboxes)
        return TeardownReport(terminated=tuple(
            RecordedSandbox(s.sandbox_id, s.sandbox_type, "sandbox") for s in context.deployed_sandboxes))

    monkeypatch.setattr(task_run, "teardown_run", teardown_run)
    monkeypatch.setattr(local_runner, "teardown_run", teardown_run)
    monkeypatch.setattr(submitted_runs, "POLL_SECONDS", 0.02)
    return torn


def _ids(contexts) -> list[str]:
    return sorted(s.sandbox_id for context in contexts for s in context.deployed_sandboxes)


def _invoke(monkeypatch, tmp_path, task, *args, signal_after=None):
    """Invoke the CLI, sending Ctrl-C ``signal_after`` seconds in; a Ctrl-C not yet sent when it returns never is."""
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: task))
    timer = threading.Timer(signal_after, lambda: os.kill(os.getpid(), signal.SIGINT)) if signal_after else None
    if timer:
        timer.start()
    logging.disable(logging.CRITICAL)
    try:
        return CliRunner().invoke(cli, ["task", *args, "--id", "t", "--output-dir", str(tmp_path)])
    finally:
        if timer:
            timer.cancel()
        logging.disable(logging.NOTSET)


def test_a_run_that_passes_is_torn_down(monkeypatch, tmp_path, torn_down):
    task = _Deploys()

    result = _invoke(monkeypatch, tmp_path, task, "run")

    assert result.exit_code == 0, result.output
    assert _ids(torn_down) == ["sb-1"]
    [record] = run_store.list_runs(task_id="t")
    assert record.status is RunStatus.COMPLETED and task.contexts[0].metadata["workflow_id"] == record.run_id


def test_a_run_that_fails_is_torn_down_and_fails_the_command(monkeypatch, tmp_path, torn_down):
    task = _Deploys(ending=RuntimeError("boom"))

    result = _invoke(monkeypatch, tmp_path, task, "run")

    assert result.exit_code == 1, result.output
    assert "FAILED: RuntimeError: boom" in result.output
    assert _ids(torn_down) == ["sb-1"]


def test_each_parallel_run_is_torn_down(monkeypatch, tmp_path, torn_down):
    task = _Deploys()

    result = _invoke(monkeypatch, tmp_path, task, "run", "--k", "2")

    assert result.exit_code == 0, result.output
    assert _ids(torn_down) == ["sb-1", "sb-2"]


def test_keep_holds_a_run_up_until_ctrl_c_then_tears_it_down(monkeypatch, tmp_path, torn_down):
    task = _Deploys()

    result = _invoke(monkeypatch, tmp_path, task, "run", "--keep", signal_after=1.0)

    assert result.exit_code == 0, result.output
    assert "Kept up:" in result.output and "sb-1" in result.output
    assert "Holding 1 sandbox up; Ctrl-C tears it down." in result.output
    assert _ids(torn_down) == ["sb-1"]


def test_keep_still_tears_down_a_run_ctrl_c_cancels(monkeypatch, tmp_path, torn_down):
    task = _Deploys(nap=30)

    result = _invoke(monkeypatch, tmp_path, task, "run", "--keep", signal_after=1.0)

    assert result.exit_code == 130, result.output
    assert _ids(torn_down) == ["sb-1"]
    assert "Holding" not in result.output


def test_ctrl_c_mid_run_tears_down_every_run_and_exits_130(monkeypatch, tmp_path, torn_down):
    task = _Deploys(nap=30)

    result = _invoke(monkeypatch, tmp_path, task, "run", "--k", "2", signal_after=1.0)

    assert result.exit_code == 130, result.output
    assert _ids(torn_down) == ["sb-1", "sb-2"]
    assert "Cancelling the runs (Ctrl-C again to stop waiting for them)" in result.output
    assert result.output.count("Run CANCELED") == 2
    assert {r.status for r in run_store.list_runs(task_id="t")} == {RunStatus.CANCELED}


def test_each_seed_of_a_batch_is_torn_down(monkeypatch, tmp_path, torn_down):
    seeds = tmp_path / "seeds.csv"
    seeds.write_text("name\nfirst\nsecond\n")
    task = _Deploys()

    result = _invoke(monkeypatch, tmp_path, task, "run-batch", "--seeds", str(seeds))

    assert result.exit_code == 0, result.output
    assert len(torn_down) == 2


class _OneQuickOneSlow(_Deploys):
    """Its first run ends at once and every later one naps, or the other way round when ``quick_last``."""

    def __init__(self, quick_last=False):
        super().__init__()
        self.quick_last = quick_last

    async def run(self, context, **kwargs):
        self.nap = 0 if bool(self.contexts) == self.quick_last else 30
        return await super().run(context, **kwargs)


def test_ctrl_c_with_keep_also_tears_down_a_run_that_already_finished(monkeypatch, tmp_path, torn_down):
    task = _OneQuickOneSlow()

    result = _invoke(monkeypatch, tmp_path, task, "run", "--k", "2", "--keep", signal_after=1.0)

    assert result.exit_code == 130, result.output
    assert _ids(torn_down) == ["sb-1", "sb-2"]
    assert "Holding" not in result.output


class _CancelledElsewhere(_Deploys):
    """Its first run is asked to cancel through another runner while it runs, then ends; the second ends only once
    the first has."""

    def __init__(self, other):
        super().__init__()
        self.other, self.first_ended, self.refused = other, asyncio.Event(), []

    async def run(self, context, **kwargs):
        first = not self.contexts
        await super().run(context, **kwargs)
        if first:
            self.refused.append(await self.other.cancel(context.metadata["workflow_id"]))
            self.first_ended.set()
        else:
            await self.first_ended.wait()
        return context


def test_a_cancel_another_runner_cant_carry_out_leaves_a_kept_run_kept_and_torn_down(monkeypatch, tmp_path, torn_down):
    def sleep(_):
        signal.raise_signal(signal.SIGINT)

    monkeypatch.setattr(task_run, "time", types.SimpleNamespace(sleep=sleep))
    task = _CancelledElsewhere(local_runner.LocalRunner())

    result = _invoke(monkeypatch, tmp_path, task, "run", "--k", "2", "--keep")

    assert result.exit_code == 0, result.output
    assert task.refused == [False]
    assert {r.status for r in run_store.list_runs(task_id="t")} == {RunStatus.COMPLETED}
    assert "Holding 2 sandboxes up" in result.output and _ids(torn_down) == ["sb-1", "sb-2"]


@pytest.mark.parametrize("quick_last", [False, True], ids=["ended-first", "ended-last"])
def test_an_error_following_kept_runs_still_tears_down_one_that_ended(monkeypatch, tmp_path, torn_down, quick_last):
    real_status = local_runner.LocalRunner.status

    async def status(self, run_id):
        record = await real_status(self, run_id)
        if record.status is not RunStatus.COMPLETED and any(
                run.status is RunStatus.COMPLETED for run in run_store.list_runs()):
            raise RuntimeError("run index unreachable")
        return record

    monkeypatch.setattr(local_runner.LocalRunner, "status", status)
    task = _OneQuickOneSlow(quick_last)

    result = _invoke(monkeypatch, tmp_path, task, "run", "--k", "2", "--keep")

    assert isinstance(result.exception, RuntimeError), result.output
    assert _ids(torn_down) == ["sb-1", "sb-2"]


def test_keep_writes_the_context_of_a_run_that_failed(monkeypatch, tmp_path, torn_down):
    task = _Deploys(ending=RuntimeError("boom"))

    result = _invoke(monkeypatch, tmp_path, task, "run", "--keep", signal_after=1.0)

    assert result.exit_code == 1, result.output
    assert list(tmp_path.glob("t_*.json")), "no context file to resume the kept run from"


def test_a_kept_failed_run_reports_its_own_error_when_its_context_cant_be_written(monkeypatch, tmp_path, torn_down):
    def disk_full(*_, **__):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(task_run, "_write_context", disk_full)
    task = _Deploys(ending=RuntimeError("boom"))

    result = _invoke(monkeypatch, tmp_path, task, "run", "--keep", signal_after=1.0)

    assert result.exit_code == 1 and "FAILED: RuntimeError: boom" in result.output
    assert "Couldn't write the run's context: [Errno 28] No space left on device" in result.output
    assert _ids(torn_down) == ["sb-1"]
