"""``agent-env task run`` and ``task run-batch`` tear down what each run deployed, however it ends; ``--keep`` holds
it up until Ctrl-C, and Ctrl-C mid-run tears it down and exits 130."""

import asyncio
import importlib
import logging
import os
import signal
import threading

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.task import Task
from agent_env.task.teardown import TORN_DOWN_KEY, RecordedSandbox, TeardownReport
from agent_env.task_step.context import DeployedSandbox

task_run = importlib.import_module("agent_env.cli.task.run")  # the package's ``run`` is the click command
pytestmark = pytest.mark.usefixtures("sigint_handled")


class _Deploys:
    """Records a sandbox on a fake backend, then naps, then ends as told."""

    id, version, steps = "t", 1, []

    def __init__(self, nap=0.0, ending=None):
        self.nap, self.ending, self.contexts = nap, ending, []

    async def run(self, context, **_):
        self.contexts.append(context)
        context.deployed_sandboxes.append(DeployedSandbox(
            sandbox_name="box", sandbox_id=f"sb-{len(self.contexts)}", sandbox_mode="vm", sandbox_type="fake"))
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
    return torn


def _invoke(monkeypatch, tmp_path, task, *args, signal_after=None):
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: task))
    if signal_after is not None:
        threading.Timer(signal_after, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    logging.disable(logging.CRITICAL)
    try:
        return CliRunner().invoke(cli, ["task", *args, "--id", "t", "--output-dir", str(tmp_path)])
    finally:
        logging.disable(logging.NOTSET)


def test_a_run_that_passes_is_torn_down(monkeypatch, tmp_path, torn_down):
    task = _Deploys()

    result = _invoke(monkeypatch, tmp_path, task, "run")

    assert result.exit_code == 0, result.output
    assert torn_down == task.contexts
    assert "Tore down 1 sandbox" in result.output


def test_a_run_that_fails_is_torn_down_and_reported_as_before(monkeypatch, tmp_path, torn_down):
    task = _Deploys(ending=RuntimeError("boom"))

    result = _invoke(monkeypatch, tmp_path, task, "run")

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert torn_down == task.contexts


def test_each_parallel_run_is_torn_down(monkeypatch, tmp_path, torn_down):
    task = _Deploys()

    result = _invoke(monkeypatch, tmp_path, task, "run", "--k", "2")

    assert result.exit_code == 0, result.output
    assert len(torn_down) == 2 and {id(c) for c in torn_down} == {id(c) for c in task.contexts}


def test_keep_holds_a_run_up_until_ctrl_c_then_tears_it_down(monkeypatch, tmp_path, torn_down):
    task = _Deploys()

    result = _invoke(monkeypatch, tmp_path, task, "run", "--keep", signal_after=1.0)

    assert result.exit_code == 0, result.output
    assert "Kept up:" in result.output and "sb-1" in result.output
    assert "Holding 1 sandbox up; Ctrl-C tears it down." in result.output
    assert torn_down == task.contexts


def test_keep_still_tears_down_a_run_ctrl_c_cancels(monkeypatch, tmp_path, torn_down):
    task = _Deploys(nap=30)

    result = _invoke(monkeypatch, tmp_path, task, "run", "--keep", signal_after=1.0)

    assert result.exit_code == 130, result.output
    assert torn_down == task.contexts
    assert "Holding" not in result.output


def test_ctrl_c_mid_run_tears_down_every_run_and_exits_130(monkeypatch, tmp_path, torn_down):
    task = _Deploys(nap=30)

    result = _invoke(monkeypatch, tmp_path, task, "run", "--k", "2", signal_after=1.0)

    assert result.exit_code == 130, result.output
    assert len(torn_down) == 2
    assert "Cancelling and tearing down (Ctrl-C again to stop now)" in result.output
    assert result.output.count("CANCELLED") == 2


def test_each_seed_of_a_batch_is_torn_down(monkeypatch, tmp_path, torn_down):
    seeds = tmp_path / "seeds.csv"
    seeds.write_text("name\nfirst\nsecond\n")
    task = _Deploys()

    result = _invoke(monkeypatch, tmp_path, task, "run-batch", "--seeds", str(seeds))

    assert result.exit_code == 0, result.output
    assert len(torn_down) == 2


class _OneQuickOneSlow(_Deploys):
    """Its first run ends at once; every later one naps."""

    async def run(self, context, **kwargs):
        self.nap = 0 if not self.contexts else 30
        return await super().run(context, **kwargs)


def test_ctrl_c_with_keep_also_tears_down_a_run_that_already_finished(monkeypatch, tmp_path, torn_down):
    task = _OneQuickOneSlow()

    result = _invoke(monkeypatch, tmp_path, task, "run", "--k", "2", "--keep", signal_after=1.0)

    assert result.exit_code == 130, result.output
    assert {id(c) for c in torn_down} == {id(c) for c in task.contexts}
    assert "Holding" not in result.output


def test_keep_writes_the_context_of_a_run_that_failed(monkeypatch, tmp_path, torn_down):
    task = _Deploys(ending=RuntimeError("boom"))

    result = _invoke(monkeypatch, tmp_path, task, "run", "--keep", signal_after=1.0)

    assert isinstance(result.exception, RuntimeError)
    assert list(tmp_path.glob("t_*.json")), "no context file to resume the kept run from"
