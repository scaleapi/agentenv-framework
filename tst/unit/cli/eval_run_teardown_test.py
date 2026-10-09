"""``agent-env eval run`` tears down each run's sandboxes, however the run ends, and a first Ctrl-C lets a
teardown already under way finish."""

import asyncio
import importlib
import logging
import signal
import types

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.eval import Eval
from agent_env.task import Task, teardown
from agent_env.task.teardown import TeardownReport
from agent_env.task_step.context import DeployedSandbox

eval_run = importlib.import_module("agent_env.cli.eval.run")  # the package's ``run`` is the click command
pytestmark = pytest.mark.usefixtures("sigint_handled")


class _Raising:
    id = "t"

    def __init__(self, ending):
        self.ending, self.context = ending, None

    async def run(self, context, **_):
        self.context = context
        raise self.ending


@pytest.mark.parametrize("ending", [RuntimeError("boom"), asyncio.CancelledError()], ids=["raised", "cancelled"])
def test_a_run_that_raises_or_is_cancelled_is_still_torn_down(monkeypatch, ending):
    torn_down = []

    async def teardown_run(context):
        torn_down.append(context)
        return TeardownReport()

    monkeypatch.setattr(eval_run, "teardown_run", teardown_run)
    task = _Raising(ending)

    with pytest.raises(type(ending)):
        asyncio.run(eval_run._run_single(task, "[t]", None))

    assert torn_down == [task.context]


class _Deploys:
    """Records a sandbox on the fake backend, then naps."""

    def __init__(self, id, nap):
        self.id, self.version, self.steps, self.nap = id, 1, [], nap

    async def run(self, context, **_):
        context.deployed_sandboxes.append(DeployedSandbox(
            sandbox_name="box", sandbox_id=f"sb-{self.id}", sandbox_mode="vm", sandbox_type="fake"))
        await asyncio.sleep(self.nap)
        return context


def test_ctrl_c_cancels_the_running_runs_lets_a_finished_ones_teardown_finish_and_exits_130(monkeypatch):
    tasks = {"done": _Deploys("done", 0), "slow": _Deploys("slow", 30)}
    listed = types.SimpleNamespace(id="e", version=1,
                                   tasks=[types.SimpleNamespace(task_id=name, task_version=1) for name in tasks])
    monkeypatch.setattr(Eval, "get", classmethod(lambda cls, id, version=None: listed))
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: tasks[id]))
    terminated = []

    class _Sandbox:
        ON_THIS_MACHINE = False

        def __init__(self, sandbox_id):
            self.sandbox_id = sandbox_id

        async def terminate(self):
            if self.sandbox_id == "sb-done":
                signal.raise_signal(signal.SIGINT)
                await asyncio.sleep(0.2)
            terminated.append(self.sandbox_id)

    class _Provider:
        async def get_sandbox(self, sandbox_id):
            return _Sandbox(sandbox_id)

    monkeypatch.setattr(teardown, "build_sandbox_provider", lambda name: _Provider())
    logging.disable(logging.CRITICAL)
    try:
        result = CliRunner().invoke(cli, ["eval", "run", "--id", "e"])
    finally:
        logging.disable(logging.NOTSET)

    assert result.exit_code == 130, result.output
    assert sorted(terminated) == ["sb-done", "sb-slow"]
    assert "Cancelling the runs and tearing them down (Ctrl-C again to stop now)" in result.output
    assert "  [done] PASSED" in result.output
    assert "  [slow] CANCELLED" in result.output
