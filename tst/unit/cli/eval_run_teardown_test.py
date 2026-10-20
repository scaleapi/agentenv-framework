"""``agent-env eval run``'s runs, through the configured runner (here ``LocalRunner``): a first Ctrl-C cancels the
running ones, which are torn down, and lets a teardown already under way finish."""

import asyncio
import logging
import signal
import types

import pytest
from click.testing import CliRunner

from agent_env.cli import cli, submitted_runs
from agent_env.eval import Eval
from agent_env.task import Task, teardown
from agent_env.task_step.context import DeployedSandbox
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

pytestmark = pytest.mark.usefixtures("sigint_handled", "local_stores")


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

    class _Provider(SandboxProvider):
        async def get_sandbox(self, sandbox_id):
            return _Sandbox(sandbox_id)

    monkeypatch.setattr(teardown, "build_sandbox_provider", lambda name: _Provider())
    monkeypatch.setattr(submitted_runs, "POLL_SECONDS", 0.02)
    logging.disable(logging.CRITICAL)
    try:
        result = CliRunner().invoke(cli, ["eval", "run", "--id", "e"])
    finally:
        logging.disable(logging.NOTSET)

    assert result.exit_code == 130, result.output
    assert sorted(terminated) == ["sb-done", "sb-slow"]
    assert "Cancelling the runs (Ctrl-C again to stop waiting for them)" in result.output
    assert "  [done] PASSED" in result.output
    assert "  [slow] CANCELED" in result.output
