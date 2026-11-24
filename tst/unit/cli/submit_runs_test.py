"""``task run`` and ``eval run`` hand their runs to the configured ``[runner]`` and follow them from the run index:
here ``LocalRunner``, which runs them in this process and tears them down itself."""

import asyncio
import importlib
import json
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
from agent_env.task.store import get_task_instance_store, register_task_instance
from agent_env.task.teardown import TeardownReport
from agent_env.task_step.context import TaskStepContext

eval_run = importlib.import_module("agent_env.cli.eval.run")  # the package's ``run`` is the click command

pytestmark = pytest.mark.usefixtures("sigint_handled", "local_stores")


class _Task:
    id, version, steps, requires, headline_verifiers = "t", 1, [], None, None

    def __init__(self, nap=0.0, failed_steps=None):
        self.nap, self.failed_steps, self.contexts, self.calls = nap, failed_steps, [], []
        self.active = self.most_active = 0

    async def run(self, context, **kwargs):
        self.contexts.append(context)
        self.calls.append(kwargs)
        self.active += 1
        self.most_active = max(self.most_active, self.active)
        try:
            await asyncio.sleep(self.nap)
        finally:
            self.active -= 1
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


def test_a_failed_submitted_run_fails_the_command(monkeypatch, tmp_path, torn_down):
    task = _Task(failed_steps=[{"step_id": "s1"}])

    result = _invoke(monkeypatch, task, "task", "run", "--id", "t", "--k", "2",
                     "--output-dir", str(tmp_path))

    assert result.exit_code == 1, result.output
    assert "0/2 runs completed, 2/2 failed." in result.output
    assert {r.status for r in run_store.list_runs(task_id="t")} == {RunStatus.FAILED}


def test_a_context_json_is_stored_as_an_instance_the_runner_continues(monkeypatch, tmp_path, torn_down):
    task = _Task()
    saved = tmp_path / "context.json"
    saved.write_text(json.dumps(TaskStepContext(metadata={"seeded": "yes"}).to_dict()))

    result = _invoke(monkeypatch, task, "task", "run", "--id", "t", "--context-json", str(saved), "--start-step", "1",
                     "--gateway-env-id", "gw-run", "--website-browser-env-id", "wb-run", "--output-dir", str(tmp_path))

    assert result.exit_code == 0, result.output
    [call], [context] = task.calls, task.contexts
    assert call["start_step"] == 1 and context.metadata["seeded"] == "yes"
    assert context.metadata["user_overrides"]["gateway_env_id"] == "gw-run"
    assert context.metadata["user_overrides"]["website_browser_env_id"] == "wb-run"
    assert call["instance_id"] == run_store.list_runs(task_id="t")[0].instance_id


def test_an_eval_submits_every_task_k_times(monkeypatch, tmp_path, torn_down):
    task = _Task()
    monkeypatch.setattr(Eval, "get", classmethod(lambda cls, id, version=None: types.SimpleNamespace(
        id="e", version=1, tasks=[types.SimpleNamespace(task_id="t", task_version=1)])))

    monkeypatch.setattr(eval_run, "get_agent_env_client_id", lambda: "agent-env-cli/alex-dev")

    result = _invoke(monkeypatch, task, "eval", "run", "--id", "e", "--k", "2")

    assert result.exit_code == 0, result.output
    assert result.output.count("PASSED") == 2 and "All 2 runs completed!" in result.output
    assert len(run_store.list_runs(task_id="t")) == 2
    assert {c.metadata["agent_env_hub"]["caller"] for c in task.contexts} == {"agent-env-cli/alex-dev"}


def test_an_eval_keeps_at_most_max_concurrency_runs_in_flight(monkeypatch, torn_down):
    task = _Task(nap=0.1)
    monkeypatch.setattr(Eval, "get", classmethod(lambda cls, id, version=None: types.SimpleNamespace(
        id="e", version=1, tasks=[types.SimpleNamespace(task_id="t", task_version=1)])))

    result = _invoke(monkeypatch, task, "eval", "run", "--id", "e", "--k", "3", "--max-concurrency", "1")

    assert result.exit_code == 0, result.output
    assert (len(task.contexts), task.most_active) == (3, 1)


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

    result = _invoke(monkeypatch, _Task(), "task", "run", "--id", "t")

    assert isinstance(result.exception, RuntimeError)
    assert runner.stopped


def test_a_first_ctrl_c_lets_the_runner_finish_stopping(monkeypatch, local_stores):
    runner = _Runner(stop_seconds=2.0)
    local_stores.set_runner(runner)

    result = _invoke(monkeypatch, _Task(), "task", "run", "--id", "t", signal_after=0.5)

    assert result.exit_code == 130, result.output
    assert runner.stopped


def test_an_error_cancelling_after_ctrl_c_still_stops_the_runner(monkeypatch, local_stores):
    runner = _Runner(cancel_error=RuntimeError("runner unreachable"), run_status=RunStatus.RUNNING)
    local_stores.set_runner(runner)

    result = _invoke(monkeypatch, _Task(), "task", "run", "--id", "t", signal_after=0.5)

    assert isinstance(result.exception, RuntimeError)
    assert runner.stopped


class _Spends(_Task):
    """Leaves one solver entry in its run's ledger, as a running step's write does."""

    async def run(self, context, **kwargs):
        register_task_instance(self.id, self.version, 0, instance_id=kwargs["instance_id"])
        get_task_instance_store().record_running_usage_sync(kwargs["instance_id"], [{
            "attempt": 0, "step_id": "ask", "role": "solver", "model": "gpt-4.1-mini", "turns": 1, "input_tokens": 1000,
            "output_tokens": 10, "cost_usd": 0.0004, "cost_status": "priced", "cost_source": "litellm@test"}])
        return await super().run(context, **kwargs)


SPENT = "at least $0.0004 (solver gpt-4.1-mini at least $0.0004, 1.0k in, 10 out, partial)"


def test_an_eval_prints_each_runs_cost_and_their_total_from_the_ledgers(monkeypatch, tmp_path, torn_down):
    """Two versions of one task, whose runs share a tag, each count."""
    monkeypatch.setattr(Eval, "get", classmethod(lambda cls, id, version=None: types.SimpleNamespace(
        id="e", version=1, tasks=[types.SimpleNamespace(task_id="t", task_version=1),
                                  types.SimpleNamespace(task_id="t", task_version=2)])))

    result = _invoke(monkeypatch, _Spends(), "eval", "run", "--id", "e")

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines.count(f"    cost: {SPENT}") == 2
    assert ("Cost over 2 runs: at least $0.0008 (solver gpt-4.1-mini at least $0.0008, 2.0k in, 20 out, partial)"
            in lines)
    assert lines[-1] == "Costs priced from LiteLLM's price map, litellm@test."


def test_a_task_run_prints_its_cost_from_the_ledger(monkeypatch, tmp_path, torn_down):
    result = _invoke(monkeypatch, _Spends(), "task", "run", "--id", "t", "--output-dir", str(tmp_path))

    assert result.exit_code == 0, result.output
    assert f"cost: {SPENT}" in result.output.splitlines()
    assert result.output.splitlines()[-1] == "Costs priced from LiteLLM's price map, litellm@test."
