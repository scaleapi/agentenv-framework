"""``task run``, ``task run-batch`` and ``eval run`` give each run the user simulator's and the judges' models as
overrides, leaving the task as stored."""

import importlib
import logging
import types

import pytest
from click.testing import CliRunner

from agent_env.cli import cli, submitted_runs
from agent_env.eval import Eval
from agent_env.runner import local_runner
from agent_env.task import Task
from agent_env.task.teardown import TeardownReport

task_run = importlib.import_module("agent_env.cli.task.run")  # the package's ``run`` is the click command
pytestmark = pytest.mark.usefixtures("sigint_handled", "local_stores")


class _Recorded:
    """Records each context it runs with."""

    id, version, steps = "t", 1, []

    def __init__(self):
        self.contexts = []

    async def run(self, context, **_):
        self.contexts.append(context)
        return context


@pytest.fixture
def task(monkeypatch):
    async def teardown_run(context):
        return TeardownReport()

    monkeypatch.setattr(task_run, "teardown_run", teardown_run)
    monkeypatch.setattr(local_runner, "teardown_run", teardown_run)
    monkeypatch.setattr(submitted_runs, "POLL_SECONDS", 0.02)
    recorded = _Recorded()
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: recorded))
    monkeypatch.setattr(Eval, "get", classmethod(lambda cls, id, version=None: types.SimpleNamespace(
        id="e", version=1, tasks=[types.SimpleNamespace(task_id="t", task_version=1)])))
    logging.disable(logging.CRITICAL)
    yield recorded
    logging.disable(logging.NOTSET)


@pytest.mark.parametrize("command", ["task run", "task run-batch", "eval run"])
def test_the_user_and_judge_models_reach_each_run_as_overrides(task, tmp_path, command):
    seeds = tmp_path / "seeds.csv"
    seeds.write_text("name\nfirst\n")
    args = {"task run": ["--id", "t"], "task run-batch": ["--id", "t", "--seeds", str(seeds)],
            "eval run": ["--id", "e"]}[command]

    result = CliRunner().invoke(cli, [*command.split(), *args, "--user-model", "u", "--judge-model", "j"])

    assert result.exit_code == 0, result.output
    assert [{key: context.metadata["user_overrides"][key] for key in ("user_model", "judge_model")}
            for context in task.contexts] == [{"user_model": "u", "judge_model": "j"}]
