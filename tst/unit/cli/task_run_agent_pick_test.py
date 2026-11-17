"""``task run --a2a-agent-id`` is a run-wide pick: the CLI leaves it run-wide, at the deploy steps' stored versions, for
each step to apply (tst/unit/task_step/run_wide_agent_pick_test.py); a step that plays the user keeps its own agent by
its class, as it does for a hub run's pick."""

import importlib
import logging

import pytest
from click.testing import CliRunner

from agent_env.cli import cli, submitted_runs
from agent_env.runner import local_runner
from agent_env.task import Task
from agent_env.task.teardown import TeardownReport

task_run = importlib.import_module("agent_env.cli.task.run")  # the package's ``run`` is the click command
pytestmark = pytest.mark.usefixtures("sigint_handled", "local_stores")

AGENT = {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "theirs", "a2a_agent_version": 4}
USER = {"id": "user", "type": "deploy_agent", "env_ids": [], "agent_name": "human_agent", "a2a_agent_id": "sim"}
ASK = {"id": "ask", "type": "prompt_agent", "prompt_id": "ask", "prompt": "hi", "max_conversation_turns": 3}


class _Recorded:
    """A stored task that records each context it runs with."""

    id, version, requires = "t", 1, None

    def __init__(self, steps):
        self.steps = Task.from_dict({"id": "t", "version": 1, "steps": steps}).steps
        self.contexts = []

    async def run(self, context, **_):
        self.contexts.append(context)
        return context


@pytest.fixture
def stored(monkeypatch):
    async def teardown_run(context):
        return TeardownReport()

    monkeypatch.setattr(task_run, "teardown_run", teardown_run)
    monkeypatch.setattr(local_runner, "teardown_run", teardown_run)
    monkeypatch.setattr(submitted_runs, "POLL_SECONDS", 0.02)

    def store(*steps):
        recorded = _Recorded(list(steps))
        monkeypatch.setattr(Task, "get", classmethod(lambda cls, id, version=None: recorded))
        return recorded

    logging.disable(logging.CRITICAL)
    yield store
    logging.disable(logging.NOTSET)


def test_the_pick_stays_run_wide_with_a_user_agent_deployed(stored):
    recorded = stored(AGENT, USER, ASK)
    result = CliRunner().invoke(cli, ["task", "run", "--id", "t", "--a2a-agent-id", "mine"])
    assert result.exit_code == 0, result.output

    [context] = recorded.contexts
    overrides = context.metadata["user_overrides"]
    assert overrides["a2a_agent_id"] == "mine" and "step_params" not in overrides
