"""``task run --a2a-agent-id`` swaps only the tested agent's deploy when the task deploys a user sim, so the user sim
keeps its own agent; a task without one keeps the run-wide pick."""

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

AGENT = {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "theirs"}
USER = {"id": "user", "type": "deploy_agent", "env_ids": [], "agent_name": "human_agent", "a2a_agent_id": "sim"}
ASK = {"id": "ask", "type": "prompt_agent", "prompt_id": "ask", "prompt": "hi", "max_conversation_turns": 3}
LINDA = {"id": "user", "type": "deploy_agent", "agent_name": "linda", "a2a_agent_id": "sim"}
WHO = {"id": "who", "type": "user_sim", "name": "linda", "model": "m"}
LOOP = {"id": "loop", "type": "always_on", "prompt": "Help Linda.", "user_sim": "linda"}


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


def _overrides(recorded):
    result = CliRunner().invoke(cli, ["task", "run", "--id", "t", "--a2a-agent-id", "mine"])
    assert result.exit_code == 0, result.output
    [context] = recorded.contexts
    return context.metadata["user_overrides"]


def test_with_a_user_sim_deployed_only_the_tested_agents_deploy_is_swapped(stored):
    overrides = _overrides(stored(AGENT, USER, ASK))

    assert "a2a_agent_id" not in overrides
    assert overrides["step_params"] == {"agent": {"a2a_agent_id": "mine"}}


def test_without_a_user_sim_the_pick_stays_run_wide(stored):
    overrides = _overrides(stored(AGENT, {**ASK, "max_conversation_turns": 1}))

    assert overrides["a2a_agent_id"] == "mine" and "step_params" not in overrides


def test_the_agent_a_user_sim_step_names_keeps_its_own():
    overrides = {"a2a_agent_id": "mine"}

    task_run._narrow_agent_pick([AGENT, LINDA, WHO, LOOP], overrides)

    assert overrides == {"step_params": {"agent": {"a2a_agent_id": "mine"}}}


def test_a_steps_own_pick_wins_over_the_moved_one():
    overrides = {"a2a_agent_id": "mine", "step_params": {"agent": {"a2a_agent_id": "pinned"}}}

    task_run._narrow_agent_pick([AGENT, LINDA, WHO, LOOP], overrides)

    assert overrides == {"step_params": {"agent": {"a2a_agent_id": "pinned"}}}


def test_a_user_sim_named_but_not_deployed_leaves_the_pick_run_wide():
    overrides = {"a2a_agent_id": "mine"}

    task_run._narrow_agent_pick([AGENT, WHO, LOOP], overrides)

    assert overrides == {"a2a_agent_id": "mine"}
