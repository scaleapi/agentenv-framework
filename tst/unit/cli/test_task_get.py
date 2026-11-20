"""`agent-env task get` prints a task as `task create` takes it."""

import json

import pytest
from click.testing import CliRunner

from agent_env.cli.task.get import get
from agent_env.task import Task

STEPS = [{"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm"},
         {"id": "tests", "type": "verify_sandbox", "sandbox_name": "box", "verifier_id": "tests", "criteria": []}]


@pytest.mark.parametrize("headlines", [None, ["tests"]], ids=["bare", "headlines"])
def test_a_task_prints_as_its_steps_or_with_its_headline_verifiers(monkeypatch, headlines):
    task = Task.from_dict({"id": "t", "version": 1, "steps": STEPS, "headline_verifiers": headlines})
    monkeypatch.setattr(Task, "get", classmethod(lambda cls, task_id, version=None: task))

    printed = json.loads(CliRunner().invoke(get, ["--id", "t"]).output)

    steps = [step.to_dict() for step in task.steps]
    assert printed == ({"headline_verifiers": headlines, "steps": steps} if headlines else steps)
