"""The `agent-env task create` preflight gate.

`step_cls.put` writes a step document unconditionally, so the gate has to run before
any put — otherwise a rejected create leaves orphan step versions behind and every
retry adds more. These assert nothing is persisted on rejection.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from agent_env.cli.task.create import create
from agent_env.store.base import NotFoundError


def _task_file(tmp_path, artifact_id="missing-artifact"):
    path = tmp_path / "steps.json"
    path.write_text(json.dumps([{
        "id": "c1", "type": "run_code", "script_artifact_id": artifact_id,
        "agent_name": "default-agent",
    }]))
    return path


@pytest.fixture
def store(monkeypatch):
    """Stub the two writes and the artifact read; yields the mocks to assert on."""
    from agent_env.task import Task
    from agent_env.task_step.task_steps.run_code import RunCodeTaskStep

    put_step = MagicMock(side_effect=lambda **kw: RunCodeTaskStep(**{**kw, "version": 1}))
    put_task = MagicMock(return_value=Task(id="t", version=1, steps=[]))
    monkeypatch.setattr(RunCodeTaskStep, "put", classmethod(lambda cls, **kw: put_step(**kw)))
    monkeypatch.setattr(Task, "put", classmethod(lambda cls, **kw: put_task(**kw)))
    yield put_step, put_task


def _run(tmp_path, *extra):
    return CliRunner().invoke(create, [
        str(_task_file(tmp_path)), "--id", "t", "--project-id", "p", *extra,
    ])


def test_a_rejected_create_persists_nothing(tmp_path, store):
    put_step, put_task = store
    with patch("agent_env.artifact.Artifact.get", side_effect=NotFoundError("nope")):
        result = _run(tmp_path)

    assert result.exit_code == 1
    assert "does not exist" in result.output
    assert "Nothing was saved" in result.output
    put_step.assert_not_called()
    put_task.assert_not_called()


def test_skip_validation_saves_and_still_prints_the_problems(tmp_path, store):
    put_step, put_task = store
    with patch("agent_env.artifact.Artifact.get", side_effect=NotFoundError("nope")):
        result = _run(tmp_path, "--skip-validation")

    assert result.exit_code == 0
    assert "does not exist" in result.output
    assert "Saving anyway" in result.output
    put_step.assert_called_once()
    put_task.assert_called_once()
