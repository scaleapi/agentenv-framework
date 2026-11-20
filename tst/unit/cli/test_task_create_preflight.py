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


def _object_file(tmp_path, **fields):
    path = tmp_path / "task.json"
    path.write_text(json.dumps({"steps": json.loads(_task_file(tmp_path).read_text()), **fields}))
    return path


def test_an_object_file_saves_its_headline_verifiers_with_the_task(tmp_path, store):
    _, put_task = store
    with patch("agent_env.artifact.Artifact.get", side_effect=NotFoundError("nope")):
        result = CliRunner().invoke(create, [str(_object_file(tmp_path, headline_verifiers=["c1"])), "--id", "t",
                                             "--skip-validation"])

    assert result.exit_code == 0, result.output
    assert put_task.call_args.kwargs["headline_verifiers"] == ["c1"]


@pytest.mark.parametrize(("fields", "message"), [
    ({"headline": ["c1"]}, "unknown key 'headline'; a task file's object takes steps and headline_verifiers"),
    ({"headline_verifiers": ["final"]}, "headline_verifiers names no step in this task: ['final']"),
], ids=["unknown-key", "unknown-headline"])
def test_an_object_file_that_is_refused_persists_nothing(tmp_path, store, fields, message):
    put_step, put_task = store

    result = CliRunner().invoke(create, [str(_object_file(tmp_path, **fields)), "--id", "t", "--skip-validation"])

    assert result.exit_code == 1 and message in result.output
    put_step.assert_not_called()
    put_task.assert_not_called()


def _run(tmp_path, *extra):
    return CliRunner().invoke(create, [
        str(_task_file(tmp_path)), "--id", "t", *extra,
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


def test_a_field_no_step_reads_is_refused_naming_the_one_it_meant_and_persists_nothing(tmp_path, store):
    put_step, put_task = store
    path = tmp_path / "steps.json"
    path.write_text(json.dumps([{"id": "c1", "type": "run_code", "script_artifact_id": "a", "agent_nme": "x"}]))

    result = CliRunner().invoke(create, [str(path), "--id", "t", "--skip-validation"])

    assert result.exit_code == 1
    assert "Error: step 'c1': run_code doesn't read 'agent_nme' (did you mean 'agent_name'?)" in result.output
    put_step.assert_not_called()
    put_task.assert_not_called()


def test_a_lint_error_is_a_preflight_problem_and_a_lint_warning_is_only_printed(tmp_path, store):
    put_step, put_task = store
    with patch("agent_env.artifact.Artifact.get", side_effect=NotFoundError("nope")):
        refused = CliRunner().invoke(create, [str(_object_file(tmp_path, headline_verifiers=["c1"])), "--id", "t"])

    assert refused.exit_code == 1
    assert "Warning: no step grades this task, so a run has no score" in refused.output
    assert "  - headline_verifiers names 'c1', a run_code step, which grades nothing" in refused.output
    put_step.assert_not_called()
    put_task.assert_not_called()
