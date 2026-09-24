"""Integration tests for Eval persistence and CLI.

These tests use real MongoDB (Atlas dev database).
Requires AWS credentials with access to secrets manager.
"""

import json
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.eval import Eval, EvalTask
from agent_env.store.base import NotFoundError
from agent_env.task import Task

REPO_ROOT = Path(__file__).resolve().parents[3]


def _unique_id(prefix: str = "test") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run agent-env CLI via subprocess (avoids CliRunner stream issues)."""
    return subprocess.run(
        ["agent-env", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**os.environ, "AGENT_ENV_ENVIRONMENT": "dev"},
    )


# ---------------------------------------------------------------------------
# Model persistence tests (hit real MongoDB)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEvalPersistence:
    """Tests for Eval put/get/query."""

    def test_put_creates_eval(self):
        eval_id = _unique_id("eval")
        tasks = [
            EvalTask(task_id="task-a", task_version=1),
            EvalTask(task_id="task-b", task_version=2),
        ]
        eval_obj = Eval.put(id=eval_id, tasks=tasks)

        assert eval_obj.id == eval_id
        assert eval_obj.version == 1
        assert len(eval_obj.tasks) == 2
        assert eval_obj.tasks[0].task_id == "task-a"
        assert eval_obj.tasks[1].task_version == 2

    def test_get_retrieves_eval(self):
        eval_id = _unique_id("eval")
        created = Eval.put(id=eval_id, tasks=[EvalTask(task_id="t1", task_version=1)])
        retrieved = Eval.get(eval_id)

        assert retrieved.id == created.id
        assert retrieved.version == created.version
        assert len(retrieved.tasks) == 1
        assert retrieved.tasks[0].task_id == "t1"

    def test_get_specific_version(self):
        eval_id = _unique_id("eval")
        Eval.put(id=eval_id, tasks=[EvalTask(task_id="t1", task_version=1)])
        Eval.put(id=eval_id, tasks=[EvalTask(task_id="t1", task_version=1), EvalTask(task_id="t2", task_version=1)])

        v1 = Eval.get(eval_id, version=1)
        v2 = Eval.get(eval_id, version=2)
        latest = Eval.get(eval_id)

        assert len(v1.tasks) == 1
        assert len(v2.tasks) == 2
        assert latest.version == 2

    def test_get_nonexistent_raises(self):
        with pytest.raises(NotFoundError):
            Eval.get("nonexistent-eval-id-12345")

    def test_versioning_increments(self):
        eval_id = _unique_id("eval")
        e1 = Eval.put(id=eval_id, tasks=[])
        e2 = Eval.put(id=eval_id, tasks=[])
        e3 = Eval.put(id=eval_id, tasks=[])

        assert e1.version == 1
        assert e2.version == 2
        assert e3.version == 3

    def test_query_by_id(self):
        eval_id = _unique_id("eval")
        Eval.put(id=eval_id, tasks=[])
        Eval.put(id=eval_id, tasks=[])

        results = Eval.query().id(eval_id).execute()
        assert len(results) == 2

    def test_query_latest(self):
        eval_id = _unique_id("eval")
        Eval.put(id=eval_id, tasks=[EvalTask(task_id="old", task_version=1)])
        Eval.put(id=eval_id, tasks=[EvalTask(task_id="new", task_version=2)])

        latest = Eval.query().id(eval_id).latest().first()
        assert latest is not None
        assert latest.tasks[0].task_id == "new"

    def test_to_dict_from_dict_roundtrip(self):
        eval_obj = Eval(
            id="roundtrip-test",
            version=5,
            tasks=[EvalTask(task_id="t1", task_version=3), EvalTask(task_id="t2", task_version=7)],
        )
        d = eval_obj.to_dict()
        restored = Eval.from_dict(d)

        assert restored.id == eval_obj.id
        assert restored.version == eval_obj.version
        assert len(restored.tasks) == 2
        assert restored.tasks[0].task_id == "t1"
        assert restored.tasks[1].task_version == 7


# ---------------------------------------------------------------------------
# CLI help / validation tests (no DB needed, safe with CliRunner)
# ---------------------------------------------------------------------------

class TestEvalCliHelp:

    def test_eval_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "--help"])
        assert result.exit_code == 0
        assert "create" in result.output
        assert "add-tasks" in result.output
        assert "run" in result.output

    def test_eval_create_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "create", "--help"])
        assert result.exit_code == 0
        assert "--id" in result.output
        assert "FILEPATH" in result.output

    def test_eval_add_tasks_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "add-tasks", "--help"])
        assert result.exit_code == 0
        assert "--id" in result.output
        assert "--version" in result.output

    def test_eval_run_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "run", "--help"])
        assert result.exit_code == 0
        assert "--id" in result.output
        assert "--version" in result.output
        assert "--output-dir" in result.output
        assert "--k" in result.output


class TestEvalCliValidation:

    def test_create_missing_id(self):
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            f.flush()
            result = runner.invoke(cli, ["eval", "create", f.name])
        assert result.exit_code != 0

    def test_create_missing_filepath(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "create", "--id", "test"])
        assert result.exit_code != 0

    def test_run_missing_id(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "run"])
        assert result.exit_code != 0

    def test_run_invalid_k(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["eval", "run", "--id", "x", "--k", "0"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI integration tests (subprocess, matching cli_test.py pattern)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestEvalCliIntegration:
    """Integration tests for eval create / add-tasks / run via subprocess.

    Creates tasks in the configured document store (in-process), then exercises CLI
    commands via subprocess to avoid CliRunner stream issues with the store clients.
    """

    def test_eval_create(self):
        task1 = Task.put(id=_unique_id("eval-cli-task"), steps=[])
        task2 = Task.put(id=_unique_id("eval-cli-task"), steps=[])
        eval_id = _unique_id("eval-cli")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"task_id": task1.id, "task_version": task1.version},
                {"task_id": task2.id, "task_version": task2.version},
            ], f)
            filepath = f.name

        result = _run_cli("eval", "create", filepath, "--id", eval_id)
        os.unlink(filepath)

        assert result.returncode == 0, f"CLI failed:\n{result.stdout}\n{result.stderr}"
        assert f"Created eval: id={eval_id} version=1" in result.stdout

        eval_obj = Eval.get(eval_id)
        assert eval_obj.version == 1
        assert len(eval_obj.tasks) == 2
        assert eval_obj.tasks[0].task_id == task1.id
        assert eval_obj.tasks[1].task_id == task2.id

    def test_eval_create_validates_tasks_exist(self):
        eval_id = _unique_id("eval-cli")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"task_id": "nonexistent-task-xyz", "task_version": 1}], f)
            filepath = f.name

        result = _run_cli("eval", "create", filepath, "--id", eval_id)
        os.unlink(filepath)

        assert result.returncode != 0

    def test_eval_add_tasks(self):
        task1 = Task.put(id=_unique_id("eval-cli-task"), steps=[])
        task2 = Task.put(id=_unique_id("eval-cli-task"), steps=[])
        task3 = Task.put(id=_unique_id("eval-cli-task"), steps=[])
        eval_id = _unique_id("eval-cli")

        # Create eval with task1
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"task_id": task1.id, "task_version": task1.version}], f)
            filepath = f.name
        result = _run_cli("eval", "create", filepath, "--id", eval_id)
        os.unlink(filepath)
        assert result.returncode == 0, f"Create failed:\n{result.stdout}\n{result.stderr}"

        # Add task2 and task3
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"task_id": task2.id, "task_version": task2.version},
                {"task_id": task3.id, "task_version": task3.version},
            ], f)
            filepath = f.name
        result = _run_cli("eval", "add-tasks", filepath, "--id", eval_id)
        os.unlink(filepath)
        assert result.returncode == 0, f"Add-tasks failed:\n{result.stdout}\n{result.stderr}"
        assert "version=2" in result.stdout

        eval_obj = Eval.get(eval_id)
        assert eval_obj.version == 2
        assert len(eval_obj.tasks) == 3

        v1 = Eval.get(eval_id, version=1)
        assert len(v1.tasks) == 1

    def test_eval_add_tasks_deduplicates(self):
        task1 = Task.put(id=_unique_id("eval-cli-task"), steps=[])
        task2 = Task.put(id=_unique_id("eval-cli-task"), steps=[])
        eval_id = _unique_id("eval-cli")

        # Create eval with both tasks
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([
                {"task_id": task1.id, "task_version": task1.version},
                {"task_id": task2.id, "task_version": task2.version},
            ], f)
            filepath = f.name
        _run_cli("eval", "create", filepath, "--id", eval_id)
        os.unlink(filepath)

        # Re-add task1 -- should replace, not duplicate
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"task_id": task1.id, "task_version": task1.version}], f)
            filepath = f.name
        result = _run_cli("eval", "add-tasks", filepath, "--id", eval_id)
        os.unlink(filepath)
        assert result.returncode == 0, f"Add-tasks failed:\n{result.stdout}\n{result.stderr}"

        eval_obj = Eval.get(eval_id)
        assert len(eval_obj.tasks) == 2

    @pytest.mark.int_test_slow
    def test_eval_run_nonexistent_eval(self):
        result = _run_cli("eval", "run", "--id", "does-not-exist-xyz")
        assert result.returncode != 0
