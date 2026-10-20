"""``task run`` on stored tasks, through ``LocalRunner`` and the real stores: an ``@local`` task's run stays in the
``@local`` namespace, seeded steps report under the ids they ran as, and a context that can't be exported fails
the command."""

import json
import logging

import pytest
from click.testing import CliRunner

from agent_env.cli import cli, submitted_runs
from agent_env.config import get_config
from agent_env.config.runtime import Config
from agent_env.runner import store as run_store
from agent_env.runner.store import RUNS_COLLECTION
from agent_env.store.document_store import Filter
from agent_env.task import Task
from agent_env.task.store import TASK_INSTANCES_COLLECTION, TASK_STEP_JOURNAL_COLLECTION, TaskInstanceStore
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_step import TaskStep

pytestmark = pytest.mark.usefixtures("sigint_handled", "local_stores", "cli_routing")

RUN_COLLECTIONS = (TASK_INSTANCES_COLLECTION, TASK_STEP_JOURNAL_COLLECTION, RUNS_COLLECTION)


class _Writes(TaskStep):
    """Makes the check ``collect_artifacts`` makes on an id it derives from the run's instance, records a score under
    its ``verifier_id``, and fails when told to."""

    type = "writes_stored_run_test"
    entity_refs = ()

    def __init__(self, verifier_id=None, fail=False, **base):
        super().__init__(**base)
        self.verifier_id, self.fail = verifier_id, fail

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), verifier_id=data.get("verifier_id"), fail=data.get("fail", False))

    def to_dict(self):
        return {**super().to_dict(), "verifier_id": self.verifier_id, "fail": self.fail}

    async def execute(self, context):
        get_config().check_local_run_write(context.instance_id)
        if self.verifier_id:
            context.metadata.setdefault("verifications", {})[self.verifier_id] = {"score": 1.0}
        if self.fail:
            raise RuntimeError("boom")
        return context


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), _Writes.type: _Writes}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    monkeypatch.setattr(submitted_runs, "POLL_SECONDS", 0.02)


def _put(task_id, *steps):
    return Task.put(id=task_id, steps=list(steps))


def _invoke(*args):
    logging.disable(logging.CRITICAL)
    try:
        return CliRunner().invoke(cli, ["task", "run", *args])
    finally:
        logging.disable(logging.NOTSET)


def _context_file(tmp_path, **metadata):
    path = tmp_path / "context.json"
    path.write_text(json.dumps(TaskStepContext(metadata=metadata).to_dict()))
    return str(path)


def _configured_run_records():
    configured = get_config().get_document_store().configured
    return {collection: configured.query(collection, Filter()) for collection in RUN_COLLECTIONS if
            configured.query(collection, Filter())}


@pytest.mark.parametrize("continued", [False, True], ids=["fresh", "context-json"])
def test_an_at_local_tasks_run_stays_in_the_local_namespace(tmp_path, continued):
    _put("@local/demo/t", _Writes(id="first", version=None), _Writes(id="second", version=None))
    resume = ["--context-json", _context_file(tmp_path, from_file="yes"), "--start-step", "1"] if continued else []

    result = _invoke("--id", "@local/demo/t", "--output-dir", str(tmp_path / "out"), *resume)

    assert result.exit_code == 0, result.output
    assert _configured_run_records() == {}
    [instance] = get_config().local_namespace_document_store().query(TASK_INSTANCES_COLLECTION, Filter())
    assert instance["instance_id"].startswith("@local/demo/t-") and instance["status"] == "completed"
    assert instance["context"]["metadata"].get("from_file") == ("yes" if continued else None)


@pytest.mark.parametrize("k", ["1", "2"])
def test_a_seeded_step_reports_under_the_id_it_ran_as(tmp_path, k):
    _put("seeded", _Writes(id="verify-<week>", version=None, verifier_id="grade-<week>"))

    result = _invoke("--id", "seeded", "--k", k, "--output-dir", str(tmp_path / "out"),
                     "--context-json", _context_file(tmp_path, seed={"week": "3"}))

    assert result.exit_code == 0, result.output
    assert "Completed step [1/1]: writes_stored_run_test (id=verify-3)" in result.output
    assert ("grade-3" if k == "1" else "score (verifier_id=grade-3): 1.0") in result.output


@pytest.mark.parametrize("fail", [False, True], ids=["passed", "failed"])
def test_a_context_that_cant_be_exported_fails_the_command(tmp_path, fail):
    _put("exported", _Writes(id="only", version=None, fail=fail))
    (tmp_path / "taken").write_text("")

    result = _invoke("--id", "exported", "--output-dir", str(tmp_path / "taken" / "out"))

    assert result.exit_code == 1, result.output
    assert "Couldn't write the run's context" in result.output and "Task completed!" not in result.output
    assert ("Run FAILED: RuntimeError: boom" in result.output) == fail


def test_a_context_json_that_cant_be_stored_fails_before_anything_is_submitted(tmp_path, monkeypatch):
    def unwritable(*args, **kwargs):
        raise ConnectionError("store unreachable")

    _put("resumed", _Writes(id="only", version=None))
    monkeypatch.setattr(TaskInstanceStore, "seed_context", unwritable)

    result = _invoke("--id", "resumed", "--context-json", _context_file(tmp_path, a=1), "--output-dir", str(tmp_path))

    assert isinstance(result.exception, ConnectionError), result.output
    assert run_store.list_runs(task_id="resumed") == []
