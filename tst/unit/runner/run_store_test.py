"""Run-store invariants for the in-process runner: a terminal run is never overwritten,
and set_running only advances a still-queued one."""

import pytest

from agent_env.config import configure, reset_config, set_document_store
from agent_env.runner import store as run_store
from agent_env.runner.runner import RunRecord, RunStatus
from agent_env.store.document_store.sqlite_document_store import LocalSqliteDocumentStore


@pytest.fixture
def docs(tmp_path):
    configure()
    store = LocalSqliteDocumentStore(str(tmp_path / "documents.db"))
    set_document_store(store)
    run_store.ensure_indexes()
    yield store
    reset_config()


def _record(run_id: str, status: RunStatus) -> RunRecord:
    return RunRecord(run_id=run_id, runner="local", task_id="t", task_version=1,
                     instance_id="i", status=status)


def test_mark_terminal_does_not_overwrite_a_terminal_run(docs):
    # A cancel that already terminated the run must win over a completion that races it.
    run_store.insert_run(_record("r", RunStatus.CANCELED))
    run_store.mark_terminal("r", RunStatus.COMPLETED)
    assert run_store.get_run("r").status == RunStatus.CANCELED


def test_mark_terminal_advances_a_running_run(docs):
    run_store.insert_run(_record("r", RunStatus.RUNNING))
    run_store.mark_terminal("r", RunStatus.COMPLETED)
    got = run_store.get_run("r")
    assert got.status == RunStatus.COMPLETED
    assert got.finished_at_utc is not None


def test_set_running_advances_only_a_queued_run(docs):
    run_store.insert_run(_record("q", RunStatus.QUEUED))
    run_store.set_running("q")
    assert run_store.get_run("q").status == RunStatus.RUNNING

    # a run canceled while it waited for a worker slot must not be revived by set_running
    run_store.insert_run(_record("c", RunStatus.CANCELED))
    run_store.set_running("c")
    assert run_store.get_run("c").status == RunStatus.CANCELED
