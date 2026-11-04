from __future__ import annotations

import pytest

from agent_env.config import set_document_store
from agent_env.store import Filter, LocalSqliteDocumentStore
from agent_env.task.step_journal import commit_ordered, replay_context
from agent_env.task.store import (
    TASK_INSTANCES_COLLECTION,
    TASK_STEP_JOURNAL_COLLECTION,
    TaskInstanceStore,
    register_task_instance,
    seed_task_instance_context,
    set_task_instance_store,
)
from agent_env.task_step.context import TaskStepContext


@pytest.fixture
def store(tmp_path) -> TaskInstanceStore:
    doc = LocalSqliteDocumentStore(str(tmp_path / "j.db"))
    s = TaskInstanceStore()
    set_document_store(doc)
    set_task_instance_store(s)
    try:
        yield s
    finally:
        set_task_instance_store(None)


def _register():
    return register_task_instance("t", 1, 3, instance_id="i")


def _seed(run, owner: str) -> None:
    seed_task_instance_context("i", TaskStepContext(metadata={"run_group_id": owner}), run.attempt_epoch)


def _doc(store):
    return store._doc_store.find_one(TASK_INSTANCES_COLLECTION, Filter.of(instance_id="i"))


def _seed_row(store):
    return store._doc_store.find_one(TASK_STEP_JOURNAL_COLLECTION, Filter.of(instance_id="i", step_id="__seed__"))


def _replayed(store, doc):
    return replay_context(commit_ordered(store.journal_entries_sync("i"), doc["completed_steps"]), doc["rerecorded_steps"])


def test_a_superseded_run_cannot_seed_over_the_run_that_replaced_it(store):
    superseded = _register()
    live = _register()
    assert (superseded.attempt_epoch, live.attempt_epoch) == (0, 1)
    _seed(live, "B")
    _seed(superseded, "A")
    d = _doc(store)
    assert d["context"]["metadata"]["run_group_id"] == "B"
    assert _replayed(store, d) == d["context"]


def test_replaying_the_live_journal_reproduces_the_stored_context_after_a_late_seed(store, monkeypatch):
    superseded = _register()
    live = _register()
    real = store._journal_step

    def a_seeds_between_bs_doc_write_and_its_journal_row(docs, instance_id, step_id, ops, status, **kw):
        monkeypatch.setattr(store, "_journal_step", real)
        _seed(superseded, "A")
        return real(docs, instance_id, step_id, ops, status, **kw)

    monkeypatch.setattr(store, "_journal_step", a_seeds_between_bs_doc_write_and_its_journal_row)
    _seed(live, "B")
    d = _doc(store)
    assert (_seed_row(store)["generation"], d["run_generation"]) == (1, 1)
    assert _replayed(store, d) == d["context"]
