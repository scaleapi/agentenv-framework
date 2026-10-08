"""TaskInstanceStore index coverage.

The hub's instance-list endpoint queries by ``task_id`` (optionally
``task_version``) and sorts by ``created_at_utc``. Without a matching compound
index Mongo falls back to an in-memory sort over every instance for the task,
which on large pass@k/batch tasks ran 60-300s and tripped ``socketTimeoutMS``.
These tests pin that the store *declares* those compound indexes (via the
backend-agnostic ``ensure_index``) so the regression can't silently come back.
``task_instances`` also breaks ties within a minute by ``instance_id``, so each
index ends in it: both sort keys descend and Mongo reads them from the index
scanned backward, and the leading fields still serve a sort on
``created_at_utc`` alone.

Direction is no longer asserted: the store declares ASC compound indexes, which
serve the DESC sort too (Mongo scans backward); a pre-existing DESC index is
kept as-is by ``MongoDocumentStore.ensure_index``.
"""
from __future__ import annotations

from agent_env.task.store import TaskInstanceStore
from tst.unit.store.fakes import FakeDocumentStore


def _fake_after_ensure(monkeypatch) -> FakeDocumentStore:
    fake = FakeDocumentStore()
    monkeypatch.setattr(
        "agent_env.config.Config.get_document_store", lambda self: fake
    )
    store = TaskInstanceStore()
    _ = store._doc_store  # first access ensures indexes
    return fake


def _declared_indexes(monkeypatch) -> list[tuple[list[str], bool]]:
    return _fake_after_ensure(monkeypatch).indexes


def test_task_instance_has_task_id_created_at_compound_index(monkeypatch):
    keys = [fields for fields, _ in _declared_indexes(monkeypatch)]
    assert ["task_id", "created_at_utc", "instance_id"] in keys


def test_task_instance_has_task_id_version_created_at_compound_index(monkeypatch):
    keys = [fields for fields, _ in _declared_indexes(monkeypatch)]
    assert ["task_id", "task_version", "created_at_utc", "instance_id"] in keys


def test_task_instance_preserves_existing_indexes(monkeypatch):
    declared = _declared_indexes(monkeypatch)
    assert (["instance_id"], True) in declared  # unique
    assert ["task_id"] in [fields for fields, _ in declared]


def test_step_journal_declares_its_indexes_and_expires_nothing(monkeypatch):
    # Same retention as the instance it describes: none. An entry that expired first would
    # silently take away the base a later undo replays over.
    fake = _fake_after_ensure(monkeypatch)
    assert (["instance_id", "step_id"], True) in fake.indexes      # one entry per step, replace target
    assert ["instance_id", "seq"] in [f for f, _ in fake.indexes]
    assert fake.ttls == {}
