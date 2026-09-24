"""Integration coverage for the DocumentStore-backed ReviewStore.

Exercises the port of the former ``$setOnInsert``/upsert register to
``insert``+catch-``DuplicateKeyError``, the awaiting-state guard on
``set_decision``, and the TTL index declared via ``ensure_index(ttl_seconds=)``.
"""

import uuid

import pytest

from agent_env.store import Filter, get_config
from agent_env.task_step.review_store import (
    ABORT,
    AWAITING,
    CONTINUE,
    REVIEWS_COLLECTION,
    ReviewStore,
)


def _docs():
    return get_config().get_document_store()


@pytest.fixture
def ids():
    instance_id = f"review-test-{uuid.uuid4().hex[:10]}"
    yield instance_id, "cp1"
    _docs().delete(REVIEWS_COLLECTION, Filter.of(instance_id=instance_id))


@pytest.mark.integration
def test_put_awaiting_is_idempotent_and_get_roundtrips(ids):
    instance_id, step_id = ids
    store = ReviewStore()
    store.put_awaiting(instance_id, step_id, label="review", upstream={"k": "v"})

    doc = store.get(instance_id, step_id)
    assert doc is not None
    assert doc["state"] == AWAITING
    assert doc["label"] == "review"
    assert "_id" not in doc

    # A retry must not clobber the existing record.
    store.set_decision(instance_id, step_id, CONTINUE, decided_by="op@example.com")
    store.put_awaiting(instance_id, step_id, label="ignored")
    doc = store.get(instance_id, step_id)
    assert doc["state"] == CONTINUE
    assert doc["label"] == "review"  # original, not clobbered


@pytest.mark.integration
def test_set_decision_only_on_awaiting(ids):
    instance_id, step_id = ids
    store = ReviewStore()

    assert store.set_decision(instance_id, step_id, CONTINUE) is False  # nothing awaiting

    store.put_awaiting(instance_id, step_id)
    assert store.set_decision(instance_id, step_id, CONTINUE, decided_by="a@b.com") is True
    # Already decided -> no longer awaiting -> guarded no-op.
    assert store.set_decision(instance_id, step_id, ABORT) is False
    assert store.get(instance_id, step_id)["state"] == CONTINUE


@pytest.mark.integration
def test_list_awaiting_filters_by_state_and_instance(ids):
    instance_id, step_id = ids
    store = ReviewStore()
    store.put_awaiting(instance_id, "cp1")
    store.put_awaiting(instance_id, "cp2")
    store.set_decision(instance_id, "cp2", CONTINUE)

    awaiting = store.list_awaiting(instance_id)
    step_ids = {d["step_id"] for d in awaiting}
    assert step_ids == {"cp1"}  # cp2 decided, excluded
    _docs().delete(REVIEWS_COLLECTION, Filter.of(instance_id=instance_id))
