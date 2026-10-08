"""DynamoDbDocumentStore runs the full DocumentStore conformance suite against moto."""

import threading

import botocore.client
import pytest
from moto import mock_aws

import agent_env.store.document_store.dynamodb_document_store as dynamodb_mod
from agent_env.store import DuplicateKeyError, DynamoDbDocumentStore, Eq, Filter, UpdateSpec
from tst.store import conformance


@pytest.fixture
def store_coll(monkeypatch):
    """moto checks a write's condition and applies the write as two steps any thread can come
    between, where DynamoDB applies each call atomically; one call at a time restores that for
    the kit's racing cases."""
    serial, make_api_call = threading.Lock(), botocore.client.BaseClient._make_api_call

    def one_call_at_a_time(client, operation, params):
        with serial:
            return make_api_call(client, operation, params)

    monkeypatch.setattr(botocore.client.BaseClient, "_make_api_call", one_call_at_a_time)
    with mock_aws():
        yield DynamoDbDocumentStore(table_prefix="test_", region="us-west-2"), "coll"


@pytest.mark.parametrize("case", conformance.CASES, ids=lambda c: c.__name__)
def test_conformance(case, store_coll):
    case(*store_coll)


def test_a_write_that_lands_between_read_and_write_is_not_lost(store_coll):
    store, coll = store_coll
    store.insert(coll, {"id": "x", "rev": 0})
    read = store._candidates

    def read_then_lose_the_race(collection, filter):
        found = read(collection, filter)
        store._candidates = read
        store.update(coll, Filter.of(id="x"), UpdateSpec(inc={"rev": 1}))
        return found

    store._candidates = read_then_lose_the_race
    assert store.update(coll, Filter.of(id="x").where("rev", Eq(0)), UpdateSpec(set={"claimed": True})) == 0
    assert store.find_one(coll, Filter.of(id="x")) == {"id": "x", "rev": 1}


def test_batch_identity_lookup_uses_primary_keys_in_bounded_requests(store_coll, monkeypatch):
    store, coll = store_coll
    store.ensure_index(coll, ["instance_id"], unique=True)
    for i in range(200):
        store.insert(coll, {"instance_id": f"i{i}"})
    requests = []
    original_batch_get = store._client.batch_get_item

    def batch_get(**kwargs):
        request = kwargs["RequestItems"][store._table_name(coll)]
        assert request["ConsistentRead"] is True
        requests.append(len(request["Keys"]))
        return original_batch_get(**kwargs)

    def refuse_scan(*args, **kwargs):
        pytest.fail("batch identity reads must not scan")

    monkeypatch.setattr(store._client, "scan", refuse_scan)
    monkeypatch.setattr(store._client, "get_item", refuse_scan)
    monkeypatch.setattr(store._client, "batch_get_item", batch_get)
    ids = [f"i{i}" for i in range(120)]
    assert store.find_many_by_id(coll, "instance_id", ids + ["missing", "i0"]) == [
        {"instance_id": identity} for identity in ids
    ]
    assert requests == [100, 21]


def test_batch_identity_lookup_retries_unprocessed_keys(store_coll, monkeypatch):
    store, coll = store_coll
    store.ensure_index(coll, ["instance_id"], unique=True)
    store.insert(coll, {"instance_id": "i"})
    original_batch_get = store._client.batch_get_item
    calls = 0

    def batch_get(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"Responses": {}, "UnprocessedKeys": kwargs["RequestItems"]}
        return original_batch_get(**kwargs)

    monkeypatch.setattr(store._client, "batch_get_item", batch_get)
    monkeypatch.setattr(dynamodb_mod.time, "sleep", lambda _: None)
    assert store.find_many_by_id(coll, "instance_id", ["i"]) == [{"instance_id": "i"}]
    assert calls == 2


def test_batch_identity_lookup_does_not_hide_exhausted_retries(store_coll, monkeypatch):
    store, coll = store_coll
    store.ensure_index(coll, ["instance_id"], unique=True)
    monkeypatch.setattr(dynamodb_mod, "_CAS_ATTEMPTS", 2)
    monkeypatch.setattr(dynamodb_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(store._client, "batch_get_item", lambda **kwargs: {"UnprocessedKeys": kwargs["RequestItems"]})
    with pytest.raises(TimeoutError, match="unprocessed keys"):
        store.find_many_by_id(coll, "instance_id", ["i"])


def test_a_write_that_changes_a_unique_index_field_is_refused(store_coll):
    store, coll = store_coll
    store.ensure_index(coll, ["id", "version"], unique=True)
    store.insert(coll, {"id": "a", "version": 1})
    with pytest.raises(NotImplementedError):
        store.update(coll, Filter.of(id="a"), UpdateSpec(set={"id": "b"}))
    assert store.find_one(coll, Filter.of(id="a")) == {"id": "a", "version": 1}


def test_a_process_that_never_ensured_the_index_keys_items_the_same_way(store_coll):
    store, coll = store_coll
    store.ensure_index(coll, ["instance_id", "step_id"], unique=True)
    other = DynamoDbDocumentStore(table_prefix="test_", region="us-west-2")
    other.insert(coll, {"instance_id": "i", "step_id": "a"})
    with pytest.raises(DuplicateKeyError):
        store.insert(coll, {"instance_id": "i", "step_id": "a"})
    assert store.find_one(coll, Filter.of(instance_id="i")) == {"instance_id": "i", "step_id": "a"}


def test_a_unique_index_cannot_be_added_after_the_first_write(store_coll):
    store, coll = store_coll
    store.insert(coll, {"id": "a"})
    with pytest.raises(NotImplementedError, match="before any unique index"):
        store.ensure_index(coll, ["id"], unique=True)


def test_a_write_outlasts_more_conflicts_than_concurrent_steps_cause(store_coll, monkeypatch):
    store, coll = store_coll
    store.insert(coll, {"id": "x", "n": 0})
    other = DynamoDbDocumentStore(table_prefix="test_", region="us-west-2")
    read = store._candidates
    competing = iter(range(20))

    def read_then_lose_the_race(collection, filter):
        found = read(collection, filter)
        if next(competing, None) is not None:
            other.update(coll, Filter.of(id="x"), UpdateSpec(inc={"n": 1}))
        return found

    monkeypatch.setattr(dynamodb_mod.time, "sleep", lambda _s: None)
    store._candidates = read_then_lose_the_race
    assert store.update(coll, Filter.of(id="x"), UpdateSpec(inc={"n": 1})) == 1
    assert store.find_one(coll, Filter.of(id="x")) == {"id": "x", "n": 21}


def test_a_table_another_process_is_still_creating_is_waited_for(store_coll, monkeypatch):
    store, coll = store_coll
    DynamoDbDocumentStore(table_prefix="test_", region="us-west-2").ensure_index(coll, ["id"], unique=True)
    describe = store._client.describe_table
    waited = []
    monkeypatch.setattr(
        store._client, "describe_table", lambda **kw: {"Table": {**describe(**kw)["Table"], "TableStatus": "CREATING"}}
    )
    monkeypatch.setattr(store, "_wait", waited.append)
    store.insert(coll, {"id": "a"})
    assert waited == [f"test_{coll}"]


def test_a_filter_that_pins_the_whole_key_reads_one_item(store_coll, monkeypatch):
    store, coll = store_coll
    store.ensure_index(coll, ["id", "version"], unique=True)
    for version in range(3):
        store.insert(coll, {"id": "a", "version": version})
    monkeypatch.setattr(store._client, "get_paginator", lambda op: pytest.fail(f"{op} for a whole-key read"))
    assert store.find_one(coll, Filter.of(id="a", version=1)) == {"id": "a", "version": 1}


@pytest.mark.parametrize(
    "upsert",
    [
        lambda s, c: s.update(c, Filter.of(id="x"), UpdateSpec(inc={"n": 1}), upsert=True),
        lambda s, c: s.update_one_and_get(c, Filter.of(id="x"), UpdateSpec(inc={"n": 1}), upsert=True),
        lambda s, c: s.replace(c, Filter.of(id="x"), {"id": "x", "n": 1}, upsert=True),
    ],
    ids=["update", "update_one_and_get", "replace"],
)
def test_an_upsert_that_loses_the_insert_race_writes_the_winner(store_coll, upsert):
    store, coll = store_coll
    store.ensure_index(coll, ["id"], unique=True)
    other = DynamoDbDocumentStore(table_prefix="test_", region="us-west-2")
    read = store._candidates

    def read_then_lose_the_insert_race(collection, filter):
        found = read(collection, filter)
        store._candidates = read
        other.insert(coll, {"id": "x", "n": 0})
        return found

    store._candidates = read_then_lose_the_insert_race
    upsert(store, coll)
    assert store.find_one(coll, Filter.of(id="x")) == {"id": "x", "n": 1}


def test_an_upsert_whose_insert_race_winner_is_deleted_inserts_after_all(store_coll):
    store, coll = store_coll
    store.ensure_index(coll, ["id"], unique=True)
    other = DynamoDbDocumentStore(table_prefix="test_", region="us-west-2")
    read = store._candidates

    def read_then_lose_the_insert_race(collection, filter):
        found = read(collection, filter)
        other.insert(coll, {"id": "x", "n": 5})
        store._candidates = delete_the_winner_then_read
        return found

    def delete_the_winner_then_read(collection, filter):
        other.delete(coll, Filter.of(id="x"))
        store._candidates = read
        return read(collection, filter)

    store._candidates = read_then_lose_the_insert_race
    assert store.update(coll, Filter.of(id="x"), UpdateSpec(inc={"n": 1}), upsert=True) == 1
    assert store.find_one(coll, Filter.of(id="x")) == {"id": "x", "n": 1}


def test_an_upsert_whose_key_holds_a_document_the_filter_does_not_match_still_raises(store_coll):
    store, coll = store_coll
    store.ensure_index(coll, ["id"], unique=True)
    store.insert(coll, {"id": "x", "status": "old"})
    with pytest.raises(DuplicateKeyError):
        store.update(coll, Filter.of(id="x", status="new"), UpdateSpec(set={"n": 1}), upsert=True)
    assert store.find_one(coll, Filter.of(id="x")) == {"id": "x", "status": "old"}
