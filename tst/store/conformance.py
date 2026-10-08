"""Backend-neutral DocumentStore conformance assertions.

Each function takes ``(store, coll)`` and exercises one behavior of the
``DocumentStore`` contract. Both `MongoDocumentStore` and
`LocalSqliteDocumentStore` must pass every case in ``CASES`` — that shared pass
is the concrete "the abstraction generalizes" guarantee.
"""

import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_env.store import (
    AbsentOrNull,
    DuplicateKeyError,
    Eq,
    Filter,
    In,
    LteOrAbsent,
    Ne,
    Sort,
    UpdateSpec,
    VersionedEntityStore,
    compare_and_swap,
)

# Enough concurrent writers on one document that some of them race.
_RACERS = 8


def insert_find_one_strips_id(store, coll):
    store.insert(coll, {"id": "a", "version": 1, "x": 1})
    doc = store.find_one(coll, Filter.of(id="a", version=1))
    assert doc == {"id": "a", "version": 1, "x": 1}
    assert "_id" not in doc


def ensure_index_unique_rejects_duplicate(store, coll):
    store.ensure_index(coll, ["id", "version"], unique=True)
    store.insert(coll, {"id": "a", "version": 1})
    with pytest.raises(DuplicateKeyError):
        store.insert(coll, {"id": "a", "version": 1})


def ensure_index_unique_on_other_fields_rejects_duplicate(store, coll):
    store.ensure_index(coll, ["instance_id", "step_id"], unique=True)
    store.insert(coll, {"instance_id": "i", "step_id": "a"})
    store.insert(coll, {"instance_id": "i", "step_id": "b"})
    with pytest.raises(DuplicateKeyError):
        store.insert(coll, {"instance_id": "i", "step_id": "a", "seq": 2})
    assert store.update(coll, Filter.of(instance_id="i", step_id="b"), UpdateSpec(set={"seq": 1})) == 1
    assert store.query(coll, Filter.of(instance_id="i"), sort=Sort.by("step_id", descending=False)) == [
        {"instance_id": "i", "step_id": "a"},
        {"instance_id": "i", "step_id": "b", "seq": 1},
    ]


def ensure_index_non_unique_allows_duplicate(store, coll):
    store.ensure_index(coll, ["x"])
    store.insert(coll, {"id": "a", "version": 1, "x": 7})
    store.insert(coll, {"id": "b", "version": 1, "x": 7})
    assert store.count(coll, Filter.of(x=7)) == 2


def absent_or_null_precondition(store, coll):
    store.insert(coll, {"id": "a", "version": 1})  # no "ts"
    assert store.update(
        coll, Filter.of(id="a").where("ts", AbsentOrNull()), UpdateSpec(set={"ts": "t1"})
    ) == 1
    assert store.update(
        coll, Filter.of(id="a").where("ts", AbsentOrNull()), UpdateSpec(set={"z": 1})
    ) == 0
    store.update(coll, Filter.of(id="a"), UpdateSpec(set={"ts": None}))
    assert store.update(
        coll, Filter.of(id="a").where("ts", AbsentOrNull()), UpdateSpec(set={"z": 2})
    ) == 1


def ne_requires_present_field(store, coll):
    store.insert(coll, {"id": "other", "status": "open"})
    store.insert(coll, {"id": "match", "status": "done"})
    store.insert(coll, {"id": "absent"})
    ids = {d["id"] for d in store.query(coll, Filter().where("status", Ne("done")))}
    assert ids == {"other"}


def two_fused_predicates_and(store, coll):
    store.insert(coll, {"id": "both-absent"})
    store.insert(coll, {"id": "one-present", "ts1": "2021"})
    f = Filter().where("ts1", AbsentOrNull()).where("ts2", AbsentOrNull())
    assert {d["id"] for d in store.query(coll, f)} == {"both-absent"}


def find_one_latest_by_sort(store, coll):
    for v in (1, 2, 3):
        store.insert(coll, {"id": "a", "version": v})
    latest = store.find_one(coll, Filter.of(id="a"), sort=Sort.by("version", descending=True))
    assert latest["version"] == 3


def query_sort_offset_limit(store, coll):
    for v in range(1, 6):
        store.insert(coll, {"id": "a", "version": v})
    res = store.query(
        coll, Filter.of(id="a"), sort=Sort.by("version", descending=False), limit=2, offset=1
    )
    assert [d["version"] for d in res] == [2, 3]


def query_limit_zero_returns_all(store, coll):
    for v in range(1, 6):
        store.insert(coll, {"id": "a", "version": v})
    assert len(store.query(coll, Filter.of(id="a"), limit=0)) == 5


def count(store, coll):
    for v in range(1, 4):
        store.insert(coll, {"id": "a", "version": v})
    store.insert(coll, {"id": "b", "version": 1})
    assert store.count(coll, Filter.of(id="a")) == 3
    assert store.count(coll, Filter()) == 4


def update_ops(store, coll):
    store.insert(coll, {"id": "a", "version": 1, "n": 1, "tags": ["x"], "log": ["a"], "drop": 1})
    matched = store.update(
        coll,
        Filter.of(id="a"),
        UpdateSpec(
            set={"s": 1}, unset={"drop"}, inc={"n": 2},
            add_to_set={"tags": ["x", "y"]}, push={"log": ["b"]},
        ),
    )
    assert matched == 1
    doc = store.find_one(coll, Filter.of(id="a"))
    assert doc["s"] == 1
    assert "drop" not in doc
    assert doc["n"] == 3
    assert sorted(doc["tags"]) == ["x", "y"]  # add_to_set dedupes existing "x"
    assert doc["log"] == ["a", "b"]           # push does not dedupe


def update_cas_miss_and_hit(store, coll):
    store.insert(coll, {"id": "a", "version": 1, "status": "open"})
    miss = store.update(
        coll, Filter.of(id="a").where("status", Ne("open")), UpdateSpec(set={"status": "done"})
    )
    assert miss == 0
    assert store.find_one(coll, Filter.of(id="a"))["status"] == "open"
    hit = store.update(
        coll, Filter.of(id="a").where("status", Eq("open")), UpdateSpec(set={"status": "done"})
    )
    assert hit == 1
    assert store.find_one(coll, Filter.of(id="a"))["status"] == "done"


def lte_or_absent_precondition(store, coll):
    store.insert(coll, {"id": "a", "version": 1})  # no updated_at
    assert store.update(
        coll, Filter.of(id="a").where("updated_at", LteOrAbsent("2020")),
        UpdateSpec(set={"updated_at": "2021"}),
    ) == 1
    assert store.update(
        coll, Filter.of(id="a").where("updated_at", LteOrAbsent("2020")),
        UpdateSpec(set={"z": 1}),
    ) == 0
    assert store.update(
        coll, Filter.of(id="a").where("updated_at", LteOrAbsent("2022")),
        UpdateSpec(set={"z": 1}),
    ) == 1


def update_one_and_get_before_after_and_miss(store, coll):
    store.insert(coll, {"id": "a", "version": 1, "n": 1})
    after = store.update_one_and_get(coll, Filter.of(id="a"), UpdateSpec(inc={"n": 1}))
    assert after["n"] == 2 and "_id" not in after
    before = store.update_one_and_get(
        coll, Filter.of(id="a"), UpdateSpec(inc={"n": 1}), return_after=False
    )
    assert before["n"] == 2  # pre-update snapshot
    assert store.update_one_and_get(coll, Filter.of(id="zzz"), UpdateSpec(set={"n": 9})) is None


def replace_and_upsert(store, coll):
    store.insert(coll, {"id": "a", "version": 1, "x": 1})
    assert store.replace(coll, Filter.of(id="a", version=1), {"id": "a", "version": 1, "y": 2}) == 1
    assert store.find_one(coll, Filter.of(id="a", version=1)) == {"id": "a", "version": 1, "y": 2}
    assert store.replace(
        coll, Filter.of(id="b", version=1), {"id": "b", "version": 1, "z": 3}, upsert=True
    ) == 1
    assert store.find_one(coll, Filter.of(id="b", version=1))["z"] == 3


def update_upsert_synthesizes_from_filter(store, coll):
    assert store.update(
        coll, Filter.of(id="c", version=1), UpdateSpec(set={"w": 1}), upsert=True
    ) == 1
    assert store.find_one(coll, Filter.of(id="c", version=1)) == {"id": "c", "version": 1, "w": 1}


def upsert_colliding_with_a_unique_index_raises_duplicate_key(store, coll):
    """What losing a concurrent upsert race looks like: the filter misses, and the document the
    upsert then inserts collides with one another writer already inserted."""
    store.ensure_index(coll, ["id", "version"], unique=True)
    store.insert(coll, {"id": "a", "version": 1, "state": "held"})
    missing = Filter.of(id="a", version=1, state="free")
    with pytest.raises(DuplicateKeyError):
        store.update(coll, missing, UpdateSpec(set={"owner": "x"}), upsert=True)
    with pytest.raises(DuplicateKeyError):
        store.update_one_and_get(coll, missing, UpdateSpec(set={"owner": "x"}), upsert=True)
    with pytest.raises(DuplicateKeyError):
        store.replace(coll, missing, {"id": "a", "version": 1, "state": "free"}, upsert=True)
    assert store.find_one(coll, Filter.of(id="a", version=1)) == {"id": "a", "version": 1, "state": "held"}


def delete(store, coll):
    store.insert(coll, {"id": "a", "version": 1})
    assert store.delete(coll, Filter.of(id="a", version=1)) == 1
    assert store.find_one(coll, Filter.of(id="a", version=1)) is None
    assert store.delete(coll, Filter.of(id="zzz")) == 0


def eq_and_in_array_containment(store, coll):
    store.insert(coll, {"id": "a", "tags": ["red", "blue"]})
    assert store.find_one(coll, Filter().where("tags", Eq("red")))["id"] == "a"
    assert store.find_one(coll, Filter().where("tags", Eq("green"))) is None
    assert store.find_one(coll, Filter().where("tags", In(["green", "blue"])))["id"] == "a"


def dotted_array_traversal(store, coll):
    store.insert(coll, {"id": "conv", "a2a_tasks": [{"a2a_task_id": "t1"}, {"a2a_task_id": "t2"}]})
    assert store.find_one(coll, Filter().where("a2a_tasks.a2a_task_id", Eq("t2")))["id"] == "conv"
    assert store.find_one(coll, Filter().where("a2a_tasks.a2a_task_id", Eq("nope"))) is None


def positional_array_index_filter_and_set(store, coll):
    store.insert(coll, {"id": "conv", "a2a_tasks": [{"state": "working"}, {"state": "working"}]})
    hit = store.update(
        coll,
        Filter.of(id="conv").where("a2a_tasks.0.state", Eq("working")),
        UpdateSpec(set={"a2a_tasks.0.state": "done"}),
    )
    assert hit == 1
    tasks = store.find_one(coll, Filter.of(id="conv"))["a2a_tasks"]
    assert tasks[0]["state"] == "done" and tasks[1]["state"] == "working"
    miss = store.update(
        coll,
        Filter.of(id="conv").where("a2a_tasks.0.state", Eq("working")),
        UpdateSpec(set={"a2a_tasks.0.state": "x"}),
    )
    assert miss == 0


def versioned_entity_store_roundtrip(store, coll):
    ves: VersionedEntityStore[dict] = VersionedEntityStore(
        store, coll, serialize=dict, deserialize=lambda d: d
    )
    assert ves.put({"id": "e"}) == 1
    assert ves.put({"id": "e"}) == 2
    assert ves.get("e")["version"] == 2
    assert ves.get("e", version=1)["version"] == 1
    assert ves.get("missing") is None
    assert ves.next_version("e") == 3
    assert ves.count(Filter.of(id="e")) == 2


def latest_per_id_reduces_to_newest_version(store, coll):
    """One row per id, the highest-version one; filter applies before grouping."""
    for doc in [
        {"id": "a", "version": 1, "kind": "env", "name": "a-v1"},
        {"id": "a", "version": 3, "kind": "env", "name": "a-v3"},
        {"id": "a", "version": 2, "kind": "env", "name": "a-v2"},
        {"id": "b", "version": 1, "kind": "env", "name": "b-v1"},
        {"id": "c", "version": 7, "kind": "task", "name": "c-v7"},
    ]:
        store.insert(coll, doc)

    rows = store.latest_per_id(coll, Filter())
    assert {(r["id"], r["version"]) for r in rows} == {("a", 3), ("b", 1), ("c", 7)}
    assert store.count_distinct(coll, Filter()) == 3

    # filter selects among ALL versions, then reduces
    envs = store.latest_per_id(coll, Filter.of(kind="env"))
    assert {(r["id"], r["version"]) for r in envs} == {("a", 3), ("b", 1)}
    assert store.count_distinct(coll, Filter.of(kind="env")) == 2

    # a filter that only matches an older version still yields that version
    old = store.latest_per_id(coll, Filter.of(name="a-v1"))
    assert [(r["id"], r["version"]) for r in old] == [("a", 1)]


def latest_per_id_sort_offset_limit_apply_after_grouping(store, coll):
    for doc in [
        {"id": "a", "version": 2, "rank": 30},
        {"id": "a", "version": 1, "rank": 99},
        {"id": "b", "version": 1, "rank": 20},
        {"id": "c", "version": 1, "rank": 10},
    ]:
        store.insert(coll, doc)

    ordered = store.latest_per_id(coll, Filter(), sort=Sort.by("rank", descending=True))
    assert [r["id"] for r in ordered] == ["a", "b", "c"]      # a ranks 30, not 99
    assert [r["rank"] for r in ordered] == [30, 20, 10]

    page = store.latest_per_id(coll, Filter(), sort=Sort.by("rank", descending=True),
                               offset=1, limit=1)
    assert [r["id"] for r in page] == ["b"]

    # the pagination total counts entities, not rows (4 docs, 3 ids)
    assert store.count(coll, Filter()) == 4
    assert store.count_distinct(coll, Filter()) == 3


def find_many_by_id_preserves_first_documents_and_request_order(store, coll):
    store.insert(coll, {"identity": "a", "value": "first"})
    store.insert(coll, {"identity": "b", "value": "second"})
    store.insert(coll, {"identity": "a", "value": "later"})

    assert store.find_many_by_id(coll, "identity", ["b", "missing", "a", "b"]) == [
        store.find_one(coll, Filter.of(identity="b")),
        store.find_one(coll, Filter.of(identity="a")),
    ]
    assert store.find_many_by_id(coll, "identity", []) == []


def latest_per_id_page_returns_entity_total_and_window(store, coll):
    for doc in [
        {"eid": "a", "rev": 1, "rank": 99},
        {"eid": "a", "rev": 2, "rank": 30},
        {"eid": "b", "rev": 1, "rank": 20},
        {"eid": "c", "rev": 1, "rank": 10},
        {"rank": 100},
    ]:
        store.insert(coll, doc)

    for limit, expected in [(1, ["b"]), (0, ["b", "c"]), (None, ["b", "c"])]:
        page, total = store.latest_per_id_page(
            coll, Filter(), id_field="eid", version_field="rev",
            sort=Sort.by("rank", descending=True), offset=1, limit=limit,
        )
        assert [doc["eid"] for doc in page] == expected
        assert total == 3

    page, total = store.latest_per_id_page(coll, Filter(), id_field="eid", offset=10, limit=1)
    assert page == []
    assert total == 3


def latest_per_id_orders_absent_last_in_both_directions(store, coll):
    """A missing sort field lands last for ascending and descending alike; otherwise
    offset/limit page the wrong entities."""
    for doc in [
        {"id": "a", "version": 1, "rank": 5},
        {"id": "b", "version": 1},  # no rank
        {"id": "c", "version": 1, "rank": 1},
    ]:
        store.insert(coll, doc)

    asc = store.latest_per_id(coll, Filter(), sort=Sort.by("rank", descending=False))
    assert [r["id"] for r in asc] == ["c", "a", "b"]

    desc = store.latest_per_id(coll, Filter(), sort=Sort.by("rank", descending=True))
    assert [r["id"] for r in desc] == ["a", "c", "b"]


def latest_per_id_skips_docs_without_identity(store, coll):
    store.insert(coll, {"id": "a", "version": 1})
    store.insert(coll, {"version": 5})                       # no id -> not groupable
    rows = store.latest_per_id(coll, Filter())
    assert [(r["id"], r["version"]) for r in rows] == [("a", 1)]
    assert store.count_distinct(coll, Filter()) == 1


def latest_per_id_missing_version_sorts_lowest(store, coll):
    store.insert(coll, {"id": "a", "name": "unversioned"})
    store.insert(coll, {"id": "a", "version": 1, "name": "versioned"})
    rows = store.latest_per_id(coll, Filter())
    assert len(rows) == 1 and rows[0]["name"] == "versioned"


def latest_per_id_honours_custom_fields(store, coll):
    store.insert(coll, {"eid": "x", "rev": 1})
    store.insert(coll, {"eid": "x", "rev": 4})
    store.insert(coll, {"eid": "y", "rev": 2})
    rows = store.latest_per_id(coll, Filter(), id_field="eid", version_field="rev")
    assert {(r["eid"], r["rev"]) for r in rows} == {("x", 4), ("y", 2)}
    assert store.count_distinct(coll, Filter(), id_field="eid") == 2


def _race(work):
    """Run ``work(racer)`` for every racer at once, each in a copy of the caller's context (so a
    run scope set around the case still applies), and return the results in racer order."""
    start = threading.Barrier(_RACERS)

    def run(racer):
        start.wait()
        return work(racer)

    with ThreadPoolExecutor(max_workers=_RACERS) as pool:
        futures = [pool.submit(contextvars.copy_context().run, run, racer) for racer in range(_RACERS)]
        return [f.result() for f in futures]


_UNCLAIMED = Filter.of(id="x").where("owner", AbsentOrNull())


def racing_guarded_updates_have_exactly_one_winner(store, coll):
    """Writers race to claim one document; each is told whether its own write landed."""
    store.insert(coll, {"id": "x"})
    won = _race(lambda racer: store.update(coll, _UNCLAIMED, UpdateSpec(set={"owner": racer})))
    assert sorted(won) == [0] * (_RACERS - 1) + [1]
    assert store.find_one(coll, Filter.of(id="x")) == {"id": "x", "owner": won.index(1)}


def racing_guarded_replaces_have_exactly_one_winner(store, coll):
    store.insert(coll, {"id": "x"})
    won = _race(lambda racer: store.replace(coll, _UNCLAIMED, {"id": "x", "owner": racer}))
    assert sorted(won) == [0] * (_RACERS - 1) + [1]
    assert store.find_one(coll, Filter.of(id="x")) == {"id": "x", "owner": won.index(1)}


def racing_update_one_and_get_returns_only_its_own_write(store, coll):
    store.insert(coll, {"id": "x"})
    got = _race(lambda racer: store.update_one_and_get(coll, _UNCLAIMED, UpdateSpec(set={"owner": racer})))
    winners = [(racer, doc) for racer, doc in enumerate(got) if doc is not None]
    assert len(winners) == 1
    racer, doc = winners[0]
    assert doc == {"id": "x", "owner": racer} == store.find_one(coll, Filter.of(id="x"))


def racing_pre_image_updates_have_exactly_one_winner(store, coll):
    store.insert(coll, {"id": "x"})
    got = _race(
        lambda racer: store.update_one_and_get(coll, _UNCLAIMED, UpdateSpec(set={"owner": racer}), return_after=False)
    )
    winners = [racer for racer, doc in enumerate(got) if doc is not None]
    assert len(winners) == 1 and got[winners[0]] == {"id": "x"}
    assert store.find_one(coll, Filter.of(id="x")) == {"id": "x", "owner": winners[0]}


def concurrent_compare_and_swap_loses_no_update(store, coll):
    """Each writer's mutation depends on what it read, so a write applied under a stale guard
    drops another writer's step."""
    store.insert(coll, {"id": "x", "steps": []})
    _race(lambda racer: compare_and_swap(
        store, coll, Filter.of(id="x"), lambda doc: UpdateSpec(set={"steps": doc["steps"] + [racer]}), counter_field="rev"
    ))
    doc = store.find_one(coll, Filter.of(id="x"))
    assert sorted(doc["steps"]) == list(range(_RACERS)) and doc["rev"] == _RACERS


def update_one_and_get_upsert_inserts_on_a_miss(store, coll):
    after = store.update_one_and_get(coll, Filter.of(id="a"), UpdateSpec(set={"n": 1}), upsert=True)
    assert after == {"id": "a", "n": 1}
    before = store.update_one_and_get(coll, Filter.of(id="b"), UpdateSpec(set={"n": 2}), return_after=False, upsert=True)
    assert before is None
    assert store.find_one(coll, Filter.of(id="b")) == {"id": "b", "n": 2}


CASES = [
    insert_find_one_strips_id,
    ensure_index_unique_rejects_duplicate,
    ensure_index_unique_on_other_fields_rejects_duplicate,
    ensure_index_non_unique_allows_duplicate,
    absent_or_null_precondition,
    ne_requires_present_field,
    two_fused_predicates_and,
    find_one_latest_by_sort,
    query_sort_offset_limit,
    query_limit_zero_returns_all,
    count,
    update_ops,
    update_cas_miss_and_hit,
    lte_or_absent_precondition,
    update_one_and_get_before_after_and_miss,
    replace_and_upsert,
    update_upsert_synthesizes_from_filter,
    upsert_colliding_with_a_unique_index_raises_duplicate_key,
    delete,
    eq_and_in_array_containment,
    dotted_array_traversal,
    positional_array_index_filter_and_set,
    versioned_entity_store_roundtrip,
    latest_per_id_reduces_to_newest_version,
    latest_per_id_sort_offset_limit_apply_after_grouping,
    latest_per_id_page_returns_entity_total_and_window,
    find_many_by_id_preserves_first_documents_and_request_order,
    latest_per_id_orders_absent_last_in_both_directions,
    latest_per_id_skips_docs_without_identity,
    latest_per_id_missing_version_sorts_lowest,
    latest_per_id_honours_custom_fields,
    update_one_and_get_upsert_inserts_on_a_miss,
    racing_guarded_updates_have_exactly_one_winner,
    racing_guarded_replaces_have_exactly_one_winner,
    racing_update_one_and_get_returns_only_its_own_write,
    racing_pre_image_updates_have_exactly_one_winner,
    concurrent_compare_and_swap_loses_no_update,
]
