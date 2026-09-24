"""Pure-dataclass unit tests for the document-store abstraction value types.

No backend / network: these exercise Filter, UpdateSpec, and Sort only.
"""

import pytest

from agent_env.store.document_store import (
    AbsentOrNull,
    DuplicateKeyError,
    Eq,
    Exists,
    Filter,
    Gte,
    In,
    Lte,
    Ne,
    Sort,
    SortKey,
    UpdateSpec,
    VersionedEntityStore,
    VersionedEntityStoreCache,
)
from agent_env.store.document_store.mongo_document_store import _to_mongo_filter
from agent_env.store.query import QueryBuilder, to_document_query
from agent_env.task_step.context_ops import ContextUpdateOps
from tst.unit.store.fakes import FakeDocumentStore


def test_mongo_translator_rejects_eq_operator_fusion():
    with pytest.raises(ValueError, match="fuses"):
        _to_mongo_filter(Filter.of(v=5).where("v", Gte(1)))
    with pytest.raises(ValueError, match="fuses"):
        _to_mongo_filter(Filter().where("v", Gte(1)).where("v", Eq(5)))
    assert _to_mongo_filter(Filter.of(id="a")) == {"id": "a"}
    assert _to_mongo_filter(Filter().where("n", Gte(1)).where("n", Lte(9))) == {"n": {"$gte": 1, "$lte": 9}}


class TestUpdateSpecPathConflicts:
    def test_same_path_under_two_operators_is_rejected(self):
        with pytest.raises(ValueError, match="multiple operators"):
            UpdateSpec(add_to_set={"x": [1]}, set={"x": 2}).validate()

    def test_ancestor_and_descendant_paths_are_rejected(self):
        with pytest.raises(ValueError, match="ancestors"):
            UpdateSpec(set={"a": {}}, add_to_set={"a.l": [1]}).validate()
        with pytest.raises(ValueError, match="ancestors"):
            UpdateSpec(unset={"a"}, set={"a.b.c": 1}).validate()
        UpdateSpec(set={"a.b": 1, "a.c": 2}, add_to_set={"a.d": [1]}).validate()  # siblings are fine


class TestFilter:
    def test_of_builds_all_equality(self):
        f = Filter.of(id="x", version=3)
        assert f.conditions == {"id": [Eq("x")], "version": [Eq(3)]}

    def test_where_adds_predicate_without_mutating_original(self):
        base = Filter.of(id="x")
        extended = base.where("metadata.updated_at", Exists(True))
        assert base.conditions == {"id": [Eq("x")]}  # original unchanged
        assert extended.conditions["metadata.updated_at"] == [Exists(True)]
        assert extended.conditions["id"] == [Eq("x")]

    def test_where_stacks_two_predicates_on_one_field(self):
        f = Filter().where("version", Gte(1)).where("version", Lte(10))
        assert f.conditions["version"] == [Gte(1), Lte(10)]


class TestPredicates:
    def test_frozen(self):
        with pytest.raises((AttributeError, Exception)):
            Eq("x").value = "y"  # type: ignore[misc]

    def test_hashable(self):
        assert {Eq("a"), Eq("a"), Ne(1)} == {Eq("a"), Ne(1)}


class TestUpdateSpec:
    def test_is_empty(self):
        assert UpdateSpec().is_empty()
        assert not UpdateSpec(set={"a": 1}).is_empty()
        assert not UpdateSpec(inc={"rev": 1}).is_empty()

    def test_validate_passes_on_disjoint_paths(self):
        UpdateSpec(
            set={"a": 1}, unset={"b"}, inc={"rev": 1},
            add_to_set={"c": [1]}, push={"d": [2]},
        ).validate()  # no raise

    def test_validate_raises_on_same_path_two_operators(self):
        with pytest.raises(ValueError, match="a"):
            UpdateSpec(set={"a": 1}, unset={"a"}).validate()

    def test_validate_raises_across_add_to_set_and_push(self):
        with pytest.raises(ValueError, match="x"):
            UpdateSpec(add_to_set={"x": [1]}, push={"x": [2]}).validate()


class TestSort:
    def test_by_single_key_descending_default(self):
        assert Sort.by("version") == Sort((SortKey("version", True),))

    def test_by_ascending(self):
        s = Sort.by("created_at", descending=False)
        assert s.keys == (SortKey("created_at", False),)


class TestApplySort:
    """`_apply_sort` must never raise on values Python can't `<`-compare — a 500 in the
    explorer list endpoint otherwise (latest_per_id sorts before returning)."""

    def test_mixed_types_do_not_raise(self):
        from agent_env.store.document_store.document_store import _apply_sort

        docs = [{"f": 10}, {"f": "a"}, {"f": 2}, {"f": None}, {"other": 1}]
        # int vs str must not reach a `<`; absent/None order last regardless of direction.
        out = _apply_sort(docs, Sort.by("f", descending=False))
        assert [d.get("f") for d in out[:3]] == [2, 10, "a"]  # ints by value, then the str
        assert {id(d) for d in out[3:]} == {id(docs[3]), id(docs[4])}  # None + missing last

    def test_same_type_uncomparable_values_do_not_raise(self):
        from agent_env.store.document_store.document_store import _apply_sort

        # Two dicts, and two lists with uncomparable elements — a raw `<` raises TypeError.
        docs = [{"f": {"b": 1}}, {"f": {"a": 2}}, {"f": [1, "x"]}, {"f": ["x", 1]}]
        out = _apply_sort(docs, Sort.by("f", descending=False))
        assert len(out) == len(docs)  # deterministic order, no exception
        # Re-sorting is stable (deterministic serialized key).
        assert out == _apply_sort(list(docs), Sort.by("f", descending=False))


class TestVersionedEntityStoreCache:
    """The cache is lock-free, so what it *returns* carries the guarantee, not what it stores."""

    def _cache(self):
        return VersionedEntityStoreCache("c", dict, lambda d: d)

    def test_one_backend_is_built_once(self):
        # Building runs ensure_index; a per-call rebuild would pay that on every read.
        docs, cache = FakeDocumentStore(), self._cache()
        assert cache.for_store(docs) is cache.for_store(docs)

    def test_a_new_backend_gets_a_new_view(self):
        cache = self._cache()
        first = cache.for_store(FakeDocumentStore())
        assert cache.for_store(FakeDocumentStore()) is not first

    def test_a_builder_that_loses_the_race_still_returns_its_own_backend(self, monkeypatch):
        # Two callers with different backends, interleaved so the one that started first
        # finishes last. It must be handed the view it built, not whichever the field ended
        # up holding -- otherwise it reads and writes a database it never asked for.
        a, b = FakeDocumentStore(), FakeDocumentStore()
        cache = self._cache()

        real_init, intruded = VersionedEntityStore.__init__, []

        def racing_init(self, store, *args, **kwargs):
            real_init(self, store, *args, **kwargs)
            if store is a and not intruded:
                intruded.append(True)
                cache.for_store(b)          # the other caller completes mid-build
        monkeypatch.setattr(VersionedEntityStore, "__init__", racing_init)

        view = cache.for_store(a)
        assert intruded, "the race was not exercised"
        assert view._doc_store is a
        # and the pair it left behind is consistent, never a marker from one with a view
        # from the other
        monkeypatch.undo()
        assert cache.for_store(a) is view


def _versioned(docs: FakeDocumentStore) -> VersionedEntityStore[dict]:
    return VersionedEntityStore(docs, "c", serialize=dict, deserialize=lambda d: d)


class TestVersionedEntityStore:
    def test_put_assigns_sequential_versions(self):
        store = _versioned(FakeDocumentStore())
        assert store.put({"id": "e"}) == 1
        assert store.put({"id": "e"}) == 2

    def test_get_exact_and_latest(self):
        docs = FakeDocumentStore()
        store = _versioned(docs)
        store.put({"id": "e"})
        store.put({"id": "e"})
        assert store.get("e")["version"] == 2       # latest
        assert store.get("e", version=1)["version"] == 1
        assert store.get("missing") is None

    def test_next_version(self):
        store = _versioned(FakeDocumentStore())
        assert store.next_version("e") == 1
        store.put({"id": "e"})
        assert store.next_version("e") == 2

    def test_put_retries_past_a_collision(self):
        docs = FakeDocumentStore(fail_inserts=1)
        store = _versioned(docs)
        # First insert "loses" version 1 to a simulated concurrent writer, so
        # the retry re-reads next_version (now 2) and succeeds there.
        assert store.put({"id": "e"}) == 2

    def test_put_gives_up_after_max_retries(self):
        store = _versioned(FakeDocumentStore(fail_inserts=99))
        with pytest.raises(DuplicateKeyError, match="after"):
            store.put({"id": "e"}, max_retries=3)


class _Q(QueryBuilder):
    def execute(self):
        return []

    def _execute_count(self):
        return 0


class TestToDocumentQuery:
    def test_translates_operators_and_sort(self):
        q = _Q().id("x").version_gte(2).sort("version", descending=True)
        q = q._add_filter("kind_in", ["a", "b"])._add_filter("n_lte", 5)
        filt, sort = to_document_query(q)
        assert filt.conditions == {
            "id": [Eq("x")],
            "version": [Gte(2)],
            "kind": [In(["a", "b"])],
            "n": [Lte(5)],
        }
        assert sort == Sort.by("version", descending=True)

    def test_no_sort_returns_none(self):
        filt, sort = to_document_query(_Q().id("x"))
        assert filt.conditions == {"id": [Eq("x")]}
        assert sort is None


class TestContextUpdateOpsToUpdateSpec:
    def test_maps_fields_one_to_one(self):
        spec = ContextUpdateOps(
            sets={"context.a": 1}, unsets={"context.b"}, add_to_sets={"context.c": [1, 2]}
        ).to_update_spec()
        assert spec.set == {"context.a": 1}
        assert spec.unset == {"context.b"}
        assert spec.add_to_set == {"context.c": [1, 2]}
        assert spec.inc == {} and spec.push == {}

    def test_empty(self):
        assert ContextUpdateOps().to_update_spec().is_empty()


def test_absent_or_null_is_frozen_and_hashable():
    assert {AbsentOrNull(), AbsentOrNull()} == {AbsentOrNull()}
    with pytest.raises((AttributeError, Exception)):
        AbsentOrNull().x = 1  # type: ignore[attr-defined]


class TestReservedIdPrefix:
    """`@` opens `@namespace/id` for a federated registry, so it cannot start a local id.
    Reserved while agent-env is private: no stored document uses it, and once third parties
    have created ids the reservation stops being free."""

    def _store(self):
        return VersionedEntityStore(
            FakeDocumentStore(), "things",
            serialize=lambda e: dict(e), deserialize=lambda d: d,
        )

    def test_an_at_prefixed_id_is_refused(self):
        with pytest.raises(ValueError, match=r"reserved '@' prefix"):
            self._store().put({"id": "@agentenv/universe-real"})

    def test_a_bare_at_is_refused_too(self):
        with pytest.raises(ValueError, match=r"reserved '@' prefix"):
            self._store().put({"id": "@"})

    @pytest.mark.parametrize("entity_id", [
        "universe-real",
        "morrowline-live-v0/attendee-export.yaml",   # `/` is load-bearing in artifact ids
        "user@example.com",                          # `@` anywhere but the front is fine
    ])
    def test_every_id_shape_already_in_use_still_writes(self, entity_id):
        assert self._store().put({"id": entity_id}) == 1
