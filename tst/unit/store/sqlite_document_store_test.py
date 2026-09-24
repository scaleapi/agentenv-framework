"""LocalSqliteDocumentStore runs the full DocumentStore conformance suite.

Fast tier — no Mongo, no network (stdlib sqlite over a temp file), so the same
assertions that prove Mongo parity also give quick backend-neutral coverage.
"""

from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_env.store import Filter, LocalSqliteDocumentStore, UpdateSpec, compare_and_swap
from tst.store import conformance


@pytest.fixture
def store_coll(tmp_path):
    return LocalSqliteDocumentStore(str(tmp_path / "docstore.db")), "coll"


@pytest.mark.parametrize("case", conformance.CASES, ids=lambda c: c.__name__)
def test_conformance(case, store_coll):
    case(*store_coll)


def test_threaded_cas_converges(store_coll):
    store, coll = store_coll
    store.ensure_index(coll, ["id"], unique=True)
    store.insert(coll, {"id": "x", "rev": 0, "steps": []})
    n = 20

    def add(k):
        return compare_and_swap(
            store, coll, Filter.of(id="x"),
            lambda doc: UpdateSpec(add_to_set={"steps": [k]}),
            counter_field="rev",
        )

    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(add, range(n)))

    doc = store.find_one(coll, Filter.of(id="x"))
    assert sorted(doc["steps"]) == list(range(n))  # every writer landed, none lost
    assert doc["rev"] == n


def test_reader_sees_a_collection_created_by_another_connection(tmp_path):
    """A long-lived reader must see a collection another connection creates after it opened,
    without reopening. Covers several primitives, not one."""
    path = str(tmp_path / "shared.db")
    reader = LocalSqliteDocumentStore(path)
    writer = LocalSqliteDocumentStore(path)   # a separate connection to the same file

    for coll in ("a2a_agents", "tasks", "evals"):
        assert reader.find_one(coll, Filter.of(id="x")) is None       # collection absent so far
        writer.insert(coll, {"id": "x", "version": 1})                # creates docs_<coll> + a row
        got = reader.find_one(coll, Filter.of(id="x"))
        assert got is not None and got["id"] == "x", f"{coll}: cross-connection write not seen"
