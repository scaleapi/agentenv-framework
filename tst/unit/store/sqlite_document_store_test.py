"""LocalSqliteDocumentStore runs the full DocumentStore conformance suite.

Fast tier — no Mongo, no network (stdlib sqlite over a temp file), so the same
assertions that prove Mongo parity also give quick backend-neutral coverage.
"""

import random
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent_env.store import (
    AbsentOrNull,
    Eq,
    Exists,
    Filter,
    Gte,
    In,
    LocalSqliteDocumentStore,
    Lte,
    LteOrAbsent,
    Ne,
    Sort,
    SortKey,
    UpdateSpec,
    VersionedEntityStore,
    compare_and_swap,
)
from agent_env.store.document_store import sqlite_document_store
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


def test_processes_opening_a_new_database_at_once_all_write(tmp_path):
    path = tmp_path / "new" / "shared.db"
    code = (
        "import sys\n"
        "from agent_env.store import LocalSqliteDocumentStore\n"
        "store = LocalSqliteDocumentStore(sys.argv[1])\n"
        "for i in range(50):\n"
        "    store.insert('coll', {'id': f'{sys.argv[2]}-{i}'})\n"
    )
    # nosemgrep: dangerous-subprocess-use-audit -- argv is sys.executable + the literal `code`; no external input
    writers = [subprocess.Popen([sys.executable, "-c", code, str(path), name], stderr=subprocess.PIPE) for name in "abcd"]
    for writer in writers:
        _, stderr = writer.communicate(timeout=120)
        assert writer.returncode == 0, stderr.decode()
    assert len(LocalSqliteDocumentStore(str(path)).query("coll", Filter.of())) == 200


_KEY = "docs_coll_id_version_unique"
_INDEXED_READS = {
    "find_one(id)": (lambda s: s.find_one("coll", Filter.of(id="a")), f"{_KEY} (<expr>=?)"),
    "find_one(id, version)": (
        lambda s: s.find_one("coll", Filter.of(id="a", version=1)), f"{_KEY} (<expr>=? AND <expr>=?)",
    ),
    "query(id)": (lambda s: s.query("coll", Filter.of(id="a"), sort=Sort.by("version")), f"{_KEY} (<expr>=?)"),
    "count(id)": (lambda s: s.count("coll", Filter.of(id="a")), f"{_KEY} (<expr>=?)"),
    "next_version": (lambda s: VersionedEntityStore(s, "coll", dict, dict).next_version("a"), f"{_KEY} (<expr>=?)"),
    "update": (
        lambda s: s.update("coll", Filter.of(id="a", version=1), UpdateSpec(set={"x": 2})), f"{_KEY} (<expr>=? AND",
    ),
    "update_one_and_get": (
        lambda s: s.update_one_and_get("coll", Filter.of(id="a").where("x", Ne(0)), UpdateSpec(inc={"x": 1})),
        f"{_KEY} (<expr>=?)",
    ),
    "replace": (
        lambda s: s.replace("coll", Filter.of(id="a", version=1), {"id": "a", "version": 1}), f"{_KEY} (<expr>=? AND",
    ),
    "delete": (lambda s: s.delete("coll", Filter.of(id="b", version=1)), f"{_KEY} (<expr>=? AND <expr>=?)"),
    "the unique index on a tie": (lambda s: s.find_one("coll", Filter.of(id="a", type="file")), f"{_KEY} (<expr>=?)"),
    "a secondary index": (lambda s: s.query("coll", Filter.of(type="file")), "docs_coll_type_index (<expr>=?)"),
    "the longest pinned prefix": (
        lambda s: s.query("coll", Filter.of(task_id="t", created_at_utc="c")),
        "docs_coll_task_id_created_at_utc_index (<expr>=? AND <expr>=?)",
    ),
}


@pytest.mark.parametrize("read", _INDEXED_READS.values(), ids=_INDEXED_READS.keys())
def test_equality_on_an_index_prefix_searches_the_index(read, store_coll):
    store, coll = store_coll
    _indexed(store, coll)
    call, plan = read
    plans = _plans(store, lambda: call(store))
    assert plans and all(f"USING INDEX {plan}" in p for p in plans), plans


@pytest.mark.parametrize("filter", [
    Filter.of(x=1),                                         # no index
    Filter.of(version=1),                                   # not an index's leading field
    Filter().where("id", Eq(None)),                         # also matches an absent id
    Filter().where("id", Eq(["a"])),                        # also matches an array holding ["a"]
    Filter().where("id", Eq(1.5)),
    Filter().where("id", In(["a"])),
    Filter().where("a2a_tasks.a2a_task_id", Eq("t")),       # a dotted path steps through arrays
], ids=repr)
def test_other_filters_scan(filter, store_coll):
    store, coll = store_coll
    _indexed(store, coll)
    assert _plans(store, lambda: store.query(coll, filter)) == ["SCAN docs_coll"]


def test_a_reader_searches_an_index_another_connection_creates(tmp_path):
    path = str(tmp_path / "shared.db")
    reader, writer = LocalSqliteDocumentStore(path), LocalSqliteDocumentStore(path)
    writer.insert("coll", {"id": "a", "version": 1})
    read = lambda: reader.find_one("coll", Filter.of(id="a"))
    assert _plans(reader, read) == ["SCAN docs_coll"]
    writer.ensure_index("coll", ["id", "version"], unique=True)
    assert "USING INDEX docs_coll_id_version_unique (<expr>=?)" in _plans(reader, read)[0]


def test_indexed_reads_and_writes_match_a_full_scan(tmp_path):
    """Random documents and operations give the same results, in the same order, as the same store with no index."""
    rng = random.Random(3205)
    indexed = LocalSqliteDocumentStore(str(tmp_path / "indexed.db"))
    scanned = LocalSqliteDocumentStore(str(tmp_path / "scanned.db"))
    for fields in (["id", "version"], ["type"], ["k", "id"], ["a.b"]):
        indexed.ensure_index("coll", fields)
    for _ in range(300):
        doc = _random_doc(rng)
        indexed.insert("coll", doc)
        scanned.insert("coll", doc)
    statements = []
    indexed._conn.set_trace_callback(statements.append)
    for _ in range(1500):
        op, call = _random_op(rng)
        assert _outcome(call, indexed) == _outcome(call, scanned), op
    assert indexed.query("coll", Filter()) == scanned.query("coll", Filter())
    searched = [sql for sql in statements if sql.startswith("SELECT rowid, doc") and " WHERE " in sql]
    assert len(searched) > 500  # the pushdown was exercised, not skipped


def test_a_write_the_busy_timeout_gives_up_on_names_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite_document_store, "_BUSY_TIMEOUT_SECONDS", 0.05)
    path = tmp_path / "locked.db"
    store = LocalSqliteDocumentStore(str(path))
    store.insert("coll", {"id": "a"})
    holder = sqlite3.connect(str(path), isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    insert = lambda: store.insert("coll", {"id": "b"})
    update = lambda: store.update("coll", Filter.of(id="a"), UpdateSpec(set={"x": 1}))
    for write in (insert, update):
        with pytest.raises(sqlite_document_store.DatabaseLockedError, match=f"{path} is locked: another process"):
            write()
    holder.execute("ROLLBACK")
    assert update() == 1


def test_a_first_open_the_busy_timeout_gives_up_on_names_the_store(tmp_path, monkeypatch):
    monkeypatch.setattr(sqlite_document_store, "_BUSY_TIMEOUT_SECONDS", 0.05)
    path = tmp_path / "locked.db"
    holder = sqlite3.connect(str(path), isolation_level=None)
    holder.execute("CREATE TABLE t (x)")
    holder.execute("BEGIN EXCLUSIVE")  # a rollback-journal database, so the switch to WAL waits on this lock
    with pytest.raises(sqlite_document_store.DatabaseLockedError, match=f"{path} is locked: another process"):
        LocalSqliteDocumentStore(str(path)).count("coll", Filter())
    holder.execute("ROLLBACK")
    assert LocalSqliteDocumentStore(str(path)).count("coll", Filter()) == 0


class _Connection:
    """Fails its first ``failures`` statements with ``code``."""

    def __init__(self, failures, code):
        self.failures, self.code, self.calls = failures, code, 0

    def execute(self, sql):
        self.calls += 1
        if self.calls <= self.failures:
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = self.code
            raise error


def test_the_wal_switch_waits_out_a_connection_holding_the_lock():
    conn = _Connection(failures=3, code=sqlite3.SQLITE_BUSY)
    sqlite_document_store._enable_wal(conn)
    assert conn.calls == 4


def test_the_wal_switch_raises_other_errors_and_a_lock_held_too_long(monkeypatch):
    conn = _Connection(failures=1, code=sqlite3.SQLITE_IOERR)
    with pytest.raises(sqlite3.OperationalError):
        sqlite_document_store._enable_wal(conn)
    assert conn.calls == 1

    monkeypatch.setattr(sqlite_document_store, "_BUSY_TIMEOUT_SECONDS", 0.05)
    with pytest.raises(sqlite3.OperationalError):
        sqlite_document_store._enable_wal(_Connection(failures=10**6, code=sqlite3.SQLITE_BUSY))


_FIELDS = ["id", "version", "type", "k", "x", "tags", "a.b", "missing"]
_VALUES = [
    "a", "b", "A", "", "1", "[1]", '{"x": 1}', "\u00e9", "\U0001f600", "\ud800", "a\x00b",
    0, 1, -1, 2, 10**20, 1.0, 2.5, True, False, None, {"x": 1}, [1], ["a"], [],
]
# Indexed fields hold scalars or objects, as agent-env writes them; an array under one is out of scope.
_KEY_VALUES = [v for v in _VALUES if not isinstance(v, list)]


def _random_doc(rng):
    doc = {}
    for field in ("id", "version", "type", "k", "x"):
        if rng.random() < 0.85:
            doc[field] = rng.choice(["a", "b", "1", 1, 2, None] if field == "id" else _KEY_VALUES)
    if rng.random() < 0.5:
        doc["tags"] = rng.sample(_VALUES, rng.randint(0, 3))
    shape = rng.random()
    if shape < 0.3:
        doc["a"] = {"b": rng.choice(_VALUES)}
    elif shape < 0.6:
        doc["a"] = [{"b": rng.choice(_VALUES)} for _ in range(rng.randint(0, 3))]
    return doc


def _random_filter(rng, values=_VALUES):
    filter = Filter()
    for field in rng.sample(["id", "version", "type", "k"], rng.randint(0, 3)):
        filter = filter.where(field, Eq(rng.choice(values)))
    for _ in range(rng.choice([0, 0, 1, 2])):
        value = rng.choice(values)
        pred = rng.choice([Eq(value), Ne(value), In([value, rng.choice(values)]), Gte(value), Lte(value),
                           Exists(rng.random() < 0.5), LteOrAbsent(value), AbsentOrNull()])
        filter = filter.where(rng.choice(_FIELDS), pred)
    return filter


def _random_op(rng):
    filter, key_filter, doc = _random_filter(rng), _random_filter(rng, _KEY_VALUES), _random_doc(rng)
    sort = rng.choice([None, Sort.by(rng.choice(_FIELDS), rng.random() < 0.5),
                       Sort((SortKey("type", False), SortKey("version")))])
    limit, offset = rng.choice([None, 1, 3]), rng.choice([None, 1])
    update = UpdateSpec(set={rng.choice(["x", "type", "version"]): rng.choice(_KEY_VALUES)}, inc={"n": 1})
    name, call = rng.choice([
        ("find_one", lambda s: s.find_one("coll", filter, sort)),
        ("query", lambda s: s.query("coll", filter, sort, limit, offset)),
        ("count", lambda s: s.count("coll", filter)),
        ("update", lambda s: s.update("coll", filter, update)),
        ("upsert", lambda s: s.update("coll", key_filter, update, upsert=True)),  # synthesizes from the filter
        ("update_one_and_get", lambda s: s.update_one_and_get("coll", filter, update, return_after=bool(limit))),
        ("replace", lambda s: s.replace("coll", filter, doc)),
        ("delete", lambda s: s.delete("coll", filter)),
    ])
    return f"{name} {key_filter if name == 'upsert' else filter} {sort}", call


def _outcome(call, store):
    try:
        return call(store)
    except TypeError as e:  # sort_docs can't order unlike types, with or without an index
        return repr(e)


def _indexed(store, coll):
    store.ensure_index(coll, ["id", "version"], unique=True)
    store.ensure_index(coll, ["type"])
    store.ensure_index(coll, ["task_id"])
    store.ensure_index(coll, ["task_id", "created_at_utc"])
    store.ensure_index(coll, ["a2a_tasks.a2a_task_id"])
    store.insert(coll, {"id": "a", "version": 1, "type": "file", "x": 1})
    store.insert(coll, {"id": "a", "version": 2, "type": "file", "x": 1})


def _plans(store, call) -> list[str]:
    """The query plan of each row read ``call`` makes."""
    statements = []
    store._conn.set_trace_callback(statements.append)
    try:
        call()
    finally:
        store._conn.set_trace_callback(None)
    reads = [sql for sql in statements if sql.startswith("SELECT rowid, doc")]
    return [" ".join(row[3] for row in store._conn.execute(f"EXPLAIN QUERY PLAN {sql}")) for sql in reads]
