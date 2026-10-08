"""MongoDocumentStore reports pymongo's duplicate-key error as the store's own, on every write, and holds
a caller for no longer than its wait on an index build."""

from __future__ import annotations

import logging
import threading
import time

import pytest
from pymongo.errors import DuplicateKeyError as PyMongoDuplicateKeyError
from pymongo.errors import NetworkTimeout, OperationFailure

from agent_env.store import DuplicateKeyError, Filter, UpdateSpec
from agent_env.store.document_store import mongo_document_store
from agent_env.store.document_store.mongo_document_store import MongoDocumentStore


class _Colliding:
    """A pymongo collection whose every write collides with a unique index."""

    def _collide(self, *args, **kwargs):
        raise PyMongoDuplicateKeyError("E11000 duplicate key error collection: db.coll index: id_version_unique")

    insert_one = update_one = find_one_and_update = replace_one = _collide


_MISSING = Filter.of(id="a", version=1, state="free")


@pytest.mark.parametrize(
    "write",
    [
        lambda store: store.insert("coll", {"id": "a", "version": 1}),
        lambda store: store.update("coll", _MISSING, UpdateSpec(set={"x": 1}), upsert=True),
        lambda store: store.update_one_and_get("coll", _MISSING, UpdateSpec(set={"x": 1}), upsert=True),
        lambda store: store.replace("coll", _MISSING, {"id": "a", "version": 1}, upsert=True),
    ],
    ids=["insert", "update-upsert", "update-one-and-get-upsert", "replace-upsert"],
)
def test_a_unique_violation_is_the_stores_duplicate_key_error(write):
    with pytest.raises(DuplicateKeyError, match="E11000"):
        write(MongoDocumentStore({"coll": _Colliding()}))


class _Indexing:
    """A pymongo collection whose index builds run until the test releases them; MongoDB lists a build once it
    has started it."""

    def __init__(self, existing: dict | None = None, error: Exception | None = None, started: bool = True) -> None:
        self.existing = existing or {"_id_": {"key": [("_id", 1)]}}
        self.error = error
        self.started = started
        self.release = threading.Event()
        self.asked: list[str] = []
        self.built: list[str] = []

    def index_information(self) -> dict:
        return dict(self.existing)

    def create_index(self, keys, *, name, **kwargs):
        self.asked.append(name)
        if self.started:
            self.existing[name] = {"key": keys}
        self.release.wait(5)
        if self.error is not None:
            raise self.error
        self.built.append(name)


def _builds() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("ensure-index-")]


def _finish_builds() -> None:
    for thread in _builds():
        thread.join(5)


@pytest.fixture
def short_wait(monkeypatch):
    monkeypatch.setattr(mongo_document_store, "_INDEX_BUILD_WAIT_SECONDS", 0.05)


@pytest.mark.parametrize(
    "existing",
    [{"id_version_index": {"key": [("other", 1)]}}, {"legacy": {"key": [("id", 1), ("version", 1)]}}],
    ids=["same-name", "same-keys"],
)
def test_an_index_already_listed_is_not_built_again(existing):
    coll = _Indexing(existing)
    MongoDocumentStore({"c": coll}).ensure_index("c", ["id", "version"])
    assert coll.asked == []


@pytest.mark.parametrize("code", [85, 86])
def test_a_build_within_the_wait_tolerates_an_equivalent_index(code):
    coll = _Indexing(error=OperationFailure("equivalent index exists", code=code))
    coll.release.set()
    MongoDocumentStore({"c": coll}).ensure_index("c", ["id"])
    assert coll.asked == ["id_index"]


def test_a_build_that_fails_within_the_wait_raises():
    coll = _Indexing(error=OperationFailure("cannot create index", code=67))
    coll.release.set()
    with pytest.raises(OperationFailure, match="cannot create index"):
        MongoDocumentStore({"c": coll}).ensure_index("c", ["id"])


def test_a_slow_build_is_left_to_finish_without_the_caller(short_wait, caplog):
    coll = _Indexing()
    with caplog.at_level(logging.INFO, logger=mongo_document_store.__name__):
        started = time.monotonic()
        MongoDocumentStore({"c": coll}).ensure_index("c", ["task_id", "created_at_utc"])
        waited = time.monotonic() - started
        assert coll.built == [] and all(t.daemon for t in _builds())
        coll.release.set()
        _finish_builds()
    assert waited < 2
    assert coll.built == ["task_id_created_at_utc_index"]
    assert "still building index task_id_created_at_utc_index" in caplog.text
    assert "finished building index task_id_created_at_utc_index" in caplog.text


def test_a_unique_index_is_awaited_however_long_it_builds(short_wait):
    coll = _Indexing()
    threading.Timer(0.3, coll.release.set).start()
    started = time.monotonic()
    MongoDocumentStore({"c": coll}).ensure_index("c", ["instance_id"], unique=True)
    assert time.monotonic() - started >= 0.3
    assert coll.built == ["instance_id_unique"]


@pytest.mark.parametrize(
    ("error", "level", "message"),
    [
        (OperationFailure("Index build failed", code=276), logging.ERROR, "did not build index id_index"),
        (NetworkTimeout("timed out"), logging.INFO, "Stopped waiting for index id_index"),
    ],
    ids=["build-failed", "client-timed-out"],
)
def test_a_build_that_ends_after_the_wait_is_logged_not_raised(short_wait, caplog, error, level, message):
    coll = _Indexing(error=error)
    with caplog.at_level(logging.INFO, logger=mongo_document_store.__name__):
        MongoDocumentStore({"c": coll}).ensure_index("c", ["id"])
        coll.release.set()
        _finish_builds()
    assert any(r.levelno == level and message in r.getMessage() for r in caplog.records)


def test_a_client_timeout_within_the_wait_leaves_a_started_build_to_the_server(caplog):
    coll = _Indexing(error=NetworkTimeout("timed out"))
    coll.release.set()
    with caplog.at_level(logging.INFO, logger=mongo_document_store.__name__):
        MongoDocumentStore({"c": coll}).ensure_index("c", ["id"])
    assert "still building index id_index" in caplog.text
    assert "Stopped waiting for index id_index" in caplog.text


def test_a_client_timeout_before_mongodb_started_the_build_raises():
    coll = _Indexing(error=NetworkTimeout("timed out"), started=False)
    coll.release.set()
    with pytest.raises(NetworkTimeout):
        MongoDocumentStore({"c": coll}).ensure_index("c", ["id"])
