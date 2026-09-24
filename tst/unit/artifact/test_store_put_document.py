"""Unit tests for put_document's version-allocation race retry (in-memory fake)."""

import logging

import pytest

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.store import _MAX_VERSION_RETRIES, ArtifactStore
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.store import DuplicateKeyError, reset_config, set_document_store, set_object_store
from tst.unit.store.fakes import FakeDocumentStore, FakeObjectStore


class _ToyArtifact(Artifact):
    type: str = "toy"


class _UniqueIndexStore(FakeDocumentStore):
    """FakeDocumentStore that enforces the (id, version) unique index."""

    def __init__(self, seed=None):
        super().__init__()
        self.docs = [dict(d) for d in (seed or [])]
        self.insert_calls = 0

    def insert(self, collection, doc):
        self.insert_calls += 1
        if any(d.get("id") == doc.get("id") and d.get("version") == doc.get("version") for d in self.docs):
            raise DuplicateKeyError("E11000 duplicate key: id_version_unique")
        self.docs.append(dict(doc))


@pytest.fixture(autouse=True)
def _reset_config():
    yield
    reset_config()


def _store_with(doc_store) -> ArtifactStore:
    set_document_store(doc_store)
    return ArtifactStore()


def test_no_collision_preserves_version():
    coll = _UniqueIndexStore()
    result = _store_with(coll).put_document(_ToyArtifact(id="x", version=1))

    assert result.version == 1
    assert coll.insert_calls == 1
    assert coll.docs[0]["id"] == "x" and coll.docs[0]["version"] == 1


def test_version_zero_is_allocated_from_max():
    coll = _UniqueIndexStore(seed=[{"id": "x", "version": 3, "type": "toy"}])
    result = _store_with(coll).put_document(_ToyArtifact(id="x", version=0))

    assert result.version == 4
    assert coll.insert_calls == 1


def test_retries_and_reallocates_on_duplicate(caplog):
    """A concurrent writer already took v1; put_document must climb to v2."""
    coll = _UniqueIndexStore(seed=[{"id": "x", "version": 1, "type": "toy"}])
    store = _store_with(coll)

    with caplog.at_level(logging.WARNING, logger="agent_env.artifact.store"):
        result = store.put_document(_ToyArtifact(id="x", version=1))

    assert result.version == 2
    assert coll.insert_calls == 2
    assert {d["version"] for d in coll.docs} == {1, 2}
    assert any("concurrent write" in r.message for r in caplog.records)


def test_reallocation_climbs_past_multiple_concurrent_writes(caplog):
    """If several versions land between retries, next_version keeps climbing."""

    class _Racing(_UniqueIndexStore):
        def insert(self, collection, doc):
            # On collision, a racer grabs the next slot too, so +1 isn't enough.
            if any(d["id"] == doc["id"] and d["version"] == doc["version"] for d in self.docs):
                nxt = max(d["version"] for d in self.docs if d["id"] == doc["id"]) + 1
                self.docs.append({"id": doc["id"], "version": nxt, "type": "toy"})
            return super().insert(collection, doc)

    coll = _Racing(seed=[{"id": "x", "version": 1, "type": "toy"}])
    store = _store_with(coll)

    with caplog.at_level(logging.WARNING, logger="agent_env.artifact.store"):
        result = store.put_document(_ToyArtifact(id="x", version=1))

    assert result.version == 3


def test_exhausts_retries_then_raises():
    class _AlwaysDup(_UniqueIndexStore):
        def insert(self, collection, doc):
            self.insert_calls += 1
            raise DuplicateKeyError("E11000 duplicate key")

    coll = _AlwaysDup()
    store = _store_with(coll)

    with pytest.raises(DuplicateKeyError):
        store.put_document(_ToyArtifact(id="x", version=1))

    assert coll.insert_calls == _MAX_VERSION_RETRIES


def test_put_document_refuses_a_reserved_at_prefixed_id():
    """Artifacts do not go through `VersionedEntityStore.put` — they have their own
    version-allocation loop — so the reservation has to be stated on this path too. Artifacts
    are also the collection most likely to receive a third party's id."""
    set_document_store(_UniqueIndexStore())
    try:
        with pytest.raises(ValueError, match=r"reserved '@' prefix"):
            ArtifactStore().put_document(_ToyArtifact(id="@agentenv/shared-fixture"))
    finally:
        reset_config()


def test_a_reserved_id_is_refused_before_anything_is_uploaded():
    """Artifact helpers write remote data — an object to S3, an image to a registry — before
    they have a document to store. Refusing the id at `put_document` would leave that data
    orphaned with nothing pointing at it, so the reservation is enforced at version
    allocation, which every helper does first."""
    objects = FakeObjectStore()
    uploads = []
    real_put = objects.put
    objects.put = lambda *a, **k: (uploads.append(1), real_put(*a, **k))[1]
    set_document_store(_UniqueIndexStore())
    set_object_store(objects)
    try:
        with pytest.raises(ValueError, match=r"reserved '@' prefix"):
            FileArtifact.put_bytes(
                id="@agentenv/shared", description="d", filename="f.txt", content=b"x"
            )
        assert uploads == []

        FileArtifact.put_bytes(id="ordinary", description="d", filename="f.txt", content=b"x")
        assert len(uploads) == 1  # and an unreserved id is untouched by the guard
    finally:
        reset_config()


def test_set_field_cannot_rename_a_stored_artifact():
    """`set_field` writes an UpdateSpec straight to the store, so `field="id"` would both
    rename a document out from under the filter that found it and bypass the reservation."""
    set_document_store(_UniqueIndexStore())
    try:
        store = ArtifactStore()
        for field in ("id", "version", "type"):
            with pytest.raises(ValueError, match="identifies the document"):
                store.set_field("toy", "x", 1, field, "@agentenv/renamed")
        store.set_field("toy", "x", 1, "harbor_zip_cds_url", "s3://b/k")  # side channels still work
    finally:
        reset_config()
