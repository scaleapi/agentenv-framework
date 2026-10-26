"""put_document: a named version must be the id's next, and version 0 is allocated, retrying the allocation race
(in-memory fake)."""

import logging

import pytest

from agent_env.artifact.artifact import Artifact
from agent_env.artifact.store import _MAX_VERSION_RETRIES, ArtifactStore
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.store import (
    ConcurrentModificationError,
    DuplicateKeyError,
    reset_config,
    set_document_store,
    set_object_store,
)
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


class _RacerTakesIt(_UniqueIndexStore):
    """A concurrent writer inserts the version this one allocated just before it does, ``races`` times."""

    def __init__(self, seed=None, races=1):
        super().__init__(seed)
        self.races = races

    def insert(self, collection, doc):
        if self.races:
            self.races -= 1
            self.docs.append({"id": doc["id"], "version": doc["version"], "type": "toy"})
        return super().insert(collection, doc)


@pytest.mark.parametrize("version", [3, 2, 5])
def test_a_named_version_other_than_the_next_writes_nothing(version):
    coll = _UniqueIndexStore(seed=[{"id": "x", "version": v, "type": "toy"} for v in (1, 2, 3)])

    with pytest.raises(ConcurrentModificationError, match=f"at version 3, so its next version is 4, not {version}"):
        _store_with(coll).put_document(_ToyArtifact(id="x", version=version))

    assert coll.insert_calls == 0


def test_a_named_version_another_write_takes_first_is_refused():
    coll = _RacerTakesIt(seed=[{"id": "x", "version": 1, "type": "toy"}])

    with pytest.raises(ConcurrentModificationError, match="another write took version 2"):
        _store_with(coll).put_document(_ToyArtifact(id="x", version=2))

    assert coll.insert_calls == 1


def test_an_allocated_version_another_write_takes_first_is_reallocated(caplog):
    coll = _RacerTakesIt(seed=[{"id": "x", "version": 1, "type": "toy"}])

    with caplog.at_level(logging.WARNING, logger="agent_env.artifact.store"):
        result = _store_with(coll).put_document(_ToyArtifact(id="x", version=0))

    assert result.version == 3 and coll.insert_calls == 2
    assert any("concurrent write" in r.message for r in caplog.records)


def test_reallocation_climbs_past_multiple_concurrent_writes():
    coll = _RacerTakesIt(seed=[{"id": "x", "version": 1, "type": "toy"}], races=2)

    result = _store_with(coll).put_document(_ToyArtifact(id="x", version=0))

    assert result.version == 4


def test_exhausts_retries_then_raises():
    class _AlwaysDup(_UniqueIndexStore):
        def insert(self, collection, doc):
            self.insert_calls += 1
            raise DuplicateKeyError("E11000 duplicate key")

    coll = _AlwaysDup()
    store = _store_with(coll)

    with pytest.raises(DuplicateKeyError):
        store.put_document(_ToyArtifact(id="x", version=0))

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
    orphaned with nothing pointing at it, so the reservation is enforced before the upload, at
    version allocation or ``check_writable``."""
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


def test_a_file_whose_version_another_writer_publishes_during_its_upload_is_still_published():
    """put_bytes keys its object by a version it reserves, then writes the document after the upload; a writer that
    publishes that version meanwhile moves the document to the next one rather than failing it."""
    objects = FakeObjectStore()
    docs = _UniqueIndexStore()
    real_put = objects.put

    def put_while_another_publishes(*args, **kwargs):
        url = real_put(*args, **kwargs)
        docs.docs.append({"id": "f", "version": 1, "type": "file"})
        return url

    objects.put = put_while_another_publishes
    set_document_store(docs)
    set_object_store(objects)
    try:
        published = FileArtifact.put_bytes(id="f", description="d", filename="f.txt", content=b"x")
    finally:
        reset_config()

    assert published.version == 2 and {d["version"] for d in docs.docs} == {1, 2}


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
