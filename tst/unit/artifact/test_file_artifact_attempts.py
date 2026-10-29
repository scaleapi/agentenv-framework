"""FileArtifact.put / put_bytes work around objects already on the version path.

Regression guard: both upload to the write-once path of the version they reserve,
before the document is inserted. When the insert then failed, the orphaned object
stayed on that path and every retry collided with it (ObjectAlreadyExistsError);
two writers reserving the same version collided the same way. The upload now
retries past a published version and falls back to a prefix of the attempt's own
past an orphan. Network-free on the local stores.
"""

import threading

import pytest

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.store import ArtifactStore
from agent_env.store.object_store.local.store import LocalFilesystemObjectStore


@pytest.fixture
def fail_first_insert(monkeypatch):
    """Make the first ArtifactStore.put_document call raise, as a failed insert does."""
    real = ArtifactStore.put_document
    calls = {"n": 0}

    def flaky(self, artifact):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated document insert failure")
        return real(self, artifact)

    monkeypatch.setattr(ArtifactStore, "put_document", flaky)
    return calls


def test_put_bytes_retry_after_document_insert_failure(local_stores, fail_first_insert):
    with pytest.raises(RuntimeError, match="simulated document insert failure"):
        FileArtifact.put_bytes("f1", description="d", filename="a.txt", content=b"hello")

    art = FileArtifact.put_bytes("f1", description="d", filename="a.txt", content=b"hello")

    assert art.version == 1
    assert art.load() == b"hello"


def test_put_retry_after_document_insert_failure(local_stores, tmp_path, fail_first_insert):
    path = tmp_path / "a.txt"
    path.write_bytes(b"from file")
    with pytest.raises(RuntimeError, match="simulated document insert failure"):
        FileArtifact.put("f2", description="d", file_path=str(path))

    art = FileArtifact.put("f2", description="d", file_path=str(path))

    assert art.version == 1
    assert art.load() == b"from file"


def _barrier_uploads(monkeypatch, method_name):
    """Hold the first two uploads in ``method_name`` until both are in flight at once."""
    real = getattr(LocalFilesystemObjectStore, method_name)
    barrier = threading.Barrier(2)
    calls = {"n": 0}
    lock = threading.Lock()

    def gated(self, *args, **kwargs):
        with lock:
            calls["n"] += 1
            gated_call = calls["n"] <= 2
        if gated_call:
            barrier.wait(timeout=10)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(LocalFilesystemObjectStore, method_name, gated)


def _put_two_concurrently(put_one):
    results, errors = {}, {}

    def run(slot):
        try:
            results[slot] = put_one(slot)
        except Exception as e:  # collected and asserted below, not swallowed
            errors[slot] = e

    threads = [threading.Thread(target=run, args=(slot,)) for slot in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors
    return results


def test_concurrent_put_bytes_uploads_do_not_collide(local_stores, monkeypatch):
    _barrier_uploads(monkeypatch, "put")

    results = _put_two_concurrently(
        lambda slot: FileArtifact.put_bytes(
            "conc", description="d", filename="a.txt", content=f"content-{slot}".encode()
        )
    )

    assert sorted(art.version for art in results.values()) == [1, 2]
    assert len({art.object_url for art in results.values()}) == 2
    for slot, art in results.items():
        assert art.load() == f"content-{slot}".encode()


def test_concurrent_put_uploads_do_not_collide(local_stores, tmp_path, monkeypatch):
    _barrier_uploads(monkeypatch, "put_file_at")
    paths = []
    for slot in (0, 1):
        path = tmp_path / f"a{slot}.txt"
        path.write_bytes(f"file-{slot}".encode())
        paths.append(path)

    results = _put_two_concurrently(
        lambda slot: FileArtifact.put("concf", description="d", file_path=str(paths[slot]))
    )

    assert sorted(art.version for art in results.values()) == [1, 2]
    assert len({art.object_url for art in results.values()}) == 2
    for slot, art in results.items():
        assert art.load() == f"file-{slot}".encode()


def test_a_second_put_does_not_overwrite_the_first_object(local_stores, tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"first")
    first = FileArtifact.put("seq", description="d", file_path=str(path))
    path.write_bytes(b"second")
    second = FileArtifact.put("seq", description="d", file_path=str(path))

    assert (first.version, second.version) == (1, 2)
    assert first.object_url != second.object_url
    assert first.load() == b"first"
    assert second.load() == b"second"
