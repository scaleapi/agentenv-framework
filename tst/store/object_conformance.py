"""Backend-neutral ObjectStore conformance assertions.

Each function takes ``(store, ns)`` — ``ns`` is a key-prefix that namespaces the
case's objects (empty for the temp-dir local store; a unique prefix for the
shared S3 bucket). Both `S3ObjectStore` and `LocalFilesystemObjectStore` must
pass every case in ``CASES`` — that shared pass is the "it generalizes" proof.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from agent_env.store import ObjectAlreadyExistsError


def put_get_roundtrip(store, ns):
    url = store.put(f"{ns}a/b.json", b"hello", content_type="application/json")
    assert store.get(url) == b"hello"


def put_is_write_once(store, ns):
    store.put(f"{ns}dup", b"one")
    with pytest.raises(ObjectAlreadyExistsError):
        store.put(f"{ns}dup", b"two")


def put_allow_overwrite_replaces(store, ns):
    store.put(f"{ns}mut", b"one")
    url = store.put(f"{ns}mut", b"two", allow_overwrite=True)
    assert store.get(url) == b"two"


def put_file_roundtrip(store, ns):
    path = _write_temp(b"filedata")
    try:
        url = store.put_file(f"{ns}f/data.bin", path)
        assert store.get(url) == b"filedata"
    finally:
        os.unlink(path)


def put_file_is_write_once(store, ns):
    path = _write_temp(b"x")
    try:
        store.put_file(f"{ns}once", path)
        with pytest.raises(ObjectAlreadyExistsError):
            store.put_file(f"{ns}once", path)
    finally:
        os.unlink(path)


def exists_reflects_writes(store, ns):
    assert store.exists(f"{ns}k") is False
    store.put(f"{ns}k", b"v")
    assert store.exists(f"{ns}k") is True


def list_and_read_by_key(store, ns):
    store.put(f"{ns}p/a.txt", b"A")
    store.put(f"{ns}p/sub/b.txt", b"B")
    store.put(f"{ns}other/c.txt", b"C")
    assert set(store.list(f"{ns}p/")) == {f"{ns}p/a.txt", f"{ns}p/sub/b.txt"}
    assert store.read(f"{ns}p/a.txt") == b"A"
    assert store.read(f"{ns}p/sub/b.txt") == b"B"


def object_url_matches_put(store, ns):
    url = store.put(f"{ns}k/x", b"v")
    assert store.object_url(f"{ns}k/x") == url
    assert store.get(url) == b"v"


def get_object_key_inverts_object_url(store, ns):
    url = store.put(f"{ns}inv/x.json", b"v")
    assert store.get_object_key(url) == f"{ns}inv/x.json"
    assert store.object_url(store.get_object_key(url)) == url


def download_to_file_streams_to_dest(store, ns):
    url = store.put(f"{ns}dl/blob.bin", b"streamed")
    tmp = tempfile.mkdtemp()
    try:
        dest = os.path.join(tmp, "nested", "out.bin")
        store.download_to_file(url, dest)
        assert Path(dest).read_bytes() == b"streamed"
    finally:
        shutil.rmtree(tmp)


def get_object_metadata_reflects_object(store, ns):
    assert store.get_object_metadata(f"{ns}md/absent") is None
    store.put(f"{ns}md/x", b"hello", content_type="application/json")
    md = store.get_object_metadata(f"{ns}md/x")
    assert md is not None
    assert md.size == 5
    assert md.last_modified is not None


def put_file_at_roundtrips(store, ns):
    path = _write_temp(b"at-data")
    try:
        url = store.put_file_at(store.object_url(f"{ns}at/f.bin"), path)
        assert store.get(url) == b"at-data"
    finally:
        os.unlink(path)


def get_object_metadata_at_matches_key_variant(store, ns):
    assert store.get_object_metadata_at(store.object_url(f"{ns}mdat/absent")) is None
    store.put(f"{ns}mdat/x", b"hello")
    at = store.get_object_metadata_at(store.object_url(f"{ns}mdat/x"))
    assert at is not None
    assert at.size == store.get_object_metadata(f"{ns}mdat/x").size == 5


def list_at_matches_list(store, ns):
    store.put(f"{ns}lp/a.txt", b"A")
    store.put(f"{ns}lp/sub/b.txt", b"B")
    store.put(f"{ns}lpother/c.txt", b"C")
    got = set(store.list_at(store.object_url(f"{ns}lp/")))
    assert got == {store.object_url(k) for k in store.list(f"{ns}lp/")}


def _write_temp(data: bytes) -> str:
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return path


CASES = [
    put_get_roundtrip,
    put_is_write_once,
    put_allow_overwrite_replaces,
    put_file_roundtrip,
    put_file_is_write_once,
    exists_reflects_writes,
    list_and_read_by_key,
    object_url_matches_put,
    get_object_key_inverts_object_url,
    download_to_file_streams_to_dest,
    get_object_metadata_reflects_object,
    put_file_at_roundtrips,
    get_object_metadata_at_matches_key_variant,
    list_at_matches_list,
]
