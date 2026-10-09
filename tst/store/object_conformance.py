"""Backend-neutral ObjectStore conformance assertions.

Each function takes ``(store, ns)`` — ``ns`` is a key-prefix that namespaces the
case's objects (empty for the temp-dir local store; a unique prefix for a
shared bucket). Every built-in store must pass every case in ``CASES`` — that
shared pass is the "it generalizes" proof.

A store that sets ``supports_transfer_grants`` must also pass ``GRANT_CASES``, except a
namespace case its ``issue_upload_policy`` declines with ``GrantUnavailableError``. They send
the grants to the provider over HTTPS, so they run against a real store only.
"""

import asyncio
import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import httpx
import pytest
from agentenv_protocol.transfers import Uploaded, upload

from agent_env.store import NotFoundError, ObjectAlreadyExistsError, ObjectNotFoundError, UploadFailedError


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


def open_streams_the_object(store, ns):
    url = store.put(f"{ns}open/blob.bin", b"streamed bytes")
    with contextlib.closing(store.open(url)) as body:
        assert body.read(8) + body.read() == b"streamed bytes"


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


def owns_its_objects_and_prefixes(store, ns):
    url = store.put(f"{ns}own/x.bin", b"v")
    assert store.owns(url)
    assert store.owns(store.object_url(f"{ns}own/"))
    assert not store.owns(f"https://example.test/{ns}own/x.bin")


def missing_object_reads_raise_object_not_found(store, ns):
    """A key with no object (absent, a prefix, below an object) reads as missing everywhere."""
    store.put(f"{ns}nf/dir/obj.bin", b"v")
    assert store.get_object_metadata(f"{ns}nf/dir") is None
    assert not store.exists(f"{ns}nf/dir")
    tmp = tempfile.mkdtemp()
    try:
        for key in (f"{ns}nf/absent.bin", f"{ns}nf/dir", f"{ns}nf/dir/obj.bin/below"):
            url = store.object_url(key)
            with pytest.raises(ObjectNotFoundError) as raised:
                store.get(url)
            assert isinstance(raised.value, NotFoundError) and isinstance(raised.value, FileNotFoundError)
            with pytest.raises(ObjectNotFoundError):
                store.open(url)
            with pytest.raises(ObjectNotFoundError):
                store.download_to_file(url, os.path.join(tmp, "out.bin"))
    finally:
        shutil.rmtree(tmp)


def read_grant_fetches_the_object(store, ns):
    url = store.put(f"{ns}grant/read.bin", b"granted")
    grant = store.issue_read_grant(url)
    response = httpx.get(grant.url, headers=grant.headers)
    assert response.status_code == 200
    assert response.content == b"granted"


def write_grant_uploads_with_the_signed_content_type(store, ns):
    url = store.object_url(f"{ns}grant/write.json")
    grant = store.issue_write_grant(url, media_type="application/json", max_bytes=64)
    response = httpx.put(grant.url, headers=grant.headers, content=b"{}")
    assert response.is_success
    assert store.get(url) == b"{}"
    assert store.get_object_metadata_at(url).content_type == "application/json"


def namespace_grant_confines_uploads_to_its_root(store, ns):
    policy = store.issue_upload_policy(
        store.object_url(f"{ns}grant/namespace/"), max_object_bytes=8, expires_in=600
    )
    root = store.get_object_key(store.object_url(f"{ns}grant/namespace")).rstrip("/")

    def post(path: str, data: bytes) -> httpx.Response:
        write = policy.write
        return httpx.post(
            write.url,
            data={"Content-Type": "application/octet-stream", **write.fields, write.path_field: path},
            files={write.file_field: ("object", data, "application/octet-stream")},
            headers=write.headers,
        )

    assert post(f"{root}/000000", b"inside").is_success
    assert store.read(f"{ns}grant/namespace/000000") == b"inside"
    assert post(f"{root}-sibling/000000", b"escaped").is_client_error
    assert not store.exists(f"{ns}grant/namespace-sibling/000000")
    assert post(f"{root}/000001", b"oversized").is_client_error
    assert not store.exists(f"{ns}grant/namespace/000001")


def object_write_completes_through_its_grant(store, ns):
    url = store.object_url(f"{ns}grant/object.zip")
    with store.begin_write(url, media_type="application/zip", max_bytes=1024, kinds=_OBJECT_KINDS, expires_in=900) as write:
        write.complete(asyncio.run(upload(write.grant, b"PK\x03\x04object")))
    assert store.get(url) == b"PK\x03\x04object"
    assert store.get_object_metadata_at(url).content_type == "application/zip"


def a_misreported_object_write_is_refused(store, ns):
    url = store.object_url(f"{ns}grant/misreported.zip")
    with store.begin_write(url, media_type="application/zip", max_bytes=1024, kinds=_OBJECT_KINDS, expires_in=900) as write:
        sent = asyncio.run(upload(write.grant, b"data"))
        with pytest.raises(UploadFailedError):
            write.complete(Uploaded(size_bytes=sent.size_bytes + 1))


def an_abandoned_object_write_leaves_no_object(store, ns):
    url = store.object_url(f"{ns}grant/abandoned.zip")
    with store.begin_write(url, media_type="application/zip", max_bytes=1024, kinds=_OBJECT_KINDS, expires_in=900):
        pass
    assert not store.exists(f"{ns}grant/abandoned.zip")


_OBJECT_KINDS = frozenset({"http-put", "http-put-parts"})


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
    open_streams_the_object,
    get_object_metadata_reflects_object,
    put_file_at_roundtrips,
    get_object_metadata_at_matches_key_variant,
    list_at_matches_list,
    owns_its_objects_and_prefixes,
    missing_object_reads_raise_object_not_found,
]

GRANT_CASES = [
    read_grant_fetches_the_object,
    write_grant_uploads_with_the_signed_content_type,
    namespace_grant_confines_uploads_to_its_root,
    object_write_completes_through_its_grant,
    a_misreported_object_write_is_refused,
    an_abandoned_object_write_leaves_no_object,
]
