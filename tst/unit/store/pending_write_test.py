"""The one-URL write every store that issues write grants offers: begin_write wraps issue_write_grant, and
completing checks the object landed at the size the receiver reported."""

import pytest
from agentenv_protocol.transfers import HttpPartsPutGrant, HttpPutGrant, Uploaded

from agent_env.store import GrantUnavailableError, UploadFailedError
from tst.util.granting_object_store import GrantingObjectStore

KINDS = frozenset({"http-put", "http-put-parts"})


class _CappedStore(GrantingObjectStore):
    max_single_upload_bytes = 10


@pytest.fixture
def store(tmp_path):
    return GrantingObjectStore(str(tmp_path))


def test_a_receiver_taking_parts_gets_the_put_as_a_one_url_parts_grant(store):
    url = store.object_url("snapshots/github.zip")
    write = store.begin_write(url, media_type="application/zip", max_bytes=1024, kinds=KINDS)

    grant = write.grant.write
    assert isinstance(grant, HttpPartsPutGrant)
    assert (grant.part_bytes, len(grant.urls), write.grant.max_bytes) == (1024, 1, 1024)
    assert grant.headers == {"Content-Type": "application/zip"}
    assert (write.object_url, store.granted) == (url, [url])


def test_a_receiver_taking_only_a_put_gets_a_put_grant(store):
    write = store.begin_write(
        store.object_url("t.json"), media_type="application/json", max_bytes=64, kinds=frozenset({"http-put"})
    )
    assert isinstance(write.grant.write, HttpPutGrant)


def test_a_receiver_taking_no_object_grant_is_refused(store):
    with pytest.raises(GrantUnavailableError):
        store.begin_write(store.object_url("t.json"), media_type="application/json", max_bytes=64,
                          kinds=frozenset({"http-post-policy"}))


def test_one_upload_bounds_the_write_on_a_store_that_caps_it(tmp_path):
    store = _CappedStore(str(tmp_path))
    write = store.begin_write(store.object_url("t.zip"), media_type="application/zip", max_bytes=100, kinds=KINDS)
    assert write.grant.max_bytes == 10


def test_completing_checks_the_object_landed_at_the_reported_size(store):
    url = store.object_url("snapshots/github.zip")
    store.put("snapshots/github.zip", b"12345")

    store.begin_write(url, media_type="application/zip", max_bytes=64, kinds=KINDS).complete(Uploaded(size_bytes=5))
    with pytest.raises(UploadFailedError):
        store.begin_write(url, media_type="application/zip", max_bytes=64, kinds=KINDS).complete(Uploaded(size_bytes=6))
    with pytest.raises(UploadFailedError):
        store.begin_write(store.object_url("absent.zip"), media_type="application/zip", max_bytes=64,
                          kinds=KINDS).complete(Uploaded(size_bytes=1))


def test_leaving_an_unfinished_one_url_write_does_nothing(store):
    url = store.object_url("snapshots/github.zip")
    with store.begin_write(url, media_type="application/zip", max_bytes=64, kinds=KINDS):
        pass
    assert not store.exists("snapshots/github.zip")
