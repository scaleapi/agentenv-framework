"""A remote party is handed an HTTPS URL for an object: a read grant where the store's grants reach it, else a
URL the store signs, lasting the store's grant lifetime or longer when asked."""

import pytest

from agent_env.store.object_store import DEFAULT_GRANT_LIFETIME_SECONDS
from agent_env.store.object_store.object_store import readable_url
from tst.util.granting_object_store import GRANT_ORIGIN, SIGNED_ORIGIN, GrantingObjectStore


def _put(store):
    return store.put("files/a.png", b"png")


def test_a_grant_where_the_stores_grants_reach(tmp_path):
    store = GrantingObjectStore(str(tmp_path))

    assert readable_url(store, _put(store), sandbox_type="local", expires_in=60) == f"{GRANT_ORIGIN}/files/a.png?sig=read"


def test_a_signed_url_where_they_do_not(tmp_path):
    store = GrantingObjectStore(str(tmp_path), signs=True)

    assert readable_url(store, _put(store), sandbox_type="modal", expires_in=60) == f"{SIGNED_ORIGIN}/files/a.png"
    assert store.granted == []


def test_none_where_the_store_neither_grants_to_it_nor_signs(tmp_path):
    store = GrantingObjectStore(str(tmp_path))

    assert readable_url(store, _put(store), sandbox_type="modal", expires_in=60) is None


def test_a_grant_a_url_alone_cannot_carry_is_signed_instead(tmp_path):
    store = GrantingObjectStore(str(tmp_path), signs=True, grant_headers={"x-token": "t"})

    assert readable_url(store, _put(store), sandbox_type="local", expires_in=60) == f"{SIGNED_ORIGIN}/files/a.png"


def test_a_grant_that_cannot_last_long_enough_is_signed_instead(tmp_path):
    store = GrantingObjectStore(str(tmp_path), signs=True, max_grant_seconds=DEFAULT_GRANT_LIFETIME_SECONDS)
    long = 2 * DEFAULT_GRANT_LIFETIME_SECONDS

    assert readable_url(store, _put(store), sandbox_type="local", expires_in=long) == f"{SIGNED_ORIGIN}/files/a.png"


@pytest.mark.parametrize("expires_in, expected", [
    (60, DEFAULT_GRANT_LIFETIME_SECONDS), (2 * DEFAULT_GRANT_LIFETIME_SECONDS, 2 * DEFAULT_GRANT_LIFETIME_SECONDS),
])
def test_it_lasts_the_grant_lifetime_or_longer_when_asked(tmp_path, expires_in, expected):
    store = GrantingObjectStore(str(tmp_path), signs=True)

    readable_url(store, _put(store), sandbox_type="modal", expires_in=expires_in)

    assert store.signed == [expected]
