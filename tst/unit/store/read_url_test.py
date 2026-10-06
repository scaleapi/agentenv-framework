"""A remote party is handed an HTTPS URL for an object: a read grant where the store's grants reach it, else a
URL the store signs, lasting the store's grant lifetime or longer when asked."""

from datetime import UTC, datetime, timedelta

import pytest
from agentenv_protocol.transfers import HttpGetGrant

from agent_env.store.object_store import DEFAULT_GRANT_LIFETIME_SECONDS, read_url
from tst.util.granting_object_store import GRANT_ORIGIN, GrantingObjectStore

SIGNED_ORIGIN = "https://signed.example.test"


class _Signing(GrantingObjectStore):
    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.signed: list[int] = []

    def signed_get_url(self, object_url: str, expires_in: int = 3600) -> str | None:
        self.signed.append(expires_in)
        return f"{SIGNED_ORIGIN}/{self.get_object_key(object_url)}"


class _HeaderGrants(_Signing):
    def issue_read_grant(self, object_url: str, *, expires_in: int | None = None) -> HttpGetGrant:
        return HttpGetGrant(
            kind="http-get", url=f"{GRANT_ORIGIN}/x", expires_at=datetime.now(UTC) + timedelta(hours=1),
            headers={"x-token": "t"},
        )


@pytest.fixture
def signing(tmp_path):
    return _Signing(str(tmp_path))


def test_a_grant_where_the_stores_grants_reach(tmp_path):
    store = GrantingObjectStore(str(tmp_path))
    url = store.put("files/a.png", b"png")

    assert read_url(store, url, sandbox_type="local", lasting=60) == f"{GRANT_ORIGIN}/files/a.png?sig=read"


def test_a_signed_url_where_they_do_not(signing):
    url = signing.put("files/a.png", b"png")

    assert read_url(signing, url, sandbox_type="modal", lasting=60) == f"{SIGNED_ORIGIN}/files/a.png"
    assert signing.granted == []


def test_none_where_the_store_neither_grants_to_it_nor_signs(tmp_path):
    store = GrantingObjectStore(str(tmp_path))
    url = store.put("files/a.png", b"png")

    assert read_url(store, url, sandbox_type="modal", lasting=60) is None


def test_a_grant_a_url_alone_cannot_carry_is_signed_instead(tmp_path):
    store = _HeaderGrants(str(tmp_path))
    url = store.put("files/a.png", b"png")

    assert read_url(store, url, sandbox_type="local", lasting=60) == f"{SIGNED_ORIGIN}/files/a.png"


@pytest.mark.parametrize("lasting, expected", [
    (60, DEFAULT_GRANT_LIFETIME_SECONDS), (2 * DEFAULT_GRANT_LIFETIME_SECONDS, 2 * DEFAULT_GRANT_LIFETIME_SECONDS),
])
def test_it_lasts_the_grant_lifetime_or_longer_when_asked(signing, lasting, expected):
    url = signing.put("files/a.png", b"png")

    read_url(signing, url, sandbox_type="modal", lasting=lasting)

    assert signing.signed == [expected]
