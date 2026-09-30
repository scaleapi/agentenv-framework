"""Grant tokens: what one signs, it verifies, until the expiry; nothing else verifies."""

import pytest

from agent_env.store.object_store.local_grants.tokens import GrantClaims, GrantSigner, InvalidGrantError

NOW = 1_800_000_000


def _claims(**overrides) -> GrantClaims:
    fields = {"op": "put", "store": "s1", "expires": NOW + 60, "key": "a/b.json", "max_bytes": 64,
              "content_type": "application/json"}
    return GrantClaims(**{**fields, **overrides})


def test_a_signed_token_verifies_to_its_claims():
    signer = GrantSigner()
    claims = _claims()
    assert signer.verify(signer.sign(claims), now=NOW) == claims


def test_a_post_token_carries_its_prefix():
    signer = GrantSigner()
    claims = GrantClaims(op="post", store="s1", expires=NOW + 60, prefix="ns/", max_bytes=8)
    assert signer.verify(signer.sign(claims), now=NOW).prefix == "ns/"


def test_an_expired_token_is_refused():
    signer = GrantSigner()
    token = signer.sign(_claims(expires=NOW))
    with pytest.raises(InvalidGrantError, match="expired"):
        signer.verify(token, now=NOW)


def test_another_processs_token_is_refused():
    with pytest.raises(InvalidGrantError, match="signature"):
        GrantSigner().verify(GrantSigner().sign(_claims()), now=NOW)


@pytest.mark.parametrize("tamper", [
    lambda t: t.replace("v1.", "v2.", 1),
    lambda t: t.rsplit(".", 1)[0],
    lambda t: t[:-2] + ("AA" if not t.endswith("AA") else "BB"),
    lambda t: "v1." + GrantSigner().sign(_claims(key="other")).split(".")[1] + "." + t.rsplit(".", 1)[1],
    lambda t: t + "!",
])
def test_a_tampered_token_is_refused(tamper):
    signer = GrantSigner()
    with pytest.raises(InvalidGrantError):
        signer.verify(tamper(signer.sign(_claims())), now=NOW)


@pytest.mark.parametrize("fields", [
    {"op": "delete"},
    {"op": "post", "key": None, "prefix": "ns"},
    {"op": "post", "prefix": "ns/"},
    {"op": "get", "key": None, "prefix": "ns/"},
    {"max_bytes": None},
    {"max_bytes": -1},
    {"content_type": None},
    {"op": "get", "max_bytes": None},
])
def test_claims_that_make_no_sense_are_refused(fields):
    with pytest.raises(ValueError):
        _claims(**fields)
