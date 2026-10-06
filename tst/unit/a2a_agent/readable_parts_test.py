"""A file part naming an object a configured store owns is sent to an agent as an HTTPS URL it can read; every
other part is sent as it is, and an owned object the agent can be given no URL for fails the send."""

import copy

import pytest

from agent_env.a2a_agent.object_transfer import readable_parts
from agent_env.config import set_object_store
from tst.util.granting_object_store import GRANT_ORIGIN, GrantingObjectStore

URL = "https://agent.example.test"


@pytest.fixture
def store(tmp_path):
    granting = GrantingObjectStore(str(tmp_path))
    set_object_store(granting)
    return granting


def _file(uri, mime="image/png"):
    return {"kind": "file", "file": {"uri": uri, "mimeType": mime, "name": "x"}}


@pytest.mark.asyncio
async def test_an_owned_object_is_sent_as_a_grant_and_the_parts_given_are_left_as_they_are(store):
    url = store.put("seeds/s1/x.png", b"png")
    parts = [{"kind": "text", "text": "look"}, _file(url)]
    given = copy.deepcopy(parts)

    async with readable_parts(parts, a2a_url=URL, card=None, sandbox_type="local", expires_in=600) as sent:
        assert sent == [{"kind": "text", "text": "look"}, _file(f"{GRANT_ORIGIN}/seeds/s1/x.png?sig=read")]

    assert parts == given
    assert store.granted == [url]


@pytest.mark.asyncio
async def test_parts_that_name_no_owned_object_are_sent_as_they_are(store):
    parts = [
        {"kind": "text", "text": "look"},
        {"kind": "data", "data": {"k": "v"}},
        {"kind": "file", "file": {"bytes": "cG5n", "mimeType": "image/png"}},
        _file("https://example.test/x.png"),
        _file("s3://another-bucket/x.png"),
    ]

    async with readable_parts(parts, a2a_url=URL, card=None, sandbox_type="modal", expires_in=600) as sent:
        assert sent == parts

    assert store.granted == []


@pytest.mark.asyncio
async def test_an_owned_object_the_agent_can_be_given_no_url_for_is_not_sent(store):
    url = store.put("seeds/s1/x.png", b"png")

    with pytest.raises(RuntimeError, match="gives no URL that agents on the 'modal' sandbox provider can read") as raised:
        async with readable_parts([_file(url)], a2a_url=URL, card=None, sandbox_type="modal", expires_in=600):
            pytest.fail("the parts must not be sent")

    assert url in str(raised.value)
