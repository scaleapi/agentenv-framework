"""``ObjectStore.signed_post``: the policy handed to boto3, not S3's response to
it — the round trip is integration territory."""
from __future__ import annotations

from agent_env.store.object_store.local_object_store import LocalFilesystemObjectStore
from agent_env.store.object_store.s3_object_store import S3ObjectStore

BUCKET = "artifact-bucket"


class _StubClient:
    def __init__(self):
        self.calls: list[dict] = []

    def generate_presigned_post(self, **kwargs):
        self.calls.append(kwargs)
        return {"url": f"https://{BUCKET}.s3.amazonaws.com/", "fields": {"key": kwargs["Key"]}}


def _sign(**kwargs):
    client = _StubClient()
    store = S3ObjectStore(client, BUCKET)
    grant = store.signed_post(f"s3://{BUCKET}/captures/one/", **kwargs)
    return client.calls[0], grant


def test_the_key_is_a_template_the_uploader_substitutes():
    call, _ = _sign()
    assert call["Bucket"] == BUCKET
    assert call["Key"] == "captures/one/${filename}"
    assert ["starts-with", "$key", "captures/one/"] in call["Conditions"]


def test_content_type_is_admitted():
    # Without this the uploader's Content-Type 403s as an extra form field.
    call, _ = _sign()
    assert ["starts-with", "$Content-Type", ""] in call["Conditions"]


def test_no_size_bound_unless_asked_for():
    call, _ = _sign()
    assert not [c for c in call["Conditions"] if c[0] == "content-length-range"]


def test_max_bytes_becomes_a_content_length_range():
    """The one guard a signed url cannot express."""
    call, _ = _sign(max_bytes=1024)
    assert ["content-length-range", 0, 1024] in call["Conditions"]


def test_the_grant_is_the_fields_not_the_url():
    _, grant = _sign()
    assert set(grant) == {"url", "fields"}
    # The url is the bare bucket endpoint; it carries no authorization.
    assert grant["url"].rstrip("/").endswith(".s3.amazonaws.com")


def test_a_backend_that_cannot_sign_returns_none(tmp_path):
    """The ABC default."""
    assert LocalFilesystemObjectStore(str(tmp_path)).signed_post(f"file://{tmp_path}/x/") is None
