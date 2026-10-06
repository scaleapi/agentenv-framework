"""Provider-neutral object-transfer grants issued by ``ObjectStore``."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import boto3
import pytest
from agentenv_protocol.transfers import (
    HttpGetGrant,
    HttpPostPolicyGrant,
    HttpPutGrant,
)
from botocore.credentials import Credentials, RefreshableCredentials

from agent_env.store import GrantUnavailableError
from agent_env.store.object_store.local.store import (
    LocalFilesystemObjectStore,
)
from agent_env.store.object_store import UploadPolicy
from agent_env.store.object_store.s3_object_store import S3ObjectStore
from agent_env.store.routing import LocalRunObjectStore

BUCKET = "artifact-bucket"


class _StubClient:
    def __init__(self) -> None:
        self.url_calls: list[tuple[str, dict]] = []
        self.post_calls: list[dict] = []
        # Long-term credentials: no session token and no expiry.
        self._request_signer = SimpleNamespace(_credentials=SimpleNamespace(token=None))

    def generate_presigned_url(self, operation: str, **kwargs) -> str:
        self.url_calls.append((operation, kwargs))
        return f"https://{BUCKET}.s3.amazonaws.com/signed"

    def generate_presigned_post(self, **kwargs) -> dict:
        self.post_calls.append(kwargs)
        return {
            "url": f"https://{BUCKET}.s3.amazonaws.com/",
            "fields": {"key": kwargs["Key"], "policy": "redacted"},
        }


@pytest.fixture
def client() -> _StubClient:
    return _StubClient()


@pytest.fixture
def store(client: _StubClient) -> S3ObjectStore:
    return S3ObjectStore(client, BUCKET)


def _assert_expiry(expires_at: datetime, *, seconds: int) -> None:
    expected = datetime.now(UTC) + timedelta(seconds=seconds)
    assert expires_at.tzinfo is UTC
    assert abs((expires_at - expected).total_seconds()) < 1


def test_s3_store_from_config_forces_signature_v4() -> None:
    with patch("boto3.client", return_value=_StubClient()) as client_factory:
        S3ObjectStore.from_config(bucket=BUCKET, region="us-west-2")

    config = client_factory.call_args.kwargs["config"]
    assert config.signature_version == "s3v4"


def test_s3_read_grant_signs_an_exact_get(
    store: S3ObjectStore, client: _StubClient
) -> None:
    grant = store.issue_read_grant(f"s3://{BUCKET}/objects/input.json", expires_in=90)

    assert isinstance(grant, HttpGetGrant)
    assert grant.kind == "http-get"
    assert grant.url.startswith("https://")
    assert grant.headers is None
    _assert_expiry(grant.expires_at, seconds=90)
    assert client.url_calls == [
        (
            "get_object",
            {
                "Params": {"Bucket": BUCKET, "Key": "objects/input.json"},
                "ExpiresIn": 90,
            },
        )
    ]


def test_s3_keys_keep_the_characters_a_url_parser_would_cut(
    store: S3ObjectStore, client: _StubClient
) -> None:
    key = "skills/review/notes #1?.md"

    store.issue_read_grant(f"s3://{BUCKET}/{key}")

    assert store.get_object_key(f"s3://{BUCKET}/{key}") == key
    assert client.url_calls[0][1]["Params"]["Key"] == key
    with pytest.raises(ValueError, match="is not an object in"):
        store.get_object_key(f"s3://other-bucket/{key}")


def test_s3_write_grant_signs_media_type_for_an_exact_put(
    store: S3ObjectStore, client: _StubClient
) -> None:
    grant = store.issue_write_grant(
        f"s3://{BUCKET}/objects/output.tar.gz",
        media_type="application/gzip",
        max_bytes=4096,
        expires_in=120,
    )

    assert isinstance(grant, HttpPutGrant)
    assert grant.kind == "http-put"
    assert grant.headers == {"Content-Type": "application/gzip"}
    _assert_expiry(grant.expires_at, seconds=120)
    assert client.url_calls == [
        (
            "put_object",
            {
                "Params": {
                    "Bucket": BUCKET,
                    "Key": "objects/output.tar.gz",
                    "ContentType": "application/gzip",
                },
                "ExpiresIn": 120,
            },
        )
    ]


@pytest.mark.parametrize("prefix", ["changelog/run-1", "changelog/run-1/"])
def test_s3_upload_policy_signs_the_prefix_and_per_object_limit(
    store: S3ObjectStore, client: _StubClient, prefix: str
) -> None:
    policy = store.issue_upload_policy(
        f"s3://{BUCKET}/{prefix}", max_object_bytes=1024, expires_in=180
    )

    _assert_expiry(policy.expires_at, seconds=180)
    assert isinstance(policy.write, HttpPostPolicyGrant)
    assert policy.write.kind == "http-post-policy"
    assert policy.write.path_field == "key"
    assert policy.write.file_field == "file"
    assert policy.write.fields["key"] == "changelog/run-1/${filename}"

    assert client.post_calls == [
        {
            "Bucket": BUCKET,
            "Key": "changelog/run-1/${filename}",
            "Conditions": [
                ["starts-with", "$key", "changelog/run-1/"],
                ["content-length-range", 0, 1024],
                ["starts-with", "$Content-Type", ""],
            ],
            "ExpiresIn": 180,
        }
    ]


def _signing_with(credentials: Credentials) -> S3ObjectStore:
    client = boto3.client(
        "s3",
        region_name="us-west-2",
        aws_access_key_id="placeholder",
        aws_secret_access_key="placeholder",
    )
    client._request_signer._credentials = credentials
    return S3ObjectStore(client, BUCKET)


def _session(expiry: datetime) -> RefreshableCredentials:
    return RefreshableCredentials(
        access_key="test-access-key",
        secret_key="test-secret-key",
        token="test-session-token",
        expiry_time=expiry,
        refresh_using=lambda: {},
        method="test",
    )


def _namespace_grant(store: S3ObjectStore, *, expires_in: int) -> UploadPolicy:
    return store.issue_upload_policy(
        f"s3://{BUCKET}/changelog/run-1", max_object_bytes=1024, expires_in=expires_in
    )


def test_s3_grants_end_when_the_signing_credentials_expire() -> None:
    credential_expiry = datetime.now(UTC) + timedelta(minutes=30)
    store = _signing_with(_session(credential_expiry))

    grant = store.issue_read_grant(f"s3://{BUCKET}/objects/input", expires_in=28_800)
    assert grant.expires_at == credential_expiry

    with pytest.raises(GrantUnavailableError) as raised:
        _namespace_grant(store, expires_in=28_800)
    assert credential_expiry.isoformat(timespec="seconds") in str(raised.value)
    assert "28800s" in str(raised.value)


@pytest.mark.parametrize(
    "credentials",
    [
        _session(datetime.now(UTC) + timedelta(hours=12)),
        Credentials("test-access-key", "test-secret-key", "test-session-token"),
    ],
    ids=["session-outlasting-the-grant", "session-token-without-expiry"],
)
def test_s3_namespace_grants_refuse_temporary_credentials_whatever_they_have_left(
    credentials: Credentials,
) -> None:
    with pytest.raises(GrantUnavailableError, match="temporary.*600s namespace grant"):
        _namespace_grant(_signing_with(credentials), expires_in=600)


def test_s3_namespace_grants_last_their_full_lifetime_on_long_term_credentials() -> None:
    store = _signing_with(Credentials("test-access-key", "test-secret-key"))

    grant = _namespace_grant(store, expires_in=28_800)

    _assert_expiry(grant.expires_at, seconds=28_800)
    assert grant.write.fields["key"] == "changelog/run-1/${filename}"


def test_s3_namespace_grants_refuse_credentials_they_cannot_inspect() -> None:
    client = _StubClient()
    del client._request_signer

    with pytest.raises(GrantUnavailableError, match="cannot be inspected"):
        _namespace_grant(S3ObjectStore(client, BUCKET), expires_in=600)


@pytest.mark.parametrize(
    ("endpoint", "grants"),
    [("https://s3.us-west-2.amazonaws.com", True), ("http://localhost:9000", False)],
)
def test_s3_store_on_a_plain_http_endpoint_issues_no_grants(endpoint: str, grants: bool) -> None:
    client = boto3.client(
        "s3",
        region_name="us-west-2",
        endpoint_url=endpoint,
        aws_access_key_id="placeholder",
        aws_secret_access_key="placeholder",
    )

    assert S3ObjectStore(client, BUCKET).supports_transfer_grants is grants


def test_s3_grants_are_capped_at_the_sigv4_maximum(store: S3ObjectStore) -> None:
    prefix = f"s3://{BUCKET}/changelog/run-1"

    policy = store.issue_upload_policy(prefix, max_object_bytes=10, expires_in=604_800)
    _assert_expiry(policy.expires_at, seconds=604_800)
    with pytest.raises(GrantUnavailableError, match="at most 604800s; 604801s"):
        store.issue_upload_policy(prefix, max_object_bytes=10, expires_in=604_801)
    with pytest.raises(GrantUnavailableError, match="at most 604800s; 604801s"):
        store.issue_read_grant(f"{prefix}/000000", expires_in=604_801)


def test_botocore_still_exposes_the_signing_credentials() -> None:
    """S3ObjectStore reads the signer's credentials through botocore's private attributes; if a
    botocore release moves them, this fails rather than every grant silently losing its clamp."""
    long_term = boto3.client(
        "s3",
        region_name="us-west-2",
        aws_access_key_id="placeholder",
        aws_secret_access_key="placeholder",
    )
    session = boto3.client(
        "s3",
        region_name="us-west-2",
        aws_access_key_id="placeholder",
        aws_secret_access_key="placeholder",
        aws_session_token="token",
    )

    assert S3ObjectStore(long_term, BUCKET)._signing_credentials() == (None, False)
    assert S3ObjectStore(session, BUCKET)._signing_credentials() == (None, True)


def test_local_store_grants_reach_only_local_sandboxes(tmp_path) -> None:
    store = LocalFilesystemObjectStore(str(tmp_path))

    assert store.supports_transfer_grants
    assert store.grants_reach("local")
    assert not store.grants_reach("modal")
    assert not store.grants_reach(None)


def test_local_store_grants_can_be_turned_off(tmp_path) -> None:
    assert not LocalFilesystemObjectStore(str(tmp_path), grants="off").supports_transfer_grants
    with pytest.raises(ValueError, match="grants must be one of"):
        LocalFilesystemObjectStore(str(tmp_path), grants="on")


def test_hosted_store_grants_reach_any_sandbox(store: S3ObjectStore) -> None:
    assert store.supports_transfer_grants
    assert store.grants_reach("modal") and store.grants_reach("local") and store.grants_reach(None)


def test_a_local_run_offers_grants_that_reach_its_local_store(tmp_path, store: S3ObjectStore) -> None:
    routed = LocalRunObjectStore(store, LocalFilesystemObjectStore(str(tmp_path)))

    assert routed.supports_transfer_grants
    assert routed.grants_reach("local")
    assert not routed.grants_reach("modal")
