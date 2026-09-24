"""Unit tests for `_maybe_resign_presigned_url`.

Covers the four cases:
  1. Pass-through: non-presigned URL (raw s3://, a custom scheme, plain HTTPS).
  2. Pass-through: presigned URL with plenty of TTL left.
  3. Re-sign: presigned URL near or past expiry.
  4. Pass-through (with warn): malformed presigned URL.

The actual `boto3.client("s3").generate_presigned_url` call is monkey-patched
because we don't need real AWS creds to validate the dispatch logic.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agent_env.task_step.task_steps import deploy_agent


def _build_presigned_url(*, signed_at: datetime, expires_in: int, bucket: str = "artifact-bucket", key: str = "agent_snapshots/x/workspace.tar.gz") -> str:
    amz_date = signed_at.strftime("%Y%m%dT%H%M%SZ")
    return (
        f"https://{bucket}.s3.us-west-2.amazonaws.com/{key}"
        f"?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        f"&X-Amz-Date={amz_date}"
        f"&X-Amz-Expires={expires_in}"
        f"&X-Amz-Signature=abc"
    )


def test_passthrough_raw_s3_url():
    url = "s3://artifact-bucket/snap/workspace.tar.gz"
    assert deploy_agent._maybe_resign_presigned_url(url) == url


def test_passthrough_custom_scheme_url():
    url = "vault://path/to/file#s3/example-bucket"
    assert deploy_agent._maybe_resign_presigned_url(url) == url


def test_passthrough_plain_https_no_amz_params():
    url = "https://example.com/foo.tar.gz"
    assert deploy_agent._maybe_resign_presigned_url(url) == url


def test_passthrough_presigned_with_long_ttl():
    # Signed now, valid for 7 days → far above the 600s refresh buffer.
    url = _build_presigned_url(
        signed_at=datetime.now(timezone.utc), expires_in=7 * 24 * 3600,
    )
    assert deploy_agent._maybe_resign_presigned_url(url) == url


def test_resign_when_near_expiry(monkeypatch):
    # Signed 7 days ago, 7-day TTL → just expired.
    url = _build_presigned_url(
        signed_at=datetime.now(timezone.utc) - timedelta(days=7, seconds=10),
        expires_in=7 * 24 * 3600,
    )
    fresh = "https://artifact-bucket.s3.us-west-2.amazonaws.com/RESIGNED"

    seen_region: list[str | None] = []

    class _StubS3:
        def generate_presigned_url(self, op, Params, ExpiresIn):  # noqa: N803
            assert op == "get_object"
            assert Params["Bucket"] == "artifact-bucket"
            assert Params["Key"] == "agent_snapshots/x/workspace.tar.gz"
            assert ExpiresIn == 7 * 24 * 3600
            return fresh

    import boto3
    monkeypatch.setattr(
        boto3,
        "client",
        lambda svc, region_name=None: (seen_region.append(region_name), _StubS3())[1],
    )
    assert deploy_agent._maybe_resign_presigned_url(url) == fresh
    # Region must be parsed out of the URL host and forwarded to boto3, otherwise
    # S3 returns AuthorizationQueryParametersError when the worker default differs.
    assert seen_region == ["us-west-2"]


def test_resign_when_expiry_within_buffer(monkeypatch):
    # 1h from now — inside the 2h refresh buffer.
    url = _build_presigned_url(
        signed_at=datetime.now(timezone.utc) - timedelta(seconds=4 * 3600 - 3600),
        expires_in=4 * 3600,
    )
    fresh = "https://artifact-bucket.s3.us-west-2.amazonaws.com/RESIGNED"

    class _StubS3:
        def generate_presigned_url(self, op, Params, ExpiresIn):  # noqa: N803
            return fresh

    import boto3
    monkeypatch.setattr(boto3, "client", lambda svc, region_name=None: _StubS3())
    assert deploy_agent._maybe_resign_presigned_url(url) == fresh


def test_passthrough_unparseable_amz_date():
    url = (
        "https://b.s3.us-west-2.amazonaws.com/k"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=notadate&X-Amz-Expires=3600"
    )
    assert deploy_agent._maybe_resign_presigned_url(url) == url


def test_passthrough_non_virtual_hosted_style():
    # Path-style URL (https://s3.region.amazonaws.com/bucket/key) — the helper
    # doesn't recover bucket+key cleanly from this shape, so it passes through
    # rather than re-signing wrong. Confirms the warn-and-return branch.
    url = (
        "https://s3.us-west-2.amazonaws.com/some-bucket/some-key"
        "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Date=" + (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y%m%dT%H%M%SZ")
        + "&X-Amz-Expires=3600"
    )
    assert deploy_agent._maybe_resign_presigned_url(url) == url
