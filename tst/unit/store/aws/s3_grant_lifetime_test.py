"""An S3 store's grants last its configured lifetime, 12 hours unless set, and never beyond what SigV4 can sign."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from agent_env.store.object_store import DEFAULT_GRANT_LIFETIME_SECONDS, S3ObjectStore

BUCKET = "artifact-bucket"


class _StubS3:
    def __init__(self) -> None:
        self._request_signer = SimpleNamespace(_credentials=SimpleNamespace(token=None))  # long-term

    def generate_presigned_url(self, operation, **kwargs):
        return f"https://{BUCKET}.s3.amazonaws.com/signed"


def _lasts(expires_at: datetime, seconds: int) -> bool:
    return abs((expires_at - (datetime.now(UTC) + timedelta(seconds=seconds))).total_seconds()) < 2


def test_an_s3_store_grants_for_twelve_hours_by_default():
    store = S3ObjectStore(_StubS3(), BUCKET)
    assert _lasts(store.issue_read_grant(f"s3://{BUCKET}/k").expires_at, DEFAULT_GRANT_LIFETIME_SECONDS)


def test_an_s3_store_grants_for_its_configured_lifetime():
    store = S3ObjectStore(_StubS3(), BUCKET, grant_lifetime_seconds=900)
    assert _lasts(store.issue_read_grant(f"s3://{BUCKET}/k").expires_at, 900)
    write = store.issue_write_grant(f"s3://{BUCKET}/k", media_type="application/json", max_bytes=8)
    assert _lasts(write.expires_at, 900)


def test_a_caller_can_still_name_a_lifetime():
    store = S3ObjectStore(_StubS3(), BUCKET, grant_lifetime_seconds=900)
    assert _lasts(store.issue_read_grant(f"s3://{BUCKET}/k", expires_in=60).expires_at, 60)


@pytest.mark.parametrize("seconds, why", [
    (True, "whole number"), (3600.0, "whole number"), ("3600", "whole number"),
    (0, "at least 900"), (-1, "at least 900"), (899, "at least 900"),
])
def test_a_lifetime_that_is_not_whole_seconds_or_is_too_short_to_outlast_a_transfer_is_refused(seconds, why):
    with pytest.raises(ValueError, match=why):
        S3ObjectStore(_StubS3(), BUCKET, grant_lifetime_seconds=seconds)


def test_an_s3_lifetime_beyond_sigv4s_seven_days_is_refused():
    S3ObjectStore(_StubS3(), BUCKET, grant_lifetime_seconds=7 * 24 * 60 * 60)
    with pytest.raises(ValueError, match="at most 604800"):
        S3ObjectStore(_StubS3(), BUCKET, grant_lifetime_seconds=7 * 24 * 60 * 60 + 1)
