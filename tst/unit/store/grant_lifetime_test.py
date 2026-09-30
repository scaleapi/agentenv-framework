"""Each store sets how long its read and write grants last, 12 hours unless configured; agent-env names no
lifetime of its own, so a grant lasts what the store that issues it says."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from agent_env.a2a_agent.object_transfer import read_object, write_object
from agent_env.config import Config
from agent_env.store import LocalFilesystemObjectStore
from agent_env.store.object_store import DEFAULT_GRANT_LIFETIME_SECONDS, S3ObjectStore
from agent_env.store.routing import LocalRunObjectStore
from tst.util.granting_object_store import GrantingObjectStore

BUCKET = "artifact-bucket"


class _StubS3:
    def __init__(self) -> None:
        self._request_signer = SimpleNamespace(_credentials=SimpleNamespace(token=None))  # long-term

    def generate_presigned_url(self, operation, **kwargs):
        return f"https://{BUCKET}.s3.amazonaws.com/signed"


def _lasts(expires_at: datetime, seconds: int) -> bool:
    return abs((expires_at - (datetime.now(UTC) + timedelta(seconds=seconds))).total_seconds()) < 2


def test_the_default_is_twelve_hours():
    assert DEFAULT_GRANT_LIFETIME_SECONDS == 12 * 60 * 60
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


@pytest.mark.parametrize("seconds", [0, -1, True, 3600.0, "3600"])
def test_a_lifetime_that_is_not_a_positive_whole_number_of_seconds_is_refused(seconds, tmp_path):
    with pytest.raises(ValueError, match="positive whole number"):
        S3ObjectStore(_StubS3(), BUCKET, grant_lifetime_seconds=seconds)
    with pytest.raises(ValueError, match="positive whole number"):
        LocalFilesystemObjectStore(str(tmp_path), grant_lifetime_seconds=seconds)


def test_an_s3_lifetime_beyond_sigv4s_seven_days_is_refused():
    S3ObjectStore(_StubS3(), BUCKET, grant_lifetime_seconds=7 * 24 * 60 * 60)
    with pytest.raises(ValueError, match="at most 604800"):
        S3ObjectStore(_StubS3(), BUCKET, grant_lifetime_seconds=7 * 24 * 60 * 60 + 1)


def test_agent_env_grants_for_the_issuing_stores_lifetime(tmp_path):
    store = GrantingObjectStore(str(tmp_path))
    store.grant_lifetime_seconds = 900
    url = store.put("a/in.json", b"{}", content_type="application/json")
    assert _lasts(read_object(store, url).read.expires_at, 900)
    assert _lasts(write_object(store, store.object_url("a/out.json"), media_type="application/json", max_bytes=8).write.expires_at, 900)


def test_a_local_run_grants_for_the_lifetime_of_the_store_that_owns_the_object(tmp_path):
    local = GrantingObjectStore(str(tmp_path))
    local.grant_lifetime_seconds = 600
    routed = LocalRunObjectStore(S3ObjectStore(_StubS3(), BUCKET, grant_lifetime_seconds=900), local)
    assert _lasts(routed.issue_read_grant(f"s3://{BUCKET}/k").expires_at, 900)
    assert _lasts(routed.issue_read_grant(local.put("k", b"v")).expires_at, 600)


def test_config_toml_sets_the_local_stores_grant_lifetime(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_OBJECT_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".agentenv").mkdir()
    (tmp_path / ".agentenv" / "config.toml").write_text(
        '[stores.object]\nimpl = "agent_env.store.object_store:LocalFilesystemObjectStore"\n'
        'config = { grant_lifetime_seconds = 3600 }\n'
    )
    assert Config().get_object_store().grant_lifetime_seconds == 3600
