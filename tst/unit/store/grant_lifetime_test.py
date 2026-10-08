"""Each store sets how long its read and write grants last, 12 hours unless configured; agent-env names no
lifetime of its own, so a grant lasts what the store that issues it says."""

from datetime import UTC, datetime, timedelta

import pytest

from agent_env.a2a_agent.object_transfer import TRANSFER_TIMEOUT_SECONDS, read_object, write_object
from agent_env.config import Config
from agent_env.store import LocalFilesystemObjectStore
from agent_env.store.object_store import DEFAULT_GRANT_LIFETIME_SECONDS, MIN_GRANT_LIFETIME_SECONDS
from agent_env.store.routing import LocalRunObjectStore
from tst.util.granting_object_store import GrantingObjectStore

def _lasts(expires_at: datetime, seconds: int) -> bool:
    return abs((expires_at - (datetime.now(UTC) + timedelta(seconds=seconds))).total_seconds()) < 2


def test_the_default_is_twelve_hours(tmp_path):
    assert DEFAULT_GRANT_LIFETIME_SECONDS == 12 * 60 * 60
    store = GrantingObjectStore(str(tmp_path))
    assert _lasts(store.issue_read_grant(store.put("k", b"v")).expires_at, DEFAULT_GRANT_LIFETIME_SECONDS)


@pytest.mark.parametrize("seconds, why", [
    (True, "whole number"), (3600.0, "whole number"), ("3600", "whole number"),
    (0, "at least 900"), (-1, "at least 900"), (899, "at least 900"),
])
def test_a_lifetime_that_is_not_whole_seconds_or_is_too_short_to_outlast_a_transfer_is_refused(seconds, why, tmp_path):
    with pytest.raises(ValueError, match=why):
        LocalFilesystemObjectStore(str(tmp_path), grant_lifetime_seconds=seconds)


def test_the_shortest_lifetime_outlasts_agent_envs_wait_for_a_transfer():
    assert TRANSFER_TIMEOUT_SECONDS < MIN_GRANT_LIFETIME_SECONDS <= DEFAULT_GRANT_LIFETIME_SECONDS


def test_a_local_lifetime_beyond_seven_days_is_refused(tmp_path):
    LocalFilesystemObjectStore(str(tmp_path), grant_lifetime_seconds=7 * 24 * 60 * 60)
    with pytest.raises(ValueError, match="at most 604800"):
        LocalFilesystemObjectStore(str(tmp_path), grant_lifetime_seconds=7 * 24 * 60 * 60 + 1)


def test_agent_env_grants_for_the_issuing_stores_lifetime(tmp_path):
    store = GrantingObjectStore(str(tmp_path))
    store.grant_lifetime_seconds = 900
    url = store.put("a/in.json", b"{}", content_type="application/json")
    assert _lasts(read_object(store, url).read.expires_at, 900)
    assert _lasts(write_object(store, store.object_url("a/out.json"), media_type="application/json", max_bytes=8).write.expires_at, 900)


def test_a_local_run_grants_for_the_lifetime_of_the_store_that_owns_the_object(tmp_path):
    configured = GrantingObjectStore(str(tmp_path / "configured"))
    configured.grant_lifetime_seconds = 900
    local = GrantingObjectStore(str(tmp_path / "local"))
    local.grant_lifetime_seconds = 1800
    routed = LocalRunObjectStore(configured, local)
    assert _lasts(routed.issue_read_grant(configured.put("k", b"v")).expires_at, 900)
    assert _lasts(routed.issue_read_grant(local.put("k", b"v")).expires_at, 1800)


def test_a_local_run_leaves_a_custom_stores_own_default_alone(tmp_path):
    class _CustomStore(GrantingObjectStore):  # a store written before lifetimes were a setting
        def issue_read_grant(self, object_url, *, expires_in: int = 300):
            return super().issue_read_grant(object_url, expires_in=expires_in)

    custom = _CustomStore(str(tmp_path / "custom"))
    routed = LocalRunObjectStore(custom, GrantingObjectStore(str(tmp_path / "local")))
    assert _lasts(routed.issue_read_grant(custom.put("k", b"v")).expires_at, 300)


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
