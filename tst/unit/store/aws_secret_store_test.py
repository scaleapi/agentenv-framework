"""AwsSecretsManagerSecretStore load-time behavior (no AWS / no network).

Fakes the ``secretsmanager`` client at the ``boto3`` boundary so the same parse /
cache / validation assertions run without credentials.
"""

import logging

import pytest

from agent_env.config import Config
from agent_env.store import AwsSecretsManagerSecretStore


class _FakeClient:
    def __init__(self, backing: "_FakeSecret"):
        self._backing = backing

    def get_secret_value(self, SecretId):  # noqa: N803 (boto3 kwarg name)
        self._backing.fetches += 1
        return {"SecretString": self._backing.secret_string}


class _FakeSecret:
    """A mutable stand-in for the AWS secret, counting how often it is fetched."""

    def __init__(self, secret_string: str):
        self.secret_string = secret_string
        self.fetches = 0


@pytest.fixture
def store(monkeypatch):
    def _build(secret_string: str, **kwargs) -> AwsSecretsManagerSecretStore:
        backing = _FakeSecret(secret_string)
        monkeypatch.setattr("boto3.client", lambda *a, **k: _FakeClient(backing))
        s = AwsSecretsManagerSecretStore("my-team/bundle", "us-west-2", **kwargs)
        s.backing = backing  # type: ignore[attr-defined]  # test handle
        return s

    return _build


def test_loads_flat_mapping(store):
    s = store("litellm_api_key: sk-aws\nmodal_token_id: tok-1\n")
    assert s.get("litellm_api_key") == "sk-aws"
    assert s.get("modal_token_id") == "tok-1"
    assert s.get("missing") is None


def test_empty_secret_string_is_empty_mapping(store):
    assert store("").get("anything") is None


def test_non_mapping_secret_rejected_at_load_time(store):
    # A plain scalar / list secret must fail loudly on first access rather than
    # raising a cryptic AttributeError later inside get().
    with pytest.raises(ValueError):
        store("- just\n- a\n- list\n").get("anything")


def test_scalar_secret_rejected_at_load_time(store):
    with pytest.raises(ValueError):
        store("just-a-bare-string").get("anything")


def test_non_string_values_coerced_to_str(store):
    assert store("port: 5432\n").get("port") == "5432"


def test_from_config_builds_store():
    s = AwsSecretsManagerSecretStore.from_config(
        secret_name="my-team/bundle", region="us-west-2"
    )
    assert isinstance(s, AwsSecretsManagerSecretStore)


def test_from_config_accepts_ttl_overrides():
    s = AwsSecretsManagerSecretStore.from_config(
        secret_name="my-team/bundle",
        region="us-west-2",
        ttl_seconds=42,
        min_refresh_interval=7,
    )
    assert s._ttl_seconds == 42.0
    assert s._min_refresh_interval == 7.0


# --- TTL / refresh: time driven through an injected monotonic clock -----------------------


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1_000.0}
    monkeypatch.setattr(
        "agent_env.store.secret_store.aws_secrets_manager_secret_store.time.monotonic",
        lambda: now["t"],
    )
    return now


def test_bundle_is_cached_within_the_ttl(store, clock):
    s = store("a: 1\n", ttl_seconds=300)
    assert s.get("a") == "1"
    clock["t"] += 299
    assert s.get("a") == "1"
    assert s.backing.fetches == 1


def test_added_key_resolves_after_the_ttl_without_a_restart(store, clock):
    s = store("a: 1\n", ttl_seconds=300)
    assert s.get("new_key") is None
    s.backing.secret_string = "a: 1\nnew_key: sk-new\n"
    assert s.get("new_key") is None  # cached bundle predates the addition

    clock["t"] += 301
    assert s.get("new_key") == "sk-new"
    assert s.backing.fetches == 2


def test_rotated_value_converges_after_the_ttl(store, clock):
    # A changed value never misses, so only the TTL can catch it.
    s = store("a: old\n", ttl_seconds=60)
    assert s.get("a") == "old"
    s.backing.secret_string = "a: new\n"

    clock["t"] += 61
    assert s.get("a") == "new"


def test_zero_ttl_caches_for_the_process_lifetime(store, clock):
    s = store("a: 1\n", ttl_seconds=0)
    assert s.get("a") == "1"
    s.backing.secret_string = "a: 2\n"
    clock["t"] += 86_400
    assert s.get("a") == "1"
    assert s.backing.fetches == 1


def test_refresh_picks_up_an_added_key_immediately(store, clock):
    s = store("a: 1\n", ttl_seconds=300)
    assert s.get("added") is None
    s.backing.secret_string = "a: 1\nadded: sk-new\n"

    assert s.refresh()["added"] == "sk-new"
    assert s.get("added") == "sk-new"


def test_refresh_is_rate_limited(store, clock):
    s = store("a: 1\n", ttl_seconds=300, min_refresh_interval=10)
    s.refresh()
    fetches = s.backing.fetches

    for _ in range(5):
        clock["t"] += 1
        s.refresh()
    assert s.backing.fetches == fetches  # all inside the cooldown

    clock["t"] += 10
    s.refresh()
    assert s.backing.fetches == fetches + 1


def test_first_refresh_always_fetches(store, clock):
    s = store("a: 1\n", min_refresh_interval=3600)
    assert s.refresh()["a"] == 1
    assert s.backing.fetches == 1


def test_a_failed_refresh_does_not_start_the_cooldown(store, clock, monkeypatch):
    # A failed attempt must not suppress its own retry.
    s = store("a: old\n", ttl_seconds=300, min_refresh_interval=10)
    assert s.get("a") == "old"

    boom = {"raise": True}
    real_fetch = s._fetch

    def _flaky():
        if boom["raise"]:
            raise RuntimeError("transient Secrets Manager error")
        return real_fetch()

    monkeypatch.setattr(s, "_fetch", _flaky)

    with pytest.raises(RuntimeError):
        s.refresh()

    boom["raise"] = False
    s.backing.secret_string = "a: new\n"
    clock["t"] += 1
    assert s.refresh()["a"] == "new"


def test_a_slow_fetch_does_not_hand_back_an_already_expired_cooldown(store, clock, monkeypatch):
    # The cooldown is stamped off a clock read taken after the network call.
    s = store("a: 1\n", ttl_seconds=300, min_refresh_interval=10)
    real_fetch = s._fetch

    def _slow_fetch():
        clock["t"] += 15  # longer than the cooldown itself
        return real_fetch()

    monkeypatch.setattr(s, "_fetch", _slow_fetch)

    s.refresh()
    fetches = s.backing.fetches

    s.refresh()  # no time has passed since the first refresh returned
    assert s.backing.fetches == fetches


def test_a_slow_fetch_starts_the_ttl_when_the_bundle_arrives(store, clock, monkeypatch):
    # Same post-fetch clock read backs `_fetched_at`.
    s = store("a: 1\n", ttl_seconds=300)
    real_fetch = s._fetch

    def _slow_fetch():
        clock["t"] += 100
        return real_fetch()

    monkeypatch.setattr(s, "_fetch", _slow_fetch)

    s.refresh()
    fetches = s.backing.fetches

    clock["t"] += 250  # 350s after the request was sent, 250s after it returned
    assert s.get("a") == "1"
    assert s.backing.fetches == fetches


def test_a_failed_refresh_leaves_the_cached_bundle_intact(store, clock, monkeypatch):
    s = store("a: old\n", ttl_seconds=300)
    assert s.get("a") == "old"
    monkeypatch.setattr(s, "_fetch", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        s.refresh()
    assert s.get("a") == "old"


# --- the live bundle view (second cache layer) ----------------------------------------------


def test_load_returns_a_live_view_that_observes_changes(store, clock):
    s = store("a: 1\n", ttl_seconds=300)
    view = s._load()
    assert s._load() is view  # one stable object — safe to memoize
    assert view.get("added") is None

    s.backing.secret_string = "a: 2\nadded: sk-new\n"
    clock["t"] += 301
    assert view.get("added") == "sk-new"
    assert view["a"] == 2
    assert "added" in view
    assert sorted(view) == ["a", "added"]
    assert len(view) == 2


def test_config_level_reader_observes_added_keys_without_runtime_changes(store, clock):
    # The layer-2 acceptance criterion, against the real (unmodified) Config memo.
    s = store("litellm_api_key: sk-old\n", ttl_seconds=300)
    cfg = Config()
    cfg.set_secret_store(s)

    secret = cfg._get_secret()
    assert secret.get("new_key") is None

    s.backing.secret_string = "litellm_api_key: sk-rotated\nnew_key: sk-added\n"
    clock["t"] += 301
    assert secret.get("new_key") == "sk-added"
    assert cfg._get_secret().get("litellm_api_key") == "sk-rotated"


def test_view_repr_never_renders_values(store):
    s = store("litellm_api_key: sk-secret\n")
    assert "sk-secret" not in repr(s._load())


def test_first_fetch_failure_raises(store, monkeypatch):
    s = store("a: 1\n")
    monkeypatch.setattr(s, "_fetch", lambda: (_ for _ in ()).throw(RuntimeError("aws down")))
    with pytest.raises(RuntimeError):
        s.get("a")  # nothing cached yet — nothing to serve
    with pytest.raises(RuntimeError):
        s._load()


def test_serve_stale_when_a_ttl_refetch_fails(store, clock, monkeypatch):
    s = store("a: old\n", ttl_seconds=300, min_refresh_interval=10)
    assert s.get("a") == "old"

    monkeypatch.setattr(
        s, "_fetch", lambda: (_ for _ in ()).throw(RuntimeError("transient SM error"))
    )
    clock["t"] += 301
    assert s.get("a") == "old"  # re-fetch failed silently; stale bundle served


def test_failed_refetch_backs_off_instead_of_hammering(store, clock, monkeypatch):
    s = store("a: old\n", ttl_seconds=300, min_refresh_interval=10)
    assert s.get("a") == "old"

    attempts = {"n": 0}

    def _failing():
        attempts["n"] += 1
        raise RuntimeError("outage")

    monkeypatch.setattr(s, "_fetch", _failing)
    clock["t"] += 301

    for _ in range(5):
        assert s.get("a") == "old"
    assert attempts["n"] == 1  # one probe, then backoff

    clock["t"] += 10
    assert s.get("a") == "old"
    assert attempts["n"] == 2


def test_recovery_after_an_outage_serves_fresh_values(store, clock, monkeypatch):
    s = store("a: old\n", ttl_seconds=300, min_refresh_interval=10)
    assert s.get("a") == "old"

    boom = {"raise": True}
    real_fetch = s._fetch

    def _flaky():
        if boom["raise"]:
            raise RuntimeError("outage")
        return real_fetch()

    monkeypatch.setattr(s, "_fetch", _flaky)
    clock["t"] += 301
    assert s.get("a") == "old"

    boom["raise"] = False
    s.backing.secret_string = "a: new\n"
    clock["t"] += 10  # backoff over
    assert s.get("a") == "new"
    assert s._last_failed_fetch_at is None


def test_a_failed_refresh_arms_the_ttl_backoff(store, clock, monkeypatch):
    s = store("a: old\n", ttl_seconds=300, min_refresh_interval=10)
    assert s.get("a") == "old"

    attempts = {"n": 0}

    def _failing():
        attempts["n"] += 1
        raise RuntimeError("outage")

    monkeypatch.setattr(s, "_fetch", _failing)
    clock["t"] += 301
    with pytest.raises(RuntimeError):
        s.refresh()
    assert attempts["n"] == 1

    assert s.get("a") == "old"  # within the backoff — no second probe
    assert attempts["n"] == 1

    clock["t"] += 10
    assert s.get("a") == "old"
    assert attempts["n"] == 2


def test_reader_is_not_blocked_by_an_inflight_refetch(store, clock):
    s = store("a: old\n", ttl_seconds=300)
    assert s.get("a") == "old"
    clock["t"] += 301

    s._lock.acquire()  # stand-in for a fetch in flight on another thread
    try:
        assert s.get("a") == "old"  # would deadlock without the non-blocking path
        assert s.backing.fetches == 1
    finally:
        s._lock.release()

    assert s.get("a") == "old"  # lock free again — the TTL re-fetch proceeds
    assert s.backing.fetches == 2


def test_bulk_accessors_bind_to_one_snapshot(store, clock):
    # A key deleted by a mid-iteration refresh must not KeyError (the Mapping mixins would).
    s = store("a: 1\nb: 2\n", ttl_seconds=300)
    view = s._load()

    items = view.items()  # bound to the first snapshot
    s.backing.secret_string = "a: 1\n"  # 'b' deleted in AWS
    clock["t"] += 301

    assert dict(items) == {"a": 1, "b": 2}
    assert dict(view.items()) == {"a": 1}
    assert view == {"a": 1}


def test_invalid_yaml_never_leaks_secret_content(store):
    # YAML error marks embed the raw secret line they point at.
    s = store("litellm_api_key: sk-SECRET-TOKEN: oops\n")
    with pytest.raises(ValueError) as excinfo:
        s.get("anything")
    assert "sk-SECRET-TOKEN" not in str(excinfo.value)
    assert excinfo.value.__suppress_context__


def test_bundle_rotated_to_garbage_serves_stale_and_logs_error(store, clock, caplog):
    s = store("a: old\nsecret_key: sk-SECRET-TOKEN\n", ttl_seconds=300)
    assert s.get("a") == "old"

    s.backing.secret_string = "just-a-bare-string"
    clock["t"] += 301
    with caplog.at_level(logging.ERROR):
        assert s.get("a") == "old"
    assert any(
        r.levelno == logging.ERROR and "invalid content" in r.message for r in caplog.records
    )
    assert "sk-SECRET-TOKEN" not in caplog.text
