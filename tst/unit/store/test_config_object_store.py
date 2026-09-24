"""Backend injection + AGENT_ENV_OBJECT_STORE / config.toml selection for get_object_store().

S3-free (the tst/unit socket guard forbids network): injection returns a fake
verbatim, and the ``local`` selector / a custom ``impl`` build real stores under
a temp cwd.
"""

import pytest

from agent_env.store import ConfigError, LocalFilesystemObjectStore
from agent_env.config import Config, configure, get_config, set_object_store
from agent_env.store.document_store import Filter
from agent_env.store.object_store import S3ObjectStore
from tst.unit.store.fakes import FakeObjectStore

_FAKE_SECTION = '[stores.object]\nimpl = "tst.unit.store.fakes:FakeObjectStore"\n'


def _write_config(tmp_path, body):
    agentenv = tmp_path / ".agentenv"
    agentenv.mkdir(exist_ok=True)
    (agentenv / "config.toml").write_text(body)


def test_set_object_store_returns_it_verbatim():
    cfg = Config()
    fake = FakeObjectStore()
    cfg.set_object_store(fake)
    assert cfg.get_object_store() is fake  # no S3 built — override short-circuits


def test_module_level_set_object_store_overrides_singleton():
    fake = FakeObjectStore()
    set_object_store(fake)
    assert get_config().get_object_store() is fake


def test_configure_carries_object_store():
    fake = FakeObjectStore()
    configure(object_store=fake)
    assert get_config().get_object_store() is fake


def test_env_selector_builds_local_filesystem(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    store = Config().get_object_store()
    assert isinstance(store, LocalFilesystemObjectStore)
    locator = store.put("artifacts/x/1/a.json", b"hi")
    assert store.get(locator) == b"hi"
    assert (tmp_path / ".agentenv" / "object_store").exists()


def test_default_backend_is_local(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_OBJECT_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    section = Config()._resolve_object_section()
    assert section["impl"] == "agent_env.store.object_store:LocalFilesystemObjectStore"


def test_s3_alias_raises_actionable(monkeypatch):
    # No built-in coordinates: the error names the table AND the overriding env var.
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "s3")
    with pytest.raises(ConfigError, match=r"\[stores\.object\].*AGENT_ENV_OBJECT_STORE"):
        Config().get_object_store()


def test_get_s3_bucket_re_sources_from_the_configured_store(monkeypatch, tmp_path):
    cfg = Config()
    cfg.set_object_store(S3ObjectStore(client=object(), bucket="my-bucket"))
    assert cfg.get_s3_bucket() == "my-bucket"


def test_get_s3_bucket_raises_actionable_without_an_s3_store(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_OBJECT_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match=r"\[stores\.object\]"):
        Config().get_s3_bucket()


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "bogus")
    with pytest.raises(ValueError, match="AGENT_ENV_OBJECT_STORE"):
        Config().get_object_store()


def test_config_toml_selects_custom_impl(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_OBJECT_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, _FAKE_SECTION)
    assert isinstance(Config().get_object_store(), FakeObjectStore)


def test_env_override_beats_config_toml(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, _FAKE_SECTION)
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    assert isinstance(Config().get_object_store(), LocalFilesystemObjectStore)


def test_s3_object_store_from_config_builds_adaptive_client(monkeypatch):
    captured: dict = {}

    def _fake_client(service, **kwargs):
        captured["service"] = service
        captured["config"] = kwargs.get("config")
        return object()

    monkeypatch.setattr("agent_env.store.object_store.s3_object_store.boto3.client", _fake_client)
    store = S3ObjectStore.from_config(bucket="my-bucket")

    assert isinstance(store, S3ObjectStore)
    assert store._bucket == "my-bucket"
    assert captured["service"] == "s3"
    assert captured["config"].retries == {"max_attempts": 10, "mode": "adaptive"}


def test_local_stores_create_nothing_until_first_write(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    config = Config()
    objects, documents = config.get_object_store(), config.get_document_store()
    assert objects.list("") == []
    assert list(tmp_path.iterdir()) == []

    objects.put("k", b"v")
    documents.insert("c", {"id": "x"})
    for state in (tmp_path / ".agentenv" / "object_store", tmp_path / ".agentenv" / "document_store"):
        assert (state / ".gitignore").read_text() == "*\n"
    assert not (tmp_path / ".agentenv" / ".gitignore").exists()
    assert objects.list("") == ["k"]


def test_existing_local_state_stays_readable_and_untouched(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    first = Config()
    first.get_object_store().put("k", b"v")
    first.get_document_store().insert("c", {"id": "x"})
    root = tmp_path / ".agentenv" / "object_store"
    (root / ".gitignore").unlink()

    again = Config()
    again.get_object_store().put("k2", b"v2")
    assert sorted(again.get_object_store().list("")) == ["k", "k2"]
    assert again.get_document_store().find_one("c", Filter.of(id="x"))["id"] == "x"
    assert not (root / ".gitignore").exists()
