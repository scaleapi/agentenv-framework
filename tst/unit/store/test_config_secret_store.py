"""Backend injection + AGENT_ENV_SECRET_STORE / config.toml selection for get_secret_store().

AWS-free (the tst/unit socket guard forbids network): injection returns a store
verbatim, and the ``local`` selector / a custom ``impl`` build real stores.
"""

import pytest

from agent_env.store import (
    ConfigError,
    LocalSecretStore,
    SecretStore,
)
from agent_env.config import Config, configure, get_config, set_secret_store

_FAKE_SECTION = '[stores.secret]\nimpl = "agent_env.store.secret_store:LocalSecretStore"\n'


def _write_config(tmp_path, body):
    agentenv = tmp_path / ".agentenv"
    agentenv.mkdir(exist_ok=True)
    (agentenv / "config.toml").write_text(body)


def test_set_secret_store_returns_it_verbatim():
    cfg = Config()
    store = LocalSecretStore(values={"k": "v"})
    cfg.set_secret_store(store)
    assert cfg.get_secret_store() is store  # no AWS built — override short-circuits


def test_module_level_set_secret_store_overrides_singleton():
    store = LocalSecretStore(values={"k": "v"})
    set_secret_store(store)
    assert get_config().get_secret_store() is store


def test_configure_carries_secret_store():
    store = LocalSecretStore(values={"k": "v"})
    configure(secret_store=store)
    assert get_config().get_secret_store() is store


def test_env_selector_builds_local(monkeypatch):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.setenv("AGENT_ENV_SECRET_STORE", "local")
    monkeypatch.setenv("MY_KEY", "from-env")
    store = Config().get_secret_store()
    assert isinstance(store, LocalSecretStore)
    assert store.get("MY_KEY") == "from-env"


def test_default_backend_is_local(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_SECRET_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # an ambient .agentenv/config.toml must not leak in
    store = Config().get_secret_store()
    assert isinstance(store, LocalSecretStore)
    assert store.get("definitely_absent_key") is None
    assert store._load() == {}


def test_aws_selector_raises_actionable_config_error(monkeypatch):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.setenv("AGENT_ENV_SECRET_STORE", "aws")
    with pytest.raises(ConfigError, match=r"stores\.secret"):
        Config().get_secret_store()


def test_config_toml_selects_local_file_backend(monkeypatch, tmp_path):
    secret_file = tmp_path / "secrets.yaml"
    secret_file.write_text("litellm_api_key: sk-file\n")
    monkeypatch.delenv("AGENT_ENV_SECRET_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path,
        '[stores.secret]\nimpl = "agent_env.store.secret_store:LocalSecretStore"\n'
        f'[stores.secret.config]\nfile_path = "{secret_file}"\n',
    )
    store = Config().get_secret_store()
    assert isinstance(store, LocalSecretStore)
    assert store.get("litellm_api_key") == "sk-file"


def test_config_toml_selects_custom_impl(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_SECRET_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, _FAKE_SECTION)
    assert isinstance(Config().get_secret_store(), LocalSecretStore)


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("AGENT_ENV_SECRET_STORE", "bogus")
    with pytest.raises(ValueError, match="AGENT_ENV_SECRET_STORE"):
        Config().get_secret_store()


def test_get_secret_reads_through_secret_store():
    cfg = Config()
    cfg.set_secret_store(LocalSecretStore(values={"litellm_api_key": "sk-x"}, use_env=False))
    assert cfg._get_secret()["litellm_api_key"] == "sk-x"


def test_get_secret_on_an_empty_bundle_raises_actionable_config_error():
    cfg = Config()
    cfg.set_secret_store(LocalSecretStore(use_env=False))
    with pytest.raises(ConfigError, match=r"stores\.secret"):
        cfg._get_secret()
    # Also a KeyError, so key-tolerant callers degrade instead of failing outright.
    with pytest.raises(KeyError):
        cfg._get_secret()


def test_get_secret_requires_bundle_backed_store():
    class _BareSecretStore(SecretStore):
        def get(self, name):
            return None

    cfg = Config()
    cfg.set_secret_store(_BareSecretStore())
    with pytest.raises(ConfigError):
        cfg._get_secret()
