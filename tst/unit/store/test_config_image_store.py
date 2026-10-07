"""Backend injection + AGENT_ENV_IMAGE_STORE / config.toml selection for get_image_store().

Infra-free (the tst/unit socket guard forbids network): injection returns a fake
verbatim, and the ``local`` selector / a custom ``impl`` build real stores.
"""

import pytest

from agent_env.config import Config, configure, get_config, set_image_store
from agent_env.store import (
    ConfigError,
    EcrCredentials,
    EcrImageStore,
    LocalRegistryImageStore,
    OciRegistryImageStore,
    SecretStoreCredentials,
)
from tst.unit.store.fakes import FakeImageStore

_FAKE_SECTION = '[stores.image]\nimpl = "tst.unit.store.fakes:FakeImageStore"\n'


def _write_config(tmp_path, body):
    agentenv = tmp_path / ".agentenv"
    agentenv.mkdir(exist_ok=True)
    (agentenv / "config.toml").write_text(body)


def test_set_image_store_returns_it_verbatim():
    cfg = Config()
    fake = FakeImageStore()
    cfg.set_image_store(fake)
    assert cfg.get_image_store() is fake  # no ECR client built — override short-circuits


def test_module_level_set_image_store_overrides_singleton():
    fake = FakeImageStore()
    set_image_store(fake)
    assert get_config().get_image_store() is fake


def test_configure_carries_image_store():
    fake = FakeImageStore()
    configure(image_store=fake)
    assert get_config().get_image_store() is fake


def test_env_selector_builds_local_registry(monkeypatch):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.setenv("AGENT_ENV_IMAGE_STORE", "local")
    store = Config().get_image_store()
    assert isinstance(store, LocalRegistryImageStore)
    assert store.image_ref("my-agent", "v3") == "localhost:5000/my-agent:v3"


def test_default_backend_is_local(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_IMAGE_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    section = Config()._resolve_image_section()
    assert section["impl"] == "agent_env.store.image_store:LocalRegistryImageStore"


def test_an_unknown_backend_names_the_table_and_the_overriding_env_var(monkeypatch):
    monkeypatch.setenv("AGENT_ENV_IMAGE_STORE", "hosted")
    with pytest.raises(ConfigError, match=r"\[stores\.image\].*AGENT_ENV_IMAGE_STORE"):
        Config().get_image_store()


def test_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("AGENT_ENV_IMAGE_STORE", "bogus")
    with pytest.raises(ValueError, match="AGENT_ENV_IMAGE_STORE"):
        Config().get_image_store()


def test_config_toml_selects_custom_impl(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_IMAGE_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, _FAKE_SECTION)
    assert isinstance(Config().get_image_store(), FakeImageStore)


def test_env_override_beats_config_toml(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, _FAKE_SECTION)
    monkeypatch.setenv("AGENT_ENV_IMAGE_STORE", "local")
    assert isinstance(Config().get_image_store(), LocalRegistryImageStore)


def test_ecr_from_config_builds_store_lazily_without_network(monkeypatch):
    store = EcrImageStore.from_config(
        registry_host="123.dkr.ecr.us-west-2.amazonaws.com",
        repository_prefix="example/images-dev",
    )
    assert isinstance(store, EcrImageStore)
    assert store.image_ref("foo", "v1") == "123.dkr.ecr.us-west-2.amazonaws.com/example/images-dev/foo:v1"


def test_ecr_rejects_a_non_ecr_credential_provider():
    with pytest.raises(TypeError, match="EcrCredentials"):
        EcrImageStore.from_config(
            registry_host="123.dkr.ecr.us-west-2.amazonaws.com",
            credentials=SecretStoreCredentials(),
        )


# ── registry_host accessor ──────────────────────────────────────────────────
# ECR-specific: the exporter reads the host from a configured EcrImageStore,
# replacing its hardcoded `ECR_HOST` (which ignored [stores.image] overrides).


def test_ecr_store_exposes_its_registry_host():
    store = EcrImageStore.from_config(
        registry_host="123.dkr.ecr.us-west-2.amazonaws.com",
        repository_prefix="example/images-dev",
    )
    assert store.registry_host == "123.dkr.ecr.us-west-2.amazonaws.com"
    # Host only — no repository prefix, unlike image_ref().
    assert "images-dev" not in store.registry_host


def test_registry_host_is_the_host_component_of_image_ref():
    """The two must agree, or a docker-login lands on a different registry."""
    for store in (
        EcrImageStore.from_config(registry_host="123.dkr.ecr.eu-west-1.amazonaws.com"),
        EcrImageStore.from_config(
            registry_host="123.dkr.ecr.eu-west-1.amazonaws.com", repository_prefix="a/b"
        ),
    ):
        assert store.image_ref("repo", "v1").split("/", 1)[0] == store.registry_host


def test_config_toml_registry_host_reaches_the_store(monkeypatch, tmp_path):
    """The point of the accessor: an override is what callers observe."""
    monkeypatch.delenv("AGENT_ENV_IMAGE_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path,
        '[stores.image]\nimpl = "agent_env.store.image_store:EcrImageStore"\n'
        '[stores.image.config]\n'
        'registry_host = "999.dkr.ecr.eu-central-1.amazonaws.com"\n'
        'repository_prefix = "team/images"\n',
    )
    assert Config().get_image_store().registry_host == "999.dkr.ecr.eu-central-1.amazonaws.com"


def test_config_toml_builds_generic_registry_with_nested_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_IMAGE_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.setenv(
        "REGISTRY_AUTHS",
        '{"auths":{"ghcr.io":{"username":"octocat","password":"pat"}}}',
    )
    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path,
        '[stores.image]\nimpl = "agent_env.store.image_store:OciRegistryImageStore"\n'
        '[stores.image.config]\nregistry_host = "ghcr.io"\nrepository_prefix = "example/images"\n'
        '[stores.image.config.credentials]\n'
        'impl = "agent_env.store.image_store:SecretStoreCredentials"\n'
        'secret_key = "REGISTRY_AUTHS"\n',
    )

    store = Config().get_image_store()
    assert isinstance(store, OciRegistryImageStore)
    assert isinstance(store.credentials, SecretStoreCredentials)
    assert store.image_ref("server", "v1") == "ghcr.io/example/images/server:v1"
    auth = store.auth("ghcr.io/example/images/server:v1")
    assert auth is not None
    assert auth.username == "octocat"


def test_ecr_store_without_credentials_builds_bare_ecr_credentials():
    store = EcrImageStore.from_config(registry_host="123.dkr.ecr.us-west-2.amazonaws.com")
    assert isinstance(store.credentials, EcrCredentials)


def test_ecr_credentials_without_a_region_fail_actionably_at_client_time():
    creds = EcrCredentials()
    with pytest.raises(ConfigError, match="region"):
        _ = creds.client


def test_generic_ecr_credentials_are_declared_in_config(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_IMAGE_STORE", raising=False)
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.setenv("TEST_ECR_ACCESS_KEY", "access")
    monkeypatch.setenv("TEST_ECR_SECRET_KEY", "secret")
    monkeypatch.chdir(tmp_path)
    _write_config(
        tmp_path,
        '[stores.image]\nimpl = "agent_env.store.image_store:OciRegistryImageStore"\n'
        '[stores.image.config]\nregistry_host = "123.dkr.ecr.us-west-2.amazonaws.com"\n'
        '[stores.image.config.credentials]\n'
        'impl = "agent_env.store.image_store:EcrCredentials"\n'
        'region = "us-west-2"\n'
        'access_key = "secret:TEST_ECR_ACCESS_KEY"\n'
        'secret_key = "secret:TEST_ECR_SECRET_KEY"\n',
    )

    store = Config().get_image_store()

    assert isinstance(store, OciRegistryImageStore)
    assert isinstance(store.credentials, EcrCredentials)
