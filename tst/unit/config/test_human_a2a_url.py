"""Human-A2A base URL resolution: env var > configure() > [conversations]; unconfigured raises."""

import pytest

from agent_env.config.errors import ConfigError
from tst.util.config import config_with_document
from agent_env.config.runtime import (
    Config,
    configure,
    get_config,
    reset_config,
)
from agent_env.store.secret_store import SecretStore

ENV_VAR = "AGENT_ENV_HUMAN_A2A_URL"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)
    monkeypatch.delenv("AGENT_ENV_ENVIRONMENT", raising=False)


def _config(config_file=None, **kwargs):
    """A Config with its config.toml read stubbed — no file discovery, no AWS."""
    return config_with_document({} if config_file is None else config_file, **kwargs)


# --- no built-in default: unconfigured resolution fails loud -------------------

def test_unconfigured_resolution_raises_naming_the_key_and_the_env_var():
    with pytest.raises(ConfigError, match=r"\[conversations\] default_human_a2a_url"):
        _config().get_default_human_a2a_url()
    with pytest.raises(ConfigError, match="AGENT_ENV_HUMAN_A2A_URL"):
        _config().get_default_human_a2a_url()


def test_the_stage_no_longer_selects_a_default(monkeypatch):
    for stage in ("dev", "prod"):
        monkeypatch.setenv("AGENT_ENV_ENVIRONMENT", stage)
        with pytest.raises(ConfigError):
            _config().get_default_human_a2a_url()


def test_an_empty_conversations_section_still_raises():
    with pytest.raises(ConfigError):
        _config(config_file={"conversations": {}}).get_default_human_a2a_url()


# --- layer 3: the config.toml key ---------------------------------------------

def test_config_key_resolves():
    cfg = _config(config_file={"conversations": {"default_human_a2a_url": "https://hub.example/a2a"}})
    assert cfg.get_default_human_a2a_url() == "https://hub.example/a2a"


def test_config_key_resolves_env_references():
    cfg = _config(config_file={"conversations": {"default_human_a2a_url": "env:MISSING?https://fallback/a2a"}})
    assert cfg.get_default_human_a2a_url() == "https://fallback/a2a"


class _RecordingSecretStore(SecretStore):
    def __init__(self, values):
        self.values = values
        self.calls = []

    def get(self, name):
        self.calls.append(name)
        return self.values.get(name)


def test_a_secret_reference_resolves_through_the_configured_secret_store():
    cfg = _config(config_file={"conversations": {"default_human_a2a_url": "secret:hub_url"}})
    store = _RecordingSecretStore({"hub_url": "https://from-secrets/a2a"})
    cfg.set_secret_store(store)
    assert cfg.get_default_human_a2a_url() == "https://from-secrets/a2a"
    assert store.calls == ["hub_url"]


def test_a_non_table_conversations_section_is_refused_as_mis_shaped():
    cfg = _config(config_file={"conversations": "nonsense"})
    with pytest.raises(ConfigError, match=r"\[conversations\] must be a table, got str"):
        cfg.get_default_human_a2a_url()


@pytest.mark.parametrize("bad", [8000, True, ["https://x/a2a"], {"url": "https://x"}, "", "   "])
def test_a_non_string_or_blank_config_value_fails_at_the_config_seam(bad):
    cfg = _config(config_file={"conversations": {"default_human_a2a_url": bad}})
    with pytest.raises(ConfigError, match="must be a non-empty string URL"):
        cfg.get_default_human_a2a_url()


# --- layer 2: an explicit configure() value -----------------------------------

def test_configure_accepts_the_override_and_it_reaches_the_singleton():
    try:
        configure(default_human_a2a_url="https://explicit/a2a")
        assert get_config().get_default_human_a2a_url() == "https://explicit/a2a"
    finally:
        reset_config()


def test_configure_without_the_override_leaves_the_field_unset():
    try:
        configure()
        assert get_config().default_human_a2a_url is None
    finally:
        reset_config()


def test_explicit_value_beats_the_config_key():
    cfg = _config(
        config_file={"conversations": {"default_human_a2a_url": "https://hub.example/a2a"}},
        default_human_a2a_url="https://explicit/a2a",
    )
    assert cfg.get_default_human_a2a_url() == "https://explicit/a2a"


@pytest.mark.parametrize("bad", [8000, True, ["https://x/a2a"], {"url": "https://x"}, "", "   "])
def test_a_non_string_or_blank_explicit_value_fails_like_a_config_one(bad):
    cfg = _config(default_human_a2a_url=bad)
    with pytest.raises(ConfigError, match="must be a non-empty string URL"):
        cfg.get_default_human_a2a_url()


# --- layer 1: the env var is absolute -----------------------------------------

def test_env_var_beats_the_config_key(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "http://localhost:18000/api/v1/a2a/human")
    cfg = _config(config_file={"conversations": {"default_human_a2a_url": "https://hub.example/a2a"}})
    assert cfg.get_default_human_a2a_url() == "http://localhost:18000/api/v1/a2a/human"


def test_env_var_beats_an_explicit_value(monkeypatch):
    # Deliberate: local dev's env-var repoint must stick even over a configured URL.
    monkeypatch.setenv(ENV_VAR, "http://localhost:18000/api/v1/a2a/human")
    cfg = _config(default_human_a2a_url="https://explicit/a2a")
    assert cfg.get_default_human_a2a_url() == "http://localhost:18000/api/v1/a2a/human"


def test_env_var_alone_resolves(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "http://localhost:18000/api/v1/a2a/human")
    assert _config().get_default_human_a2a_url() == "http://localhost:18000/api/v1/a2a/human"


# --- the config.toml read must stay lazy --------------------------------------

def test_the_env_var_does_not_require_reading_a_config_file(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "http://localhost:18000/api/v1/a2a/human")
    cfg = Config()  # the document deliberately left unstubbed
    cfg._config_file = lambda: pytest.fail("config file read on the env-var path")
    assert cfg.get_default_human_a2a_url() == "http://localhost:18000/api/v1/a2a/human"


def test_constructing_config_reads_no_config_file():
    cfg = Config()
    assert cfg._snapshot is None  # constructing a Config resolves no document
    assert cfg.default_human_a2a_url is None
