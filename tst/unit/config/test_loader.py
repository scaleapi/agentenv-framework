"""Unit tests for the .agentenv/config.toml loader.

Backend-agnostic: discovery, parsing, env:/secret: interpolation, and the
import + ABC-guard + from_config construction step, driven with fakes (no
network — the tst/unit socket guard forbids it).
"""

import pytest

from agent_env.config import ConfigError
from agent_env.config import loader as config_loader
from agent_env.store.document_store import DocumentStore
from tst.unit.store.fakes import FakeDocumentStore

_SECRET = {"mongodb_uri": "mongodb://real", "blank": ""}


def _resolver(key):
    return _SECRET.get(key)


class _MarkerStore(FakeDocumentStore):
    """Records the kwargs from_config received (proves build_store routes through from_config)."""

    @classmethod
    def from_config(cls, **config):
        inst = cls()
        inst.received = config
        return inst


# --- discovery ---


def test_discover_prefers_env_override(monkeypatch, tmp_path):
    target = tmp_path / "custom.toml"
    target.write_text("")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(target))
    assert config_loader.discover_config_path() == target


def test_discover_env_override_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "nope.toml"))
    with pytest.raises(ConfigError, match="AGENT_ENV_CONFIG"):
        config_loader.discover_config_path()


def test_discover_walks_up_to_nearest_agentenv(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    cfg = tmp_path / ".agentenv" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert config_loader.discover_config_path(start=nested) == cfg.resolve()


def test_discover_returns_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    assert config_loader.discover_config_path(start=tmp_path) is None


# --- parsing ---


def test_load_config_file_none_returns_empty():
    assert config_loader.load_config_file(None) == {}


def test_load_config_file_with_no_path_at_all_is_empty():
    """No config anywhere is the bare install, and it resolves to built-in defaults."""
    assert config_loader.load_config_file(None) == {}


def test_a_path_that_is_not_there_is_empty_not_an_error(tmp_path):
    """Deliberately lenient: this is public API and the sdk's exporter calls it with a path
    of its own. Knowing a path was *just discovered* — so that losing it is a race — belongs
    to the caller that discovered it. `snapshot` owns that check."""
    assert config_loader.load_config_file(tmp_path / "nope.toml") == {}


def test_a_utf8_bom_is_stripped_rather_than_failing_at_line_one(tmp_path):
    """An editor that writes a BOM produces a file correct to its author and rejected by
    tomllib at line 1, column 1 — an error that names nothing the author can see."""
    path = tmp_path / "config.toml"
    path.write_bytes(b"\xef\xbb\xbf[stores.document]\nimpl = \"x:Y\"\n")

    assert config_loader.load_config_file(path) == {"stores": {"document": {"impl": "x:Y"}}}


def test_a_file_that_is_not_utf8_says_so(tmp_path):
    path = tmp_path / "config.toml"
    path.write_bytes(b'[stores]\nimpl = "\xff\xfe not utf-8"\n')

    with pytest.raises(config_loader.ConfigError, match="is not valid UTF-8"):
        config_loader.load_config_file(path)


def test_load_config_file_parses(tmp_path):
    p = tmp_path / "c.toml"
    p.write_text('[stores.document]\nimpl = "x:Y"\n')
    assert config_loader.load_config_file(p) == {"stores": {"document": {"impl": "x:Y"}}}


def test_load_config_file_malformed_raises(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("this is = = not toml")
    with pytest.raises(ConfigError, match="Malformed"):
        config_loader.load_config_file(p)


# --- interpolation ---


def test_interpolate_passes_plain_string_through():
    assert config_loader.interpolate("plain", secret_resolver=_resolver) == "plain"


def test_interpolate_passes_non_string_through():
    assert config_loader.interpolate(42, secret_resolver=_resolver) == 42
    assert config_loader.interpolate(True, secret_resolver=_resolver) is True


def test_interpolate_env(monkeypatch):
    monkeypatch.setenv("SOME_VAR", "barval")
    assert config_loader.interpolate("env:SOME_VAR", secret_resolver=_resolver) == "barval"


def test_interpolate_env_default(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    assert config_loader.interpolate("env:MISSING_VAR?fallback", secret_resolver=_resolver) == "fallback"


def test_interpolate_env_missing_no_default_raises(monkeypatch):
    monkeypatch.delenv("MISSING_VAR", raising=False)
    with pytest.raises(ConfigError, match="env:MISSING_VAR"):
        config_loader.interpolate("env:MISSING_VAR", secret_resolver=_resolver)


def test_interpolate_secret():
    assert config_loader.interpolate("secret:mongodb_uri", secret_resolver=_resolver) == "mongodb://real"


def test_interpolate_secret_default():
    assert config_loader.interpolate("secret:absent?agent_env", secret_resolver=_resolver) == "agent_env"


def test_interpolate_secret_missing_no_default_raises():
    with pytest.raises(ConfigError, match="secret:absent"):
        config_loader.interpolate("secret:absent", secret_resolver=_resolver)


def test_interpolate_empty_secret_is_a_value_not_missing():
    assert config_loader.interpolate("secret:blank", secret_resolver=_resolver) == ""


def test_interpolate_recurses_into_dict(monkeypatch):
    monkeypatch.setenv("SOME_VAR", "v")
    out = config_loader.interpolate({"a": "env:SOME_VAR", "b": "plain"}, secret_resolver=_resolver)
    assert out == {"a": "v", "b": "plain"}


def test_interpolate_secret_without_resolver_raises():
    with pytest.raises(ConfigError, match="secret store"):
        config_loader.interpolate("secret:mongodb_uri")


# --- build_store ---


def test_build_store_default_from_config():
    section = {"impl": "tst.unit.store.fakes:FakeDocumentStore", "config": {"fail_inserts": 2}}
    store = config_loader.build_store(section, DocumentStore, secret_resolver=_resolver)
    assert isinstance(store, FakeDocumentStore)
    assert store._to_fail == 2


def test_build_store_routes_through_from_config_and_interpolates(monkeypatch):
    monkeypatch.setenv("ENDPOINT", "https://real")
    section = {
        "impl": "tst.unit.config.test_loader:_MarkerStore",
        "config": {"endpoint": "env:ENDPOINT", "plain": "x"},
    }
    store = config_loader.build_store(section, DocumentStore, secret_resolver=_resolver)
    assert isinstance(store, _MarkerStore)
    assert store.received == {"endpoint": "https://real", "plain": "x"}


def test_build_store_missing_impl_raises():
    with pytest.raises(ConfigError, match="impl"):
        config_loader.build_store({"config": {}}, DocumentStore, secret_resolver=_resolver)


def test_build_store_bad_impl_format_raises():
    with pytest.raises(ConfigError, match="module.path:ClassName"):
        config_loader.build_store({"impl": "no_colon"}, DocumentStore, secret_resolver=_resolver)


def test_build_store_unimportable_module_raises():
    with pytest.raises(ConfigError, match="Cannot import"):
        config_loader.build_store(
            {"impl": "nonexistent.module:Thing"}, DocumentStore, secret_resolver=_resolver
        )


def test_build_store_missing_attr_raises():
    with pytest.raises(ConfigError, match="Cannot import"):
        config_loader.build_store(
            {"impl": "agent_env.store.document_store:NoSuchClass"},
            DocumentStore,
            secret_resolver=_resolver,
        )


def test_build_store_abc_guard_raises():
    with pytest.raises(ConfigError, match="not a subclass"):
        config_loader.build_store({"impl": "builtins:dict"}, DocumentStore, secret_resolver=_resolver)


def test_build_store_secret_ref_without_resolver_raises():
    section = {"impl": "tst.unit.store.fakes:FakeDocumentStore", "config": {"x": "secret:k"}}
    with pytest.raises(ConfigError, match="secret store"):
        config_loader.build_store(section, DocumentStore)


# --- load_impl (the shared import + ABC-guard seam) ---


def test_load_impl_returns_the_class_not_an_instance():
    cls = config_loader.load_impl("tst.unit.store.fakes:FakeDocumentStore", DocumentStore)
    assert cls is FakeDocumentStore


def test_load_impl_bad_format_raises():
    with pytest.raises(ConfigError, match="module.path:ClassName"):
        config_loader.load_impl("no_colon", DocumentStore)


def test_load_impl_unimportable_module_raises():
    with pytest.raises(ConfigError, match="Cannot import"):
        config_loader.load_impl("nonexistent.module:Thing", DocumentStore)


def test_load_impl_missing_attr_raises():
    with pytest.raises(ConfigError, match="Cannot import"):
        config_loader.load_impl("agent_env.store.document_store:NoSuchClass", DocumentStore)


def test_load_impl_not_subclass_raises():
    with pytest.raises(ConfigError, match="not a subclass"):
        config_loader.load_impl("builtins:dict", DocumentStore)
