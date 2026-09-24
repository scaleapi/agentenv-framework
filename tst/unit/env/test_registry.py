"""Env registry guards + config.toml-declared custom envs.

The registry is what lets stored env documents be deserialized by `Env.get` /
`EnvStore._deserialize` — the path the hub backend and the Temporal
worker use to list/load envs. An env missing here can be `put()` but never
loaded back ("Unknown env type: ..."), so it can't be browsed or run from the
hub. A custom `Env` named under `[envs]` in `.agentenv/config.toml` is imported,
ABC-guarded, and registered under its own `type` — no change to the built-in
path. Backend-agnostic; no network.
"""

from __future__ import annotations

import textwrap

import pytest

from agent_env.config import ConfigError
from agent_env.env.env import Env
from agent_env.env.registry import get_env_registry
from agent_env.env.store import EnvStore, set_env_store
from agent_env.store import reset_config, set_document_store
from tst.unit.store.fakes import FakeDocumentStore


class _CustomEnv(Env):
    type = "custom_env_test"

    @classmethod
    def from_dict(cls, data: dict) -> "_CustomEnv":
        return cls(id=data["id"], version=data.get("version"), metadata=data.get("metadata"))


class _CollidingEnv(Env):
    type = "mcp_server"


class _CustomEnvDup(Env):
    type = "custom_env_test"


class _NoTypeEnv(Env):
    pass


class _NotAnEnv:
    pass


def _write_config(tmp_path, body: str):
    cfg = tmp_path / ".agentenv" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(textwrap.dedent(body))
    return cfg


_HERE = "tst.unit.env.test_registry"


@pytest.fixture(autouse=True)
def _reset_registry():
    from agent_env.env.store import reset_env_store

    reset_config()
    yield
    reset_config()
    reset_env_store()
    reset_config()


# --- built-in registry guards ---



def test_registry_keys_match_class_type(monkeypatch, tmp_path):
    # Each registry key must equal the class's `type`, or _deserialize resolves
    # the wrong class for a stored document.
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    for key, cls in get_env_registry().items():
        assert cls.type == key, f"registry key {key!r} != {cls.__name__}.type {cls.type!r}"


# --- config.toml custom envs ---


def test_absent_config_leaves_builtins_only(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    reg = get_env_registry()
    assert "custom_env_test" not in reg
    assert "mcp_server" in reg


def test_no_envs_section_leaves_builtins_only(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, '[stores]\ndocument = "local"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reg = get_env_registry()
    assert "custom_env_test" not in reg
    assert "mcp_server" in reg


def test_empty_impls_leaves_builtins_only(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, "[envs]\nimpls = []\n")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reg = get_env_registry()
    assert "custom_env_test" not in reg
    assert "mcp_server" in reg


def test_config_toml_env_is_registered(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_CustomEnv"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    reg = get_env_registry()
    assert reg["custom_env_test"] is _CustomEnv
    assert "mcp_server" in reg


def test_collision_with_builtin_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_CollidingEnv"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="already registered"):
        get_env_registry()


def test_two_custom_envs_same_type_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_CustomEnv", "{_HERE}:_CustomEnvDup"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="already registered"):
        get_env_registry()


def test_env_without_own_type_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_NoTypeEnv"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="does not define its own 'type'"):
        get_env_registry()


def test_unimportable_impl_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [envs]
        impls = ["no.such.module:Thing"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="Cannot import"):
        get_env_registry()


def test_non_env_impl_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_NotAnEnv"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="not a subclass"):
        get_env_registry()


def test_non_list_impls_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f'[envs]\nimpls = "{_HERE}:_CustomEnv"\n')
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="must be a list"):
        get_env_registry()


def test_non_string_impl_element_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, "[envs]\nimpls = [123]\n")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="must be a 'module:Class' string"):
        get_env_registry()


def test_failed_merge_does_not_memoize_partial_registry(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [envs]
        impls = ["no.such.module:Thing"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError):
        get_env_registry()
    # A bad manifest must fail loud on EVERY call, not just the first.
    with pytest.raises(ConfigError):
        get_env_registry()


def test_store_get_routes_custom_type_through_registry(monkeypatch, tmp_path):
    """EnvStore.get -> _deserialize resolves a config-registered custom type via
    get_env_registry() — the real read path, not just the registry dict."""
    cfg = _write_config(tmp_path, f"""
        [envs]
        impls = ["{_HERE}:_CustomEnv"]
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    doc_store = FakeDocumentStore()
    doc_store.docs.append({"id": "cw1", "version": 1, "type": "custom_env_test", "metadata": {}})
    set_document_store(doc_store)
    set_env_store(EnvStore())

    loaded = Env.get("cw1")

    assert type(loaded) is _CustomEnv
    assert loaded.id == "cw1"


def test_store_get_unregistered_type_raises_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # no .agentenv -> built-ins only
    doc_store = FakeDocumentStore()
    doc_store.docs.append({"id": "u1", "version": 1, "type": "never_registered_env", "metadata": {}})
    set_document_store(doc_store)
    set_env_store(EnvStore())

    with pytest.raises(ValueError, match="Unknown env type"):
        Env.get("u1")
