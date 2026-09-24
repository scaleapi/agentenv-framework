"""config.toml-declared state providers + the provider-owned deploy-context hook — together, what
lets a backend live outside this package and still be acquired by the deploy path."""

import textwrap
from dataclasses import dataclass

import pytest

from agent_env.config import ConfigError
from agent_env.providers.state import env_state_provider
from agent_env.config import reset_config
from agent_env.providers.state.env_state_provider import (
    EnvStateInstance,
    EnvStateProvider,
    LOCAL_POSTGRES_STATE_TYPE,
    StateContext,
    acquire_state_for_deploy,
    build_state_provider,
)
from agent_env.providers.state.local_postgres import LocalPostgresStateProvider


@dataclass
class _RecordingContext(StateContext):
    ttl_seconds: int = 10800
    name_hint: str | None = None


class _RecordingStateProvider(EnvStateProvider):
    type = "custom_test"

    def __init__(self, **config):
        self.config = config
        self.acquired_with = None

    def deploy_state_context(self, *, ttl_seconds: int, name_hint: str | None) -> StateContext:
        return _RecordingContext(ttl_seconds=ttl_seconds, name_hint=name_hint)

    async def acquire(self, ctx: StateContext) -> EnvStateInstance:
        self.acquired_with = ctx
        return EnvStateInstance(state_type=self.type, instance_id="custom-1")

    async def _teardown(self, instance: EnvStateInstance) -> None:
        return None


class _UnwiredStateProvider(_RecordingStateProvider):
    type = "unwired_test"
    deploy_state_context = EnvStateProvider.deploy_state_context


class _MisnamedStateProvider(_RecordingStateProvider):
    type = "not_the_table_name"


class _NotAProvider:
    pass


_HERE = "tst.unit.providers.state.test_config_state_providers"


def _write_config(tmp_path, body: str):
    cfg = tmp_path / ".agentenv" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(textwrap.dedent(body))
    return cfg


@pytest.fixture(autouse=True)
def _reset_state_registry():
    reset_config()
    yield
    reset_config()


# --- registry ---


def test_absent_config_resolves_builtins_and_rejects_unknown(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert isinstance(build_state_provider(LOCAL_POSTGRES_STATE_TYPE), LocalPostgresStateProvider)
    with pytest.raises(ValueError, match="Unknown env state type"):
        build_state_provider("snowflake")


def test_custom_provider_registered_table_form(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [state.providers.custom_test]
        impl = "{_HERE}:_RecordingStateProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    assert isinstance(build_state_provider("custom_test"), _RecordingStateProvider)


def test_custom_provider_registered_string_form(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [state.providers]
        custom_test = "{_HERE}:_RecordingStateProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    assert isinstance(build_state_provider("custom_test"), _RecordingStateProvider)


def test_custom_provider_receives_interpolated_config(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [state.providers.custom_test]
        impl = "{_HERE}:_RecordingStateProvider"
        [state.providers.custom_test.config]
        region = "env:MY_TEST_STATE_REGION?us-east-1"
        dbname = "runs"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    monkeypatch.delenv("MY_TEST_STATE_REGION", raising=False)
    provider = build_state_provider("custom_test")
    assert provider.config == {"region": "us-east-1", "dbname": "runs"}


def test_a_broken_entry_is_caught_even_when_another_backend_is_requested(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [state.providers.custom_test]
        impl = "does.not.exist:Nope"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError):
        build_state_provider(LOCAL_POSTGRES_STATE_TYPE)


def test_a_failed_merge_does_not_memoize_a_partial_registry(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [state.providers.custom_test]
        impl = "does.not.exist:Nope"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError):
        build_state_provider(LOCAL_POSTGRES_STATE_TYPE)
    # Must fail on EVERY call: a half-built registry would carry local_postgres and pass.
    with pytest.raises(ConfigError):
        build_state_provider(LOCAL_POSTGRES_STATE_TYPE)


def test_collision_with_a_builtin_fails_loud(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [state.providers.{LOCAL_POSTGRES_STATE_TYPE}]
        impl = "{_HERE}:_RecordingStateProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="collides with a built-in"):
        build_state_provider(LOCAL_POSTGRES_STATE_TYPE)


def test_missing_impl_fails_loud(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [state.providers.custom_test]
        ttl_seconds = 60
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="missing an 'impl'"):
        build_state_provider("custom_test")


def test_impl_of_the_wrong_base_fails_loud(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [state.providers.custom_test]
        impl = "{_HERE}:_NotAProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError):
        build_state_provider("custom_test")


def test_registry_name_must_equal_the_impls_type(monkeypatch, tmp_path):
    """The name is the reconnect identity: a mismatch would strand the store it provisions."""
    cfg = _write_config(tmp_path, f"""
        [state.providers.custom_test]
        impl = "{_HERE}:_MisnamedStateProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="must match"):
        build_state_provider("custom_test")


def test_a_malformed_entry_fails_as_a_config_error(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [state.providers]
        custom_test = 5
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="must be a 'module:Class' string or a table"):
        build_state_provider("custom_test")


def test_stray_keys_beside_impl_are_reported(monkeypatch, tmp_path, caplog):
    cfg = _write_config(tmp_path, f"""
        [state.providers.custom_test]
        impl = "{_HERE}:_RecordingStateProvider"
        region = "us-west-2"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    provider = build_state_provider("custom_test")
    assert provider.config == {}
    assert "region" in caplog.text


# --- the deploy hook ---


def test_local_postgres_allocates_nothing_upstream():
    assert LocalPostgresStateProvider().deploy_state_context(ttl_seconds=60, name_hint="x") is None


@pytest.mark.asyncio
async def test_acquire_for_deploy_returns_none_for_local(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert await acquire_state_for_deploy(env_state_type=LOCAL_POSTGRES_STATE_TYPE) is None
    assert await acquire_state_for_deploy() is None  # default backend


@pytest.mark.asyncio
async def test_out_of_tree_provider_is_acquirable(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [state.providers.custom_test]
        impl = "{_HERE}:_RecordingStateProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    instance = await acquire_state_for_deploy(
        env_state_type="custom_test", ttl_seconds=42, name_hint="hint-1"
    )
    assert instance is not None
    assert instance.state_type == "custom_test"
    # the store is reachable again from its persisted identity — the reattach/teardown path
    assert isinstance(build_state_provider(instance.state_type), _RecordingStateProvider)


@pytest.mark.asyncio
async def test_run_identity_reaches_the_providers_context(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [state.providers.custom_test]
        impl = "{_HERE}:_RecordingStateProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    captured = {}

    class _Capturing(_RecordingStateProvider):
        async def acquire(self, ctx):
            captured["ctx"] = ctx
            return await super().acquire(ctx)

    monkeypatch.setattr(env_state_provider, "build_state_provider", lambda t: _Capturing())
    await acquire_state_for_deploy(env_state_type="custom_test", ttl_seconds=42, name_hint="hint-1")
    assert (captured["ctx"].ttl_seconds, captured["ctx"].name_hint) == (42, "hint-1")


@pytest.mark.asyncio
async def test_a_backend_without_the_hook_fails_loud(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [state.providers.unwired_test]
        impl = "{_HERE}:_UnwiredStateProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(NotImplementedError, match="not wired into the deploy path"):
        await acquire_state_for_deploy(env_state_type="unwired_test")
