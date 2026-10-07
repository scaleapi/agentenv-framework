"""Unit tests for config.toml-declared custom sandbox providers + the deploy-time type guard."""

import asyncio
import os
import textwrap

import pytest

from agent_env.config import ConfigError
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.config import reset_config
from agent_env.providers.sandbox_providers.e2b.provider import E2BSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.sandbox_providers.sail_vm.provider import SailVmSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import Sandbox
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxProvider,
    SandboxProviderTypeError,
    build_sandbox_provider,
)


class _RecordingProvider(SandboxProvider):
    """A custom, out-of-SDK provider that records the config it was built with."""

    def __init__(self, **config):
        self.config = config

    async def create_sandbox(self, **kwargs) -> Sandbox:
        raise NotImplementedError


class _NotAProvider:
    pass


class _FakeSandbox(Sandbox):
    """A Sandbox whose .type is set per-instance, recording whether it was terminated.
    `terminate_raises` exercises the guard's best-effort-cleanup branch."""

    def __init__(self, type_: str, terminate_raises: bool = False):
        self.type = type_
        self.sandbox_id = "fake-1"
        self.tunnel_urls = {}
        self.vnc_url = None
        self.mode = "container"
        self.terminated = False
        self._terminate_raises = terminate_raises

    async def terminate(self) -> None:
        if self._terminate_raises:
            raise RuntimeError("terminate boom")
        self.terminated = True


class _MintingProvider(SandboxProvider):
    """A custom provider that mints a _FakeSandbox with a configurable .type from every create_*
    method — used to exercise the deploy-time name==Sandbox.type guard (including the mismatch case)."""

    def __init__(self, produced_type: str, terminate_raises: bool = False):
        self._produced_type = produced_type
        self._terminate_raises = terminate_raises
        self.last_sandbox = None

    def _mint(self) -> Sandbox:
        self.last_sandbox = _FakeSandbox(self._produced_type, self._terminate_raises)
        return self.last_sandbox

    async def create_sandbox(self, **kwargs) -> Sandbox:
        return self._mint()

    async def create_vm(self, **kwargs) -> Sandbox:
        return self._mint()

    async def create_container(self, **kwargs) -> Sandbox:
        return self._mint()

    async def get_sandbox(self, sandbox_id: str) -> Sandbox:
        return _FakeSandbox(self._produced_type, self._terminate_raises)


_HERE = "tst.unit.providers.sandbox_providers.test_config_sandbox_providers"


def _write_config(tmp_path, body: str):
    cfg = tmp_path / ".agentenv" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(textwrap.dedent(body))
    return cfg


@pytest.fixture(autouse=True)
def _reset_sandbox_state():
    def _reset():
        reset_config()
        sandbox_provider.reset_sandbox_provider()
        sandbox_provider.reset_env_sandbox_provider()
        sandbox_provider.reset_agent_sandbox_provider()

    _reset()
    yield
    _reset()


# --- registry: custom providers + reconnect ---


def test_absent_config_resolves_builtin_and_rejects_unknown(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert isinstance(build_sandbox_provider("local"), LocalSandboxProvider)
    with pytest.raises(ValueError, match="Unknown sandbox backend"):
        build_sandbox_provider("custom_test")


def test_custom_provider_registered(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers.custom_test]
        impl = "{_HERE}:_RecordingProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    assert isinstance(build_sandbox_provider("custom_test"), _RecordingProvider)


def test_custom_provider_string_form(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers]
        custom_test = "{_HERE}:_RecordingProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    assert isinstance(build_sandbox_provider("custom_test"), _RecordingProvider)


def test_custom_provider_receives_interpolated_config(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers.custom_test]
        impl = "{_HERE}:_RecordingProvider"
        [sandbox.providers.custom_test.config]
        region = "env:MY_TEST_REGION?us-east-1"
        instance_type = "m6i.xlarge"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    monkeypatch.delenv("MY_TEST_REGION", raising=False)
    provider = build_sandbox_provider("custom_test")
    assert provider.config == {"region": "us-east-1", "instance_type": "m6i.xlarge"}


def test_reconnect_round_trip_resolves_custom_by_persisted_type(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers.custom_test]
        impl = "{_HERE}:_RecordingProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    # A deployed record persists the concrete sandbox's .type; reconnect rebuilds from it.
    persisted_sandbox_type = "custom_test"
    provider = build_sandbox_provider(persisted_sandbox_type)
    assert isinstance(provider, _RecordingProvider)


def test_collision_with_builtin_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers.local]
        impl = "{_HERE}:_RecordingProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="collides with a built-in"):
        build_sandbox_provider("local")


def test_builtin_string_form_collision_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers]
        modal = "{_HERE}:_RecordingProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="collides with a built-in"):
        build_sandbox_provider("modal")


def test_builtin_accepts_a_config_only_table(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers.modal.config]
        ecr_pull_secret_name = "my-ecr-reader"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    provider = build_sandbox_provider("modal")
    assert provider._ecr_pull_secret_name == "my-ecr-reader"


def test_e2b_builtin_receives_interpolated_key_and_versioned_base(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers.e2b.config]
        api_key = "env:E2B_TEST_API_KEY"
        base_template = "agent-env-docker-base-v1"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    monkeypatch.setenv("E2B_TEST_API_KEY", "resolved-test-key")

    provider = build_sandbox_provider("e2b")

    assert isinstance(provider, E2BSandboxProvider)
    assert provider._api_key == "resolved-test-key"
    assert provider.base_template == "agent-env-docker-base-v1"


def test_e2b_missing_base_template_is_a_config_error(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers.e2b.config]
        api_key = "env:E2B_TEST_API_KEY"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    monkeypatch.setenv("E2B_TEST_API_KEY", "resolved-test-key")

    with pytest.raises(ConfigError, match="requires a non-empty 'base_template'"):
        build_sandbox_provider("e2b")


def test_sail_builtin_receives_interpolated_key_without_touching_the_sdk(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers.sail_vm.config]
        api_key = "env:SAIL_TEST_API_KEY"
        app = "agent-env-test"
        auto_sleep_min_idle_seconds = 600
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    monkeypatch.setenv("SAIL_TEST_API_KEY", "resolved-sail-key")
    monkeypatch.delenv("SAIL_API_KEY", raising=False)

    provider = build_sandbox_provider("sail_vm")

    assert isinstance(provider, SailVmSandboxProvider)
    assert provider._api_key == "resolved-sail-key"
    assert provider._app_name == "agent-env-test"
    assert provider._auto_sleep_min_idle_seconds == 600
    assert "SAIL_API_KEY" not in os.environ
    assert "resolved-sail-key" not in repr(provider)


def test_sail_missing_api_key_is_a_config_error(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers.sail_vm.config]
        app = "agent-env-test"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))

    with pytest.raises(ConfigError, match="requires a non-empty 'api_key'"):
        build_sandbox_provider("sail_vm")


def test_builtin_config_reaches_chain_members(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers.modal.config]
        ecr_pull_secret_name = "my-ecr-reader"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    chain = build_sandbox_provider("local,modal")
    assert chain._providers[1]._ecr_pull_secret_name == "my-ecr-reader"


def test_builtin_table_with_non_config_keys_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers.modal]
        region = "us-west-2"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="only a 'config' table"):
        build_sandbox_provider("modal")


def test_builtin_non_table_config_values_raise(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers.modal]
        config = "not-a-table"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="must be a table"):
        build_sandbox_provider("modal")


def test_builtin_scalar_entry_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers]
        modal = 3
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="collides with a built-in"):
        build_sandbox_provider("modal")


def test_missing_impl_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers.custom_test.config]
        region = "us-west-2"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="missing an 'impl'"):
        build_sandbox_provider("custom_test")


def test_unimportable_impl_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers.custom_test]
        impl = "no.such.module:Thing"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="Cannot import"):
        build_sandbox_provider("custom_test")


def test_non_subclass_impl_raises(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers.custom_test]
        impl = "{_HERE}:_NotAProvider"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError, match="not a subclass"):
        build_sandbox_provider("custom_test")


def test_failed_merge_does_not_memoize_partial_registry(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox.providers.custom_test]
        impl = "no.such.module:Thing"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    with pytest.raises(ConfigError):
        build_sandbox_provider("local")
    # A bad manifest must fail loud on EVERY call, not just the first.
    with pytest.raises(ConfigError):
        build_sandbox_provider("local")


# --- config-driven getter defaults ---


def test_default_spec_falls_back_to_builtins(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    assert sandbox_provider._default_sandbox_spec() == "local"
    assert sandbox_provider._agent_sandbox_spec() == "local"


def test_default_getter_honors_config(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox]
        default = "local"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    assert isinstance(sandbox_provider.get_sandbox_provider(), LocalSandboxProvider)
    assert isinstance(sandbox_provider.get_env_sandbox_provider(), LocalSandboxProvider)


def test_agent_default_getter_honors_config(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox]
        agent_default = "local"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    assert isinstance(sandbox_provider.get_agent_sandbox_provider(), LocalSandboxProvider)


def test_explicit_set_beats_config_default(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, """
        [sandbox]
        default = "local"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    injected = _RecordingProvider()
    sandbox_provider.set_sandbox_provider(injected)
    assert sandbox_provider.get_sandbox_provider() is injected


# --- deploy-time name == Sandbox.type guard ---


def test_type_mismatch_terminates_sandbox_and_fails_loud(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers.myprov]
        impl = "{_HERE}:_MintingProvider"
        [sandbox.providers.myprov.config]
        produced_type = "wrong"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    provider = build_sandbox_provider("myprov")
    with pytest.raises(SandboxProviderTypeError, match=r"produced a sandbox with .type='wrong'"):
        asyncio.run(provider.create_sandbox(image_name="x", port=0, env={}))
    # the mis-typed sandbox is terminated before raising, so it can't leak
    assert provider.last_sandbox.terminated is True


def test_matching_type_passes_the_guard(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers.myprov]
        impl = "{_HERE}:_MintingProvider"
        [sandbox.providers.myprov.config]
        produced_type = "myprov"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    provider = build_sandbox_provider("myprov")
    sandbox = asyncio.run(provider.create_sandbox(image_name="x", port=0, env={}))
    assert sandbox.type == "myprov"
    assert sandbox.terminated is False


def test_builtin_provider_is_not_guarded(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    provider = build_sandbox_provider("local")
    assert isinstance(provider, LocalSandboxProvider)
    # built-ins satisfy name==type by construction, so create_* stay the class methods (no per-instance guard)
    assert "create_sandbox" not in vars(provider)


def test_guard_preserves_provider_identity(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers.myprov]
        impl = "{_HERE}:_MintingProvider"
        [sandbox.providers.myprov.config]
        produced_type = "myprov"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    provider = build_sandbox_provider("myprov")
    assert isinstance(provider, _MintingProvider)  # decorated in place -> isinstance/logging unaffected
    assert "create_sandbox" in vars(provider)  # guard installed on the instance


def test_create_vm_guard_rejects_mismatch(monkeypatch, tmp_path):
    # env/gateway deploys call create_vm (not create_sandbox), so the guard must cover it too.
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers.myprov]
        impl = "{_HERE}:_MintingProvider"
        [sandbox.providers.myprov.config]
        produced_type = "wrong"
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    provider = build_sandbox_provider("myprov")
    with pytest.raises(SandboxProviderTypeError, match=r"produced a sandbox with .type='wrong'"):
        asyncio.run(provider.create_vm())
    assert provider.last_sandbox.terminated is True


def test_terminate_failure_does_not_mask_config_error(monkeypatch, tmp_path):
    # best-effort cleanup: if terminate() raises, the actionable config error must still surface.
    cfg = _write_config(tmp_path, f"""
        [sandbox.providers.myprov]
        impl = "{_HERE}:_MintingProvider"
        [sandbox.providers.myprov.config]
        produced_type = "wrong"
        terminate_raises = true
    """)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(cfg))
    provider = build_sandbox_provider("myprov")
    with pytest.raises(SandboxProviderTypeError):  # NOT the RuntimeError from terminate()
        asyncio.run(provider.create_sandbox(image_name="x", port=0, env={}))
