"""Unit tests for the provider-declared reachability contributions.

The generic deploy paths used to hardcode one platform's Cloudflare-Access plumbing —
credential injection and a host gate. They now ask the registered sandbox providers for
those contributions: a declarative ``CONTAINER_ENV`` map (name -> env:/secret: reference)
resolved generically, and a ``REQUEST_HEADERS`` map matched generically per URL. These
tests pin the generic contract (defaults, resolution, aggregation, best-effort isolation)
against ``_CfProvider``, a config-registered platform with that exact shape; the real
platform's parity tests live with its provider, in the plugin that ships it.
"""

import textwrap

import pytest

from agent_env.config import ENV_REF_PREFIX, reset_config
from agent_env.providers import sandbox_provider
from agent_env.providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.sandbox import Sandbox
from agent_env.providers.sandbox_provider import (
    SandboxProvider,
    _resolve_container_env,
    all_sandbox_container_env,
    sandbox_request_headers_for_url,
)

_CF_ID = "PLATFORM_CF_ACCESS_CLIENT_ID"
_CF_SECRET = "PLATFORM_CF_ACCESS_CLIENT_SECRET"
_PLATFORM_MCP_URL = "https://gw.internal.platform.test/sandbox/vm-8000/mcp"
_CF_TOML = '[sandbox.providers.cf]\nimpl = "tst.unit.providers.test_reachability_contributions:_CfProvider"\n'


@pytest.fixture(autouse=True)
def _cf_registry(monkeypatch, tmp_path):
    """The built-ins plus the Cloudflare-fronted platform, no ambient CF env vars, a clean registry."""
    (tmp_path / "config.toml").write_text(_CF_TOML)
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    for name in (_CF_ID, _CF_SECRET, "GADGET_TOKEN", "GADGET_TOKEN_SOURCE", "UNRESOLVABLE_TOKEN",
                 "TYPO_TOKEN", "TYPO_TOKEN_SOURCE", "EXTRA_TOKEN", "EXTRA_TOKEN_SOURCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    reset_config()
    yield
    reset_config()


class _StubSecretStore:
    def __init__(self, values: dict[str, str]):
        self._values = values

    def get(self, key: str):
        return self._values.get(key)


class _StubConfig:
    """Just the secret-store accessor the ``secret:`` resolver uses.

    Stubbing the Config singleton keeps these tests off AWS Secrets Manager —
    a ``secret:`` reference resolves through the configured secret store, which
    would be a network call in CI.
    """

    def __init__(self, values: dict[str, str]):
        self._store = _StubSecretStore(values)

    def get_secret_store(self) -> _StubSecretStore:
        return self._store


@pytest.fixture
def _stub_config(monkeypatch):
    import agent_env.config as config_mod

    def _install(values: dict[str, str]):
        monkeypatch.setattr(config_mod, "get_config", lambda: _StubConfig(values))

    return _install


@pytest.fixture
def cf_creds(_stub_config):
    _stub_config({"platform_cf_clientid": "cf-id", "platform_cf_clientsecret": "cf-secret"})


@pytest.fixture
def no_cf_creds(_stub_config):
    _stub_config({})


# --- the generic contract -------------------------------------------------------


def test_base_provider_contributes_nothing():
    assert SandboxProvider.CONTAINER_ENV == {}
    assert SandboxProvider.REQUEST_HEADERS == {}
    assert _resolve_container_env(SandboxProvider) == {}


def test_a_provider_without_reachability_needs_stays_silent():
    # LocalSandboxProvider inherits the defaults — nothing to inject, no headers.
    assert LocalSandboxProvider.CONTAINER_ENV == {}
    assert sandbox_request_headers_for_url("http://localhost:8080/mcp") == {}


def test_env_var_of_the_same_name_wins_over_the_reference(monkeypatch, cf_creds):
    """Process env first, secret bundle fallback — per entry."""
    monkeypatch.setenv(_CF_ID, "from-env")
    resolved = _resolve_container_env(_CfProvider)
    assert resolved == {_CF_ID: "from-env", _CF_SECRET: "cf-secret"}


def test_resolution_is_all_or_nothing(_stub_config):
    """A half-configured platform must not inject half its credentials."""
    _stub_config({"platform_cf_clientid": "cf-id"})  # secret half missing
    assert _resolve_container_env(_CfProvider) == {}


# --- aggregation is best-effort -------------------------------------------------


def test_unbuildable_registry_contributes_nothing_instead_of_raising(monkeypatch, tmp_path):
    """A malformed config must not fail a deploy over contributions nobody asked for.

    It still fails loud wherever a provider is actually built.
    """
    (tmp_path / "config.toml").write_text("[sandbox.providers.broken]\nno_impl = true\n")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()
    assert all_sandbox_container_env() == {}
    assert sandbox_provider.registered_sandbox_provider_classes() == []
    assert sandbox_request_headers_for_url(_PLATFORM_MCP_URL) == {}


def test_one_broken_provider_does_not_suppress_the_others(monkeypatch, cf_creds, tmp_path):
    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            """
            [sandbox.providers.unresolvable]
            impl = "tst.unit.providers.test_reachability_contributions:_UnresolvableProvider"
            """
        )
        + _CF_TOML
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()
    # The platform's contributions still land despite the sibling's unresolvable env references.
    assert all_sandbox_container_env()[_CF_ID] == "cf-id"
    assert "UNRESOLVABLE_TOKEN" not in all_sandbox_container_env()
    assert sandbox_request_headers_for_url(_PLATFORM_MCP_URL)["CF-Access-Client-Id"] == "cf-id"
    # A malformed URL degrades per provider inside the aggregator, never raises out.
    assert sandbox_request_headers_for_url("http://[") == {}


def test_headers_union_across_providers_and_a_raising_sibling_is_skipped(monkeypatch, cf_creds, tmp_path):
    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            """
            [sandbox.providers.typo]
            impl = "tst.unit.providers.test_reachability_contributions:_TypoHeaderProvider"
            [sandbox.providers.extra]
            impl = "tst.unit.providers.test_reachability_contributions:_ExtraHeaderProvider"
            """
        )
        + _CF_TOML
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("TYPO_TOKEN_SOURCE", "typo-tok")
    monkeypatch.setenv("EXTRA_TOKEN_SOURCE", "extra-tok")
    reset_config()
    headers = sandbox_request_headers_for_url(_PLATFORM_MCP_URL)
    assert headers["CF-Access-Client-Id"] == "cf-id"
    assert headers["X-Extra"] == "extra-tok"
    assert headers["X-Extra-Static"] == "v1"
    # The typo provider's header names a missing CONTAINER_ENV entry (KeyError);
    # the aggregator skips it without suppressing the others.
    assert "X-Typo" not in headers


def test_static_headers_emit_without_any_resolution(monkeypatch, tmp_path):
    import agent_env.config as config_mod

    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            """
            [sandbox.providers.static]
            impl = "tst.unit.providers.test_reachability_contributions:_StaticHeaderProvider"
            """
        )
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()

    def _boom():
        raise AssertionError("literal headers must not resolve config or secrets")

    monkeypatch.setattr(config_mod, "get_config", _boom)
    assert sandbox_request_headers_for_url("https://api.static.test/mcp") == {"X-Static": "v1"}


def test_mixed_headers_suppressed_when_refs_unresolvable(monkeypatch, no_cf_creds, tmp_path):
    """All-or-nothing covers literals too: no partial header sets from a half-broken provider."""
    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            """
            [sandbox.providers.extra]
            impl = "tst.unit.providers.test_reachability_contributions:_ExtraHeaderProvider"
            """
        )
        + _CF_TOML
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()
    assert sandbox_request_headers_for_url(_PLATFORM_MCP_URL) == {}


def test_a_custom_provider_contribution_is_picked_up(monkeypatch, cf_creds, tmp_path):
    (tmp_path / "config.toml").write_text(
        textwrap.dedent(
            """
            [sandbox.providers.gadget]
            impl = "tst.unit.providers.test_reachability_contributions:_GadgetProvider"
            """
        )
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    monkeypatch.setenv("GADGET_TOKEN_SOURCE", "gadget-tok")
    reset_config()
    assert all_sandbox_container_env()["GADGET_TOKEN"] == "gadget-tok"
    # Host patterns gate per provider, so URLs don't cross-contaminate.
    assert sandbox_request_headers_for_url("https://api.gadget.test/mcp") == {"X-Gadget": "gadget-tok"}
    assert "X-Gadget" not in sandbox_request_headers_for_url(_PLATFORM_MCP_URL)
    # Pattern forms: exact host matches; near-misses don't.
    assert sandbox_request_headers_for_url("https://gadget.test/mcp") == {"X-Gadget": "gadget-tok"}
    assert sandbox_request_headers_for_url("https://gadget.test.evil/mcp") == {}
    assert sandbox_request_headers_for_url("https://xgadget.test/mcp") == {}


# --- call-site wiring -----------------------------------------------------------


def test_a2a_merged_env_carries_contributions_and_they_win_over_caller_env(cf_creds):
    """Parity: the A2A path assigned CF creds after the caller merge, so contributions win
    (the opposite of `_build_container_env`, where caller `extra_env` is applied last)."""
    from types import SimpleNamespace

    from agent_env.a2a_agent.a2a_agent import A2AAgent

    record = SimpleNamespace(default_env_vars={"AWS_ACCESS_KEY_ID": "preset"})
    merged = A2AAgent._build_merged_env(record, {_CF_ID: "caller-supplied"}, 8000)
    assert merged[_CF_ID] == "cf-id"
    assert merged[_CF_SECRET] == "cf-secret"
    assert merged["A2A_PORT"] == "8000"
    assert "SANDBOX_URL_REWRITES" in merged


def test_verifier_reserves_the_contributed_names_against_seed_columns(monkeypatch):
    """A CSV seed column named `platform_cf_access_client_id` must not swap the creds.

    Runs with a booby-trapped Config: the guard is a validation path, so it must
    hold on declared names alone — no secret resolution, and still reserved when
    the values would be unresolvable.
    """
    import agent_env.config as config_mod

    from agent_env.task_step.context import TaskStepContext
    from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
        RunContainerUnitTestsVerifierTaskStep,
    )

    def _boom():
        raise AssertionError("the seed guard must not resolve config or secrets")

    monkeypatch.setattr(config_mod, "get_config", _boom)
    step = RunContainerUnitTestsVerifierTaskStep(
        id="v", version=1, sandbox_name="s", container_name="c", command="true"
    )
    context = TaskStepContext()
    context.metadata["seed"] = {"platform_cf_access_client_id": "attacker", "harmless": "ok"}
    _command, extra_env = step._resolve_command(context)
    assert _CF_ID not in extra_env
    assert extra_env["HARMLESS"] == "ok"


# --- stub providers referenced by the config.toml tests above --------------------


class _CfProvider(SandboxProvider):
    """A platform behind a Cloudflare-Access edge: credentials injected, headers gated per host."""

    URL_REWRITES = {"gw.internal.platform.test": "gw.external.platform.test"}
    CONTAINER_ENV = {_CF_ID: "secret:platform_cf_clientid", _CF_SECRET: "secret:platform_cf_clientsecret"}
    REQUEST_HEADERS = {
        "*.platform.test": {
            "CF-Access-Client-Id": f"env:{_CF_ID}",
            "CF-Access-Client-Secret": f"env:{_CF_SECRET}",
        },
    }

    async def create_sandbox(self, **kwargs) -> Sandbox:
        raise NotImplementedError


class _UnresolvableProvider(SandboxProvider):
    CONTAINER_ENV = {"UNRESOLVABLE_TOKEN": "secret:no_such_bundle_key"}

    async def create_sandbox(self, **kwargs) -> Sandbox:
        raise NotImplementedError


class _GadgetProvider(SandboxProvider):
    CONTAINER_ENV = {"GADGET_TOKEN": "env:GADGET_TOKEN_SOURCE"}
    REQUEST_HEADERS = {
        "*.gadget.test": {"X-Gadget": "env:GADGET_TOKEN"},
        "gadget.test": {"X-Gadget": "env:GADGET_TOKEN"},
    }

    async def create_sandbox(self, **kwargs) -> Sandbox:
        raise NotImplementedError


class _TypoHeaderProvider(SandboxProvider):
    CONTAINER_ENV = {"TYPO_TOKEN": "env:TYPO_TOKEN_SOURCE"}
    REQUEST_HEADERS = {"*.platform.test": {"X-Typo": "env:NO_SUCH_ENTRY"}}

    async def create_sandbox(self, **kwargs) -> Sandbox:
        raise NotImplementedError


class _ExtraHeaderProvider(SandboxProvider):
    CONTAINER_ENV = {"EXTRA_TOKEN": "env:EXTRA_TOKEN_SOURCE"}
    REQUEST_HEADERS = {"*.platform.test": {"X-Extra": "env:EXTRA_TOKEN", "X-Extra-Static": "v1"}}

    async def create_sandbox(self, **kwargs) -> Sandbox:
        raise NotImplementedError


class _StaticHeaderProvider(SandboxProvider):
    REQUEST_HEADERS = {"*.static.test": {"X-Static": "v1"}}

    async def create_sandbox(self, **kwargs) -> Sandbox:
        raise NotImplementedError
