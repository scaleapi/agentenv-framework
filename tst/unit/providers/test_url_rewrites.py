"""Unit tests for the provider-owned URL-rewrite hooks."""

import json
from types import SimpleNamespace

import pytest

from agent_env.config import reset_config
from agent_env.providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_provider import (
    SandboxProvider,
    all_sandbox_url_rewrites,
    reachable_url,
)
from agent_env.task_step.task_steps.deploy_agent import _choose_mcp_url

_INTERNAL = "gw.internal.test"
_EXTERNAL = "gw.external.test"


class _PlatformProvider(SandboxProvider):
    """A platform whose sandboxes share one network and whose gateways need an external host."""

    URL_REWRITES = {_INTERNAL: _EXTERNAL}

    @classmethod
    def shares_network_with(cls, sandbox_type):
        return sandbox_type in {None, "platform", "platform_vm"}

    async def create_sandbox(self, **kwargs):  # pragma: no cover - never provisioned here
        raise NotImplementedError


class _PlatformVmProvider(_PlatformProvider):
    pass


class _NestedProvider(SandboxProvider):
    """Two rewrite keys where one contains the other; the longer key must win on its URLs."""

    URL_REWRITES = {_INTERNAL: _EXTERNAL, f"mac.{_INTERNAL}": f"mac.{_EXTERNAL}"}

    async def create_sandbox(self, **kwargs):  # pragma: no cover - never provisioned here
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _platform_registry(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text(
        "[sandbox]\n"
        'default = "platform"\n'
        "[sandbox.providers.platform]\n"
        'impl = "tst.unit.providers.test_url_rewrites:_PlatformProvider"\n'
        "[sandbox.providers.platform_vm]\n"
        'impl = "tst.unit.providers.test_url_rewrites:_PlatformVmProvider"\n'
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()
    yield
    reset_config()


# --- provider hooks -------------------------------------------------------------

def test_base_provider_has_no_rewrites():
    # A provider that doesn't override the hook (the base default) leaves URLs unchanged.
    assert SandboxProvider.URL_REWRITES == {}
    assert SandboxProvider.get_external_url("http://localhost:8080") == "http://localhost:8080"


def test_default_network_membership_is_same_provider():
    # Modal inherits the default: a provider shares a network only with its own sandbox type.
    assert ModalSandboxProvider.shares_network_with("modal") is True
    assert ModalSandboxProvider.shares_network_with("local") is False
    assert ModalSandboxProvider.shares_network_with(None) is False
    assert ModalSandboxProvider.shares_network_with("bogus") is False


def test_local_provider_isolates_network_and_externalizes_localhost():
    # The local backend never shares a Docker network (agent docker-run bridge vs env compose net),
    # so cross-sandbox URLs are externalized: localhost -> host.docker.internal.
    for sandbox_type in ("local", "modal", None, "bogus"):
        assert LocalSandboxProvider.shares_network_with(sandbox_type) is False
    assert (
        LocalSandboxProvider.get_external_url("http://localhost:8080")
        == "http://host.docker.internal:8080"
    )


def test_nested_rewrite_keys_apply_longest_first():
    # The mac key CONTAINS the base key — rewriting on the shorter key first would yield a host that does not exist.
    assert _NestedProvider.get_external_url(f"https://t.mac.{_INTERNAL}:443/x") == f"https://t.mac.{_EXTERNAL}:443/x"
    assert _NestedProvider.get_external_url(f"https://{_INTERNAL}/y") == f"https://{_EXTERNAL}/y"


# --- registry-level helpers -----------------------------------------------------

def test_reachable_url_resolves_the_issuing_provider():
    url = f"https://{_INTERNAL}/sandbox/vm-8000/mcp"
    assert reachable_url(url, from_sandbox_type="platform", to_sandbox_type="modal") == url.replace(
        _INTERNAL, _EXTERNAL
    )
    assert reachable_url(url, from_sandbox_type="platform", to_sandbox_type="platform_vm") == url
    # No issuer recorded: the configured default spec is the issuer.
    assert reachable_url(url, from_sandbox_type=None, to_sandbox_type="modal") == url.replace(_INTERNAL, _EXTERNAL)
    assert reachable_url(url, from_sandbox_type="bogus", to_sandbox_type="modal") == url


def test_all_sandbox_url_rewrites_is_the_registry_union():
    union = all_sandbox_url_rewrites()
    assert union == {**LocalSandboxProvider.URL_REWRITES, _INTERNAL: _EXTERNAL}
    assert json.loads(json.dumps(union, sort_keys=True)) == union


def test_all_sandbox_url_rewrites_is_empty_when_the_registry_is_unbuildable(tmp_path, monkeypatch):
    # A malformed config must not fail injection callers (deploys) — it fails
    # loud wherever a provider is actually built instead.
    (tmp_path / "config.toml").write_text("[sandbox.providers.broken]\nno_impl = true\n")
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()
    assert all_sandbox_url_rewrites() == {}


# --- deploy_agent._choose_mcp_url ------------------------------------------------

def _record(sandbox_type, mcp_url=f"https://{_INTERNAL}/sandbox/vm-8000/mcp"):
    return SimpleNamespace(sandbox_type=sandbox_type, mcp_url=mcp_url)


@pytest.mark.parametrize(
    ("env_type", "agent_type", "rewritten"),
    [
        ("platform", "modal", True),
        ("platform", "platform", False),
        ("platform", "platform_vm", False),
        ("platform", None, False),
        (None, "modal", True),  # no issuer recorded: [sandbox] default (platform) is the issuer
        ("modal", "platform", False),
        ("modal", "modal", False),
        ("bogus", "modal", False),  # unknown platform: leave the URL alone
    ],
)
def test_choose_mcp_url_matrix(env_type, agent_type, rewritten):
    env = _record(env_type)
    url = _choose_mcp_url(_record(agent_type), env)
    assert url == (env.mcp_url.replace(_INTERNAL, _EXTERNAL) if rewritten else env.mcp_url)
