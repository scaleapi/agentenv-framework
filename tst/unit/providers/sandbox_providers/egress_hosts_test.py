"""The auto-union that keeps an allowlisted sandbox able to reach its own platform."""

import pytest

import agent_env.providers.sandbox_providers.sandbox_provider as sp
from agent_env.config import reset_config
from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider, all_sandbox_egress_hosts

_INTERNAL = "gw.internal.platform.test"
_EXTERNAL = "gw.external.platform.test"


class _Platform(SandboxProvider):
    """A config-registered platform whose gateways are issued on an internal host."""

    URL_REWRITES = {_INTERNAL: _EXTERNAL}

    async def create_sandbox(self, **kwargs):  # pragma: no cover - never provisioned here
        raise NotImplementedError


@pytest.fixture
def platform_registry(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text(
        '[sandbox.providers.platform]\nimpl = "tst.unit.providers.sandbox_providers.egress_hosts_test:_Platform"\n'
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()
    yield
    reset_config()


@pytest.fixture
def builtin_registry(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    reset_config()
    yield
    reset_config()


def test_union_covers_every_registered_platforms_tunnel_hosts(platform_registry):
    hosts = all_sandbox_egress_hosts()
    assert _INTERNAL in hosts                          # a registered platform's URL_REWRITES key
    assert _EXTERNAL in hosts                          # ...and its rewritten value
    assert "*.modal.host" in hosts                     # a Modal EGRESS_HOSTS entry
    assert "*.e2b.app" in hosts                        # E2B's <port>-<sandbox-id> ingress URL
    assert "*.sandbox.tensorlake.ai" in hosts


def test_union_holds_only_platform_hosts():
    """Nothing a workload chose to depend on is pre-approved: a model endpoint (LiteLLM,
    Bedrock, ...) or an object store is the caller's to allowlist, not the SDK's to assume."""
    hosts = all_sandbox_egress_hosts()
    assert not [h for h in hosts if "litellm" in h or "amazonaws" in h or "bedrock" in h]


def test_union_is_deduplicated_and_order_stable():
    hosts = all_sandbox_egress_hosts()
    assert len(hosts) == len(set(hosts))
    assert hosts == all_sandbox_egress_hosts()


def test_a_bare_star_is_dropped_rather_than_defeating_the_allowlist(monkeypatch):
    class _Star(SandboxProvider):
        EGRESS_HOSTS = ("*", "real.com")

        async def create_sandbox(self, **kwargs):  # pragma: no cover
            raise NotImplementedError

    monkeypatch.setattr(sp, "registered_sandbox_provider_classes", lambda: [_Star])
    hosts = all_sandbox_egress_hosts()
    assert "*" not in hosts
    assert "real.com" in hosts


def test_effective_policy_widens_an_allowlist_with_the_platform_hosts(platform_registry):
    allowlist = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("task-specific.com",))
    widened = SandboxProvider.effective_network_policy(allowlist)
    assert "task-specific.com" in widened.allow_hosts
    assert _INTERNAL in widened.allow_hosts


def test_effective_policy_leaves_allow_all_alone():
    """Nothing to widen, and widening would send kwargs Modal reads as a restriction."""
    unrestricted = NetworkPolicy()
    assert SandboxProvider.effective_network_policy(unrestricted) == unrestricted


def test_effective_policy_of_none_is_allow_all():
    assert SandboxProvider.effective_network_policy(None) == NetworkPolicy()


def test_the_builtin_floor_is_pinned(builtin_registry):
    """Every restricted sandbox gets these, so an empty allowlist is not total denial.
    Pinned rather than counted: a new entry should be argued for in review. Platforms
    registered from config add theirs on top."""
    assert set(all_sandbox_egress_hosts()) == {
        "*.modal.host", "*.w.modal.host", "*.e2b.app", "*.sail.box", "*.sandbox.tensorlake.ai",
    }
