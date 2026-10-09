"""Provider-side NetworkPolicy contract: capability defaults, fail-loud, the base
create_container forward, and the chained pre-filter."""

import pytest

from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandboxProvider
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import (
    NetworkMode,
    NetworkPolicy,
    NetworkPolicyUnsupportedError,
    Sandbox,
)
from agent_env.providers.sandbox_providers.vercel.provider import VercelSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider

RESTRICTIVE = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("only-this.example.com",))
ALLOWLIST = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("llm-proxy.example.com",))
ALLOW_ALL = NetworkPolicy()


class _RecordingProvider(SandboxProvider):
    """Only implements create_vm, so it exercises the *base* create_container."""

    def __init__(self):
        self.seen = {}

    async def create_sandbox(self, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    async def create_vm(self, **kwargs):
        self.seen = kwargs
        raise RuntimeError("stop here: we only care that the kwarg arrived")


@pytest.mark.parametrize("provider_cls", [LocalSandboxProvider])
@pytest.mark.parametrize("policy", [RESTRICTIVE, ALLOWLIST])
def test_backends_without_enforcement_refuse_restrictive_policies(provider_cls, policy):
    assert provider_cls.supports_network_policy(policy) is False
    assert provider_cls.supports_network_policy(ALLOW_ALL) is True


@pytest.mark.parametrize(
    "provider_cls", [ModalSandboxProvider, ModalVmSandboxProvider, VercelSandboxProvider]
)
@pytest.mark.parametrize("policy", [RESTRICTIVE, ALLOWLIST, ALLOW_ALL])
def test_modal_backends_advertise_full_support(provider_cls, policy):
    assert provider_cls.supports_network_policy(policy) is True


@pytest.mark.asyncio
async def test_base_create_container_forwards_the_policy_to_create_vm():
    """Container mode silently losing the policy is the easiest bug to introduce here."""
    provider = _RecordingProvider()
    with pytest.raises(RuntimeError, match="stop here"):
        await provider.create_container(image_name="img", port=8000, env={}, network_policy=RESTRICTIVE)
    assert provider.seen["network_policy"] is RESTRICTIVE


@pytest.mark.asyncio
@pytest.mark.parametrize("provider, method, kwargs", [
    (LocalSandboxProvider(), "create_vm", {}),
    (LocalSandboxProvider(), "create_container", {"image_name": "i", "port": 1, "env": {}}),
])
async def test_a_backend_refuses_before_provisioning_rather_than_ignoring_the_policy(provider, method, kwargs):
    """Silently returning an unrestricted sandbox would make an eval that was supposed to be
    isolated look valid. Each entry point refuses before it provisions anything."""
    with pytest.raises(NetworkPolicyUnsupportedError, match="cannot enforce network policy mode"):
        await getattr(provider, method)(network_policy=RESTRICTIVE, **kwargs)


@pytest.mark.asyncio
async def test_local_provider_still_accepts_allow_all():
    sandbox = await LocalSandboxProvider().create_vm(network_policy=ALLOW_ALL)
    assert isinstance(sandbox, Sandbox)


def test_an_unset_policy_reads_as_unknown_not_unrestricted():
    """get_sandbox cannot recover the policy from the backend, so a reconnected handle
    must report None. Defaulting to ALLOW_ALL would record a restricted run as open."""
    assert Sandbox.network_policy is None


def test_a_sandbox_built_without_a_policy_reports_unknown():
    """This is the constructor shape get_sandbox uses on reconnect: no policy argument,
    because the backend cannot tell us what was applied."""
    from unittest.mock import MagicMock

    from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandbox
    from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox

    for cls in (ModalSandbox, ModalVmSandbox):
        assert cls(MagicMock(), {}).network_policy is None


# --- ChainedSandboxProvider -------------------------------------------------------------


class _Capable(SandboxProvider):
    def __init__(self):
        self.called = False

    @classmethod
    def supports_network_policy(cls, policy):
        return True

    async def create_sandbox(self, **kwargs):
        self.called = True
        sandbox = _FakeSandbox()
        sandbox.network_policy = kwargs.get("network_policy") or NetworkPolicy()
        return sandbox


class _Incapable(SandboxProvider):
    def __init__(self):
        self.called = False

    async def create_sandbox(self, **kwargs):
        self.called = True
        raise AssertionError("an incapable member must never be called")


class _FakeSandbox(Sandbox):
    type = "fake"

    def __init__(self):
        self.sandbox_id = "fake-1"
        self.tunnel_urls = {}
        self.vnc_url = None
        self.mode = "vm"

    async def terminate(self):  # pragma: no cover
        pass


@pytest.mark.asyncio
async def test_chain_prefilters_members_that_cannot_enforce():
    incapable, capable = _Incapable(), _Capable()
    chain = ChainedSandboxProvider([incapable, capable])
    sandbox = await chain.create_sandbox(network_policy=RESTRICTIVE)
    assert incapable.called is False, "the incapable member must be skipped, not attempted"
    assert capable.called is True
    assert sandbox.network_policy.mode is NetworkMode.ALLOWLIST


@pytest.mark.asyncio
async def test_chain_leaves_unrestricted_ordering_untouched():
    """No policy => today's behaviour: the first member is still attempted and still falls
    through on failure. Only a restrictive policy pre-filters."""
    first, second = _Incapable(), _Capable()
    chain = ChainedSandboxProvider([first, second])
    sandbox = await chain.create_sandbox(network_policy=None)
    assert first.called is True
    assert second.called is True
    assert sandbox.network_policy.mode is NetworkMode.ALLOW_ALL


@pytest.mark.asyncio
async def test_chain_raises_when_no_member_can_enforce():
    chain = ChainedSandboxProvider([_Incapable(), _Incapable()])
    with pytest.raises(NetworkPolicyUnsupportedError, match="No sandbox backend can enforce"):
        await chain.create_sandbox(network_policy=RESTRICTIVE)


def test_chain_capability_is_the_union_of_its_members():
    assert ChainedSandboxProvider([_Incapable(), _Capable()]).supports_network_policy(RESTRICTIVE) is True
    assert ChainedSandboxProvider([_Incapable()]).supports_network_policy(RESTRICTIVE) is False
