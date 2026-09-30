"""The validator's image-by-URI probe uses a transfer grant where the store signs no URL, but only one that
reaches the provider the validation agents will deploy on."""

from types import SimpleNamespace

import pytest

from agent_env.a2a_agent.validator import A2AAgentValidator
from agent_env.config import reset_config, set_object_store
from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import (
    SandboxProvider,
    reset_agent_sandbox_provider,
    set_agent_sandbox_provider,
)
from tst.util.granting_object_store import GRANT_ORIGIN, GrantingObjectStore

_AGENT = SimpleNamespace(id="solver", version=3)


class _RemoteProvider(SandboxProvider):
    async def create_sandbox(self, **kwargs):
        raise NotImplementedError


@pytest.fixture
def store(tmp_path):
    granting = GrantingObjectStore(str(tmp_path))
    set_object_store(granting)
    yield granting
    reset_config()
    reset_agent_sandbox_provider()


@pytest.mark.parametrize("provider", [
    LocalSandboxProvider(), ChainedSandboxProvider([LocalSandboxProvider(), _RemoteProvider()]),
], ids=["local", "local-first-in-a-chain"])
def test_agents_on_the_local_provider_get_the_image_through_a_grant(store, provider):
    set_agent_sandbox_provider(provider)
    fixtures = A2AAgentValidator._upload_probe_fixtures(_AGENT, "file:///skill")
    assert fixtures.png_signed_url.startswith(GRANT_ORIGIN)


@pytest.mark.parametrize("provider", [
    _RemoteProvider(), ChainedSandboxProvider([_RemoteProvider(), LocalSandboxProvider()]),
], ids=["remote", "remote-first-in-a-chain"])
def test_agents_the_grants_do_not_reach_are_not_sent_one(store, provider):
    set_agent_sandbox_provider(provider)
    with pytest.raises(RuntimeError, match="signs URLs or issues grants"):
        A2AAgentValidator._upload_probe_fixtures(_AGENT, "file:///skill")
    assert store.granted == []
