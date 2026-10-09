"""EnvironmentProvider: deploy() and close() are what a provider must implement; _run is the built-ins' shared loop,
with the fallback over a chained sandbox provider, the attempt lines, and closing a failed deploy. The topology is a
scripted fake."""

from __future__ import annotations

import logging
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent_env.providers.sandbox_providers.chained_sandbox_provider import ChainedSandboxProvider
from agent_env.providers.env_providers.env_provider import EnvironmentProvider, _SandboxEnvironmentProvider

_ATTEMPT = r"env_deploy_provider_attempt env_id=e1 provider={} status={} duration_s=\d+\.\d"
_ENV = SimpleNamespace(id="e1")
_RECORD = SimpleNamespace(env_id="e1")


class ProviderA(MagicMock):
    @property
    def links(self):
        return (self,)


class ProviderB(MagicMock):
    @property
    def links(self):
        return (self,)


class _Topology(_SandboxEnvironmentProvider):
    """Runs the next scripted outcome on each sandbox provider it's given: an exception raises, anything else is the record."""

    def __init__(self, *outcomes):
        super().__init__()
        self.outcomes, self.seen = list(outcomes), []

    async def deploy(self, env, sandbox_provider, **options):
        return await self._run(sandbox_provider, env.id, lambda p: self._deploy_one(env, p, **options))

    async def _deploy_one(self, env, sandbox_provider, **options):
        self.seen.append(SimpleNamespace(env=env, sandbox_provider=sandbox_provider, options=options))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_a_provider_must_implement_deploy_and_close():
    with pytest.raises(TypeError, match="close.*deploy"):
        type("Partial", (EnvironmentProvider,), {})()


@pytest.fixture
def log(caplog) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO, logger="agent_env.providers.env_providers.env_provider")
    return caplog


@pytest.mark.asyncio
async def test_a_deploy_returns_the_record_and_logs_one_attempt(log):
    provider = ProviderA()
    topology = _Topology(_RECORD)

    assert await topology.deploy(_ENV, provider, cpu=1.0) == _RECORD

    [seen] = topology.seen
    assert (seen.env, seen.sandbox_provider, seen.options) == (_ENV, provider, {"cpu": 1.0})
    assert len(log.messages) == 1 and re.fullmatch(_ATTEMPT.format("ProviderA", "success"), log.messages[0])


@pytest.mark.asyncio
async def test_a_single_providers_error_is_raised_unwrapped_and_closed(log):
    topology = _Topology(ValueError("no card"))

    with patch.object(topology, "close", wraps=topology.close) as close, pytest.raises(ValueError, match="no card"):
        await topology.deploy(_ENV, ProviderA())

    close.assert_awaited_once()
    [line] = log.messages
    assert re.fullmatch(_ATTEMPT.format("ProviderA", "failure") + r" error_type=ValueError error=ValueError\('no card'\)", line)


@pytest.mark.asyncio
async def test_a_chain_falls_back_to_the_next_member_and_closes_the_failed_attempt(log):
    topology = _Topology(RuntimeError("boom"), _RECORD)

    with patch.object(topology, "close", wraps=topology.close) as close:
        assert await topology.deploy(_ENV, ChainedSandboxProvider([ProviderA(), ProviderB()])) == _RECORD

    assert [type(s.sandbox_provider).__name__ for s in topology.seen] == ["ProviderA", "ProviderB"]
    close.assert_awaited_once()
    first, second = log.messages
    assert re.fullmatch(_ATTEMPT.format("ProviderA", "failure") + r" error_type=RuntimeError error=RuntimeError\('boom'\)", first)
    assert re.fullmatch(_ATTEMPT.format("ProviderB", "success"), second)


@pytest.mark.asyncio
async def test_a_chain_that_fails_everywhere_names_every_member():
    topology = _Topology(RuntimeError("a"), ValueError("b"))

    with pytest.raises(RuntimeError) as raised:
        await topology.deploy(_ENV, ChainedSandboxProvider([ProviderA(), ProviderB()]))

    assert str(raised.value) == "All 2 chained providers failed to deploy: ProviderA: RuntimeError('a'); ProviderB: ValueError('b')"


@pytest.mark.asyncio
async def test_a_chain_narrowed_to_one_member_raises_that_members_error_unwrapped():
    topology = _Topology(ValueError("no card"))
    first = ProviderA()
    topology._sandbox_providers = lambda sandbox_provider, env_id: [first]

    with pytest.raises(ValueError, match="no card"):
        await topology.deploy(_ENV, ChainedSandboxProvider([first, ProviderB()]))

    assert [s.sandbox_provider for s in topology.seen] == [first]


@pytest.mark.asyncio
async def test_a_refused_spec_closes_the_provider_and_deploys_nothing():
    topology = _Topology()

    def refuse(sandbox_provider, env_id):
        raise ValueError(f"env '{env_id}' can't deploy here")

    topology._sandbox_providers = refuse
    with patch.object(topology, "close", wraps=topology.close) as close, pytest.raises(ValueError, match="can't deploy here"):
        await topology.deploy(_ENV, ProviderA())

    close.assert_awaited_once()
    assert topology.seen == []


def test_a_built_in_provider_type_names_the_record_class_it_writes_and_any_other_type_loads_by_shape():
    from agent_env.env.env import DeployedGatewayEnv, DeployedSandboxEnv
    from agent_env.providers.env_providers.env_provider import record_class_for

    assert (record_class_for("gateway"), record_class_for("server")) == (DeployedGatewayEnv, DeployedSandboxEnv)
    assert (record_class_for("newer"), record_class_for(None)) == (None, None)
