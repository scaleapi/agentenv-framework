"""``A2AAgent.deploy`` closes the sandbox it created when the deploy fails or is cancelled, so an agent that never
made it into a run's context doesn't outlive it."""

import asyncio
import types

import pytest

from agent_env.a2a_agent.a2a_agent import DEFAULT_A2A_PORT, A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.sandbox_providers.sandbox_provider import SANDBOX_MODE_CONTAINER


class _Sandbox:
    sandbox_id, type, mode, network_policy = "sb-agent", "fake", SANDBOX_MODE_CONTAINER, None
    tunnel_urls = {DEFAULT_A2A_PORT: "http://agent"}
    terminated = False

    async def terminate(self):
        self.terminated = True


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["cancel", "raise"])
async def test_a_deploy_that_stops_waiting_for_its_card_closes_its_sandbox(monkeypatch, local_stores, ending):
    sandbox, waiting = _Sandbox(), asyncio.Event()

    class _Provider:
        async def create_sandbox(self, **_):
            return sandbox

    async def wait_for_card(self, url):
        waiting.set()
        if ending == "raise":
            raise RuntimeError("no card")
        await asyncio.Event().wait()

    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: _Provider())
    monkeypatch.setattr(A2AAgent, "_wait_for_agent_card", wait_for_card)
    agent = A2AAgent(id="solver", version=1, docker_image_artifact=DockerImageArtifact(id="img", description="d", image_name="img"))
    deploying = asyncio.ensure_future(agent.deploy(
        sandbox_type="fake", env_vars={"LITELLM_API_KEY": "k", "LITELLM_BASE_URL": "http://llm"}))
    await waiting.wait()

    deploying.cancel()
    with pytest.raises((asyncio.CancelledError, RuntimeError)):
        await deploying

    assert sandbox.terminated
