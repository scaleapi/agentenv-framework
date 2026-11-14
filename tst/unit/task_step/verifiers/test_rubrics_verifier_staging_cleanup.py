"""A judge rubrics_verifier deployed itself is terminated when loading the trajectory fails (an override's staging
here), not only after grading."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import agent_env.config as agent_env_config
from agent_env.a2a_agent import A2AAgent
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers import rubrics_verifier as rv


class _StagingFails(rv.RubricsVerifierTaskStep):
    async def stage_trajectory(self, *args, **kwargs):
        raise ConnectionError("the write to the judge's sandbox dropped")


@pytest.mark.asyncio
async def test_a_failed_staging_terminates_the_judge_the_verifier_deployed(monkeypatch):
    terminated: list[str] = []

    class Sandbox:
        def __init__(self, sandbox_id):
            self.sandbox_id = sandbox_id

        async def terminate(self):
            terminated.append(self.sandbox_id)

    class Provider:
        async def get_sandbox(self, sandbox_id):
            return Sandbox(sandbox_id)

    class Judge:
        async def deploy(self, **kwargs):
            return SimpleNamespace(a2a_url="http://judge.test", sandbox_id="sb-judge", agent_card={}, sandbox_type=None)

    config = SimpleNamespace(resolve_reference=lambda v: v, get_litellm_api_key=lambda: "k",
                             get_litellm_base_url=lambda: "http://llm.test", get_default_a2a_agent_id=lambda: "judge")
    monkeypatch.setattr(agent_env_config, "get_config", lambda: config)
    monkeypatch.setattr(A2AAgent, "get", staticmethod(lambda agent_id: Judge()))
    monkeypatch.setattr(sandbox_provider, "get_agent_sandbox_provider", lambda: Provider())
    verifier = _StagingFails(id="verify", version=1, criteria=[{"id": "c1", "title": "asked first"}], prompt_id="ask",
                             use_trajectory=True, use_agent_judge=True, verifier_id="v")
    monkeypatch.setattr(verifier, "_compact_trajectory", lambda *a, **k: rv._CompactedTrajectory.empty())
    ctx = TaskStepContext()
    ctx.prompt_responses.append(PromptResponse(prompt_id="ask", response="done", prompt_text="book it",
                                               agent_trajectory_object_url="mem://run/turn.json"))

    with pytest.raises(ConnectionError):
        await verifier._grade(ctx)

    assert terminated == ["sb-judge"]
