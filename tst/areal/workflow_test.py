"""``AgentEnvWorkflow`` on AReaL itself: AReaL's rollout engine resolves it from its import path into its proxy workflow
in inline mode, grouped as GRPO groups a row's samples, and each sample plays the unit tier's bundle, the solver
calling the stand-in model at the proxy address under that sample's session key; a played sample's reward goes on its
last completion. A sample whose run fails is left out of its group, its siblings still training, on an AReaL that asks
the agent; 2.1.0 doesn't, and rejects the group. CI runs it on AReaL 2.1.0 and main."""

import asyncio
import inspect
import logging
from unittest.mock import AsyncMock

import pytest

from agent_env.training.areal import task_rows
from tst.unit.training.harness_test import MODEL, asked
from tst.util.capabilities import AREAL, missing_capability_reason
from tst.util.fake_vllm import fake_vllm, says

proxy = pytest.importorskip("areal.experimental.openai.proxy.workflow", reason=missing_capability_reason(AREAL))
remote = pytest.importorskip("areal.infra.remote_inf_engine", reason=missing_capability_reason(AREAL))
cli_args = pytest.importorskip("areal.api.cli_args", reason=missing_capability_reason(AREAL))
workflow_context = pytest.importorskip("areal.infra.workflow_context", reason=missing_capability_reason(AREAL))
openai_types = pytest.importorskip("areal.experimental.openai.types", reason=missing_capability_reason(AREAL))

WORKFLOW = "agent_env.training.areal.AgentEnvWorkflow"
#: Whether this AReaL asks an inline agent if a sample whose run failed leaves its group alone, or rejects the group.
REJECTS_A_SAMPLE = "should_reject_failed_sample" in inspect.getsource(proxy.OpenAIProxyWorkflow.arun_episode)


class _Session:
    """AReaL's proxy client for one sample: the session key it hands the agent, and the reward set on the session's
    last completion, which it exports."""

    def __init__(self, *args, task_id, **kwargs):
        self.session_api_key, self.session_id = f"key-{task_id}", f"session-{task_id}"
        self.context_overflow = self.system_error = self.invalid_tool_json = False
        self.interaction_count = 1
        self.reward = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None

    async def set_last_reward(self, reward):
        self.reward = reward

    async def export_interactions(self, **kwargs):
        return {self.session_id: openai_types.InteractionWithTokenLogpReward(reward=self.reward)}


@pytest.fixture
def areal(monkeypatch):
    """Plays a row through the workflow AReaL's rollout engine resolves from ``WORKFLOW`` with ``kwargs``, a group of
    ``group`` samples, its proxy at ``url``."""
    monkeypatch.setattr(proxy, "OpenAIProxyClient", _Session)
    monkeypatch.setattr(proxy.OpenAIProxyWorkflow, "_grant_capacity", AsyncMock())
    monkeypatch.setattr(workflow_context, "get_httpx_client", AsyncMock(return_value=None))
    monkeypatch.setattr(workflow_context, "get_aiohttp_session", AsyncMock(return_value=None))
    engine = remote.RemoteInfEngine(cli_args.InferenceEngineConfig(agent=cli_args.AgentConfig(mode="inline")), None)
    engine.logger = logging.getLogger("areal-tier")

    def plays(row, url, group, **kwargs):
        workflow = engine._resolve_workflow(WORKFLOW, kwargs, group_size=group, proxy_addr=url)

        async def episode():
            workflow_context.set(workflow_context.WorkflowContext(task_id=7))
            return await workflow.arun_episode(None, row)

        return asyncio.run(episode())

    return plays


def _sum(bundle):
    return next(row for row in task_rows(bundle) if row["task_id"].endswith("/sum"))


def test_each_sample_plays_the_task_under_its_own_session_and_its_reward_goes_on_its_last_completion(areal, bundle):
    with fake_vllm(MODEL, says("It is 42.")) as engine:
        played = areal(_sum(bundle), engine.url, 2, bundle=str(bundle))

    assert {session: interaction.reward for session, interaction in played.items()} == {
        "session-7:0": 1.0, "session-7:1": 1.0}
    assert sorted(call["key"] for call in asked) == ["key-7:0", "key-7:1"]
    assert {call["overrides"]["step_params"]["agent"]["litellm_base_url"] for call in asked} == {engine.url}
    assert len(engine.requests) == 2


def test_a_sample_whose_run_fails_is_rejected(areal, bundle):
    answered = []

    def second_fails(body):
        answered.append(body)
        if len(answered) == 2:
            raise RuntimeError("the model server fell over")
        return {"role": "assistant", "content": "It is 42."}

    with fake_vllm(MODEL, second_fails) as engine:
        if REJECTS_A_SAMPLE:
            played = areal(_sum(bundle), engine.url, 2, bundle=str(bundle))
            assert [interaction.reward for interaction in played.values()] == [1.0]
        else:
            with pytest.raises(ValueError):
                areal(_sum(bundle), engine.url, 2, bundle=str(bundle))
