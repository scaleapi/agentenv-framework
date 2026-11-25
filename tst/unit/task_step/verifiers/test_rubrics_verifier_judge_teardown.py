"""An auto-deployed judge is torn down on the provider it was deployed on, not the configured agent default, and its
work folder goes with it on this machine."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers import rubrics_verifier
from agent_env.task_step.task_steps.verifiers.judge_utils.judge_output_format import ResultRows
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

CRITERIA = [{"id": "done", "description": "The task is done"}]


def _judged(*, eval_prompt, context, model, criteria, **kw):
    rows = [{**c, "justification": "yes", "score": 1.0, "result": True} for c in criteria]
    return ResultRows(rows, reasoning="", checks=[]), 0, [], None


async def _grade_with(monkeypatch, sandbox, sandbox_type=None):
    """Grade once with a judge the verifier deploys onto ``sandbox``; the provider and its lookups, to check."""
    step = RubricsVerifierTaskStep(id="judge", version=1, criteria=CRITERIA, prompt_id="p1", verifier_id="v",
                                   use_agent_judge=True, use_trajectory=False, judge_a2a_agent_id="judge-agent")
    judge = SimpleNamespace(a2a_url="http://judge", sandbox_id="sb-judge", agent_card={}, sandbox_type=sandbox_type)
    agent = MagicMock()
    agent.deploy = AsyncMock(return_value=judge)
    monkeypatch.setattr(A2AAgent, "get", MagicMock(return_value=agent))

    async def run_judge(**kw):
        return _judged(**kw)

    monkeypatch.setattr(step, "_run_judge_with_output_retries", run_judge)
    provider = MagicMock()
    provider.get_sandbox = AsyncMock(return_value=sandbox)
    context = TaskStepContext(
        prompt_responses=[PromptResponse(prompt_id="p1", response="done", prompt_text="do it")],
        metadata={"user_overrides": {"judge_litellm_api_key": "k", "judge_litellm_base_url": "http://litellm"}},
    )
    with patch("agent_env.providers.sandbox_providers.sandbox_provider.build_sandbox_provider", return_value=provider) as build, \
            patch("agent_env.providers.sandbox_providers.sandbox_provider.get_agent_sandbox_provider", return_value=provider) as default:
        await step.execute(context)
    return provider, build, default


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox_type, built", [("modal", "modal"), (None, None)], ids=["recorded", "unrecorded"])
async def test_the_judge_is_torn_down_where_it_was_deployed(monkeypatch, sandbox_type, built):
    sandbox = AsyncMock(ON_THIS_MACHINE=False)

    provider, build, default = await _grade_with(monkeypatch, sandbox, sandbox_type)

    provider.get_sandbox.assert_awaited_once_with("sb-judge")
    sandbox.terminate.assert_awaited_once()
    if built:
        build.assert_called_once_with(built)
        default.assert_not_called()
    else:
        build.assert_not_called()
        default.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("here, terminates, removed", [(True, True, ["sb-judge"]), (False, True, []), (True, False, [])],
                         ids=["on-this-machine", "elsewhere", "terminate-failed"])
async def test_a_judge_on_this_machine_leaves_no_work_folder(monkeypatch, here, terminates, removed):
    """A run's teardown removes the folders of the sandboxes it records, which a judge the verifier deploys isn't: the
    verifier removes the folder once the sandbox is terminated, and keeps one it couldn't terminate."""
    sandbox = AsyncMock(ON_THIS_MACHINE=here)
    if not terminates:
        sandbox.terminate.side_effect = RuntimeError("Local sandbox teardown incomplete")
    gone: list[str] = []
    monkeypatch.setattr(rubrics_verifier, "remove_local_work_dir", gone.append)

    await _grade_with(monkeypatch, sandbox)

    sandbox.terminate.assert_awaited_once()
    assert gone == removed
