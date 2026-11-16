"""prompt_agent fails the step with ModelKeyUnusable when the agent turn was refused for its model key."""

from unittest.mock import MagicMock

import pytest
from a2a.types import TaskState

from agent_env.task_step.context import DeployedAgent, ModelKeyUnusable, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


def _failed(error_type):
    data = {"error_type": error_type, "error_message": "refused"}
    return {"status": {"state": TaskState.failed, "message": {"parts": [{"kind": "data", "data": data}]}}}


def _wire(monkeypatch, result):
    cfg = MagicMock()
    cfg.get_model_params.return_value = {}
    monkeypatch.setattr(pa_mod, "get_config", lambda: cfg)
    for fn in ("create_conversation", "add_a2a_task", "complete_a2a_task", "mark_closed", "get_conversation"):
        monkeypatch.setattr(pa_mod.conversation_store, fn, lambda *a, **kw: None)

    async def send(url, parts, message_id, context_id, timeout):
        return f"{url}#task", None

    async def poll(url, task_id, timeout, interval, *, sandbox_id=None):
        return result

    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    monkeypatch.setattr(pa_mod.protocol, "poll_a2a_task", poll)


def _context():
    ctx = TaskStepContext(instance_id="ti-1")
    ctx.deployed_agents.append(DeployedAgent(
        agent_name="solver", api_url="http://solver", a2a_url="http://solver", a2a_card={}, sandbox_id="sb-solver",
    ))
    return ctx


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type, cause", [("model_key_budget", "budget"), ("model_key_auth", "auth")])
async def test_a_turn_refused_for_its_model_key_raises_model_key_unusable(monkeypatch, error_type, cause):
    _wire(monkeypatch, _failed(error_type))
    step = PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver")
    ctx = _context()

    with pytest.raises(ModelKeyUnusable) as raised:
        await step.execute(ctx)

    assert raised.value.cause == cause
    assert ctx.prompt_responses[-1].error_type == error_type


@pytest.mark.asyncio
async def test_any_other_failed_turn_still_raises_the_plain_error(monkeypatch):
    _wire(monkeypatch, _failed("agent_error"))
    step = PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver")

    with pytest.raises(RuntimeError, match="A2A task failed") as raised:
        await step.execute(_context())

    assert not isinstance(raised.value, ModelKeyUnusable)
