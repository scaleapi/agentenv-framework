"""prompt_agent's max_output_tokens: stored with the step, replaced by a run's agent_max_output_tokens, and sent in the
agent config only to an agent whose card takes it."""

from unittest.mock import MagicMock

import pytest
from a2a.types import TaskState

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


@pytest.fixture
def posted(monkeypatch):
    """Mocks the turn's HTTP and store boundary, and collects each agent config posted."""
    cfg = MagicMock()
    cfg.get_model_params.return_value = {}
    cfg.get_model_for_role.return_value = None
    monkeypatch.setattr(pa_mod, "get_config", lambda: cfg)
    for fn in ("create_conversation", "add_a2a_task", "complete_a2a_task", "mark_closed", "get_conversation"):
        monkeypatch.setattr(pa_mod.conversation_store, fn, lambda *a, **kw: None)
    configs = []

    async def post_agent_config(url, fields, headers=None):
        configs.append(fields)

    async def send(url, parts, message_id, context_id, timeout):
        return f"task-{message_id}", None

    async def poll(url, task_id, timeout, interval, **_):
        return {"status": {"state": TaskState.completed, "message": {"parts": [{"kind": "text", "text": "done"}]}}}

    monkeypatch.setattr(pa_mod.protocol, "post_agent_config", post_agent_config)
    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    monkeypatch.setattr(pa_mod.protocol, "poll_a2a_task", poll)
    return configs


def _context(supported: list[str], overrides: dict | None = None) -> TaskStepContext:
    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {
        "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": supported}}}}}]}}
    context = TaskStepContext(instance_id="ti-1", metadata={"user_overrides": overrides} if overrides else {})
    context.deployed_agents.append(DeployedAgent(agent_name="solver", api_url="http://solver", a2a_url="http://solver",
                                                 a2a_card=card))
    return context


def _step(**fields) -> PromptAgentTaskStep:
    return PromptAgentTaskStep.from_dict({"id": "p", "version": 1, "prompt": "go", "agent_name": "solver", **fields})


def test_max_output_tokens_round_trips():
    assert PromptAgentTaskStep.from_dict(_step(max_output_tokens=64000).to_dict()).max_output_tokens == 64000
    assert _step().max_output_tokens is None


@pytest.mark.asyncio
async def test_the_steps_max_output_tokens_reaches_an_agent_that_takes_it(posted):
    await _step(max_output_tokens=64000).execute(_context(["model", "max_output_tokens"]))

    assert [fields.get("max_output_tokens") for fields in posted] == [64000]


@pytest.mark.asyncio
async def test_a_runs_override_replaces_the_steps_max_output_tokens(posted):
    await _step(max_output_tokens=64000).execute(
        _context(["model", "max_output_tokens"], {"agent_max_output_tokens": 96000}))

    assert [fields.get("max_output_tokens") for fields in posted] == [96000]


@pytest.mark.asyncio
async def test_an_agent_whose_card_does_not_take_it_is_not_sent_max_output_tokens(posted):
    await _step(max_output_tokens=64000, model="m").execute(_context(["model"]))

    assert posted and all("max_output_tokens" not in fields for fields in posted)


@pytest.mark.asyncio
async def test_an_unset_max_output_tokens_is_not_sent(posted):
    await _step(model="m").execute(_context(["model", "max_output_tokens"]))

    assert posted and all("max_output_tokens" not in fields for fields in posted)
