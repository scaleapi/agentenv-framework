"""prompt_agent sends the agent's selection of its own tools again on every turn, as deploy_agent recorded it on the
agent's card, so an agent that restarted since its deploy doesn't get its own tools back; an agent deployed without
one is sent none."""

from unittest.mock import MagicMock

import pytest
from a2a.types import TaskState

from agent_env.a2a_agent import A2AAgent
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

SUPPORTED = ["model", "builtin_tools"]


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

    async def post_agent_config(url, fields):
        configs.append(fields)

    async def send(url, parts, message_id, context_id, timeout):
        return f"task-{message_id}", None

    async def poll(url, task_id, timeout, interval, **_):
        return {"status": {"state": TaskState.completed, "message": {"parts": [{"kind": "text", "text": "done"}]}}}

    monkeypatch.setattr(pa_mod.protocol, "post_agent_config", post_agent_config)
    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    monkeypatch.setattr(pa_mod.protocol, "poll_a2a_task", poll)
    return configs


def _context(**stamped) -> TaskStepContext:
    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {
        "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": SUPPORTED}}}}}]}, **stamped}
    context = TaskStepContext(instance_id="ti-1")
    context.deployed_agents.append(DeployedAgent(agent_name="solver", api_url="http://solver", a2a_url="http://solver",
                                                 a2a_card=card))
    return context


@pytest.mark.asyncio
async def test_every_turn_sends_the_selection_deploy_agent_recorded(posted):
    context = _context(builtin_tools={"disable": "*"})

    for turn in ("one", "two"):
        await PromptAgentTaskStep(id=turn, version=None, prompt=turn, agent_name="solver").execute(context)

    assert [fields.get("builtin_tools") for fields in posted] == [{"disable": "*"}, {"disable": "*"}]


@pytest.mark.asyncio
async def test_an_agent_deployed_without_a_selection_is_sent_none(posted):
    await PromptAgentTaskStep(id="one", version=None, prompt="one", agent_name="solver").execute(_context())

    assert all("builtin_tools" not in fields for fields in posted)
