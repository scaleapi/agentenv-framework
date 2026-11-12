"""A user simulator is sent the resume prompt that keeps it the user when its card takes one: an agent resumes a
turn cut short by asking its model to finish the request, and a user simulator's request is the agent's reply."""

from unittest.mock import MagicMock

import pytest
from a2a.types import TaskState

from agent_env.a2a_agent import A2AAgent, protocol
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.prompt_agent import USER_SIM_RESUME_PROMPT, PromptAgentTaskStep


def _card(*supported: str) -> dict:
    return {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {
        "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": list(supported)}}}}}]}}


@pytest.fixture
def posted(monkeypatch):
    """Mocks the conversation's HTTP and store boundary, and collects each agent config posted, by URL."""
    cfg = MagicMock()
    cfg.get_model_params.return_value = {}
    cfg.get_model_for_role.return_value = None
    monkeypatch.setattr(pa_mod, "get_config", lambda: cfg)
    for fn in ("create_conversation", "add_a2a_task", "complete_a2a_task", "mark_closed", "get_conversation"):
        monkeypatch.setattr(pa_mod.conversation_store, fn, lambda *a, **kw: None)
    configs = {}

    async def post_agent_config(url, fields):
        configs[url] = fields

    async def send(url, parts, message_id, context_id, timeout):
        return f"task-{message_id}", None

    async def poll(url, task_id, timeout, interval, **_):
        text = '{"message": "thanks", "done": true}' if "user-sim" in url else "done"
        return {"status": {"state": TaskState.completed, "message": {"parts": [{"kind": "text", "text": text}]}}}

    monkeypatch.setattr(protocol, "post_agent_config", post_agent_config)
    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    monkeypatch.setattr(pa_mod.protocol, "poll_a2a_task", poll)
    return configs


def _conversation(solver_card: dict, user_card: dict) -> TaskStepContext:
    context = TaskStepContext(instance_id="ti-1")
    context.deployed_agents.append(DeployedAgent(agent_name="solver", api_url="http://solver",
                                                 a2a_url="http://solver", a2a_card=solver_card))
    context.deployed_agents.append(DeployedAgent(agent_name="usersim", api_url="http://user-sim",
                                                 a2a_url="http://user-sim", a2a_card=user_card))
    return context


def _step() -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver", model="task-solver",
                               max_conversation_turns=2, user_agent_name="usersim", user_model="task-user")


@pytest.mark.asyncio
async def test_a_user_simulator_whose_card_takes_a_resume_prompt_is_sent_the_one_that_keeps_it_the_user(posted):
    card = _card("model", "output_format", "resume_prompt")
    await _step().execute(_conversation(card, card))

    assert posted["http://user-sim/ext/agent-config"]["resume_prompt"] == USER_SIM_RESUME_PROMPT
    assert "resume_prompt" not in posted["http://solver/ext/agent-config"]


@pytest.mark.asyncio
async def test_a_user_simulator_whose_card_takes_no_resume_prompt_is_sent_none(posted):
    await _step().execute(_conversation(_card("model"), _card("model", "output_format")))

    assert "resume_prompt" not in posted["http://user-sim/ext/agent-config"]
