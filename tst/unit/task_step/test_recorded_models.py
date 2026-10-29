"""A solver, user simulator or agent judge records the model its agent took: the one asked for when the agent's
card accepts a model, and None, the agent's own, when it doesn't."""

from unittest.mock import MagicMock

import pytest
from a2a.types import TaskState

from agent_env.a2a_agent import A2AAgent
from agent_env.a2a_agent import protocol
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep


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
    context = TaskStepContext(instance_id="ti-1", agent_model="flag-solver",
                              metadata={"user_overrides": {"user_model": "flag-user"}})
    context.deployed_agents.append(DeployedAgent(agent_name="solver", api_url="http://solver",
                                                 a2a_url="http://solver", a2a_card=solver_card))
    context.deployed_agents.append(DeployedAgent(agent_name="usersim", api_url="http://user-sim",
                                                 a2a_url="http://user-sim", a2a_card=user_card))
    return context


def _step() -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver", model="task-solver",
                               max_conversation_turns=2, user_agent_name="usersim", user_model="task-user")


@pytest.mark.asyncio
async def test_a_solver_and_user_simulator_whose_cards_take_a_model_record_the_one_they_were_given(posted):
    context = await _step().execute(_conversation(_card("model"), _card("model", "output_format")))

    assert context.metadata["models"] == {"solver": {"solve": "flag-solver"}, "user simulator": {"solve": "flag-user"}}
    assert posted["http://solver/ext/agent-config"]["model"] == "flag-solver"
    assert posted["http://user-sim/ext/agent-config"]["model"] == "flag-user"


@pytest.mark.asyncio
async def test_an_agent_whose_card_takes_no_model_records_none_and_is_sent_none(posted):
    context = await _step().execute(_conversation(_card("system_prompt"), {}))

    assert context.metadata["models"] == {"solver": {"solve": None}, "user simulator": {"solve": None}}
    assert "model" not in posted.get("http://solver/ext/agent-config", {})
    assert "http://user-sim/ext/agent-config" not in posted


def _judge() -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep.from_dict({"id": "judge", "type": "rubrics_verifier", "prompt_id": "p",
                                              "criteria": [{"id": "c1", "description": "says hi"}]})


@pytest.mark.asyncio
@pytest.mark.parametrize(("card", "recorded"), [(_card("model", "output_format"), "flag-judge"),
                                                (_card("output_format"), None), ({}, None)],
                         ids=["takes a model", "takes no model", "takes no config"])
async def test_an_agent_judge_records_the_model_it_took(monkeypatch, card, recorded):
    async def post_agent_config(url, fields):
        pass

    monkeypatch.setattr(protocol, "post_agent_config", post_agent_config)
    context = TaskStepContext(metadata={"task_id": "t"})

    await _judge()._configure_judge_a2a(judge_a2a_url="http://judge", judge_agent_card=card, model="flag-judge",
                                        judge_effort=None, judge_max_thinking_tokens=None, context=context)

    assert context.metadata["models"] == {"judge": {"judge": recorded}}
