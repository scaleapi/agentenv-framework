"""prompt_agent gives up on an agent once its sandbox stops answering, naming the sandbox, but waits on a human."""

from unittest.mock import MagicMock

import pytest
from a2a.types import TaskState

from agent_env.a2a_agent.protocol import AgentUnreachableError
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep

_USER_SIM = DeployedAgent(
    agent_name="human_agent", api_url="http://user-sim", a2a_url="http://user-sim", a2a_card={}, sandbox_id="sb-user",
)


def _completed(text):
    return {"status": {"state": TaskState.completed, "message": {"parts": [{"kind": "text", "text": text}]}}}


def _wire(monkeypatch, poll):
    cfg = MagicMock()
    cfg.get_model_params.return_value = {}
    cfg.get_default_human_a2a_url.return_value = "http://hub/a2a/human"
    monkeypatch.setattr(pa_mod, "get_config", lambda: cfg)
    for fn in ("create_conversation", "add_a2a_task", "complete_a2a_task", "mark_closed", "get_conversation"):
        monkeypatch.setattr(pa_mod.conversation_store, fn, lambda *a, **kw: None)

    async def send(url, parts, message_id, context_id, timeout):
        return f"{url}#task", None

    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    monkeypatch.setattr(pa_mod.protocol, "poll_a2a_task", poll)


def _context(*agents):
    ctx = TaskStepContext(instance_id="ti-1")
    ctx.deployed_agents.append(DeployedAgent(
        agent_name="solver", api_url="http://solver", a2a_url="http://solver", a2a_card={},
        sandbox_id="sb-solver", sandbox_type="modal",
    ))
    ctx.deployed_agents.extend(agents)
    return ctx


@pytest.mark.asyncio
async def test_a_solver_that_stops_answering_fails_the_step_naming_its_sandbox(monkeypatch):
    async def poll(url, task_id, timeout, interval, *, sandbox_id=None):
        raise AgentUnreachableError(f"The agent on sandbox {sandbox_id} stopped answering")

    _wire(monkeypatch, poll)
    step = PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver")

    with pytest.raises(AgentUnreachableError, match="sandbox sb-solver"):
        await step.execute(_context())


@pytest.mark.asyncio
@pytest.mark.parametrize("user_sim, user_url, watched", [
    (_USER_SIM, "http://user-sim", "sb-user"),
    (None, "http://hub/a2a/human", None),
], ids=["deployed-user-sim", "human"])
async def test_a_user_sim_on_a_sandbox_is_watched_but_a_human_is_not(monkeypatch, user_sim, user_url, watched):
    polls = []

    async def poll(url, task_id, timeout, interval, *, sandbox_id=None):
        polls.append((url, sandbox_id))
        return _completed("and then?")

    _wire(monkeypatch, poll)
    step = PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver", max_conversation_turns=2)

    await step.execute(_context(*([user_sim] if user_sim else [])))

    assert polls == [("http://solver", "sb-solver"), (user_url, watched), ("http://solver", "sb-solver")]


@pytest.mark.asyncio
async def test_a_user_sim_that_stops_answering_ends_the_conversation_not_the_run(monkeypatch):
    async def poll(url, task_id, timeout, interval, *, sandbox_id=None):
        if url == "http://user-sim":
            raise AgentUnreachableError(f"The agent on sandbox {sandbox_id} stopped answering")
        return _completed("solver answer")

    _wire(monkeypatch, poll)
    step = PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver", max_conversation_turns=3)

    result = await step.execute(_context(_USER_SIM))

    assert result.prompt_responses[-1].response == "solver answer"
