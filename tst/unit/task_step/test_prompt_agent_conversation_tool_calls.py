"""A multi-turn conversation's tool_call_count is the sum of its turns, not the last turn's.

Each turn's terminal message reports only that turn's tool calls, so recording the last one
made a usersim run with 3 + 39 + 19 calls report 19.
"""

from unittest.mock import MagicMock

import pytest
from a2a.types import TaskState

from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep


def _wire(monkeypatch, solver_tool_calls: list):
    """Mock the conversation's HTTP and store boundary. The solver's n-th reply reports
    solver_tool_calls[n] (omitted when None); the user-sim never says done."""
    cfg = MagicMock()
    cfg.get_model_params.return_value = {}
    cfg.get_default_human_a2a_url.return_value = "http://hub/a2a/human"
    monkeypatch.setattr(pa_mod, "get_config", lambda: cfg)
    for fn in ("create_conversation", "add_a2a_task", "complete_a2a_task", "mark_closed", "get_conversation"):
        monkeypatch.setattr(pa_mod.conversation_store, fn, lambda *a, **kw: None)
    solver_turns = iter(solver_tool_calls)

    async def post_agent_config(url, fields):
        return None

    async def send(url, parts, message_id, context_id, timeout):
        return f"task-{message_id}", None

    async def poll(url, task_id, timeout, interval, **_):
        if "user-sim" in url:
            parts = [{"kind": "text", "text": '{"message": "keep going", "done": false}'}]
        else:
            count = next(solver_turns)
            parts = [{"kind": "text", "text": "done"}]
            if count is not None:
                parts.append({"kind": "data", "data": {"tool_call_count": count}})
        return {"status": {"state": TaskState.completed, "message": {"parts": parts}}}

    monkeypatch.setattr(pa_mod.protocol, "post_agent_config", post_agent_config)
    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    monkeypatch.setattr(pa_mod.protocol, "poll_a2a_task", poll)


def _context() -> TaskStepContext:
    ctx = TaskStepContext(instance_id="ti-1")
    ctx.deployed_agents.append(DeployedAgent(agent_name="solver", api_url="http://solver", a2a_url="http://solver",
                                             a2a_card={}))
    ctx.deployed_agents.append(DeployedAgent(agent_name="usersim", api_url="http://user-sim",
                                             a2a_url="http://user-sim", a2a_card={}))
    return ctx


def _step(turns: int) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="solve", version=None, prompt="hi", agent_name="solver",
                               max_conversation_turns=turns, user_agent_name="usersim")


@pytest.mark.asyncio
async def test_the_count_is_the_sum_of_every_turn(monkeypatch):
    _wire(monkeypatch, [3, 39, 19])

    ctx = await _step(3).execute(_context())

    assert ctx.prompt_responses[-1].tool_call_count == 61


@pytest.mark.asyncio
async def test_a_turn_without_a_count_adds_nothing(monkeypatch):
    _wire(monkeypatch, [4, None, 6])

    ctx = await _step(3).execute(_context())

    assert ctx.prompt_responses[-1].tool_call_count == 10


@pytest.mark.asyncio
async def test_it_stays_none_when_no_turn_reports_one(monkeypatch):
    _wire(monkeypatch, [None, None])

    ctx = await _step(2).execute(_context())

    assert ctx.prompt_responses[-1].tool_call_count is None
