"""A real two-turn conversation (echo agent as solver and user-sim) reads back whole as a ``Run``."""

import uuid

import pytest

from agent_env.task import Task
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.verifiers.run_context import load_run_context
from tst.integration.task_step.test_prompt_agent_file_parts_local import _deploy, local_registry as local_registry
from tst.util.a2a_test_agent import put_test_agent

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_the_run_context_reads_back_every_turn_of_a_real_conversation(local_registry):
    suffix = uuid.uuid4().hex[:8]
    agent = put_test_agent(f"run-context-agent-{suffix}")
    task = Task.put(id=f"run-context-{suffix}", steps=[
        _deploy(agent, "solver"),
        _deploy(agent, "user"),
        PromptAgentTaskStep(id="ask", version=None, prompt_id="ask", agent_name="solver", user_agent_name="user",
                            max_conversation_turns=2, poll_interval_seconds=1, prompt="Plan a lunch for Friday."),
    ])

    context = await task.run()
    assert not context.metadata.get("failed_steps"), context.metadata.get("failed_steps")
    run = await load_run_context(context, next(pr for pr in context.prompt_responses if pr.prompt_id == "ask"))

    assert run.unavailable == {} and len(run.turns) == 2
    assert [(t.number, t.message.author, t.reply.author) for t in run.turns] == [(1, "user", "solver"), (2, "user", "solver")]
    assert run.turn(1).message.text == "Plan a lunch for Friday."
    assert run.turn(2).message.text and run.turn(2).message.text != run.turn(1).message.text
    assert run.turn(1).message.sent_at < run.turn(1).reply.sent_at < run.turn(2).message.sent_at
    assert run.as_of(1).turns == run.turns[:1] and run.firings() == ()
