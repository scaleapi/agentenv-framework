"""A verifier is handed the run it grades: after a real two-turn conversation on the local defaults, with the echo
agent (tst/data/a2a_agent) as both solver and user-sim, the run context reads back every turn prompt_agent stored, the
user-sim's included. Needs Docker and a throwaway local registry."""

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

    assert run.unavailable == {} and run.turns == 2
    assert [(m["turn"], m["role"], m.get("author")) for m in run.conversation] == [
        (1, "user", "user"), (1, "agent", "solver"), (2, "user", "user"), (2, "agent", "solver")]
    assert run.conversation[0]["text"] == "Plan a lunch for Friday."
    assert run.conversation[2]["text"] and run.conversation[2]["text"] != run.conversation[0]["text"]
    assert run.as_of_turn(1).conversation == run.conversation[:2]
