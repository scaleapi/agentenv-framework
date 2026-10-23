"""``teardown_run`` blocks a run's model budget keys once its agents are down."""

from __future__ import annotations

import pytest

from agent_env import model_budget
from agent_env.task import teardown
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from tst.util.fake_litellm import serve


@pytest.mark.asyncio
async def test_the_keys_are_blocked_after_the_agents_are_down(monkeypatch, local_stores):
    litellm = serve(monkeypatch, local_stores)
    events = []

    async def terminate(sandbox):
        events.append(("terminate", sandbox.sandbox_id))
        return False

    monkeypatch.setattr(teardown, "_terminate", terminate)
    handle = litellm.handle

    def recording(request):
        events.append(("litellm", request.url.path))
        return handle(request)

    monkeypatch.setattr(litellm, "handle", recording)
    key = await model_budget.make_key(agent_name="solver", max_usd=1, ttl_seconds=60, attribution={})
    context = TaskStepContext(deployed_agents=[
        DeployedAgent(agent_name="solver", api_url="http://agent.test", sandbox_id="sb-1", sandbox_type="fake")])
    model_budget.record(context, "solver", key)
    events.clear()

    report = await teardown_run(context)

    assert [s.sandbox_id for s in report.terminated] == ["sb-1"]
    assert events == [("terminate", "sb-1"), ("litellm", "/key/block")]
    assert context.metadata["model_budgets"]["solver"]["key_blocked"] is True and litellm.keys[key.key_id].blocked


@pytest.mark.asyncio
async def test_a_run_without_budgets_asks_litellm_nothing(monkeypatch, local_stores):
    litellm = serve(monkeypatch, local_stores)

    await teardown_run(TaskStepContext())

    assert litellm.requests == []
