"""The advance-to-next-event example hook on a real gateway: a world event lands silently, and the hook moves the clock only
to the agent's own check and a notify action's notification, which prompt_agent delivers, each labelled with its time
and source."""

import json
import uuid
from pathlib import Path

import httpx
import pytest

import agent_env
from agent_env.a2a_agent import conversation_store
from agent_env.artifact import FileArtifact
from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.configure_scheduled_checks import ConfigureScheduledChecksTaskStep
from agent_env.task_step.task_steps.prompt_agent import PromptAgentTaskStep
from agent_env.task_step.task_steps.register_env_triggers import RegisterEnvTriggersStep
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs
from tst.integration.task_step.test_prompt_agent_live_trajectory_local import _deploy, local_registry as local_registry
from tst.integration.task_step.test_register_env_triggers_local import _deploy_items
from tst.util.a2a_test_agent import put_test_agent

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

HOOK = Path(agent_env.__file__).parent / "examples" / "advance-to-next-event" / "artifacts" / "advance-to-next-event" / "hook.py"
MONDAY = "2026-03-02T13:00:00Z"


@pytest.mark.asyncio
async def test_only_the_agent_s_check_and_a_declared_notification_wake_it(local_registry):
    _seed_bootstrap_envs()
    env, deployed, _ = await _deploy_items("wakeups")
    agent = put_test_agent(f"wakeups-agent-{uuid.uuid4().hex[:8]}")
    hook = FileArtifact.put(f"wakeups-hook-{uuid.uuid4().hex[:8]}", description="advance-to-next-event", file_path=str(HOOK))
    context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[], metadata={})
    triggers = [
        {"id": "w-1", "when": {"type": "time", "at": "2026-03-03T17:00:00Z"},
         "actions": [{"type": "tool", "tool": "items_add_item", "args": {"item": "rug listed"}}]},
        {"id": "w-2", "when": {"type": "time", "at": "2026-03-04T16:00:00Z"},
         "actions": [{"type": "notify", "channel": "Messages", "text": "Jamie: did the rug thing get sorted?"}]},
    ]
    try:
        await SyncEnvClockTaskStep(id="clock", version=None, env_id=env.id, virtual_time=MONDAY,
                                   virtual_seconds_per_real_second=0).execute(context)
        await ConfigureScheduledChecksTaskStep(id="checks", version=None, env_id=env.id,
                                               roles=["default"]).execute(context)
        await RegisterEnvTriggersStep(id="world", version=None, env_id=env.id, triggers=triggers,
                                      watch_roles=["default"]).execute(context)
        async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
            # what the agent would do in its first turn: schedule a check of its own, through the tool
            made = await client.post("/step", headers={AGENT_ENV_ROLE_HEADER: "default"}, json={
                "action": "call_tool", "tool_name": "schedule_check",
                "arguments": {"note": "look for new rugs", "at": "2026-03-04T02:00:00Z"}})
            assert made.status_code == 200 and not made.json().get("isError"), made.text
        await _deploy(agent, "solver").execute(context)
        await PromptAgentTaskStep(
            id="watch", version=None, prompt="Keep an eye on things.", agent_name="solver", max_conversation_turns=6,
            poll_interval_seconds=1,
            post_turn_hooks=[{"file_artifact_id": hook.id, "file_artifact_version": hook.version,
                              "params": {"end_time": "P5D"}}],
        ).execute(context)
        async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
            stored = await client.post("/step", json={"action": "call_tool", "tool_name": "list_items", "arguments": {}},
                                       headers={AGENT_ENV_ROLE_HEADER: "harness"})
            state = (await client.get("/notifications/state")).raise_for_status().json()
    finally:
        await env.close()

    messages = conversation_store.get_conversation(context.metadata["a2a_conversations"]["watch"])["messages"]
    assert [[p["text"] for p in m["parts"]] for m in messages if m["role"] == "user"] == [
        ["Keep an eye on things."],
        ["[Wed Mar 4, 2:00 AM UTC · your scheduled check] look for new rugs"],
        ["[Wed Mar 4, 4:00 PM UTC · Messages] Jamie: did the rug thing get sorted?"],
    ]
    assert "rug listed" in json.loads(stored.json()["content"][0]["text"])["items"]
    assert state["pending_notifications"] == [] and state["next_due_at"] is None
    records = context.metadata["post_turn_hooks"]["watch"]
    assert [r.get("for") for r in records] == [["scheduled_check"], ["trigger"], None]
    assert records[-1]["end"] == "nothing_can_wake_the_agent"
