"""prompt_agent delivers notifications/v1 on a real gateway: the solver's own check and a trigger's notify open the
agent's next message, ahead of what the next-env-event hook says, while a role the config doesn't name can't
schedule at all."""

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

HOOK = Path(agent_env.__file__).parent / "examples" / "next-env-event" / "artifacts" / "next-env-event" / "hook.py"
MONDAY = "2026-03-02T13:00:00Z"
TUESDAY = "2026-03-03T16:40:00Z"  # 8:40 AM in Los Angeles
CHECK_AT = "2026-03-03T15:40:00Z"  # 7:40 AM
NUDGE = "[{}] Time has passed. Check for anything new and handle it."


def _schedule(note: str, at: str) -> dict:
    return {"action": "call_tool", "tool_name": "schedule_check", "arguments": {"note": note, "at": at}}


@pytest.mark.asyncio
async def test_what_came_due_for_the_solver_opens_its_next_message_on_a_real_gateway(local_registry):
    _seed_bootstrap_envs()
    env, deployed, _ = await _deploy_items("notify")
    agent = put_test_agent(f"notify-agent-{uuid.uuid4().hex[:8]}")
    hook = FileArtifact.put(f"notify-hook-{uuid.uuid4().hex[:8]}", description="next-env-event", file_path=str(HOOK))
    context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[], metadata={})
    triggers = [{"id": "ivy-listed", "when": {"type": "time", "at": TUESDAY},
                 "actions": [{"type": "notify", "channel": "Curbfinch alert", "text": "New listing: 2BR on Ivy St"}]}]
    try:
        await SyncEnvClockTaskStep(id="clock", version=None, env_id=env.id, virtual_time=MONDAY,
                                   virtual_seconds_per_real_second=0).execute(context)
        await ConfigureScheduledChecksTaskStep(id="checks", version=None, env_id=env.id,
                                               roles=["default"]).execute(context)
        await RegisterEnvTriggersStep(id="world", version=None, env_id=env.id, triggers=triggers,
                                      watch_roles=["default"]).execute(context)
        async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
            # what the agent would do in its first turn: schedule a check of its own, through the tool
            made = await client.post("/step", json=_schedule("look for new rugs", CHECK_AT),
                                     headers={AGENT_ENV_ROLE_HEADER: "default"})
            assert made.status_code == 200 and not made.json().get("isError"), made.text
            refused = await client.post("/step", json=_schedule("snoop", CHECK_AT),
                                        headers={AGENT_ENV_ROLE_HEADER: "landlord"})
            assert refused.status_code == 403, refused.text
        await _deploy(agent, "solver").execute(context)
        await PromptAgentTaskStep(
            id="watch", version=None, prompt="Keep an eye on things.", agent_name="solver", max_conversation_turns=4,
            poll_interval_seconds=1,
            post_turn_hooks=[{"file_artifact_id": hook.id, "file_artifact_version": hook.version,
                              "params": {"end_time": "P5D"}}],
        ).execute(context)
        async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
            state = (await client.get("/notifications/state")).raise_for_status().json()
    finally:
        await env.close()

    messages = conversation_store.get_conversation(context.metadata["a2a_conversations"]["watch"])["messages"]
    sent = [m["parts"] for m in messages if m["role"] == "user"]
    assert [[p["text"] for p in parts] for parts in sent] == [
        ["Keep an eye on things."],
        ["[Tue Mar 3, 3:40 PM UTC · your scheduled check] look for new rugs",
         "[Tue Mar 3, 4:40 PM UTC · Curbfinch alert] New listing: 2BR on Ivy St",
         NUDGE.format(TUESDAY)],
    ]
    assert [p.get("metadata", {}).get("agentenv", {}).get("source") for p in sent[1]] == [
        "scheduled_check", "trigger", None]
    assert sent[1][1]["metadata"]["agentenv"]["source_id"] == "ivy-listed"
    assert state["pending_notifications"] == [] and state["next_due_at"] is None
    assert [(c["to"], c["status"]) for c in state["checks"]] == [("default", "done")]
    assert context.metadata["post_turn_hooks"]["watch"][-1]["end"] == "no_pending_events"
