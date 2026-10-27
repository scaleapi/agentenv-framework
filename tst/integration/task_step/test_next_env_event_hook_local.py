"""The next-env-event example hook on a real gateway: one conversation, event to event over simulated days."""

import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
import pytest

import agent_env
from agent_env.a2a_agent import conversation_store
from agent_env.artifact import FileArtifact
from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER
from agent_env.task_step.context import TaskStepContext
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
TUESDAY = "2026-03-03T16:40:00Z"
TUESDAY_LATER = "2026-03-03T18:40:00Z"
THURSDAY = "2026-03-05T03:15:00Z"
FRIDAY = "2026-03-06T12:00:00Z"
NUDGE = "[{}] Time has passed. Check for anything new and handle it."


def _vt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _add(item: str, **action) -> list[dict]:
    return [{"type": "tool", "tool": "items_add_item", "args": {"item": item}, **action}]


async def _run(label: str, triggers: list[dict], params: dict) -> tuple:
    """Run the echo agent over the items env with these triggers and the hook; its context, env, state and wall time."""
    _seed_bootstrap_envs()
    env, deployed, _ = await _deploy_items(label)
    agent = put_test_agent(f"{label}-agent-{uuid.uuid4().hex[:8]}")
    hook = FileArtifact.put(f"{label}-hook-{uuid.uuid4().hex[:8]}", description="next-env-event", file_path=str(HOOK))
    context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[], metadata={})
    try:
        await SyncEnvClockTaskStep(id="clock", version=None, env_id=env.id, virtual_time=MONDAY,
                                   virtual_seconds_per_real_second=0).execute(context)
        await RegisterEnvTriggersStep(id="events", version=None, env_id=env.id, triggers=triggers,
                                      watch_roles=["default"]).execute(context)
        await _deploy(agent, "solver").execute(context)
        started = time.monotonic()
        await PromptAgentTaskStep(
            id="watch", version=None, prompt="Watch the inbox.", agent_name="solver", max_conversation_turns=8,
            poll_interval_seconds=1,
            post_turn_hooks=[{"file_artifact_id": hook.id, "file_artifact_version": hook.version, "params": params}],
        ).execute(context)
        wall = time.monotonic() - started
        async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
            state = (await client.get("/triggers/state")).raise_for_status().json()
            clock = (await client.get("/clock/state")).raise_for_status().json()
            stored = await client.post("/step", json={"action": "call_tool", "tool_name": "list_items", "arguments": {}},
                                       headers={AGENT_ENV_ROLE_HEADER: "harness"})
    finally:
        await env.close()
    items = json.loads(stored.json()["content"][0]["text"])["items"]
    return context, env, state, clock, items, wall


def _due(record: dict) -> list | None:
    return [e["due"] for e in record["envs"]] if "envs" in record else None


def _conversation(context: TaskStepContext) -> list[dict]:
    return conversation_store.get_conversation(context.metadata["a2a_conversations"]["watch"])["messages"]


@pytest.mark.asyncio
async def test_each_event_days_apart_wakes_the_agent_once_in_one_run(local_registry):
    triggers = [{"id": "school-email", "when": {"type": "time", "at": TUESDAY}, "actions": _add("school email")},
                {"id": "calendar-clash", "when": {"type": "time", "at": "P2DT14H15M"},
                 "actions": _add("calendar clash")}]

    context, env, state, clock, items, wall = await _run("multiday", triggers, {"end_time": FRIDAY})

    messages = _conversation(context)
    assert [m["parts"][0]["text"] for m in messages if m["role"] == "user"] == [
        "Watch the inbox.", NUDGE.format(TUESDAY), NUDGE.format(THURSDAY)]
    assert [(c["env_id"], c["env_step_id"], c["virtual_time"]) for m in messages for c in m["env_clock"]] == [
        (env.id, None, at) for at in (MONDAY, MONDAY, TUESDAY, TUESDAY, THURSDAY, THURSDAY)]
    assert {t["id"]: (t["status"], t["fire_count"]) for t in state["triggers"]} == {
        "school-email": ("fired", 1), "calendar-clash": ("fired", 1)}
    fired = {e["trigger_id"]: _vt(e["virtual_time"]) for e in state["events"] if e["kind"] == "fired"}
    assert fired == {"school-email": _vt(TUESDAY), "calendar-clash": _vt(THURSDAY)}
    assert "reanchored" not in {e["kind"] for e in state["events"]}
    assert items[-2:] == ["school email", "calendar clash"]
    assert _vt(clock["virtual_time"]) == _vt(THURSDAY) and clock["t0"] == MONDAY and state["next_event_at"] is None
    records = context.metadata["post_turn_hooks"]["watch"]
    assert [(r.get("event_at"), _due(r), r.get("end")) for r in records] == [
        (TUESDAY, [["school-email"]], None), (THURSDAY, [["calendar-clash"]], None), (None, None, "no_pending_events")]
    assert all(e["env_id"] == env.id and e["settled"] for r in records[:2] for e in r["envs"])
    assert wall < 90, f"four simulated days took {wall:.0f}s of wall time"


@pytest.mark.asyncio
async def test_a_delayed_action_lands_at_its_mark_plus_the_delay_and_wakes_the_agent_then(local_registry):
    triggers = [{"id": "school-email", "when": {"type": "time", "at": TUESDAY},
                 "actions": _add("school email", delay="PT2H")},
                {"id": "calendar-clash", "when": {"type": "time", "at": THURSDAY}, "actions": _add("calendar clash")}]

    context, env, state, clock, items, wall = await _run("delayed", triggers, {})

    assert [m["parts"][0]["text"] for m in _conversation(context) if m["role"] == "user"] == [
        "Watch the inbox.", NUDGE.format(TUESDAY), NUDGE.format(TUESDAY_LATER), NUDGE.format(THURSDAY)]
    acted = [_vt(e["virtual_time"]) for e in state["events"] if e["kind"] == "action_ok"]
    assert acted == [_vt(TUESDAY_LATER), _vt(THURSDAY)]
    assert items[-2:] == ["school email", "calendar clash"]
    records = context.metadata["post_turn_hooks"]["watch"]
    assert [r.get("event_at") for r in records] == [TUESDAY, TUESDAY_LATER, THURSDAY, None]
    assert _due(records[1]) == [["school-email"]] and records[-1]["end"] == "no_pending_events"
    assert wall < 90, f"three simulated days took {wall:.0f}s of wall time"
