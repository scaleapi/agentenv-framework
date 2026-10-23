"""An event-driven multi-day run on the local defaults: a gateway env whose frozen clock starts on a Monday, two time
triggers days apart that write into its items server, and the echo agent (tst/data/a2a_agent) prompted with
between_turns next_env_event. Between turns prompt_agent advances the gateway clock to each mark, the trigger writes
land, the agent is woken once per event with no user side, and every message carries the env clock, all in seconds of
wall time. Needs Docker and a throwaway local registry."""

import json
import time
import uuid
from datetime import datetime

import httpx
import pytest

from agent_env.a2a_agent import conversation_store
from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.prompt_agent import DEFAULT_WAKE_MESSAGE, PromptAgentTaskStep
from agent_env.task_step.task_steps.register_env_triggers import RegisterEnvTriggersStep
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs
from tst.integration.task_step.test_prompt_agent_live_trajectory_local import _deploy, local_registry as local_registry
from tst.integration.task_step.test_register_env_triggers_local import _deploy_items
from tst.util.a2a_test_agent import put_test_agent

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

MONDAY = "2026-03-02T13:00:00Z"
TUESDAY = "2026-03-03T16:40:00Z"
THURSDAY = "2026-03-05T03:15:00Z"
FRIDAY = "2026-03-06T12:00:00Z"


def _vt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


@pytest.mark.asyncio
async def test_each_event_days_apart_wakes_the_agent_once_in_one_run(local_registry):
    _seed_bootstrap_envs()
    env, deployed, _ = await _deploy_items("multiday")
    agent = put_test_agent(f"multiday-agent-{uuid.uuid4().hex[:8]}")
    triggers = [{"id": "school-email", "when": {"type": "time", "at": TUESDAY},
                 "actions": [{"type": "tool", "tool": "items_add_item", "args": {"item": "school email"}}]},
                {"id": "calendar-clash", "when": {"type": "time", "at": "P2DT14H15M"},
                 "actions": [{"type": "tool", "tool": "items_add_item", "args": {"item": "calendar clash"}}]}]
    context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[], metadata={})
    try:
        await SyncEnvClockTaskStep(id="clock", version=None, env_id=env.id, virtual_time=MONDAY,
                                   virtual_seconds_per_real_second=0).execute(context)
        await RegisterEnvTriggersStep(id="events", version=None, env_id=env.id, triggers=triggers,
                                      watch_roles=["default"]).execute(context)
        await _deploy(agent, "solver").execute(context)
        started = time.monotonic()
        await PromptAgentTaskStep(id="watch", version=None, prompt="Watch the inbox.", agent_name="solver",
                                  max_conversation_turns=6, poll_interval_seconds=1,
                                  between_turns={"mode": "next_env_event", "end_time": FRIDAY}).execute(context)
        wall = time.monotonic() - started
        async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=60) as client:
            state = (await client.get("/triggers/state")).raise_for_status().json()
            clock = (await client.get("/clock/state")).raise_for_status().json()
            stored = await client.post("/step", json={"action": "call_tool", "tool_name": "list_items", "arguments": {}},
                                       headers={AGENT_ENV_ROLE_HEADER: "harness"})
    finally:
        await env.close()

    conversation = conversation_store.get_conversation(context.metadata["a2a_conversations"]["watch"])
    sent = [m["parts"][0]["text"] for m in conversation["messages"] if m["role"] == "user"]
    assert sent == ["Watch the inbox.", DEFAULT_WAKE_MESSAGE.format(virtual_time=TUESDAY),
                    DEFAULT_WAKE_MESSAGE.format(virtual_time=THURSDAY)]
    assert [m["env_clock"][env.id] for m in conversation["messages"]] == [MONDAY, MONDAY, TUESDAY, TUESDAY,
                                                                          THURSDAY, THURSDAY]
    assert {t["id"]: (t["status"], t["fire_count"]) for t in state["triggers"]} == {
        "school-email": ("fired", 1), "calendar-clash": ("fired", 1)}
    fired = {e["trigger_id"]: e["virtual_time"] for e in state["events"] if e["kind"] == "fired"}
    assert {k: _vt(v) for k, v in fired.items()} == {"school-email": _vt(TUESDAY), "calendar-clash": _vt(THURSDAY)}
    assert "reanchored" not in {e["kind"] for e in state["events"]}
    assert json.loads(stored.json()["content"][0]["text"])["items"][-2:] == ["school email", "calendar clash"]
    assert _vt(clock["virtual_time"]) == _vt(FRIDAY) and clock["t0"] == MONDAY
    advances = context.metadata["env_event_advances"]["watch"]
    assert [a.get("trigger_ids") for a in advances[:2]] == [["school-email"], ["calendar-clash"]]
    assert all(a["moved"] == {env.id: "advance"} for a in advances[:2])
    assert advances[2] == {"turn": 3, "stopped": "no_pending_time_triggers", "virtual_time": FRIDAY}
    assert wall < 90, f"four simulated days took {wall:.0f}s of wall time"
