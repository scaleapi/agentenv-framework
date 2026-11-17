"""Triggers on a multi env with a computer: the agent's computer calls set off action and state triggers, a world event
and a check wake it, a counterpart waits for its working hours, and the advance-to-next-event hook moves the frozen
clock between turns with the computer following. Needs Docker and the local registry on localhost:5000."""

import asyncio
import importlib.util
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import agent_env
import httpx
import pytest

from agent_env.env import Env
from agent_env.env.env import DeployedEnv
from agent_env.env.envs.computer.utils import inside
from agent_env.env.gateway import AGENT_ENV_ROLE_HEADER
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.configure_scheduled_checks import ConfigureScheduledChecksTaskStep
from agent_env.task_step.task_steps.register_env_triggers import RegisterEnvTriggersStep
from tst.integration.computer.conftest import CONTAINER, multi_env
from tst.integration.computer.multi_env_computers_test import _reaches, shop  # noqa: F401  shop is a fixture
from tst.integration.task_step.test_register_env_triggers_local import _Executor

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

HOOK = (Path(agent_env.__file__).parent / "examples" / "advance-to-next-event" / "artifacts" / "advance-to-next-event"
        / "hook.py")
AGENT, ZONE, T0 = "person", "America/Denver", "2027-03-01T16:00:00Z"  # 09:00 in Denver
MAIL = "MAIL from Robert: the lockbox code is 4821"
TRIGGERS = [
    {"id": "mail-drops", "when": {"type": "time", "at": "2027-03-01T10:00:00-07:00"},
     "actions": [{"type": "tool", "tool": "items_add_item", "as": "world", "args": {"item": MAIL}},
                 {"type": "notify", "channel": "Mail", "to": AGENT, "text": "New mail from Robert"}]},
    {"id": "sees-a-click", "when": {"type": "action", "tool": "desktop_computer_click", "roles": [AGENT]},
     "actions": [{"type": "tool", "tool": "items_add_item", "as": "world", "args": {"item": "CLICK-SEEN"}}]},
    {"id": "sees-the-flag", "when": {"type": "state", "check": {"tool": "list_items", "extract": "items",
                                                                 "predicate": {"regex": "FLAG"}}},
     "actions": [{"type": "tool", "tool": "items_add_item", "as": "world", "args": {"item": "STATE-SEEN"}}]},
    {"id": "bob-after-hours", "when": {"type": "time", "at": "2027-03-01T18:00:00-07:00"},
     "actions": [{"type": "nl", "executor": "bob", "instruction": "BOB-REPLIES"}]},
]
FOLLOW_S = 10  # how long the computer's follower may take to show a jump


class _Turn:
    """What the hook reads of prompt_agent's turn, for a scripted agent."""
    last = False

    def __init__(self, envs):
        self.envs, self._records = envs, []

    async def user_triggers(self):
        return None

    def records(self):
        return self._records

    def record(self, entry):
        self._records.append(entry)

    def heartbeat(self):
        pass


def _hook():
    spec = importlib.util.spec_from_file_location("advance_to_next_event", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_an_agent_on_a_computer_sets_off_triggers_and_wakes_for_world_events_and_checks_as_the_clock_jumps(
        desktop, items):
    started: list[_Executor] = []
    with namespace_routing():
        office = multi_env(items, Env.get(desktop.id))
        context = TaskStepContext()
        try:
            deployed = await office.deploy(sandbox_type="local", ttl_seconds=1800, child_envs={"desktop": CONTAINER})
            bob = _Executor("bob", "bob@example.com", deployed.gateway_url, office._sandbox.host_ips[-1], started)
            context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[bob.agent], metadata={})
            computer = DeployedEnv.from_dict(deployed.child_envs["desktop"])
            async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=120,
                                         headers=deployed.control_headers) as client:
                async def call(tool, args=None, role=AGENT):
                    answer = await client.post("/step", json={"action": "call_tool", "tool_name": tool,
                                                              "arguments": args or {}},
                                               headers={AGENT_ENV_ROLE_HEADER: role})
                    assert answer.status_code == 200, answer.text
                    return answer.json()

                async def status(trigger_id):
                    await _quiet(client)
                    row = next(t for t in (await _triggers(client))["triggers"] if t["id"] == trigger_id)
                    return row["status"]

                (await client.put("/clock/set-time", json={"virtual_time": T0, "virtual_seconds_per_real_second": 0,
                                                           "time_zone": ZONE})).raise_for_status()
                await ConfigureScheduledChecksTaskStep(id="checks", version=None, env_id=office.id,
                                                       roles=[AGENT]).execute(context)
                await RegisterEnvTriggersStep(id="world", version=None, env_id=office.id, triggers=TRIGGERS,
                                              watch_roles=[AGENT], executor_timeout_seconds=60,
                                              executor_hours={"bob": {"start": "09:00", "end": "17:00",
                                                                      "time_zone": ZONE}}).execute(context)
                listed = (await client.post("/step", json={"action": "list_tools"},
                                            headers={AGENT_ENV_ROLE_HEADER: AGENT})).json()
                tools = {tool["name"] for tool in listed["tools"]}

                await call("desktop_computer_screenshot")
                await call("desktop_computer_click", {"x": 5, "y": 5})
                after_click = await status("sees-the-flag")
                await call("items_add_item", {"item": "FLAG"}, role="harness")
                after_unwatched_write = await status("sees-the-flag")
                await call("desktop_computer_wait", {"seconds": 1})
                after_read_only_call = await status("sees-the-flag")
                await call("desktop_computer_pointer_move", {"x": 20, "y": 20})
                after_input_call = await status("sees-the-flag")
                await call("schedule_check", {"note": "Read Robert's mail", "at": "2027-03-01T12:00:00-07:00"})

                hook, turn, wakes = _hook(), _Turn([deployed]), []
                for _ in range(4):
                    out = await hook.post_turn(turn, {"role": AGENT, "end_time": "2027-03-03T00:00:00-07:00"})
                    now = (await client.get("/clock/state")).json()["virtual_time"]
                    followed = await _computer_shows(computer, now)
                    if "end" in out:
                        wakes.append((out["end"], now, followed))
                        break
                    taken = (await client.post("/notifications/take", json={"to": AGENT})).json()["items"]
                    mail = json.loads((await call("list_items"))["content"][0]["text"])["items"]
                    wakes.append(([(item["source"], item.get("channel")) for item in taken], now, followed,
                                  MAIL in mail))
                state = await _triggers(client)
                trajectory = [json.loads(line) for line in
                              (await client.get("/trajectory")).raise_for_status().text.splitlines()]
        finally:
            for executor in started:
                executor.stop()
            report = await teardown_run(context)

    assert not report.failed
    assert {"desktop_computer_click", "schedule_check", "list_items"} <= tools, sorted(tools)
    assert (after_click, after_unwatched_write, after_read_only_call, after_input_call) == (
        "armed", "armed", "armed", "fired")
    detected = {e["trigger_id"]: e["provoking"] for e in state["events"] if e["kind"] == "detected"}
    assert (detected["sees-a-click"]["tool"], detected["sees-a-click"]["role"]) == ("desktop_computer_click", AGENT)
    assert detected["sees-the-flag"]["tool"] == "desktop_computer_pointer_move"
    assert wakes == [([("trigger", "Mail")], "2027-03-01T17:00:00Z", True, True),
                     ([("scheduled_check", "your scheduled check")], "2027-03-01T19:00:00Z", True, True),
                     ("nothing_can_wake_the_agent", "2027-03-02T16:00:00Z", True)], wakes
    deferred = next(e for e in state["events"] if e["kind"] == "deferred")
    assert (deferred["trigger_id"], deferred["until"]) == ("bob-after-hours", "2027-03-02T16:00:00Z")
    assert bob.instructions == ["BOB-REPLIES"]
    bobs = [e["virtual_time"] for e in trajectory if e.get("event_type") == "tool_call" and e.get("role") == bob.role]
    assert bobs == ["2027-03-02T16:00:00Z"] * 2, bobs
    clicks = [e for e in trajectory if e.get("event_type") == "tool_call"
              and e["tool_call"]["function_name"] == "desktop_computer_click"]
    assert [(e["role"], e["virtual_time"]) for e in clicks] == [(AGENT, T0)], clicks


async def _triggers(client: httpx.AsyncClient) -> dict:
    return (await client.get("/triggers/state")).raise_for_status().json()


async def _quiet(client: httpx.AsyncClient, seconds: float = 5) -> None:
    """Until no trigger is firing or evaluating, for at most ``seconds``."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rows = (await _triggers(client))["triggers"]
        if not any(t.get("evaluating") or t["status"] in ("firing", "queued") for t in rows):
            return
        await asyncio.sleep(0.2)


async def _computer_shows(computer: DeployedEnv, virtual_time: str) -> bool:
    """Whether the computer's session clock reads ``virtual_time`` (to the second) within FOLLOW_S."""
    want = int(datetime.fromisoformat(virtual_time.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp())
    deadline = time.monotonic() + FOLLOW_S
    while True:
        ran = await inside.run(computer, ["date", "-u", "+%s"], timeout_s=30)
        if int(ran["stdout"].strip()) == want:
            return True
        if time.monotonic() > deadline:
            return False
        await asyncio.sleep(0.5)


@pytest.mark.asyncio
async def test_a_computers_browser_requests_take_the_role_driving_it_now(desktop, items, shop):  # noqa: F811
    routed = {**CONTAINER, "env_params": {"form": "container", "site_routing": True}}
    health = {"id": "health", "actions": [], "when": {"type": "website", "app": "shop", "method": "GET",
                                                       "path": {"regex": "^/?api/health$"}, "repeat": True}}
    with namespace_routing():
        office = multi_env(items, Env.get(desktop.id), websites=[shop])
        context = TaskStepContext()
        try:
            deployed = await office.deploy(sandbox_type="local", ttl_seconds=1800, child_envs={"desktop": routed})
            context = TaskStepContext(deployed_envs=[deployed], metadata={})
            computer = DeployedEnv.from_dict(deployed.child_envs["desktop"])
            await RegisterEnvTriggersStep(id="seen", version=None, env_id=office.id, triggers=[health],
                                          watch_roles=["person", "helper"]).execute(context)
            async with httpx.AsyncClient(base_url=deployed.gateway_url, timeout=120,
                                         headers=deployed.control_headers) as client:
                loads = []
                for role in ("person", "helper"):
                    moved = await client.post("/step", json={"action": "call_tool", "arguments": {"x": 10, "y": 10},
                                                             "tool_name": "desktop_computer_pointer_move"},
                                              headers={AGENT_ENV_ROLE_HEADER: role})
                    assert moved.status_code == 200, moved.text
                    loads.append(await _reaches(computer, "https://shop.test/api/health"))
                await _quiet(client)
                state = await _triggers(client)
        finally:
            report = await teardown_run(context)

    assert not report.failed and loads == ["200", "200"]
    assert [e["provoking"]["role"] for e in state["events"] if e["kind"] == "detected"] == ["person", "helper"]

