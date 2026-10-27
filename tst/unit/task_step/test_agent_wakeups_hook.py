"""The agent-wakeups example hook, turn by turn, against an in-process gateway's real clock, triggers and wakeups."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

import agent_env
from agent_env.env.env import DeployedGatewayEnv
from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.clock import Clock
from agent_env.env.gateway.constants import GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.env.gateway.gateway import Gateway
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.post_turn_hooks import PostTurnHook, Turn, _Answer

HOOK = Path(agent_env.__file__).parent / "examples" / "agent-wakeups" / "artifacts" / "agent-wakeups" / "hook.py"
GATEWAY = "http://gateway.test"
MONDAY = "2026-03-02T08:00:00Z"


@pytest.fixture
def hook(monkeypatch):
    spec = importlib.util.spec_from_file_location("agent_wakeups_hook", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "POLL_SECONDS", 0.01)
    return module


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """A gateway on a frozen clock armed Monday 08:00 whose world writes record the virtual time they ran at."""
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    gw = Gateway(host="127.0.0.1", port=0, server_name="AgentEnvGateway", internal_mcp_servers=[])
    gw._clock = Clock()
    gw._clock.set_time(MONDAY, rate=0)
    gw._tool_server_urls = {"items_add_item": "http://items/mcp"}
    gw.calls = []

    async def call_child_tool(server_url, tool_name, arguments, *, caller=None):
        gw.calls.append((arguments["item"], gw._clock.read()["virtual_time"]))
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)

    gw._call_child_tool = call_child_tool
    gw._wakeups.configure({"scheduled_checks": True})
    real_client = httpx.AsyncClient
    transport = httpx.ASGITransport(app=gw._mcp.streamable_http_app())
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_client(*a, **{**kw, "transport": transport}))
    yield gw
    gw._close_trajectory_file()


def _context() -> TaskStepContext:
    card = {"capabilities": {"extensions": list(GATEWAY_EXTENSIONS)}}
    env = DeployedGatewayEnv(env_id="items", env_version=1, gateway_url=GATEWAY, mcp_url="", sandbox_id="sb",
                             environment_card_url=f"{GATEWAY}{WELL_KNOWN_PATH}", environment_card=card,
                             metadata={"deploy_step_id": "deploy_items"})
    return TaskStepContext(deployed_envs=[env])


async def _turns(hook, context, params=None, on_turn=None, triggers=None) -> list:
    """Each turn's decision until the hook ends the conversation; ``on_turn(n)`` plays the agent's turn n first."""
    answers = list(triggers or [])

    async def user_triggers():
        return answers.pop(0) if answers else None

    async def ask_user():
        raise AssertionError("the hook never asks the user side")

    decisions = []
    while not decisions or "end" not in decisions[-1]:
        assert len(decisions) < 12, decisions
        number = len(decisions) + 1
        if on_turn:
            on_turn(number)
        turn = Turn(context=context, step_id="week", hook=PostTurnHook("agent-wakeups", None, {}, hook.post_turn),
                    answer=_Answer(ask_user, user_triggers), number=number, last=False, agent_reply="ok",
                    agent_parts=[{"kind": "text", "text": "ok"}])
        decisions.append(await hook.post_turn(turn, dict(params or {})))
    return decisions


def _register(gw: Gateway, *triggers: dict) -> None:
    gw._trigger_engine.register({"watch_roles": ["default"], "triggers": list(triggers)})


@pytest.mark.asyncio
async def test_a_world_event_with_no_channel_happens_silently_and_the_agent_s_own_check_wakes_it(hook, gateway):
    _register(gateway, {"id": "w-1", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"},
                        "actions": [{"type": "tool", "tool": "items_add_item", "args": {"item": "rug listed"}}]})

    def agent(n):
        if n == 1:
            gateway._wakeups.schedule("check for new rugs", at="2026-03-03T18:00:00Z")

    decisions = await _turns(hook, _context(), on_turn=agent)

    assert decisions == [{"message": "[Tue Mar 3, 6:00 PM · your scheduled check] check for new rugs"},
                         {"end": "nothing_can_wake_the_agent"}]
    assert gateway.calls == [("rug listed", "2026-03-03T09:00:00Z")]


@pytest.mark.asyncio
async def test_a_notify_agent_action_wakes_the_agent_when_it_fires_with_its_channel(hook, gateway):
    _register(gateway, {"id": "w-2", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"},
                        "actions": [{"type": "notify_agent", "channel": "Curbfinch alert",
                                     "text": 'New for "rug 8x10": 1970s wool rug · $360'}]})
    decisions = await _turns(hook, _context(), params={"time_zone": "America/Los_Angeles"})
    assert decisions == [{"message": '[Tue Mar 3, 1:00 AM · Curbfinch alert] New for "rug 8x10": 1970s wool rug · $360'},
                         {"end": "nothing_can_wake_the_agent"}]


@pytest.mark.asyncio
async def test_the_person_s_messages_arrive_at_their_times_once_each(hook, gateway):
    params = {"person_messages": [{"at": "2026-03-04T16:30:00Z", "text": "got Leo, we're home"},
                                  {"at": "2026-03-03T15:00:00Z", "text": "I'm up"}]}
    decisions = await _turns(hook, _context(), params=params)
    assert decisions == [{"message": "[Tue Mar 3, 3:00 PM] I'm up"},
                         {"message": "[Wed Mar 4, 4:30 PM] got Leo, we're home"},
                         {"end": "nothing_can_wake_the_agent"}]


@pytest.mark.asyncio
async def test_what_comes_due_together_goes_in_one_message_oldest_first(hook, gateway):
    _register(gateway, {"id": "w-3", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"},
                        "actions": [{"type": "notify_agent", "channel": "Gmail", "text": "School: early dismissal"}]})

    def agent(n):
        if n == 1:
            gateway._wakeups.schedule("morning sweep", at="2026-03-03T09:00:00Z")

    [first, end] = await _turns(hook, _context(), on_turn=agent)
    assert first == {"message": "[Tue Mar 3, 9:00 AM · Gmail] School: early dismissal\n"
                                "[Tue Mar 3, 9:00 AM · your scheduled check] morning sweep"}


@pytest.mark.asyncio
async def test_it_ends_at_end_time_without_waking_for_what_lies_beyond(hook, gateway):
    gateway._wakeups.schedule("next week", at="2026-03-10T09:00:00Z")
    assert await _turns(hook, _context(), params={"end_time": "P5D"}) == [{"end": "end_time"}]


@pytest.mark.asyncio
async def test_a_state_check_the_agents_turn_set_off_finishes_before_the_clock_jumps(hook, gateway, monkeypatch):
    async def slow_check(check, ctx=None, role=None):
        await asyncio.sleep(0.3)
        return True

    monkeypatch.setattr(gateway._trigger_engine, "_run_check", slow_check)
    _register(gateway, {"id": "seen", "when": {"type": "state", "check": {"tool": "items_list", "predicate": {"exists": True}}},
                        "actions": [{"type": "tool", "tool": "items_add_item", "args": {"item": "seen"}}]},
              {"id": "tue", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"},
               "actions": [{"type": "tool", "tool": "items_add_item", "args": {"item": "tue"}}]})
    gateway._trigger_engine.on_tool_call("default", "send_email", {}, CallToolResult(content=[], isError=False))
    await _turns(hook, _context())
    assert gateway.calls[0] == ("seen", MONDAY)


@pytest.mark.asyncio
async def test_a_firing_agent_trigger_answers_before_any_jump(hook, gateway):
    gateway._wakeups.schedule("later", at="2026-03-03T09:00:00Z")
    decisions = await _turns(hook, _context(), triggers=[{"message": "the person pushes back"}])
    assert decisions[0] == {"message": "the person pushes back"}
    assert decisions[1] == {"message": "[Tue Mar 3, 9:00 AM · your scheduled check] later"}


@pytest.mark.asyncio
async def test_a_repeating_check_wakes_once_per_time_and_the_record_names_why(hook, gateway):
    gateway._wakeups.schedule("sweep", every="P1D", until="P2D")
    context = _context()
    decisions = await _turns(hook, context)
    assert decisions == [{"message": "[Tue Mar 3, 8:00 AM · your scheduled check] sweep"},
                         {"message": "[Wed Mar 4, 8:00 AM · your scheduled check] sweep"},
                         {"end": "nothing_can_wake_the_agent"}]
    records = context.metadata["post_turn_hooks"]["week"]
    assert [r.get("for") for r in records[:2]] == [["scheduled_check"], ["scheduled_check"]]


def test_unknown_params_are_refused(hook):
    with pytest.raises(ValueError, match="takes"):
        hook._params({"nudge": "x"})
