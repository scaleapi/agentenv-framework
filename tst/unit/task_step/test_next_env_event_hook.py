"""The next-env-event example hook, turn by turn, against an in-process gateway's real clock and trigger engine."""

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
from agent_env.env.gateway.constants import EXT_CLOCK_URI, GATEWAY_EXTENSIONS, WELL_KNOWN_PATH
from agent_env.env.gateway.gateway import Gateway
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.post_turn_hooks import PostTurnHook, Turn, _Answer

HOOK = Path(agent_env.__file__).parent / "examples" / "next-env-event" / "artifacts" / "next-env-event" / "hook.py"
GATEWAY = "http://gateway.test"
MONDAY = "2026-03-02T08:00:00Z"


def _load_hook():
    spec = importlib.util.spec_from_file_location("next_env_event_hook", HOOK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def hook(monkeypatch):
    module = _load_hook()
    monkeypatch.setattr(module, "POLL_SECONDS", 0.01)
    return module


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """A gateway on a frozen clock armed Monday 08:00 whose tool calls record the virtual time they ran at."""
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
    real_client = httpx.AsyncClient
    transport = httpx.ASGITransport(app=gw._mcp.streamable_http_app())
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: real_client(*a, **{**kw, "transport": transport}))
    yield gw
    gw._close_trajectory_file()


def _context(*extensions: dict) -> TaskStepContext:
    card = {"capabilities": {"extensions": list(extensions or GATEWAY_EXTENSIONS)}}
    env = DeployedGatewayEnv(env_id="items", env_version=1, gateway_url=GATEWAY, mcp_url="", sandbox_id="sb",
                             environment_card_url=f"{GATEWAY}{WELL_KNOWN_PATH}", environment_card=card,
                             metadata={"deploy_step_id": "deploy_items"})
    return TaskStepContext(deployed_envs=[env])


async def _turns(hook, context: TaskStepContext, params: dict | None = None, triggers: list | None = None) -> list:
    """Each turn's decision until the hook ends the conversation, with agent triggers answering from ``triggers``."""
    answers = list(triggers or [])

    async def user_triggers():
        return answers.pop(0) if answers else None

    async def ask_user():
        raise AssertionError("the hook never asks the user side")

    decisions = []
    while not decisions or "end" not in decisions[-1]:
        assert len(decisions) < 10, decisions
        number = len(decisions) + 1
        turn = Turn(context=context, step_id="week", hook=PostTurnHook("next-env-event", None, {}, hook.post_turn),
                    answer=_Answer(ask_user, user_triggers), number=number, last=False, agent_reply="ok",
                    agent_parts=[{"kind": "text", "text": "ok"}])
        decisions.append(await hook.post_turn(turn, dict(params or {})))
    return decisions


def _nudge(at: str) -> dict:
    return {"message": f"[{at}] Time has passed. Check for anything new and handle it."}


def _register(gw: Gateway, *triggers: dict) -> None:
    gw._trigger_engine.register({"watch_roles": ["default"], "triggers": list(triggers)})


def _add(item: str, **action) -> list[dict]:
    return [{"type": "tool", "tool": "items_add_item", "args": {"item": item}, **action}]


@pytest.mark.asyncio
async def test_a_delayed_action_gets_a_wake_of_its_own_at_its_mark_plus_the_delay(hook, gateway):
    _register(gateway,
              {"id": "school", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"},
               "actions": _add("school email", delay="PT2H")},
              {"id": "calendar", "when": {"type": "time", "at": "2026-03-05T09:00:00Z"}, "actions": _add("clash")})
    context = _context()

    decisions = await _turns(hook, context)

    assert decisions == [_nudge("2026-03-03T09:00:00Z"), _nudge("2026-03-03T11:00:00Z"),
                         _nudge("2026-03-05T09:00:00Z"), {"end": "no_pending_events"}]
    assert gateway.calls == [("school email", "2026-03-03T11:00:00Z"), ("clash", "2026-03-05T09:00:00Z")]
    records = context.metadata["post_turn_hooks"]["week"]
    assert [(r["event_at"], r["envs"]) for r in records[:3]] == [
        (at, [{"env_id": "items", "env_step_id": "deploy_items", "due": [due], "settled": True}])
        for at, due in (("2026-03-03T09:00:00Z", "school"), ("2026-03-03T11:00:00Z", "school"),
                        ("2026-03-05T09:00:00Z", "calendar"))]
    assert records[-1] == {"hook": "next-env-event", "turn": 4, "end": "no_pending_events"}


@pytest.mark.asyncio
async def test_a_delayed_action_with_no_later_event_and_no_end_time_still_runs(hook, gateway):
    _register(gateway, {"id": "school", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"},
                        "actions": _add("school email", delay="PT2H")})

    decisions = await _turns(hook, _context())

    assert decisions[-1] == {"end": "no_pending_events"} and len(decisions) == 3
    assert gateway.calls == [("school email", "2026-03-03T11:00:00Z")]


@pytest.mark.asyncio
async def test_a_recurring_trigger_whose_action_outlasts_its_interval_still_moves_on(hook, gateway):
    _register(gateway, {"id": "beat", "when": {"type": "time", "every": "PT1H", "count": 2},
                        "actions": _add("beat", delay="PT2H")})

    decisions = await _turns(hook, _context())

    assert decisions[-1] == {"end": "no_pending_events"}
    assert gateway.calls == [("beat", "2026-03-02T11:00:00Z"), ("beat", "2026-03-02T12:00:00Z")]


@pytest.mark.asyncio
async def test_a_fire_the_agents_turn_set_off_lands_before_the_clock_jumps(hook, gateway, monkeypatch):
    real = gateway._call_child_tool

    async def slow(server_url, tool_name, arguments, *, caller=None):
        await asyncio.sleep(0.3)
        return await real(server_url, tool_name, arguments, caller=caller)

    gateway._call_child_tool = slow
    _register(gateway, {"id": "ack", "when": {"type": "action", "tool": "send_email"}, "actions": _add("ack")},
              {"id": "tue", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"}, "actions": _add("tue")})
    gateway._trigger_engine.on_tool_call("default", "send_email", {}, CallToolResult(content=[], isError=False))

    decisions = await _turns(hook, _context())

    assert decisions[0] == _nudge("2026-03-03T09:00:00Z")
    assert gateway.calls == [("ack", MONDAY), ("tue", "2026-03-03T09:00:00Z")]


@pytest.mark.asyncio
async def test_a_state_check_the_agents_turn_set_off_finishes_before_the_clock_jumps(hook, gateway, monkeypatch):
    async def slow_check(check, ctx=None, role=None):
        await asyncio.sleep(0.3)
        return True

    monkeypatch.setattr(gateway._trigger_engine, "_run_check", slow_check)
    _register(gateway, {"id": "seen", "when": {"type": "state", "check": {"tool": "items_list", "predicate": {"exists": True}}},
                        "actions": _add("seen")},
              {"id": "tue", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"}, "actions": _add("tue")})
    gateway._trigger_engine.on_tool_call("default", "send_email", {}, CallToolResult(content=[], isError=False))
    assert gateway._trigger_engine.state()["triggers"][0]["evaluating"] is True

    await _turns(hook, _context())

    assert gateway.calls[0] == ("seen", MONDAY)


@pytest.mark.asyncio
async def test_an_agent_trigger_reply_is_sent_first_at_the_current_time_with_no_jump(hook, gateway):
    _register(gateway, {"id": "school", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"},
                        "actions": _add("school email")})
    reply = {"message": [{"kind": "text", "text": "[Julia]: thanks, that works"}]}

    decisions = await _turns(hook, _context(), triggers=[reply])

    assert decisions[0] == reply
    assert decisions[1] == _nudge("2026-03-03T09:00:00Z")
    assert gateway.calls == [("school email", "2026-03-03T09:00:00Z")]


@pytest.mark.asyncio
async def test_an_agent_trigger_that_answers_the_event_speaks_instead_of_the_nudge(hook, gateway):
    _register(gateway, {"id": "school", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"},
                        "actions": _add("school email")})
    reply = {"message": [{"kind": "text", "text": "[school]: did you see our email?"}]}

    decisions = await _turns(hook, _context(), triggers=[None, reply])

    assert decisions[0] == reply and gateway._clock.read()["virtual_time"] == "2026-03-03T09:00:00Z"


@pytest.mark.asyncio
@pytest.mark.parametrize("end_time", ["2026-03-04T00:00:00Z", "P1DT16H"])
async def test_an_event_past_end_time_ends_the_conversation_without_jumping_to_it(hook, gateway, end_time):
    _register(gateway, {"id": "tue", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"}, "actions": _add("a")},
              {"id": "thu", "when": {"type": "time", "at": "2026-03-05T09:00:00Z"}, "actions": _add("b")})
    context = _context()

    decisions = await _turns(hook, context, {"end_time": end_time})

    assert decisions == [_nudge("2026-03-03T09:00:00Z"), {"end": "end_time"}]
    assert gateway._clock.read()["virtual_time"] == "2026-03-03T09:00:00Z"
    assert context.metadata["post_turn_hooks"]["week"][-1]["next_event_at"] == "2026-03-05T09:00:00Z"


@pytest.mark.asyncio
async def test_a_custom_nudge_names_the_time(hook, gateway):
    _register(gateway, {"id": "tue", "when": {"type": "time", "at": "PT1H"}, "actions": _add("a")})

    decisions = await _turns(hook, _context(), {"nudge": "It is now {virtual_time}."})

    assert decisions[0] == {"message": "It is now 2026-03-02T09:00:00Z."}


@pytest.mark.asyncio
async def test_with_no_armed_clock_the_conversation_ends(hook, gateway):
    gateway._clock.clear()

    assert await _turns(hook, _context()) == [{"end": "no_armed_env_clock"}]


@pytest.mark.asyncio
async def test_a_gateway_without_advance_ends_the_conversation_naming_the_env_and_never_re_arms(hook, gateway, caplog):
    _register(gateway, {"id": "tue", "when": {"type": "time", "at": "PT1H"}, "actions": _add("a")})
    old_clock = next(e for e in GATEWAY_EXTENSIONS if e["uri"] == EXT_CLOCK_URI)
    methods = {k: v for k, v in old_clock["params"]["methods"].items() if k != "advance"}
    extensions = [e for e in GATEWAY_EXTENSIONS if e["uri"] != EXT_CLOCK_URI]
    context = _context(*extensions, {**old_clock, "params": {**old_clock["params"], "methods": methods}})

    decisions = await _turns(hook, context)

    assert decisions == [{"end": "gateway_without_advance"}]
    assert context.metadata["post_turn_hooks"]["week"][-1]["envs"] == ["items"]
    assert "rebuild their gateway image" in caplog.text
    assert gateway._clock.read()["virtual_time"] == MONDAY


@pytest.mark.asyncio
@pytest.mark.parametrize("params, fragment", [
    ({"wake": "decide"}, "not ['wake']"),
    ({"end_time": "2026-03-04T00:00:00"}, "RFC 3339 instant with an offset"),
    ({"end_time": 4}, "RFC 3339 instant with an offset"),
    ({"end_time": "P"}, "a duration like 'P4D'"),
    ({"nudge": "{trigger_ids}"}, "one placeholder is {virtual_time}"),
    ({"settle_timeout_seconds": 0}, "positive number"),
])
async def test_params_the_hook_cannot_use_are_refused(hook, gateway, params, fragment):
    with pytest.raises(ValueError) as e:
        await _turns(hook, _context(), params)
    assert fragment in str(e.value)
