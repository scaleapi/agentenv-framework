"""prompt_agent between_turns next_env_event: between turns the env clock jumps to the next pending time-trigger mark,
the run waits for those triggers, and the agent is woken; with the option unset nothing changes."""
from __future__ import annotations

import copy
import time
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import httpx
import pytest
from a2a.types import TaskState

from agent_env.a2a_agent import A2AAgent
from agent_env.env.gateway.constants import EXT_CLOCK_URI, EXT_TRIGGERS_URI
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps import prompt_agent as pa_mod
from agent_env.task_step.task_steps.prompt_agent import DEFAULT_WAKE_MESSAGE, PromptAgentTaskStep

START = "2026-03-02T13:00:00Z"
DAY1 = "2026-03-03T09:30:00Z"
DAY2 = "2026-03-04T21:15:00Z"


def _dt(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class _Env:
    """A deployed env whose clock and time triggers behave as a gateway's: a trigger fires once the clock reaches its mark."""

    def __init__(self, marks: dict, *, rate: float = 0.0, advance: bool = True, outcome: str = "fired",
                 env_id: str = "env-1", overrun: timedelta | None = None):
        self.env_id, self.rate, self.offers_advance, self.outcome = env_id, rate, advance, outcome
        self.overrun = overrun
        self.now = _dt(START)
        self.calls: list[tuple[str, dict | None]] = []
        self.rows = {tid: {"id": tid, "type": "time", "status": "armed", "next_mark": at if not at.startswith("P") else
                           _z(self.now + timedelta(days=1)), "fire_count": 0, "failure_count": 0,
                           "when": {"type": "time", "at": at}} for tid, at in marks.items()}

    def supports(self, uri: str, method: str) -> bool:
        if uri == EXT_CLOCK_URI:
            return method in ("state", "set_time") or (method == "advance" and self.offers_advance)
        return uri == EXT_TRIGGERS_URI and method == "state"

    async def invoke(self, uri, method, params=None, *, timeout=30, environment_name=None):
        self.calls.append((method, params))
        if uri == EXT_CLOCK_URI and method == "advance" and self.overrun is not None:
            self.now = _dt(params["virtual_time"]) + self.overrun
            raise httpx.HTTPStatusError("advance only moves forward", request=httpx.Request("PUT", "http://gw"),
                                        response=httpx.Response(400, request=httpx.Request("PUT", "http://gw")))
        if uri == EXT_CLOCK_URI and method in ("advance", "set_time"):
            self.now = _dt(params["virtual_time"])
            return {}
        if uri == EXT_CLOCK_URI:
            return {"armed": True, "virtual_time": _z(self.now), "virtual_seconds_per_real_second": self.rate}
        for row in self.rows.values():
            if row.get("stuck"):
                continue
            if row["status"] == "firing" and self.outcome == "fired":
                row.update(status="armed", fire_count=row["fire_count"] + 1)
            elif row["status"] == "armed" and row["next_mark"] and _dt(row["next_mark"]) <= self.now:
                if "every" in row["when"] and self.outcome == "fired":
                    row.update(fire_count=row["fire_count"] + 1, next_mark=_z(_dt(row["next_mark"]) + timedelta(days=1)))
                elif self.outcome == "fired":
                    row.update(status="fired", fire_count=1, next_mark=None)
                elif self.outcome == "failed":
                    row.update(status="failed", failure_count=1, next_mark=None)
        return {"triggers": copy.deepcopy(list(self.rows.values()))}

    def moves(self) -> list[tuple[str, dict]]:
        return [(m, p) for m, p in self.calls if m in ("advance", "set_time")]


def _wire(monkeypatch, *, decide=None):
    """Mock the conversation's HTTP and store boundary; returns what the solver was sent and what the store recorded."""
    cfg = MagicMock()
    cfg.get_model_params.return_value = {}
    cfg.get_default_human_a2a_url.side_effect = AssertionError("no human peer in an event-driven run")
    monkeypatch.setattr(pa_mod, "get_config", lambda: cfg)
    seen = {"solver": [], "stored": [], "closed": 0}
    monkeypatch.setattr(pa_mod.conversation_store, "create_conversation", lambda *a, **kw: None)
    monkeypatch.setattr(pa_mod.conversation_store, "get_conversation", lambda *a, **kw: None)
    monkeypatch.setattr(pa_mod.conversation_store, "add_a2a_task", lambda *a, **kw: seen["stored"].append(("add", kw)))
    monkeypatch.setattr(pa_mod.conversation_store, "complete_a2a_task",
                        lambda *a, **kw: seen["stored"].append(("complete", kw)))

    def closed(*a, **kw):
        seen["closed"] += 1

    monkeypatch.setattr(pa_mod.conversation_store, "mark_closed", closed)

    async def send(url, parts, message_id, context_id, timeout):
        if "user" in url:
            raise AssertionError("the user side is never asked in an event-driven run")
        seen["solver"].append(parts[0]["text"])
        return f"task-{message_id}", None

    async def poll(url, task_id, timeout, interval, **_):
        return {"status": {"state": TaskState.completed, "message": {"parts": [{"kind": "text", "text": "ok"}]}}}

    monkeypatch.setattr(pa_mod.protocol, "send_a2a_message", send)
    monkeypatch.setattr(pa_mod.protocol, "poll_a2a_task", poll)
    monkeypatch.setattr(pa_mod.asyncio, "sleep", _no_sleep)
    if decide is not None:
        monkeypatch.setattr(PromptAgentTaskStep, "_decide", decide)
    return seen


async def _no_sleep(_seconds):
    return None


def _context(*envs, user_sim_triggers: bool = False) -> TaskStepContext:
    ctx = TaskStepContext(instance_id="ti-1", deployed_envs=list(envs))
    ctx.deployed_agents.append(DeployedAgent(agent_name="solver", api_url="http://solver", a2a_url="http://solver",
                                             a2a_card={}))
    if user_sim_triggers:
        card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_TRIGGERS, "params": {
            "endpoint": "/ext/triggers", "methods": {"decide": {"method": "POST", "endpoint": "/ext/triggers/decide"}}}}]}}
        ctx.deployed_agents.append(DeployedAgent(agent_name="stakeholder", api_url="http://user-sim",
                                                 a2a_url="http://user-sim", a2a_card=card))
        ctx.metadata["agent_trigger_registrations"] = [{"agent_name": "stakeholder", "added": ["news"]}]
    return ctx


def _step(between_turns, turns: int = 6, **kw) -> PromptAgentTaskStep:
    return PromptAgentTaskStep(id="watch", version=None, prompt="Watch my day.", agent_name="solver",
                               max_conversation_turns=turns, between_turns=between_turns, **kw)


def test_unset_adds_nothing_to_the_stored_step():
    data = PromptAgentTaskStep(id="s", version=None, prompt="hi").to_dict()
    assert "between_turns" not in data and "record_env_clock" not in data
    again = PromptAgentTaskStep.from_dict(data)
    assert (again.between_turns, again.record_env_clock) == (None, False)


@pytest.mark.parametrize("given", ["next_env_event", {"mode": "next_env_event", "end_time": DAY2, "wake": "message"}])
def test_the_option_round_trips_as_given(given):
    step = PromptAgentTaskStep.from_dict(_step(given, record_env_clock=True).to_dict())
    assert (step.between_turns, step.record_env_clock) == (given, True)


@pytest.mark.parametrize("given,fragment", [
    ("every_turn", "mode must be"),
    ({"mode": "next_env_event", "until": DAY1}, "unknown key"),
    ({"mode": "next_env_event", "wake": "user"}, "wake must be one of"),
    ({"mode": "next_env_event", "wake_message": "{now}"}, "may use only"),
    ({"mode": "next_env_event", "wake_message": " "}, "non-empty"),
    ({"mode": "next_env_event", "end_time": "Tuesday"}, "end_time"),
    ({"mode": "next_env_event", "end_time": "2026-03-04T21:15:00"}, "end_time"),
    ({"mode": "next_env_event", "fire_timeout_seconds": 0}, "positive"),
    (["next_env_event"], "or an object with a mode"),
])
def test_a_bad_option_is_refused_at_construction(given, fragment):
    with pytest.raises(ValueError, match=fragment):
        _step(given)


def test_a_single_turn_step_cannot_act_between_turns():
    with pytest.raises(ValueError, match="max_conversation_turns >= 2"):
        _step("next_env_event", turns=1)


@pytest.mark.asyncio
async def test_two_events_days_apart_each_wake_the_agent_once_then_the_run_ends(monkeypatch):
    seen = _wire(monkeypatch)
    env = _Env({"school-email": DAY1, "calendar-clash": DAY2})

    ctx = await _step("next_env_event").execute(_context(env))

    assert seen["solver"] == ["Watch my day.", DEFAULT_WAKE_MESSAGE.format(virtual_time=DAY1),
                              DEFAULT_WAKE_MESSAGE.format(virtual_time=DAY2)]
    assert env.moves() == [("advance", {"virtual_time": DAY1}), ("advance", {"virtual_time": DAY2})]
    advances = ctx.metadata["env_event_advances"]["watch"]
    assert advances[0] == {"turn": 1, "mark": DAY1, "trigger_ids": ["school-email"], "statuses": {"env-1": {"school-email": "fired"}},
                           "moved": {"env-1": "advance"}}
    assert advances[1]["trigger_ids"] == ["calendar-clash"]
    assert advances[2] == {"turn": 3, "stopped": "no_pending_time_triggers", "virtual_time": None}
    assert seen["closed"] >= 1


@pytest.mark.asyncio
async def test_every_message_carries_the_env_clock_when_the_run_is_event_driven(monkeypatch):
    seen = _wire(monkeypatch)
    env = _Env({"school-email": DAY1})

    await _step("next_env_event").execute(_context(env))

    stamps = [(kind, kw["env_clock"]) for kind, kw in seen["stored"]]
    assert stamps == [("add", {"env-1": START}), ("complete", {"env-1": START}),
                      ("add", {"env-1": DAY1}), ("complete", {"env-1": DAY1})]


@pytest.mark.asyncio
async def test_end_time_stops_the_run_and_brings_the_clock_to_it(monkeypatch):
    seen = _wire(monkeypatch)
    env = _Env({"day-1": DAY1, "day-5": "2026-03-07T09:00:00Z"})
    end = "2026-03-05T07:00:00Z"

    ctx = await _step({"mode": "next_env_event", "end_time": end}).execute(_context(env))

    assert len(seen["solver"]) == 2
    assert env.moves()[-1] == ("advance", {"virtual_time": end})
    assert ctx.metadata["env_event_advances"]["watch"][-1] == {"turn": 2, "stopped": "end_time", "virtual_time": end}
    assert env.rows["day-5"]["status"] == "armed"


@pytest.mark.asyncio
async def test_max_conversation_turns_still_bounds_the_run(monkeypatch):
    seen = _wire(monkeypatch)
    env = _Env({"a": DAY1, "b": DAY2, "c": "2026-03-06T00:00:00Z"})

    await _step("next_env_event", turns=2).execute(_context(env))

    assert len(seen["solver"]) == 2 and env.moves() == [("advance", {"virtual_time": DAY1})]


@pytest.mark.asyncio
async def test_triggers_due_at_one_mark_share_one_wake(monkeypatch):
    seen = _wire(monkeypatch)
    env = _Env({"sale": DAY1, "text": DAY1})

    await _step({"mode": "next_env_event", "wake_message": "ping {virtual_time}: {trigger_ids}"}).execute(_context(env))

    assert seen["solver"][1:] == [f"ping {DAY1}: sale, text"]


@pytest.mark.asyncio
async def test_one_trigger_id_on_two_envs_is_awaited_on_each_without_waiting_out_the_timeout(monkeypatch):
    seen = _wire(monkeypatch)
    envs = (_Env({"news": DAY1}, env_id="env-1"), _Env({"news": DAY1}, env_id="env-2"))
    started = time.monotonic()

    ctx = await _step({"mode": "next_env_event", "fire_timeout_seconds": 5}, turns=2).execute(_context(*envs))

    assert time.monotonic() - started < 2
    advance = ctx.metadata["env_event_advances"]["watch"][0]
    assert advance["statuses"] == {"env-1": {"news": "fired"}, "env-2": {"news": "fired"}}
    assert (advance["trigger_ids"], advance["moved"]) == (["news"], {"env-1": "advance", "env-2": "advance"})
    assert seen["solver"][1] == DEFAULT_WAKE_MESSAGE.format(virtual_time=DAY1)


@pytest.mark.asyncio
async def test_a_running_clock_that_passes_the_mark_before_the_advance_lands_is_not_an_error(monkeypatch):
    seen = _wire(monkeypatch)
    env = _Env({"school-email": DAY1}, rate=60.0, overrun=timedelta(seconds=2))

    ctx = await _step("next_env_event", turns=2).execute(_context(env))

    advance = ctx.metadata["env_event_advances"]["watch"][0]
    assert (advance["moved"], advance["statuses"]) == ({"env-1": "passed"}, {"env-1": {"school-email": "fired"}})
    assert len(seen["solver"]) == 2


@pytest.mark.asyncio
async def test_an_advance_refused_for_another_reason_still_fails_the_step(monkeypatch):
    _wire(monkeypatch)
    env = _Env({"school-email": DAY1}, overrun=timedelta(seconds=-2))

    with pytest.raises(httpx.HTTPStatusError):
        await _step("next_env_event", turns=2).execute(_context(env))


@pytest.mark.asyncio
async def test_a_recurring_trigger_still_mid_fire_keeps_its_next_mark_in_the_schedule(monkeypatch):
    seen = _wire(monkeypatch)
    env = _Env({})
    env.rows["daily"] = {"id": "daily", "type": "time", "status": "firing", "next_mark": DAY1, "fire_count": 3,
                         "failure_count": 0, "when": {"type": "time", "every": "P1D"}}

    ctx = await _step("next_env_event", turns=3).execute(_context(env))

    assert env.moves() == [("advance", {"virtual_time": DAY1}), ("advance", {"virtual_time": "2026-03-04T09:30:00Z"})]
    assert [a.get("trigger_ids") for a in ctx.metadata["env_event_advances"]["watch"]] == [["daily"], ["daily"]]
    assert len(seen["solver"]) == 3


@pytest.mark.asyncio
async def test_an_unfinished_recurring_fire_does_not_hold_time_at_its_mark(monkeypatch):
    seen = _wire(monkeypatch)
    env = _Env({"later": DAY2})
    env.rows["daily"] = {"id": "daily", "type": "time", "status": "firing", "next_mark": DAY1, "fire_count": 1,
                         "failure_count": 0, "when": {"type": "time", "every": "P1D"}, "stuck": True}

    ctx = await _step({"mode": "next_env_event", "fire_timeout_seconds": 0.01}, turns=5).execute(_context(env))

    assert env.moves() == [("advance", {"virtual_time": DAY1}), ("advance", {"virtual_time": DAY2})]
    advances = ctx.metadata["env_event_advances"]["watch"]
    assert advances[0]["statuses"] == {"env-1": {"daily": "timeout"}}
    assert advances[1]["statuses"] == {"env-1": {"later": "fired"}}
    assert advances[2]["stopped"] == "no_pending_time_triggers" and len(seen["solver"]) == 3


@pytest.mark.asyncio
async def test_a_gateway_without_advance_is_rearmed_at_its_own_rate(monkeypatch):
    _wire(monkeypatch)
    env = _Env({"school-email": DAY1}, rate=60.0, advance=False)

    await _step("next_env_event").execute(_context(env))

    assert env.moves() == [("set_time", {"virtual_time": DAY1, "virtual_seconds_per_real_second": 60.0})]


@pytest.mark.asyncio
async def test_a_gateway_without_advance_refuses_to_rearm_over_a_relative_mark(monkeypatch):
    _wire(monkeypatch)
    env = _Env({"school-email": DAY1, "reminder": "P1D"}, advance=False)

    with pytest.raises(RuntimeError, match=r"re-anchor .*\['reminder'\]"):
        await _step("next_env_event").execute(_context(env))
    assert env.moves() == []


@pytest.mark.asyncio
async def test_a_gateway_without_advance_refuses_to_rearm_to_end_time_over_a_relative_mark(monkeypatch):
    _wire(monkeypatch)
    env = _Env({"reminder": "P1D"}, advance=False)

    with pytest.raises(RuntimeError, match=r"re-anchor .*\['reminder'\]"):
        await _step({"mode": "next_env_event", "end_time": "2026-03-02T20:00:00Z"}).execute(_context(env))
    assert env.moves() == []


@pytest.mark.asyncio
async def test_a_failed_or_stuck_trigger_is_reported_and_the_agent_still_woken(monkeypatch):
    seen = _wire(monkeypatch)
    failed = await _step("next_env_event", turns=2).execute(_context(_Env({"e": DAY1}, outcome="failed")))
    stuck = await _step({"mode": "next_env_event", "fire_timeout_seconds": 0.01}, turns=2).execute(
        _context(_Env({"e": DAY1}, outcome="stuck")))

    assert failed.metadata["env_event_advances"]["watch"][0]["statuses"] == {"env-1": {"e": "failed"}}
    assert stuck.metadata["env_event_advances"]["watch"][0]["statuses"] == {"env-1": {"e": "timeout"}}
    assert len(seen["solver"]) == 4


@pytest.mark.asyncio
async def test_wake_decide_lets_agent_triggers_speak_and_falls_back_to_the_wake_message(monkeypatch):
    calls = []

    async def decide(self, url, turn, message, context_id, env_triggers):
        calls.append((turn, env_triggers))
        return {"fired": ["news"], "done": False, "parts": [{"kind": "text", "text": "[news] the school wrote"}]} \
            if turn == 1 else {"fired": [], "done": False, "parts": []}

    async def readback(self, url, timeout):
        return httpx.Response(200, json={"firing_log": []}, request=httpx.Request("GET", url))

    seen = _wire(monkeypatch, decide=decide)
    monkeypatch.setattr(httpx.AsyncClient, "get", readback)
    env = _Env({"school-email": DAY1, "calendar-clash": DAY2})

    await _step({"mode": "next_env_event", "wake": "decide"}, user_agent_name="stakeholder").execute(
        _context(env, user_sim_triggers=True))

    assert seen["solver"][1:] == ["[news] the school wrote", DEFAULT_WAKE_MESSAGE.format(virtual_time=DAY2)]
    assert calls[0] == (1, {"env-1": {"school-email": "fired", "calendar-clash": "armed"}})


@pytest.mark.asyncio
async def test_a_run_with_no_armed_env_clock_ends_after_the_first_turn(monkeypatch):
    seen = _wire(monkeypatch)

    ctx = await _step("next_env_event").execute(_context())

    assert len(seen["solver"]) == 1
    assert ctx.metadata["env_event_advances"]["watch"] == [{"turn": 1, "stopped": "no_armed_env_clock", "virtual_time": None}]


@pytest.mark.asyncio
async def test_an_agent_trigger_that_ends_the_run_records_its_message_with_the_env_clock(monkeypatch):
    async def decide(self, url, turn, message, context_id, env_triggers):
        return {"fired": ["bye"], "done": True, "parts": [{"kind": "text", "text": "[news] that's all"}]}

    async def readback(self, url, timeout):
        return httpx.Response(200, json={"firing_log": []}, request=httpx.Request("GET", url))

    seen = _wire(monkeypatch, decide=decide)
    monkeypatch.setattr(httpx.AsyncClient, "get", readback)

    await _step({"mode": "next_env_event", "wake": "decide"}, user_agent_name="stakeholder").execute(
        _context(_Env({"school-email": DAY1}), user_sim_triggers=True))

    assert seen["stored"][-1] == ("add", {"parts": [{"kind": "text", "text": "[news] that's all"}],
                                          "a2a_task_id": seen["stored"][-1][1]["a2a_task_id"], "role": "user",
                                          "env_clock": {"env-1": DAY1}})


@pytest.mark.asyncio
async def test_wake_decide_without_agent_triggers_is_refused(monkeypatch):
    _wire(monkeypatch)
    with pytest.raises(RuntimeError, match="wake 'decide' needs"):
        await _step({"mode": "next_env_event", "wake": "decide"}).execute(_context(_Env({"e": DAY1})))


@pytest.mark.asyncio
async def test_record_env_clock_stamps_a_plain_conversation_and_unset_stamps_nothing(monkeypatch):
    seen = _wire(monkeypatch)
    env = _Env({})
    await PromptAgentTaskStep(id="s", version=None, prompt="hi", agent_name="solver",
                              record_env_clock=True).execute(_context(env))
    await PromptAgentTaskStep(id="s", version=None, prompt="hi", agent_name="solver").execute(_context(env))

    assert [kw.get("env_clock", "absent") for _, kw in seen["stored"]] == [{"env-1": START}, {"env-1": START},
                                                                           "absent", "absent"]
    assert env.calls == [("state", None), ("state", None)]
