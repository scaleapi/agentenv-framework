"""The engine's next_event_at and resumes_at, what an advance makes due, and the /step tag of a hook's tool calls."""
from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.clock import Clock, _iso
from agent_env.env.gateway.constants import (
    EXT_STEP_URI, EXT_TRIGGERS_URI, GATEWAY_EXTENSIONS, STEP_SOURCE, TRIGGER_NEXT_EVENT_AT,
)
from agent_env.env.gateway.gateway import Gateway
from agent_env.env.gateway.triggers import TriggerEngine
from tst.unit.env.gateway.triggers_time_test import FakeNow, _FakeGateway, _perm, _result, _settle

MONDAY = "2026-03-02T00:00:00Z"


def _engine(rate: float = 0.0) -> tuple[TriggerEngine, Clock, FakeNow]:
    now = FakeNow()
    clock = Clock(now_fn=now)
    clock.set_time(MONDAY, rate=rate)
    return TriggerEngine(_FakeGateway(clock)), clock, now


def _time(tid: str, **when) -> dict:
    return {"id": tid, "when": {"type": "time", **when}, "actions": [_perm()]}


def _next(engine: TriggerEngine) -> str | None:
    return engine.state()["next_event_at"]


def _row(engine: TriggerEngine, tid: str) -> dict:
    return next(t for t in engine.state()["triggers"] if t["id"] == tid)


async def _until(predicate, seconds: float = 2.0) -> None:
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached"
        await asyncio.sleep(0.01)


# ---- next_event_at ---------------------------------------------------------------------------------------------------

def test_the_next_event_is_the_earliest_armed_mark_before_the_engine_has_evaluated_any():
    engine, _, _ = _engine()
    engine.register({"triggers": [_time("thu", at="P2DT14H15M"), _time("tue", at="2026-03-03T09:00:00Z")]})

    assert _next(engine) == "2026-03-03T09:00:00Z"
    assert [_row(engine, tid)["next_mark"] for tid in ("thu", "tue")] == [None, None]


@pytest.mark.asyncio
async def test_a_recurring_trigger_is_next_due_one_interval_on_and_again_after_each_arrival():
    engine, clock, _ = _engine()
    engine.register({"triggers": [_time("beat", every="PT6H")]})
    assert _next(engine) == "2026-03-02T06:00:00Z"

    clock.advance("2026-03-02T07:00:00Z")
    engine._eval_time_triggers()
    await _settle(engine)

    assert _next(engine) == "2026-03-02T12:00:00Z"


def test_a_stochastic_mark_is_read_without_drawing_it_so_the_engine_stamps_the_same_one():
    engine, _, _ = _engine()
    engine.register({"triggers": [_time("poisson", every={"dist": "exp", "mean": "PT1H"}, seed=7)]})

    first, again = _next(engine), _next(engine)
    engine._eval_time_triggers()

    assert first == again == _row(engine, "poisson")["next_mark"]


def test_after_a_re_arm_a_relative_mark_is_read_on_the_new_t0():
    engine, clock, _ = _engine()
    engine.register({"triggers": [_time("rel", at="PT3H")]})
    engine._eval_time_triggers()
    clock.set_time("2026-03-05T00:00:00Z", rate=0)

    assert _next(engine) == "2026-03-05T03:00:00Z"


@pytest.mark.asyncio
async def test_a_mark_past_until_or_past_count_is_no_event():
    engine, clock, _ = _engine()
    engine.register({"triggers": [_time("capped", every="PT1H", until="PT90M"), _time("twice", every="PT4H", count=2)]})
    clock.advance("2026-03-02T01:00:00Z")
    engine._eval_time_triggers()
    await _settle(engine)
    assert _next(engine) == "2026-03-02T04:00:00Z"

    clock.advance("2026-03-02T08:00:00Z")
    engine._eval_time_triggers()
    await _settle(engine)

    assert _next(engine) is None


@pytest.mark.asyncio
async def test_a_deadline_after_another_trigger_is_next_due_only_once_its_anchor_fires():
    engine, clock, _ = _engine()
    engine.register({"triggers": [_time("ask", at="PT1H"), _time("chase", after="ask", offset="P1D")]})
    assert _next(engine) == "2026-03-02T01:00:00Z"

    clock.advance("2026-03-02T01:00:00Z")
    engine._eval_time_triggers()
    await _settle(engine)

    assert _next(engine) == "2026-03-03T01:00:00Z"


@pytest.mark.asyncio
async def test_nothing_is_next_once_every_time_trigger_fired_and_while_the_clock_is_unarmed():
    engine, clock, _ = _engine()
    engine.register({"triggers": [_time("once", at="PT1H")]})
    clock.advance("2026-03-02T02:00:00Z")
    engine._eval_time_triggers()
    await _settle(engine)
    assert (_row(engine, "once")["status"], _next(engine)) == ("fired", None)

    engine.register({"triggers": [_time("later", at="PT5H")]})
    clock.clear()

    assert _next(engine) is None


# ---- resumes_at -----------------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_delayed_action_resumes_at_its_mark_plus_the_delay_and_is_the_next_event():
    engine, clock, _ = _engine()
    engine.register({"triggers": [{"id": "reply", "when": {"type": "time", "at": "2026-03-03T09:00:00Z"},
                                   "actions": [{**_perm(), "delay": "PT2H"}]}]})
    clock.advance("2026-03-03T09:00:00Z")
    assert engine.clock_advanced() == ["reply"]

    await _until(lambda: _row(engine, "reply")["waiting"])

    assert _row(engine, "reply")["resumes_at"] == "2026-03-03T11:00:00Z"
    assert _next(engine) == "2026-03-03T11:00:00Z"
    await engine.cancel_scheduled()


@pytest.mark.asyncio
async def test_a_recurring_trigger_waiting_out_a_delay_is_next_due_when_its_action_resumes_not_at_its_next_mark():
    engine, clock, _ = _engine()
    engine.register({"triggers": [{"id": "beat", "when": {"type": "time", "every": "PT1H"},
                                   "actions": [{**_perm(), "delay": "PT2H"}]}]})
    clock.advance("2026-03-02T01:00:00Z")
    engine.clock_advanced()
    await _until(lambda: _row(engine, "beat")["waiting"])

    assert (_row(engine, "beat")["next_mark"], _next(engine)) == ("2026-03-02T02:00:00Z", "2026-03-02T03:00:00Z")
    await engine.cancel_scheduled()


@pytest.mark.asyncio
async def test_an_action_delay_resumes_from_the_provoking_call():
    engine, clock, now = _engine(rate=1.0)
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "ack", "when": {"type": "action", "tool": "send_email"}, "actions": [{**_perm(), "delay": "PT30M"}]}]})
    now.tick(60)
    engine.on_tool_call("default", "send_email", {}, _result())

    await _until(lambda: _row(engine, "ack")["waiting"])

    assert _row(engine, "ack")["resumes_at"] == _next(engine) == "2026-03-02T00:31:00Z"
    await engine.cancel_scheduled()


@pytest.mark.asyncio
async def test_an_action_waiting_for_its_executor_s_hours_resumes_when_they_open(monkeypatch):
    async def reply(self, executor, instruction, context_id):
        return "done"

    monkeypatch.setattr(TriggerEngine, "_call_executor", reply)
    engine, _, _ = _engine()
    engine.register({"watch_roles": ["default"],
                     "executors": {"bob": {"a2a_url": "http://bob", "role": "bob",
                                           "hours": {"start": "09:00", "end": "17:00"}}},
                     "triggers": [{"id": "ask-bob", "when": {"type": "action", "tool": "send_email"},
                                   "actions": [{"type": "nl", "executor": "bob", "instruction": "reply"}]}]})
    engine.on_tool_call("default", "send_email", {}, _result())

    await _until(lambda: _row(engine, "ask-bob")["waiting"])

    assert _row(engine, "ask-bob")["resumes_at"] == _next(engine) == "2026-03-02T09:00:00Z"
    await engine.cancel_scheduled()


@pytest.mark.asyncio
async def test_an_advance_past_the_close_of_an_executors_hours_waits_for_the_next_opening(monkeypatch):
    calls = []

    async def reply(self, executor, instruction, context_id):
        calls.append(_iso(self._gw._clock.now()))
        return "done"

    monkeypatch.setattr(TriggerEngine, "_call_executor", reply)
    engine, clock, _ = _engine()
    clock.advance("2026-03-02T08:00:00Z")
    engine.register({"watch_roles": ["default"],
                     "executors": {"bob": {"a2a_url": "http://bob", "role": "bob",
                                           "hours": {"start": "09:00", "end": "17:00"}}},
                     "triggers": [{"id": "ask-bob", "when": {"type": "action", "tool": "send_email"},
                                   "actions": [{"type": "nl", "executor": "bob", "instruction": "reply"}]}]})
    engine.on_tool_call("default", "send_email", {}, _result())
    await _until(lambda: _row(engine, "ask-bob")["waiting"])

    clock.advance("2026-03-02T18:00:00Z")
    assert engine.clock_advanced() == []
    await _until(lambda: _row(engine, "ask-bob")["resumes_at"] == "2026-03-03T09:00:00Z")
    assert calls == [] and _next(engine) == "2026-03-03T09:00:00Z"

    clock.advance("2026-03-03T09:00:00Z")
    assert engine.clock_advanced() == ["ask-bob"]
    await _until(lambda: _row(engine, "ask-bob")["status"] == "fired" and calls)
    assert calls == ["2026-03-03T09:00:00Z"]
    assert [e["until"] for e in engine.state()["events"] if e["kind"] == "deferred"] == [
        "2026-03-02T09:00:00Z", "2026-03-03T09:00:00Z"]


# ---- what an advance makes due --------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_advance_names_the_time_triggers_it_reaches_and_they_are_firing_when_it_returns():
    engine, clock, _ = _engine()
    engine.register({"triggers": [_time("tue", at="2026-03-03T09:00:00Z"), _time("wed", at="2026-03-04T09:00:00Z")]})
    clock.advance("2026-03-03T09:00:00Z")

    assert engine.clock_advanced() == ["tue"]
    assert (_row(engine, "tue")["status"], _row(engine, "wed")["status"]) == ("firing", "armed")
    await _settle(engine)


@pytest.mark.asyncio
async def test_an_advance_past_a_waiting_action_names_it_and_wakes_it_at_once():
    engine, clock, _ = _engine()
    engine.register({"triggers": [{"id": "reply", "when": {"type": "time", "at": "PT1M"},
                                   "actions": [{**_perm(), "delay": "PT2H"}]}]})
    clock.advance("2026-03-02T00:01:00Z")
    engine.clock_advanced()
    await _until(lambda: _row(engine, "reply")["waiting"])

    clock.advance("2026-03-02T02:01:00Z")
    started = time.monotonic()
    assert engine.clock_advanced() == ["reply"]
    await _until(lambda: _row(engine, "reply")["status"] == "fired")

    assert time.monotonic() - started < 1.0
    assert _next(engine) is None


@pytest.fixture
def gateway(tmp_path, monkeypatch) -> Gateway:
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    gw = Gateway(host="127.0.0.1", port=0, server_name="AgentEnvGateway", internal_mcp_servers=[])
    gw._clock = Clock(now_fn=FakeNow())
    gw._tool_server_urls = {"send_email": "http://svc/mcp"}

    async def call_child_tool(server_url, tool_name, arguments, *, caller=None):
        return _result()

    gw._call_child_tool = call_child_tool
    yield gw
    gw._close_trajectory_file()


async def _request(gw: Gateway, method: str, path: str, body: dict, role: str | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=gw._mcp.streamable_http_app())
    headers = {"AgentEnv-Role": role} if role else {}
    async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
        return await client.request(method, path, json=body, headers=headers)


@pytest.mark.asyncio
async def test_the_advance_route_returns_what_came_due(gateway):
    gateway._clock.set_time(MONDAY, rate=0)
    gateway._trigger_engine.register({"triggers": [_time("tue", at="2026-03-03T09:00:00Z")]})

    resp = await _request(gateway, "PUT", "/clock/advance", {"virtual_time": "2026-03-03T09:30:00Z"})

    assert resp.status_code == 200
    assert (resp.json()["virtual_time"], resp.json()["due"]) == ("2026-03-03T09:30:00Z", ["tue"])


# ---- the /step source tag -------------------------------------------------------------------------------------------

def _trajectory(gw: Gateway) -> list[dict]:
    with open(gw_mod.GATEWAY_TRAJECTORY_FILE) as f:
        return [json.loads(line) for line in f]


@pytest.mark.asyncio
async def test_a_step_call_a_hook_makes_is_recorded_as_the_hooks(gateway):
    body = {"action": "call_tool", "tool_name": "send_email", "arguments": {"to": "a@example.com"},
            "source": "post_turn_hook", "hook_artifact_id": "next-env-event"}

    resp = await _request(gateway, "POST", "/step", body, role="assistant")

    assert resp.status_code == 200
    assert [(e["event_type"], e["source"], e["hook_artifact_id"]) for e in _trajectory(gateway)] == [
        ("tool_call", "post_turn_hook", "next-env-event"), ("tool_call_result", "post_turn_hook", "next-env-event")]
    assert _trajectory(gateway)[0]["role"] == "assistant"


@pytest.mark.asyncio
async def test_a_step_call_naming_no_source_is_still_step_api(gateway):
    resp = await _request(gateway, "POST", "/step", {"action": "call_tool", "tool_name": "send_email"})

    assert resp.status_code == 200
    assert {e["source"] for e in _trajectory(gateway)} == {"step_api"}
    assert not any("hook_artifact_id" in e for e in _trajectory(gateway))


@pytest.mark.asyncio
@pytest.mark.parametrize("tag", [{"source": "trigger_engine", "hook_artifact_id": "h"}, {"source": "post_turn_hook"},
                                 {"source": "post_turn_hook", "hook_artifact_id": ""}])
async def test_a_step_call_naming_another_source_or_no_hook_is_refused(gateway, tag):
    resp = await _request(gateway, "POST", "/step", {"action": "call_tool", "tool_name": "send_email", **tag})

    assert resp.status_code == 400 and "'source' may only be 'post_turn_hook'" in resp.json()["error"]
    assert _trajectory(gateway) == []


def test_the_card_declares_next_event_at_and_the_step_source_tag():
    extensions = {e["uri"]: e for e in GATEWAY_EXTENSIONS}
    assert TRIGGER_NEXT_EVENT_AT in extensions[EXT_TRIGGERS_URI]["params"]["features"]
    assert extensions[EXT_STEP_URI]["params"]["features"] == [STEP_SOURCE]
