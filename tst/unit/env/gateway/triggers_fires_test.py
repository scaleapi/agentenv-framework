"""What the trigger engine's state says of each fire in flight (running, waits, executor_calls), and its opt-in
settings (config): a repeat trigger's fires side by side, and a time limit for executors that set none."""
from __future__ import annotations

import asyncio
import random
from datetime import timedelta
from types import SimpleNamespace

import httpx
import pytest

from agent_env.env.gateway import gateway as gw_mod
from agent_env.env.gateway.clock import Clock, _iso, _parse_rfc3339
from agent_env.env.gateway.constants import EXT_TRIGGERS_URI, GATEWAY_EXTENSIONS
from agent_env.env.gateway.gateway import Gateway
from agent_env.env.gateway.triggers import TriggerEngine, TriggerError
from tst.unit.env.gateway.next_event_test import MONDAY, _engine, _next, _row, _until
from tst.unit.env.gateway.triggers_time_test import _FakeGateway, _perm, _result


def _reply_trigger(tid: str = "reply", *, delay=None, executor: dict | None = None) -> dict:
    action = {"type": "nl", "executor": "bob", "instruction": "answer Linda", **({"delay": delay} if delay else {})}
    return {"watch_roles": ["default"],
            "executors": {"bob": {"a2a_url": "http://bob", "role": "bob", **(executor or {})}},
            "triggers": [{"id": tid, "when": {"type": "action", "tool": "send_text", "repeat": True},
                          "actions": [action]}]}


class _Executor:
    """A stand-in executor: each call waits until `release()` lets it answer; records each call and how many ran at
    once."""

    def __init__(self, monkeypatch) -> None:
        self.calls: list[dict] = []
        self.running = self.most_at_once = 0
        self.go = asyncio.Event()
        executor = self

        async def call(engine, ex, instruction, context_id):
            executor.calls.append({"timeout_seconds": ex["timeout_seconds"], "context_id": context_id,
                                   "at": engine._gw._clock.now()})
            executor.running += 1
            executor.most_at_once = max(executor.most_at_once, executor.running)
            try:
                await executor.go.wait()
            finally:
                executor.running -= 1
            return "replied"

        monkeypatch.setattr(TriggerEngine, "_call_executor", call)

    def release(self) -> None:
        self.go.set()


# ---- executor calls in state --------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_state_lists_an_executor_call_in_progress_with_the_time_left_of_its_limit(monkeypatch):
    bob = _Executor(monkeypatch)
    engine, _, _ = _engine()
    engine.register(_reply_trigger(executor={"timeout_seconds": 300}))
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: bob.calls)

    row = _row(engine, "reply")
    [call] = row["executor_calls"]
    assert (call["executor"], call["action_index"], row["running"]) == ("bob", 0, 1)
    assert 295 <= call["seconds_left"] <= 300

    bob.release()
    await _until(lambda: _row(engine, "reply")["status"] == "armed")
    assert _row(engine, "reply")["executor_calls"] == [] and _row(engine, "reply")["running"] == 0


# ---- concurrent_fires -----------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_by_default_a_second_call_queues_behind_the_fire_running(monkeypatch):
    bob = _Executor(monkeypatch)
    engine, _, _ = _engine()
    engine.register(_reply_trigger())
    engine.on_tool_call("default", "send_text", {}, _result())
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: bob.calls)

    row = _row(engine, "reply")
    assert (row["status"], row["pending"], row["running"], len(bob.calls)) == ("queued", 1, 1, 1)
    bob.release()
    await _until(lambda: _row(engine, "reply")["status"] == "armed")
    assert len(bob.calls) == 2 and bob.most_at_once == 1


@pytest.mark.asyncio
async def test_side_by_side_each_fire_waits_out_its_own_delay(monkeypatch):
    """A quick reply doesn't wait behind a slow one: each fire's delay runs from its own provoking call."""
    bob = _Executor(monkeypatch)
    bob.release()
    engine, clock, _ = _engine()
    engine.config({"concurrent_fires": True})
    engine.register({**_reply_trigger(delay="PT2H")})
    engine.on_tool_call("default", "send_text", {}, _result())
    clock.advance("2026-03-02T01:00:00Z")
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: len(_row(engine, "reply")["waits"]) == 2)

    row = _row(engine, "reply")
    assert (row["status"], row["running"], row["pending"]) == ("firing", 2, 0)
    assert [w["resumes_at"] for w in row["waits"]] == ["2026-03-02T02:00:00Z", "2026-03-02T03:00:00Z"]
    assert row["resumes_at"] == _next(engine) == "2026-03-02T02:00:00Z"

    clock.advance("2026-03-02T02:00:00Z")
    assert engine.clock_advanced() == ["reply"]
    await _until(lambda: len(bob.calls) == 1 and len(_row(engine, "reply")["waits"]) == 1)
    assert _next(engine) == "2026-03-02T03:00:00Z" and _row(engine, "reply")["running"] == 1

    clock.advance("2026-03-02T03:00:00Z")
    engine.clock_advanced()
    await _until(lambda: _row(engine, "reply")["status"] == "armed")
    assert [c["at"].isoformat() for c in bob.calls] == ["2026-03-02T02:00:00+00:00", "2026-03-02T03:00:00+00:00"]
    assert _row(engine, "reply")["fire_count"] == 2 and _next(engine) is None


@pytest.mark.asyncio
async def test_side_by_side_fires_of_one_trigger_take_turns_with_its_executor_conversation(monkeypatch):
    bob = _Executor(monkeypatch)
    engine, _, _ = _engine()
    engine.config({"concurrent_fires": True})
    engine.register(_reply_trigger())
    for _ in range(3):
        engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: bob.calls)
    await asyncio.sleep(0.05)

    assert _row(engine, "reply")["running"] == 3 and len(bob.calls) == 1
    bob.release()
    await _until(lambda: _row(engine, "reply")["status"] == "armed")
    assert len(bob.calls) == 3 and bob.most_at_once == 1
    assert {c["context_id"] for c in bob.calls} == {"trigger-reply"}


@pytest.mark.asyncio
async def test_two_triggers_reach_their_executors_at_once(monkeypatch):
    bob = _Executor(monkeypatch)
    engine, _, _ = _engine()
    engine.register({"watch_roles": ["default"], "executors": {"bob": {"a2a_url": "http://bob", "role": "bob"}},
                     "triggers": [{"id": tid, "when": {"type": "action", "tool": "send_text"},
                                   "actions": [{"type": "nl", "executor": "bob", "instruction": "answer"}]}
                                  for tid in ("group", "dm")]})
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: len(bob.calls) == 2)

    assert bob.most_at_once == 2
    bob.release()
    await _until(lambda: all(_row(engine, t)["status"] == "fired" for t in ("group", "dm")))


@pytest.mark.asyncio
async def test_removing_a_trigger_cancels_its_fires_side_by_side(monkeypatch):
    bob = _Executor(monkeypatch)
    engine, _, _ = _engine()
    engine.config({"concurrent_fires": True})
    engine.register(_reply_trigger(delay="PT1H"))
    engine.on_tool_call("default", "send_text", {}, _result())
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: len(_row(engine, "reply")["waits"]) == 2)
    tasks = [engine._triggers["reply"]["task"], *engine._triggers["reply"]["alongside"]]

    engine.remove({"ids": ["reply"]})
    await asyncio.gather(*tasks, return_exceptions=True)

    assert all(t.cancelled() for t in tasks) and bob.calls == []


@pytest.mark.asyncio
async def test_side_by_side_is_only_for_contact_replies_a_tool_action_trigger_still_queues(monkeypatch):
    """A tool action may read a row and write it back; two of its fires at once could lose a write, so it queues."""
    bob = _Executor(monkeypatch)
    engine, _, _ = _engine()
    engine.config({"concurrent_fires": True})
    engine.register({"watch_roles": ["default"],
                     "executors": {"bob": {"a2a_url": "http://bob", "role": "bob"}},
                     "triggers": [{"id": "mirror", "when": {"type": "action", "tool": "send_text", "repeat": True},
                                   "actions": [{"type": "nl", "executor": "bob", "instruction": "answer"}, _perm()]}]})
    engine.on_tool_call("default", "send_text", {}, _result())
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: bob.calls)

    row = _row(engine, "mirror")
    assert (row["status"], row["pending"], row["running"]) == ("queued", 1, 1)
    bob.release()
    await _until(lambda: _row(engine, "mirror")["status"] == "armed")
    assert len(bob.calls) == 2


@pytest.mark.asyncio
async def test_a_reply_that_notifies_the_agent_runs_side_by_side_so_its_due_time_shows_and_it_lands_on_time(
        monkeypatch):
    """A contact's reply followed by a notify, as a host's reply on a booking site: the second fire, with the shorter
    delay, used to queue behind the first, unseen in next_event_at, and only drew its delay once the first ended, 17
    minutes late. Its delay now runs from its own call, its resumes_at shows, and it lands then, the first still
    waiting."""
    bob = _Executor(monkeypatch)
    bob.release()
    notes: list[str] = []
    engine, clock, _ = _engine()
    def notify(channel, text, trigger_id, to):
        notes.append(_iso(clock.now()))
        return {"id": f"n{len(notes)}", "to": to}

    engine._gw._notifications = SimpleNamespace(notify=notify)
    engine.config({"concurrent_fires": True})
    trigger = _reply_trigger(delay={"dist": "exp", "mean": "PT40M", "seed": 30})
    trigger["triggers"][0]["actions"].append({"type": "notify", "channel": "Stayncook", "text": "New message"})
    engine.register(trigger)
    draws = random.Random(30)
    slow, quick = draws.expovariate(1 / 2400), draws.expovariate(1 / 2400)  # 1859 s, then 819 s
    engine.on_tool_call("default", "send_text", {}, _result())
    clock.advance("2026-03-02T00:00:01Z")
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: len(_row(engine, "reply")["waits"]) == 2)

    row = _row(engine, "reply")
    due = _iso(_parse_rfc3339("2026-03-02T00:00:01Z") + timedelta(seconds=quick))
    assert (row["status"], row["pending"], row["running"]) == ("firing", 0, 2)
    assert _next(engine) == due < _iso(_parse_rfc3339(MONDAY) + timedelta(seconds=slow))

    clock.advance(due)
    engine.clock_advanced()
    await _until(lambda: len(notes) == 1)
    assert [_iso(c["at"]) for c in bob.calls] == [due] and notes == [due]
    assert _row(engine, "reply")["running"] == 1


@pytest.mark.asyncio
async def test_each_fire_side_by_side_keeps_its_own_number_and_only_its_own_executor_calls(monkeypatch):
    """Fire 1 waits out its delay while fire 2 starts: fire 1's call still says it is fire 1. Fire 2, waiting for its
    turn on the executor conversation, is not credited with the calls the executor makes for fire 1."""
    seen: list[list[int]] = []
    engine, clock, _ = _engine()
    engine.config({"concurrent_fires": True})
    engine.register(_reply_trigger(delay="PT1H"))

    async def call(eng, ex, instruction, context_id):
        seen.append([c["fire"] for c in eng.caused_by("bob")])
        eng.on_tool_call("bob", "messaging_send", {}, _result())  # the executor's own tool call, made for this fire
        await asyncio.sleep(0.05)
        return "replied"

    monkeypatch.setattr(TriggerEngine, "_call_executor", call)
    engine.on_tool_call("default", "send_text", {}, _result())
    clock.advance("2026-03-02T00:30:00Z")
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: len(_row(engine, "reply")["waits"]) == 2)
    clock.advance("2026-03-02T01:30:00Z")  # both delays end: the two calls take turns
    engine.clock_advanced()
    await _until(lambda: _row(engine, "reply")["status"] == "armed")

    assert seen == [[1], [2]]
    oks = [e["detail"] for e in engine._events if e["kind"] == "action_ok"]
    assert [d["calls"] for d in oks] == [{"messaging_send": 1}, {"messaging_send": 1}]
    assert not any("calls_shared_with" in d for d in oks)


@pytest.mark.asyncio
async def test_a_call_queued_once_side_by_side_is_turned_off_still_fires(monkeypatch):
    """Fires side by side don't drain a queue; turned off meanwhile, a queued call used to be left behind for good."""
    bob = _Executor(monkeypatch)
    bob.release()
    engine, clock, _ = _engine()
    engine.config({"concurrent_fires": True})
    engine.register(_reply_trigger(delay="PT1H"))
    engine.on_tool_call("default", "send_text", {}, _result())              # the first fire, resumes 01:00
    clock.advance("2026-03-02T00:30:00Z")
    engine.on_tool_call("default", "send_text", {}, _result())              # side by side, resumes 01:30
    clock.advance("2026-03-02T01:00:00Z")
    engine.clock_advanced()
    await _until(lambda: len(bob.calls) == 1 and _row(engine, "reply")["running"] == 1)

    engine.config({})                                                        # side by side off
    engine.on_tool_call("default", "send_text", {}, _result())              # queued behind the fire left
    assert (_row(engine, "reply")["status"], _row(engine, "reply")["pending"]) == ("queued", 1)
    clock.advance("2026-03-02T01:30:00Z")
    engine.clock_advanced()
    await _until(lambda: len(bob.calls) == 2 and _row(engine, "reply")["pending"] == 0)
    clock.advance("2026-03-02T02:00:00Z")                                   # its delay runs from its own call
    engine.clock_advanced()
    await _until(lambda: _row(engine, "reply")["status"] == "armed")
    assert len(bob.calls) == 3 and _row(engine, "reply")["fire_count"] == 3


def test_a_fire_that_can_t_be_scheduled_leaves_no_count():
    engine, _, _ = _engine()
    engine.register(_reply_trigger())
    trig = engine._triggers["reply"]

    async def fire(ticket):
        return None

    with pytest.raises(RuntimeError):  # no running event loop
        engine._start_fire(trig, fire)
    assert trig["running"] == 0


# ---- executor time limits -------------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_each_executor_keeps_the_time_limit_it_was_registered_with_and_state_reports_it(monkeypatch):
    bob = _Executor(monkeypatch)
    bob.release()
    engine, _, _ = _engine()
    engine.register({"watch_roles": ["default"],
                     "executors": {"bob": {"a2a_url": "http://bob", "role": "bob"},
                                   "ann": {"a2a_url": "http://ann", "role": "ann", "timeout_seconds": 300}},
                     "triggers": [{"id": tid, "when": {"type": "action", "tool": "send_text"},
                                   "actions": [{"type": "nl", "executor": tid, "instruction": "answer"}]}
                                  for tid in ("bob", "ann")]})
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: len(bob.calls) == 2)

    assert sorted((c["context_id"], c["timeout_seconds"]) for c in bob.calls) == [
        ("trigger-ann", 300), ("trigger-bob", 120)]
    assert engine.state()["config"]["executor_timeouts"] == [{"name": "ann", "timeout_seconds": 300.0},
                                                             {"name": "bob", "timeout_seconds": 120.0}]


def test_state_lists_the_unnamed_executor_s_time_limit_first():
    engine, _, _ = _engine()
    assert engine.state()["config"]["executor_timeouts"] == []
    engine.register({"watch_roles": ["default"], "executor": {"a2a_url": "http://exec", "role": "exec"},
                     "executors": {"ann": {"a2a_url": "http://ann", "role": "ann", "timeout_seconds": 45}},
                     "triggers": []})
    assert engine.state()["config"]["executor_timeouts"] == [{"name": None, "timeout_seconds": 120.0},
                                                             {"name": "ann", "timeout_seconds": 45.0}]


@pytest.mark.parametrize("body", [{"concurrent_fires": "yes"}, {"executor_timeout_seconds": 300},
                                  {"parallel": True}, []])
def test_bad_settings_are_refused(body):
    engine, _, _ = _engine()
    with pytest.raises(TriggerError):
        engine.config(body)


def test_settings_are_off_until_set_and_clear_resets_them():
    engine, _, _ = _engine()
    assert engine.state()["settings"] == {"concurrent_fires": False}
    assert engine.config({"concurrent_fires": True})["concurrent_fires"] is True
    assert engine.state()["settings"] == {"concurrent_fires": True}
    engine.clear()
    assert engine.state()["settings"] == {"concurrent_fires": False}


def test_the_card_offers_config():
    triggers = next(e for e in GATEWAY_EXTENSIONS if e["uri"] == EXT_TRIGGERS_URI)
    assert triggers["params"]["methods"]["config"]["endpoint"] == "/triggers/config"
    assert "fires_in_state" in triggers["params"]["features"]


@pytest.mark.asyncio
async def test_a_fire_counts_as_running_from_its_detection_before_its_task_first_runs(monkeypatch):
    bob = _Executor(monkeypatch)
    bob.release()
    engine, clock, _ = _engine()
    engine.register({"watch_roles": ["default"], "triggers": [
        {"id": "ping", "when": {"type": "time", "at": "PT1M"}, "actions": [_perm()]}]})
    clock.advance("2026-03-02T00:01:00Z")

    assert engine.clock_advanced() == ["ping"]
    assert (_row(engine, "ping")["status"], _row(engine, "ping")["running"]) == ("firing", 1)
    await _until(lambda: _row(engine, "ping")["status"] == "fired")
    assert _row(engine, "ping")["running"] == 0


@pytest.mark.asyncio
async def test_a_fire_cancelled_before_it_began_leaves_no_count(monkeypatch):
    _Executor(monkeypatch)
    engine, _, _ = _engine()
    engine.register(_reply_trigger())
    engine.on_tool_call("default", "send_text", {}, _result())
    trig = engine._triggers["reply"]
    trig["task"].cancel()
    await asyncio.gather(trig["task"], return_exceptions=True)
    await asyncio.sleep(0)

    assert trig["running"] == 0


@pytest.mark.asyncio
async def test_a_fire_advanced_to_its_resumes_at_goes_on_though_its_delay_ends_between_microseconds(monkeypatch):
    """On a frozen clock, nothing else moves time: an advance to the rounded resumes_at must be enough."""
    bob = _Executor(monkeypatch)
    bob.release()
    engine, clock, _ = _engine()
    engine.register(_reply_trigger(delay={"dist": "exp", "mean": "PT40M", "seed": 30}))
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: _row(engine, "reply")["waits"])
    resumes_at = _row(engine, "reply")["resumes_at"]
    assert resumes_at == "2026-03-02T00:30:58.882035Z"  # the draw is 1858.882035... s

    clock.advance(resumes_at)
    engine.clock_advanced()
    await _until(lambda: _row(engine, "reply")["status"] == "armed")
    assert len(bob.calls) == 1


# ---- a clock that runs only during the agent's turns ----------------------------------------------------------------

class _RunningNow:
    """Real time under a running clock: it moves on a microsecond at every read, as time.monotonic does between two."""

    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        self.t += 1e-6
        return self.t


def _running_engine(rate: float) -> tuple[TriggerEngine, Clock]:
    clock = Clock(now_fn=_RunningNow())
    clock.set_time(MONDAY, rate=rate)
    return TriggerEngine(_FakeGateway(clock)), clock


@pytest.mark.parametrize("past", [0, 90], ids=["to its resumes_at", "past it"])
@pytest.mark.asyncio
async def test_a_delay_begun_on_a_running_clock_goes_on_when_the_frozen_clock_is_advanced_to_its_resumes_at(
        monkeypatch, past):
    """The always_on step's jump: a contact's reply starts its delay during the agent's turn, with the clock running
    at the task's rate; between turns the step freezes it (set_rate 0) and advances it to the reply's resumes_at.
    Nothing else moves the clock then, so that must be enough."""
    bob = _Executor(monkeypatch)
    bob.release()
    engine, clock = _running_engine(rate=60)
    engine.register(_reply_trigger(delay={"dist": "exp", "mean": "PT40M", "seed": 30}))
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: _row(engine, "reply")["waits"])
    clock.set_rate(0)
    to = _parse_rfc3339(_row(engine, "reply")["resumes_at"]) + timedelta(seconds=past)

    clock.advance(_iso(to))
    assert engine.clock_advanced() == ["reply"]
    await _until(lambda: _row(engine, "reply")["status"] == "armed")
    assert [c["at"] for c in bob.calls] == [to]


@pytest.mark.asyncio
async def test_a_fire_waiting_on_a_running_clock_for_its_executor_s_hours_goes_on_when_the_frozen_clock_reaches_them(
        monkeypatch):
    bob = _Executor(monkeypatch)
    bob.release()
    engine, clock = _running_engine(rate=60)
    engine.register(_reply_trigger(executor={"hours": {"start": "09:00", "end": "17:00"}}))
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: _row(engine, "reply")["waits"])
    clock.set_rate(0)
    assert _row(engine, "reply")["resumes_at"] == "2026-03-02T09:00:00Z"

    clock.advance("2026-03-02T09:00:00Z")
    assert engine.clock_advanced() == ["reply"]
    await _until(lambda: _row(engine, "reply")["status"] == "armed")
    assert [_iso(c["at"]) for c in bob.calls] == ["2026-03-02T09:00:00Z"]


@pytest.mark.asyncio
async def test_a_rate_change_wakes_a_waiting_fire_to_go_on_when_the_clock_gets_to_its_time(monkeypatch, tmp_path):
    """On a frozen clock a waiting fire looks at it again only every _VIRTUAL_RECHECK_S real seconds; the next turn's
    set_rate wakes it to wait out the rest at the new rate, so it goes on when the clock gets to its time, not up to
    that long after (five virtual minutes at 60x)."""
    monkeypatch.setattr(gw_mod, "GATEWAY_TRAJECTORY_FILE", str(tmp_path / "trajectory.jsonl"))
    gw = Gateway(host="127.0.0.1", port=0, server_name="AgentEnvGateway", internal_mcp_servers=[])
    bob = _Executor(monkeypatch)
    bob.release()
    gw._clock.set_time(MONDAY, rate=0)
    engine = gw._trigger_engine
    engine.register(_reply_trigger(delay="PT1M"))
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: _row(engine, "reply")["waits"])
    try:
        transport = httpx.ASGITransport(app=gw._mcp.streamable_http_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            r = await client.put("/clock/set-rate", json={"virtual_seconds_per_real_second": 3600})
            r.raise_for_status()
        await _until(lambda: bob.calls, seconds=1)  # a virtual minute at 3600x is a sixtieth of a real second
    finally:
        await engine.cancel_scheduled()
        gw._close_trajectory_file()
