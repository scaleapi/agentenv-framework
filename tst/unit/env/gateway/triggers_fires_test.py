"""What the trigger engine's state says of each fire in flight (running, waits, executor_calls), and its opt-in
settings (config): a repeat trigger's fires side by side, and a time limit for executors that set none."""
from __future__ import annotations

import asyncio

import pytest

from agent_env.env.gateway.constants import EXT_TRIGGERS_URI, GATEWAY_EXTENSIONS
from agent_env.env.gateway.triggers import TriggerEngine, TriggerError
from tst.unit.env.gateway.next_event_test import _engine, _next, _row, _until
from tst.unit.env.gateway.triggers_time_test import _result


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


# ---- executor_timeout_seconds ---------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_configured_limit_applies_only_to_executors_that_set_none(monkeypatch):
    bob = _Executor(monkeypatch)
    bob.release()
    engine, _, _ = _engine()
    engine.config({"executor_timeout_seconds": 300})
    engine.register({"watch_roles": ["default"],
                     "executors": {"bob": {"a2a_url": "http://bob", "role": "bob"},
                                   "ann": {"a2a_url": "http://ann", "role": "ann", "timeout_seconds": 120}},
                     "triggers": [{"id": tid, "when": {"type": "action", "tool": "send_text"},
                                   "actions": [{"type": "nl", "executor": tid, "instruction": "answer"}]}
                                  for tid in ("bob", "ann")]})
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: len(bob.calls) == 2)

    assert sorted((c["context_id"], c["timeout_seconds"]) for c in bob.calls) == [
        ("trigger-ann", 120), ("trigger-bob", 300)]
    assert engine.state()["settings"] == {"concurrent_fires": False, "executor_timeout_seconds": 300.0}


@pytest.mark.asyncio
async def test_without_config_an_executor_that_sets_no_limit_keeps_120_s(monkeypatch):
    bob = _Executor(monkeypatch)
    bob.release()
    engine, _, _ = _engine()
    engine.register(_reply_trigger())
    engine.on_tool_call("default", "send_text", {}, _result())
    await _until(lambda: bob.calls)
    assert bob.calls[0]["timeout_seconds"] == 120


@pytest.mark.parametrize("body", [{"concurrent_fires": "yes"}, {"executor_timeout_seconds": 0},
                                  {"executor_timeout_seconds": True}, {"executor_timeout_seconds": float("inf")},
                                  {"parallel": True}, []])
def test_bad_settings_are_refused(body):
    engine, _, _ = _engine()
    with pytest.raises(TriggerError):
        engine.config(body)


def test_settings_are_off_until_set_and_clear_resets_them():
    engine, _, _ = _engine()
    assert engine.state()["settings"] == {"concurrent_fires": False, "executor_timeout_seconds": None}
    assert engine.config({"concurrent_fires": True, "executor_timeout_seconds": 300})["concurrent_fires"] is True
    engine.config({"concurrent_fires": True})
    assert engine.state()["settings"] == {"concurrent_fires": True, "executor_timeout_seconds": None}
    engine.clear()
    assert engine.state()["settings"] == {"concurrent_fires": False, "executor_timeout_seconds": None}


def test_the_card_offers_config():
    triggers = next(e for e in GATEWAY_EXTENSIONS if e["uri"] == EXT_TRIGGERS_URI)
    assert triggers["params"]["methods"]["config"]["endpoint"] == "/triggers/config"
    assert "fires_in_state" in triggers["params"]["features"]
