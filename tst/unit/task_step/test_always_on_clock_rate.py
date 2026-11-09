"""The task's clock runs at its rate only while the agent works (FD-5366): between turns the step freezes it, so a slow
contact reply or user sim read doesn't move the task's time and alerts arrive when they happen; at the end the task's
rate is back. Every other always_on test runs a frozen clock (rate 0); these run one at 60x."""

from __future__ import annotations

import asyncio
import time

import pytest

from agent_env.task_step.task_steps import always_on as ao
from agent_env.task_step.task_steps.always_on import _utc
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

RATE = 60.0  # a real second is a virtual minute


def _estimate(rig, n):
    rig.alert("Live Oak: estimate attached, $2,400", channel="Gmail · Live Oak Arbor")


@pytest.mark.asyncio
async def test_a_slow_contact_reply_between_turns_doesn_t_move_the_clock(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, rate=RATE)
    await rig.arm()
    liveoak = rig.contact("liveoak", seconds=1.5, reply=_estimate, timeout_seconds=10)  # 90 virtual s at 60x
    rig.triggers({"id": "estimate", "when": {"type": "time", "at": "2026-11-09T15:00:00Z"},
                  "actions": [{"type": "nl", "executor": "liveoak", "instruction": "send Maya your estimate"}]},
                 executors={"liveoak": liveoak})
    ctx = await rig.run(rig.step(settle_timeout_seconds=0.2))

    [_, alerted] = rig.events(ctx, "wake")
    assert "[Mon Nov 9, 3:00 PM UTC · Gmail · Live Oak Arbor]" in alerted["parts_text"]
    assert alerted["virtual_time"].startswith("2026-11-09T15:00:00")
    assert rig.record(ctx)["settings"]["clock"] == {"envs": [{"env_id": "world", "rate": RATE,
                                                              "frozen_between_turns": True}]}


@pytest.mark.asyncio
async def test_the_clock_runs_at_the_task_s_rate_during_a_turn_and_the_rate_is_back_at_the_end(monkeypatch, tmp_path):
    rates: list[float] = []
    moved: list[float] = []

    def agent(n, texts, rig):
        rates.append(rig.gw._clock.rate)
        before = rig.gw._clock.elapsed()
        time.sleep(0.5)  # the agent at work: half a real second, half a virtual minute
        moved.append(rig.gw._clock.elapsed() - before)
        rig.message("found two houses") if n == 1 else rig.quiet()
        return ""

    class SlowReader:
        @staticmethod
        def setup(version, person):
            return {"system_prompt": "you are Linda", "output_format": None}

        @staticmethod
        async def answer(version, person, history, read, ask):
            rates.append(rig.gw._clock.rate)
            await asyncio.sleep(0.5)  # she takes her time reading
            return {"outcome": "reply", "message": "the blue one"}

    rig = Rig(monkeypatch, tmp_path, rate=RATE, agent=agent, plugin=SlowReader)
    await rig.arm()
    ctx = await rig.run(rig.step())

    assert rates == [RATE, 0.0, RATE]  # turn, read, turn
    assert all(25 <= m <= 40 for m in moved)
    [read] = rig.events(ctx, "user_sim_read")
    assert read["deliver_at"] == read["virtual_time"]  # nothing moved while she read
    assert rig.gw._clock.rate == RATE


def _crew_texted(rig, n):
    rig.alert(f"Sam: reply {n}", channel="Messages · noor 30")


CREW_DM = {"id": "crew-dm", "when": {"type": "action", "tool": "messaging_send_message", "repeat": True},
           "actions": [{"type": "nl", "executor": "crew", "instruction": "answer Linda's newest text",
                        "delay": {"dist": "exp", "mean": "PT40M", "seed": 30}}]}


@pytest.mark.asyncio
async def test_a_contact_s_delayed_reply_goes_on_when_the_step_jumps_the_frozen_clock_to_it(monkeypatch, tmp_path):
    """The reply's delay starts during the agent's turn, on the running clock; between turns the step freezes the
    clock and jumps it to the reply's time. That must be enough: the reply used to wait out settle_timeout_seconds,
    then go only at the next jump, hours of the task's time late."""
    def texts_a_friend(n, texts, rig):
        if n == 1:
            rig.provoke("messaging_send_message", {"user_id": "+13475550167"})
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, rate=RATE, agent=texts_a_friend)
    turn = rig._send_and_wait

    async def the_turn_goes_on_after_the_text(url, parts, **kw):
        out = await turn(url, parts, **kw)
        await asyncio.sleep(0.05)  # the reply's delay starts while the clock still runs, as in a real turn
        return out

    monkeypatch.setattr(ao, "send_and_wait", the_turn_goes_on_after_the_text)
    await rig.arm()
    rig.app_tools("messaging_send_message")
    rig.triggers(CREW_DM, executors={"crew": rig.contact("crew", seconds=0, reply=_crew_texted)})
    try:
        ctx = await rig.run(rig.step(settle_timeout_seconds=1))
    finally:
        await rig.close()

    assert rig.events(ctx, "settle_timeout") == []
    [jump] = [j for j in rig.events(ctx, "clock_jump") if j["to"].startswith("2026-11-09T14:3")]
    assert jump["settled"]
    [call] = rig.contact_calls
    assert _utc(call["at"]) == _utc(jump["to"])  # the reply went at the time the step jumped to
    [_, replied] = rig.events(ctx, "wake")
    assert "Sam: reply 1" in replied["parts_text"] and _utc(replied["virtual_time"]) == _utc(jump["to"])
