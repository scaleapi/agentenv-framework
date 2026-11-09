"""The task's clock runs at its rate only while the agent works (FD-5366): between turns the step freezes it, so a slow
contact reply or user sim read doesn't move the task's time and alerts arrive when they happen; at the end the task's
rate is back. Every other always_on test runs a frozen clock (rate 0); these run one at 60x."""

from __future__ import annotations

import asyncio
import time

import pytest

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
