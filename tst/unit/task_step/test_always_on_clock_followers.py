"""The always-on step's clock writes wait for the computers that follow the gateway's clock: one a computer doesn't
show fails the step naming it, without the write being retried, so the agent is never woken on a computer still
showing the old time."""

from __future__ import annotations

import asyncio

import pytest

from agent_env.env import env as env_mod
from agent_env.env.env import ClockFollowersBehind
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

DAY = {"schedule": {"wake": "07:30", "sleep": "23:30", "work": ["09:00", "18:00"]},
       "response_times": {"work": 150, "free": 30, "morning": 15}}
WAIT_S = 0.3


def _asks_once(n, texts, rig):
    if n == 1:
        rig.message("blue house or green?")
    else:
        rig.quiet()
    return ""


def _changes(rig: Rig) -> int:
    """How many times the gateway's clock has changed: the count in its opaque version."""
    return int(rig.gw._clock.version.rsplit(".", 1)[1])


@pytest.mark.asyncio
async def test_a_clock_write_a_computer_doesnt_show_fails_the_step_naming_it_and_isnt_retried(monkeypatch, tmp_path):
    monkeypatch.setattr(env_mod, "CLOCK_WAIT_FOLLOWERS_S", WAIT_S)
    rig = Rig(monkeypatch, tmp_path, agent=_asks_once, answers=[{"outcome": "reply", "message": "blue"}],
              followers={"laptop": "token-laptop"})
    await rig.arm()
    armed = _changes(rig)
    ctx = rig.context(user=DAY)

    with pytest.raises(ClockFollowersBehind) as behind:
        await asyncio.wait_for(rig.run(rig.step(), ctx), 30)

    assert behind.value.stale == ["laptop"]
    [error] = rig.events(ctx, "error")
    assert "ClockFollowersBehind" in error["error"] and "laptop didn't show it" in error["error"]
    assert _changes(rig) - armed == 1, rig.timeline(ctx)
