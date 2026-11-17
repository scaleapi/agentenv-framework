"""Without the task's clock: read times and reply delays are task time, so a user sim with a schedule or response times
is refused at the start instead of the loop spinning in real time; a run with neither still works. And a message the
chat env couldn't time (its clock unarmed) is read on the task's clock when another env's is armed."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

DAY = {"schedule": {"wake": "07:30", "sleep": "23:30", "work": ["09:00", "18:00"]},
       "response_times": {"work": 150, "free": 30, "morning": 15}}


def _asks_once(n, texts, rig):
    if n == 1:
        rig.message("blue house or green?")
    else:
        rig.quiet()
    return ""


@pytest.mark.parametrize("user", [DAY, {"schedule": DAY["schedule"]}, {"response_times": DAY["response_times"]}],
                         ids=["both", "schedule", "response_times"])
@pytest.mark.asyncio
async def test_a_schedule_or_response_times_with_no_task_clock_is_refused_at_the_start(monkeypatch, tmp_path, user):
    rig = Rig(monkeypatch, tmp_path, agent=_asks_once, answers=[{"outcome": "reply", "message": "blue"}])
    ctx = rig.context(user=user)
    with pytest.raises(ValueError, match="need the task's clock, and no env's clock is armed"):
        await asyncio.wait_for(rig.run(rig.step(), ctx), 5)  # the old loop spun until real time caught up
    assert rig.sent == [] and rig.gw._user_chat.roles == ()  # refused before the gateway was touched


@pytest.mark.asyncio
async def test_with_no_clock_and_no_schedule_a_run_reads_and_replies_at_once(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_asks_once, answers=[{"outcome": "reply", "message": "blue"}])
    ctx = await asyncio.wait_for(rig.run(rig.step()), 10)

    assert [w["woke_by"] for w in rig.events(ctx, "wake")] == [["user_request"], ["user_reply"]]
    assert rig.record(ctx)["end_reason"] == "finished_nothing_pending"


class _ClockEnv:
    """Another env whose clock is armed (the chat env's isn't): only clock/v1, frozen at ``at``."""

    env_id, env_version, metadata = "calendar", 1, {"deploy_step_id": "deploy-calendar"}

    def __init__(self, at: datetime) -> None:
        self.at = at

    def supports(self, uri, method=None) -> bool:
        return uri == EXT_CLOCK_URI and method in (None, "state", "advance")

    async def invoke(self, uri, method, params=None, **kw):
        if method == "advance":
            self.at = max(self.at, ao._utc(params["virtual_time"]))
        return {"armed": True, "t0": "2027-03-01T15:00:00Z", "virtual_time": ao._iso(self.at),
                "virtual_seconds_per_real_second": 0}


@pytest.mark.asyncio
async def test_a_message_the_chat_env_couldn_t_time_is_read_on_the_task_s_clock(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, agent=_asks_once, answers=[{"outcome": "reply", "message": "blue"}])
    ctx = rig.context(user=DAY)
    task_time = datetime(2027, 3, 1, 15, 0, tzinfo=timezone.utc)  # Mon 10 AM in New York: she reads at once
    assert abs(datetime.now(timezone.utc) - task_time) > timedelta(days=30)
    ctx.deployed_envs.append(_ClockEnv(task_time))
    ctx = await asyncio.wait_for(rig.run(rig.step(), ctx), 10)

    [read] = rig.reads
    assert read["assistant_messages"][0]["at"] is None  # the chat env's clock is unarmed
    assert read["time"] == "2027-03-01T15:00:00Z"
    [scheduled] = rig.events(ctx, "user_read_scheduled")
    assert scheduled["read_at"] == "2027-03-01T15:00:00Z"
