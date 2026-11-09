"""The always_on step gives the agent the task's time, not the computer's: each wake opens with the task's date and
time, and says any other date is the computer's clock."""

from __future__ import annotations

import pytest

from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

CLOCK = "Any other date you see is the computer's clock, not the task's."
PROMPT = "You are Linda's assistant."


@pytest.mark.asyncio
async def test_each_wake_opens_with_the_task_s_date_and_time(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    rig.triggers(rig.notify_at("sam", "2026-11-10T14:00:00Z", "Sam: shh", channel="noor 30"))
    ctx = await rig.run(rig.step(system_prompt=PROMPT))

    assert [texts[0] for texts in rig.sent] == [
        f"It's Mon Nov 9, 2026, 2:00 PM UTC (Mon 9:00 AM for Linda, America/New_York). {CLOCK}",
        f"It's Tue Nov 10, 2026, 2:00 PM UTC (Tue 9:00 AM for Linda, America/New_York). {CLOCK}",
    ]
    assert rig.parts[1][0]["metadata"] == {"agentenv": {"source": "time"}}
    assert rig.events(ctx, "wake")[1]["parts_text"].startswith("It's Tue Nov 10, 2026")


@pytest.mark.asyncio
async def test_a_user_in_utc_gets_the_time_once(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    await rig.run(rig.step(system_prompt=PROMPT), rig.context(user={"time_zone": "UTC"}))
    assert rig.sent[0][0] == f"It's Mon Nov 9, 2026, 2:00 PM UTC. {CLOCK}"


@pytest.mark.asyncio
async def test_without_a_task_clock_there_is_no_time_line(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.run(rig.step())
    assert not any(CLOCK in t for t in rig.sent[0])


@pytest.mark.asyncio
async def test_get_time_stays_with_the_agent_when_the_task_hides_every_tool(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    async with rig.http() as client:
        await client.post("/tools/disable", json={"role": "default", "tools": ["*"]})
    assert "get_time" not in await rig.tools()
    await rig.run(rig.step())
    assert {"get_time", "message_user", "stay_quiet"} <= await rig.tools()
