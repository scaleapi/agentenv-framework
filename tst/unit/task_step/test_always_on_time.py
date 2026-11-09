"""The always_on step gives the agent the task's time through what it sends, not a date line: each alert and each of
the user's lines is labelled with its time, and get_time reads the task's clock, even when the task hides every tool.
No wake opens with a line of its own about the date."""

from __future__ import annotations

import pytest

from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

PROMPT = "You are Linda's assistant."


@pytest.mark.asyncio
async def test_wakes_carry_the_task_s_time_on_each_line_and_no_date_line(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, takes_config=True)  # the rules go in its system prompt
    await rig.arm()
    rig.triggers(rig.notify_at("sam", "2026-11-10T14:00:00Z", "Sam: shh", channel="noor 30"))
    ctx = await rig.run(rig.step(system_prompt=PROMPT))

    assert [texts[0] for texts in rig.sent] == ["[Mon Nov 9, 2:00 PM UTC] Linda: sort the cdmx weekend",
                                                "[Tue Nov 10, 2:00 PM UTC · noor 30] Sam: shh"]
    assert rig.parts[0][0]["metadata"]["agentenv"]["opening"] is True
    sources = {p.get("metadata", {}).get("agentenv", {}).get("source") for parts in rig.parts for p in parts}
    assert "time" not in sources
    assert not any("computer's clock" in t for texts in rig.sent for t in texts)
    assert rig.events(ctx, "wake")[1]["parts_text"].startswith("[Tue Nov 10, 2:00 PM UTC · noor 30]")


@pytest.mark.asyncio
async def test_get_time_stays_with_the_agent_when_the_task_hides_every_tool(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    async with rig.http() as client:
        await client.post("/tools/disable", json={"role": "default", "tools": ["*"]})
    assert "get_time" not in await rig.tools()
    await rig.run(rig.step())
    assert {"get_time", "message_user", "stay_quiet"} <= rig.during[0]["tools"]
    assert not {"message_user", "stay_quiet"} & await rig.tools()  # the user chat is off again after the run
