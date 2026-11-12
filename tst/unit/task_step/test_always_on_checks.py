"""The always_on step lets the agent wake itself: scheduled checks are on by default when a run spans more than one
day of the task's time, at least PT3H apart, and the agent is told it can schedule them."""

from __future__ import annotations

import pytest

from agent_env.env.gateway.notifications import NotificationError
from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import Rig

pytestmark = pytest.mark.usefixtures("local_stores")

CHECK_TOOLS = {"schedule_check", "list_scheduled_checks", "cancel_scheduled_check"}


@pytest.mark.asyncio
async def test_a_multi_day_run_gives_the_agent_checks_through_a_deny_all_and_says_so(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    async with rig.http() as client:
        await client.post("/tools/disable", json={"role": "default", "tools": ["*"]})
    ctx = await rig.run(rig.step(end_time="P5D"))

    during = rig.during[0]
    assert during["check_roles"] == ("default",) and during["min_interval"] == "PT3H"
    assert CHECK_TOOLS <= during["tools"]
    assert ao.CHECKS_RULE in rig.sent[0][0]
    assert rig.gw._notifications.roles == () and not CHECK_TOOLS & await rig.tools()  # off again after the run
    assert rig.record(ctx)["settings"]["scheduled_checks"] == {
        "on": True, "envs": [{"env_id": "world", "set_by": "step", "min_interval": "PT3H", "max_active": 10}]}


@pytest.mark.asyncio
async def test_checks_closer_than_the_minimum_gap_are_refused(monkeypatch, tmp_path):
    refused: list[str] = []

    def agent(n, texts, rig):
        if n == 1:
            try:
                rig.gw._notifications.schedule("sync stock", every="PT1H", to="default")
            except NotificationError as e:
                refused.append(str(e))
            rig.gw._notifications.schedule("sync stock", every="PT3H", to="default")
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    await rig.arm()
    await rig.run(rig.step(end_time="P1D"))

    [why] = refused
    assert "at least PT3H" in why


@pytest.mark.asyncio
async def test_a_check_the_agent_schedules_moves_the_clock_and_wakes_it(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n == 1:
            rig.schedule("book La Ola, it opens at 9", at="2026-11-12T14:00:00Z")
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent)
    await rig.arm()
    ctx = await rig.run(rig.step(end_time="P5D"))

    wake = rig.events(ctx, "wake")[1]
    assert wake["virtual_time"] == "2026-11-12T14:00:00Z" and wake["woke_by"] == ["notification:your scheduled check"]
    assert "[Thu Nov 12, 2:00 PM UTC · your scheduled check] book La Ola, it opens at 9" in rig.sent[1]


@pytest.mark.parametrize("end_time,setting,on", [(None, None, False), ("PT6H", None, False), ("P5D", False, False),
                                                 (None, True, True)],
                         ids=["one sitting", "same day", "turned off", "turned on"])
@pytest.mark.asyncio
async def test_otherwise_a_task_decides(monkeypatch, tmp_path, end_time, setting, on):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    settings = {**({"end_time": end_time} if end_time else {}),
                **({"scheduled_checks": setting} if setting is not None else {})}
    ctx = await rig.run(rig.step(**settings))

    assert (rig.during[0]["check_roles"] == ("default",)) is on
    assert rig.record(ctx)["settings"]["scheduled_checks"]["on"] is on
    assert (ao.CHECKS_RULE in rig.sent[0][0]) is on


@pytest.mark.asyncio
async def test_the_line_goes_in_the_system_prompt_when_checks_are_on(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    run = ao._Run(rig.step(system_prompt="You are Linda's assistant.", end_time="P5D"), rig.context())
    run.checks = {"on": True}
    assert run._system_prompt().endswith(ao.CHECKS_RULE)


@pytest.mark.asyncio
async def test_the_task_s_own_check_settings_are_kept(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    rig.gw._notifications.configure({"roles": ["default"], "max_active": 3, "min_interval": "PT1H"})
    ctx = await rig.run(rig.step(end_time="P5D", scheduled_checks=False))

    assert rig.gw._notifications.min_interval == "PT1H" and rig.gw._notifications.max_active == 3
    assert rig.record(ctx)["settings"]["scheduled_checks"]["envs"][0]["set_by"] == "task"


@pytest.mark.asyncio
async def test_other_roles_the_task_gave_checks_keep_them(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    rig.gw._notifications.configure({"roles": ["planner"], "min_interval": "PT2H"})
    await rig.run(rig.step(end_time="P5D"))

    assert rig.during[0]["check_roles"] == ("planner", "default") and rig.during[0]["min_interval"] == "PT2H"
    assert rig.gw._notifications.roles == ("planner",) and rig.gw._notifications.min_interval == "PT2H"


def test_the_settings_survive_a_round_trip():
    step = ao.AlwaysOnTaskStep("week", None, user_sim="linda", prompt_id="p", agent_name="a",
                               scheduled_checks=False, scheduled_check_min_interval="PT6H")
    again = ao.AlwaysOnTaskStep.from_dict(step.to_dict())
    assert (again.scheduled_checks, again.scheduled_check_min_interval) == (False, "PT6H")
    assert ao.AlwaysOnTaskStep.from_dict({**step.to_dict(), "scheduled_checks": None}).scheduled_checks is None
