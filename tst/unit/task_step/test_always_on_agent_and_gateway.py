"""What the step sets on the agent and the gateways, and what it leaves: the message_user rules reach the agent even when
its card won't take a system prompt; the step's context_id is the agent's session, as prompt_agent's is; and the
gateway settings the step changed for its run are as they were once it ends, for the task's later steps."""

from __future__ import annotations

import logging

import pytest

from agent_env.task_step.task_steps import always_on as ao
from tst.unit.task_step.always_on_harness import SOLVER, Rig

pytestmark = pytest.mark.usefixtures("local_stores")

RULES = f"{ao.MESSAGE_RULES} {ao.QUIET_RULE}"
PROMPT = "You are Linda's assistant."


@pytest.mark.asyncio
async def test_the_rules_go_on_the_first_wake_when_the_agent_won_t_take_a_system_prompt(monkeypatch, tmp_path, caplog):
    rig = Rig(monkeypatch, tmp_path)  # its card takes no config
    await rig.arm()
    caplog.set_level("WARNING", logger=ao.logger.name)
    ctx = await rig.run(rig.step(system_prompt=PROMPT))

    assert rig.sent[0][0] == RULES and rig.parts[0][0]["metadata"] == {"agentenv": {"source": "rules"}}
    assert rig.record(ctx)["settings"]["rules_via"] == "first_wake"
    assert any("did not accept system_prompt" in r.getMessage() for r in caplog.records
               if r.levelno == logging.WARNING)


@pytest.mark.asyncio
async def test_the_rules_go_in_the_system_prompt_when_the_agent_takes_it(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, takes_config=True)
    await rig.arm()
    ctx = await rig.run(rig.step(system_prompt=PROMPT))

    [(url, fields)] = [(u, f) for u, f in rig.configs if u.startswith(SOLVER)]
    assert fields["system_prompt"] == f"{PROMPT}\n\n{RULES}"
    assert RULES not in rig.sent[0]
    assert rig.record(ctx)["settings"]["rules_via"] == "system_prompt"


@pytest.mark.asyncio
async def test_the_step_s_context_id_is_the_agent_s_session(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        rig.message("blue or green?") if n == 1 else rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, answers=[{"outcome": "reply", "message": "blue"}])
    await rig.arm()
    ctx = await rig.run(rig.step(context_id="cdmx-weekend"))

    assert rig.solver_contexts == ["cdmx-weekend", "cdmx-weekend"]
    assert ctx.prompt_responses[0].a2a_context_id == "cdmx-weekend"
    assert ao.AlwaysOnTaskStep.from_dict(rig.step(context_id="cdmx-weekend").to_dict()).context_id == "cdmx-weekend"


@pytest.mark.asyncio
async def test_without_a_context_id_each_run_has_a_session_of_its_own(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    ctx = await rig.run(rig.step())
    [session] = set(rig.solver_contexts)
    assert session and ctx.prompt_responses[0].a2a_context_id == session


@pytest.mark.asyncio
async def test_the_gateway_settings_the_step_changed_are_as_they_were_after_the_run(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path, rate=60)
    await rig.arm()
    gw = rig.gw
    gw._user_chat.configure({"roles": ["planner"], "user_name": "Tess"})
    gw._notifications.configure({"roles": ["planner"], "min_interval": "PT2H", "max_active": 4})
    await rig.run(rig.step(end_time="P2D"))

    during = rig.during[0]
    assert during["user_chat_roles"] == ("default",) and during["stay_quiet"]
    assert during["trigger_settings"] == {"concurrent_fires": True}
    assert during["check_roles"] == ("planner", "default") and during["clock_rate"] == 60
    assert (gw._user_chat.roles, gw._user_chat.user_name, gw._user_chat.stay_quiet) == (("planner",), "Tess", False)
    assert gw._trigger_engine.state()["settings"] == {"concurrent_fires": False}
    assert (gw._notifications.roles, gw._notifications.min_interval, gw._notifications.max_active) == (
        ("planner",), "PT2H", 4)
    assert gw._clock.rate == 60


@pytest.mark.asyncio
async def test_a_gateway_the_task_left_alone_is_back_to_its_defaults(monkeypatch, tmp_path):
    rig = Rig(monkeypatch, tmp_path)
    await rig.arm()
    await rig.run(rig.step(end_time="P2D"))

    gw = rig.gw
    assert (gw._user_chat.roles, gw._user_chat.stay_quiet, gw._notifications.roles) == ((), False, ())
    assert gw._trigger_engine.state()["settings"] == {"concurrent_fires": False}
    assert not {"message_user", "stay_quiet", "schedule_check"} & await rig.tools()
