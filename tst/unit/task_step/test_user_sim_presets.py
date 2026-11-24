"""The built-in user sim's presets: chat and always_on keep today's prompt byte for byte, milestone adds how the user
takes a post-turn hook's note, each names the features a judge's rules may depend on, and a preset that doesn't fit
the task's other steps is refused when the task loads."""

from __future__ import annotations

import hashlib

import pytest

from agent_env.task import Task
from agent_env.task_step.task_steps.agent import user_sim_versions as usv
from agent_env.task_step.task_steps.agent.deploy_user_sim_agent import recorded_instructions
from tst.unit.task_step.test_deploy_user_sim_agent import HANNAH, PERSON, PROMPT, _Deploy, _published, _step

PROMPT_AGENT = {"id": "talk", "type": "prompt_agent", "prompt": "sort the weekend", "agent_name": "solver",
                "user_agent_name": "hannah", "max_conversation_turns": 6}
HOOK = [{"file_artifact_id": "milestone-gates", "params": {}}]


def _person(**over) -> dict:
    return {**PERSON, **over}


# ---- the prompt ----------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_an_always_on_preset_keeps_todays_prompt_byte_for_byte(monkeypatch):
    deploy = _Deploy(monkeypatch)
    context = await deploy.run(_step(preset="always_on"))

    assert deploy.fields("system_prompt") == [PROMPT]
    record = context.metadata["user_sims"]["hannah"]
    assert record["prompt_sha256"] == hashlib.sha256(PROMPT.encode()).hexdigest()
    assert record["preset"] == "always_on"


def test_a_chat_preset_writes_the_prompt_a_single_session_writes_today():
    single = _person(done_ends_run=True)
    assert usv.system_prompt({**single, "preset": "chat"}) == usv.system_prompt(single)


def test_a_milestone_preset_tells_the_user_how_to_take_a_hooks_note_and_when_it_is_done():
    chat = usv.system_prompt(_person(done_ends_run=True))
    milestone = usv.system_prompt(_person(done_ends_run=True, preset="milestone"))

    assert usv.MILESTONE_LINES in milestone and usv.MILESTONE_LINES not in chat
    assert usv.DONE_MILESTONE in milestone and usv.DONE_ENDS_RUN not in milestone


@pytest.mark.asyncio
async def test_the_prompt_rebuilt_from_the_record_is_the_one_the_milestone_user_got(monkeypatch):
    deploy = _Deploy(monkeypatch)
    context = await deploy.run(_step(preset="milestone", multi_day=False))
    record = context.metadata["user_sims"]["hannah"]

    assert recorded_instructions(record, {}) == deploy.fields("system_prompt")[0]


# ---- features ------------------------------------------------------------------------------------------------------
def test_features_follow_the_sections_the_persona_fills_in_and_the_preset():
    plain = {"name": "Ada Park", "description": {"who_you_are": ["You run finance."]}, "done_ends_run": True}
    assert usv.features(plain) == frozenset()
    assert usv.features({**plain, "description": {**plain["description"], "already_told": ["The budget is 5k."]},
                         "permissions": {"ask_first": ["sending money"]}}) == {"told_lines", "permission_rules"}
    assert usv.features({**plain, "done_ends_run": False}) == {"phone"}
    assert usv.features({**plain, "preset": "milestone"}) == {"milestones", "hook_notes"}
    assert set(usv.features(_person(done_ends_run=False))) <= set(usv.FEATURES)


@pytest.mark.asyncio
async def test_the_record_carries_the_features_for_the_judge(monkeypatch):
    record = await _published(monkeypatch, _step(preset="milestone", multi_day=False))

    assert record["preset"] == "milestone"
    assert {"milestones", "hook_notes"} <= set(record["features"])
    assert "phone" not in record["features"]


@pytest.mark.asyncio
async def test_a_step_without_a_preset_records_none_and_its_features(monkeypatch):
    record = await _published(monkeypatch, _step())

    assert record["preset"] is None and "phone" in record["features"]


# ---- checks --------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fields, problem", [
    ({"preset": "weekly"}, "is not one of"),
    ({"preset": "milestone", "multi_day": True}, "multi_day must be false"),
    ({"preset": "chat", "multi_day": True}, "multi_day must be false"),
])
def test_a_bad_preset_is_refused(fields, problem):
    with pytest.raises(ValueError, match=problem):
        _step(**fields)


def _task(*steps: dict) -> Task:
    sim = {"id": "user", "type": "deploy_user_sim_agent", "a2a_agent_id": "sim-image", "agent_name": "hannah",
           "model": "sim-model", "opening_prompt": PERSON["opening"]["text"], "persona": HANNAH}
    return Task.from_dict({"id": "t", "version": 1, "steps": [{**sim, **steps[0]}, *steps[1:]]})


def test_a_milestone_user_with_a_hooked_conversation_loads():
    task = _task({"preset": "milestone", "multi_day": False}, {**PROMPT_AGENT, "post_turn_hooks": HOOK})
    assert [s.type for s in task.steps] == ["deploy_user_sim_agent", "prompt_agent"]


@pytest.mark.parametrize("steps, problem", [
    (({"preset": "milestone", "multi_day": False}, PROMPT_AGENT), "has no post_turn_hooks"),
    (({"preset": "milestone", "multi_day": False},), "no prompt_agent step talks to it"),
    (({"preset": "chat", "multi_day": False},
      {"id": "week", "type": "always_on", "user_sim": "hannah", "prompt_id": "ask", "agent_name": "solver"}),
     "one conversation, but always_on"),
])
def test_a_preset_that_doesnt_fit_the_task_is_refused_when_it_loads(steps, problem):
    with pytest.raises(ValueError, match=problem):
        _task(*steps)
