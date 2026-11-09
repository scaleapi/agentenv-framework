"""The user_sim step: the settings it records, the checks they get when the task loads, and a run of the always_on
loop with the built-in user sim."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_env.task_step.task_steps import user_sim_versions
from agent_env.task_step.task_steps.user_sim import UserSimTaskStep
from tst.unit.task_step.always_on_harness import Rig

DESCRIPTION = {"who_you_are": ["Linda Murphy, planning Noor's 30th in Mexico City."]}
DATA = Path(__file__).parents[2] / "data" / "user_sim"


def _step(**settings) -> UserSimTaskStep:
    return UserSimTaskStep.from_dict({"id": "u", "type": "user_sim", "name": "linda", "model": "m",
                                      "description": DESCRIPTION, **settings})


def test_the_user_sim_step_defaults_to_ours_and_has_no_apps_setting():
    step = _step(apps=["Gmail"], time_zone="America/New_York", email="linda.murphy@example.test")
    assert step.user_version == "ours" and "apps" not in step.to_dict()
    assert UserSimTaskStep.from_dict(step.to_dict()).email == "linda.murphy@example.test"
    with pytest.raises(Exception):
        _step(time_zone="Mars/Olympus")


def test_the_tested_settings_load():
    person = json.loads((DATA / "hannah_person.json").read_text())
    step = _step(display_name=person["name"], first_name=person["first"], description=person["description"],
                 permissions=person["permissions"], keep_private=person["keep_private"],
                 schedule={"wake": "07:30", "sleep": "23:30", "work_days": ["mon", "tue", "wed", "thu", "fri"],
                           "work": ["09:00", "18:00"],
                           "busy": [{"start": "2026-11-12T09:00", "end": "2026-11-12T18:00"}]},
                 response_times={"work": 150, "free": 30, "morning": 15}, max_replies=40)
    assert step.description == person["description"]


@pytest.mark.parametrize("settings, problem", [
    ({"description": None}, "description must be an object of sections"),
    ({"description": "You're Linda, planning Noor's 30th."}, "description must be an object of sections"),
    ({"description": {"who_you_are": ["Linda."], "events": ["Tue: an email"]}}, "unknown sections ['events']"),
    ({"permissions": {"maybe": ["book it"]}}, "permissions has unknown kinds ['maybe']"),
    ({"keep_private": "her age"}, "keep_private must be a list of non-empty lines"),
    ({"user_version": "today"}, "the built-in user sim has one version, 'ours'"),
])
def test_the_built_in_user_sim_s_settings_are_checked_when_the_task_loads(settings, problem):
    with pytest.raises(ValueError, match="user_sim 'linda'") as e:
        _step(**settings)
    assert problem in str(e.value)


def test_a_step_that_names_its_own_plug_in_writes_its_description_its_own_way():
    step = _step(description="You're Linda.", user_version="today",
                 plugin={"file_artifact_id": "fa-1", "params": {}})
    assert step.description == "You're Linda."


@pytest.mark.parametrize("settings, problem", [
    ({"schedule": ["07:30", "23:30"]}, "schedule must be an object"),
    ({"schedule": {"wake": "7am"}}, "schedule.wake must be a time like 07:30"),
    ({"schedule": {"lunch": "12:00"}}, "schedule has unknown keys ['lunch']"),
    ({"schedule": {"work": ["09:00"]}}, "schedule.work must be [start, end]"),
    ({"schedule": {"work_days": ["monday"]}}, "schedule.work_days must be a list of"),
    ({"schedule": {"busy": [{"start": "Thu 9am", "end": "Thu 6pm"}]}}, "schedule.busy must be a list of {start, end}"),
    ({"response_times": {"work": -5}}, "response_times.work must be a number of minutes"),
    ({"response_times": {"evening": 30}}, "response_times has unknown keys ['evening']"),
    ({"max_replies": 0}, "max_replies must be a whole number"),
])
def test_when_she_reads_and_how_long_she_takes_are_checked_for_any_plug_in(settings, problem):
    for plugin in (None, {"file_artifact_id": "fa-1"}):
        with pytest.raises(ValueError, match="user_sim 'linda'") as e:
            _step(plugin=plugin, **settings)
        assert problem in str(e.value)


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_the_always_on_loop_runs_the_built_in_user_sim(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n == 1:
            rig.message("Blue house or green?")
            return "sent"
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=user_sim_versions,
              user_texts=['{"outcome": "reply", "message": "blue!", "used": "W1"}'])
    await rig.arm()
    ctx = await rig.run(rig.step(end_time="P3D"), rig.context(user={"description": DESCRIPTION}))

    [call] = rig.user_calls
    assert call["parts"][0]["text"] == (
        "It's Mon, Nov 9, 9:00 AM.\n"
        "You last messaged your AI assistant at 9:00 AM.\n"
        "\nYOU SENT YOUR AI ASSISTANT:\n  [9:00 AM] sort the cdmx weekend\n"
        "\nNEW FROM YOUR AI ASSISTANT (this is what you're answering):\n  [9:00 AM]\n    Blue house or green?\n"
        "\nYour AI assistant sent you this. As Linda, decide: reply, don't reply, or say you're done.\n"
        "Answer in JSON: " + user_sim_versions.ANSWER_LINE)
    [read] = rig.events(ctx, "user_sim_read")
    assert (read["outcome"], read["message"], read["used"], read["schema_errors"]) == ("reply", "blue!", "W1", [])
    assert any("] Linda: blue!" in t for texts in rig.sent for t in texts)


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_after_the_final_check_in_the_built_in_user_sim_reads_the_wrap_up_like_any_other_read(monkeypatch,
                                                                                                     tmp_path):
    def agent(n, texts, rig):
        if any("send me the rundown" in t for t in texts):
            rig.message("House booked; La Ola is open Friday.")
            return "sent"
        if n == 1:
            rig.message("Blue house or green?")
            return "sent"
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=user_sim_versions,
              user_texts=['{"outcome": "reply", "message": "blue!", "used": "W1"}',
                          '{"outcome": "done", "message": "perfect, thank you!!", "used": "none"}'])
    await rig.arm()
    ctx = await rig.run(rig.step(end_time="P2D", final="send me the rundown"),
                        rig.context(user={"description": DESCRIPTION}))

    first, last = rig.user_calls
    text = last["parts"][0]["text"]
    assert text.startswith("It's Wed, Nov 11, 9:00 AM.\nYou last messaged your AI assistant at 9:00 AM.\n")
    assert "\nYOU SENT YOUR AI ASSISTANT:\n  [9:00 AM] send me the rundown\n" in text
    assert "(this is what you're answering):\n  [9:00 AM]\n    House booked; La Ola is open Friday.\n" in text
    assert text.endswith("Answer in JSON: " + user_sim_versions.ANSWER_LINE)
    reads = rig.events(ctx, "user_sim_read")
    assert [r["last_read"] for r in reads] == [False, True]
    assert (reads[-1]["outcome"], reads[-1]["message"], reads[-1]["schema_errors"]) == (
        "done", "perfect, thank you!!", [])
    assert rig.record(ctx)["end_reason"] == "end_time"
    assert not any("perfect, thank you" in t for texts in rig.sent for t in texts)


@pytest.mark.usefixtures("local_stores")
@pytest.mark.asyncio
async def test_a_malformed_answer_is_retried_once_in_the_same_user_sim_session(monkeypatch, tmp_path):
    def agent(n, texts, rig):
        if n == 1:
            rig.message("Blue house or green?")
            return "sent"
        rig.quiet()
        return ""

    rig = Rig(monkeypatch, tmp_path, agent=agent, plugin=user_sim_versions,
              user_texts=["blue, obviously", '{"outcome": "reply", "message": "blue!", "used": "W1"}'])
    await rig.arm()
    ctx = await rig.run(rig.step(end_time="P3D"), rig.context(user={"description": DESCRIPTION}))

    first, retry = rig.user_calls
    assert retry["parts"] == [{"kind": "text", "text": user_sim_versions.RETRY}]
    assert first["context_id"] == retry["context_id"]
    [read] = rig.events(ctx, "user_sim_read")
    assert (read["outcome"], read["message"], read["schema_errors"]) == ("reply", "blue!", [])
    assert read["user_sim_turns"] == 2
    assert any("] Linda: blue!" in t for texts in rig.sent for t in texts)
