"""The user_sim step and the built-in user sim plug-in: the settings the step records, and the ``ours`` version's
prompt and read."""

from __future__ import annotations

import pytest

from agent_env.task_step.task_steps import user_sim_versions
from agent_env.task_step.task_steps.user_sim import UserSimTaskStep


def test_the_user_sim_step_defaults_to_ours_and_has_no_apps_setting():
    step = UserSimTaskStep.from_dict({"id": "u", "type": "user_sim", "name": "linda", "model": "m",
                                         "apps": ["Gmail"], "time_zone": "America/New_York",
                                         "email": "linda.murphy@example.test"})
    assert step.user_version == "ours" and "apps" not in step.to_dict()
    assert UserSimTaskStep.from_dict(step.to_dict()).email == "linda.murphy@example.test"
    with pytest.raises(Exception):
        UserSimTaskStep("u", None, name="linda", model="m", time_zone="Mars/Olympus")


def test_ours_works_from_the_description_alone():
    person = {"name": "Linda Murphy", "first": "Linda", "description": "You're 31, planning Noor's 30th."}
    setup = user_sim_versions.setup("ours", person)
    assert setup["system_prompt"].endswith("WHO YOU ARE\nYou're 31, planning Noor's 30th.\n")
    assert "Any other date you see is the computer's clock" in setup["system_prompt"]
    assert setup["output_format"]["schema"]["properties"]["outcome"]["enum"] == ["reply", "no_reply", "done"]
    own = user_sim_versions.setup("ours", {**person, "versions": {"ours": {"system_prompt": "tagged lines"}}})
    assert own["system_prompt"] == "tagged lines"


def test_ours_renders_one_labelled_read():
    person = {"name": "Linda Murphy", "first": "Linda", "time_zone": "America/New_York"}
    read = {"time": "2026-11-10T15:00:00Z",
            "assistant_messages": [{"at": "2026-11-10T13:00:00Z", "text": "house?"},
                                   {"at": "2026-11-10T14:00:00Z", "text": "and dinner?"}],
            "phone": [{"at": "2026-11-10T12:00:00Z", "channel": "Gmail", "text": "New email from Sam"}],
            "you_sent": [{"at": "2026-11-09T14:00:00Z", "text": "sort the cdmx weekend"}]}
    text = user_sim_versions.render_ours("ours", person, [], read)
    assert text.startswith("It's Tue, Nov 10, 10:00 AM.")
    for label in ("NEW ON YOUR PHONE", "YOU SENT YOUR AI ASSISTANT", "NEW FROM YOUR AI ASSISTANT"):
        assert label in text
    assert text.index("house?") < text.index("and dinner?")


def test_response_times_without_a_schedule_is_refused():
    for schedule in (None, {}):
        with pytest.raises(ValueError, match="response_times needs a schedule"):
            UserSimTaskStep("u", None, name="linda", model="m", schedule=schedule, response_times={"free": 30})
    UserSimTaskStep("u", None, name="linda", model="m", schedule={"wake": "07:30"}, response_times={"free": 30})
