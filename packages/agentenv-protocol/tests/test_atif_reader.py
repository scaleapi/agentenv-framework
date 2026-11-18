"""Reading documents with or without the profile: strict loading, slicing into tasks, media paths, stripping the
profile, and telling a live page from a later read."""
import json

import pytest

from agentenv_protocol.atif import AtifError, is_prefix, load, media_paths, strip_profile, tasks
from atif_cases import FIXTURES


def fixture(name):
    return load(FIXTURES / f"{name}.json")


def test_loading_is_strict_json_checked_with_atifs_models():
    with pytest.raises(AtifError, match="JSON"):
        load('{"agent": {"name": "a", "version": "1"}, "steps": [], "x": NaN}')
    with pytest.raises(AtifError, match="UTF-8"):
        load(b"\xff\xfe")
    with pytest.raises(AtifError, match="ATIF"):
        load({"agent": {"name": "a", "version": "1"}, "steps": []})
    assert load(json.dumps(fixture("coding")).encode())["session_id"] == "ctx-7f3a"


def test_a_document_slices_into_its_tasks_by_the_user_steps_profile_ids():
    sliced = tasks(fixture("two_turns"))
    assert [(t.turn, t.task_id, len(t.steps), t.state) for t in sliced] == [
        (0, "task-91c2", 6, "completed"), (1, "task-91c3", 3, "completed")]


@pytest.mark.parametrize("name", ["coding", "two_turns", "multi_agent", "subagent"])
def test_a_document_without_the_profile_slices_into_the_same_tasks(name):
    with_profile = tasks(fixture(name))
    without = tasks(strip_profile(fixture(name)))
    assert [strip_profile({"steps": t.steps}) for t in with_profile] == [{"steps": t.steps} for t in without]
    assert [t.turn for t in without] == list(range(len(without))) and all(t.task_id is None for t in without)


def test_the_stripped_fixture_is_the_coding_fixture_without_its_profile():
    stripped = fixture("coding.stripped")
    assert stripped == strip_profile(fixture("coding"))
    assert "agentenv" not in json.dumps(stripped)
    assert stripped["steps"][1]["tool_calls"][0]["function_name"] == "Bash"


def test_media_paths_lists_every_file_once_including_the_ones_atif_has_no_part_for():
    assert len(media_paths(fixture("computer_use"))) == 3
    odd = media_paths(fixture("odd_parts"))
    assert {p.rsplit(".", 1)[1] for p in odd} == {"pdf", "mp4", "heic"}
    assert media_paths(fixture("coding")) == []
    for name in ("computer_use", "odd_parts"):
        assert all((FIXTURES / path).exists() for path in media_paths(fixture(name)))


def test_a_page_is_a_prefix_only_when_its_steps_lead_every_trajectory():
    final = fixture("subagent")
    page = {**final, "steps": final["steps"][:2],
            "subagent_trajectories": [{**final["subagent_trajectories"][0],
                                       "steps": final["subagent_trajectories"][0]["steps"][:1]}]}
    assert is_prefix(page, final) and is_prefix(None, final)
    assert not is_prefix({**page, "steps": final["steps"][1:3]}, final)
    assert not is_prefix(final, {**final, "subagent_trajectories": []})
