"""Reading documents with or without the profile: strict loading, slicing into tasks, media paths, stripping the
profile, and telling a live page from a later read."""
import copy
import json

import pytest

from agentenv_protocol.atif import (
    AtifError,
    check,
    errors_in,
    is_prefix,
    load,
    media_paths,
    media_refs,
    strip_profile,
    task_document,
    tasks,
    validate,
)
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


def test_media_refs_name_each_place_a_file_is_named_with_its_details_and_media_type():
    refs = media_refs(fixture("computer_use"))
    assert list(dict.fromkeys(ref.path for ref in refs)) == media_paths(fixture("computer_use"))
    part = next(ref for ref in refs if ref.where.endswith(".source.path"))
    assert part.media_type and part.media_type.startswith("image/") and part.info.get("sha256") == part.path[6:70]
    odd = media_refs(fixture("odd_parts"))
    noted = [ref for ref in odd if ".extra.agentenv.media[" in ref.where]
    assert noted and all(ref.media_type == ref.info.get("media_type") for ref in noted)
    nested = {**fixture("coding"), "subagent_trajectories": [fixture("computer_use")]}
    assert all(ref.where.startswith("subagent_trajectories[0].steps[") for ref in media_refs(nested))
    assert len(media_refs(nested)) == len(refs)


def test_a_task_cut_out_of_a_document_is_a_document_of_its_own():
    document = fixture("two_turns")
    first, second = tasks(document)
    cut = task_document(document, [second.task_id])
    assert errors_in(validate(cut)) == []
    assert [step["step_id"] for step in cut["steps"]] == [1, 2, 3]
    assert [{**step, "step_id": 0} for step in cut["steps"]] == [{**step, "step_id": 0} for step in second.steps]
    assert "final_metrics" not in cut and cut["session_id"] == document["session_id"]
    both = task_document(document, [first.task_id, second.task_id])
    assert errors_in(validate(both)) == [] and len(both["steps"]) == len(document["steps"])
    with pytest.raises(KeyError, match="task-none"):
        task_document(document, ["task-none"])


def test_a_cut_keeps_the_sub_agents_its_steps_reach_and_remaps_arrivals():
    document = fixture("subagent")
    [task] = tasks(document)
    assert [sub["trajectory_id"] for sub in task_document(document, [task.task_id])["subagent_trajectories"]] == ["sub1"]
    nested = copy.deepcopy(document)
    sub1 = nested["subagent_trajectories"][0]
    nested["subagent_trajectories"].append({**copy.deepcopy(sub1), "trajectory_id": "sub2"})
    nested["subagent_trajectories"].append({**copy.deepcopy(sub1), "trajectory_id": "unreached"})
    sub1["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"] = [{"trajectory_id": "sub2"}]
    cut = task_document(nested, [task.task_id])
    assert [sub["trajectory_id"] for sub in cut["subagent_trajectories"]] == ["sub1", "sub2"]
    late = fixture("two_turns")
    [result] = late["steps"][7]["observation"]["results"]  # step 8, the second task's call
    result.setdefault("extra", {}).setdefault("agentenv", {})["arrived"] = {"at": "2026-11-04T17:09:12.004Z", "step_id": 9}
    earlier = late["steps"][1]["observation"]["results"][0]  # step 2, arriving during the second task
    earlier.setdefault("extra", {}).setdefault("agentenv", {})["arrived"] = {"at": "2026-11-04T17:09:12.004Z", "step_id": 8}
    second = task_document(late, [tasks(late)[1].task_id])
    assert second["steps"][1]["observation"]["results"][0]["extra"]["agentenv"]["arrived"]["step_id"] == 3
    first = task_document(late, [tasks(late)[0].task_id])
    assert first["steps"][1]["observation"]["results"][0]["extra"]["agentenv"]["arrived"] == {
        "at": "2026-11-04T17:09:12.004Z"}



def test_a_cut_resolves_each_reference_where_it_sits_at_any_depth():
    """sub1 embeds inner, whose reference names the root's sibling, so the cut keeps sibling; sub1's own reference names
    its own inner, so the root's inner, which that shadows, is left out."""
    document = fixture("subagent")
    [task] = tasks(document)
    sub1 = document["subagent_trajectories"][0]
    inner, decoy, sibling = ({**copy.deepcopy(sub1), "trajectory_id": name} for name in ("inner", "inner", "sibling"))
    inner["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"] = [{"trajectory_id": "sibling"}]
    sub1["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"] = [{"trajectory_id": "inner"}]
    sub1["subagent_trajectories"] = [inner]
    document["subagent_trajectories"] += [decoy, sibling]
    cut = task_document(document, [task.task_id])
    assert [sub["trajectory_id"] for sub in cut["subagent_trajectories"]] == ["sub1", "sibling"]
    assert [finding for finding in validate(cut) if finding.rule == "reference"] == []

def test_a_page_is_a_prefix_only_when_its_steps_lead_every_trajectory():
    final = fixture("subagent")
    page = {**final, "steps": final["steps"][:2],
            "subagent_trajectories": [{**final["subagent_trajectories"][0],
                                       "steps": final["subagent_trajectories"][0]["steps"][:1]}]}
    assert is_prefix(page, final) and is_prefix(None, final)
    assert not is_prefix({**page, "steps": final["steps"][1:3]}, final)
    assert not is_prefix(final, {**final, "subagent_trajectories": []})


def test_stripping_the_profile_leaves_tool_arguments_and_other_extras_alone():
    document = fixture("coding")
    payload = {"extra": {"agentenv": {"x": 1}}, "nested": [{"extra": {"agentenv": 2}}]}
    call = document["steps"][1]["tool_calls"][0]
    call["arguments"] = payload
    call["extra"] = {"raw_arguments": payload, "agentenv": {}}
    document["extra"]["other"] = {"agentenv": "kept"}
    stripped = strip_profile(document)
    assert stripped["steps"][1]["tool_calls"][0] == {**call, "extra": {"raw_arguments": payload}}
    assert stripped["extra"] == {"other": {"agentenv": "kept"}}
    assert "extra" not in stripped["steps"][0] and document["steps"][1]["tool_calls"][0]["extra"]["agentenv"] == {}
    check(stripped)
