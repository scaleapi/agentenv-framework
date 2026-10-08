"""PromptResponse names its trajectory fields for the object store; its stored form keeps the S3-named keys.

Stored instances and raw-doc readers use the S3-named keys, so the stored form writes each beside its neutral one,
and from_dict reads either. The S3-named keywords and attributes are gone."""

import dataclasses

import pytest

from agent_env.cli.task.run import _format_prompt_response, _format_verifier_output
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.context_ops import build_context_update_ops

_URL, _PREFIX, _TURNS = "mem://b/t/2.json", "mem://b/t/", ["mem://b/t/1.json", None, "mem://b/t/2.json"]
_PAIRS = {
    "agent_trajectory_s3_uri": ("agent_trajectory_object_url", _URL),
    "agent_trajectory_s3_prefix": ("agent_trajectory_object_prefix", _PREFIX),
    "target_agent_per_turn_trajectory_s3_uris": ("target_agent_per_turn_trajectory_object_urls", _TURNS),
}


def _neutral() -> PromptResponse:
    return PromptResponse(prompt_id="p1", response="done", agent_trajectory_object_url=_URL,
                          agent_trajectory_object_prefix=_PREFIX, target_agent_per_turn_trajectory_object_urls=_TURNS)


def test_the_stored_form_carries_both_keys():
    stored = _neutral().to_dict()
    assert all(stored[legacy] == stored[neutral] == value for legacy, (neutral, value) in _PAIRS.items())


def test_a_persisted_response_carries_both_keys():
    pre = TaskStepContext()
    post = TaskStepContext(prompt_responses=[_neutral()])
    [pushed] = build_context_update_ops(pre, post).add_to_sets["context.prompt_responses"]
    assert all(pushed[legacy] == pushed[neutral] == value for legacy, (neutral, value) in _PAIRS.items())


@pytest.mark.parametrize("serialize", [TaskStepContext.to_dict, TaskStepContext.to_safe_dict])
def test_a_saved_context_carries_both_keys(serialize):
    [saved] = serialize(TaskStepContext(prompt_responses=[_neutral()]))["prompt_responses"]
    assert all(saved[legacy] == saved[neutral] == value for legacy, (neutral, value) in _PAIRS.items())


@pytest.mark.parametrize("spelling", ["legacy", "neutral"])
def test_a_stored_response_loads_keyed_either_way(spelling):
    keyed = {(legacy if spelling == "legacy" else neutral): value for legacy, (neutral, value) in _PAIRS.items()}
    response = PromptResponse.from_dict({"prompt_id": "p1", "response": "done", **keyed})
    assert response == _neutral()


def test_the_stored_form_round_trips():
    response = _neutral()
    assert PromptResponse.from_dict(response.to_dict()) == response


def test_the_legacy_key_wins_when_the_two_disagree():
    """Only a raw-doc writer outside the model can make them differ, and today's know only the legacy key."""
    response = PromptResponse.from_dict({"prompt_id": "p1", "response": "done", "agent_trajectory_s3_uri": _URL,
                                         "agent_trajectory_object_url": "mem://b/stale.json",
                                         "target_agent_per_turn_trajectory_s3_uris": [],
                                         "target_agent_per_turn_trajectory_object_urls": ["mem://b/stale.json"]})
    assert response.agent_trajectory_object_url == _URL
    assert response.target_agent_per_turn_trajectory_object_urls == []


def test_a_legacy_key_holding_none_falls_through_to_the_neutral_one():
    response = PromptResponse.from_dict({"prompt_id": "p1", "response": "done", "agent_trajectory_s3_uri": None,
                                         "agent_trajectory_object_url": _URL})
    assert response.agent_trajectory_object_url == _URL


@pytest.mark.parametrize("legacy", [*_PAIRS, "compact_trajectory_s3_uri"])
def test_the_s3_named_fields_are_gone(legacy):
    assert legacy not in {f.name for f in dataclasses.fields(PromptResponse)}
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{legacy}'"):
        PromptResponse(prompt_id="p1", response="done", **{legacy: _URL})


def test_the_run_output_prints_the_neutral_labels():
    printed = _format_prompt_response(_neutral())
    assert f"agent_trajectory_object_url: {_URL}" in printed and f"agent_trajectory_object_prefix: {_PREFIX}" in printed
    assert "s3" not in printed


@pytest.mark.parametrize("spelling", ["legacy", "neutral"])
def test_the_verifier_output_reads_metadata_keyed_either_way(spelling):
    def keyed(legacy, neutral, value):
        return {legacy if spelling == "legacy" else neutral: value}

    vdata = {**keyed("compact_trajectory_s3_uri", "compact_trajectory_object_url", "mem://b/compact.json"),
             **keyed("judge_trajectory_s3_uri", "judge_trajectory_object_url", "mem://b/judge.json"),
             "stdout_artifact": {"id": "out", "version": 1, **keyed("s3_url", "object_url", "mem://b/stdout.txt")},
             "stderr_artifact": {"id": "err", "version": 1, **keyed("s3_url", "object_url", "mem://b/stderr.txt")}}
    assert _format_verifier_output("v", vdata) == [
        "compact_trajectory_object_url (verifier_id=v): mem://b/compact.json",
        "judge_trajectory_object_url (verifier_id=v): mem://b/judge.json",
        "stdout_artifact (verifier_id=v): out v1 -> mem://b/stdout.txt",
        "stderr_artifact (verifier_id=v): err v1 -> mem://b/stderr.txt",
    ]
