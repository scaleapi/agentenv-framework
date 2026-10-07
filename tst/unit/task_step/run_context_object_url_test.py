"""The S3-named run-context keys get neutral twins.

Stored instances carry the S3-named keys, so they are read forever and written beside the neutral ones while workers
on older versions read only them. The S3-named ``PromptResponse`` keywords still work, with a warning."""

import dataclasses
import logging
import warnings

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


def _neutral(**kw) -> PromptResponse:
    return PromptResponse(prompt_id="p1", response="done", agent_trajectory_object_url=_URL,
                          agent_trajectory_object_prefix=_PREFIX, target_agent_per_turn_trajectory_object_urls=_TURNS,
                          **kw)


def test_a_response_built_with_the_neutral_keywords_carries_both_spellings_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        response = _neutral()
    dumped = dataclasses.asdict(response)
    assert all(dumped[legacy] == dumped[neutral] == value for legacy, (neutral, value) in _PAIRS.items())


def test_a_persisted_response_carries_both_keys():
    pre = TaskStepContext()
    post = TaskStepContext(prompt_responses=[_neutral()])
    [pushed] = build_context_update_ops(pre, post).add_to_sets["context.prompt_responses"]
    assert all(pushed[legacy] == pushed[neutral] == value for legacy, (neutral, value) in _PAIRS.items())


@pytest.mark.parametrize("spelling", ["legacy", "neutral"])
def test_a_stored_response_loads_keyed_either_way(spelling):
    keyed = {(legacy if spelling == "legacy" else neutral): value for legacy, (neutral, value) in _PAIRS.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        response = PromptResponse.from_dict({"prompt_id": "p1", "response": "done", **keyed})
    assert dataclasses.asdict(response) == dataclasses.asdict(_neutral())


def test_the_legacy_key_wins_when_the_two_disagree():
    """Only a raw-doc writer outside the model can make them differ, and today's know only the legacy key."""
    response = PromptResponse.from_dict({"prompt_id": "p1", "response": "done", "agent_trajectory_s3_uri": _URL,
                                         "agent_trajectory_object_url": "mem://b/stale.json",
                                         "target_agent_per_turn_trajectory_s3_uris": [],
                                         "target_agent_per_turn_trajectory_object_urls": ["mem://b/stale.json"]})
    assert response.agent_trajectory_object_url == _URL
    assert response.target_agent_per_turn_trajectory_object_urls == []


def test_the_legacy_keywords_still_work_and_are_counted(caplog):
    with caplog.at_level(logging.WARNING, logger="agent_env.utils.deprecation"), \
            pytest.warns(DeprecationWarning) as caught:
        response = PromptResponse(prompt_id="p1", response="done", agent_trajectory_s3_uri=_URL,
                                  agent_trajectory_s3_prefix=_PREFIX, target_agent_per_turn_trajectory_s3_uris=_TURNS)
    assert dataclasses.asdict(response) == dataclasses.asdict(_neutral())
    symbols = [f"PromptResponse({legacy}=)" for legacy in _PAIRS]
    assert [str(w.message).split(" is deprecated")[0] for w in caught] == symbols
    assert [r.deprecated_symbol for r in caplog.records if getattr(r, "event", None) == "agent_env_deprecated_symbol"] \
        == symbols


def test_the_two_spellings_disagreeing_is_an_error_and_agreeing_is_a_copy():
    with pytest.raises(ValueError, match="agent_trajectory_s3_uri='mem://b/other.json' and agent_trajectory_object_url"):
        _neutral(agent_trajectory_s3_uri="mem://b/other.json")
    response = _neutral()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert dataclasses.replace(response) == response
        assert PromptResponse(**dataclasses.asdict(response)) == response


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
