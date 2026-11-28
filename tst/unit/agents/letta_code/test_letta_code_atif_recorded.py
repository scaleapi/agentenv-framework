"""Letta Code's ATIF adapter on real runs, recorded through the agent's image on the pinned CLI with atif/1 prescribed
(``tst/data/letta_code_atif``): each scenario's turns as the agent served them, its stream-json events, its aux log of
the model relay's calls and its turn context, replayed through the adapter kit as the SDK ran it live.

The scenarios: ``coding`` (two turns of one conversation), ``browser`` (a screenshot a browser's MCP tool returned to
the model), ``refused`` (the endpoint refusing every call), ``retried`` (calls the endpoint failed, which the CLI
retried within its run) and ``subagent`` (a sub-agent the Agent tool started in the background). gpt-4.1-mini streams
no reasoning, so none holds any.
Regenerate the expected documents after an intended change with: python
tst/unit/agents/letta_code/test_letta_code_atif_recorded.py. Record the runs again, as a bump of the pinned CLI asks,
with ``tst/manual/agents_atif/record.py`` and ``make_fixtures.py``, whose docstrings have the steps."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import check_media, profile_of  # noqa: E402

from tst.unit.agents.letta_code.agent import letta_code_atif  # noqa: E402
from tst.util.atif_recorded import (  # noqa: E402
    Recordings,
    agent_steps,
    assert_every_call_has_its_result,
    assert_every_model_call_has_its_time_and_usage,
    assert_steps_sum_to_reported_usage,
    images,
    trajectories,
)

RECORDINGS = Recordings(REPO / "tst" / "data" / "letta_code_atif", letta_code_atif.LettaCodeAtif)


def relay_calls(scenario: str) -> list[dict]:
    return [entry["record"] for turn in RECORDINGS.data(scenario)["turns"] for entry in turn["aux"]
            if entry["name"] == "calls"]


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    RECORDINGS.check(scenario)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """Each call's own usage, its sub-agents' and side calls' included, adds up to what the relay counted for the
    turn."""
    assert_steps_sum_to_reported_usage(RECORDINGS.replay(scenario).document, RECORDINGS.recorded(scenario)[1])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_model_call_of_a_recorded_run_is_a_step_with_its_time_model_and_usage(scenario):
    assert_every_model_call_has_its_time_and_usage(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_call_of_a_recorded_run_has_its_result(scenario):
    assert_every_call_has_its_result(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_relay_call_of_a_recorded_run_is_one_step_and_each_printed_response_the_one_its_counts_name(scenario):
    """A step per call the relay answered, a step with no model call per one it refused beside the SDK's own closing
    step, and each run's printed usage, its prompt with the cache added back, the counts of a step in its order."""
    document, calls = RECORDINGS.replay(scenario).document, relay_calls(scenario)
    steps = [step for trajectory in trajectories(document) for step in agent_steps(trajectory)]
    assert sum(step["llm_call_count"] for step in steps) == sum(call["outcome"] != "refused" for call in calls)
    errors = [step for step in steps if step["llm_call_count"] == 0 and profile_of(step).get("error", {}).get(
        "kind") == "infra_error"]
    assert len(errors) == sum(call["outcome"] == "refused" for call in calls)
    counted = iter((step.get("metrics", {}).get("prompt_tokens"), step.get("metrics", {}).get("completion_tokens"))
                   for step in agent_steps(document) if step["llm_call_count"])
    printed = [(event["prompt_tokens"] + event["cached_input_tokens"] + event["cache_write_tokens"],
                event["completion_tokens"]) for turn in RECORDINGS.data(scenario)["turns"] for event in turn["native"]
               if event.get("message_type") == "usage_statistics"]
    assert all(counts in counted for counts in printed), printed


def test_the_recordings_cover_what_the_adapter_maps():
    documents = {scenario: RECORDINGS.replay(scenario) for scenario in RECORDINGS.scenarios}
    browser = documents["browser"]
    assert images(browser.document)
    for image in images(browser.document):
        check_media(image["source"]["path"], browser.media[image["source"]["path"]])
    assert any(call["function_name"].startswith("mcp__browser__") for step in agent_steps(browser.document)
               for call in step.get("tool_calls") or ())
    refused, closing = agent_steps(documents["refused"].document)
    assert refused["llm_call_count"] == 0 and profile_of(refused)["error"]["kind"] == "infra_error"
    assert "refused by the recording proxy" in profile_of(refused)["error"]["message"]
    assert profile_of(closing)["state"] == "failed"
    *calls, answer = agent_steps(documents["retried"].document)
    assert [profile_of(step)["error"]["message"] for step in calls if step["llm_call_count"] == 0] == [
        '503: {"message":"service unavailable","code":503}'] * 2
    assert answer["message"] == "first second" and calls[0]["tool_calls"]
    subagent = documents["subagent"].document
    spawns = [result for step in agent_steps(subagent) for result in (step.get("observation") or {}).get("results", ())
              if "\nAgent ID: agent-local-" in result["content"]]
    assert len(spawns) == 1
    fed = any(entry["name"] == "subagents" for turn in RECORDINGS.data("subagent")["turns"] for entry in turn["aux"])
    assert bool(subagent.get("subagent_trajectories")) == fed == ("subagent_trajectory_ref" in spawns[0])


if __name__ == "__main__":
    RECORDINGS.regenerate()
