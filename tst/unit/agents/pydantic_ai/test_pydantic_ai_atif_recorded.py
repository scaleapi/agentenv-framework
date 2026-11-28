"""Pydantic AI's ATIF adapter on real runs, recorded through the agent's image on the pinned Pydantic AI with atif/1
prescribed (``tst/data/pydantic_ai_atif``): each scenario's turns as the agent served them, its native messages, its aux
log and its turn context, replayed through the adapter kit as the SDK ran it live.

The scenarios: ``coding`` (two turns of one conversation), ``browser`` (Playwright MCP's screenshot, which the harness's
``Coder`` cuts to 64,000 characters of JSON text before the model sees it, so the run holds no image), ``reasoning`` (a
high effort, which Pydantic AI never sends Claude through an OpenAI-compatible endpoint), ``refused`` (the endpoint
refusing every call), ``retried`` (two 503s the OpenAI client retried inside one model call, which the messages never
show) and ``subagent`` (a delegate_task run, a sub-agent). Regenerate the expected documents after an intended change
with: python tst/unit/agents/pydantic_ai/test_pydantic_ai_atif_recorded.py. Record the runs again, as a bump of the
pinned Pydantic AI asks, with ``tst/manual/agents_atif/record.py`` and ``make_fixtures.py``, whose docstrings have the
steps."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import profile_of  # noqa: E402

from tst.unit.agents.pydantic_ai.agent import pydantic_ai_atif  # noqa: E402
from tst.util.atif_recorded import (  # noqa: E402
    Recordings,
    agent_steps,
    assert_every_call_has_its_result,
    assert_every_model_call_has_its_time_and_usage,
    assert_steps_sum_to_reported_usage,
    images,
    trajectories,
)

RECORDINGS = Recordings(REPO / "tst" / "data" / "pydantic_ai_atif", pydantic_ai_atif.PydanticAIAtif)


def messages(turn: dict) -> list[dict[str, Any]]:
    """A turn's Pydantic AI messages as the agent fed them: its own, its delegated runs' and its failed run's."""
    fed = list(turn["native"])
    for entry in turn["aux"]:
        if entry["name"] == pydantic_ai_atif.DELEGATED:
            fed.append(entry["record"]["message"])
        elif entry["name"] == pydantic_ai_atif.FAILED:
            fed.extend(entry["record"]["messages"])
    return fed


def parts(scenario: str, kind: str) -> list[dict[str, Any]]:
    return [part for turn in RECORDINGS.data(scenario)["turns"] for message in messages(turn)
            for part in message.get("parts") or () if part.get("part_kind") == kind]


def turns_of(trajectory: dict) -> list[list[dict]]:
    """The root trajectory's agent steps, a list per turn."""
    turns: list[list[dict]] = []
    for step in trajectory["steps"]:
        if step["source"] == "user":
            turns.append([])
        elif step["source"] == "agent":
            turns[-1].append(step)
    return turns


def calls(document: dict) -> list[dict]:
    return [call for trajectory in trajectories(document) for step in agent_steps(trajectory)
            for call in step.get("tool_calls") or ()]


def results(document: dict) -> list[dict]:
    return [result for trajectory in trajectories(document) for step in agent_steps(trajectory)
            for result in (step.get("observation") or {}).get("results") or ()]


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    RECORDINGS.check(scenario)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """Each response's own usage, delegated runs' included, adds up to what the agent reported for the turn."""
    assert_steps_sum_to_reported_usage(RECORDINGS.replay(scenario).document, RECORDINGS.recorded(scenario)[1])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_model_call_of_a_recorded_run_is_a_step_with_its_time_model_and_usage(scenario):
    assert_every_model_call_has_its_time_and_usage(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_call_of_a_recorded_run_has_its_result(scenario):
    assert_every_call_has_its_result(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_has_a_step_per_model_request_the_turn_counted(scenario):
    """A failed call's cut-off response (``interrupted``) is no answer, so it is the failed run's error step instead."""
    document = RECORDINGS.replay(scenario).document
    answered = [step for trajectory in trajectories(document) for step in agent_steps(trajectory)
                if step["llm_call_count"] == 1]
    responses = [message for turn in RECORDINGS.data(scenario)["turns"] for message in messages(turn)
                 if message["kind"] == "response" and message["state"] != "interrupted"]

    assert len(answered) == len(responses) == sum(usage["provider_details"]["requests"]
                                                  for usage in RECORDINGS.recorded(scenario)[1])
    assert [step["timestamp"] for step in answered] == [response["timestamp"] for response in responses]


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_turns_last_step_is_its_reply(scenario):
    document = RECORDINGS.replay(scenario).document
    for steps, turn in zip(turns_of(document), RECORDINGS.data(scenario)["turns"], strict=True):
        assert steps[-1]["message"] == "".join(turn["turn"]["final"])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_call_and_result_of_a_recorded_run_is_the_one_its_messages_hold(scenario):
    """Nothing is dropped or doubled: each call the model made, with its arguments, and each tool return, with the text
    the model was sent."""
    document = RECORDINGS.replay(scenario).document
    made = [(part["tool_call_id"], part["tool_name"], json.loads(part["args"])) for part in parts(scenario, "tool-call")]
    texts = {part["tool_call_id"]: part["content"] for part in parts(scenario, "tool-return")
             if isinstance(part["content"], str)}
    answered = {result["source_call_id"]: result["content"] for result in results(document)}

    assert [(call["tool_call_id"], call["function_name"], call["arguments"]) for call in calls(document)] == made
    assert len(answered) == len(results(document)) == len(made)
    assert {call_id: answered[call_id] for call_id in texts} == texts


def test_the_recordings_cover_what_the_adapter_maps():
    documents = {scenario: RECORDINGS.replay(scenario).document for scenario in RECORDINGS.scenarios}

    browser = documents["browser"]
    assert {"browser_navigate", "browser_take_screenshot"} <= {call["function_name"] for call in calls(browser)}
    binaries = [item for part in parts("browser", "tool-return") if isinstance(part["content"], list)
                for item in part["content"] if isinstance(item, dict) and item.get("kind") == "binary"]
    assert len(images(browser)) == len(binaries)

    (sub,) = documents["subagent"]["subagent_trajectories"]
    (spawn,) = [call for call in calls(documents["subagent"]) if call["function_name"] == "delegate_task"]
    (spawned,) = [result for result in results(documents["subagent"]) if result["source_call_id"] == spawn["tool_call_id"]]
    assert spawned["subagent_trajectory_ref"] == [{"trajectory_id": sub["trajectory_id"]}]
    assert sub["agent"]["name"] == "Pydantic AI" and agent_steps(sub)[-1]["message"] == spawned["content"]
    assert sub["steps"][0]["tool_calls"][0]["function_name"] == "shell"

    ((refused,),) = turns_of(documents["refused"])
    (turn,) = RECORDINGS.data("refused")["turns"]
    assert refused["llm_call_count"] == 0 and profile_of(refused)["state"] == "failed"
    assert profile_of(refused)["error"] == turn["turn"]["error"]
    assert refused["message"] == turn["turn"]["error"]["message"]

    retried = agent_steps(documents["retried"])
    assert [step["llm_call_count"] for step in retried] == [1, 1]
    assert not any("error" in profile_of(step) for step in retried)

    (reasoning,) = agent_steps(documents["reasoning"])
    (fed,) = [entry["record"] for entry in RECORDINGS.data("reasoning")["turns"][0]["aux"]
              if entry["name"] == pydantic_ai_atif.TURN]
    assert reasoning.get("reasoning_effort") == fed["effort"]
    for scenario, document in documents.items():
        thought = [part["content"] for part in parts(scenario, "thinking") if part["content"]]
        assert [step["reasoning_content"] for trajectory in trajectories(document)
                for step in agent_steps(trajectory) if step.get("reasoning_content")] == thought


if __name__ == "__main__":
    RECORDINGS.regenerate()
