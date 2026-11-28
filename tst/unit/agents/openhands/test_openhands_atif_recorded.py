"""OpenHands' ATIF adapter on real runs, recorded through the agent's image on the pinned OpenHands SDK with atif/1
prescribed (``tst/data/openhands_atif``): each scenario's turns as the agent served them, the SDK's events, the aux log
the runtime fed from the LLMs' metrics (``llm_call``, ``tools``, ``clock``, ``turn``) and the turn context, replayed
through the adapter kit as the SDK ran it live.

The scenarios: ``coding`` (two turns of one conversation, the second resuming it), ``browser`` (Playwright MCP's
screenshot returned to the model as an image), ``refused`` (the endpoint refusing every call, which the SDK doesn't
retry), ``retried`` (two 503s the SDK's LLM retried before the call went through) and ``reasoning`` (a high effort, which
Claude answers with signed thinking blocks). The agent spawns no sub-agent: its tools have no delegate.
Regenerate the expected documents after an intended change with: python
tst/unit/agents/openhands/test_openhands_atif_recorded.py. Record the runs again, as a bump of the pinned SDK asks, with
``tst/manual/agents_atif/record.py`` and ``make_fixtures.py``, whose docstrings have the steps."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import check_media, profile_of  # noqa: E402

from tst.unit.agents.openhands.agent import openhands_atif  # noqa: E402
from tst.util.atif_recorded import (  # noqa: E402
    Recordings,
    agent_steps,
    assert_every_call_has_its_result,
    assert_every_model_call_has_its_time_and_usage,
    assert_steps_sum_to_reported_usage,
    images,
)

RECORDINGS = Recordings(REPO / "tst" / "data" / "openhands_atif", openhands_atif.OpenHandsAtif)


def events(scenario: str, kind: str) -> list[dict[str, Any]]:
    return [event for turn in RECORDINGS.data(scenario)["turns"] for event in turn["native"] if event["kind"] == kind]


def calls_fed(scenario: str, outcome: str) -> list[dict[str, Any]]:
    """The ``llm_call`` records the runtime fed: ``answered`` a response the metrics hold, ``failed`` an attempt the
    SDK made again."""
    return [entry["record"] for turn in RECORDINGS.data(scenario)["turns"] for entry in turn["aux"]
            if entry["name"] == "llm_call" and entry["record"]["outcome"] == outcome]


def turns_of(document: dict) -> list[list[dict]]:
    """The agent steps, a list per turn."""
    turns: list[list[dict]] = []
    for step in document["steps"]:
        if step["source"] == "user":
            turns.append([])
        elif step["source"] == "agent":
            turns[-1].append(step)
    return turns


def texts(content: list[dict[str, Any]]) -> list[str]:
    return [part["text"] for part in content if part.get("type") == "text"]


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    RECORDINGS.check(scenario)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """Each response's usage read off the LLMs' metrics adds up to what the runtime reported for the turn, its cost
    too."""
    assert_steps_sum_to_reported_usage(RECORDINGS.replay(scenario).document, RECORDINGS.recorded(scenario)[1])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_model_call_of_a_recorded_run_is_a_step_with_its_time_model_and_usage(scenario):
    assert_every_model_call_has_its_time_and_usage(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_call_of_a_recorded_run_has_its_result(scenario):
    assert_every_call_has_its_result(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_has_a_step_per_response_the_metrics_hold_at_the_time_it_was_read(scenario):
    """One step per response the events name, read just before its first event; and each attempt the SDK made again a
    step of its own naming the model it called, with no model call, holding the attempt's error."""
    document = RECORDINGS.replay(scenario).document
    answered, failed = calls_fed(scenario, "answered"), calls_fed(scenario, "failed")
    steps = agent_steps(document)
    responses = list(dict.fromkeys(event["llm_response_id"] for turn in RECORDINGS.data(scenario)["turns"]
                                   for event in turn["native"] if event.get("llm_response_id")))

    assert [record["response_id"] for record in answered] == responses
    assert [step["timestamp"] for step in steps if step["llm_call_count"] == 1] == [
        record["timestamp"] for record in answered]
    assert [(step["timestamp"], profile_of(step)["error"]["code"]) for step in steps
            if step["llm_call_count"] == 0 and step.get("model_name")] == [
        (record["timestamp"], record["error_type"]) for record in failed]


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_turns_last_step_is_its_reply_said_once(scenario):
    document = RECORDINGS.replay(scenario).document
    for steps, turn in zip(turns_of(document), RECORDINGS.data(scenario)["turns"], strict=True):
        reply = "".join(turn["turn"]["final"])
        assert steps[-1]["message"] == reply
        assert [step["message"] for step in steps].count(reply) == 1


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_call_and_result_of_a_recorded_run_is_the_one_its_events_hold(scenario):
    """Nothing is dropped or doubled: each call the model made, with the arguments it gave, and each observation, with
    the text the model was sent, on the step of the response that made the call."""
    document = RECORDINGS.replay(scenario).document
    made = [(event["tool_call_id"], event["tool_name"], json.loads(event["tool_call"]["arguments"]))
            for event in events(scenario, "ActionEvent")]
    observed = {event["tool_call_id"]: texts(event["observation"]["content"])
                for event in events(scenario, "ObservationEvent")}
    steps = agent_steps(document)

    assert [(call["tool_call_id"], call["function_name"], call["arguments"]) for step in steps
            for call in step.get("tool_calls") or ()] == made
    for step in steps:
        for result in (step.get("observation") or {}).get("results") or ():
            content = result["content"]
            got = [content] if isinstance(content, str) else texts(content)
            assert "\n\n".join(got) == "\n\n".join(observed.pop(result["source_call_id"])), step["step_id"]
    assert not observed


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_runs_reasoning_is_its_thinking_blocks_with_their_signatures(scenario):
    """Claude's signed thinking blocks, which LiteLLM's ``reasoning_content`` repeats, each written once."""
    document = RECORDINGS.replay(scenario).document
    blocks = [block for turn in RECORDINGS.data(scenario)["turns"] for event in turn["native"]
              for block in (event.get("thinking_blocks") or (event.get("llm_message") or {}).get("thinking_blocks")
                            or ())]
    steps = [step for step in agent_steps(document) if step.get("reasoning_content")]

    assert "\n\n".join(step["reasoning_content"] for step in steps) == "\n\n".join(block["thinking"] for block in blocks)
    assert [block["signature"] for step in steps for block in profile_of(step)["reasoning"]] == [
        block["signature"] for block in blocks]


def test_the_recordings_come_from_the_sdk_the_adapter_is_tested_with():
    assert {RECORDINGS.data(scenario)["cli"] for scenario in RECORDINGS.scenarios} == set(openhands_atif.TESTED_WITH)


def test_the_recordings_cover_what_the_adapter_maps():
    replays = {scenario: RECORDINGS.replay(scenario) for scenario in RECORDINGS.scenarios}
    documents = {scenario: replay.document for scenario, replay in replays.items()}

    browser = replays["browser"]
    shots = [url for event in events("browser", "ObservationEvent") for part in event["observation"]["content"]
             if part["type"] == "image" for url in part["image_urls"]]
    assert shots and len(images(browser.document)) == len(shots)
    for image in images(browser.document):
        check_media(image["source"]["path"], browser.media[image["source"]["path"]])
    names = {call["function_name"] for step in agent_steps(browser.document) for call in step.get("tool_calls") or ()}
    assert {"browser_browser_navigate", "browser_browser_take_screenshot"} <= names
    assert {"browser_browser_take_screenshot", "terminal"} <= {
        tool["function"]["name"] for tool in browser.document["agent"]["tool_definitions"]}

    ((error, closing),) = turns_of(documents["refused"])
    (turn,) = RECORDINGS.data("refused")["turns"]
    (failure,) = events("refused", "ConversationErrorEvent")
    assert error["llm_call_count"] == closing["llm_call_count"] == 0
    assert profile_of(error)["error"] == {"kind": "infra_error", "message": failure["detail"], "code": failure["code"]}
    assert profile_of(closing)["state"] == "failed" and closing["message"] == turn["turn"]["error"]["message"]

    retried = agent_steps(documents["retried"])
    assert [step["llm_call_count"] for step in retried] == [1, 0, 0, 1, 1]
    assert [profile_of(step)["error"]["code"] for step in retried[1:3]] == ["ServiceUnavailableError"] * 2
    assert {step["model_name"] for step in retried} == {RECORDINGS.data("retried")["turns"][0]["turn"]["agent_model"]}

    (reasoning,) = agent_steps(documents["reasoning"])
    assert reasoning["reasoning_effort"] == "high" and profile_of(reasoning)["reasoning"][0]["signature"]
    assert not any(document.get("subagent_trajectories") for document in documents.values())
    assert not any(profile_of(document).get("unmatched") for document in documents.values())
    assert len(turns_of(documents["coding"])) == 2 and not events("coding", "SystemPromptEvent")[1:]


if __name__ == "__main__":
    RECORDINGS.regenerate()
