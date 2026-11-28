"""Antigravity's ATIF adapter on real runs, recorded through the agent's image on the pinned SDK with atif/1 prescribed
(``tst/data/antigravity_atif``), on Gemini 2.5 Flash: each scenario's turns as the agent served them, the SDK's chunks
and hook results, its aux log of the relay's model calls and its turn context, replayed through the adapter kit as the
SDK ran it live.

The scenarios: ``coding`` (two turns of one conversation, the first with a call the harness found invalid and a file
the SDK streams as edit_file), ``browser`` (a screenshot through a browser's MCP tool, which the harness keeps on disk
and names to the model), ``reasoning`` (a high reasoning effort), ``refused`` (the endpoint refusing every call),
``retried`` (refused calls the SDK retried itself) and ``subagent`` (an ``invoke_subagent`` call, whose sub-agent runs on
after it returns and reports back while the turn goes on).
Regenerate the expected documents after an intended change with: python
tst/unit/agents/antigravity/test_antigravity_atif_recorded.py. Record the runs again, as a bump of the pinned SDK asks,
with ``tst/manual/agents_atif/record.py`` and ``make_fixtures.py``, whose docstrings have the steps."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import profile_of  # noqa: E402

from tst.unit.agents.antigravity.agent import antigravity_atif  # noqa: E402
from tst.util.atif_recorded import (  # noqa: E402
    Recordings,
    agent_steps,
    assert_every_call_has_its_result,
    assert_steps_sum_to_reported_usage,
    images,
    trajectories,
)

RECORDINGS = Recordings(REPO / "tst" / "data" / "antigravity_atif", antigravity_atif.AntigravityAtif)


def relay_calls(scenario: str, **match) -> list[dict]:
    """The relay's record of each model call that ended, of the conversation ``match`` names when it does."""
    turns = RECORDINGS.data(scenario)["turns"]
    requests = {(index, entry["record"]["seq"]): entry["record"] for index, turn in enumerate(turns)
                for entry in turn["aux"] if entry["record"]["type"] == "agentenv.model_request"}
    return [entry["record"] for index, turn in enumerate(turns) for entry in turn["aux"]
            if entry["record"]["type"] == "agentenv.model_call" and all(
                requests[index, entry["record"]["seq"]].get(key) == value for key, value in match.items())]


def model_steps(trajectory: dict) -> list[dict]:
    """The steps of the relay's calls, not the closing reply the SDK adds."""
    return [step for step in agent_steps(trajectory) if "model_name" in step]


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    RECORDINGS.check(scenario)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """Each call's own usage, the retries' and the sub-agent's included, adds up to what the relay counted for the
    turn."""
    assert_steps_sum_to_reported_usage(RECORDINGS.replay(scenario).document, RECORDINGS.recorded(scenario)[1])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_model_call_of_a_recorded_run_is_a_step_with_its_time_model_and_the_relays_counts(scenario):
    """One step per call the relay ended, at its end time, naming its model, a refused one with no model call; counts
    on exactly the calls the relay read usage from, which were none when these were recorded, as the harness asked
    its streams for no usage then."""
    document = RECORDINGS.replay(scenario).document
    ended = relay_calls(scenario)
    steps = [step for trajectory in trajectories(document) for step in model_steps(trajectory)]
    assert sorted((step["timestamp"], step["model_name"], step["llm_call_count"]) for step in steps) == sorted(
        (record["timestamp"], record["model"], int(record["outcome"] != "refused")) for record in ended)
    assert sum("completion_tokens" in step.get("metrics", {}) for step in steps) == sum(
        record["outcome"] == "answered" for record in ended)
    assert sum(record["outcome"] != "refused" for record in ended) == sum(
        ((usage or {}).get("provider_details") or {}).get("model_calls", 0) for usage in RECORDINGS.recorded(scenario)[1])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_call_of_a_recorded_run_has_its_result(scenario):
    assert_every_call_has_its_result(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_each_turn_of_a_recorded_run_ends_on_its_answer_and_holds_every_tool_call_streamed(scenario):
    document = RECORDINGS.replay(scenario).document
    tasks: list[list[dict]] = []
    for step in document["steps"]:
        tasks.append([]) if step["source"] == "user" else tasks[-1].append(step)
    for task, turn in zip(tasks, RECORDINGS.data(scenario)["turns"]):
        assert task[-1]["message"] == "\n\n".join(turn["turn"]["final"])
    copied = [call["tool_call_id"] for trajectory in trajectories(document) for step in agent_steps(trajectory)
              for call in step.get("tool_calls") or ()]
    assert copied == list(dict.fromkeys(copied))
    assert sorted(copied) == sorted(record["id"] for turn in RECORDINGS.data(scenario)["turns"]
                                    for record in turn["native"] if record.get("type") == "ToolCall")


def test_the_recordings_cover_what_the_adapter_maps():
    documents = {scenario: RECORDINGS.replay(scenario) for scenario in RECORDINGS.scenarios}

    browser = documents["browser"].document
    screenshot = next(result for step in agent_steps(browser) for call in step.get("tool_calls") or ()
                      if call["function_name"] == "browser_take_screenshot"
                      for result in step["observation"]["results"] if result["source_call_id"] == call["tool_call_id"])
    assert "[Resource offloaded to file:///" in screenshot["content"] and not images(browser)

    (refused,), closing = model_steps(documents["refused"].document), agent_steps(documents["refused"].document)[-1]
    assert refused["llm_call_count"] == 0 and profile_of(refused)["error"] == {
        "kind": "infra_error", "message": "the endpoint refused the call (HTTP 400)"}
    assert profile_of(closing)["state"] == "failed"

    retried = model_steps(documents["retried"].document)
    errors = [index for index, step in enumerate(retried) if "error" in profile_of(step)]
    assert len(errors) == 2 and all(retried[index]["llm_call_count"] == 0 and profile_of(retried[index])["error"][
        "message"] == "the endpoint refused the call (HTTP 503)" for index in errors)
    assert retried[errors[-1] + 1]["tool_calls"][0]["arguments"]["command_line"] == "echo second"

    subagent = documents["subagent"].document
    (sub,) = subagent["subagent_trajectories"]
    spawn = next(result for step in agent_steps(subagent) for result in (step.get("observation") or {}).get(
        "results") or () if "subagent_trajectory_ref" in result)
    assert spawn["subagent_trajectory_ref"] == [{"trajectory_id": sub["trajectory_id"]}]
    assert len(model_steps(sub)) == len(relay_calls("subagent", turn=False)) == 4
    assert [call["function_name"] for step in agent_steps(sub) for call in step.get("tool_calls") or ()] == [
        "edit_file"]
    assert sub["steps"][-1]["message"].startswith("Task complete.")
    assert [step["message"] for step in model_steps(subagent) if step["message"]] == ["forty-two", "forty-two"]

    coding = documents["coding"].document
    created = next(step for step in agent_steps(coding) for call in step.get("tool_calls") or ()
                   if call["function_name"] == "edit_file")
    assert [result["content"] for result in created["observation"]["results"]] == ["Create hello.py"]
    assert [step["source"] for step in coding["steps"]].count("user") == 2

    reasoning = model_steps(documents["reasoning"].document)
    natives = [record for turn in RECORDINGS.data("reasoning")["turns"] for record in turn["native"]]
    assert reasoning and all(step["reasoning_effort"] == "high" for step in reasoning)
    assert [bool(step.get("reasoning_content")) for step in reasoning] == [
        any(record.get("type") == "Thought" for record in natives)]


if __name__ == "__main__":
    RECORDINGS.regenerate()
