"""OpenCode's ATIF adapter on real runs, recorded through the agent's image on the pinned CLI with atif/1 prescribed
(``tst/data/opencode_atif``): each scenario's turns as the agent served them, the CLI's events, its aux log of the
turn's settings, the relay's record of each model call and the sub-agent sessions read from the CLI's session store,
and its turn context, replayed through the adapter kit as the SDK ran it live.

The scenarios: ``coding`` (two turns of one conversation, the first with the session title's call), ``browser`` (a
screenshot a browser's MCP tool returned to the model), ``reasoning`` (a high reasoning effort), ``refused`` (the
endpoint refusing every call), ``retried`` (failed attempts the CLI retried itself), ``subagent`` (a task call's
sub-agent) and ``subagents`` (two task calls of one step, their sub-agents run at once).
Regenerate the expected documents after an intended change with: python
tst/unit/agents/opencode/test_opencode_atif_recorded.py. Record the runs again, as a bump of the pinned CLI asks, with
``tst/manual/agents_atif/record.py`` and ``make_fixtures.py``, whose docstrings have the steps."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import check_media, profile_of  # noqa: E402

from tst.unit.agents.opencode.agent import opencode_atif  # noqa: E402
from tst.util.atif_recorded import (  # noqa: E402
    Recordings,
    agent_steps,
    assert_every_call_has_its_result,
    assert_every_model_call_has_its_time_and_usage,
    assert_steps_sum_to_reported_usage,
    images,
    trajectories,
)

RECORDINGS = Recordings(REPO / "tst" / "data" / "opencode_atif", opencode_atif.OpenCodeAtif)


def tasks_of(document: dict) -> list[list[dict]]:
    """The document's agent steps, task by task."""
    tasks: list[list[dict]] = []
    for step in document["steps"]:
        tasks.append([]) if step["source"] == "user" else tasks[-1].append(step)
    return tasks


def messages_of(native: list[dict]) -> list[list[dict]]:
    """A turn's CLI events, message by message."""
    messages: dict[str, list[dict]] = {}
    for event in native:
        if isinstance(event.get("part"), dict) and "messageID" in event["part"]:
            messages.setdefault(event["part"]["messageID"], []).append(event)
    return list(messages.values())


def counts_of(tokens: dict) -> tuple[int, int, int]:
    """A step_finish's counts as ATIF's: the cache in the prompt, the reasoning in the completion."""
    cache = tokens.get("cache") or {}
    return (tokens["input"] + cache.get("read", 0) + cache.get("write", 0), tokens["output"] + tokens["reasoning"],
            cache.get("read", 0))


def task_calls(native: list[dict]) -> list[dict]:
    return [event["part"] for event in native if event["type"] == "tool_use" and event["part"]["tool"] == "task"]


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    RECORDINGS.check(scenario)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """Each call's own usage, the title's, the retries' and the sub-agents' included, adds up to what the relay counted
    for the turn."""
    assert_steps_sum_to_reported_usage(RECORDINGS.replay(scenario).document, RECORDINGS.recorded(scenario)[1])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_model_call_of_a_recorded_run_is_a_step_with_its_time_model_and_usage(scenario):
    document = RECORDINGS.replay(scenario).document
    assert_every_model_call_has_its_time_and_usage(document)
    model_steps = [step for trajectory in trajectories(document) for step in agent_steps(trajectory)
                   if step.get("llm_call_count") != 0]
    turns = RECORDINGS.data(scenario)["turns"]
    answered = [entry for turn in turns for entry in turn["aux"]
                if entry["name"] == "calls" and entry["record"]["outcome"] in ("answered", "unreported")]
    assert len(model_steps) == len(answered) == sum(
        ((turn["usage"] or {}).get("provider_details") or {}).get("model_calls", 0) for turn in turns)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_call_of_a_recorded_run_has_its_result(scenario):
    assert_every_call_has_its_result(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_each_message_of_a_recorded_run_is_one_step_with_its_content_and_the_counts_its_step_finish_gives(scenario):
    """The join of the CLI's messages to the relay's calls: each message the CLI printed is one root step, with its
    text, reasoning and calls, none lost or written twice, and the counts of the call it counted."""
    document = RECORDINGS.replay(scenario).document
    for task, turn in zip(tasks_of(document), RECORDINGS.data(scenario)["turns"]):
        messages = [events for events in messages_of(turn["native"]) if any(
            event["type"] in ("text", "tool_use", "reasoning") for event in events)]
        steps = [step for step in task if step.get("llm_call_count") != 0
                 and (step["message"] or step.get("tool_calls") or step.get("reasoning_content"))]
        assert [(step["metrics"]["prompt_tokens"], step["metrics"]["completion_tokens"],
                 step["metrics"]["cached_tokens"], step["message"], bool(step.get("reasoning_content")),
                 [call["tool_call_id"] for call in step.get("tool_calls") or ()]) for step in steps] == [
            (*counts_of(next(event["part"]["tokens"] for event in events if event["type"] == "step_finish")),
             "\n".join(event["part"]["text"] for event in events
                       if event["type"] == "text" and event["part"]["text"].strip()).strip(),
             any(event["type"] == "reasoning" for event in events),
             [event["part"]["callID"] for event in events if event["type"] == "tool_use"])
            for events in messages]


def test_the_recordings_cover_what_the_adapter_maps():
    documents = {scenario: RECORDINGS.replay(scenario) for scenario in RECORDINGS.scenarios}
    browser = documents["browser"]
    assert images(browser.document)
    for image in images(browser.document):
        check_media(image["source"]["path"], browser.media[image["source"]["path"]])
    assert any(call["function_name"].startswith("browser_") for step in agent_steps(browser.document)
               for call in step.get("tool_calls") or ())

    *refused, closing = agent_steps(documents["refused"].document)
    assert refused and all(step["llm_call_count"] == 0 and profile_of(step)["error"]["code"] == "400"
                           for step in refused)
    assert profile_of(refused[-1])["error"]["message"].startswith("APIError:")
    assert profile_of(closing)["state"] == "failed"

    retried = agent_steps(documents["retried"].document)
    errors = [index for index, step in enumerate(retried) if "error" in profile_of(step)]
    assert errors and all(profile_of(retried[index])["error"]["code"] == "503" for index in errors)
    assert any(step.get("tool_calls") for step in retried[errors[-1]:])

    for scenario, spawned in (("subagent", 1), ("subagents", 2)):
        document, turns = documents[scenario].document, RECORDINGS.data(scenario)["turns"]
        calls = [part for turn in turns for part in task_calls(turn["native"])]
        assert len(calls) == spawned
        fed = {entry["record"]["session"]: entry["record"]["events"] for turn in turns for entry in turn["aux"]
               if entry["name"] == "subagents"}
        subs = {sub["trajectory_id"]: sub for sub in document.get("subagent_trajectories") or ()}
        for session, events in fed.items():
            assert [call["tool_call_id"] for step in subs[session]["steps"] for call in step.get("tool_calls") or ()] \
                == [event["part"]["callID"] for event in events if event["type"] == "tool_use"]
        if spawned == 1:
            (sub,) = subs.values()
            assert sub["steps"][-1]["message"] == "21"
        else:
            (start, end), (other_start, other_end) = (
                (part["state"]["time"]["start"], part["state"]["time"]["end"]) for part in calls)
            assert start <= other_end and other_start <= end
            assert len(subs) == len(fed)

    reasoning = [step for step in agent_steps(documents["reasoning"].document) if step.get("llm_call_count") != 0]
    assert reasoning and all(step["reasoning_effort"] == "high" for step in reasoning)


if __name__ == "__main__":
    RECORDINGS.regenerate()
