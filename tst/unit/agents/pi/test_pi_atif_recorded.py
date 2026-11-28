"""pi's ATIF adapter on real runs, recorded through the agent's image on the pinned pi with atif/1 prescribed
(``tst/data/pi_atif``): each scenario's turns as the agent served them, its JSON-mode events, its aux log of the bridge's
records and its turn context, replayed through the adapter kit as the SDK ran it live. ``sessions/coding.jsonl`` is
the session file pi wrote in ``coding``, which the bridge replays a resumed session's tools from.

The scenarios: ``coding`` (two turns of one conversation), ``browser`` (a screenshot a browser's MCP tool returned to
the model), ``refused`` (the endpoint refusing every call), ``retried`` (two refused calls pi retried until one was
answered), ``thinking`` (a thinking level, which the endpoint answered with no reasoning text) and ``compaction`` (two
turns in a small context window, recorded with a spec under which pi found nothing to compact). pi spawns no
sub-agents. Regenerate the expected documents after an intended change with: python
tst/unit/agents/pi/test_pi_atif_recorded.py. Record the runs again, as a bump of the pinned pi asks, with
``tst/manual/agents_atif/record.py``, whose docstring has the steps."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import Fed, check_adapter, check_media, profile_of  # noqa: E402

from tst.unit.agents.pi.agent import pi_agent, pi_atif  # noqa: E402
from tst.util.atif_recorded import (  # noqa: E402
    Recordings,
    agent_steps,
    assert_every_call_has_its_result,
    assert_every_model_call_has_its_time_and_usage,
    assert_steps_sum_to_reported_usage,
    images,
    trajectories,
)

FIXTURES = REPO / "tst" / "data" / "pi_atif"
RECORDINGS = Recordings(FIXTURES, pi_atif.PiAtif)


def _natives(scenario: str, kind: str, role: str | None = None) -> list[dict]:
    """The scenario's native records of ``kind`` (a ``message_end``'s message, of ``role``), in the order pi printed
    them, every turn's."""
    tasks, _ = RECORDINGS.recorded(scenario)
    records = [fed.record for task in tasks for fed in task.records if fed.aux is None and fed.record["type"] == kind]
    return records if role is None else [record["message"] for record in records if record["message"]["role"] == role]


def _tasks_steps(document: dict) -> list[list[dict]]:
    """The root's agent steps, by task: those after each user step."""
    grouped: list[list[dict]] = []
    for step in document["steps"]:
        if step["source"] == "user":
            grouped.append([])
        elif step["source"] == "agent":
            grouped[-1].append(step)
    return grouped


def _reasoning_blocks(step: dict) -> int:
    blocks = profile_of(step).get("reasoning")
    return len(blocks) if blocks is not None else int(bool(step.get("reasoning_content")))


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    RECORDINGS.check(scenario)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """Each call's own usage, a compaction's summary call's included, adds up to what the bridge reported for the
    turn."""
    assert_steps_sum_to_reported_usage(RECORDINGS.replay(scenario).document, RECORDINGS.recorded(scenario)[1])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_model_call_of_a_recorded_run_is_a_step_with_its_time_model_and_usage(scenario):
    assert_every_model_call_has_its_time_and_usage(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_each_assistant_message_is_one_step_named_by_the_model_called_and_timed_when_pi_sent_it(scenario):
    """One step per call pi made, in order: a call the endpoint refused (an error, no usage and no content) a step with
    no model call, every other a model call, each named as the turn's model is, whatever name the endpoint answered
    with."""
    document = RECORDINGS.replay(scenario).document
    calls = [step for step in agent_steps(document) if step.get("model_name") and "compaction" not in profile_of(step)]
    want = [(pi_atif._iso(message["timestamp"]), message["model"],
             0 if message["stopReason"] == "error" and pi_atif.counted(message["usage"]) is None
             and not message["content"] else 1)
            for message in _natives(scenario, "message_end", "assistant")]

    assert [(step["timestamp"], step["model_name"], step["llm_call_count"]) for step in calls] == want
    assert {step["model_name"] for step in calls} == {document["agent"]["model_name"]}


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_call_of_a_recorded_run_is_in_its_step_with_its_result(scenario):
    document = RECORDINGS.replay(scenario).document
    called = [block["id"] for message in _natives(scenario, "message_end", "assistant")
              if message["stopReason"] not in ("error", "aborted")
              for block in message["content"] if block["type"] == "toolCall"]

    assert [call["tool_call_id"] for step in agent_steps(document) for call in step.get("tool_calls") or ()] == called
    assert_every_call_has_its_result(document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_completed_turns_reply_is_its_last_calls_text_written_once(scenario):
    """pi's reply is its last call's text, so the SDK adds no step repeating it."""
    tasks, _ = RECORDINGS.recorded(scenario)
    for task, steps in zip(tasks, _tasks_steps(RECORDINGS.replay(scenario).document), strict=True):
        if task.state == "completed":
            assert steps[-1]["llm_call_count"] == 1 and steps[-1]["message"] == "".join(task.final)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_thinking_block_and_compaction_pi_printed_is_in_the_copy(scenario):
    document = RECORDINGS.replay(scenario).document
    thought = sum(block["type"] == "thinking" for message in _natives(scenario, "message_end", "assistant")
                  for block in message["content"])

    assert sum(_reasoning_blocks(step) for step in agent_steps(document)) == thought
    assert sum("compaction" in profile_of(step) for step in agent_steps(document)) == len(
        [end for end in _natives(scenario, "compaction_end") if end.get("result")])


def test_a_new_document_on_the_recorded_session_gets_the_tools_its_first_turn_declared():
    """What the bridge feeds a copy that starts on a session pi resumes, which declares no tools: those it replays from
    the session file pi wrote."""
    tools = pi_agent._declared_tools(FIXTURES / "sessions" / "coding.jsonl")
    tasks, _ = RECORDINGS.recorded("coding")
    resumed = dataclasses.replace(tasks[1], records=[Fed({"type": "tools", "tools": tools}, aux=pi_atif.AUX),
                                                     *tasks[1].records])
    kit = check_adapter(pi_atif.PiAtif, [resumed])
    kit.raise_for_errors()

    assert not _natives("coding", "message_end", "system")[1:]
    assert kit.document["agent"]["tool_definitions"] == RECORDINGS.replay("coding").document["agent"][
        "tool_definitions"]


def test_the_recordings_cover_what_the_adapter_maps():
    documents = {scenario: RECORDINGS.replay(scenario) for scenario in RECORDINGS.scenarios}
    browser = documents["browser"]
    assert images(browser.document)
    for image in images(browser.document):
        check_media(image["source"]["path"], browser.media[image["source"]["path"]])
    assert "mcp__browser__browser_take_screenshot" in [
        call["function_name"] for step in agent_steps(browser.document) for call in step.get("tool_calls") or ()]
    assert "mcp__browser__browser_take_screenshot" in [
        tool["function"]["name"] for tool in browser.document["agent"]["tool_definitions"]]
    coding = documents["coding"].document
    assert [step["source"] for step in coding["steps"]].count("user") == 2
    assert [tool["function"]["name"] for tool in coding["agent"]["tool_definitions"]] == ["read", "bash", "edit",
                                                                                           "write"]
    refused, closing = agent_steps(documents["refused"].document)
    assert refused["llm_call_count"] == 0 and profile_of(refused)["error"]["kind"] == "infra_error"
    assert profile_of(closing)["state"] == "failed"
    retried = agent_steps(documents["retried"].document)
    assert [(step["llm_call_count"], profile_of(step).get("attempt", 1)) for step in retried] == [
        (1, 1), (0, 1), (0, 2), (1, 3)]
    assert [profile_of(step)["error"]["kind"] for step in retried if step["llm_call_count"] == 0] == ["infra_error"] * 2
    assert profile_of(retried[-1])["state"] == "completed"
    (thinking,) = agent_steps(documents["thinking"].document)
    assert thinking["reasoning_effort"] == "low"
    assert all(len(trajectories(run.document)) == 1 for run in documents.values())


if __name__ == "__main__":
    RECORDINGS.regenerate()
