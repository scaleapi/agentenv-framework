"""Hermes' ATIF adapter on real runs, recorded through the agent's image on the pinned Hermes with atif/1 prescribed
(``tst/data/hermes_atif``): each scenario's turns as the agent served them, the plugin's hook events, the server's
``turn`` aux records and the turn context, replayed through the adapter kit as the SDK ran it live.

The scenarios: ``coding`` (two turns of one conversation, a title call after the first), ``browser`` (Playwright MCP
beside the agent: a tool search batch entry Hermes refused before running it, a screenshot Hermes hands the model by its
cache path, and the vision call that reads it), ``refused`` (the endpoint refusing every call), ``retried`` (the fault
proxy's two failures, which reached no model call, so it holds no failed attempt), ``reasoning`` (a high reasoning
effort, for which the model gave no reasoning), ``subagent`` (a ``delegate_task`` child) and ``batch`` (one
``delegate_task`` call with two children). The browser run predates the plugin reading cached images, so its screenshot
is a path there.
Regenerate the expected documents after an intended change with: python
tst/unit/agents/hermes/test_hermes_atif_recorded.py. Record the runs again, as a bump of the pinned Hermes asks, with
``tst/manual/agents_atif/record.py`` and ``make_fixtures.py`` beside it, whose docstrings have the steps."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import check_media, profile_of  # noqa: E402

from tst.unit.agents.hermes.agent import hermes_atif  # noqa: E402
from tst.util.atif_recorded import (  # noqa: E402
    Recordings,
    agent_steps,
    assert_every_call_has_its_result,
    assert_every_model_call_has_its_time_and_usage,
    assert_steps_sum_to_reported_usage,
    images,
    trajectories,
)

RECORDINGS = Recordings(REPO / "tst" / "data" / "hermes_atif", hermes_atif.HermesAtif)


def natives(scenario: str) -> list[dict]:
    return [record for turn in RECORDINGS.data(scenario)["turns"] for record in turn["native"]]


def calls(document: dict) -> dict[str, dict]:
    return {call["tool_call_id"]: call for trajectory in trajectories(document) for step in agent_steps(trajectory)
            for call in step.get("tool_calls") or ()}


def results(document: dict) -> dict[str, dict]:
    return {result["source_call_id"]: result for trajectory in trajectories(document)
            for step in agent_steps(trajectory) for result in (step.get("observation") or {}).get("results") or ()}


def text(result: dict) -> str:
    content = result.get("content") or ""
    return content if isinstance(content, str) else "".join(part.get("text", "") for part in content)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    RECORDINGS.check(scenario)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """Each call's own usage, the auxiliary calls' and the children's included, adds up to what the turn reported."""
    assert_steps_sum_to_reported_usage(RECORDINGS.replay(scenario).document, RECORDINGS.recorded(scenario)[1])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_model_call_of_a_recorded_run_is_a_step_with_its_start_model_and_usage(scenario):
    assert_every_model_call_has_its_time_and_usage(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_call_of_a_recorded_run_has_its_result(scenario):
    assert_every_call_has_its_result(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_each_model_call_and_failed_attempt_hermes_recorded_is_one_step_and_no_event_is_left_out(scenario):
    run, records = RECORDINGS.replay(scenario), natives(scenario)
    steps = [step for trajectory in trajectories(run.document) for step in agent_steps(trajectory)]
    made = [r for r in records if r["type"] == "post_api_request"
            or r["type"] == "post_auxiliary_call" and not r.get("error")]

    assert sum(step["llm_call_count"] for step in steps) == len(made)
    assert sum("error" in profile_of(step) and step["llm_call_count"] == 0 for step in steps) == len(
        [r for r in records if r["type"] == "api_request_error"])
    assert {call for call in calls(run.document) if ":" not in call} == {
        r["tool_call_id"] for r in records if r["type"] in ("pre_tool_call", "post_tool_call")}
    assert "unmatched" not in profile_of(run.document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_each_tasks_last_step_is_its_reply_and_its_state(scenario):
    document, turns = RECORDINGS.replay(scenario).document, RECORDINGS.data(scenario)["turns"]
    users = [index for index, step in enumerate(document["steps"]) if step["source"] == "user"]
    for turn, end in zip(turns, [*users[1:], len(document["steps"])]):
        last = document["steps"][end - 1]
        assert last["message"] == "\n\n".join(turn["turn"]["final"])
        assert profile_of(last)["state"] == turn["turn"]["state"]
        assert (last["llm_call_count"] == 0) == (turn["turn"]["state"] == "failed")


def test_the_recordings_cover_what_the_adapter_maps():
    documents = {scenario: RECORDINGS.replay(scenario) for scenario in RECORDINGS.scenarios}

    browser = documents["browser"]
    assert len(images(browser.document)) == sum(len(r.get("media") or ()) for r in natives("browser"))
    for image in images(browser.document):
        check_media(image["source"]["path"], browser.media[image["source"]["path"]])
    named = {call["function_name"]: call_id for call_id, call in calls(browser.document).items()}
    assert "/cache/images/" in text(results(browser.document)[named["mcp__browser__browser_take_screenshot"]])
    refused = results(browser.document)[named["tool_call"]]
    assert profile_of(refused)["status"] == "error" and "missing required argument" in text(refused)
    assert {r.get("aux_task") for r in natives("browser")} >= {"vision", "title_generation"}

    coding = documents["coding"].document
    assert [profile_of(step)["turn"] for step in coding["steps"] if step["source"] == "user"] == [0, 1]
    assert {profile_of(result).get("exit_code") for call_id, result in results(coding).items()
            if calls(coding)[call_id]["function_name"] == "terminal"} == {0}

    error, closing = agent_steps(documents["refused"].document)
    assert error["llm_call_count"] == 0 and error["model_name"] and error["timestamp"]
    assert {key: profile_of(error)["error"][key] for key in ("kind", "code")} == {"kind": "infra_error", "code": "400"}
    assert profile_of(closing)["state"] == "failed"

    reasoning = [step for step in agent_steps(documents["reasoning"].document) if step.get("reasoning_effort")]
    assert [step["reasoning_effort"] for step in reasoning] == ["high"]
    assert all(step.get("reasoning_effort") is None for name in ("coding", "retried")
               for step in agent_steps(documents[name].document))

    for name, children in (("subagent", 1), ("batch", 2)):
        document = documents[name].document
        subs = document.get("subagent_trajectories") or []
        referenced = {ref["trajectory_id"] for result in results(document).values()
                      for ref in result.get("subagent_trajectory_ref") or ()}
        assert len(subs) == children and referenced == {sub["trajectory_id"] for sub in subs}
        assert all(agent_steps(sub)[-1]["message"] for sub in subs)
    spawned = [call for call in calls(documents["batch"].document).values() if call["function_name"] == "delegate_task"]
    assert len(spawned) == 2 and spawned[1]["tool_call_id"].startswith(spawned[0]["tool_call_id"] + ":")


if __name__ == "__main__":
    RECORDINGS.regenerate()
