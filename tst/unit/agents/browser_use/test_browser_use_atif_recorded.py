"""browser-use's ATIF adapter on real runs, recorded through the agent's image on the pinned browser-use with atif/1
prescribed (``tst/data/browser_use_atif``): each scenario's turns as the agent served them, its Pydantic AI messages,
its aux log (the turn, agent.py's model requests and pruned screenshots) and its turn context, replayed through the
adapter kit as the SDK ran it.

The scenarios, Claude Haiku driving a headless Chromium over CDP: ``browse`` (two turns of one conversation, seven
screenshots returned to the model, three of which browser-use pruned within their own turn), ``thinking`` (signed
thinking at a high effort), ``delivery`` (a two-step budget, ending in browser-use's delivery-only turn), ``cut`` (a
one-second run budget, which this recording's bare reply finished within), ``refused`` (the endpoint refusing the
call) and ``retried`` (the endpoint failing two calls the SDK retried).
Regenerate the expected documents after an intended change with: python
tst/unit/agents/browser_use/test_browser_use_atif_recorded.py. Record the runs again, as a bump of the pinned
browser-use asks, with ``tst/manual/agents_atif/record.py``, whose docstring has the steps."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import check_media, profile_of  # noqa: E402

from tst.unit.agents.browser_use.agent import browser_use_atif  # noqa: E402
from tst.util.atif_recorded import (  # noqa: E402
    Recordings,
    agent_steps,
    assert_every_call_has_its_result,
    assert_every_model_call_has_its_time_and_usage,
    assert_steps_sum_to_reported_usage,
    images,
)

RECORDINGS = Recordings(REPO / "tst" / "data" / "browser_use_atif", browser_use_atif.BrowserUseAtif)


def turns(document: dict) -> list[list[dict]]:
    """The document's agent steps, turn by turn."""
    split: list[list[dict]] = []
    for step in document["steps"]:
        if step["source"] == "user":
            split.append([])
        elif step["source"] == "agent":
            split[-1].append(step)
    return split


def result_texts(document: dict) -> list[str]:
    texts = []
    for step in agent_steps(document):
        for result in (step.get("observation") or {}).get("results") or ():
            content = result["content"]
            texts += [content] if isinstance(content, str) else [p["text"] for p in content if p["type"] == "text"]
    return texts


def infra_error(step: dict) -> bool:
    return profile_of(step).get("error", {}).get("kind") == "infra_error"


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    RECORDINGS.check(scenario)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """Each response's own usage adds up to what browser-use reported for its turn, and its cost to the turn's."""
    assert_steps_sum_to_reported_usage(RECORDINGS.replay(scenario).document, RECORDINGS.recorded(scenario)[1])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_model_call_of_a_recorded_run_is_a_step_with_its_time_model_and_usage(scenario):
    assert_every_model_call_has_its_time_and_usage(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_call_of_a_recorded_run_has_its_result(scenario):
    assert_every_call_has_its_result(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_each_turn_is_a_step_per_model_request_agent_py_saw(scenario):
    """Each response browser-use counted, and each request the endpoint refused, is a step of its own turn; a later
    turn's copy holds its own run's calls alone, though its native repeats the conversation before it."""
    split = turns(RECORDINGS.replay(scenario).document)
    for turn, steps in zip(RECORDINGS.data(scenario)["turns"], split, strict=True):
        attempts = [entry["record"] for entry in turn["aux"] if entry["name"] == "attempt"]
        answered = [step for step in steps if step["llm_call_count"] == 1 and not infra_error(step)]
        assert len(answered) == turn["usage"]["provider_details"]["requests"]
        assert len(answered) + sum(infra_error(step) for step in steps) == len(attempts)
        assert [profile_of(step)["error"]["message"] for step in steps if infra_error(step)
                and step["llm_call_count"] == 0] == [f"the endpoint refused the call (HTTP {attempt['status']})"
                                                     for attempt in attempts if attempt.get("status", 0) >= 400]
    calls = [{call["tool_call_id"] for step in steps for call in step.get("tool_calls") or ()} for steps in split]
    assert sum(len(ids) for ids in calls) == len(set().union(*calls))


def test_the_recordings_cover_what_the_adapter_maps():
    documents = {scenario: RECORDINGS.replay(scenario) for scenario in RECORDINGS.scenarios}
    assert not any(run.document.get("subagent_trajectories") for run in documents.values())

    browse = documents["browse"]
    assert [len(images({"steps": steps})) for steps in turns(browse.document)] == [3, 4]
    for image in images(browse.document):
        check_media(image["source"]["path"], browse.media[image["source"]["path"]])
    for scenario in ("browse", "retried"):  # every screenshot browser-use pruned comes back from agent.py's copy
        assert not any(browser_use_atif.PRUNED in text for text in result_texts(documents[scenario].document))

    refused, closing = agent_steps(documents["refused"].document)
    assert refused["llm_call_count"] == 0 and infra_error(refused)
    assert profile_of(closing)["state"] == "failed"
    retried = agent_steps(documents["retried"].document)
    assert [step["llm_call_count"] for step in retried] == [1, 0, 0, 1, 1, 1, 1]
    assert len(images(documents["retried"].document)) == 3

    thinking = agent_steps(documents["thinking"].document)
    assert {step["reasoning_effort"] for step in thinking} == {"high"}
    assert profile_of(thinking[0])["reasoning"][0]["visibility"] == "summary" and thinking[0]["reasoning_content"]
    for scenario in ("browse", "cut", "delivery", "retried"):  # browser-use's own effort, the turn configuring none
        assert {step.get("reasoning_effort") for step in agent_steps(documents[scenario].document)
                if step["llm_call_count"]} == {"medium"}

    delivery = documents["delivery"].document
    (deliver,) = [step for step in delivery["steps"] if step["source"] == "system"]
    assert deliver["message"].startswith("Budget nearly exhausted.")
    final = delivery["steps"][-1]
    assert final["tool_calls"][0]["function_name"] == "finish"
    assert final["message"] == RECORDINGS.data("delivery")["turns"][0]["turn"]["final"][0]


if __name__ == "__main__":
    RECORDINGS.regenerate()
