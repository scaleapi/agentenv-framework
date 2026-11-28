"""Gemini's ATIF adapter on real runs, recorded through the agent's image on the pinned CLI with atif/1 prescribed
(``tst/data/gemini_atif``): each scenario's turns as the agent served them, its native events and proxy records, its
aux log of transcript messages and its turn context, replayed through the adapter kit as the SDK ran it live.

The scenarios: ``coding`` (two turns of one conversation), ``browser`` (a screenshot a browser's MCP tool returned to
the model), ``refused`` (the endpoint refusing every call) and ``retried`` (calls the endpoint failed that the CLI
retried). Regenerate the expected documents after an intended change with: python
tst/unit/agents/gemini/test_gemini_atif_recorded.py. Record the runs again, as a bump of the pinned CLI asks, with
``tst/manual/codex_gemini_atif/record.py``, whose docstring has the steps."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import check_media, profile_of  # noqa: E402

from tst.unit.agents.gemini.agent import gemini_atif  # noqa: E402
from tst.util.atif_recorded import (  # noqa: E402
    Recordings,
    agent_steps,
    assert_every_call_has_its_result,
    assert_every_model_call_has_its_time_and_usage,
    assert_steps_sum_to_reported_usage,
    images,
)

RECORDINGS = Recordings(REPO / "tst" / "data" / "gemini_atif", gemini_atif.GeminiAtif)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    RECORDINGS.check(scenario)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """Each call the proxy forwarded, a message's or not, adds up to the turn's usage, which is the proxy's sum."""
    assert_steps_sum_to_reported_usage(RECORDINGS.replay(scenario).document, RECORDINGS.recorded(scenario)[1])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_response_of_a_recorded_run_is_a_step_with_its_time_model_and_usage(scenario):
    assert_every_model_call_has_its_time_and_usage(RECORDINGS.replay(scenario).document)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_call_of_a_recorded_run_has_its_result(scenario):
    assert_every_call_has_its_result(RECORDINGS.replay(scenario).document)


def test_the_recordings_cover_what_the_adapter_maps():
    documents = {scenario: RECORDINGS.replay(scenario) for scenario in RECORDINGS.scenarios}
    browser = documents["browser"]
    assert images(browser.document)
    for image in images(browser.document):
        check_media(image["source"]["path"], browser.media[image["source"]["path"]])
    assert any(step.get("reasoning_content") for step in agent_steps(documents["coding"].document))
    retried = agent_steps(documents["retried"].document)
    assert any(step["llm_call_count"] == 0 and "error" in profile_of(step) for step in retried)
    assert retried[-1]["message"]
    refused, closing = agent_steps(documents["refused"].document)
    assert refused["llm_call_count"] == 0 and profile_of(refused)["error"]["message"].startswith("[API Error")
    assert profile_of(closing)["state"] == "failed"


if __name__ == "__main__":
    RECORDINGS.regenerate()
