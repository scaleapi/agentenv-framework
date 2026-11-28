"""Codex's ATIF adapter on real runs, recorded through the agent's image on the pinned CLI with atif/1 prescribed
(``tst/data/codex_atif``): each scenario's turns as the agent served them, its native events, its aux log of rollout
records and its turn context, replayed through the adapter kit as the SDK ran it live.

The scenarios: ``coding`` (two turns of one conversation), ``browser`` (a screenshot a browser's MCP tool returned to
the model), ``reasoning`` (a high reasoning effort), ``refused`` (the endpoint refusing every call), ``retried`` (a
failed attempt the bridge resumed), ``subagent`` (a spawned sub-agent) and ``forked`` (a second turn's sub-agent forked
with the conversation so far, whose rollout begins with a copy of it, on gpt-6.2-astra in the CLI's code mode).
Regenerate the expected documents after an intended change with: python
tst/unit/agents/codex/test_codex_atif_recorded.py. Record the runs again, as a bump of the pinned CLI asks, with
``tst/manual/codex_gemini_atif/record.py``, whose docstring has the steps."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import check_media, profile_of  # noqa: E402

from tst.unit.agents.codex.agent import codex_atif  # noqa: E402
from tst.util.atif_recorded import (  # noqa: E402
    Recordings,
    agent_steps,
    assert_every_call_has_its_result,
    assert_every_model_call_has_its_time_and_usage,
    assert_steps_sum_to_reported_usage,
    images,
)

RECORDINGS = Recordings(REPO / "tst" / "data" / "codex_atif", codex_atif.CodexAtif)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    RECORDINGS.check(scenario)


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """Each response's own usage, sub-agents' included, adds up to what the CLI reported for the turn."""
    assert_steps_sum_to_reported_usage(RECORDINGS.replay(scenario).document, RECORDINGS.recorded(scenario)[1])


@pytest.mark.parametrize("scenario", RECORDINGS.scenarios)
def test_every_model_call_of_a_recorded_run_is_a_step_with_its_rollouts_time_model_and_usage(scenario):
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
    assert any(call["function_name"].startswith("mcp__browser__") for step in agent_steps(browser.document)
               for call in step.get("tool_calls") or ())
    assert [len(documents[name].document.get("subagent_trajectories") or ()) for name in ("subagent", "forked")] \
        == [1, 1]
    retried = agent_steps(documents["retried"].document)
    assert any(profile_of(step).get("attempt") == 2 for step in retried)
    assert any("error" in profile_of(step) for step in retried)
    refused, closing = agent_steps(documents["refused"].document)
    assert refused["llm_call_count"] == 0 and profile_of(refused)["error"]["kind"] == "infra_error"
    assert profile_of(closing)["state"] == "failed"
    reasoning = agent_steps(documents["reasoning"].document)
    assert reasoning[0]["reasoning_effort"] == "high" and profile_of(reasoning[0])["reasoning"]


if __name__ == "__main__":
    RECORDINGS.regenerate()
