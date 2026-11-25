"""Gemini's ATIF adapter on real runs, recorded through the agent's image on the pinned CLI with atif/1 prescribed
(``tst/data/gemini_atif``): each scenario's turns as the agent served them, its native events and proxy records, its
aux log of transcript messages and its turn context, replayed through the adapter kit as the SDK ran it live.

The scenarios: ``coding`` (two turns of one conversation), ``browser`` (a screenshot a browser's MCP tool returned to
the model), ``refused`` (the endpoint refusing every call) and ``retried`` (calls the endpoint failed that the CLI
retried). Regenerate the expected documents after an intended change with: python
tst/unit/agents/gemini/test_gemini_atif_recorded.py. Record the runs again, as a bump of the pinned CLI asks, with
``tst/manual/codex_gemini_atif/record.py``, whose docstring has the steps."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import RecordedTask, check_adapter, check_media, profile_of  # noqa: E402

from tst.unit.agents.gemini.agent import gemini_atif  # noqa: E402

FIXTURES = REPO / "tst" / "data" / "gemini_atif"
SCENARIOS = sorted(path.stem for path in FIXTURES.glob("*.json") if not path.name.endswith(".expected.json"))
GeminiAtif = gemini_atif.GeminiAtif


def recorded(scenario: str) -> tuple[list[RecordedTask], list[dict | None]]:
    data = json.loads((FIXTURES / f"{scenario}.json").read_text())
    return ([RecordedTask.from_served(turn["turn"], turn["native"], turn["aux"]) for turn in data["turns"]],
            [turn["usage"] for turn in data["turns"]])


def expected(scenario: str) -> dict:
    return json.loads((FIXTURES / f"{scenario}.expected.json").read_text())


def replay(scenario: str):
    return check_adapter(GeminiAtif, recorded(scenario)[0])


def _agent_steps(document: dict) -> list[dict]:
    return [step for step in document["steps"] if step["source"] == "agent"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    kit = check_adapter(GeminiAtif, recorded(scenario)[0], expected=expected(scenario))
    kit.raise_for_errors()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    """Each call the proxy forwarded, a message's or not, adds up to the turn's usage, which is the proxy's sum."""
    _, usage = recorded(scenario)
    final = replay(scenario).document["final_metrics"]
    reported = [turn or {} for turn in usage]
    for metric, field in (("total_prompt_tokens", "input_tokens"), ("total_completion_tokens", "output_tokens"),
                          ("total_cached_tokens", "cache_read_tokens")):
        assert final.get(metric, 0) == sum(turn.get(field) or 0 for turn in reported), metric


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_response_of_a_recorded_run_is_a_step_with_its_time_model_and_usage(scenario):
    for step in _agent_steps(replay(scenario).document):
        if step.get("llm_call_count") != 0:
            assert step.get("timestamp") and step.get("model_name"), step["step_id"]
            assert "completion_tokens" in step.get("metrics", {}), step["step_id"]


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_call_of_a_recorded_run_has_its_result(scenario):
    for step in _agent_steps(replay(scenario).document):
        answered = {result["source_call_id"] for result in (step.get("observation") or {}).get("results") or ()}
        assert {call["tool_call_id"] for call in step.get("tool_calls") or ()} <= answered, step["step_id"]


def test_the_recordings_cover_what_the_adapter_maps():
    documents = {scenario: replay(scenario) for scenario in SCENARIOS}
    browser = documents["browser"]
    images = [part for step in _agent_steps(browser.document)
              for result in (step.get("observation") or {}).get("results") or ()
              for part in result["content"] if isinstance(result["content"], list) and part.get("type") == "image"]
    assert images
    for image in images:
        check_media(image["source"]["path"], browser.media[image["source"]["path"]])
    assert any(step.get("reasoning_content") for step in _agent_steps(documents["coding"].document))
    retried = _agent_steps(documents["retried"].document)
    assert any(step["llm_call_count"] == 0 and "error" in profile_of(step) for step in retried)
    assert retried[-1]["message"]
    refused, closing = _agent_steps(documents["refused"].document)
    assert refused["llm_call_count"] == 0 and profile_of(refused)["error"]["message"].startswith("[API Error")
    assert profile_of(closing)["state"] == "failed"


if __name__ == "__main__":
    for name in SCENARIOS:
        run = replay(name)
        run.raise_for_errors()
        (FIXTURES / f"{name}.expected.json").write_text(json.dumps(run.document, indent=1, ensure_ascii=False) + "\n")
        print("wrote", name)
