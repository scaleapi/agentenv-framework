"""Claude Code's ATIF adapter on real runs, recorded through the agent's image on the pinned CLI with atif/1 prescribed
(``tst/data/claude_code_atif``): each scenario's turns as the agent served them, its native events, its aux log and
its turn context, replayed through the adapter kit as the SDK ran it live. Each request shape's tools are kept once,
under ``tools/``, and named where they were fed.

The scenarios: ``coding`` (two turns of one conversation), ``subagent`` (an Agent call), ``browser`` (screenshots from a
browser's MCP tools), ``thinking`` (a thinking model), ``api-error`` (the endpoint refusing every call) and ``retried``
(a stalled call that the bridge resumed). Regenerate the expected documents after an intended change with:
python tst/unit/agents/claude_code/test_claude_code_atif_recorded.py"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
if __name__ == "__main__":
    sys.path.insert(0, str(REPO))

from agentenv_protocol.atif import RecordedTask, check_adapter  # noqa: E402

from tst.unit.agents.claude_code.agent import claude_code_atif  # noqa: E402

FIXTURES = REPO / "tst" / "data" / "claude_code_atif"
SCENARIOS = sorted(path.stem for path in FIXTURES.glob("*.json") if not path.name.endswith(".expected.json"))
ClaudeCodeAtif = claude_code_atif.ClaudeCodeAtif


def _shared(marker: dict) -> list:
    return json.loads((FIXTURES / marker["shared"]).read_text())


def recorded(scenario: str) -> tuple[list[RecordedTask], list[dict | None]]:
    data = json.loads((FIXTURES / f"{scenario}.json").read_text())
    tasks, usage = [], []
    for turn in data["turns"]:
        aux = [{**entry, "record": {**entry["record"], "tools": _shared(entry["record"]["tools"])}}
               if entry["name"] == "tools" else entry for entry in turn["aux"]]
        tasks.append(RecordedTask.from_served(turn["turn"], turn["native"], aux))
        usage.append(turn["usage"])
    return tasks, usage


def _trajectories(document: dict) -> list[dict]:
    return [document, *(document.get("subagent_trajectories") or [])]


def _tool_files() -> dict[str, list]:
    return {f"tools/{path.name}": [claude_code_atif._function(tool) for tool in json.loads(path.read_text())]
            for path in sorted((FIXTURES / "tools").glob("*.json"))}


def expected(scenario: str) -> dict:
    """The checked-in document, each agent block's tool definitions read back from the shared file it names."""
    document = json.loads((FIXTURES / f"{scenario}.expected.json").read_text())
    files = _tool_files()
    for trajectory in _trajectories(document):
        named = trajectory["agent"].get("tool_definitions")
        if isinstance(named, dict):
            trajectory["agent"]["tool_definitions"] = files[named["shared"]]
    return document


def replay(scenario: str):
    tasks, _ = recorded(scenario)
    return check_adapter(ClaudeCodeAtif, tasks)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_recorded_run_replays_to_its_document_under_the_live_rule(scenario):
    tasks, _ = recorded(scenario)
    kit = check_adapter(ClaudeCodeAtif, tasks, expected=expected(scenario))
    kit.raise_for_errors()


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_a_recorded_runs_steps_sum_to_the_usage_each_turn_reported(scenario):
    _, usage = recorded(scenario)
    final = replay(scenario).document["final_metrics"]
    reported = [turn or {} for turn in usage]
    for metric, field in (("total_prompt_tokens", "input_tokens"), ("total_completion_tokens", "output_tokens"),
                          ("total_cached_tokens", "cache_read_tokens")):
        total = sum(turn.get(field) or 0 for turn in reported)
        assert final.get(metric, 0) == total, (metric, final.get(metric), total)
    if all(turn.get("cost_usd") is not None for turn in reported):
        assert math.isclose(final["total_cost_usd"], sum(turn["cost_usd"] for turn in reported))


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_every_model_call_of_a_recorded_run_is_a_step_with_its_transcripts_time_and_usage(scenario):
    for trajectory in _trajectories(replay(scenario).document):
        for step in trajectory["steps"]:
            if step["source"] == "agent" and step.get("llm_call_count") != 0:
                assert step.get("timestamp") and "completion_tokens" in step.get("metrics", {}), step["step_id"]


def test_the_recordings_cover_what_the_adapter_maps():
    documents = {scenario: replay(scenario).document for scenario in SCENARIOS}
    assert documents["subagent"]["subagent_trajectories"][0]["agent"]["tool_definitions"]
    assert any(part.get("type") == "image" for step in documents["browser"]["steps"]
               for result in (step.get("observation") or {}).get("results") or ()
               for part in result.get("content") if isinstance(part, dict))
    retried = [step for step in documents["retried"]["steps"] if step["source"] == "agent"]
    assert any(step.get("extra", {}).get("agentenv", {}).get("attempt") == 2 for step in retried)
    assert any("error" in step.get("extra", {}).get("agentenv", {}) for step in documents["api-error"]["steps"])
    assert any(step.get("extra", {}).get("agentenv", {}).get("reasoning") for step in documents["thinking"]["steps"])


def test_from_atif_reads_harbors_own_conversion_of_a_recorded_session():
    harbor = json.loads((FIXTURES / "harbor" / "subagent.harbor.json").read_text())
    tasks, _ = recorded("subagent")
    kit = check_adapter(ClaudeCodeAtif, tasks, from_atif=[harbor])
    assert not [finding for finding in kit.findings if finding.rule == "kit-from-atif"]
    assert ClaudeCodeAtif.from_atif(harbor).log


def _regenerate() -> None:
    files = _tool_files()
    for scenario in SCENARIOS:
        document = replay(scenario).document
        for trajectory in _trajectories(document):
            definitions = trajectory["agent"].get("tool_definitions")
            named = next((name for name, converted in files.items() if converted == definitions), None)
            if named is not None:
                trajectory["agent"]["tool_definitions"] = {"shared": named}
        (FIXTURES / f"{scenario}.expected.json").write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    _regenerate()
