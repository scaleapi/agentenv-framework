"""An agent's recorded atif/1 runs (``tst/data/<agent>_atif``), recorded through its image on the pinned CLI with
atif/1 prescribed: each scenario's turns as the agent served them, its native events, its aux log and its turn
context, replayed through its adapter as the SDK ran it live; and the checks every agent's recorded runs share."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from agentenv_protocol.atif import KitResult, RecordedTask, check_adapter


class Recordings:
    """The recordings in ``folder``, replayed through ``adapter``. ``aux`` turns a fixture's aux entry into the one the
    agent fed, ``expected`` a checked-in document into the one a replay writes, and ``saved`` the other way, for an
    agent whose fixtures keep what every run shares, its tools say, in files of their own."""

    def __init__(self, folder: Path, adapter: type, *, aux: Callable[[dict], dict] = lambda entry: entry,
                 expected: Callable[[dict], dict] = lambda document: document,
                 saved: Callable[[dict], dict] = lambda document: document) -> None:
        self.folder, self.adapter = folder, adapter
        self._aux, self._expected, self._saved = aux, expected, saved
        self.scenarios = sorted(path.stem for path in folder.glob("*.json") if not path.name.endswith(".expected.json"))

    def data(self, scenario: str) -> dict:
        return json.loads((self.folder / f"{scenario}.json").read_text())

    def recorded(self, scenario: str) -> tuple[list[RecordedTask], list[dict | None]]:
        """The scenario's turns as recorded tasks, and the usage each turn reported."""
        turns = self.data(scenario)["turns"]
        return ([RecordedTask.from_served(turn["turn"], turn["native"], [self._aux(entry) for entry in turn["aux"]])
                 for turn in turns], [turn["usage"] for turn in turns])

    def expected(self, scenario: str) -> dict:
        return self._expected(json.loads((self.folder / f"{scenario}.expected.json").read_text()))

    def replay(self, scenario: str) -> KitResult:
        return check_adapter(self.adapter, self.recorded(scenario)[0])

    def check(self, scenario: str) -> None:
        """The scenario replays to its checked-in document under the live rule."""
        check_adapter(self.adapter, self.recorded(scenario)[0], expected=self.expected(scenario)).raise_for_errors()

    def regenerate(self) -> None:
        """Write each scenario's expected document again, after an intended change."""
        for scenario in self.scenarios:
            run = self.replay(scenario)
            run.raise_for_errors()
            document = self._saved(run.document)
            (self.folder / f"{scenario}.expected.json").write_text(json.dumps(document, indent=1, ensure_ascii=False)
                                                                  + "\n")
            print("wrote", scenario)


def trajectories(document: dict) -> list[dict]:
    """The document's own trajectory and each embedded sub-agent's."""
    return [document, *(document.get("subagent_trajectories") or [])]


def agent_steps(trajectory: dict) -> list[dict]:
    return [step for step in trajectory["steps"] if step["source"] == "agent"]


def assert_steps_sum_to_reported_usage(document: dict, usage: list[dict | None]) -> None:
    """The copy's totals are the turns' reported usage, and its cost theirs when every turn reported one."""
    final, reported = document["final_metrics"], [turn or {} for turn in usage]
    for metric, field in (("total_prompt_tokens", "input_tokens"), ("total_completion_tokens", "output_tokens"),
                          ("total_cached_tokens", "cache_read_tokens")):
        total = sum(turn.get(field) or 0 for turn in reported)
        assert final.get(metric, 0) == total, (metric, final.get(metric), total)
    if all(turn.get("cost_usd") is not None for turn in reported):
        assert math.isclose(final["total_cost_usd"], sum(turn["cost_usd"] for turn in reported))


def assert_every_model_call_has_its_time_and_usage(document: dict, *, model: bool = True) -> None:
    """Each step that made a model call, a sub-agent's included, has its time, its completion tokens and, with
    ``model``, the model it called."""
    for trajectory in trajectories(document):
        for step in agent_steps(trajectory):
            if step.get("llm_call_count") != 0:
                assert step.get("timestamp") and "completion_tokens" in step.get("metrics", {}), step["step_id"]
                assert not model or step.get("model_name"), step["step_id"]


def assert_every_call_has_its_result(document: dict) -> None:
    for trajectory in trajectories(document):
        for step in agent_steps(trajectory):
            answered = {result["source_call_id"] for result in (step.get("observation") or {}).get("results") or ()}
            assert {call["tool_call_id"] for call in step.get("tool_calls") or ()} <= answered, step["step_id"]


def images(document: dict) -> list[dict[str, Any]]:
    """Every image the document's results hold, a sub-agent's included."""
    return [part for trajectory in trajectories(document) for step in agent_steps(trajectory)
            for result in (step.get("observation") or {}).get("results") or ()
            for part in (result.get("content") if isinstance(result.get("content"), list) else ())
            if isinstance(part, dict) and part.get("type") == "image"]
