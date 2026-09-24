"""Unit tests for compact_otel_trajectory's ``result_file_prefix`` param.

The prefix namespaces externalized tool-result filenames so a caller compacting
several trajectories into one destination (e.g. per-turn files in one judge
container dir) doesn't get colliding ``tool_call_result_N.json`` names. The path
is built in one place and reused for both the returned file and the event's
``output_file`` reference, so they can't drift.
"""
from __future__ import annotations

import json

from agent_env.task_step.task_steps.verifiers.judge_utils.trajectory_filter import (
    _TOOL_RESULT_FILES_DIR,
    TrajectoryFilter,
    compact_otel_trajectory,
)


def _tool_span(output) -> dict:
    return {
        "name": "some_tool",
        "attributes": {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.prompt": json.dumps({"tool": "some_tool", "input": {}}),
            "gen_ai.completion": json.dumps({"output": output}),
        },
        "start_time": "1",
        "end_time": "2",
    }


def test_result_file_prefix_namespaces_externalized_files():
    spans = [_tool_span({"result": "x" * 2000})]      # >1KB -> externalized
    events, files = compact_otel_trajectory(spans, TrajectoryFilter(), result_file_prefix="turn_02_")

    assert files, "large tool result should be externalized"
    path = files[0][0]
    assert path == f"{_TOOL_RESULT_FILES_DIR}/turn_02_tool_call_result_1.json"
    ref = next(e["output_file"] for e in events if e.get("type") == "tool_result")
    assert ref == path, "in-event reference must match the externalized filename"


def test_result_file_prefix_default_is_backward_compatible():
    spans = [_tool_span({"result": "x" * 2000})]
    _, files = compact_otel_trajectory(spans, TrajectoryFilter())      # no prefix
    assert files[0][0] == f"{_TOOL_RESULT_FILES_DIR}/tool_call_result_1.json"
