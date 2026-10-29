"""`task run` shows a snapshot_agent_state step its own capture, never another writer's row."""

from __future__ import annotations

from types import SimpleNamespace

from agent_env.cli.task.run import _step_output_lines
from agent_env.task_step.context import TaskStepContext

_STEP = SimpleNamespace(type="snapshot_agent_state", id="capture")


def _context(*rows: dict) -> TaskStepContext:
    return TaskStepContext(metadata={"agent_snapshots": list(rows)})


def test_the_steps_own_capture_is_shown_over_a_later_row():
    lines = _step_output_lines(_STEP, _context(
        {"id": "own", "version": 2, "capture_step_id": "capture"},
        {"id": "series-row", "version": 5, "capture_step_id": "ask"},
    ))

    assert len(lines) == 1
    assert "artifact_id: own" in lines[0]


def test_nothing_is_shown_without_a_row_from_the_step():
    assert _step_output_lines(_STEP, _context({"id": "series-row", "capture_step_id": "ask"}, {"id": "untagged"})) == []
