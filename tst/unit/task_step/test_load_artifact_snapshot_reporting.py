"""load_artifact carries the snapshot decision into the task doc and the result back out.

`snapshot_after_load` has to survive to_dict/from_dict because the step is persisted to
Mongo and rehydrated by the Temporal worker — a field that round-trips to None silently
turns the bake off for every task that asked for it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep


def _step(**kwargs) -> LoadArtifactTaskStep:
    return LoadArtifactTaskStep(
        id="load-universe", version=1, env_id="env-1", artifact_id="uni", **kwargs
    )


@pytest.mark.parametrize("value", [True, False, None])
def test_snapshot_after_load_round_trips(value):
    restored = LoadArtifactTaskStep.from_dict(_step(snapshot_after_load=value).to_dict())
    assert restored.snapshot_after_load is value


def test_defaults_to_none_so_the_worker_flag_decides():
    """A task doc shouldn't hard-code a cost decision an operator may want to flip."""
    assert _step().snapshot_after_load is None
    assert LoadArtifactTaskStep.from_dict({
        "id": "load-universe", "version": 1, "env_id": "env-1", "artifact_id": "uni",
    }).snapshot_after_load is None


def _context() -> TaskStepContext:
    return TaskStepContext()


def test_records_a_snapshot_restore():
    context = _context()
    LoadArtifactTaskStep._record_universe_load(
        context,
        SimpleNamespace(id="uni", version=39),
        SimpleNamespace(restored_from_snapshot=True, snapshot_db_image_artifact_id="snap-img",
                        snapshot_baked=None, snapshot_bake_error=None),
    )
    entry = context.metadata["loaded_environment_universes"][0]
    assert entry == {
        "id": "uni", "version": 39,
        "restored_from_snapshot": True,
        "snapshot_db_image_artifact_id": "snap-img",
    }


def test_records_a_slow_load_that_baked():
    context = _context()
    LoadArtifactTaskStep._record_universe_load(
        context,
        SimpleNamespace(id="uni", version=39),
        SimpleNamespace(restored_from_snapshot=False, snapshot_db_image_artifact_id=None,
                        snapshot_baked=True, snapshot_bake_error=None),
    )
    entry = context.metadata["loaded_environment_universes"][0]
    assert entry["restored_from_snapshot"] is False
    assert entry["snapshot_baked"] is True
    assert "snapshot_bake_error" not in entry


def test_records_a_bake_failure_reason():
    context = _context()
    LoadArtifactTaskStep._record_universe_load(
        context,
        SimpleNamespace(id="uni", version=39),
        SimpleNamespace(restored_from_snapshot=False, snapshot_db_image_artifact_id=None,
                        snapshot_baked=False, snapshot_bake_error="RuntimeError: boom"),
    )
    entry = context.metadata["loaded_environment_universes"][0]
    assert entry["snapshot_baked"] is False
    assert entry["snapshot_bake_error"] == "RuntimeError: boom"


def test_multiple_universes_accumulate():
    context = _context()
    for uid in ("uni-a", "uni-b"):
        LoadArtifactTaskStep._record_universe_load(
            context,
            SimpleNamespace(id=uid, version=1),
            SimpleNamespace(restored_from_snapshot=True, snapshot_db_image_artifact_id="img",
                            snapshot_baked=None, snapshot_bake_error=None),
        )
    assert [e["id"] for e in context.metadata["loaded_environment_universes"]] == ["uni-a", "uni-b"]


def test_a_none_result_records_nothing():
    """Env types that return None from the load must not inject a bogus entry."""
    context = _context()
    LoadArtifactTaskStep._record_universe_load(context, SimpleNamespace(id="uni", version=1), None)
    assert "loaded_environment_universes" not in context.metadata
