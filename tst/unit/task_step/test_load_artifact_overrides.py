"""Unit tests for per-run ``step_overrides`` on LoadArtifactTaskStep.

The seam is generic (keyed off the step's own id), so these tests use neutral
ids. ``_resolve_inputs`` must honor an override (swap artifacts / urls /
destination for one run) while preserving stored / collected-wiring behavior
when none is set.
"""

from __future__ import annotations

import pytest

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep


def _ctx(
    step_params: dict | None = None,
    collected: dict | None = None,
    snapshotted: dict | None = None,
) -> TaskStepContext:
    metadata: dict = {}
    if step_params is not None:
        metadata["user_overrides"] = {"step_params": step_params}
    if collected is not None:
        metadata["collected_artifacts"] = collected
    if snapshotted is not None:
        metadata["env_snapshotted_universes"] = snapshotted
    return TaskStepContext(metadata=metadata)


def _load_step(**kw) -> LoadArtifactTaskStep:
    kw.setdefault("id", "load")
    kw.setdefault("version", None)
    kw.setdefault("agent_name", "agent")
    kw.setdefault("artifact_id", "stored-artifact")
    kw.setdefault("destination_path", "/dest")
    return LoadArtifactTaskStep(**kw)


class TestResolveInputs:
    def test_no_override_uses_stored_values(self):
        res = _load_step()._resolve_inputs(_ctx())
        assert res.artifacts == [{"id": "stored-artifact", "version": None}]
        assert res.urls == []
        assert res.destination_path == "/dest"

    def test_override_artifacts_list(self):
        ctx = _ctx({"load": {"artifacts": [{"id": "other-artifact", "version": 3}]}})
        res = _load_step()._resolve_inputs(ctx)
        assert res.artifacts == [{"id": "other-artifact", "version": 3}]
        assert res.destination_path == "/dest"  # unrelated field left alone

    def test_override_artifact_id_and_version(self):
        ctx = _ctx({"load": {"artifact_id": "other-artifact", "artifact_version": 2}})
        res = _load_step()._resolve_inputs(ctx)
        assert res.artifacts == [{"id": "other-artifact", "version": 2}]

    def test_override_destination_path(self):
        ctx = _ctx({"load": {"destination_path": "/other"}})
        res = _load_step()._resolve_inputs(ctx)
        assert res.destination_path == "/other"
        assert res.artifacts == [{"id": "stored-artifact", "version": None}]  # unchanged

    def test_override_urls(self):
        step = LoadArtifactTaskStep(
            id="load", version=None, agent_name="agent",
            urls=["https://old/a"], destination_path="/tmp",
        )
        ctx = _ctx({"load": {"urls": ["https://new/x", "https://new/y"]}})
        res = step._resolve_inputs(ctx)
        assert res.urls == ["https://new/x", "https://new/y"]
        assert res.artifacts == []

    def test_override_urls_none_clears_list(self):
        # {"urls": null} in the JSON override must clear the list, not crash on list(None)
        step = LoadArtifactTaskStep(id="load", version=None, agent_name="agent", urls=["https://old/a"])
        res = step._resolve_inputs(_ctx({"load": {"urls": None}}))
        assert res.urls == []

    def test_override_artifact_id_null_raises(self):
        # {"artifact_id": null} is malformed — fail clear, not with an opaque Artifact.get(None) error
        with pytest.raises(ValueError, match="artifact_id"):
            _load_step()._resolve_inputs(_ctx({"load": {"artifact_id": None}}))

    def test_artifact_override_wins_over_collected_wiring(self):
        step = LoadArtifactTaskStep(
            id="load", version=None, agent_name="agent", collected_artifacts_step_id="collect",
        )
        ctx = _ctx(
            step_params={"load": {"artifact_id": "explicit"}},
            collected={"collect": {"file_artifact_universe": {"id": "from-collect", "version": 1}}},
        )
        res = step._resolve_inputs(ctx)
        assert res.artifacts == [{"id": "explicit", "version": None}]

    def test_collected_wiring_used_when_no_artifact_override(self):
        step = LoadArtifactTaskStep(
            id="load", version=None, agent_name="agent", collected_artifacts_step_id="collect",
        )
        ctx = _ctx(collected={"collect": {"file_artifact_universe": {"id": "from-collect", "version": 1}}})
        res = step._resolve_inputs(ctx)
        assert res.artifacts == [{"id": "from-collect", "version": 1}]

    def test_collected_missing_entry_still_raises(self):
        step = LoadArtifactTaskStep(
            id="load", version=None, agent_name="agent", collected_artifacts_step_id="collect",
        )
        with pytest.raises(RuntimeError, match="No artifact recorded by step 'collect'"):
            step._resolve_inputs(_ctx())


class TestArtifactFromStepWiring:
    """`artifact_from_step_id` — an artifact minted mid-run can't be named
    statically, so config points at the step that produced it. The step is
    producer-agnostic: it resolves whatever a prior step recorded, without
    knowing which kind of step that was."""

    def test_resolves_a_snapshot_step_output(self):
        step = LoadArtifactTaskStep(
            id="load", version=None, agent_name="agent", artifact_from_step_id="t-snap",
        )
        ctx = _ctx(snapshotted={"t-snap": {"id": "snapshot-env-abc", "version": 3}})
        assert step._resolve_inputs(ctx).artifacts == [{"id": "snapshot-env-abc", "version": 3}]

    def test_resolves_a_collect_step_output(self):
        """Same param, different producer — no snapshot-specific surface."""
        step = LoadArtifactTaskStep(
            id="load", version=None, agent_name="agent", artifact_from_step_id="collect",
        )
        ctx = _ctx(collected={"collect": {"file_artifact_universe": {"id": "from-collect", "version": 1}}})
        assert step._resolve_inputs(ctx).artifacts == [{"id": "from-collect", "version": 1}]

    def test_producer_that_made_nothing_skips_the_load(self):
        step = LoadArtifactTaskStep(
            id="load", version=None, agent_name="agent", artifact_from_step_id="collect",
        )
        ctx = _ctx(collected={"collect": {"file_artifact_universe": None}})
        assert step._resolve_inputs(ctx).artifacts == []

    def test_env_target_is_allowed(self):
        """Restoring into an env is a legitimate target, unlike the legacy
        `collected_artifacts_step_id` which is agent/container-only."""
        step = LoadArtifactTaskStep(
            id="load", version=None, env_id="multi", artifact_from_step_id="t-snap",
        )
        ctx = _ctx(snapshotted={"t-snap": {"id": "snapshot-env-abc", "version": 3}})
        assert step._resolve_inputs(ctx).artifacts == [{"id": "snapshot-env-abc", "version": 3}]

    @pytest.mark.parametrize(
        "conflicting",
        [
            {"artifact_id": "explicit"},
            {"urls": ["https://x/y"]},
            {"collected_artifacts_step_id": "collect"},
        ],
    )
    def test_mutually_exclusive_with_other_sources(self, conflicting):
        with pytest.raises(ValueError, match="artifact_from_step_id"):
            LoadArtifactTaskStep(
                id="load", version=None, agent_name="agent",
                artifact_from_step_id="t-snap", **conflicting,
            )

    def test_survives_a_to_dict_from_dict_roundtrip(self):
        step = LoadArtifactTaskStep(
            id="load", version=None, agent_name="agent", artifact_from_step_id="t-snap",
        )
        assert LoadArtifactTaskStep.from_dict(step.to_dict()).artifact_from_step_id == "t-snap"

    def test_legacy_collected_param_still_resolves(self):
        """Persisted task-step docs carry `collected_artifacts_step_id`; it must
        keep working through the generic resolver."""
        step = LoadArtifactTaskStep(
            id="load", version=None, agent_name="agent", collected_artifacts_step_id="collect",
        )
        ctx = _ctx(collected={"collect": {"file_artifact_universe": {"id": "from-collect", "version": 1}}})
        assert step._resolve_inputs(ctx).artifacts == [{"id": "from-collect", "version": 1}]
        assert LoadArtifactTaskStep.from_dict(step.to_dict()).collected_artifacts_step_id == "collect"


class TestStepParamOverridesSeam:
    def test_returns_only_this_steps_params(self):
        """Keying: an unset seam yields {}, and an override leaks only to its own step id."""
        assert _load_step().step_param_overrides(_ctx()) == {}
        ctx = _ctx({"load": {"artifact_id": "x"}, "other": {"artifact_id": "y"}})
        assert _load_step().step_param_overrides(ctx) == {"artifact_id": "x"}
