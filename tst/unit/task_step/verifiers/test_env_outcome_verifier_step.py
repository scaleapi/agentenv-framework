"""EnvOutcomeVerifierTaskStep constructor accepts the JSON's raw aggregator string (the `task create` path)."""
import pytest

from agent_env.task_step.task_steps.verifiers.env_outcome_verifier import EnvOutcomeVerifierTaskStep, ScoreAggregator


def test_string_aggregator_is_coerced_and_round_trips():
    step = EnvOutcomeVerifierTaskStep(id="g", version=None, env_id="e", file_artifact_id="f", score_aggregator="weighted_average")
    assert step.score_aggregator is ScoreAggregator.WEIGHTED_AVERAGE
    assert step.to_dict()["score_aggregator"] == "weighted_average"
    assert EnvOutcomeVerifierTaskStep.from_dict(step.to_dict()).score_aggregator is ScoreAggregator.WEIGHTED_AVERAGE
    assert EnvOutcomeVerifierTaskStep(id="g", version=None, env_id="e", file_artifact_id="f").score_aggregator is ScoreAggregator.ALL_PASS


def test_unknown_aggregator_is_rejected_at_construction():
    with pytest.raises(ValueError):
        EnvOutcomeVerifierTaskStep(id="g", version=None, env_id="e", file_artifact_id="f", score_aggregator="nope")


def test_put_accepts_an_existing_artifact_reference(monkeypatch):
    from agent_env.task_step import store as step_store

    saved = {}

    class _Store:
        def put_document(self, instance):
            saved["instance"] = instance
            return instance
    monkeypatch.setattr(step_store, "get_task_step_store", lambda: _Store())
    step = EnvOutcomeVerifierTaskStep.put(id="g", env_id="e", file_artifact_id="gsuite-m1-gates-verifier",
                                          file_artifact_version=1, verifier_id="gates", score_aggregator="all_pass")
    assert saved["instance"] is step
    assert step.file_artifact_id == "gsuite-m1-gates-verifier" and step.file_artifact_version == 1
    assert step.score_aggregator is ScoreAggregator.ALL_PASS and step.version is None


def test_put_with_a_script_path_uploads_then_stores(monkeypatch, tmp_path):
    from agent_env import artifact as artifact_mod
    from agent_env.task_step import store as step_store

    script = tmp_path / "v.py"
    script.write_text("async def verify(url):\n    return []\n")
    uploads = {}

    class _Artifact:
        id, version = "g-verifier-script", 3

    class _FA:
        @staticmethod
        def put(**kw):
            uploads.update(kw)
            return _Artifact()

    class _Store:
        def put_document(self, instance):
            return instance
    monkeypatch.setattr(artifact_mod, "FileArtifact", _FA)
    monkeypatch.setattr(step_store, "get_task_step_store", lambda: _Store())
    step = EnvOutcomeVerifierTaskStep.put(id="g", env_id="e", verify_script_file_path=str(script))
    assert uploads["id"] == "g-verifier-script" and uploads["file_path"] == str(script)
    assert step.file_artifact_id == "g-verifier-script" and step.file_artifact_version == 3
