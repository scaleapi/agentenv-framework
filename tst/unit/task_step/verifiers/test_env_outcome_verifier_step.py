"""EnvOutcomeVerifierTaskStep constructor accepts the JSON's raw aggregator string (the `task create` path)."""
import asyncio
from types import SimpleNamespace

import pytest

from agent_env.artifact import FileArtifact
from agent_env.env.env import DeployedEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.env_outcome_verifier import EnvOutcomeVerifierTaskStep, ScoreAggregator
from tst.unit.event_loop_probe import on_event_loop


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


def test_the_verifier_script_is_read_off_the_event_loop(monkeypatch):
    on_loop: list[bool] = []

    def load():
        on_loop.append(on_event_loop())
        return b"async def verify(url):\n    return [{'id': 'c', 'score': 1.0}]\n"

    monkeypatch.setattr(FileArtifact, "get", staticmethod(lambda id, version=None: SimpleNamespace(load=load)))
    step = EnvOutcomeVerifierTaskStep(id="outcome", version=None, env_id="e", file_artifact_id="script", verifier_id="v")
    ctx = TaskStepContext()
    ctx.deployed_envs.append(DeployedEnv(env_id="e", env_version=1, mcp_url="http://mcp"))

    asyncio.run(step.execute(ctx))

    assert on_loop == [False] and ctx.metadata["verifications"]["v"]["results"] == [{"id": "c", "score": 1.0}]


def test_a_trajectory_parameter_gets_the_graded_prompts_trajectory_per_turn(monkeypatch):
    from agent_env.task_step.context import PromptResponse
    from agent_env.task_step.task_steps.verifiers import env_outcome_verifier

    stored = {"mem://t1": b'{"format": "cli/1", "events": [{"n": 1}]}', "mem://t2": b'[{"n": 2}]'}
    monkeypatch.setattr(env_outcome_verifier, "get_config", lambda: SimpleNamespace(
        get_object_store_at=lambda url: SimpleNamespace(get=stored.__getitem__)))
    script = (b"async def verify(url, trajectory):\n"
              b"    return [{'id': 'c', 'score': 1.0, 'trajectory': trajectory}]\n")
    monkeypatch.setattr(FileArtifact, "get", staticmethod(lambda id, version=None: SimpleNamespace(load=lambda: script)))

    def graded(**urls):
        ctx = TaskStepContext()
        ctx.deployed_envs.append(DeployedEnv(env_id="e", env_version=1, mcp_url="http://mcp"))
        ctx.prompt_responses.append(PromptResponse(prompt_id="p", response="done", **urls))
        step = EnvOutcomeVerifierTaskStep(id="outcome", version=None, env_id="e", file_artifact_id="s", verifier_id="v")
        return asyncio.run(step.execute(ctx)).metadata["verifications"]["v"]["results"][0]["trajectory"]

    assert graded(target_agent_per_turn_trajectory_object_urls=["mem://t1", None, "mem://t2"]) == [
        {"format": "cli/1", "events": [{"n": 1}]}, None, [{"n": 2}]]
    assert graded(agent_trajectory_object_url="mem://t2") == [[{"n": 2}]]
    assert graded() is None


def test_a_script_whose_object_was_replaced_after_its_put_is_never_run(local_stores, tmp_path):
    script = tmp_path / "v.py"
    script.write_text("async def verify(url):\n    return [{'id': 'c', 'score': 1.0}]\n")
    fa = FileArtifact.put("script", description="d", file_path=str(script))
    marker = tmp_path / "ran"
    store = local_stores.get_object_store()
    store.put(store.get_object_key(fa.object_url), f"open({str(marker)!r}, 'w')\n".encode(), allow_overwrite=True)
    step = EnvOutcomeVerifierTaskStep(id="outcome", version=None, env_id="e", file_artifact_id="script", verifier_id="v")
    ctx = TaskStepContext()
    ctx.deployed_envs.append(DeployedEnv(env_id="e", env_version=1, mcp_url="http://mcp"))

    with pytest.raises(ValueError, match="no longer holds the bytes file artifact script v1 was put with"):
        asyncio.run(step.execute(ctx))
    assert not marker.exists()
