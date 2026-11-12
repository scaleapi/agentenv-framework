"""context_facts_for_judge in a bundle run: a path naming an env as the task wrote it also finds the key the bundle
namespaced, ``<bundle id root>/<id>``, within the bundle the run's task came from."""
from __future__ import annotations

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import (
    CONTEXT_FACT_NOT_AVAILABLE,
    RubricsVerifierTaskStep,
    resolve_context_facts,
)

FACTS = {"wf-1 status": "env_trigger_state.world.statuses.wf-1"}


def _resolved(metadata: dict) -> str:
    return dict(resolve_context_facts(FACTS, metadata))["wf-1 status"]


def _state(key: str, status: str) -> dict:
    return {key: {"statuses": {"wf-1": status}}}


@pytest.mark.parametrize("task_id, key", [
    ("@local/~/runs/pn/task-a", "@local/~/runs/pn/world"),
    ("@hf/org/repo/pn/task-a", "@hf/org/repo/pn/world"),
])
def test_a_bundle_run_finds_the_key_its_bundle_namespaced(task_id, key):
    assert _resolved({"task_id": task_id, "env_trigger_state": _state(key, "armed")}) == '"armed"'


def test_a_key_as_written_still_wins():
    state = {**_state("world", "fired"), **_state("@local/b/world", "armed")}
    assert _resolved({"task_id": "@local/b/task-a", "env_trigger_state": state}) == '"fired"'


def test_another_bundles_key_is_not_found():
    assert _resolved({"task_id": "@local/b/task-a", "env_trigger_state": _state("@local/other/world", "fired")}) \
        == CONTEXT_FACT_NOT_AVAILABLE


@pytest.mark.parametrize("metadata", [{"task_id": "task-a"}, {}], ids=["store task", "no task id"])
def test_a_task_from_the_store_resolves_only_keys_as_written(metadata):
    assert _resolved({**metadata, "env_trigger_state": _state("@local/b/world", "fired")}) == CONTEXT_FACT_NOT_AVAILABLE


@pytest.mark.asyncio
async def test_the_judge_sees_the_fact_in_a_bundle_run(monkeypatch):
    v = RubricsVerifierTaskStep(
        id="validity", version=1, criteria=[{"id": "world", "description": "the events fired"}], prompt_id="p1",
        use_agent_judge=False, use_trajectory=True, verifier_id="validity", context_facts_for_judge=FACTS,
    )
    captured = {}

    async def fake_run(*, eval_prompt, **kwargs):
        captured["eval_prompt"] = eval_prompt
        return [{**v.criteria[0], "score": 1.0, "result": True}], 0, [], None

    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: "EVENTS")
    monkeypatch.setattr(v, "_run_judge_with_output_retries", fake_run)
    ctx = TaskStepContext(
        prompt_responses=[PromptResponse(prompt_id="p1", response="done", prompt_text="do it",
                                         agent_trajectory_object_url="s3://t/1.json")],
        metadata={"task_id": "@local/~/runs/pn/task-a", "env_trigger_state": _state("@local/~/runs/pn/world", "armed")},
    )

    await v.execute(ctx)

    assert '- wf-1 status: "armed"' in captured["eval_prompt"]
