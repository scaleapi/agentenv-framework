"""context_facts_for_judge in a bundle run: a path naming an entity as the task wrote it also finds the key under the
id the run's task gave it (``entity_ids``), as a bundle-written task's run records."""
from __future__ import annotations

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import (
    CONTEXT_FACT_NOT_AVAILABLE,
    RubricsVerifierTaskStep,
    resolve_context_facts,
)

ENV = "@local/~/runs/pn/world"
FACTS = {"wf-1 status": "env_trigger_state.world.statuses.wf-1"}


def _resolved(metadata: dict, facts: dict = FACTS) -> str:
    return next(iter(dict(resolve_context_facts(facts, metadata)).values()))


def _state(key: str, status: str) -> dict:
    return {key: {"statuses": {"wf-1": status}}}


def test_a_renamed_env_is_found_by_the_id_the_task_names_it_by():
    assert _resolved({"entity_ids": {"world": ENV}, "env_trigger_state": _state(ENV, "armed")}) == '"armed"'


def test_a_renamed_step_output_is_found_too():
    snapshot = "@local/~/runs/pn/task-a/snap"
    metadata = {"entity_ids": {"snap": snapshot}, "env_changes": {snapshot: {"rows": 3}}}
    assert _resolved(metadata, {"rows": "env_changes.snap.rows"}) == "3"


def test_a_key_as_written_still_wins():
    state = {**_state("world", "fired"), **_state(ENV, "armed")}
    assert _resolved({"entity_ids": {"world": ENV}, "env_trigger_state": state}) == '"fired"'


@pytest.mark.parametrize("entity_ids", [None, {}, {"other": ENV}], ids=["no map", "empty map", "not in the map"])
def test_without_a_rename_a_key_resolves_only_as_written(entity_ids):
    metadata = {"env_trigger_state": _state(ENV, "fired"), **({"entity_ids": entity_ids} if entity_ids is not None else {})}
    assert _resolved(metadata) == CONTEXT_FACT_NOT_AVAILABLE


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
        metadata={"entity_ids": {"world": ENV}, "env_trigger_state": _state(ENV, "armed")},
    )

    await v.execute(ctx)

    assert '- wf-1 status: "armed"' in captured["eval_prompt"]
