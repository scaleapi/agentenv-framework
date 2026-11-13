"""context_facts_for_judge in a bundle run: the bundle writes a path naming one of its entities with the id it gives
that entity, in brackets, and the judge's fact resolves through it as written."""
from __future__ import annotations

import pytest

from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import (
    CONTEXT_FACT_NOT_AVAILABLE,
    RubricsVerifierTaskStep,
    resolve_context_facts,
)

ENV = "@local/~/runs/pn/world"
FACTS = {"wf-1 status": f"env_trigger_state[{ENV}].statuses.wf-1"}


def _resolved(metadata: dict, facts: dict = FACTS) -> str:
    return next(iter(dict(resolve_context_facts(facts, metadata)).values()))


def _state(key: str, status: str) -> dict:
    return {key: {"statuses": {"wf-1": status}}}


def _verifier(facts: dict) -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(
        id="validity", version=1, criteria=[{"id": "world", "description": "the events fired"}], prompt_id="p1",
        use_agent_judge=False, use_trajectory=True, verifier_id="validity", context_facts_for_judge=facts,
    )


def test_a_bracketed_id_is_one_key():
    assert _resolved({"env_trigger_state": _state(ENV, "armed")}) == '"armed"'


def test_a_path_resolves_as_written_only():
    metadata = {"entity_ids": {"world": ENV}, "env_trigger_state": _state(ENV, "armed")}
    assert _resolved(metadata, {"wf-1 status": "env_trigger_state.world.statuses.wf-1"}) == CONTEXT_FACT_NOT_AVAILABLE


def test_a_string_that_isnt_a_path_resolves_to_nothing():
    assert _resolved({"a": 1}, {"fact": "a b"}) == CONTEXT_FACT_NOT_AVAILABLE


@pytest.mark.parametrize("path", [FACTS["wf-1 status"], "verifications.cart-check.count"])
def test_the_step_keeps_a_path_with_or_without_a_bracketed_id(path):
    step = _verifier({"fact": path})
    assert RubricsVerifierTaskStep.from_dict(step.to_dict()).context_facts_for_judge == {"fact": path}


@pytest.mark.parametrize("path", ["env_trigger_state.@local/x.statuses", "a..b", "a[b"])
def test_the_step_refuses_anything_else(path):
    with pytest.raises(ValueError, match="context_facts_for_judge"):
        _verifier({"fact": path})


@pytest.mark.asyncio
async def test_the_judge_sees_the_fact_in_a_bundle_run(monkeypatch):
    v = _verifier(FACTS)
    captured = {}

    async def fake_run(*, eval_prompt, **kwargs):
        captured["eval_prompt"] = eval_prompt
        return [{**v.criteria[0], "score": 1.0, "result": True}], 0, [], None

    monkeypatch.setattr(v, "_read_trajectory_text", lambda uri: "EVENTS")
    monkeypatch.setattr(v, "_run_judge_with_output_retries", fake_run)
    ctx = TaskStepContext(
        prompt_responses=[PromptResponse(prompt_id="p1", response="done", prompt_text="do it",
                                         agent_trajectory_object_url="s3://t/1.json")],
        metadata={"env_trigger_state": _state(ENV, "armed")},
    )

    await v.execute(ctx)

    assert '- wf-1 status: "armed"' in captured["eval_prompt"]
