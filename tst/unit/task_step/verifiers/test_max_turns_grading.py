"""A max-turns budget exhaustion records an ``error_type``, but its partial work is
still gradable. Verifiers skip responses with an ``error_type`` *unless* the error is
a ``MAX_TURNS_ERROR_CODE`` budget outcome — these tests pin that carve-out so a
budget-exhausted run reaches grading instead of being silently skipped."""

import pytest

from agent_env.a2a_agent.protocol import MAX_TURNS_ERROR_CODE
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.agent_prompt_response_verifier import (
    AgentPromptResponseVerifierTaskStep,
)


def test_is_gradable_despite_error_only_for_max_turns():
    assert PromptResponse(
        prompt_id="p1", response="x", error_type="agent_error", error_code=MAX_TURNS_ERROR_CODE,
    ).is_gradable_despite_error
    assert not PromptResponse(
        prompt_id="p1", response="x", error_type="agent_error", error_code="some_crash",
    ).is_gradable_despite_error
    assert not PromptResponse(prompt_id="p1", response="x").is_gradable_despite_error


def _context_with_response(**kw) -> TaskStepContext:
    ctx = TaskStepContext(instance_id="ti-1")
    ctx.prompt_responses.append(PromptResponse(prompt_id="p1", response="the answer is 42", **kw))
    return ctx


def _verifier() -> AgentPromptResponseVerifierTaskStep:
    return AgentPromptResponseVerifierTaskStep(
        id="v", version=None, prompt_id="p1", verifier_id="vid",
        criteria=[{"type": "response_contains", "needles": ["42"]}],
    )


@pytest.mark.asyncio
async def test_max_turns_error_reaches_grading():
    # A budget-exhausted response carries error_type but is gradable: the verifier
    # must evaluate its criteria against the partial response, not skip it.
    ctx = _context_with_response(error_type="agent_error", error_code=MAX_TURNS_ERROR_CODE)

    result = await _verifier().execute(ctx)

    entry = result.metadata["verifications"]["vid"]
    assert entry["results"][0].get("id") != "prompt_error"  # not the skipped sentinel
    assert entry["results"][0]["result"] is True
    assert entry["score"] == 1.0


@pytest.mark.asyncio
async def test_other_error_type_still_skips_grading():
    # A real crash (any other error_code) must still short-circuit to a skip.
    ctx = _context_with_response(error_type="agent_error", error_code="some_crash")

    result = await _verifier().execute(ctx)

    entry = result.metadata["verifications"]["vid"]
    assert entry["results"][0]["id"] == "prompt_error"
    assert entry["score"] == 0
