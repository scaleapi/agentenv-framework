"""AGENT_ENV_ERRORED_TURN_UNGRADED: a verifier whose agent turn errored leaves the run ungraded, failing its own step
with AgentTurnError, instead of recording a score of 0 that counts as the agent failing the task."""
from __future__ import annotations

import pytest

from agent_env.task import Task
from agent_env.task_step.context import ERRORED_TURN_UNGRADED_ENV_VAR, PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.agent_prompt_response_verifier import (
    AgentPromptResponseVerifierTaskStep,
)
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep


def _rubrics(monkeypatch) -> RubricsVerifierTaskStep:
    step = RubricsVerifierTaskStep(
        id="grade", version=1, criteria=[{"id": "done", "description": "it is done"}], prompt_id="p1",
        use_agent_judge=False, use_trajectory=True, verifier_id="grade", fail_task_on_error=False)

    async def judge(**kwargs):
        return [{"id": "done", "description": "it is done", "score": 1.0, "result": True}], 0, [], None

    monkeypatch.setattr(step, "_read_trajectory_text", lambda uri: "EVENTS")
    monkeypatch.setattr(step, "_run_judge_with_output_retries", judge)
    return step


def _response(monkeypatch) -> AgentPromptResponseVerifierTaskStep:
    return AgentPromptResponseVerifierTaskStep(
        id="grade", version=None, prompt_id="p1", criteria=[{"type": "response_contains", "needles": ["done"]}],
        verifier_id="grade", fail_task_on_error=False)


VERIFIERS = pytest.mark.parametrize("make_verifier", [_rubrics, _response], ids=["rubrics", "prompt-response"])


def _context(error_type: str | None) -> TaskStepContext:
    return TaskStepContext(prompt_responses=[PromptResponse(
        prompt_id="p1", response="done", prompt_text="do it", agent_trajectory_object_url="s3://t/1.json",
        error_type=error_type)])


@VERIFIERS
@pytest.mark.asyncio
async def test_off_an_errored_turn_scores_zero(monkeypatch, local_stores, make_verifier):
    monkeypatch.delenv(ERRORED_TURN_UNGRADED_ENV_VAR, raising=False)
    ctx = _context("model_key_budget")

    await Task(id="t", version=None, steps=[make_verifier(monkeypatch)]).run(context=ctx)

    entry = ctx.metadata["verifications"]["grade"]
    assert (entry["score"], entry["results"][0]["id"]) == (0, "prompt_error")
    assert "failed_steps" not in ctx.metadata


@VERIFIERS
@pytest.mark.asyncio
async def test_on_an_errored_turn_is_left_ungraded_and_its_verifier_fails(monkeypatch, local_stores, make_verifier):
    monkeypatch.setenv(ERRORED_TURN_UNGRADED_ENV_VAR, "1")
    ctx = _context("model_key_budget")

    await Task(id="t", version=None, steps=[make_verifier(monkeypatch)]).run(context=ctx)

    assert "grade" not in ctx.metadata.get("verifications", {})
    [failed] = ctx.metadata["failed_steps"]
    assert (failed["step_id"], failed["error_type"]) == ("grade", "AgentTurnError")
    assert "error_type=model_key_budget" in failed["error"]


@VERIFIERS
@pytest.mark.parametrize("platform_error", ["infra_error", "model_key_auth"])
@pytest.mark.asyncio
async def test_on_other_platform_failures_are_left_ungraded(monkeypatch, local_stores, make_verifier, platform_error):
    monkeypatch.setenv(ERRORED_TURN_UNGRADED_ENV_VAR, "1")
    ctx = _context(platform_error)

    await Task(id="t", version=None, steps=[make_verifier(monkeypatch)]).run(context=ctx)

    assert "grade" not in ctx.metadata.get("verifications", {})
    [failed] = ctx.metadata["failed_steps"]
    assert f"error_type={platform_error}" in failed["error"]


@VERIFIERS
@pytest.mark.asyncio
async def test_on_an_agent_error_still_scores_zero(monkeypatch, local_stores, make_verifier):
    monkeypatch.setenv(ERRORED_TURN_UNGRADED_ENV_VAR, "1")
    ctx = _context("agent_error")

    await Task(id="t", version=None, steps=[make_verifier(monkeypatch)]).run(context=ctx)

    entry = ctx.metadata["verifications"]["grade"]
    assert (entry["score"], entry["results"][0]["id"]) == (0, "prompt_error")
    assert "failed_steps" not in ctx.metadata


@VERIFIERS
@pytest.mark.asyncio
async def test_on_a_turn_without_an_error_is_graded(monkeypatch, local_stores, make_verifier):
    monkeypatch.setenv(ERRORED_TURN_UNGRADED_ENV_VAR, "1")
    ctx = _context(None)

    await Task(id="t", version=None, steps=[make_verifier(monkeypatch)]).run(context=ctx)

    assert ctx.metadata["verifications"]["grade"]["score"] == 1.0
    assert "failed_steps" not in ctx.metadata
