"""Unit tests for rubrics verifier judge output retry behavior."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep


def _criteria(ids: list[str]) -> list[dict]:
    return [{"id": cid, "title": f"Test {cid}"} for cid in ids]


def _valid_response(ids: list[str]) -> str:
    return json.dumps({"results": [
        {"id": cid, "score": 1.0, "justification": "ok"} for cid in ids
    ]})


def _verifier(*, use_agent_judge: bool = False) -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(
        id="verify",
        version=1,
        criteria=_criteria(["c1", "c2"]),
        prompt_id="prompt-1",
        use_agent_judge=use_agent_judge,
        use_trajectory=False,
        verifier_id="verifier-test",
    )


@pytest.mark.asyncio
async def test_retries_then_succeeds_on_second_attempt(monkeypatch):
    verifier = _verifier()
    bad = json.dumps({"results": [{"id": "c1", "score": 1.0, "justification": "ok"}]})
    prompts: list[str] = []

    async def fake_prompt_llm_judge(prompt: str, *, model: str, context: TaskStepContext) -> dict:
        prompts.append(prompt)
        if len(prompts) == 1:
            return {"response": bad, "trajectory_object_url": "mem://judge/rejected.json"}
        return {"response": _valid_response(["c1", "c2"]), "trajectory_object_url": "mem://judge/accepted.json"}

    monkeypatch.setattr(verifier, "_prompt_llm_judge", fake_prompt_llm_judge)

    results, retries, discrepancies, judge_trajectory_url = await verifier._run_judge_with_output_retries(
        eval_prompt="Evaluate the agent.",
        criteria=verifier.criteria,
        context=TaskStepContext(),
        model="claude-sonnet-4-6",
        judge_a2a_url=None,
        judge_agent_card={},
        judge_agent=None,
        judge_effort=None,
        judge_max_thinking_tokens=None,
        use_agent_judge=False,
    )

    assert len(results) == 2
    assert retries == 1
    assert judge_trajectory_url == "mem://judge/accepted.json"
    assert len(discrepancies) == 1
    assert discrepancies[0]["missing_ids"] == ["c2"]
    assert len(prompts) == 2
    assert prompts[0] == "Evaluate the agent."
    assert "Correction Required" in prompts[1]
    assert bad in prompts[1]


@pytest.mark.asyncio
async def test_raises_after_max_retries(monkeypatch):
    verifier = _verifier()
    bad = json.dumps({"results": [{"id": "c1", "score": 1.0, "justification": "ok"}]})

    monkeypatch.setattr(
        verifier,
        "_prompt_llm_judge",
        AsyncMock(return_value={"response": bad, "trajectory_object_url": None}),
    )

    with pytest.raises(ValueError, match=r"expected 2 criteria"):
        await verifier._run_judge_with_output_retries(
            eval_prompt="Evaluate the agent.",
            criteria=verifier.criteria,
            context=TaskStepContext(),
            model="claude-sonnet-4-6",
            judge_a2a_url=None,
            judge_agent_card={},
            judge_agent=None,
            judge_effort=None,
            judge_max_thinking_tokens=None,
            use_agent_judge=False,
        )

    assert verifier._prompt_llm_judge.await_count == verifier.DEFAULT_MAX_RETRIES


@pytest.mark.asyncio
async def test_a2a_path_retries_with_correction(monkeypatch):
    verifier = _verifier(use_agent_judge=True)
    bad = json.dumps({"results": [{"id": "c1", "score": 1.0, "justification": "ok"}]})
    prompts: list[str] = []

    monkeypatch.setattr(verifier, "_configure_judge_a2a", AsyncMock())
    monkeypatch.setattr(verifier, "DEFAULT_MAX_RETRIES", 2)

    async def fake_invoke(*, eval_prompt: str, judge_a2a_url: str, judge_agent_card: dict, judge_agent) -> dict:
        prompts.append(eval_prompt)
        if len(prompts) == 1:
            return {"response": bad, "trajectory_object_url": None}
        return {"response": _valid_response(["c1", "c2"]), "trajectory_object_url": None}

    monkeypatch.setattr(verifier, "_invoke_judge_a2a", fake_invoke)

    results, retries, discrepancies, _ = await verifier._run_judge_with_output_retries(
        eval_prompt="Evaluate the agent.",
        criteria=verifier.criteria,
        context=TaskStepContext(),
        model="claude-sonnet-4-6",
        judge_a2a_url="http://judge.example/a2a",
        judge_agent_card={},
        judge_agent=SimpleNamespace(sandbox_id="sb-test", sandbox_type=None),
        judge_effort=None,
        judge_max_thinking_tokens=None,
        use_agent_judge=True,
    )

    assert len(results) == 2
    assert retries == 1
    assert len(discrepancies) == 1
    assert len(prompts) == 2
    assert "Correction Required" in prompts[1]


@pytest.mark.asyncio
async def test_retries_on_malformed_json(monkeypatch):
    verifier = _verifier()
    prompts: list[str] = []

    async def fake_prompt_llm_judge(prompt: str, *, model: str, context: TaskStepContext) -> dict:
        prompts.append(prompt)
        if len(prompts) == 1:
            return {"response": "Sure! ```json\nnot valid", "trajectory_object_url": None}
        return {"response": _valid_response(["c1", "c2"]), "trajectory_object_url": None}

    monkeypatch.setattr(verifier, "_prompt_llm_judge", fake_prompt_llm_judge)

    results, retries, discrepancies, _ = await verifier._run_judge_with_output_retries(
        eval_prompt="Evaluate the agent.",
        criteria=verifier.criteria,
        context=TaskStepContext(),
        model="claude-sonnet-4-6",
        judge_a2a_url=None,
        judge_agent_card={},
        judge_agent=None,
        judge_effort=None,
        judge_max_thinking_tokens=None,
        use_agent_judge=False,
    )

    assert len(results) == 2
    assert retries == 1
    assert len(discrepancies) == 1
    assert discrepancies[0]["parse_error"]
    assert len(prompts) == 2
    assert "not valid JSON" in prompts[1]
