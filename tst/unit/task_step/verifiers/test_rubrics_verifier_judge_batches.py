"""judge_batch_size: grade a long rubric in batches, each its own judge call, and retry bulk verdicts.

A single judge facing hundreds of criteria greps for a few keywords and gives the rest a stock verdict;
smaller batches and the bulk-verdict check keep each row graded on its own.
"""
from __future__ import annotations

import asyncio
import json
import re

import pytest

from types import SimpleNamespace

from agent_env.a2a_agent import protocol
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import (
    RubricsVerifierTaskStep,
    _bulk_verdicts,
)


def _criteria(n: int) -> list[dict]:
    return [{"id": f"real-{i}", "title": f"Criterion {i}"} for i in range(1, n + 1)]


def _verifier(**kw) -> RubricsVerifierTaskStep:
    params = dict(
        id="verify", version=1, criteria=_criteria(7), prompt_id="p1",
        use_agent_judge=False, use_trajectory=False, verifier_id="vt",
    )
    params.update(kw)
    return RubricsVerifierTaskStep(**params)


def _context(**overrides) -> TaskStepContext:
    ctx = TaskStepContext(prompt_responses=[PromptResponse(prompt_id="p1", response="done", prompt_text="do it")])
    if overrides:
        ctx.metadata["user_overrides"] = overrides
    return ctx


def _prompted_ids(prompt: str) -> list[str]:
    criteria_json = prompt[prompt.index("["):prompt.index("]", prompt.index("[")) + 1]
    return [c["id"] for c in json.loads(criteria_json)]


def _graded(ids: list[str], justification=lambda cid: f"Checked record {cid} with the sheet tool") -> str:
    return json.dumps({"results": [{"id": cid, "score": 1.0, "justification": justification(cid)} for cid in ids]})


def _fake_judge(verifier, prompts: list[str], respond=_graded):
    async def fake_prompt_llm_judge(prompt: str, *, model: str, context: TaskStepContext) -> dict:
        prompts.append(prompt)
        return {"response": respond(_prompted_ids(prompt)), "trajectory_object_url": f"mem://judge/{len(prompts)}.json"}

    verifier._prompt_llm_judge = fake_prompt_llm_judge


def test_shared_or_short_justifications_are_bulk_verdicts():
    distinct = [{"justification": f"Sheet row {i} shows the value the criterion asks for"} for i in range(10)]
    assert _bulk_verdicts(distinct) is None

    shared = distinct[:4] + [{"justification": "Addressed."} for _ in range(6)]
    bulk = _bulk_verdicts(shared)
    assert bulk["shared_count"] == 6 and bulk["shared_justification"] == "addressed"

    short = [{"justification": f"Row {i} ok"} for i in range(10)]
    assert _bulk_verdicts(short)["short_count"] == 10


@pytest.mark.asyncio
async def test_criteria_are_graded_in_batches_and_merged_in_order():
    v = _verifier(judge_batch_size=3)
    prompts: list[str] = []
    _fake_judge(v, prompts)
    ctx = _context()

    await v.execute(ctx)

    assert sorted(len(_prompted_ids(p)) for p in prompts) == [1, 3, 3]
    assert all("Do not assign verdicts with a script" in p for p in prompts)
    assert any("These are criteria 4–6 of 7" in p for p in prompts)
    entry = ctx.metadata["verifications"]["vt"]
    assert [r["id"] for r in entry["results"]] == [c["id"] for c in _criteria(7)]
    assert entry["judge_batches"] == 3
    assert len(entry["judge_batch_trajectory_object_urls"]) == 3
    assert entry["score"] == 1.0


@pytest.mark.asyncio
async def test_batch_size_from_user_overrides():
    v = _verifier()
    prompts: list[str] = []
    _fake_judge(v, prompts)

    await v.execute(_context(judge_batch_size=4))

    assert sorted(len(_prompted_ids(p)) for p in prompts) == [3, 4]


@pytest.mark.asyncio
async def test_without_batch_size_one_call_and_no_bulk_check():
    v = _verifier()
    prompts: list[str] = []
    _fake_judge(v, prompts, respond=lambda ids: _graded(ids, justification=lambda cid: "ok"))
    ctx = _context()

    await v.execute(ctx)

    assert len(prompts) == 1
    assert "How to grade" not in prompts[0]
    entry = ctx.metadata["verifications"]["vt"]
    assert "judge_batches" not in entry and "judge_output_discrepancies" not in entry


@pytest.mark.asyncio
async def test_batches_run_at_most_judge_batch_concurrency_at_once():
    v = _verifier(criteria=_criteria(8), judge_batch_size=1, judge_batch_concurrency=2)
    running = peak = 0

    async def fake_prompt_llm_judge(prompt: str, *, model: str, context: TaskStepContext) -> dict:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return {"response": _graded(_prompted_ids(prompt)), "trajectory_object_url": None}

    v._prompt_llm_judge = fake_prompt_llm_judge
    await v.execute(_context())

    assert peak == 2


@pytest.mark.asyncio
async def test_bulk_verdicts_are_retried_with_a_correction():
    v = _verifier(criteria=_criteria(6), judge_batch_size=6)
    prompts: list[str] = []

    def respond(ids):
        if len(prompts) == 1:
            return _graded(ids, justification=lambda cid: "Addressed.")
        return _graded(ids)

    _fake_judge(v, prompts, respond=respond)
    ctx = _context()

    await v.execute(ctx)

    assert len(prompts) == 2
    assert "bulk verdict" in prompts[1] and "6 of 6 criteria got the same justification ('addressed')" in prompts[1]
    entry = ctx.metadata["verifications"]["vt"]
    assert entry["judge_output_retries"] == 1
    assert entry["judge_output_discrepancies"][0]["kind"] == "bulk_verdicts"
    assert entry["results"][0]["justification"].startswith("Checked record")


@pytest.mark.asyncio
async def test_bulk_verdicts_on_the_last_attempt_are_kept_and_flagged():
    v = _verifier(criteria=_criteria(6), judge_batch_size=6)
    prompts: list[str] = []
    _fake_judge(v, prompts, respond=lambda ids: _graded(ids, justification=lambda cid: "no"))
    ctx = _context()

    await v.execute(ctx)

    assert len(prompts) == v.DEFAULT_MAX_RETRIES
    discrepancies = ctx.metadata["verifications"]["vt"]["judge_output_discrepancies"]
    assert [d.get("accepted") for d in discrepancies] == [None, None, True]
    assert len(ctx.metadata["verifications"]["vt"]["results"]) == 6


@pytest.mark.asyncio
async def test_a_failed_batch_fails_the_step():
    v = _verifier(judge_batch_size=3)

    async def fake_prompt_llm_judge(prompt: str, *, model: str, context: TaskStepContext) -> dict:
        if "criteria 4–6" in prompt:
            raise RuntimeError("judge down")
        return {"response": _graded(_prompted_ids(prompt)), "trajectory_object_url": None}

    v._prompt_llm_judge = fake_prompt_llm_judge
    with pytest.raises(RuntimeError, match="judge down"):
        await v.execute(_context())


def test_batch_options_round_trip_and_validate():
    v = _verifier(judge_batch_size=30, judge_batch_concurrency=6)
    data = v.to_dict()
    assert (data["judge_batch_size"], data["judge_batch_concurrency"]) == (30, 6)
    restored = RubricsVerifierTaskStep.from_dict(data)
    assert (restored.judge_batch_size, restored.judge_batch_concurrency) == (30, 6)
    assert "judge_batch_size" not in _verifier().to_dict()

    for bad in (0, -1, True, "30"):
        with pytest.raises(ValueError, match=re.escape("judge_batch_size must be a positive integer")):
            _verifier(judge_batch_size=bad)


@pytest.mark.asyncio
async def test_batch_concurrency_from_user_overrides():
    v = _verifier(criteria=_criteria(6), judge_batch_size=1)
    running = peak = 0

    async def fake_prompt_llm_judge(prompt: str, *, model: str, context: TaskStepContext) -> dict:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)
        running -= 1
        return {"response": _graded(_prompted_ids(prompt)), "trajectory_object_url": None}

    v._prompt_llm_judge = fake_prompt_llm_judge
    await v.execute(_context(judge_batch_concurrency=1))

    assert peak == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [0, -2, "30", True])
async def test_an_invalid_batch_override_is_reported_not_ignored(bad):
    v = _verifier()
    _fake_judge(v, [])

    with pytest.raises(ValueError, match="judge_batch_size must be a positive integer"):
        await v.execute(_context(judge_batch_size=bad))


def test_formats_with_whole_run_rows_cannot_be_batched():
    with pytest.raises(ValueError, match="judge_batch_size does not apply to output_format=trajectory_mistakes"):
        _verifier(output_format="trajectory_mistakes", judge_batch_size=3)


@pytest.mark.asyncio
async def test_a_batch_override_on_an_unbatchable_format_fails():
    v = _verifier(output_format="trajectory_mistakes")

    with pytest.raises(RuntimeError, match="does not apply to output_format=trajectory_mistakes"):
        await v.execute(_context(judge_batch_size=3))


@pytest.mark.asyncio
async def test_a_cancelled_batch_cancels_its_remote_judge_task(monkeypatch):
    v = _verifier(use_agent_judge=True)
    cancelled: list[tuple[str, str]] = []
    polling = asyncio.Event()

    async def fake_send(url, **kwargs):
        return "a2a-task-1", None

    async def fake_poll(url, task_id, **kwargs):
        polling.set()
        await asyncio.sleep(3600)

    async def fake_cancel(url, task_id):
        cancelled.append((url, task_id))

    monkeypatch.setattr(protocol, "send_a2a_message", fake_send)
    monkeypatch.setattr(protocol, "poll_a2a_task", fake_poll)
    monkeypatch.setattr(protocol, "cancel_a2a_task", fake_cancel)
    call = asyncio.create_task(v._invoke_judge_a2a(
        eval_prompt="grade", judge_a2a_url="http://judge", judge_agent_card={},
        judge_agent=SimpleNamespace(sandbox_id="sb-1")))
    await polling.wait()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    assert cancelled == [("http://judge", "a2a-task-1")]
