"""A rubrics judge's model calls are the step's usage: one ``judge`` entry in ``metadata.step_usage[<step id>]``,
summed over the step's calls (every batch, every output retry), so the task instance's usage ledger counts judging as
it counts the agents' turns. A direct judge's call is read from its response, the endpoint's bill its cost when it sent
one; an agent judge's from the usage its turn reported."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from agent_env.a2a_agent import A2AAgent, protocol
from agent_env.config.model import ModelRole
from agent_env.task_step.context import PromptResponse, TaskStepContext
from agent_env.task_step.task_steps.verifiers import rubrics_verifier
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from agent_env.task_step.usage import StepUsage

CRITERIA = [{"id": f"c{i}", "title": f"Criterion {i}"} for i in range(1, 8)]


def _verifier(**kw) -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(id="verify", version=1, criteria=CRITERIA, prompt_id="p1", use_agent_judge=False,
                                   use_trajectory=False, verifier_id="vt", **kw)


def _context() -> TaskStepContext:
    return TaskStepContext(prompt_responses=[PromptResponse(prompt_id="p1", response="done", prompt_text="do it")])


def _direct_judge(verifier, *, billed=None):
    """Answers every batch it's sent, each call reporting 1,000 prompt and 50 completion tokens."""
    calls = []

    async def judge(prompt: str, *, model: str, context: TaskStepContext) -> dict:
        calls.append(prompt)
        ids = [c["id"] for c in json.loads(prompt[prompt.index("["):prompt.index("]", prompt.index("[")) + 1])]
        return {"response": json.dumps({"results": [{"id": i, "score": 1.0, "justification": f"Checked record {i} with the sheet tool"} for i in ids]}),
                "trajectory_object_url": None, "usage": {"prompt_tokens": 1000, "completion_tokens": 50},
                "model": "gpt-4.1-mini", "endpoint_host": "proxy.example:4000", "reported_cost_usd": billed}

    verifier._prompt_llm_judge = judge
    return calls


@pytest.mark.asyncio
async def test_a_direct_judges_calls_over_every_batch_are_one_judge_entry():
    verifier = _verifier(judge_batch_size=3)
    calls = _direct_judge(verifier)
    context = _context()

    await verifier.execute(context)

    [entry] = context.metadata["step_usage"]["verify"]
    assert len(calls) == 3
    assert (entry["role"], entry["model"], entry["endpoint_host"], entry["turns"]) == (
        "judge", "gpt-4.1-mini", "proxy.example:4000", 3)
    assert (entry["input_tokens"], entry["output_tokens"], entry["cost_status"]) == (3000, 150, "priced")
    assert context.metadata["model_endpoints"]["judge"]["verify"]["endpoint_host"] == "proxy.example:4000"


@pytest.mark.asyncio
async def test_a_direct_judges_bill_is_its_cost():
    verifier = _verifier()
    _direct_judge(verifier, billed=0.02)
    context = _context()

    await verifier.execute(context)

    [entry] = context.metadata["step_usage"]["verify"]
    assert (entry["cost_status"], entry["cost_source"], entry["cost_usd"]) == ("reported", "provider", 0.02)


def _agent_judge(monkeypatch, state: str, answer: str, reported: dict) -> RubricsVerifierTaskStep:
    """A rubrics verifier whose agent judge ends its one turn in ``state``, replying ``answer`` and its usage."""
    verifier = RubricsVerifierTaskStep(id="verify", version=1, criteria=CRITERIA[:1], prompt_id="p", verifier_id="vt",
                                       use_trajectory=False)

    async def send(url, **kwargs):
        return "task-1", None

    async def poll(url, task_id, **kwargs):
        return {"status": {"state": state, "message": {"parts": [
            {"kind": "text", "text": answer}, {"kind": "data", "data": {"usage": reported}}]}}}

    async def nothing(*args, **kwargs):
        return None

    monkeypatch.setattr(protocol, "send_a2a_message", send)
    monkeypatch.setattr(protocol, "poll_a2a_task", poll)
    monkeypatch.setattr(protocol, "post_agent_config", nothing)
    monkeypatch.setattr(verifier, "_fetch_judge_trajectory", nothing)
    monkeypatch.setattr(rubrics_verifier, "fetch_container_logs", nothing)
    return verifier


_TAKES_A_MODEL = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {
    "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": ["model"]}}}}}]}}


async def _judge_once(verifier: RubricsVerifierTaskStep, context: TaskStepContext, card: dict = _TAKES_A_MODEL) -> None:
    await verifier._run_judge_with_output_retries(
        eval_prompt="grade it", criteria=CRITERIA[:1], context=context, model="claude-haiku-4-5",
        judge_a2a_url="http://judge.test", judge_agent_card=card,
        judge_agent=SimpleNamespace(sandbox_id=None, control_headers={}, model_endpoint_host="proxy.example:4000"),
        judge_effort=None, judge_max_thinking_tokens=None, use_agent_judge=True,
        usage=StepUsage(context, "verify"))


@pytest.mark.asyncio
async def test_an_agent_judge_that_runs_its_own_model_is_priced_by_none(monkeypatch):
    answer = json.dumps({"results": [{"id": "c1", "score": 1.0, "justification": "asked first"}]})
    verifier = _agent_judge(monkeypatch, "completed", answer, {"input_tokens": 4000, "output_tokens": 80,
                                                               "input_includes_cache": True})
    context = TaskStepContext()

    await _judge_once(verifier, context, card={})

    [entry] = context.metadata["step_usage"]["verify"]
    assert (entry["model"], entry["input_tokens"], entry["cost_status"]) == (None, 4000, "unpriced")
    assert context.metadata["model_endpoints"]["judge"]["verify"]["model"] is None


@pytest.mark.asyncio
async def test_an_agent_judges_failed_reply_still_counts_its_usage(monkeypatch):
    verifier = _agent_judge(monkeypatch, "failed", "rate limited", {"input_tokens": 900, "output_tokens": 0,
                                                                     "input_includes_cache": True})
    context = TaskStepContext()

    with pytest.raises(RuntimeError, match="Judge A2A task failed"):
        await _judge_once(verifier, context)

    [entry] = context.metadata["step_usage"]["verify"]
    assert (entry["role"], entry["turns"], entry["input_tokens"]) == ("judge", 1, 900)


def test_a_billed_call_without_counts_keeps_its_bill():
    context = TaskStepContext()
    usage = StepUsage(context, "verify")
    RubricsVerifierTaskStep._record_call(usage, {"response": "", "model": "m", "endpoint_host": None,
                                                 "reported_cost_usd": 0.01}, model="m")
    RubricsVerifierTaskStep._record_call(usage, {"response": "", "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                                                 "model": "m", "endpoint_host": None, "reported_cost_usd": 0.02},
                                         model="m")
    [entry] = context.metadata["step_usage"]["verify"]
    assert (entry["cost_status"], entry["cost_usd"], entry["partial"]) == ("reported", pytest.approx(0.03), True)


@pytest.mark.asyncio
async def test_an_agent_judges_reported_usage_is_its_calls(monkeypatch):
    answer = json.dumps({"results": [{"id": "c1", "score": 1.0, "justification": "asked first"}]})
    reported = {"input_tokens": 4000, "output_tokens": 80, "input_includes_cache": True, "cost_usd": 0.01}
    verifier = _agent_judge(monkeypatch, "completed", answer, reported)
    context = TaskStepContext()

    await _judge_once(verifier, context)

    [entry] = context.metadata["step_usage"]["verify"]
    assert (entry["role"], entry["model"], entry["endpoint_host"], entry["turns"]) == (
        "judge", "claude-haiku-4-5", "proxy.example:4000", 1)
    assert (entry["input_tokens"], entry["output_tokens"], entry["reported_cost_usd"]) == (4000, 80, 0.01)


def test_a_call_with_no_usage_is_a_turn_with_no_counts():
    context = TaskStepContext()
    usage = StepUsage(context, "verify")
    RubricsVerifierTaskStep._record_call(usage, {"response": "", "model": "m", "endpoint_host": None}, model="m")
    RubricsVerifierTaskStep._record_call(usage, {"response": "", "usage": {"prompt_tokens": 10, "completion_tokens": 1},
                                                 "model": "m", "endpoint_host": None}, model="m")
    [entry] = context.metadata["step_usage"]["verify"]
    assert (entry["turns"], entry["input_tokens"], entry["partial"]) == (2, 10, True)
    assert entry["role"] == ModelRole.JUDGE.value
