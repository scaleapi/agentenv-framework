"""judge_timeout_retries: an agent judge call that runs out of judge_timeout_seconds is tried again, and the timed-out
attempt's trajectory is kept. Without it, a timeout fails the call at once, as it always has."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from agent_env.a2a_agent import A2AAgent, protocol
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import JudgeTimeoutError, RubricsVerifierTaskStep

CRITERIA = [{"id": "c1", "title": "asked first"}]
ANSWER = json.dumps({"results": [{"id": "c1", "score": 1.0, "justification": "wake 2 call 3 asked"}]})


def _verifier(**kw) -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(id="verify", version=1, criteria=CRITERIA, prompt_id="p", use_trajectory=False,
                                   verifier_id="vt", judge_timeout_seconds=60, **kw)


def _judge(monkeypatch, verifier, outcomes: list) -> list[str]:
    """The judge's calls: each outcome is "poll-timeout", "failed:<error code>:<message>", or a completed answer."""
    sent: list[str] = []
    fetched: list[str] = []

    async def send(url, **kwargs):
        sent.append(kwargs["parts"][0]["text"])
        return f"task-{len(sent)}", None

    async def poll(url, task_id, **kwargs):
        outcome = outcomes.pop(0)
        if outcome == "poll-timeout":
            raise TimeoutError(f"A2A task {task_id} did not finish in 60s")
        if outcome.startswith("failed:"):
            code, _, message = outcome[7:].partition(":")
            return {"status": {"state": "failed", "message": {"parts": [
                {"kind": "text", "text": ""},
                {"kind": "data", "data": {"error_message": message, **({"error_code": code} if code else {})}}]}}}
        return {"status": {"state": "completed", "message": {"parts": [{"kind": "text", "text": outcome}]}}}

    async def fetch(*, a2a_server_task_id, **kwargs):
        fetched.append(a2a_server_task_id)
        return f"mem://judge/{a2a_server_task_id}.json"

    monkeypatch.setattr(protocol, "send_a2a_message", send)
    monkeypatch.setattr(protocol, "poll_a2a_task", poll)
    monkeypatch.setattr(verifier, "_fetch_judge_trajectory", fetch)
    return fetched


async def _grade(verifier):
    return await verifier._run_judge_with_output_retries(
        eval_prompt="grade it", criteria=CRITERIA, context=TaskStepContext(), model="m",
        judge_a2a_url="http://judge.test", judge_agent_card={}, judge_agent=SimpleNamespace(sandbox_id=None),
        judge_effort=None, judge_max_thinking_tokens=None, use_agent_judge=True)


@pytest.mark.asyncio
async def test_without_retries_a_timeout_fails_at_once_and_nothing_is_fetched(monkeypatch):
    verifier = _verifier()
    fetched = _judge(monkeypatch, verifier, ["poll-timeout", ANSWER])

    with pytest.raises(TimeoutError) as raised:
        await _grade(verifier)

    assert type(raised.value) is TimeoutError and fetched == []


@pytest.mark.asyncio
async def test_a_timed_out_call_is_tried_again_and_its_trajectory_kept(monkeypatch):
    verifier = _verifier(judge_timeout_retries=1)
    fetched = _judge(monkeypatch, verifier, ["poll-timeout", ANSWER])

    results, output_retries, discrepancies, url = await _grade(verifier)

    assert results[0]["score"] == 1.0 and output_retries == 0 and url == "mem://judge/task-2.json"
    assert discrepancies == [{"kind": "judge_timeout", "attempt": 1, "timeout_seconds": 60,
                              "judge_trajectory_object_url": "mem://judge/task-1.json"}]
    assert fetched == ["task-1", "task-2"]


@pytest.mark.asyncio
async def test_the_judge_s_own_timeout_code_counts_too_and_the_last_one_fails_with_its_trajectory(monkeypatch):
    verifier = _verifier(judge_timeout_retries=1)
    _judge(monkeypatch, verifier, ["failed:claude_code.timeout:claude CLI timed out after 60s", "poll-timeout"])

    with pytest.raises(JudgeTimeoutError, match=re.escape("its trajectory so far: mem://judge/task-2.json")) as raised:
        await _grade(verifier)

    assert raised.value.trajectory_object_url == "mem://judge/task-2.json"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    "failed:cli_error:claude CLI timed out after 60s",   # an image from before the timeout code: its message
    "failed::claude CLI timed out after 60s",            # a harness that sends no code
    "failed:pi.timeout:pi did not finish within 60s",    # another harness's code
])
async def test_a_timeout_the_code_or_else_the_message_tells_is_retried(monkeypatch, failure):
    verifier = _verifier(judge_timeout_retries=1)
    _judge(monkeypatch, verifier, [failure, ANSWER])

    results, _, discrepancies, _ = await _grade(verifier)

    assert results[0]["score"] == 1.0 and discrepancies[0]["kind"] == "judge_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [
    "failed:cli_error:the CLI crashed",
    "failed:cli_error:API Error: The operation timed out.",
    "failed:pi.model_error:the model timed out after 30s",
])
async def test_another_failure_is_not_retried(monkeypatch, failure):
    verifier = _verifier(judge_timeout_retries=1)
    _judge(monkeypatch, verifier, [failure, ANSWER])

    with pytest.raises(RuntimeError, match="Judge A2A task failed"):
        await _grade(verifier)


def test_judge_timeout_retries_and_max_turns_round_trip_and_are_left_out_unset():
    assert not {"judge_timeout_retries", "judge_max_turns"} & set(_verifier().to_dict())
    restored = RubricsVerifierTaskStep.from_dict(_verifier(judge_timeout_retries=2, judge_max_turns=300).to_dict())
    assert (restored.judge_timeout_retries, restored.judge_max_turns) == (2, 300)
    for bad in (-1, True, "1", 1.5):
        with pytest.raises(ValueError, match="judge_timeout_retries must be a non-negative integer"):
            _verifier(judge_timeout_retries=bad)
    with pytest.raises(ValueError, match="judge_max_turns must be a positive integer"):
        _verifier(judge_max_turns=0)


@pytest.mark.asyncio
async def test_judge_agent_config_extras_go_to_a_judge_that_takes_them(monkeypatch, caplog):
    posted: list[dict] = []

    async def post(url, payload):
        posted.append(payload)

    monkeypatch.setattr(protocol, "post_agent_config", post)

    class Extra(RubricsVerifierTaskStep):
        def judge_agent_config(self) -> dict:
            return {"builtin_tools": {"disable": ["Agent"]}, "effort_level": "high"}

    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {
        "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": [
            "model", "builtin_tools", "max_turns"]}}}}}]}}
    verifier = Extra(id="verify", version=1, criteria=CRITERIA, prompt_id="p", verifier_id="vt", judge_max_turns=300)
    await verifier._configure_judge_a2a(judge_a2a_url="http://judge.test", judge_agent_card=card, model="m",
                                        judge_effort=None, judge_max_thinking_tokens=None, context=TaskStepContext())

    assert posted == [{"model": "m", "max_turns": 300, "builtin_tools": {"disable": ["Agent"]}}]
    assert "doesn't take ['effort_level']" in caplog.text

    posted.clear()
    plain = RubricsVerifierTaskStep(id="verify", version=1, criteria=CRITERIA, prompt_id="p", verifier_id="vt")
    await plain._configure_judge_a2a(judge_a2a_url="http://judge.test", judge_agent_card=card, model="m",
                                     judge_effort=None, judge_max_thinking_tokens=None, context=TaskStepContext())
    assert posted == [{"model": "m"}]
