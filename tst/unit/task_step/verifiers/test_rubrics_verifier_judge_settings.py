"""A judge's settings as its calls carried them, and its record of each call: the direct judge sends the run's effort
and thinking budget where LiteLLM takes them on its route, leaves out and records those it doesn't, fails at once on a
value its model refuses, and every judge's grade keeps one record per call."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import litellm
import pytest

from agent_env.a2a_agent import A2AAgent, protocol
from agent_env.task_step.context import TaskStepContext, VerifierNotPassingError
from agent_env.task_step.task_steps.verifiers.judge_utils import judge_usage
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import JudgeSettingRefused
from tst.unit.task_step.verifiers.test_rubrics_verifier_golden import TWO, _binary, _context, _ids, _step
from tst.util.config import install_model_config

ENDPOINT = "https://gw.test/v1"
USAGE = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}


def _llm(monkeypatch, *, answers=None, refuse_effort=False, failures=(), **config) -> list[dict]:
    """LiteLLM faked through the configured endpoint: each request kept, each answer a binary verdict for the ids its
    prompt asks about (``answers`` overrides them in turn); ``refuse_effort`` raises as LiteLLM refuses a value, and
    ``failures`` are raised by the first requests, one each."""
    install_model_config(monkeypatch, base_url=ENDPOINT, api_key="configured-key", **config)
    requests: list[dict] = []
    queued = list(answers or [])
    failing = list(failures)

    async def acompletion(**kwargs):
        requests.append(kwargs)
        if failing:
            raise failing.pop(0)
        if refuse_effort and "reasoning_effort" in kwargs:
            raise litellm.UnsupportedParamsError(message="reasoning_effort=xhigh is not supported for gpt-5",
                                                 llm_provider="openai", model="gpt-5")
        prompt = kwargs["messages"][-1]["content"]
        text = prompt if isinstance(prompt, str) else "".join(block.get("text", "") for block in prompt)
        content = queued.pop(0) if queued else _binary(_ids(text))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=dict(USAGE),
                               _hidden_params={})

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    return requests


async def _grade(step, context: TaskStepContext) -> dict:
    try:
        await step.execute(context)
    except VerifierNotPassingError:
        pass
    return context.metadata["grades"][step.id]


@pytest.mark.asyncio
async def test_the_direct_judge_sends_the_effort_and_thinking_budget_its_route_takes_and_records_them(monkeypatch):
    requests = _llm(monkeypatch)
    step = _step(default_model="claude-sonnet-4-6", effort="high", max_thinking_tokens=2048)

    grade = await _grade(step, _context())

    assert {(r["reasoning_effort"], r["thinking"]["budget_tokens"]) for r in requests} == {("high", 2048)}
    settings = grade["verifier_settings"]
    assert (settings["effort"], settings["max_thinking_tokens"], settings["judge_endpoint_host"]) == (
        "high", 2048, "gw.test")
    assert "not_sent" not in settings and grade["verifier_status"] == "graded"


@pytest.mark.asyncio
async def test_a_setting_the_route_doesnt_take_is_left_out_and_the_grade_says_so(monkeypatch, caplog):
    requests = _llm(monkeypatch)
    step = _step(default_model="gpt-4o-mini", effort="high", max_thinking_tokens=2048)

    with caplog.at_level(logging.WARNING):
        grade = await _grade(step, _context())

    assert not {"reasoning_effort", "thinking"} & set(requests[0])
    settings = grade["verifier_settings"]
    assert (settings["effort"], settings["max_thinking_tokens"]) == (None, None)
    assert settings["not_sent"] == {"effort": "high", "max_thinking_tokens": 2048}
    assert grade["verifier_status"] == "graded"
    assert "gpt-4o-mini takes no ['effort', 'max_thinking_tokens']; the judge grades without them" in caplog.text


@pytest.mark.asyncio
async def test_gpt_5_takes_effort_but_no_thinking_budget(monkeypatch):
    requests = _llm(monkeypatch)
    grade = await _grade(_step(default_model="gpt-5", effort="low", max_thinking_tokens=2048), _context())

    assert requests[0]["reasoning_effort"] == "low" and "thinking" not in requests[0]
    assert grade["verifier_settings"]["not_sent"] == {"max_thinking_tokens": 2048}


@pytest.mark.asyncio
async def test_a_value_the_model_refuses_fails_the_grade_at_once_with_the_call_in_its_record(monkeypatch):
    requests = _llm(monkeypatch, refuse_effort=True)
    context = _context()

    with pytest.raises(JudgeSettingRefused, match=r"gpt-5 refused the judge's \['effort'\]"):
        await _step(default_model="gpt-5", effort="xhigh").execute(context)

    assert len(requests) == 1
    grade = context.metadata["grades"]["verify"]
    assert grade["verifier_status"] == "failed" and "reasoning_effort=xhigh is not supported" in (
        grade["verifier_error_reasoning"])
    assert [(c["outcome"], c["model"], c["usage"]) for c in grade["judge_calls"]] == [("refused", "gpt-5", None)]


@pytest.mark.asyncio
async def test_the_runs_judge_effort_reaches_a_direct_judge(monkeypatch):
    requests = _llm(monkeypatch)

    grade = await _grade(_step(default_model="claude-sonnet-4-6"), _context({"judge_effort": "low"}))

    assert requests[0]["reasoning_effort"] == "low"
    assert grade["verifier_settings"]["effort"] == "low" and "effort" in grade["verifier_settings"]["overridden"]


@pytest.mark.asyncio
async def test_the_configs_temperature_and_seed_are_recorded_as_sent(monkeypatch):
    _llm(monkeypatch, params={"temperature": 0, "seed": 7})

    grade = await _grade(_step(default_model="gpt-4o-mini"), _context())

    assert (grade["verifier_settings"]["temperature"], grade["verifier_settings"]["seed"]) == (0, 7)


@pytest.mark.asyncio
async def test_the_runs_effort_wins_over_the_configs_and_the_configs_alone_is_recorded_as_sent(monkeypatch):
    requests = _llm(monkeypatch, params={"reasoning_effort": "low"})

    over = await _grade(_step(default_model="claude-sonnet-4-6", effort="high"), _context())
    alone = await _grade(_step(default_model="claude-sonnet-4-6"), _context())

    assert [request["reasoning_effort"] for request in requests] == ["high", "low"]
    assert (over["verifier_settings"]["effort"], alone["verifier_settings"]["effort"]) == ("high", "low")


@pytest.mark.asyncio
async def test_a_request_retried_after_a_timeout_is_a_record_of_its_own(monkeypatch):
    _llm(monkeypatch, failures=[litellm.Timeout("timed out", model="gpt-4o-mini", llm_provider="openai")])
    context = _context()

    grade = await _grade(_step(default_model="gpt-4o-mini"), context)

    assert [call["outcome"] for call in grade["judge_calls"]] == ["cut_off", "answered"]
    [entry] = context.metadata["step_usage"]["verify"]
    assert entry["turns"] == 2 and entry["partial"] is True


@pytest.mark.asyncio
async def test_a_judge_whose_requests_all_fail_keeps_its_settings_and_each_request_in_its_failed_grade(monkeypatch):
    _llm(monkeypatch, failures=[litellm.RateLimitError("slow down", llm_provider="openai", model="gpt-4o-mini")] * 3)
    context = _context()

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        await _step(default_model="gpt-4o-mini", effort="high").execute(context)

    grade = context.metadata["grades"]["verify"]
    assert grade["verifier_status"] == "failed"
    assert (grade["verifier_settings"]["not_sent"], grade["verifier_settings"]["judge_endpoint_host"]) == (
        {"effort": "high"}, "gw.test")
    assert [call["outcome"] for call in grade["judge_calls"]] == ["refused"] * 3
    assert all("slow down" in call["error"] for call in grade["judge_calls"])


@pytest.mark.asyncio
async def test_the_grade_keeps_one_record_per_call_with_its_batch_and_attempt(monkeypatch):
    _llm(monkeypatch)
    context = _context({"judge_batch_size": 1})

    grade = await _grade(_step(default_model="gpt-4o-mini"), context)

    calls = grade["judge_calls"]
    assert sorted((c["batch"], c["attempt"], c["outcome"]) for c in calls) == [(1, 1, "answered"), (2, 1, "answered")]
    assert {(c["model"], c["endpoint_host"], c["usage"]["input_tokens"], c["usage"]["output_tokens"]) for c in calls} == {
        ("gpt-4o-mini", "gw.test", 100, 20)}
    assert all(c["cost_status"] == "priced" and c["cost_usd"] > 0 for c in calls)
    [entry] = context.metadata["step_usage"]["verify"]
    assert entry["role"] == "judge" and entry["turns"] == 2


@pytest.mark.asyncio
async def test_an_output_retry_is_a_record_of_its_own(monkeypatch):
    ids = [f"c{i}" for i in range(1, len(TWO) + 1)]
    _llm(monkeypatch, answers=[_binary(ids[:1]), _binary(ids)])

    grade = await _grade(_step(default_model="gpt-4o-mini"), _context())

    assert [(c["attempt"], c["outcome"]) for c in grade["judge_calls"]] == [(1, "answered"), (2, "answered")]


@pytest.mark.asyncio
async def test_an_agent_judge_card_without_effort_leaves_it_out_and_the_record_says_so(monkeypatch, caplog):
    posted: list[dict] = []

    async def post(url, payload, headers=None):
        posted.append(payload)

    monkeypatch.setattr(protocol, "post_agent_config", post)
    card = {"capabilities": {"extensions": [{"uri": A2AAgent.EXT_AGENT_CONFIG, "params": {
        "endpoint": "/ext/agent-config", "methods": {"set": {"request": {"supported": ["model"]}}}}}]}}
    context = TaskStepContext()
    with judge_usage.scope(), caplog.at_level(logging.WARNING):
        judge = judge_usage.start(context, "verify")
        await _step(use_agent_judge=True)._configure_judge_a2a(
            judge_a2a_url="http://judge.test", judge_agent_card=card, model="m", judge_effort="high",
            judge_max_thinking_tokens=None, context=context, headers={}, endpoint_host="gw.test")

    assert posted == [{"model": "m"}]
    assert (judge.sent, judge.not_sent, judge.endpoint_host) == ({}, {"effort": "high"}, "gw.test")
    assert "the judge agent doesn't take ['effort']; it grades without them" in caplog.text
