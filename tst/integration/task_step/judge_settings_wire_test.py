"""A direct judge's settings and stored input on the wire: through LiteLLM's real client to a fake OpenAI-compatible endpoint, the run's
effort goes out as ``reasoning_effort`` where LiteLLM takes it, a thinking budget the route doesn't take stays out and
the grade says so, a value LiteLLM refuses never reaches the endpoint, and the grade keeps each call with the usage the
endpoint reported."""

from __future__ import annotations

import hashlib
import json
import re

import pytest

from agent_env.config import get_config
from agent_env.task_step.context import PromptResponse, TaskStepContext, VerifierNotPassingError
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import JudgeSettingRefused, RubricsVerifierTaskStep
from tst.util.config import install_model_config
from tst.util.fake_openai import USAGE, FakeOpenAI, Text

pytestmark = [pytest.mark.integration]

CRITERIA = [{"id": "greets", "description": "It greets the user"}, {"id": "polite", "description": "It is polite"}]


def _verdict(body: dict) -> Text:
    """Every criterion the request asks about passes."""
    ids = list(dict.fromkeys(re.findall(r'"id": "(c\d+)"', body["messages"][-1]["content"])))
    return Text(json.dumps({"results": [{"id": i, "score": 1.0, "justification": "It says hello."} for i in ids]}))


@pytest.fixture
def endpoint(monkeypatch):
    with FakeOpenAI(respond=_verdict) as fake:
        install_model_config(monkeypatch, base_url=f"{fake.url}/v1", api_key="test-key")
        yield fake


def _judge(model: str, **fields) -> RubricsVerifierTaskStep:
    return RubricsVerifierTaskStep(id="judge", version=1, criteria=CRITERIA, prompt_id="ask", use_agent_judge=False,
                                   use_trajectory=False, verifier_id="judge", default_model=model, **fields)


def _context() -> TaskStepContext:
    return TaskStepContext(prompt_responses=[PromptResponse(prompt_id="ask", response="Hello, nice to meet you!",
                                                            prompt_text="Greet me.")], instance_id="inst-1")


async def _grade(step: RubricsVerifierTaskStep, context: TaskStepContext) -> dict:
    try:
        await step.execute(context)
    except VerifierNotPassingError:
        pass
    return context.metadata["grades"][step.id]


@pytest.mark.asyncio
async def test_gpt_5_gets_the_effort_on_the_wire_and_the_thinking_budget_is_recorded_as_not_sent(endpoint):
    context = _context()

    grade = await _grade(_judge("gpt-5", effort="low", max_thinking_tokens=2048), context)

    [request] = endpoint.completions()
    assert request.body["reasoning_effort"] == "low" and "thinking" not in request.body
    settings = grade["verifier_settings"]
    assert (settings["effort"], settings["max_thinking_tokens"], settings["not_sent"]) == (
        "low", None, {"max_thinking_tokens": 2048})
    [call] = grade["judge_calls"]
    assert (call["outcome"], call["model"], call["usage"]["input_tokens"], call["usage"]["output_tokens"]) == (
        "answered", "gpt-5", USAGE["prompt_tokens"], USAGE["completion_tokens"])
    assert grade["verifier_status"] == "graded" and grade["score"] == 1.0
    stored = get_config().get_object_store().get(call["input"]["object_url"])
    assert hashlib.sha256(stored).hexdigest() == call["input"]["sha256"]
    assert json.loads(stored)["messages"] == request.body["messages"]


@pytest.mark.asyncio
async def test_a_model_without_reasoning_control_grades_without_the_effort_and_says_so(endpoint):
    grade = await _grade(_judge("gpt-4o-mini", effort="high"), _context())

    [request] = endpoint.completions()
    assert "reasoning_effort" not in request.body
    assert grade["verifier_settings"]["not_sent"] == {"effort": "high"} and grade["verifier_status"] == "graded"


@pytest.mark.asyncio
async def test_a_value_litellm_refuses_never_reaches_the_endpoint_and_fails_the_grade(endpoint):
    context = _context()

    with pytest.raises(JudgeSettingRefused, match="xhigh"):
        await _judge("gpt-5", effort="xhigh").execute(context)

    assert endpoint.completions() == []
    grade = context.metadata["grades"]["judge"]
    assert grade["verifier_status"] == "failed"
    assert [call["outcome"] for call in grade["judge_calls"]] == ["refused"]
