"""A model key the endpoint refuses fails the judge on its first call; a transient failure still retries."""
from __future__ import annotations

import json

import litellm
import pytest

from agent_env.a2a_agent import protocol
from agent_env.config.model import ModelConfig
from agent_env.store import LocalSecretStore
from agent_env.task_step.context import ModelKeyUnusable, TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

_VERDICT = {"results": [{"criterion_id": "c1", "passed": True}]}
_BUDGET = litellm.BadRequestError(
    message="Budget has been exceeded! Current cost: 51.0, Max budget: 50.0", model="m", llm_provider="openai",
)
_AUTH = litellm.AuthenticationError(
    message="Incorrect API key provided: sk-secret123", model="m", llm_provider="openai",
)


@pytest.fixture
def judge(monkeypatch):
    import agent_env.config as cfg_mod

    config = cfg_mod.Config()
    config.set_secret_store(LocalSecretStore(values={"litellm_api_key": "k"}, use_env=False))
    monkeypatch.setattr(config, "_model_cfg", ModelConfig())
    monkeypatch.setattr(cfg_mod, "get_config", lambda: config)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    return RubricsVerifierTaskStep(
        id="v", version=1, criteria=[], prompt_id="p", use_agent_judge=False,
        default_model_api_base="https://judge.invalid/v1", verifier_id="vt",
    )


def _answer_with(monkeypatch, *answers):
    real, calls, pending = litellm.acompletion, [], list(answers)

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return await real(**kwargs, mock_response=pending.pop(0))

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal, cause", [(_BUDGET, "budget"), (_AUTH, "auth")], ids=["budget", "auth"])
async def test_a_judge_whose_key_is_refused_is_called_once_and_raises_model_key_unusable(
    judge, monkeypatch, refusal, cause
):
    calls = _answer_with(monkeypatch, *[refusal] * judge.DEFAULT_MAX_RETRIES)

    with pytest.raises(ModelKeyUnusable) as raised:
        await judge._prompt_llm_judge("grade it", model="openai/gpt-4o", context=TaskStepContext())

    assert len(calls) == 1
    assert raised.value.cause == cause
    assert "sk-secret123" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_a_transient_judge_error_is_still_retried(judge, monkeypatch):
    outage = litellm.ServiceUnavailableError(message="upstream down", model="m", llm_provider="openai")
    rate_limit = litellm.RateLimitError(message="slow down", model="m", llm_provider="openai")
    calls = _answer_with(monkeypatch, outage, rate_limit, json.dumps(_VERDICT))

    out = await judge._prompt_llm_judge("grade it", model="openai/gpt-4o", context=TaskStepContext())

    assert json.loads(out["response"]) == _VERDICT
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_an_agent_judge_task_failed_for_its_key_raises_model_key_unusable(monkeypatch):
    async def send(*args, **kwargs):
        return "task-1", "ctx-1"

    async def poll(*args, **kwargs):
        return {"status": {"state": "failed", "message": {"parts": [
            {"kind": "data", "data": {"error_type": "model_key_budget", "error_message": "Budget has been exceeded"}},
        ]}}}

    monkeypatch.setattr(protocol, "send_a2a_message", send)
    monkeypatch.setattr(protocol, "poll_a2a_task", poll)
    verifier = RubricsVerifierTaskStep(
        id="v", version=1, criteria=[], prompt_id="p", use_agent_judge=True, verifier_id="vt",
    )

    with pytest.raises(ModelKeyUnusable) as raised:
        await verifier._invoke_judge_a2a(
            eval_prompt="grade it", judge_a2a_url="http://judge.example", judge_agent_card={}, judge_agent=None,
        )

    assert raised.value.cause == "budget"
