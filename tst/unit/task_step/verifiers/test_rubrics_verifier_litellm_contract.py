"""The rubrics judge's calls go through the installed litellm, not a stand-in.

``_prompt_llm_judge`` calls litellm through ``config.model_call``: ``acompletion`` with a strict ``json_schema`` response_format over
text or text-and-image content, read through ``.choices[0].message.content``, sent again without the response_format
when an endpoint refuses it with images. Each test hands the real ``litellm.acompletion`` exactly the kwargs the
judge builds, adding only litellm's ``mock_response`` so nothing leaves the process, so a litellm release that stops
accepting one of those shapes fails here instead of in a run.
"""
from __future__ import annotations

import json

import litellm
import pytest

from agent_env.config.model import ModelConfig
from agent_env.store import LocalSecretStore
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

_VERDICT = {"results": [{"criterion_id": "c1", "passed": True}]}
_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}


@pytest.fixture
def judge(monkeypatch):
    import agent_env.config as cfg_mod

    config = cfg_mod.Config()
    config.set_secret_store(LocalSecretStore(values={"litellm_api_key": "k"}, use_env=False))
    monkeypatch.setattr(config, "_model_cfg", ModelConfig(base_url="https://judge.invalid/v1"))
    monkeypatch.setattr(cfg_mod, "get_config", lambda: config)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    return RubricsVerifierTaskStep(
        id="v", version=1, criteria=[], prompt_id="p", use_agent_judge=False,
        default_model_api_base="https://judge.invalid/v1", verifier_id="vt",
    )


def _answer_with(monkeypatch, *answers):
    """Send each call to the real acompletion with the next of ``answers`` as its mock_response; return the calls."""
    real, calls, pending = litellm.acompletion, [], list(answers)

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return await real(**kwargs, mock_response=pending.pop(0))

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    return calls


@pytest.mark.asyncio
async def test_a_text_judge_call_is_answered_through_a_strict_json_schema(judge, monkeypatch):
    calls = _answer_with(monkeypatch, json.dumps(_VERDICT))

    out = await judge._prompt_llm_judge("grade it", model="openai/gpt-4o", context=TaskStepContext())

    assert json.loads(out["response"]) == _VERDICT
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["strict"] is True
    assert calls[0]["api_base"] == "https://judge.invalid/v1"


@pytest.mark.asyncio
async def test_a_judge_call_with_screenshots_sends_text_and_image_content(judge, monkeypatch):
    calls = _answer_with(monkeypatch, json.dumps(_VERDICT))

    out = await judge._prompt_llm_judge("grade it", model="openai/gpt-4o", context=TaskStepContext(),
                                        image_blocks=[_IMAGE])

    assert json.loads(out["response"]) == _VERDICT
    assert calls[0]["messages"][0]["content"] == [{"type": "text", "text": "grade it"}, _IMAGE]


@pytest.mark.asyncio
async def test_an_endpoint_refusing_the_schema_with_images_is_asked_again_without_it(judge, monkeypatch):
    refusal = Exception("response_format json_schema is not supported with image inputs")
    calls = _answer_with(monkeypatch, refusal, json.dumps(_VERDICT))

    out = await judge._prompt_llm_judge("grade it", model="openai/gpt-4o", context=TaskStepContext(),
                                        image_blocks=[_IMAGE])

    assert json.loads(out["response"]) == _VERDICT
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


@pytest.mark.asyncio
async def test_a_default_call_sends_one_user_message_and_returns_its_usage_and_cost(judge, monkeypatch):
    calls = _answer_with(monkeypatch, json.dumps(_VERDICT))

    out = await judge._prompt_llm_judge("grade it", model="openai/gpt-4o", context=TaskStepContext())

    assert calls[0]["messages"] == [{"role": "user", "content": "grade it"}]
    assert out["usage"]["total_tokens"] > 0
    assert (out["model"], out["endpoint_host"], out["cost_status"]) == ("openai/gpt-4o", "judge.invalid", "priced")
    assert out["cost_source"].startswith("litellm@") and out["cost_usd"] > 0 and out["reported_cost_usd"] is None


@pytest.mark.asyncio
async def test_a_system_message_and_a_shared_prefix_marked_for_the_prompt_cache(judge, monkeypatch):
    calls = _answer_with(monkeypatch, json.dumps(_VERDICT))

    await judge._prompt_llm_judge("THE RUN\n\nTHE CHECK", model="anthropic/claude-sonnet-4-6",
                                  context=TaskStepContext(), system="You grade the user sim.",
                                  cached_prefix="THE RUN\n\n")

    assert calls[0]["messages"] == [
        {"role": "system", "content": "You grade the user sim."},
        {"role": "user", "content": [
            {"type": "text", "text": "THE RUN\n\n", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "THE CHECK"},
        ]},
    ]


@pytest.mark.asyncio
async def test_a_shared_prefix_rides_before_the_screenshots(judge, monkeypatch):
    calls = _answer_with(monkeypatch, json.dumps(_VERDICT))

    await judge._prompt_llm_judge("THE RUN then the check", model="claude-sonnet-4-6", context=TaskStepContext(),
                                  image_blocks=[_IMAGE], cached_prefix="THE RUN")

    assert calls[0]["messages"][0]["content"] == [
        {"type": "text", "text": "THE RUN", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": " then the check"},
        _IMAGE,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "prompt"), [
    ("openai/gpt-4o", "THE RUN then the check"),  # not a Claude model: no cache marks
    ("anthropic/claude-sonnet-4-6", "a correction, then THE RUN"),  # doesn't start with the prefix
])
async def test_a_prefix_that_cant_be_cached_sends_the_prompt_whole(judge, monkeypatch, model, prompt):
    calls = _answer_with(monkeypatch, json.dumps(_VERDICT))

    await judge._prompt_llm_judge(prompt, model=model, context=TaskStepContext(), cached_prefix="THE RUN")

    assert calls[0]["messages"] == [{"role": "user", "content": prompt}]
