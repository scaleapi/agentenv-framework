"""The rubrics judge's calls go through the installed litellm, not a stand-in.

``_prompt_llm_judge`` is the only caller of litellm: ``acompletion`` with a strict ``json_schema`` response_format over
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
    monkeypatch.setattr(config, "_model_cfg", ModelConfig())
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
