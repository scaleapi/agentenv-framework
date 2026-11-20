"""The direct judge sends LiteLLM the requests its golden file holds, first recorded on main before its call moved onto
``config.model_call``: the same model name, URL, key, params, headers and message shapes, keys in the same order, and
the same refusal of a URL with no key of its own. Re-record as ``tst/util/verifier_golden.py`` says."""

from __future__ import annotations

from types import SimpleNamespace

import litellm
import pytest

import agent_env.config as cfg_mod
from agent_env.config.errors import ConfigError
from agent_env.config.model import ModelConfig
from agent_env.store import LocalSecretStore
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep
from tst.util import verifier_golden

FILE = "judge_requests.json"
ENDPOINT = "https://gw.test/v1"
_IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
_REFUSAL = "response_format json_schema is not supported with image inputs"

CASES = {
    "gpt_through_the_endpoint": dict(model="gpt-4.1-mini"),
    "claude_through_the_endpoint": dict(model="claude-sonnet-4-6"),
    "anthropic_prefixed_through_a_base_without_v1": dict(model="anthropic/claude-sonnet-4-6", base_url="https://gw.test"),
    "gemini_through_the_endpoint_as_openai": dict(model="gemini-2.5-flash"),
    "a_name_litellm_cant_place": dict(model="qwen3:8b"),
    "a_prefixed_name_with_no_endpoint": dict(model="openai/gpt-4o", base_url=None),
    "params_and_headers": dict(model="claude-sonnet-4-6", params={"custom_llm_provider": "openai", "temperature": 0},
                               headers={"X-Team": "evals"}),
    "the_judges_own_url_and_key": dict(model="gpt-4.1-mini", headers={"X-Team": "evals"}, overrides={
        "judge_litellm_base_url": "https://judge.test/v1", "judge_litellm_api_key": "secret:judge-key"}),
    "the_runs_url_on_the_endpoint": dict(model="gpt-4.1-mini", headers={"X-Team": "evals"},
                                         overrides={"litellm_base_url": "https://GW.test/v1/"}),
    "the_steps_url_elsewhere_with_the_runs_key": dict(model="gpt-4.1-mini", step={"default_model_api_base":
                                                      "https://step.test/v1"}, overrides={"litellm_api_key": "run-key"}),
    "a_url_elsewhere_without_a_key": dict(model="gpt-4.1-mini", overrides={"judge_litellm_base_url": "https://x.test"}),
    "screenshots": dict(model="claude-sonnet-4-6", call={"image_blocks": [_IMAGE]}),
    "screenshots_refused_with_the_schema": dict(model="gpt-4.1-mini", call={"image_blocks": [_IMAGE]}, refuse=True),
    "a_system_message_and_a_cached_run": dict(model="claude-sonnet-4-6", call={
        "system": "You check the run.", "cached_prefix": "THE RUN"}),
    "a_cached_run_on_a_model_without_a_prompt_cache": dict(model="gpt-4.1-mini", call={"cached_prefix": "THE RUN"}),
    "a_timeout": dict(model="gpt-4.1-mini", step={"judge_timeout_seconds": 42}),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES)
async def test_the_direct_judge_sends_the_requests_it_always_has(monkeypatch, case):
    spec = CASES[case]
    config = cfg_mod.Config()
    config.set_secret_store(LocalSecretStore(values={"judge-key": "judge-secret"}, use_env=False))
    monkeypatch.setattr(config, "_model_cfg", ModelConfig(
        base_url=spec.get("base_url", ENDPOINT), api_key="configured-key", params=spec.get("params", {}),
        headers=spec.get("headers", {})))
    monkeypatch.setattr(cfg_mod, "get_config", lambda: config)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    requests: list[dict] = []

    async def acompletion(**kwargs):
        requests.append(kwargs)
        if spec.get("refuse") and len(requests) == 1:
            raise Exception(_REFUSAL)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"results": []}'))])

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    judge = RubricsVerifierTaskStep(id="v", version=1, criteria=[], prompt_id="p", use_agent_judge=False,
                                    verifier_id="vt", **spec.get("step", {}))
    context = TaskStepContext(metadata={"user_overrides": spec.get("overrides", {})})
    raised = None
    try:
        await judge._prompt_llm_judge("THE RUN then the check", model=spec["model"], context=context,
                                      **spec.get("call", {}))
    except ConfigError as error:
        raised = str(error)

    verifier_golden.check(FILE, case, {"requests": requests, "raised": raised})
