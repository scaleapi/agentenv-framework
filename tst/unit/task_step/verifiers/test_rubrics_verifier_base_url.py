"""The direct judge reaches the configured endpoint whether its base URL ends in ``/v1`` or not, whatever bare model
name it's given: the real litellm calls a loopback server that records each call's path and model name. A Claude
name keeps Anthropic's Messages route, its ``/v1`` not doubled; an OpenAI name gets the URL as configured; any other
bare name, one litellm can't place or one it would send to another provider's own API (``gemini-*`` to Vertex), goes
through the endpoint as ``openai/<name>``. A prefixed name keeps its provider's route. The rows without ``/v1`` on
Claude, every OpenAI row and the prefixed DeepSeek row called these same paths before."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pytest_socket import enable_socket

import agent_env.config as cfg_mod
from agent_env.config.model import ModelConfig
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.verifiers.rubrics_verifier import RubricsVerifierTaskStep

_VERDICT = json.dumps({"results": [{"criterion_id": "c1", "passed": True}]})


class _Endpoint(BaseHTTPRequestHandler):
    calls: list = []

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        self.calls.append((self.path, body.get("model")))
        if self.path.endswith("/messages"):
            answer = {"id": "m", "type": "message", "role": "assistant", "model": body.get("model"),
                      "content": [{"type": "text", "text": _VERDICT}], "stop_reason": "end_turn",
                      "usage": {"input_tokens": 1, "output_tokens": 1}}
        else:
            answer = {"id": "c", "object": "chat.completion", "created": 0, "model": body.get("model"),
                      "choices": [{"index": 0, "message": {"role": "assistant", "content": _VERDICT},
                                   "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        data = json.dumps(answer).encode()
        self.send_response(200 if self.path.endswith(("/messages", "/chat/completions")) else 404)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture(autouse=True)
def _disable_network():
    """Override the suite-wide socket block (tst/unit/conftest.py) for this module: the endpoint is on loopback."""
    enable_socket()
    yield


@pytest.fixture
def endpoint():
    _Endpoint.calls = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def _judge(monkeypatch, base_url: str, params: dict | None = None) -> RubricsVerifierTaskStep:
    config = cfg_mod.Config()
    monkeypatch.setattr(config, "_model_cfg", ModelConfig(base_url=base_url, api_key="k", params=params or {}))
    monkeypatch.setattr(cfg_mod, "get_config", lambda: config)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    return RubricsVerifierTaskStep(id="v", version=1, criteria=[], prompt_id="p", use_agent_judge=False,
                                   verifier_id="vt")


@pytest.mark.asyncio
@pytest.mark.parametrize("model, suffix, path, sent", [
    ("claude-sonnet-4-6", "", "/v1/messages", "claude-sonnet-4-6"),
    ("claude-sonnet-4-6", "/v1", "/v1/messages", "claude-sonnet-4-6"),
    ("claude-sonnet-4-6", "/v1/", "/v1/messages", "claude-sonnet-4-6"),
    ("anthropic/claude-sonnet-4-6", "/v1", "/v1/messages", "claude-sonnet-4-6"),
    ("gpt-4.1-mini", "", "/chat/completions", "gpt-4.1-mini"),
    ("gpt-4.1-mini", "/v1", "/v1/chat/completions", "gpt-4.1-mini"),
    ("openai/qwen3:8b", "/v1", "/v1/chat/completions", "qwen3:8b"),
    ("qwen3:8b", "", "/chat/completions", "qwen3:8b"),
    ("qwen3:8b", "/v1", "/v1/chat/completions", "qwen3:8b"),
    ("Qwen/Qwen3-8B", "/v1", "/v1/chat/completions", "Qwen/Qwen3-8B"),
    ("gemini-2.5-flash", "", "/chat/completions", "gemini-2.5-flash"),
    ("gemini-2.5-flash", "/v1", "/v1/chat/completions", "gemini-2.5-flash"),
    ("command-a-03-2025", "/v1", "/v1/chat/completions", "command-a-03-2025"),
    ("deepseek/deepseek-chat", "/v1", "/v1/chat/completions", "deepseek-chat"),
])
async def test_the_direct_judge_reaches_the_endpoint_however_its_base_url_ends(
    monkeypatch, endpoint, model, suffix, path, sent,
):
    judge = _judge(monkeypatch, endpoint + suffix)

    out = await judge._prompt_llm_judge("grade it", model=model, context=TaskStepContext())

    assert _Endpoint.calls == [(path, sent)]
    assert json.loads(out["response"]) == json.loads(_VERDICT)


@pytest.mark.asyncio
async def test_a_judge_whose_params_name_its_provider_keeps_that_providers_url(monkeypatch, endpoint):
    """``custom_llm_provider = "openai"`` in ``[model] params`` sends even a Claude name through the OpenAI route,
    whose base keeps its ``/v1``."""
    judge = _judge(monkeypatch, endpoint + "/v1", params={"custom_llm_provider": "openai"})

    await judge._prompt_llm_judge("grade it", model="claude-sonnet-4-6", context=TaskStepContext())

    assert _Endpoint.calls == [("/v1/chat/completions", "claude-sonnet-4-6")]
