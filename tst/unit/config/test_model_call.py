"""Core's one model call: its route and key, and its record of what it called and what it cost. A call's cost is what
the endpoint billed, a LiteLLM proxy's ``x-litellm-response-cost`` header, else LiteLLM's price map's figure, never
LiteLLM's own estimate. The real LiteLLM calls a loopback server on the OpenAI and the Anthropic routes."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import litellm
import pytest
from pytest_socket import enable_socket

import agent_env.config as cfg_mod
from agent_env.config import model_call
from agent_env.config.errors import ConfigError
from agent_env.config.model import ModelConfig
from agent_env.config.model_pricing import litellm_prices
from agent_env.store import LocalSecretStore

_HOW = "judge_litellm_api_key in the run's overrides"
_USER = [{"role": "user", "content": "grade it"}]


class _Endpoint(BaseHTTPRequestHandler):
    """Answers OpenAI's chat completions and Anthropic's messages with ``usage``, and the ``billed`` header when set."""

    billed: str | None = None
    usage: dict = {}

    def log_message(self, *args) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        if self.path.endswith("/messages"):
            answer = {"id": "m", "type": "message", "role": "assistant", "model": body.get("model"),
                      "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": self.usage}
        else:
            answer = {"id": "c", "object": "chat.completion", "created": 0, "model": body.get("model"),
                      "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                   "finish_reason": "stop"}], "usage": self.usage}
        data = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        if self.billed is not None:
            self.send_header("x-litellm-response-cost", self.billed)
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
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Endpoint)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        _Endpoint.billed, _Endpoint.usage = None, {}


def _configure(monkeypatch, base_url: str | None) -> None:
    config = cfg_mod.Config()
    config.set_secret_store(LocalSecretStore(values={"own": "own-key"}, use_env=False))
    monkeypatch.setattr(config, "_model_cfg", ModelConfig(base_url=base_url, api_key="configured-key"))
    monkeypatch.setattr(cfg_mod, "get_config", lambda: config)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)


async def _call(monkeypatch, endpoint: str, model: str, *, billed: str | None, usage: dict) -> model_call.ModelReply:
    _configure(monkeypatch, endpoint + "/v1")
    _Endpoint.billed, _Endpoint.usage = billed, usage
    route = model_call.resolve_route(model, api_key=None, api_base=None, who="judge", key_how=_HOW)
    return await model_call.complete(route, _USER, timeout=30)


_OPENAI_USAGE = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
_ANTHROPIC_USAGE = {"input_tokens": 100, "cache_read_input_tokens": 900, "output_tokens": 50}


@pytest.mark.asyncio
@pytest.mark.parametrize("model, usage", [("gpt-4.1-mini", _OPENAI_USAGE), ("claude-sonnet-4-6", _ANTHROPIC_USAGE)],
                         ids=["openai route", "anthropic route"])
async def test_a_call_a_proxy_bills_costs_what_it_billed(monkeypatch, endpoint, model, usage):
    reply = await _call(monkeypatch, endpoint, model, billed="0.0123", usage=usage)

    assert reply.content == "ok" and reply.model == model and reply.endpoint_host == endpoint.removeprefix("http://")
    assert reply.fields() | {"usage": None} == {
        "usage": None, "model": model, "endpoint_host": endpoint.removeprefix("http://"), "cost_usd": 0.0123,
        "cost_status": "reported", "cost_source": "provider", "reported_cost_usd": 0.0123}


@pytest.mark.asyncio
@pytest.mark.parametrize("model, usage, usd", [
    ("gpt-4.1-mini", _OPENAI_USAGE, (1000 * 0.4 + 100 * 1.6) / 1e6),
    ("claude-sonnet-4-6", _ANTHROPIC_USAGE, (100 * 3 + 900 * 0.3 + 50 * 15) / 1e6),
], ids=["openai route", "anthropic route, its cache reads at their own rate"])
async def test_a_call_no_one_billed_is_priced_from_litellms_map(monkeypatch, endpoint, model, usage, usd):
    reply = await _call(monkeypatch, endpoint, model, billed=None, usage=usage)

    assert reply.cost.status == "priced" and reply.cost.source == litellm_prices().source
    assert reply.cost.usd == pytest.approx(usd, rel=1e-9) and reply.cost.reported_usd is None
    assert reply.usage["completion_tokens"] == usage.get("completion_tokens", usage.get("output_tokens"))


@pytest.mark.asyncio
async def test_a_model_the_map_lacks_costs_what_it_was_billed_and_otherwise_nothing(monkeypatch, endpoint):
    billed = await _call(monkeypatch, endpoint, "qwen3:8b", billed="0.002", usage=_OPENAI_USAGE)
    unbilled = await _call(monkeypatch, endpoint, "qwen3:8b", billed=None, usage=_OPENAI_USAGE)

    assert (billed.cost.usd, billed.cost.status, billed.cost.source) == (0.002, "reported", "provider")
    assert unbilled.cost.fields() == {"cost_usd": None, "cost_status": "unpriced", "cost_source": None,
                                      "reported_cost_usd": None, "unpriced_reason": "no_price"}


def _answer(monkeypatch, hidden: dict) -> None:
    """Every call answered with LiteLLM's own response, its ``_hidden_params`` replaced by ``hidden``."""
    async def acompletion(**kwargs):
        response = litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "ok"}}],
                                         usage=litellm.Usage(prompt_tokens=1000, completion_tokens=100))
        response._hidden_params = hidden
        return response

    monkeypatch.setattr(litellm, "acompletion", acompletion)


@pytest.mark.asyncio
@pytest.mark.parametrize("model, status, usd", [("gpt-4.1-mini", "priced", 0.00056), ("qwen3:8b", "unpriced", None)])
async def test_litellms_own_estimate_is_never_the_cost(monkeypatch, model, status, usd):
    _configure(monkeypatch, "https://gw.test/v1")
    _answer(monkeypatch, {"response_cost": 9.99, "additional_headers": {}})
    route = model_call.resolve_route(model, api_key=None, api_base=None, who="judge", key_how=_HOW)

    reply = await model_call.complete(route, _USER, timeout=30)

    assert reply.cost.status == status and reply.cost.usd == pytest.approx(usd) and reply.cost.reported_usd is None


@pytest.mark.asyncio
@pytest.mark.parametrize("header", ["n/a", "-1", "nan", None])
async def test_a_header_that_isnt_a_cost_is_no_bill(monkeypatch, header):
    _configure(monkeypatch, "https://gw.test/v1")
    _answer(monkeypatch, {"additional_headers": {"llm_provider-x-litellm-response-cost": header}})
    route = model_call.resolve_route("gpt-4.1-mini", api_key=None, api_base=None, who="judge", key_how=_HOW)

    reply = await model_call.complete(route, _USER, timeout=30)

    assert reply.cost.status == "priced" and reply.cost.reported_usd is None


@pytest.mark.asyncio
async def test_a_response_without_usage_is_unpriced_for_want_of_tokens(monkeypatch):
    _configure(monkeypatch, "https://gw.test/v1")

    async def acompletion(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])

    monkeypatch.setattr(litellm, "acompletion", acompletion)
    route = model_call.resolve_route("gpt-4.1-mini", api_key=None, api_base=None, who="judge", key_how=_HOW)

    reply = await model_call.complete(route, _USER, timeout=30)

    assert (reply.content, reply.usage, reply.cost.status, reply.cost.unpriced) == ("", None, "unpriced", "no_tokens")


def test_a_url_elsewhere_needs_a_key_of_its_own(monkeypatch):
    _configure(monkeypatch, "https://gw.test/v1")

    with pytest.raises(ConfigError, match=r"^judge 'j': other.test isn't the configured model endpoint.*judge_litellm"):
        model_call.resolve_route("gpt-4.1-mini", api_key=None, api_base="https://other.test/v1", who="judge 'j'",
                                 key_how=_HOW)
    own = model_call.resolve_route("gpt-4.1-mini", api_key="secret:own", api_base="https://other.test/v1", who="j",
                                   key_how=_HOW)
    configured = model_call.resolve_route("claude-sonnet-4-6", api_key=None, api_base="https://GW.test/v1/", who="j",
                                          key_how=_HOW)

    assert (own.call.api_key, own.call.api_base, own.endpoint_host) == ("own-key", "https://other.test/v1", "other.test")
    assert (configured.call.api_key, configured.call.api_base) == ("configured-key", "https://GW.test")
    assert (configured.model, configured.litellm_model) == ("claude-sonnet-4-6", "claude-sonnet-4-6")


def test_a_prefixed_name_with_no_endpoint_goes_by_its_provider(monkeypatch):
    _configure(monkeypatch, None)

    route = model_call.resolve_route("openai/gpt-4o", api_key=None, api_base=None, who="j", key_how=_HOW)

    assert (route.litellm_model, route.call.api_base, route.endpoint_host) == ("openai/gpt-4o", None, None)


def _provider(name: str, custom_llm_provider: str | None = None):
    providers = {"claude-sonnet-4-6": "anthropic", "gpt-4.1-mini": "openai", "gemini-2.5-flash": "vertex_ai",
                 "deepseek/deepseek-chat": "deepseek"}
    if custom_llm_provider:
        return name, custom_llm_provider, None, None
    if name not in providers:
        print("Provider List")
        raise ValueError(name)
    return name, providers[name], None, None


def test_the_route_keeps_a_native_name_and_sends_an_unknown_one_as_openai(capsys):
    route = model_call.litellm_route
    assert route("claude-sonnet-4-6", "https://gw.example/v1", _provider) == ("claude-sonnet-4-6", "https://gw.example")
    assert route("gpt-4.1-mini", "https://gw.example/v1", _provider) == ("gpt-4.1-mini", "https://gw.example/v1")
    assert route("qwen3:8b", "https://gw.example", _provider) == ("openai/qwen3:8b", "https://gw.example")
    assert route("qwen3:8b", None, _provider) == ("qwen3:8b", None)
    assert route("claude-sonnet-4-6", "https://gw.example/v1", _provider, "openai") == (
        "claude-sonnet-4-6", "https://gw.example/v1")
    assert route("qwen3:8b", "https://gw.example", _provider, "hosted_vllm") == ("qwen3:8b", "https://gw.example")
    assert route("gemini-2.5-flash", "https://gw.example/v1", _provider) == (
        "openai/gemini-2.5-flash", "https://gw.example/v1")
    assert route("deepseek/deepseek-chat", "https://gw.example/v1", _provider) == (
        "deepseek/deepseek-chat", "https://gw.example/v1")
    assert route("gemini-2.5-flash", "https://gw.example", _provider, "vertex_ai") == (
        "gemini-2.5-flash", "https://gw.example")
    assert capsys.readouterr().out == ""


def test_setup_errors_are_the_ones_no_retry_fixes():
    setup = model_call.setup_errors()

    assert ConfigError in setup and litellm.AuthenticationError in setup and litellm.BudgetExceededError in setup
    assert not issubclass(litellm.Timeout, setup) and not issubclass(litellm.RateLimitError, setup)
