"""The one place core calls a model itself, as its judges do: the route a call takes, the key it carries, and what it
called and what it cost. Nothing else in core imports LiteLLM (``tst/unit/config/test_one_model_call_path.py``), and
this module imports it only where it's used, since its import takes seconds."""

from __future__ import annotations

import asyncio
import contextlib
import io
import math
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional

from agentenv_protocol.a2a_agent import Usage

from agent_env.config.describe import is_secret_key, mask_value
from agent_env.config.errors import ConfigError
from agent_env.config.model import ModelCallConfig, ModelParam, api_root, endpoint_host, no_key_for
from agent_env.config.model_pricing import Cost, ReportedBy, Tokens, litellm_prices, price
from agent_env.config.runtime import get_config


@dataclass(frozen=True)
class ModelRoute:
    """How a call to ``model`` goes: the name and base LiteLLM is handed (``litellm_model``, ``call.api_base``) and
    the key and params it carries. ``model`` is what the call records."""

    model: str
    litellm_model: str
    call: ModelCallConfig

    @property
    def endpoint_host(self) -> Optional[str]:
        return endpoint_host(self.call.api_base)


@dataclass(frozen=True)
class ModelReply:
    """A call's answer, the model it called and the host it reached, its usage as the response carried it, and its
    cost: what the endpoint billed when it says, else LiteLLM's price map's figure, else unpriced."""

    content: str
    model: str
    endpoint_host: Optional[str]
    usage: Optional[dict]
    cost: Cost

    def fields(self) -> dict[str, Any]:
        """The record of the call, beside its answer: ``usage``, ``model``, ``endpoint_host`` and the cost's keys."""
        return {"usage": self.usage, "model": self.model, "endpoint_host": self.endpoint_host, **self.cost.fields()}


def resolve_route(model: str, *, api_key: Optional[str], api_base: Optional[str], who: str,
                  key_how: str) -> ModelRoute:
    """The route a call to ``model`` takes, given the key and base its caller was handed (either may be None, a key an
    ``env:`` or ``secret:`` reference). A base other than the configured endpoint needs a key of its own, since the
    configured one isn't sent there: without one this refuses, as ``who``, saying ``key_how`` to give one."""
    import litellm

    config = get_config()
    key = config.resolve_reference(api_key)
    if api_base and not key and not config.is_model_endpoint(api_base):
        raise ConfigError(f"{who}: {no_key_for(api_base, key_how)}")
    call = config.resolve_model_call(model, default_api_key=key, base_override=api_base)
    name, base = litellm_route(model, call.api_base, litellm.get_llm_provider, call.params.get("custom_llm_provider"))
    return ModelRoute(model, name, replace(call, api_base=base))


def litellm_route(model: str, api_base: Optional[str], provider_of: Callable[..., Any],
                  custom_provider: Optional[str] = None) -> tuple[str, Optional[str]]:
    """The model name and base a call hands LiteLLM. Through an endpoint, a bare name goes by what LiteLLM makes of
    it: an Anthropic one keeps that route, with the base's trailing ``/v1`` dropped since its client appends
    ``/v1/messages``; an OpenAI one goes as configured; any other, which LiteLLM would send to that provider's own API
    (``gemini-*`` to Vertex), or one it can't place, goes through the endpoint as ``openai/<name>``. A provider prefix
    in the name, or a ``custom_llm_provider`` among the call's params, keeps the route it names. ``provider_of`` is
    LiteLLM's ``get_llm_provider``, which prints a banner before refusing a name it can't place."""
    if api_base is None:
        return model, None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            provider = provider_of(model, custom_llm_provider=custom_provider)[1]
    except Exception:
        return f"openai/{model}", api_base
    if provider == "anthropic":
        return model, api_root(api_base)
    if provider == "openai" or custom_provider or "/" in model:
        return model, api_base
    return f"openai/{model}", api_base


async def complete(route: ModelRoute, messages: list[dict], *, timeout: Optional[float],
                   response_format: Optional[dict] = None, settings: Optional[dict[str, Any]] = None) -> ModelReply:
    """One chat completion by ``route``, raising what LiteLLM raises. ``settings`` are LiteLLM params for this call, such
    as ``reasoning_effort``, over the config's ``[model.params]``; a caller sends only those ``takes`` says the route
    takes."""
    import litellm

    request: dict[str, Any] = {ModelParam.MODEL: route.litellm_model, ModelParam.MESSAGES: messages,
                               ModelParam.TIMEOUT: timeout, **route.call.client_kwargs(), **(settings or {})}
    if response_format is not None:
        request[ModelParam.RESPONSE_FORMAT] = response_format
    response = await litellm.acompletion(**request)
    usage = getattr(response, "usage", None)
    usage = usage.model_dump() if hasattr(usage, "model_dump") else usage
    usage = usage if isinstance(usage, dict) else None
    tokens = Tokens.of(Usage.from_model_response(usage).to_dict()) if usage else None
    cost = price(tokens, _billed(getattr(response, "_hidden_params", None)), model=route.model,
                 prices=litellm_prices(), reported_by=ReportedBy.PROVIDER)
    return ModelReply(response.choices[0].message.content or "", route.model, route.endpoint_host, usage, cost)


def sent_params(route: ModelRoute, settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The params a call by ``route`` with ``settings`` sends (``complete``), the config's ``[model.params]`` with
    ``settings`` over them, as a record of the call keeps them: a value under a secret-shaped key (``config show``'s
    rule, which takes in ``extra_headers``) is left out, through tables and lists, unless it's a number, as
    ``max_tokens`` is, and a URL's userinfo is masked."""
    return _without_secrets({**route.call.params, **(settings or {})})


def _without_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _without_secrets(item) for key, item in value.items()
                if not is_secret_key(key) or isinstance(item, (int, float))}
    if isinstance(value, list):
        return [_without_secrets(item) for item in value]
    return mask_value("", value)


def takes(route: ModelRoute, param: str) -> bool:
    """Whether LiteLLM sends ``param`` (``reasoning_effort``, ``thinking``) on ``route``: one of the params it takes for
    the route's provider and model, as LiteLLM's own table says. A route it can't place takes none."""
    import litellm

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            model, provider, _, _ = litellm.get_llm_provider(
                route.litellm_model, custom_llm_provider=route.call.params.get("custom_llm_provider"))
        supported = litellm.get_supported_openai_params(model=model, custom_llm_provider=provider)
    except Exception:
        return False
    return param in (supported or ())


def refused_setting(exc: BaseException) -> bool:
    """Whether ``exc`` is LiteLLM refusing a param's value for the model before sending the call, as GPT-5 refuses
    ``reasoning_effort="xhigh"``: the call wasn't made, so nothing was spent."""
    import litellm

    return isinstance(exc, litellm.UnsupportedParamsError)


def cut_off(exc: BaseException) -> bool:
    """Whether ``exc`` stopped a call that may have reached the model, so it may be billed with no answer: a cancel, or
    LiteLLM's ``Timeout``, raised itself or as the cause of a failure after retries. A timeout LiteLLM's client retried
    on its own and then answered (the OpenAI SDK's ``max_retries``) never reaches the caller, so it isn't seen."""
    import litellm

    return isinstance(exc, asyncio.CancelledError) or any(
        isinstance(e, litellm.Timeout) for e in (exc, exc.__cause__))


def setup_errors() -> tuple[type[BaseException], ...]:
    """What a call raises when it can't work as set up, as opposed to a flaky failure: agent-env's ConfigError (no key,
    missing config) and LiteLLM's AuthenticationError, PermissionDeniedError, NotFoundError (an unknown model),
    BadRequestError (with its kinds: an unknown provider, unsupported params, a prompt over the context window),
    UnprocessableEntityError and BudgetExceededError."""
    import litellm

    return (ConfigError, litellm.AuthenticationError, litellm.PermissionDeniedError, litellm.NotFoundError,
            litellm.BadRequestError, litellm.UnprocessableEntityError, litellm.BudgetExceededError)


def _billed(hidden: object) -> Optional[float]:
    """What the endpoint billed for the call: a LiteLLM proxy's ``x-litellm-response-cost`` header, as LiteLLM reads
    it, never LiteLLM's own estimate, which its ``response_cost`` holds when the header is missing. None without the
    header, or for one that isn't a cost."""
    from litellm.cost_calculator import get_response_cost_from_hidden_params

    if not isinstance(hidden, dict):
        return None
    try:
        billed = get_response_cost_from_hidden_params(hidden)
    except (TypeError, ValueError):
        return None
    return billed if billed is not None and math.isfinite(billed) and billed >= 0 else None
