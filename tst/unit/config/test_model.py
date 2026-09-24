"""Unit tests for the pure ``[model]`` resolver — no ``Config``, no AWS."""

import pytest

from agent_env.config.errors import ConfigError
from agent_env.config.model import (
    MODEL_PARAMS_RESERVED,
    ModelCallConfig,
    ModelConfig,
    ModelParam,
)


@pytest.fixture(autouse=True)
def _clear_litellm_env(monkeypatch):
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)


def _secrets(values):
    return lambda key: values.get(key)


# --- ModelParam ---------------------------------------------------------------

def test_reserved_set_is_derived_from_enum():
    assert MODEL_PARAMS_RESERVED == {p.value for p in ModelParam}


def test_model_param_members_are_plain_strings():
    assert ModelParam.API_BASE == "api_base"
    assert {ModelParam.API_BASE: 1}["api_base"] == 1


# --- ModelConfig.from_section -------------------------------------------------

def test_from_section_resolves_secret_refs_except_api_key():
    """api_key keeps its raw reference — resolving it eagerly would make every
    ``[model]`` read (e.g. get_litellm_base_url) need secret-store credentials."""
    calls = []
    cfg = ModelConfig.from_section(
        {"base_url": "secret:gw", "api_key": "secret:k", "params": {"aws_region_name": "us-west-2"}},
        secret_resolver=lambda ref: calls.append(ref) or "resolved",
    )
    assert cfg == ModelConfig(base_url="resolved", api_key="secret:k", params={"aws_region_name": "us-west-2"})
    assert calls == ["gw"]


@pytest.mark.parametrize("key", sorted(MODEL_PARAMS_RESERVED))
def test_from_section_rejects_reserved_params_key(key):
    with pytest.raises(ConfigError, match="reserved"):
        ModelConfig.from_section({"params": {key: "x"}}, secret_resolver=lambda ref: None)


def test_from_section_rejects_unknown_key():
    with pytest.raises(ConfigError, match="unknown"):
        ModelConfig.from_section({"base_ur": "typo"}, secret_resolver=lambda ref: None)


# --- model_for_role -----------------------------------------------------------

def test_model_for_role_hit_and_fallback():
    cfg = ModelConfig(default="m-default", roles={"judge": "m-judge"})
    assert cfg.model_for_role("judge") == "m-judge"
    assert cfg.model_for_role("agent") == "m-default"
    assert ModelConfig().model_for_role("judge") is None


# --- resolve_api_key ----------------------------------------------------------

def test_api_key_env_wins_and_refs_stay_lazy(monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "sk-env")
    calls = []
    got = ModelConfig(api_key="secret:k").resolve_api_key(
        secret_resolver=lambda k: calls.append(k) or "sk-secret"
    )
    assert got == "sk-env"
    assert calls == []


def test_api_key_literal_and_secret_ref_resolve():
    assert ModelConfig(api_key="sk-cfg").resolve_api_key(secret_resolver=_secrets({})) == "sk-cfg"
    assert (
        ModelConfig(api_key="secret:k").resolve_api_key(secret_resolver=_secrets({"k": "sk-secret"}))
        == "sk-secret"
    )


def test_api_key_unconfigured_raises_actionable():
    with pytest.raises(ConfigError, match=r"\[model\] api_key"):
        ModelConfig().resolve_api_key(secret_resolver=_secrets({}))


# --- resolve_call -------------------------------------------------------------

def test_resolve_call_inline_override_wins():
    call = ModelConfig().resolve_call(
        "m",
        model_overrides={"m": {"secret_key": "k", "base_url": "https://ov"}},
        secret_resolver=_secrets({"k": "sk-k"}),
    )
    assert call == ModelCallConfig(api_key="sk-k", api_base="https://ov", params={})


def test_resolve_call_override_with_missing_secret_raises():
    with pytest.raises(ConfigError, match="no value"):
        ModelConfig().resolve_call(
            "m",
            model_overrides={"m": {"secret_key": "absent", "base_url": "https://ov"}},
            secret_resolver=_secrets({}),
        )


def test_resolve_call_bare_model_without_endpoint_raises_actionable():
    with pytest.raises(ConfigError, match=r"\[model\] base_url"):
        ModelConfig().resolve_call("claude-sonnet-4-6", secret_resolver=_secrets({}))


def test_resolve_call_prefixed_model_routes_native_without_config():
    call = ModelConfig(params={"aws_region_name": "us-west-2"}).resolve_call(
        "bedrock/x", secret_resolver=_secrets({})
    )
    assert call == ModelCallConfig(api_key=None, api_base=None, params={"aws_region_name": "us-west-2"})


def test_resolve_call_api_key_ref_resolves_lazily():
    call = ModelConfig(base_url="https://gw/v1", api_key="secret:k").resolve_call(
        "claude-sonnet-4-6", secret_resolver=_secrets({"k": "sk-secret"})
    )
    assert call == ModelCallConfig(api_key="sk-secret", api_base="https://gw/v1", params={})


def test_base_override_to_a_foreign_url_does_not_inherit_the_configured_key():
    """The configured key is scoped to the configured endpoint — a caller-supplied
    override URL must not receive the configured proxy's credential."""
    call = ModelConfig(base_url="https://proxy.example", api_key="secret:k").resolve_call(
        "m", base_override="https://elsewhere.example", secret_resolver=_secrets({"k": "sk-proxy"})
    )
    assert call.api_key is None
    assert call.api_base == "https://elsewhere.example"


def test_base_override_equal_to_the_configured_base_keeps_the_key():
    call = ModelConfig(base_url="https://proxy.example", api_key="secret:k").resolve_call(
        "m", base_override="https://proxy.example", secret_resolver=_secrets({"k": "sk-proxy"})
    )
    assert call.api_key == "sk-proxy"


def test_resolve_call_default_api_key_and_base_override():
    call = ModelConfig().resolve_call(
        "bedrock/x",
        default_api_key="k",
        base_override="https://over",
        secret_resolver=_secrets({}),
    )
    assert call == ModelCallConfig(api_key="k", api_base="https://over", params={})


def test_resolve_call_cfg_base_url_honored_for_prefixed_model():
    call = ModelConfig(base_url="https://gw/v1", api_key="sk-or").resolve_call(
        "openrouter/x", secret_resolver=_secrets({})
    )
    assert call == ModelCallConfig(api_key="sk-or", api_base="https://gw/v1", params={})


def test_client_kwargs_omits_none_api_base_and_splats_params():
    cc = ModelCallConfig(api_key="k", api_base=None, params={"aws_region_name": "us-west-2"})
    assert cc.client_kwargs() == {"api_key": "k", "aws_region_name": "us-west-2"}
    assert ModelCallConfig("k", "https://b", {}).client_kwargs() == {"api_key": "k", "api_base": "https://b"}
