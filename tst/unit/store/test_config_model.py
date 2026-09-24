"""Unit tests for the ``[model]`` gateway config.

Covers the getter precedence (env > [model] > actionable error — no built-in
endpoint), the open ``[model.roles]`` map, ``[model.params]`` pass-through +
reserved-key guard, and ``resolve_model_call``'s prefix rule — all AWS-free via a
fresh ``Config`` + injected ``LocalSecretStore``.
"""

import pytest

from agent_env.config.model import ModelCallConfig
from agent_env.store import ConfigError, LocalSecretStore
from agent_env.config import Config


def _cfg(tmp_path, monkeypatch, body=None, *, secrets=None, env=None):
    monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    if body is not None:
        agentenv = tmp_path / ".agentenv"
        agentenv.mkdir(exist_ok=True)
        (agentenv / "config.toml").write_text(body)
    cfg = Config()
    cfg.set_secret_store(LocalSecretStore(values=secrets or {"litellm_api_key": "sk-bundle"}, use_env=False))
    return cfg


_MODEL = '[model]\nbase_url = "https://gw.example/v1"\ndefault = "openrouter/x"\n'
_ROLES = '[model]\ndefault = "m-default"\n[model.roles]\njudge = "m-judge"\nagent = "m-agent"\n'


def test_no_config_base_url_raises_actionable(tmp_path, monkeypatch):
    with pytest.raises(ConfigError, match=r"\[model\] base_url"):
        _cfg(tmp_path, monkeypatch).get_litellm_base_url()


def test_no_config_api_key_raises_actionable(tmp_path, monkeypatch):
    with pytest.raises(ConfigError, match=r"\[model\] api_key"):
        _cfg(tmp_path, monkeypatch, secrets={"litellm_api_key": "sk-bundle"}).get_litellm_api_key()


def test_no_config_default_and_role_are_none(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch)
    assert cfg.get_default_model() is None
    assert cfg.get_model_for_role("judge") is None


def test_no_config_resolve_bare_model_raises_actionable(tmp_path, monkeypatch):
    with pytest.raises(ConfigError, match=r"\[model\] base_url"):
        _cfg(tmp_path, monkeypatch).resolve_model_call("claude-sonnet-4-6", default_api_key="k")


def test_no_config_prefixed_model_routes_natively(tmp_path, monkeypatch):
    call = _cfg(tmp_path, monkeypatch).resolve_model_call("bedrock/anthropic.claude", default_api_key="k")
    assert call == ModelCallConfig(api_key="k", api_base=None, params={})


def test_default_only_prefixed_model_routes_natively(tmp_path, monkeypatch):
    body = '[model]\ndefault = "bedrock/x"\n'
    call = _cfg(tmp_path, monkeypatch, body, secrets={"litellm_api_key": "sk-bundle"}).resolve_model_call("bedrock/x")
    assert call.api_base is None
    assert call.api_key is None


def test_provider_configured_prefixed_model_routes_natively(tmp_path, monkeypatch):
    body = '[model]\ndefault = "bedrock/x"\n[model.params]\naws_region_name = "us-west-2"\n'
    call = _cfg(tmp_path, monkeypatch, body).resolve_model_call("bedrock/x", default_api_key="k")
    assert call.api_base is None


def test_prefixed_model_gets_no_bundle_api_key(tmp_path, monkeypatch):
    body = '[model]\n[model.params]\naws_region_name = "us-west-2"\n'
    call = _cfg(tmp_path, monkeypatch, body, secrets={"litellm_api_key": "sk-bundle"}).resolve_model_call("bedrock/anthropic.claude")
    assert call.api_key is None
    assert call.api_base is None


def test_get_model_params_returns_schema_free_map(tmp_path, monkeypatch):
    body = ('[model]\ndefault = "bedrock/x"\n[model.params]\n'
            'aws_region_name = "us-west-2"\naws_secret_access_key = "secret:aws_sk"\n')
    cfg = _cfg(tmp_path, monkeypatch, body, secrets={"aws_sk": "resolved-sk"})
    assert cfg.get_model_params() == {"aws_region_name": "us-west-2", "aws_secret_access_key": "resolved-sk"}


def test_get_model_params_empty_when_unconfigured(tmp_path, monkeypatch):
    assert _cfg(tmp_path, monkeypatch).get_model_params() == {}


def test_get_model_params_merges_and_resolves_overrides(tmp_path, monkeypatch):
    body = '[model]\n[model.params]\naws_region_name = "us-west-2"\napi_version = "2024-01"\n'
    cfg = _cfg(tmp_path, monkeypatch, body, secrets={"tok": "resolved-tok"})
    merged = cfg.get_model_params({"api_version": "2024-09", "extra": "secret:tok"})
    assert merged == {"aws_region_name": "us-west-2", "api_version": "2024-09", "extra": "resolved-tok"}


def test_configured_proxy_shape_resolves_key_ref_for_bare_model(tmp_path, monkeypatch):
    """The platform-plugin shape: explicit base_url + a secret: api_key ref, resolved lazily."""
    body = '[model]\nbase_url = "https://proxy.example"\napi_key = "secret:litellm_api_key"\n'
    call = _cfg(tmp_path, monkeypatch, body, secrets={"litellm_api_key": "sk-bundle"}).resolve_model_call("claude-sonnet-4-6")
    assert call.api_key == "sk-bundle"
    assert call.api_base == "https://proxy.example"


def test_config_base_url_used(tmp_path, monkeypatch):
    assert _cfg(tmp_path, monkeypatch, _MODEL).get_litellm_base_url() == "https://gw.example/v1"


def test_env_base_url_beats_config(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, _MODEL, env={"LITELLM_BASE_URL": "https://env.example"})
    assert cfg.get_litellm_base_url() == "https://env.example"


def test_empty_env_base_url_falls_through_to_config(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, _MODEL, env={"LITELLM_BASE_URL": ""})
    assert cfg.get_litellm_base_url() == "https://gw.example/v1"


def test_config_base_url_honored_for_prefixed_model(tmp_path, monkeypatch):
    call = _cfg(tmp_path, monkeypatch, _MODEL).resolve_model_call("openrouter/x", default_api_key="k")
    assert call.api_base == "https://gw.example/v1"


def test_config_api_key_secret_resolves(tmp_path, monkeypatch):
    body = '[model]\napi_key = "secret:openrouter_key"\n'
    cfg = _cfg(tmp_path, monkeypatch, body, secrets={"openrouter_key": "sk-or", "litellm_api_key": "sk-bundle"})
    assert cfg.get_litellm_api_key() == "sk-or"


def test_env_api_key_beats_config(tmp_path, monkeypatch):
    body = '[model]\napi_key = "secret:openrouter_key"\n'
    cfg = _cfg(tmp_path, monkeypatch, body, secrets={"openrouter_key": "sk-or"}, env={"LITELLM_API_KEY": "sk-env"})
    assert cfg.get_litellm_api_key() == "sk-env"


def test_empty_env_api_key_falls_through_to_config(tmp_path, monkeypatch):
    body = '[model]\napi_key = "secret:openrouter_key"\n'
    cfg = _cfg(tmp_path, monkeypatch, body, secrets={"openrouter_key": "sk-or"}, env={"LITELLM_API_KEY": ""})
    assert cfg.get_litellm_api_key() == "sk-or"


def test_default_model(tmp_path, monkeypatch):
    assert _cfg(tmp_path, monkeypatch, _ROLES).get_default_model() == "m-default"


def test_role_lookup(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, _ROLES)
    assert cfg.get_model_for_role("judge") == "m-judge"
    assert cfg.get_model_for_role("agent") == "m-agent"


def test_missing_role_falls_back_to_default(tmp_path, monkeypatch):
    assert _cfg(tmp_path, monkeypatch, _ROLES).get_model_for_role("usersim") == "m-default"


def test_arbitrary_role_falls_back_to_default(tmp_path, monkeypatch):
    assert _cfg(tmp_path, monkeypatch, _ROLES).get_model_for_role("some_custom_role") == "m-default"


def test_model_overrides_tier(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, secrets={"ov_key": "sk-ov", "litellm_api_key": "sk-bundle"})
    overrides = {"m": {"secret_key": "ov_key", "base_url": "https://ov.example"}}
    assert cfg.resolve_model_call("m", model_overrides=overrides) == ModelCallConfig(
        api_key="sk-ov",
        api_base="https://ov.example",
        params={},
    )


def test_base_override_wins(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, monkeypatch, _MODEL)
    call = cfg.resolve_model_call("claude-sonnet-4-6", default_api_key="k", base_override="https://ovr")
    assert call.api_base == "https://ovr"


def test_params_splatted_with_resolved_secret(tmp_path, monkeypatch):
    body = (
        '[model]\ndefault = "bedrock/x"\n'
        '[model.params]\naws_region_name = "us-west-2"\naws_secret_access_key = "secret:aws_sk"\n'
    )
    cfg = _cfg(tmp_path, monkeypatch, body, secrets={"aws_sk": "SECRET", "litellm_api_key": "sk-bundle"})
    call = cfg.resolve_model_call("bedrock/x", default_api_key="k")
    assert call.params == {"aws_region_name": "us-west-2", "aws_secret_access_key": "SECRET"}
    assert call.api_base is None


@pytest.mark.parametrize("reserved", ["model", "messages", "api_key", "api_base", "user", "metadata", "timeout", "response_format"])
def test_reserved_param_key_fails_loud(tmp_path, monkeypatch, reserved):
    body = f'[model]\ndefault = "x"\n[model.params]\n{reserved} = "nope"\n'
    cfg = _cfg(tmp_path, monkeypatch, body)
    with pytest.raises(ConfigError, match="reserved"):
        cfg.get_default_model()
