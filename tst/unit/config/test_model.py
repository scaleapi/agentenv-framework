"""Unit tests for the pure ``[model]`` resolver — no ``Config``, no AWS."""

import pytest

from agent_env.config.errors import ConfigError
from agent_env.config.model import (
    JUDGE_FALLBACK_MODEL,
    MODEL_PARAMS_RESERVED,
    ModelCallConfig,
    ModelConfig,
    ModelParam,
    ModelRole,
    ModelSource,
    ResolvedModel,
    config_model,
    named_model,
    resolve_model,
    same_endpoint,
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


def test_from_section_keeps_header_refs_raw():
    calls = []
    cfg = ModelConfig.from_section(
        {"base_url": "https://gw/v1", "headers": {"X-Bill-To": "secret:org", "X-Plain": "v"}},
        secret_resolver=lambda ref: calls.append(ref) or "resolved",
    )
    assert cfg.headers == {"X-Bill-To": "secret:org", "X-Plain": "v"}
    assert calls == []


@pytest.mark.parametrize("headers", [{"X-Retries": 3}, ["X-Bill-To"]])
def test_from_section_rejects_headers_that_are_not_strings(headers):
    with pytest.raises(ConfigError, match=r"\[model.headers\]"):
        ModelConfig.from_section({"headers": headers}, secret_resolver=lambda ref: None)


@pytest.mark.parametrize("headers, problem", [
    ({"X Bill To": "acme"}, "not an HTTP header name"),
    ({"X-Bill-To": "café"}, "value HTTP can't carry"),
    ({"X-Bill-To": "acme\r\nX-Injected: 1"}, "value HTTP can't carry"),
])
def test_from_section_rejects_a_header_http_cant_carry(headers, problem):
    with pytest.raises(ConfigError, match=problem):
        ModelConfig.from_section({"headers": headers}, secret_resolver=lambda ref: None)


# --- model_for_role -----------------------------------------------------------

def test_model_for_role_hit_and_fallback():
    cfg = ModelConfig(default="m-default", roles={"judge": "m-judge"})
    assert cfg.model_for_role("judge") == "m-judge"
    assert cfg.model_for_role("agent") == "m-default"
    assert ModelConfig().model_for_role("judge") is None


def test_a_role_asked_for_without_the_default_is_only_the_one_the_config_names():
    cfg = ModelConfig(default="m-default", roles={"user": "m-user"})
    assert cfg.model_for_role("user", or_default=False) == "m-user"
    assert cfg.model_for_role("judge", or_default=False) is None


@pytest.mark.parametrize("roles", [{"solver": "m-s", "user_simulator": "m-u"}, {"agent": "m-s", "user": "m-u"}],
                         ids=["canonical", "synonyms"])
def test_a_role_is_found_under_its_name_or_its_synonym_whichever_is_asked(roles):
    cfg = ModelConfig.from_section({"roles": roles}, secret_resolver=_secrets({}))
    for asked in ("solver", "agent", ModelRole.SOLVER):
        assert cfg.role_model(asked) == "m-s"
    for asked in ("user_simulator", "user", ModelRole.USER_SIMULATOR):
        assert cfg.role_model(asked) == "m-u"


def test_a_plugin_s_own_role_is_looked_up_by_its_name():
    assert ModelConfig(roles={"reviewer": "m-r"}).model_for_role("reviewer") == "m-r"


@pytest.mark.parametrize("roles, problem", [
    ({"agent": "a", "solver": "s"}, "sets both 'solver' and 'agent'"),
    ({"user": "a", "user_simulator": "s"}, "sets both 'user_simulator' and 'user'"),
    ({"judge": {"endpoint": "local", "model": "fast"}}, "maps each role to a model name"),
    ("claude", "maps each role to a model name"),
], ids=["solver-twice", "user-simulator-twice", "table", "not-a-table"])
def test_from_section_refuses_a_role_named_twice_or_mapped_to_anything_but_a_model(roles, problem):
    with pytest.raises(ConfigError, match=problem):
        ModelConfig.from_section({"roles": roles}, secret_resolver=_secrets({}))


# --- resolve_model ------------------------------------------------------------

_LAYERS = ("step_params", "run", "step", "agent")


@pytest.mark.parametrize("first", range(len(_LAYERS)), ids=_LAYERS)
@pytest.mark.parametrize("role", list(ModelRole))
def test_the_first_layer_that_names_a_model_wins(role, first):
    given = {layer: f"m-{layer}" for layer in _LAYERS[first:]}
    resolved = resolve_model(role, config=ModelConfig(default="m-default", roles={role.value: "m-role"}), **given)
    assert resolved == ResolvedModel(role, f"m-{_LAYERS[first]}", ModelSource(_LAYERS[first]))


@pytest.mark.parametrize("role, config, expected", [
    (ModelRole.SOLVER, ModelConfig(default="D", roles={"agent": "A"}), ("A", ModelSource.CONFIG_ROLE)),
    (ModelRole.SOLVER, ModelConfig(default="D"), ("D", ModelSource.CONFIG_DEFAULT)),
    (ModelRole.SOLVER, ModelConfig(), (None, None)),
    (ModelRole.USER_SIMULATOR, ModelConfig(default="D", roles={"user": "U"}), ("U", ModelSource.CONFIG_ROLE)),
    (ModelRole.USER_SIMULATOR, ModelConfig(default="D"), (None, None)),
    (ModelRole.JUDGE, ModelConfig(default="D", roles={"judge": "J"}), ("J", ModelSource.CONFIG_ROLE)),
    (ModelRole.JUDGE, ModelConfig(default="D"), ("D", ModelSource.CONFIG_DEFAULT)),
    (ModelRole.JUDGE, ModelConfig(), (JUDGE_FALLBACK_MODEL, ModelSource.FALLBACK)),
    (ModelRole.JUDGE, None, (JUDGE_FALLBACK_MODEL, ModelSource.FALLBACK)),
    (ModelRole.SOLVER, None, (None, None)),
], ids=["solver-role", "solver-default", "solver-none", "user-role", "user-never-the-default", "judge-role",
        "judge-default", "judge-fallback", "judge-without-config", "solver-without-config"])
def test_with_no_layer_the_config_decides_per_role(role, config, expected):
    resolved = resolve_model(role, config=config)
    assert (resolved.model, resolved.source) == expected


def test_resolving_reads_no_environment_and_no_secret(monkeypatch):
    def refuse(key):
        raise AssertionError(f"resolution read the secret {key!r}")

    monkeypatch.setattr("os.environ", {})
    cfg = ModelConfig.from_section({"api_key": "secret:litellm_api_key", "headers": {"X-Org": "secret:org"},
                                    "default": "D", "roles": {"judge": "J"}}, secret_resolver=refuse)
    assert [resolve_model(role, config=cfg).model for role in ModelRole] == ["D", None, "J"]


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


def test_base_override_to_a_foreign_url_without_a_key_is_refused():
    """The configured key is scoped to the configured endpoint, and a call there with no key would have litellm send
    whatever provider key the environment holds."""
    with pytest.raises(ConfigError, match="elsewhere.example isn't the configured model endpoint"):
        ModelConfig(base_url="https://proxy.example", api_key="secret:k").resolve_call(
            "m", base_override="https://elsewhere.example", secret_resolver=_secrets({"k": "sk-proxy"}))


@pytest.mark.parametrize("override", ["https://proxy.example", "https://proxy.example/", "https://proxy.example/v1",
                                      "HTTPS://Proxy.Example:443"])
def test_base_override_equal_to_the_configured_base_keeps_the_key(override):
    call = ModelConfig(base_url="https://proxy.example", api_key="secret:k").resolve_call(
        "m", base_override=override, secret_resolver=_secrets({"k": "sk-proxy"})
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


# --- headers ------------------------------------------------------------------

_HEADERS = {"X-Bill-To": "secret:org", "X-Plain": "v"}
_RESOLVED = {"X-Bill-To": "acme", "X-Plain": "v"}
_WITH_HEADERS = ModelConfig(base_url="https://gw/v1", headers=_HEADERS)


@pytest.mark.parametrize("configured, env_base, asked, expected", [
    ("https://gw/v1", None, "https://gw/v1", _RESOLVED),
    ("https://gw/v1", None, "https://gw/v1/", _RESOLVED),
    ("https://gw/v1/", None, "https://gw/v1", _RESOLVED),
    ("https://gw/v1", "https://env-gw/v1", "https://env-gw/v1", _RESOLVED),
    ("https://gw/v1", "https://env-gw/v1", "https://gw/v1", {}),
    ("https://gw/v1", None, "https://elsewhere.example/v1", {}),
    ("https://gw/v1", None, None, {}),
    (None, None, "https://gw/v1", {}),
], ids=["configured", "trailing-slash-asked", "trailing-slash-configured", "env-configured", "file-shadowed-by-env",
        "elsewhere", "no-url", "no-endpoint"])
def test_headers_go_only_to_the_configured_endpoint_and_resolve_only_for_it(monkeypatch, configured, env_base, asked,
                                                                           expected):
    if env_base:
        monkeypatch.setenv("LITELLM_BASE_URL", env_base)
    resolved = []

    headers = ModelConfig(base_url=configured, headers=_HEADERS).headers_for(
        asked, secret_resolver=lambda key: resolved.append(key) or {"org": "acme"}.get(key))

    assert headers == expected
    assert bool(resolved) is bool(expected)


def test_headers_held_back_from_another_endpoint_are_logged(caplog):
    with caplog.at_level("INFO", logger="agent_env.config.model"):
        _WITH_HEADERS.headers_for("https://elsewhere.example/v1", secret_resolver=_secrets({}))

    assert "Not sending [model.headers] to elsewhere.example" in caplog.text


@pytest.mark.parametrize("secrets, problem", [
    ({}, "Unresolved secret:org"),
    ({"org": "acme\nX-Injected: 1"}, "value HTTP can't carry"),
], ids=["unresolvable", "resolves-to-a-value-http-cant-carry"])
def test_a_header_ref_that_cant_be_sent_fails_loud(secrets, problem):
    with pytest.raises(ConfigError, match=problem):
        _WITH_HEADERS.headers_for("https://gw/v1", secret_resolver=_secrets(secrets))


@pytest.mark.parametrize("call, expected", [
    ({}, {"extra_headers": _RESOLVED}),
    ({"base_override": "https://elsewhere.example", "default_api_key": "theirs"}, {}),
], ids=["configured", "foreign-override"])
def test_resolve_call_sends_the_headers_only_to_the_configured_endpoint(call, expected):
    assert _WITH_HEADERS.resolve_call("m", secret_resolver=_secrets({"org": "acme", "k": "sk"}), **call).params == expected


def test_params_extra_headers_win_per_header_whatever_its_case():
    cfg = ModelConfig(base_url="https://gw/v1", headers={"X-Bill-To": "acme", "X-Plain": "v"},
                      params={"temperature": 0, "extra_headers": {"x-bill-to": "other"}})
    call = cfg.resolve_call("m", secret_resolver=_secrets({}))
    assert call.params == {"temperature": 0, "extra_headers": {"X-Plain": "v", "x-bill-to": "other"}}
    assert cfg.params == {"temperature": 0, "extra_headers": {"x-bill-to": "other"}}


def test_each_role_records_under_the_key_metadata_models_has_always_used():
    assert {role: role.record_key for role in ModelRole} == {
        ModelRole.SOLVER: "solver", ModelRole.USER_SIMULATOR: "user simulator", ModelRole.JUDGE: "judge"}


def test_no_named_layer_leaves_the_role_to_its_config():
    assert named_model(ModelRole.SOLVER) is None
    assert config_model(ModelRole.SOLVER, ModelConfig(default="D")) == ResolvedModel(
        ModelRole.SOLVER, "D", ModelSource.CONFIG_DEFAULT)


@pytest.mark.parametrize("a, b", [
    ("https://gw.example/v1", "https://gw.example"),
    ("https://gw.example/v1/", "https://gw.example/"),
    ("HTTPS://GW.Example/v1", "https://gw.example/v1"),
    ("https://gw.example:443/v1", "https://gw.example/v1"),
    ("http://gw.example:80", "http://gw.example"),
    ("https://gw.example/litellm/v1", "https://gw.example/litellm"),
    (" https://gw.example ", "https://gw.example"),
], ids=["v1", "slashes", "case", "https-default-port", "http-default-port", "path-prefix", "whitespace"])
def test_same_endpoint_reads_one_endpoint_spelled_two_ways_as_one(a, b):
    assert same_endpoint(a, b) and same_endpoint(b, a)


@pytest.mark.parametrize("a, b", [
    ("https://gw.example", "http://gw.example"),
    ("https://gw.example", "https://gw.example:8443"),
    ("https://gw.example", "https://other.example"),
    ("https://gw.example/team-a", "https://gw.example/team-b"),
    ("https://gw.example/Team", "https://gw.example/team"),
    ("https://gw.example/v1/v1", "https://gw.example"),
    ("https://gw.example?team=a", "https://gw.example?team=b"),
    ("https://gw.example", None),
    (None, None),
    ("", ""),
], ids=["scheme", "port", "host", "path", "path-case", "two-v1s", "query", "none", "both-none", "both-empty"])
def test_same_endpoint_tells_two_endpoints_apart(a, b):
    assert not same_endpoint(a, b)
