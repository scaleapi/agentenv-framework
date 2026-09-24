"""The capability probes in ``tst/util/capabilities.py``."""

from __future__ import annotations

import pytest

from agent_env.config.errors import ConfigError
from tst.util import capabilities


class _FakeConfig:
    """Stands in for the resolved config; ``base_url=None`` is a profile with no ``[model]`` endpoint."""

    def __init__(
        self,
        base_url: str | None,
        *,
        default_model_error: Exception | None = None,
        base_url_error: Exception | None = None,
    ):
        self._base_url = base_url
        self._default_model_error = default_model_error
        self._base_url_error = base_url_error

    def get_default_model(self):
        if self._default_model_error is not None:
            raise self._default_model_error
        return "gpt-4o"

    def get_litellm_base_url(self) -> str:
        if self._base_url_error is not None:
            raise self._base_url_error
        if not self._base_url:
            raise ConfigError(
                "No model endpoint configured: set [model] base_url in "
                ".agentenv/config.toml or the LITELLM_BASE_URL env var."
            )
        return self._base_url


@pytest.fixture
def fake_config(monkeypatch):
    """Install a ``_FakeConfig`` as what ``get_config()`` returns."""

    def install(config):
        monkeypatch.setattr("agent_env.config.get_config", lambda: config)
        return config

    return install


def test_configured_endpoint_reports_the_capability_present(fake_config):
    fake_config(_FakeConfig("https://litellm.example.com"))
    assert capabilities.model_endpoint_is_configured() is True


def test_missing_endpoint_reports_the_capability_absent(fake_config):
    fake_config(_FakeConfig(None))
    assert capabilities.model_endpoint_is_configured() is False


def test_empty_endpoint_reports_the_capability_absent(fake_config):
    """An empty ``base_url`` is unconfigured, not a zero-length endpoint."""
    fake_config(_FakeConfig(""))
    assert capabilities.model_endpoint_is_configured() is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"default_model_error": ConfigError("[model] has unknown keys ['base_ur']")},
        {"base_url_error": RuntimeError("secret store unreachable")},
    ],
    ids=["malformed-model-section", "unreadable-config"],
)
def test_a_broken_config_reads_as_present_rather_than_absent(fake_config, kwargs):
    """Neither a raise (fails collection) nor False (hides an operator error) will do; True lets the test fail on the real problem."""
    fake_config(_FakeConfig(None, **kwargs))
    assert capabilities.model_endpoint_is_configured() is True


def test_a_broken_config_does_not_break_collection(fake_config):
    """The mark is built at import time, so building it must not raise either."""
    fake_config(_FakeConfig(None, base_url_error=RuntimeError("secret store unreachable")))
    assert capabilities.skip_without_model_endpoint().args == (False,)


def test_skip_reason_is_machine_readable():
    """CI greps this exact shape to assert the skips taken equal those declared."""
    reason = capabilities.missing_capability_reason(capabilities.MODEL_ENDPOINT)
    assert reason == "agentenv-capability-missing: model_endpoint_configured"
    assert reason.startswith(capabilities.MISSING_CAPABILITY_PREFIX)


def test_the_gate_skips_when_the_endpoint_is_absent(fake_config):
    fake_config(_FakeConfig(None))
    mark = capabilities.skip_without_model_endpoint()
    assert mark.name == "skipif"
    assert mark.args == (True,)
    assert mark.kwargs["reason"] == capabilities.missing_capability_reason(capabilities.MODEL_ENDPOINT)


def test_the_gate_does_not_skip_when_the_endpoint_is_present(fake_config):
    fake_config(_FakeConfig("https://litellm.example.com"))
    assert capabilities.skip_without_model_endpoint().args == (False,)


def _fake_default_agent_lookup(monkeypatch, outcome):
    from agent_env.store.base import NotFoundError  # noqa: F401  (documents the branch under test)

    def get(agent_id, version=None):
        assert agent_id == "the-default"
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("agent_env.a2a_agent.A2AAgent.get", staticmethod(get))
    monkeypatch.setattr(
        "agent_env.config.get_config",
        lambda: type("Cfg", (), {"get_default_a2a_agent_id": lambda self: "the-default"})(),
    )


def test_default_agent_present_when_registered(monkeypatch):
    _fake_default_agent_lookup(monkeypatch, object())
    assert capabilities.default_a2a_agent_is_registered() is True
    assert capabilities.skip_without_default_a2a_agent().args == (False,)


def test_default_agent_absent_when_not_in_the_store(monkeypatch):
    from agent_env.store.base import NotFoundError

    _fake_default_agent_lookup(monkeypatch, NotFoundError("A2AAgent the-default not found"))
    mark = capabilities.skip_without_default_a2a_agent()
    assert mark.args == (True,)
    assert mark.kwargs["reason"] == "agentenv-capability-missing: default_a2a_agent"


def test_default_agent_other_failures_read_as_present_rather_than_absent(monkeypatch):
    _fake_default_agent_lookup(monkeypatch, RuntimeError("store unreachable"))
    assert capabilities.default_a2a_agent_is_registered() is True


def _fake_e2b_builder(monkeypatch, outcome):
    def build(spec):
        assert spec == "e2b"
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("agent_env.providers.sandbox_provider.build_sandbox_provider", build)


def test_e2b_absent_when_its_config_is_missing(monkeypatch):
    _fake_e2b_builder(monkeypatch, ConfigError("[sandbox.providers.e2b.config] requires a non-empty 'base_template'"))
    assert capabilities.remote_sandbox_is_available("e2b") is False
    assert capabilities.skip_without_remote_sandbox("e2b").kwargs["reason"] == "agentenv-capability-missing: remote_sandbox"


def test_e2b_present_when_the_provider_builds(monkeypatch):
    _fake_e2b_builder(monkeypatch, object())
    assert capabilities.remote_sandbox_is_available("e2b") is True


def test_e2b_other_failures_read_as_present_rather_than_absent(monkeypatch):
    _fake_e2b_builder(monkeypatch, RuntimeError("api key rejected"))
    assert capabilities.remote_sandbox_is_available("e2b") is True
