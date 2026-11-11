"""An MCP server env's ``secret_env``: env vars its server's container gets from the secret store, each kept as a
``secret:KEY`` reference in its record and env.toml and resolved only when its container is written, never logged."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import yaml

from agent_env.artifact import Artifact
from agent_env.config import get_config
from agent_env.env.envs.mcp_server import MCPServerEnv, secret_env_problems
from agent_env.env.envs.service_db import ServiceDBConfig
from agent_env.providers.env_providers.env_gateway_provider import (EnvironmentGatewayProvider, MCPServerConfig,
                                                                     _gateway_topology, _redact_compose_secrets)
from agent_env.providers.env_state import LocalPostgresStateProvider
from agent_env.store.secret_store import LocalSecretStore

TOKEN_JSON = '{"token": "ya29.$abc", "refresh_token": "1//r3fresh ${HOME}", "scopes": ["gmail.readonly"]}'
SECRET_ENV = {"WORKSPACE_MCP_TOKEN_JSON": "secret:env__gmail_token_json"}


def _env(**overrides) -> MCPServerEnv:
    fields = dict(id="gmail-mcp", version=1, docker_image_artifact=MagicMock(image_name="gmail-mcp:1"),
                  environment_name="gmail", secret_env=SECRET_ENV)
    return MCPServerEnv(**{**fields, **overrides})


@pytest.fixture
def store():
    config = get_config()
    config.set_secret_store(LocalSecretStore(values={"env__gmail_token_json": TOKEN_JSON,
                                                     "mongodb_uri": "mongodb://platform"}, use_env=False))
    yield
    config._secret_store = None


def test_its_record_and_env_toml_keep_the_reference_and_an_env_without_one_is_unchanged(monkeypatch):
    env = _env()
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: env.docker_image_artifact))
    data = env.to_dict()

    assert data["secret_env"] == SECRET_ENV
    assert MCPServerEnv.from_dict(data).secret_env == SECRET_ENV
    assert "secret_env" not in _env(secret_env=None).to_dict()


@pytest.mark.parametrize("secret_env, problem", [
    ({"T": "ya29.plain"}, "secret_env.T is a secret:KEY reference"),
    ({"T": "secret:env__k?fallback"}, "secret_env.T is a secret:KEY reference"),
    ({"T": "env:HOME"}, "secret_env.T is a secret:KEY reference"),
    ({"T": "secret:mongodb_uri"}, "an env may name only secret store keys starting 'env__'"),
    ({"DATABASE_URL": "secret:env__k"}, "secret_env.DATABASE_URL is set by the gateway itself"),
    ({"1T": "secret:env__k"}, "secret_env: '1T' isn't an env var's name"),
])
def test_a_secret_env_names_only_env_keys_by_reference(secret_env, problem):
    assert any(problem in p for p in secret_env_problems(secret_env))
    with pytest.raises(ValueError, match="MCP server env 'gmail-mcp'"):
        _env(secret_env=secret_env)


def test_its_compose_hands_the_resolved_value_quoted_and_the_log_masks_it(store):
    compose = EnvironmentGatewayProvider().create_docker_compose(
        mcp_servers=[MCPServerConfig(image="gmail-mcp:1", environment_name="gmail", secret_env=SECRET_ENV)],
        gateway_image="agent-gateway",
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
        state_instance=LocalPostgresStateProvider.default_instance())

    service = next(s for name, s in yaml.safe_load(compose)["services"].items() if "gmail" in name)
    handed = dict(item.split("=", 1) for item in service["environment"])
    # Compose reads $$ as a $ and interpolates no other.
    assert handed["WORKSPACE_MCP_TOKEN_JSON"].replace("$$", "$") == TOKEN_JSON
    assert "r3fresh" not in _redact_compose_secrets(compose)
    assert "WORKSPACE_MCP_TOKEN_JSON=***" in _redact_compose_secrets(compose)


def test_a_key_the_store_lacks_fails_the_compose(store):
    with pytest.raises(Exception, match="env__missing"):
        EnvironmentGatewayProvider().create_docker_compose(
            mcp_servers=[MCPServerConfig(image="i", environment_name="gmail",
                                         secret_env={"T": "secret:env__missing"})],
            state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
            state_instance=LocalPostgresStateProvider.default_instance())


def test_the_gateway_topology_carries_each_servers_secret_env():
    topology = _gateway_topology(_env())
    assert topology.mcp_servers[0].secret_env == SECRET_ENV
