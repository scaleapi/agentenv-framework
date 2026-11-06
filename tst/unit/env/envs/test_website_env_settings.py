"""A website env's settings: environment variables of its server's container, a live site's credentials among them.
A secret is stored as its reference, never its value, and resolved from the deploying user's config only into the
compose file the gateway runs."""

import pytest
import yaml

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.env import Env
from agent_env.env.envs.service_db import ServiceDBConfig
from agent_env.env.envs.website import WebsiteEnv, settings_problems
from agent_env.providers.env_providers.env_gateway_provider import (
    EnvironmentGatewayProvider,
    _redact_compose_secrets,
    website_config,
)
from agent_env.providers.env_state import LocalPostgresStateProvider


def _image(id):
    return get_artifact_store().put_document(DockerImageArtifact(
        id=id, description="", image_name=f"registry.example/{id}@sha256:" + "a" * 64))


def test_a_secret_setting_is_stored_as_its_reference(local_stores):
    WebsiteEnv.put(id="github-site", server_docker_image_artifact=_image("github-server"), environment_name="github",
                   settings={"GITHUB_TOKEN": "secret:github-token", "GITHUB_ORG": "acme"})
    stored = Env.get("github-site")
    assert stored.settings == {"GITHUB_TOKEN": "secret:github-token", "GITHUB_ORG": "acme"}
    assert stored.to_dict()["settings"] == stored.settings
    assert "settings" not in WebsiteEnv.put(id="plain", server_docker_image_artifact=_image("plain-server"),
                                            environment_name="plain").to_dict()


@pytest.mark.parametrize("settings, problem", [
    ({"GITHUB_TOKEN": "ghp_the_value_itself"}, "settings.GITHUB_TOKEN is a secret"),
    ({"API_PASSWORD": "secret:pw?fallback"}, "with no ?default"),
    ({"DATABASE_URL": "postgres://elsewhere"}, "the gateway doesn't set itself"),
    ({"not a name": "x"}, "an environment variable's name"),
    ({"RETRIES": 3}, "settings.RETRIES is a string, not a int"),
])
def test_a_setting_that_isnt_one_is_refused_without_its_value(settings, problem):
    (found,) = settings_problems(settings)
    assert problem in found
    assert "ghp_the_value_itself" not in found and "fallback" not in found
    with pytest.raises(ValueError, match="website env 'x'"):
        WebsiteEnv(id="x", version=None, environment_name="x", settings=settings)


def test_the_compose_file_holds_the_resolved_settings_and_the_gateway_config_the_references(local_stores,
                                                                                            monkeypatch):
    env = WebsiteEnv(id="github-site", version=None, server_docker_image_artifact=_image("github-server"),
                     environment_name="github", settings={"GITHUB_TOKEN": "env:TEST_GITHUB_TOKEN", "ORG": "acme"})
    monkeypatch.setenv("TEST_GITHUB_TOKEN", "ghp_resolved")
    config = website_config(env)
    assert config.settings == {"GITHUB_TOKEN": "env:TEST_GITHUB_TOKEN", "ORG": "acme"}
    compose = EnvironmentGatewayProvider().create_docker_compose(
        mcp_servers=[], website_configs=[config],
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
        state_instance=LocalPostgresStateProvider.default_instance())
    environment = yaml.safe_load(compose)["services"]["github-website-backend"]["environment"]
    assert "GITHUB_TOKEN=ghp_resolved" in environment
    assert "ORG=acme" in environment


def test_a_setting_holding_compose_syntax_stays_one_variable(local_stores):
    hostile = 'ok\n    privileged: true\n    volumes:\n      - /:/host\n# "q" \\ $HOME ${X} \u2028 é 😀 \x85'
    env = WebsiteEnv(id="site", version=None, server_docker_image_artifact=_image("site-server"),
                     environment_name="site", settings={"EXTRA": hostile})
    compose = EnvironmentGatewayProvider().create_docker_compose(
        mcp_servers=[], website_configs=[website_config(env)],
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
        state_instance=LocalPostgresStateProvider.default_instance())
    service = yaml.safe_load(compose)["services"]["site-website-backend"]
    assert "privileged" not in service and "volumes" not in service
    (extra,) = [entry for entry in service["environment"] if entry.startswith("EXTRA=")]
    assert extra == "EXTRA=" + hostile.replace("$", "$$")


def test_the_logged_compose_masks_every_resolved_setting(local_stores, monkeypatch):
    env = WebsiteEnv(id="github-site", version=None, server_docker_image_artifact=_image("github-server"),
                     environment_name="github", settings={"GITHUB_TOKEN": "env:TEST_GITHUB_TOKEN",
                                                          "WEBHOOK": "env:TEST_WEBHOOK", "ORG": 'ac"me'})
    monkeypatch.setenv("TEST_GITHUB_TOKEN", "ghp_resolved")
    monkeypatch.setenv("TEST_WEBHOOK", "https://hooks.example/s3cret")
    compose = EnvironmentGatewayProvider().create_docker_compose(
        mcp_servers=[], website_configs=[website_config(env)],
        state_provider=LocalPostgresStateProvider(service_db_config=ServiceDBConfig()),
        state_instance=LocalPostgresStateProvider.default_instance())
    logged = _redact_compose_secrets(compose)
    assert "ghp_resolved" not in logged and "s3cret" not in logged
    environment = yaml.safe_load(logged)["services"]["github-website-backend"]["environment"]
    assert {"GITHUB_TOKEN=***", "WEBHOOK=***", "ORG=***", "SERVICE_NAME=github"} <= set(environment)
