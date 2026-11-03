"""A website env is one server image serving the site's pages and API, checked at its health path; a site still built
on a separate frontend image keeps loading and deploying, and put warns that it's deprecated."""

import logging
from types import SimpleNamespace

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.env import Env
from agent_env.env.envs.website import WebsiteEnv


def _image(id):
    return get_artifact_store().put_document(DockerImageArtifact(
        id=id, description="", image_name=f"registry.example/{id}@sha256:" + "a" * 64))


def test_a_site_is_one_server_image_checked_at_its_health_path(local_stores):
    env = WebsiteEnv.put(id="shop-site", server_docker_image_artifact=_image("shop-server"), environment_name="shop")
    stored = Env.get("shop-site")
    assert (stored.server_docker_image_artifact.id, stored.frontend_docker_image_artifact) == ("shop-server", None)
    assert stored.health_path == "/api/health"
    assert stored.backend_docker_image_artifact is stored.server_docker_image_artifact
    doc = env.to_dict()
    assert doc["server_docker_image_artifact"] == doc["backend_docker_image_artifact"]
    assert "frontend_docker_image_artifact" not in doc


def test_a_site_declares_its_health_path_and_a_bad_one_is_refused(local_stores):
    WebsiteEnv.put(id="slack-site", server_docker_image_artifact=_image("slack-server"), environment_name="slack",
                   health_path="/healthz")
    assert Env.get("slack-site").health_path == "/healthz"
    with pytest.raises(ValueError) as refused:
        WebsiteEnv(id="bad", version=None, server_docker_image_artifact=_image("bad-server"), environment_name="bad",
                   health_path="healthz")
    assert str(refused.value) == ("website env 'bad': health: 'healthz' is not a path of letters, digits, '.', '_', "
                                  "'~', '-' and '/' starting with '/'")


def test_a_site_stored_before_the_server_image_reads_its_backend_image_as_it(local_stores):
    backend, frontend = _image("old-backend"), _image("old-frontend")
    old = {"type": "website", "id": "old-site", "version": 1, "environment_name": "old", "metadata": {},
           "backend_docker_image_artifact": {"id": backend.id, "version": backend.version},
           "frontend_docker_image_artifact": {"id": frontend.id, "version": frontend.version}}
    env = WebsiteEnv.from_dict(old)
    assert (env.server_docker_image_artifact.id, env.frontend_docker_image_artifact.id) == ("old-backend", "old-frontend")
    with pytest.raises(ValueError, match="names its server image once"):
        WebsiteEnv(id="x", version=None, server_docker_image_artifact=backend, backend_docker_image_artifact=backend,
                   environment_name="x")


def test_a_separate_frontend_image_is_deprecated_but_still_deploys(local_stores, caplog):
    with caplog.at_level(logging.WARNING, logger="agent_env.env.envs.website"):
        env = WebsiteEnv.put(id="shop-site", backend_docker_image_artifact=_image("shop-backend"),
                             frontend_docker_image_artifact=_image("shop-frontend"), environment_name="shop")
    assert "separate frontend image, which is deprecated" in caplog.text
    assert env.deploy_refusal() is None


def test_an_env_toml_takes_a_health_path_and_refuses_a_bad_one():
    def refuse(problems):
        raise ValueError(problems)

    ctx = SimpleNamespace(config_problem=lambda problem: f"env.toml: {problem}", refuse=refuse,
                          accept_env=lambda data, cls, named_by: data)
    data = {"backend_image": "server", "environment_name": "slack", "health": "/healthz"}
    assert WebsiteEnv.accept_toml(data, ctx) == data
    with pytest.raises(ValueError, match="env.toml: health: 'healthz' is not a path"):
        WebsiteEnv.accept_toml({**data, "health": "healthz"}, ctx)
    with pytest.raises(ValueError, match="is not a path of letters"):
        WebsiteEnv.accept_toml({**data, "health": '/ok" x'}, ctx)
