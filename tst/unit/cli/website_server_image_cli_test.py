"""agent-env env website put: a site's server image, from a Dockerfile stored with its build context as an a2a
agent's is, or from a registry, checked at its health path; a separate frontend image is optional and warned of."""

from unittest.mock import patch

from click.testing import CliRunner

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store

from agent_env.cli.env.website import website
from agent_env.env import Env


def _stored(id, **_):
    return get_artifact_store().put_document(DockerImageArtifact(
        id=id, description="", image_name=f"registry.example/{id}@sha256:" + "a" * 64))


def _put(*args):
    return CliRunner().invoke(website, ["put", "--id", "mail-site", *args, "--skip-validation"])


def test_a_site_is_built_from_its_server_dockerfile_alone_with_its_build_context(local_stores, tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    built = []
    with patch("agent_env.cli.env.website.build_image", side_effect=lambda *a, **k: built.append(a[2])), \
            patch("agent_env.artifact.artifacts.docker_image.DockerImageArtifact.put",
                  side_effect=lambda **k: _stored(k["id"])) as stored:
        result = _put("--server-dockerfile", str(tmp_path / "Dockerfile"), "--environment-name", "mail",
                      "--health-path", "/healthz")
    assert result.exit_code == 0, result.output
    assert len(built) == 1 and "deprecated" not in result.output
    assert stored.call_args.kwargs["build_context_path"] == str(tmp_path)
    assert stored.call_args.kwargs["dockerfile_path"] == str(tmp_path / "Dockerfile")
    env = Env.get("mail-site")
    assert (env.frontend_docker_image_artifact, env.health_path) == (None, "/healthz")


def test_a_site_registers_a_server_image_already_in_a_registry(local_stores):
    with patch("agent_env.cli.env.website.DockerImageArtifact.put_ref",
               side_effect=lambda id, **k: _stored(id)) as put_ref:
        result = _put("--server-image-ref", "ghcr.io/acme/mail:3", "--environment-name", "mail")
    assert result.exit_code == 0, result.output
    assert put_ref.call_args.kwargs["image_name"] == "ghcr.io/acme/mail:3"
    assert Env.get("mail-site").server_docker_image_artifact.id == "mail-site__backend_image"


def test_put_refuses_no_server_image_two_of_them_an_unnamed_registry_image_and_a_bad_health_path(local_stores, tmp_path):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")
    assert "exactly one of --server-dockerfile" in _put("--environment-name", "mail").output
    assert "exactly one of" in _put("--server-dockerfile", str(tmp_path / "Dockerfile"),
                                    "--server-image-ref", "ghcr.io/acme/mail:3").output
    assert "needs --environment-name" in _put("--server-image-ref", "ghcr.io/acme/mail:3").output
    with patch("agent_env.cli.env.website.DockerImageArtifact.put_ref", side_effect=lambda id, **k: _stored(id)):
        refused = _put("--server-image-ref", "ghcr.io/acme/mail:3", "--environment-name", "mail", "--health-path", "ok")
    assert refused.exit_code == 1 and "health: 'ok' is not a path" in refused.output


def test_a_separate_frontend_image_still_builds_and_is_warned_of(local_stores, tmp_path):
    (tmp_path / "server.Dockerfile").write_text("FROM scratch\n")
    (tmp_path / "frontend.Dockerfile").write_text("FROM scratch\n")
    with patch("agent_env.cli.env.website.build_image"), \
            patch("agent_env.artifact.artifacts.docker_image.DockerImageArtifact.put",
                  side_effect=lambda **k: _stored(k["id"])):
        result = _put("--backend-dockerfile", str(tmp_path / "server.Dockerfile"), "--frontend-dockerfile",
                      str(tmp_path / "frontend.Dockerfile"), "--environment-name", "mail")
    assert result.exit_code == 0, result.output
    assert "separate frontend image is deprecated" in result.output
    assert Env.get("mail-site").frontend_docker_image_artifact.id == "mail-site__frontend_image"
