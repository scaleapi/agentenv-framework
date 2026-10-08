"""`env mcp-server put --image-ref` and `a2a-agent put --image-ref` register an env or agent over an image already in a
registry, through `DockerImageArtifact.put_ref`, on the local stores; the registry API is patched out."""

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts import docker_image
from agent_env.cli import cli
from agent_env.env import Env

DIGEST = "sha256:" + "d" * 64
REF = "ghcr.io/org/server:v1"


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(docker_image, "pin_digest", lambda ref, auth: f"{ref}@{DIGEST}")


def _run(*args):
    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


def test_an_mcp_server_is_registered_over_an_image_in_a_registry(local_stores, pinned):
    result = _run("env", "mcp-server", "put", "--id", "server", "--image-ref", REF, "--environment-name", "items")

    assert result.exit_code == 0, result.output
    env = Env.get("server")
    image = env.docker_image_artifact
    assert (env.environment_name, image.id, image.image_name, image.tar_gz_object_url) == (
        "items", "server__env_image", f"{REF}@{DIGEST}", None)
    assert f"Registered image: id=server__env_image version=1 image={REF}@{DIGEST}" in result.output


def test_an_mcp_server_over_an_image_ref_needs_its_environment_name(local_stores, pinned):
    result = _run("env", "mcp-server", "put", "--id", "server", "--image-ref", REF)

    assert result.exit_code == 1
    assert "--image-ref needs --environment-name" in result.output


@pytest.mark.parametrize("extra, message", [
    (["--dockerfile", "Dockerfile"], "--dockerfile and --image-ref are mutually exclusive"),
    (["--dockerfile-github-url", "https://github.com/o/r/tree/main/Dockerfile"],
     "--dockerfile-github-url and --image-ref are mutually exclusive"),
    (["--context", "."], "--context needs --dockerfile"),
])
def test_an_mcp_server_takes_one_image_source(local_stores, pinned, tmp_path, extra, message):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")

    result = _run("env", "mcp-server", "put", "--id", "server", "--image-ref", REF, "--environment-name", "items", *extra)

    assert result.exit_code == 1 and message in result.output


def test_an_agent_is_registered_over_an_image_in_a_registry(local_stores, pinned):
    result = _run("a2a-agent", "put", "--id", "solver", "--image-ref", REF, "--default-model", "claude-sonnet-4-6",
                  "--skip-validation")

    assert result.exit_code == 0, result.output
    agent = A2AAgent.get("solver")
    assert (agent.docker_image_artifact.id, agent.docker_image_artifact.image_name) == ("solver__agent_image",
                                                                                        f"{REF}@{DIGEST}")
    assert agent.metadata == {"default_model": "claude-sonnet-4-6"}


@pytest.mark.parametrize("extra, message", [
    ([], "exactly one of --dockerfile and --image-ref is required"),
    (["--image-ref", REF, "--dockerfile", "Dockerfile"], "exactly one of --dockerfile and --image-ref is required"),
    (["--image-ref", REF, "--context", "."], "--context needs --dockerfile"),
])
def test_an_agent_takes_one_image_source(local_stores, pinned, tmp_path, extra, message):
    (tmp_path / "Dockerfile").write_text("FROM scratch\n")

    result = _run("a2a-agent", "put", "--id", "solver", "--skip-validation", *extra)

    assert result.exit_code == 1 and message in result.output
