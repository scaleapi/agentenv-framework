"""Fixtures the bundle tests share."""

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import materialize as materialize_module
from tst.unit.bundle._support import listing


@pytest.fixture
def docker_on_path(monkeypatch):
    monkeypatch.setattr(materialize_module.shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.fixture
def builds(monkeypatch, docker_on_path):
    """Stands in for docker: each build is recorded with the files of its context, a link marked with a
    trailing ``@``, and the image written as a docker_image document."""
    calls = []

    def build(dockerfile, context, tag, *, platform):
        calls.append({"build": (dockerfile.relative_to(context).as_posix(), listing(context), tag, platform)})

    def put(id, *, description, image_name, build_context_path=None, dockerfile_path=None):
        calls[-1]["put"] = (id, image_name, listing(build_context_path), dockerfile_path)
        return get_artifact_store().put_document(DockerImageArtifact(
            id=id, description=description, image_name=image_name, tar_gz_s3_url=f"file:///{id}.tar.gz"))

    monkeypatch.setattr(materialize_module, "build_image", build)
    monkeypatch.setattr(materialize_module.DockerImageArtifact, "put", put)
    return calls
