"""`DockerImageArtifact.put_context` registers an image by its build context alone, and `put` keeps the same whole,
deterministic context beside its tarball, on the local stores with docker faked out."""

import io
import os
import tarfile

import pytest

from agent_env.artifact.artifacts import docker_image
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.store.image_store.local_registry_image_store import LocalRegistryImageStore
from agent_env.store.image_store.oci_registry_credentials import names_registry
from agent_env.utils.build_context import BuildContext

SERVER = {"Dockerfile": "FROM python:3.12\nCOPY . /app\n", "src/app.py": "print(1)\n", ".dockerignore": "*.log\n",
          "debug.log": "noise"}


@pytest.fixture
def context(tmp_path):
    root = tmp_path / "ctx"
    for name, text in SERVER.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def _names(image: DockerImageArtifact) -> list[str]:
    data = get_artifact_store().get_object(image.build_context_object_url)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        return tar.getnames()


def test_a_context_is_registered_with_no_image_and_a_name_that_names_no_registry(local_stores, context, monkeypatch):
    monkeypatch.setenv("PATH", "")  # no docker needed

    image = DockerImageArtifact.put_context("solver-image", description="d", context_path=str(context),
                                            dockerfile_path=str(context / "Dockerfile"))

    stored = DockerImageArtifact.get("solver-image", image.version)
    assert (stored.tar_gz_object_url, stored.dockerfile_path, stored.platform) == (None, "Dockerfile", "linux/amd64")
    assert stored.image_name.startswith("local/solver-image-") and stored.image_name.endswith(":v1")
    assert not names_registry(stored.image_name)
    assert stored.source_digest == BuildContext.of(context, context / "Dockerfile").source_digest("linux/amd64")
    assert _names(stored) == [".dockerignore", "Dockerfile", "src", "src/app.py"]


def test_an_id_that_looks_like_a_registry_still_gets_a_name_that_names_none(local_stores, context):
    image = DockerImageArtifact.put_context("acme.io/tools__env_image", description="d", context_path=str(context),
                                            dockerfile_path="Dockerfile")

    assert image.image_name.startswith("local/acme-io-tools-env-image-") and not names_registry(image.image_name)


def test_equal_sources_give_equal_digests_and_bytes(local_stores, context):
    one = DockerImageArtifact.put_context("a", description="d", context_path=str(context), dockerfile_path="Dockerfile")
    two = DockerImageArtifact.put_context("b", description="d", context_path=str(context), dockerfile_path="Dockerfile",
                                          platform="linux/arm64")
    three = DockerImageArtifact.put_context("a", description="d", context_path=str(context), dockerfile_path="Dockerfile")

    store = get_artifact_store()
    assert one.source_digest == three.source_digest != two.source_digest
    assert store.get_object(one.build_context_object_url) == store.get_object(three.build_context_object_url)
    assert three.version == 2


def test_a_context_only_image_is_built_by_a_vm_and_refused_where_run_by_name(local_stores, context):
    image = DockerImageArtifact.put_context("a", description="d", context_path=str(context), dockerfile_path="Dockerfile")

    assert image.context_only and image.load_problem() is None
    assert image.by_name_problem() == "'a' v1 is only a build context, which only a VM sandbox builds"
    with pytest.raises(ValueError, match="has no tar.gz; its image is built from its build context"):
        image.load()


@pytest.mark.parametrize("dockerfile, message", [
    ("missing.Dockerfile", "no Dockerfile at"),
    ("../Dockerfile.outside", "is outside the build context"),
])
def test_a_dockerfile_the_context_cant_build_writes_nothing(local_stores, context, dockerfile, message):
    (context.parent / "Dockerfile.outside").write_text("FROM scratch\n")

    with pytest.raises(ValueError, match=message):
        DockerImageArtifact.put_context("a", description="d", context_path=str(context),
                                        dockerfile_path=str(context / dockerfile))
    assert get_artifact_store().next_version("a") == 1


def test_a_context_it_cant_take_writes_nothing(local_stores, context):
    os.mkfifo(context / "pipe")

    with pytest.raises(ValueError, match="pipe: neither a regular file, a folder nor a link"):
        DockerImageArtifact.put_context("a", description="d", context_path=str(context), dockerfile_path="Dockerfile")
    assert get_artifact_store().next_version("a") == 1


def test_a_link_leaving_the_context_is_kept_as_docker_sends_it(local_stores, context):
    (context / ".venv").mkdir()
    (context / ".venv" / "python").symlink_to("/usr/bin/python3")

    image = DockerImageArtifact.put_context("a", description="d", context_path=str(context), dockerfile_path="Dockerfile")

    assert ".venv/python" in _names(image)


class _DockerSave:
    def __init__(self, *args, **kwargs):
        self.stdout = io.BytesIO(b"image-tar-bytes")
        self.stderr = io.BytesIO(b"")
        self.returncode = 0

    def wait(self, timeout=None):
        return 0


def test_put_keeps_the_whole_context_and_records_how_to_build_it(local_stores, context, monkeypatch):
    monkeypatch.setattr(LocalRegistryImageStore, "ensure_repository", lambda self, repository: None)
    monkeypatch.setattr(docker_image, "_push_local_image", lambda src, ref, store: None)
    monkeypatch.setattr(docker_image, "_image_platform", lambda ref: None)
    monkeypatch.setattr(docker_image.subprocess, "Popen", _DockerSave)

    image = DockerImageArtifact.put("server", description="d", image_name="server:latest",
                                    build_context_path=str(context), dockerfile_path=str(context / "Dockerfile"),
                                    platform="linux/amd64")

    assert image.tar_gz_object_url and image.dockerfile_path == "Dockerfile" and image.platform == "linux/amd64"
    assert image.source_digest == BuildContext.of(context, context / "Dockerfile").source_digest("linux/amd64")
    assert _names(image) == [".dockerignore", "Dockerfile", "src", "src/app.py"]


def test_put_refuses_a_context_it_cant_take_before_pushing_anything(local_stores, context, monkeypatch):
    pushed = []
    monkeypatch.setattr(docker_image, "_push_local_image", lambda src, ref, store: pushed.append(ref))
    monkeypatch.setattr(docker_image, "_image_platform", lambda ref: None)
    os.mkfifo(context / "pipe")

    with pytest.raises(ValueError, match="pipe: neither a regular file"):
        DockerImageArtifact.put("server", description="d", image_name="server:latest", build_context_path=str(context))
    assert pushed == []


@pytest.mark.parametrize("inspected, platform", [
    ("linux/arm64/v8\n", "linux/arm64"), ("linux/amd64/\n", "linux/amd64"), ("linux/arm/v7\n", "linux/arm/v7"),
])
def test_the_platform_of_an_image_is_named_as_docker_build_takes_it(monkeypatch, inspected, platform):
    monkeypatch.setattr(docker_image.subprocess, "run",
                        lambda *a, **k: docker_image.subprocess.CompletedProcess(a, 0, stdout=inspected, stderr=""))

    assert docker_image._image_platform("x:v1") == platform


def test_put_records_the_platform_it_built_for_when_none_is_given(local_stores, context, monkeypatch):
    monkeypatch.setattr(LocalRegistryImageStore, "ensure_repository", lambda self, repository: None)
    monkeypatch.setattr(docker_image, "_push_local_image", lambda src, ref, store: None)
    monkeypatch.setattr(docker_image, "_image_platform", lambda ref: "linux/arm64")
    monkeypatch.setattr(docker_image.subprocess, "Popen", _DockerSave)

    native = DockerImageArtifact.put("server", description="d", image_name="server:latest",
                                     build_context_path=str(context), dockerfile_path="Dockerfile")
    explicit = DockerImageArtifact.put("server", description="d", image_name="server:latest",
                                       build_context_path=str(context), dockerfile_path="Dockerfile",
                                       platform="linux/amd64")

    assert (native.platform, explicit.platform) == ("linux/arm64", "linux/amd64")
    assert native.source_digest != explicit.source_digest
