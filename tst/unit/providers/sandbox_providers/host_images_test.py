"""Images built from their build context on this machine: natively, once per sources, labeled with them, and saved once
for a VM here to load."""

import gzip
import io
import threading
import time

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.sandbox_providers import host_images
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.utils.docker.build import DockerBuildError


@pytest.fixture
def docker(monkeypatch):
    """This machine's Docker: each build is recorded with its Dockerfile, the files of its context, tag, platform and
    labels, and Docker then holds the tag, labeled."""
    builds, held = [], {}

    def build(dockerfile, context, tag, *, platform, labels=None):
        builds.append((dockerfile.relative_to(context).as_posix(),
                       sorted(p.relative_to(context).as_posix() for p in context.rglob("*") if p.is_file()),
                       tag, platform, labels))
        held[tag] = labels[host_images._SOURCES_LABEL]

    monkeypatch.setattr(host_images, "build_image", build)
    monkeypatch.setattr(host_images, "_built_from", held.get)
    return builds, held


@pytest.fixture
def context(tmp_path):
    folder = tmp_path / "ctx"
    folder.mkdir()
    (folder / "Dockerfile").write_text("FROM scratch\nCOPY run.sh /\n")
    (folder / "run.sh").write_text("echo hi\n")
    return folder


def _put(context, id="solver__agent_image"):
    return DockerImageArtifact.put_context(id, description="d", context_path=str(context),
                                           dockerfile_path=str(context / "Dockerfile"))


def test_an_image_is_built_natively_from_its_stored_context_labeled_with_its_sources_and_once(local_stores, context,
                                                                                             docker):
    builds, _ = docker
    image = _put(context)

    host_images.build(image)
    host_images.build(image)

    assert builds == [("Dockerfile", ["Dockerfile", "run.sh"], image.image_name, None,
                       {host_images._SOURCES_LABEL: image.source_digest})]
    assert host_images.held(image) and host_images.built_here(image.image_name)


def test_an_image_docker_holds_from_other_sources_is_built_again(local_stores, context, docker):
    builds, held = docker
    image = _put(context)
    held[image.image_name] = "sha256:" + "0" * 64

    host_images.build(image)

    assert len(builds) == 1 and host_images.held(image)


def test_a_staged_build_says_so_first_only_when_it_builds(local_stores, context, docker):
    builds, _ = docker
    said = []
    sources, name = DockerImageArtifact.context_identity("solver__agent_image", str(context), str(context / "Dockerfile"))

    for _ in range(2):
        host_images.build_staged(context, "Dockerfile", name, sources, lambda: said.append(name))

    assert said == [name] and len(builds) == 1
    assert name == _put(context).image_name


def test_deploys_building_one_image_at_once_build_it_once(local_stores, context, docker, monkeypatch):
    builds, held = docker
    image = _put(context)
    slow = host_images.build_image

    def build(*args, **kwargs):
        time.sleep(0.3)
        slow(*args, **kwargs)

    monkeypatch.setattr(host_images, "build_image", build)
    threads = [threading.Thread(target=host_images.build, args=(image,)) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(builds) == 1 and held[image.image_name] == image.source_digest


def test_a_failed_build_raises_what_docker_says(local_stores, context, monkeypatch):
    def fail(dockerfile, context, tag, *, platform, labels=None):
        raise DockerBuildError(f"docker build of {tag} failed (exit 1):\nboom")

    monkeypatch.setattr(host_images, "build_image", fail)
    monkeypatch.setattr(host_images, "_built_from", lambda name: None)

    with pytest.raises(DockerBuildError, match="boom"):
        host_images.build(_put(context))


def test_no_docker_on_path_holds_nothing(monkeypatch):
    monkeypatch.setenv("PATH", "")

    assert not host_images.built_here("local/solver-0123:abc")


@pytest.mark.asyncio
async def test_a_local_sandbox_pulls_only_images_this_machine_didnt_build(monkeypatch, docker):
    _, held = docker
    held["local/solver-0123:abc"] = "sha256:" + "1" * 64
    pulled = []

    async def pull(self, names):
        pulled.extend(names)

    monkeypatch.setattr(VmSandbox, "pull_images", pull)
    await LocalSandbox().pull_images(["local/solver-0123:abc", "ghcr.io/acme/tool:1"])

    assert pulled == ["ghcr.io/acme/tool:1"]


@pytest.mark.asyncio
async def test_a_local_sandbox_builds_a_context_with_this_machines_docker(local_stores, context, docker):
    builds, _ = docker
    image = _put(context)

    await LocalSandbox().build_images([image, image])

    assert [build[2:4] for build in builds] == [(image.image_name, None)]


class _Save:
    """``docker save``, writing the name it saves."""

    def __init__(self, argv, **kwargs):
        self.stdout, self.stderr = io.BytesIO(argv[-1].encode()), io.BytesIO()

    def wait(self):
        return 0


def test_images_of_equal_sources_under_two_names_are_saved_once_each_so_each_loads_under_its_own(local_stores, context,
                                                                                                docker, monkeypatch):
    monkeypatch.setattr(host_images.subprocess, "Popen", _Save)
    first, second = _put(context, "solver__agent_image"), _put(context, "helper__agent_image")

    urls = [host_images._saved_here(image) for image in (first, second, first)]

    assert first.source_digest == second.source_digest and urls[0] != urls[1] and urls[2] == urls[0]
    store = local_stores.get_object_store()
    assert [gzip.decompress(store.get(url)).decode() for url in urls[:2]] == [first.image_name, second.image_name]


@pytest.mark.asyncio
async def test_a_local_sandbox_given_a_platform_builds_as_a_vm_does(local_stores, context, docker, monkeypatch):
    builds, _ = docker
    built_in_vm = []

    async def build_in_vm(self, artifacts):
        built_in_vm.extend(artifact.image_name for artifact in artifacts)

    monkeypatch.setattr(VmSandbox, "build_images", build_in_vm)
    sandbox = LocalSandbox()
    sandbox.platform = "linux/amd64"
    image = _put(context)

    await sandbox.build_images([image])

    assert builds == [] and built_in_vm == [image.image_name]

