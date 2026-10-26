"""Images built from their build context on this machine: natively, once per sources, labeled with them, and saved once
for a VM here to load; and published ones pulled here by their digest."""

import gzip
import io
import logging
import subprocess
import threading
import time

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.providers.sandbox_providers import host_images
from agent_env.providers.sandbox_providers import local_sandbox
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
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
async def test_a_local_sandbox_doesnt_pull_a_tag_again_over_the_image_its_digest_named(monkeypatch, docker):
    """One its provider prepared, or held under a name pinned to a digest; another held tag is pulled again, as is one
    its provider prepared that Docker no longer holds."""
    pinned, gone = f"ghcr.io/acme/tool:1@{_PUBLISHED_DIGEST}", "ghcr.io/acme/gone:1"
    held = {_PUBLISHED_REF, pinned, "ghcr.io/acme/tool:1"}
    monkeypatch.setattr(local_sandbox, "present", lambda name: name in held)
    sandbox = LocalSandbox()
    sandbox.prepared = frozenset({_PUBLISHED_REF, gone})
    pulled = []

    async def pull(self, names):
        pulled.extend(names)

    monkeypatch.setattr(VmSandbox, "pull_images", pull)
    await sandbox.pull_images([_PUBLISHED_REF, pinned, "ghcr.io/acme/tool:1", gone])

    assert pulled == ["ghcr.io/acme/tool:1", gone]


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


_PUBLISHED_REF = "ghcr.io/scaleapi/agentenv-claude-code:src-" + "b" * 64
_PUBLISHED_DIGEST = "sha256:" + "a" * 64
_PINNED = f"{_PUBLISHED_REF}@{_PUBLISHED_DIGEST}"


def _published(context):
    return DockerImageArtifact.put_ref("solver__agent_image", description="d", image_name=_PUBLISHED_REF,
                                       image_digest=_PUBLISHED_DIGEST, context_path=str(context),
                                       dockerfile_path=str(context / "Dockerfile"))


def _docker_cli(monkeypatch, *, pulled_by=None, pull_fails=False):
    """The docker commands ``pull`` runs, each recorded; ``pulled_by`` maps each image Docker holds to the digests it was
    pulled by, and a pull exits 1 when ``pull_fails``."""
    ran = []

    def run(argv, **kwargs):
        ran.append(argv[1:])
        failed = pull_fails and argv[1] == "pull"
        return subprocess.CompletedProcess(argv, int(failed), "", "Error response from daemon: denied" if failed else "")

    monkeypatch.setattr(host_images, "_repo_digests", lambda name: (pulled_by or {}).get(name, set()))
    monkeypatch.setattr(host_images.subprocess, "run", run)
    return ran


def test_a_published_image_is_pulled_by_its_digest_and_tagged_with_its_name(local_stores, context, docker,
                                                                            monkeypatch):
    ran = _docker_cli(monkeypatch)

    host_images.pull(_published(context))

    assert ran == [["pull", _PINNED], ["tag", _PINNED, _PUBLISHED_REF]]
    assert docker[0] == []


def test_a_published_image_docker_holds_by_its_digest_isnt_pulled_again(local_stores, context, docker, monkeypatch):
    ran = _docker_cli(monkeypatch, pulled_by={_PUBLISHED_REF: {_PUBLISHED_DIGEST}})

    host_images.pull(_published(context))

    assert ran == [] and docker[0] == []


def test_a_published_image_built_here_from_its_sources_isnt_pulled(local_stores, context, docker, monkeypatch):
    ran = _docker_cli(monkeypatch)
    image = _published(context)
    docker[1][_PUBLISHED_REF] = image.source_digest

    host_images.pull(image)

    assert ran == [] and docker[0] == []


def test_a_tag_docker_holds_as_another_image_is_pulled_by_the_digest_and_moved_to_it(local_stores, context, docker,
                                                                                   monkeypatch):
    """Two versions can share a tag that names other images over time: the one a version records is the one run."""
    ran = _docker_cli(monkeypatch, pulled_by={_PUBLISHED_REF: {"sha256:" + "d" * 64}})

    host_images.pull(_published(context))

    assert ran == [["pull", _PINNED], ["tag", _PINNED, _PUBLISHED_REF]]


@pytest.mark.parametrize("printed, code, digests", [
    ('["ghcr.io/acme/tool@sha256:aa","mirror.gcr.io/acme/tool@sha256:bb"]', 0, {"sha256:aa", "sha256:bb"}),
    ("null", 0, set()),
    ("", 1, set()),
], ids=["pulled", "built", "absent"])
def test_the_digests_an_image_was_pulled_by_are_read_from_docker(monkeypatch, printed, code, digests):
    monkeypatch.setattr(host_images.subprocess, "run",
                        lambda argv, **kwargs: subprocess.CompletedProcess(argv, code, printed, ""))

    assert host_images._repo_digests("ghcr.io/acme/tool:1") == digests


def test_a_published_image_that_cant_be_pulled_is_built_from_its_build_context(local_stores, context, docker,
                                                                              monkeypatch, caplog):
    ran = _docker_cli(monkeypatch, pull_fails=True)

    with caplog.at_level(logging.WARNING):
        host_images.pull(_published(context))

    assert ran == [["pull", _PINNED]]
    assert [tag for _, _, tag, _, _ in docker[0]] == [_PUBLISHED_REF]
    assert f"Couldn't pull {_PINNED}, so it's built from its build context: Error response from daemon: denied" in (
        caplog.text)


def test_a_published_image_with_no_build_context_that_cant_be_pulled_says_why(monkeypatch):
    _docker_cli(monkeypatch, pull_fails=True)
    image = DockerImageArtifact(id="x", version=1, description="d", image_name=_PUBLISHED_REF,
                                image_digest=_PUBLISHED_DIGEST)

    with pytest.raises(RuntimeError, match=f"Couldn't pull {_PINNED}: Error response from daemon: denied"):
        host_images.pull(image)


@pytest.mark.asyncio
async def test_the_local_provider_pulls_a_published_image_and_builds_one_that_is_only_its_context(local_stores, context,
                                                                                                 monkeypatch):
    """It starts containers from an image by name, so this machine's Docker holds the image first."""
    pulled, built = [], []

    async def pull_here(image):
        pulled.append(image.image_name)

    async def build_here(image):
        built.append(image.image_name)

    monkeypatch.setattr(local_sandbox, "pull_here", pull_here)
    monkeypatch.setattr(local_sandbox, "build_here", build_here)
    provider, context_only = LocalSandboxProvider(), _put(context)

    for image in (_published(context), context_only,
                  DockerImageArtifact(id="t", version=1, description="d", image_name="img:v1",
                                      tar_gz_object_url="s3://bucket/img.tar.gz")):
        await provider.prepare_image(image)

    assert (pulled, built) == ([_PUBLISHED_REF], [context_only.image_name])
    assert (await provider.create_vm()).prepared == {_PUBLISHED_REF}  # its containers start from it as it is
    assert LocalSandboxProvider()._prepared == set()

