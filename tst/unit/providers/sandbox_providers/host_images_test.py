"""Images built from their build context on this machine: natively, once per sources, labeled with them, and saved once
for a VM here to load; and published ones pulled here by their digest."""

import gzip
import io
import json
import logging
import subprocess
import tarfile
import threading
import time

import pytest

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.config import get_config
from agent_env.providers.sandbox_providers import host_images
from agent_env.providers.sandbox_providers.local_docker import provider as local_sandbox
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.utils.docker.build import DockerBuildError
from tst.util.docker_save import saved_image, tar_bytes


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



_TAGGED = saved_image([{"app/hello.txt": "hi\n"}], oci=True, tags=["acme/crm:1"])
_LOADED_ID = "sha256:" + "c" * 64


def _archived(tmp_path):
    archive = tmp_path / "image.tar"
    archive.write_bytes(_TAGGED)
    return DockerImageArtifact.put_archive("crm_image", description="d", archive_path=str(archive))


def _stored_tarball(tmp_path, data):
    """A tar.gz another writer stored under a bare tag, as it saved it: naming whatever images it names."""
    path = tmp_path / "saved.tar.gz"
    path.write_bytes(gzip.compress(data))
    prefix = get_artifact_store().attempt_prefix("docker_image", "crm_image")
    url = get_config().get_object_store_to_write(prefix, "crm_image").put_file_at(
        f"{prefix}crm.tar.gz", str(path), "application/gzip")
    return DockerImageArtifact.put_tar("crm_image", description="d", image_name="local/crm:v1", tar_gz_object_url=url)


def _docker_load(monkeypatch, *, held=(), printed=f"Loaded image ID: {_LOADED_ID}\n", failure=None):
    """The docker commands ``load`` runs, each load recorded with the names the archive it was given gives its image;
    Docker holds ``held``, and its load prints ``printed``, or exits 1 saying ``failure``."""
    ran = []

    def run(argv, **kwargs):
        if argv[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(argv, int(argv[3] not in held), "", "")
        if argv[1] == "load":
            with tarfile.open(argv[3]) as tar:
                names = [entry["RepoTags"] for entry in json.load(tar.extractfile("manifest.json"))]
                names += [d.get("annotations") for d in json.load(tar.extractfile("index.json"))["manifests"]]
            ran.append(["load", names])
        else:
            ran.append(argv[1:])
        if argv[1] == "load" and failure:
            return subprocess.CompletedProcess(argv, 1, "", failure)
        return subprocess.CompletedProcess(argv, 0, printed if argv[1] == "load" else "", "")

    monkeypatch.setattr(host_images.subprocess, "run", run)
    return ran


def test_a_saved_image_is_loaded_naming_no_image_and_tagged_with_its_name_by_its_id(local_stores, tmp_path,
                                                                                    monkeypatch):
    image = _archived(tmp_path)
    ran = _docker_load(monkeypatch)

    host_images.load(image)

    assert ran == [["load", [None, {"org.example.kept": "yes"}]], ["tag", _LOADED_ID, image.image_name]]


def test_a_stored_tarball_naming_images_is_given_to_docker_naming_none(local_stores, tmp_path, monkeypatch):
    """Docker would otherwise tag what it loads with every name the tarball gives, over images it holds."""
    image = _stored_tarball(tmp_path, _TAGGED)
    ran = _docker_load(monkeypatch)

    host_images.load(image)

    assert ran == [["load", [None, {"org.example.kept": "yes"}]], ["tag", _LOADED_ID, "local/crm:v1"]]


def test_a_saved_image_docker_holds_isnt_loaded_again(local_stores, tmp_path, monkeypatch):
    image = _archived(tmp_path)
    ran = _docker_load(monkeypatch, held={image.image_name})

    host_images.load(image)

    assert ran == []


@pytest.mark.parametrize("load, problem", [
    ({"failure": "open /x: no space left on device\n"},
     "Couldn't load 'crm_image' v1's saved image: open /x: no space"),
    ({"printed": "Loaded image: acme/crm:1\n"}, "'crm_image' v1's saved image holds no image to load"),
], ids=["refused", "no id"])
def test_a_saved_image_docker_cant_load_says_why(local_stores, tmp_path, monkeypatch, load, problem):
    image = _archived(tmp_path)
    _docker_load(monkeypatch, **load)

    with pytest.raises(RuntimeError, match=problem):
        host_images.load(image)


def test_a_stored_tarball_docker_save_didnt_write_isnt_given_to_docker(local_stores, tmp_path, monkeypatch):
    image = _stored_tarball(tmp_path, tar_bytes({"hello.txt": "hi"}))
    ran = _docker_load(monkeypatch)

    with pytest.raises(RuntimeError, match="Couldn't load 'crm_image' v1's saved image: it holds no manifest.json"):
        host_images.load(image)
    assert ran == []


@pytest.mark.asyncio
async def test_the_local_provider_pulls_a_published_image_builds_a_context_and_loads_a_saved_one(local_stores, context,
                                                                                                   monkeypatch):
    """It starts containers from an image by name, so this machine's Docker holds the image first."""
    pulled, built, loaded = [], [], []

    async def pull_here(image):
        pulled.append(image.image_name)

    async def build_here(image):
        built.append(image.image_name)

    async def load_here(image):
        loaded.append(image.image_name)

    monkeypatch.setattr(local_sandbox, "pull_here", pull_here)
    monkeypatch.setattr(local_sandbox, "build_here", build_here)
    monkeypatch.setattr(local_sandbox, "load_here", load_here)
    provider, context_only = LocalSandboxProvider(), _put(context)

    for image in (_published(context), context_only,
                  DockerImageArtifact(id="t", version=1, description="d", image_name="img:v1",
                                      tar_gz_object_url="s3://bucket/img.tar.gz")):
        await provider.prepare_image(image)

    assert (pulled, built, loaded) == ([_PUBLISHED_REF], [context_only.image_name], ["img:v1"])
    # its containers start from them as they are
    assert (await provider.create_vm()).prepared == {_PUBLISHED_REF, "img:v1"}
    assert LocalSandboxProvider()._prepared == set()

