"""The computer artifact: what it declares is checked, each form's image is read from its registry before it's written
(its architectures, and the OS its lowest os-release layer names), and a mismatch refuses the put naming its form and
architecture. An image no registry serves yet is written uninspected."""

import hashlib
import io
import json
import tarfile

import pytest

from agent_env.artifact import Artifact
from agent_env.artifact.artifacts import computer as computer_module
from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.store.image_store.registry_api import NoSuchImageError

UBUNTU = b'PRETTY_NAME="Ubuntu 24.04.3 LTS"\nNAME="Ubuntu"\nVERSION_ID="24.04"\nID=ubuntu\n'
INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"


class Registry:
    """A registry serving the manifests and blobs put in it, by digest, and recording the blobs fetched."""

    def __init__(self):
        self.manifests, self.blobs, self.fetched = {}, {}, []

    def blob(self, data: bytes) -> str:
        digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
        self.blobs[digest] = data
        return digest

    def image(self, *layers: bytes, arch: str = "amd64") -> str:
        config = self.blob(json.dumps({"os": "linux", "architecture": arch}).encode())
        manifest = {"mediaType": MANIFEST, "config": {"digest": config},
                    "layers": [{"digest": self.blob(layer)} for layer in layers]}
        return self._manifest(manifest)

    def index(self, **images: str) -> str:
        entries = [{"digest": digest, "platform": {"os": "linux", "architecture": arch}} for arch, digest in images.items()]
        entries.append({"digest": self.blob(b"attestation"), "platform": {"os": "unknown", "architecture": "unknown"}})
        return self._manifest({"mediaType": INDEX, "manifests": entries})

    def _manifest(self, manifest: dict) -> str:
        digest = self.blob(json.dumps(manifest).encode())
        self.manifests[digest] = manifest
        return digest

    def read_manifest(self, ref, auth):
        digest = ref.rpartition("@")[2]
        if digest not in self.manifests:
            raise NoSuchImageError(f"{ref}: ghcr.io has no such image")
        return self.manifests[digest]

    def read_blob(self, ref, digest, auth, path):
        self.fetched.append(digest)
        path.write_bytes(self.blobs[digest])


def layer(files: dict[str, bytes | str]) -> bytes:
    """A tar.gz layer of ``files`` by path; a str is a link to that path."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            if isinstance(content, str):
                info.type, info.linkname = tarfile.SYMTYPE, content
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


UBUNTU_LAYER = layer({"etc/os-release": "../usr/lib/os-release", "usr/lib/os-release": UBUNTU})


@pytest.fixture
def registry(local_stores, monkeypatch):
    registry = Registry()
    monkeypatch.setattr(computer_module, "read_manifest", registry.read_manifest)
    monkeypatch.setattr(computer_module, "read_blob", registry.read_blob)
    return registry


def _image(name: str) -> DockerImageArtifact:
    return get_artifact_store().put_document(DockerImageArtifact(id="desktop-image", description="", image_name=name))


def _put(image: DockerImageArtifact, **changes) -> ComputerArtifact:
    fields = {"id": "desktop", "description": "a desktop", "arch": ["amd64", "arm64"],
              "os": {"type": "linux", "distribution": "ubuntu", "version": "24.04", "display": "x11"},
              "session": {"user": "user"},
              "forms": {"container": {"image": {"id": image.id, "version": image.version}, "init": "supervisord",
                                      "input_channel": "env"}},
              "config": {"resolution": {"default": "1280x800", "phase": "deploy"}}}
    return ComputerArtifact.put(**{**fields, **changes})


def test_a_computer_whose_image_is_what_it_declares_is_written_with_what_was_read(registry):
    amd64, arm64 = registry.image(UBUNTU_LAYER), registry.image(UBUNTU_LAYER, arch="arm64")
    image = _image(f"ghcr.io/acme/desktop:1@{registry.index(amd64=amd64, arm64=arm64)}")

    computer = _put(image)

    found = Artifact.get("desktop", computer.version).metadata["validated_images"]["container"]
    assert {arch: (found[arch]["status"], found[arch]["manifest"], found[arch]["os_release"]) for arch in found} == {
        "amd64": ("passed", amd64, {"ID": "ubuntu", "VERSION_ID": "24.04"}),
        "arm64": ("passed", arm64, {"ID": "ubuntu", "VERSION_ID": "24.04"})}
    assert found["amd64"]["image"] == image.image_name
    assert computer.forms.container.image.version == image.version


@pytest.mark.parametrize("served, changes, problem", [
    ("amd64-only", {}, "forms.container (arm64): it has no linux/arm64 image, only linux/amd64"),
    ("both", {"os": {"type": "linux", "distribution": "ubuntu", "version": "22.04", "display": "x11"}},
     "os-release has VERSION_ID='24.04', not '22.04'"),
    ("both", {"os": {"type": "linux", "distribution": "debian", "version": "24.04", "display": "x11"}},
     "os-release has ID='ubuntu', not 'debian'"),
    ("nothing", {}, "forms.container (amd64): ghcr.io/acme/desktop:1@sha256:"),
], ids=["missing-arch", "wrong-version", "wrong-distribution", "unreachable"])
def test_an_image_that_isnt_what_the_computer_declares_refuses_the_put(registry, served, changes, problem):
    amd64 = registry.image(UBUNTU_LAYER)
    digest = {"amd64-only": registry.index(amd64=amd64),
              "both": registry.index(amd64=amd64, arm64=registry.image(UBUNTU_LAYER, arch="arm64")),
              "nothing": "sha256:" + "0" * 64}[served]

    with pytest.raises(ValueError, match="computer 'desktop'") as refused:
        _put(_image(f"ghcr.io/acme/desktop:1@{digest}"), **changes)

    assert problem in str(refused.value)
    assert get_artifact_store().next_version("desktop") == 1


def test_os_release_comes_from_the_lowest_layer_holding_it_and_a_single_image_is_read_by_its_config(registry):
    desktop = layer({"etc/motd": b"welcome\n", "etc/os-release": b"ID=xfce\n"})
    single = registry.image(UBUNTU_LAYER, desktop)

    computer = _put(_image(f"ghcr.io/acme/desktop:1@{single}"), arch=["amd64"])

    assert computer.metadata["validated_images"]["container"]["amd64"]["status"] == "passed"
    manifest = registry.manifests[single]
    assert manifest["layers"][1]["digest"] not in registry.fetched


def test_an_image_no_registry_serves_yet_or_a_skipped_inspection_is_written_uninspected(registry):
    built_here = get_artifact_store().put_document(DockerImageArtifact(
        id="desktop-image", description="", image_name="local/desktop-1234:abcd",
        build_context_object_url="file:///context.tar.gz", dockerfile_path="container/Dockerfile"))

    from_context = _put(built_here, arch=["amd64"])
    skipped = ComputerArtifact.put(inspect=False, **{**from_context.model_dump(exclude={"version", "metadata"})})

    assert from_context.metadata["validated_images"]["container"]["amd64"]["status"] == "uninspected"
    assert "only its build context is stored" in from_context.metadata["validated_images"]["container"]["amd64"]["reason"]
    assert skipped.metadata["validated_images"]["container"]["amd64"]["reason"] == "inspection was skipped"
    assert registry.fetched == []


@pytest.mark.parametrize("changes, message", [
    ({"os": {"type": "windows", "version": "11"}}, "os.type 'windows' isn't supported yet"),
    ({"forms": {"disk": {}}}, "the disk form isn't supported yet"),
    ({"arch": ["amd64", "amd64"]}, "lists an architecture twice"),
    ({"arch": ["riscv64"]}, "arch.0"),
    ({"session": {"user": "Root User"}}, "session.user"),
    ({"config": {"Screen-Size": {"default": "1", "phase": "deploy"}}}, "setting names are lowercase"),
    ({"config": {"theme": {"default": "dark", "phase": "build", "scope": "user"}}}, "a build-phase setting is task-scoped"),
    ({"config": {"token": {"default": "", "phase": "build", "generate": True}}}, "a generated setting is minted"),
    ({"ports": [8080]}, "ports"),
], ids=["os-type", "form", "arch-twice", "arch-unknown", "user", "setting-name", "build-user-scope", "build-generate",
        "unknown-key"])
def test_what_a_computer_declares_is_checked(registry, changes, message):
    image = _image(f"ghcr.io/acme/desktop:1@{registry.index(amd64=registry.image(UBUNTU_LAYER))}")
    with pytest.raises(ValueError, match=message):
        _put(image, **changes)


def test_an_absolute_link_is_followed_and_an_image_without_os_release_or_not_linux_fails(registry):
    linked = registry.image(layer({"etc/os-release": "/usr/lib/os-release", "usr/lib/os-release": UBUNTU}))
    bare = registry.image(layer({"etc/motd": b"hi\n"}))
    windows = registry._manifest({"mediaType": MANIFEST, "layers": [],
                                  "config": {"digest": registry.blob(b'{"os": "windows", "architecture": "amd64"}')}})

    assert _put(_image(f"ghcr.io/acme/desktop:1@{linked}"), arch=["amd64"]).metadata["validated_images"][
        "container"]["amd64"]["status"] == "passed"
    with pytest.raises(ValueError, match="no layer holds /etc/os-release or /usr/lib/os-release"):
        _put(_image(f"ghcr.io/acme/desktop:2@{bare}"), arch=["amd64"])
    with pytest.raises(ValueError, match="it has no linux/amd64 image, only none"):
        _put(_image(f"ghcr.io/acme/desktop:3@{windows}"), arch=["amd64"])
