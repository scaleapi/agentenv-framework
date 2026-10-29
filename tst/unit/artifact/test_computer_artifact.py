"""The computer artifact: what it declares is checked, each form's image is read from its registry before it's written
(its architectures, and the OS its lowest os-release layer names), and a mismatch refuses the put naming its form and
architecture. An image no registry serves yet, or one with a layer this Python can't decompress, is written
uninspected."""

import pytest

from agent_env.artifact import Artifact
from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.store.image_store import image_contents
from tst.util.oci_registry import MANIFEST, UBUNTU, UBUNTU_LAYER, Registry, layer

@pytest.fixture
def registry(local_stores, monkeypatch):
    return Registry().serve(monkeypatch)


def _image(name: str) -> DockerImageArtifact:
    return get_artifact_store().put_document(DockerImageArtifact(id="desktop-image", description="", image_name=name))


def _put(image: DockerImageArtifact, **changes) -> ComputerArtifact:
    forms = {"container": {"image": {"id": image.id, "version": image.version}, "init": "supervisord",
                           "input_channel": "env"}}
    return ComputerArtifact.put(**{**_fields(), "forms": forms, **changes})


def _fields() -> dict:
    return {"id": "desktop", "description": "a desktop", "arch": ["amd64", "arm64"],
            "os": {"type": "linux", "distribution": "ubuntu", "version": "24.04", "display": "x11"},
            "session": {"user": "user"}, "config": {"resolution": {"default": "1280x800", "phase": "deploy"}}}


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


@pytest.mark.parametrize("served", ["single", "index"])
def test_an_image_recorded_by_its_tag_alone_is_read_at_the_digest_its_registry_serves_for_it(registry, served):
    single = registry.image(UBUNTU_LAYER)
    registry.tags["localhost:5000/desktop-image:v1"] = single if served == "single" else registry.index(amd64=single)

    computer = _put(_image("localhost:5000/desktop-image:v1"), arch=["amd64"])

    found = computer.metadata["validated_images"]["container"]["amd64"]
    assert (found["status"], found["manifest"]) == ("passed", single)
    assert found["image"] == f"localhost:5000/desktop-image:v1@{registry.tags['localhost:5000/desktop-image:v1']}"


def test_os_release_comes_from_the_lowest_layer_holding_it_and_a_single_image_is_read_by_its_config(registry):
    desktop = layer({"etc/motd": b"welcome\n", "etc/os-release": b"ID=xfce\n"})
    single = registry.image(UBUNTU_LAYER, desktop)

    computer = _put(_image(f"ghcr.io/acme/desktop:1@{single}"), arch=["amd64"])

    assert computer.metadata["validated_images"]["container"]["amd64"]["status"] == "passed"
    manifest = registry.manifests[single]
    assert manifest["layers"][1]["digest"] not in registry.fetched


def test_an_image_no_registry_serves_yet_a_skipped_inspection_or_a_layer_python_cant_read_is_written_uninspected(
        registry, monkeypatch):
    built_here = get_artifact_store().put_document(DockerImageArtifact(
        id="desktop-image", description="", image_name="local/desktop-1234:abcd",
        build_context_object_url="file:///context.tar.gz", dockerfile_path="container/Dockerfile"))
    monkeypatch.setattr(image_contents, "_READS_ZSTD", False)
    zstd = registry._manifest({"mediaType": MANIFEST, "config": {"digest": registry.blob(b'{"os": "linux"}')}, "layers": [
        {"mediaType": "application/vnd.oci.image.layer.v1.tar+zstd", "digest": registry.blob(b"zstd frames")}]})

    from_context = _put(built_here, arch=["amd64"])
    skipped = ComputerArtifact.put(inspect=False, **{**from_context.model_dump(exclude={"version", "metadata"})})
    compressed = _put(_image(f"ghcr.io/acme/desktop:1@{registry.index(amd64=zstd)}"), arch=["amd64"])

    assert from_context.metadata["validated_images"]["container"]["amd64"]["status"] == "uninspected"
    assert "only its build context is stored" in from_context.metadata["validated_images"]["container"]["amd64"]["reason"]
    assert skipped.metadata["validated_images"]["container"]["amd64"]["reason"] == "inspection was skipped"
    assert compressed.metadata["validated_images"]["container"]["amd64"] == {
        "status": "uninspected", "image": compressed.image().image_name, "manifest": zstd,
        "reason": f"its layer {registry.manifests[zstd]['layers'][0]['digest']} is zstd-compressed, which Python reads "
                  "from 3.14 on, so its first boot checks it",
        "checked_at": compressed.metadata["validated_images"]["container"]["amd64"]["checked_at"]}
    assert registry.fetched == []


@pytest.mark.parametrize("inspect", [True, False], ids=["inspected", "uninspected"])
@pytest.mark.parametrize("image_id, version, problem", [
    ("nothing", 1, "Artifact nothing version=1 not found"),
    ("desktop-image", 2, "Artifact desktop-image version=2 not found"),
    ("notes", 1, "artifact 'notes' is a 'file' artifact, not a DockerImageArtifact"),
], ids=["missing", "missing-version", "not-an-image"])
def test_an_image_the_store_has_no_image_artifact_for_refuses_the_put_inspected_or_not(
        registry, inspect, image_id, version, problem):
    _image(f"ghcr.io/acme/desktop:1@{registry.index(amd64=registry.image(UBUNTU_LAYER))}")
    get_artifact_store().put_document(FileArtifact(id="notes", description="", filename="a", content_type="text/plain",
                                                   object_url="mem://b/a"))
    with pytest.raises(ValueError, match="computer 'desktop'") as refused:
        ComputerArtifact.put(inspect=inspect, **{**_fields(), "forms": {"container": {
            "image": {"id": image_id, "version": version}, "init": "supervisord", "input_channel": "env"}}})

    assert str(refused.value) == f"computer 'desktop': forms.container.image: {problem}"
    assert get_artifact_store().next_version("desktop") == 1 and registry.fetched == []


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


def test_an_absolute_link_is_followed_and_an_image_without_os_release_or_not_linux_or_unreadable_fails(registry):
    linked = registry.image(layer({"etc/os-release": "/usr/lib/os-release", "usr/lib/os-release": UBUNTU}))
    bare = registry.image(layer({"etc/motd": b"hi\n"}))
    garbled = registry.image(b"not a tar")
    windows = registry._manifest({"mediaType": MANIFEST, "layers": [],
                                  "config": {"digest": registry.blob(b'{"os": "windows", "architecture": "amd64"}')}})

    assert _put(_image(f"ghcr.io/acme/desktop:1@{linked}"), arch=["amd64"]).metadata["validated_images"][
        "container"]["amd64"]["status"] == "passed"
    with pytest.raises(ValueError, match="no layer holds /etc/os-release or /usr/lib/os-release"):
        _put(_image(f"ghcr.io/acme/desktop:2@{bare}"), arch=["amd64"])
    with pytest.raises(ValueError, match="it has no linux/amd64 image, only none"):
        _put(_image(f"ghcr.io/acme/desktop:3@{windows}"), arch=["amd64"])
    with pytest.raises(ValueError, match="its os-release couldn't be read: its layer sha256:[0-9a-f]+ couldn't be read"):
        _put(_image(f"ghcr.io/acme/desktop:4@{garbled}"), arch=["amd64"])


@pytest.mark.parametrize("served, problem", [
    ("index", "its os-release couldn't be read: its layer sha256:[0-9a-f]+ couldn't be read"),
    ("single", "its config sha256:[0-9a-f]+ couldn't be read"),
])
def test_a_blob_that_cant_be_written_where_it_is_read_fails_the_put_naming_it(registry, monkeypatch, served, problem):
    def full(ref, digest, auth, path):
        raise OSError(28, "No space left on device")

    single = registry.image(UBUNTU_LAYER)
    digest = registry.index(amd64=single) if served == "index" else single
    monkeypatch.setattr(image_contents, "read_blob", full)

    with pytest.raises(ValueError, match=rf"forms.container \(amd64\): {problem}: \[Errno 28\] No space left"):
        _put(_image(f"ghcr.io/acme/desktop:1@{digest}"), arch=["amd64"])
