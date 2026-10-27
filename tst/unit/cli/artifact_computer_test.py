"""`artifact computer put FOLDER` writes the computer a folder's artifact.toml declares, its form's image written as a
bundle writes one and read back from its registry unless `--no-inspect`; `artifact computer get` prints it. The
registry is patched out."""

import json

import pytest
from click.testing import CliRunner

from agent_env.artifact.artifacts import computer as computer_module
from agent_env.artifact.artifacts import docker_image
from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.cli import cli
from tst.util.oci_registry import UBUNTU_LAYER, Registry

DECLARED = ('type = "computer"\ndescription = "a desktop"\narch = ["amd64"]\n'
            'os = { type = "linux", distribution = "ubuntu", version = "24.04", display = "x11" }\n'
            'session = { user = "user" }\n\n[forms.container]\nimage = %s\ninit = "supervisord"\ninput_channel = "env"\n')


@pytest.fixture
def registry(local_stores, monkeypatch):
    served = Registry()
    index = served.index(amd64=served.image(UBUNTU_LAYER))
    monkeypatch.setattr(docker_image, "pin_digest", lambda ref, auth: f"{ref}@{index}")
    monkeypatch.setattr(computer_module, "read_manifest", served.read_manifest)
    monkeypatch.setattr(computer_module, "read_blob", served.read_blob)
    return served


def _folder(tmp_path, image: str):
    folder = tmp_path / "desktop"
    folder.mkdir()
    (folder / "artifact.toml").write_text(DECLARED % image)
    (folder / "Dockerfile").write_text("FROM scratch\n")
    return folder


def _run(*args):
    return CliRunner().invoke(cli, list(args), catch_exceptions=False)


def test_a_folder_naming_a_registry_image_is_written_over_its_pin_and_inspected(registry, tmp_path):
    result = _run("artifact", "computer", "put", "--id", "desktop", str(_folder(tmp_path, '{ ref = "ghcr.io/acme/desk:1" }')))

    assert result.exit_code == 0, result.output
    assert "Created ComputerArtifact: id=desktop version=1\n  container amd64: passed\n" in result.output
    image = ComputerArtifact.get("desktop").forms.container.image
    assert DockerImageArtifact.get(image.id, image.version).image_name.startswith("ghcr.io/acme/desk:1@sha256:")
    printed = json.loads(_run("artifact", "computer", "get", "--id", "desktop").output)
    assert printed["metadata"]["validated_images"]["container"]["amd64"]["os_release"]["VERSION_ID"] == "24.04"


def test_a_folders_dockerfile_is_its_build_context_and_no_inspect_skips_the_registry(registry, tmp_path):
    result = _run("artifact", "computer", "put", "--id", "desktop", "--no-inspect",
                  str(_folder(tmp_path, '{ dockerfile = "Dockerfile" }')))

    assert result.exit_code == 0, result.output
    assert "container amd64: uninspected (inspection was skipped)" in result.output
    image = ComputerArtifact.get("desktop").forms.container.image
    assert (image.id, DockerImageArtifact.get(image.id).context_only) == ("desktop__forms_container_image", True)
    assert registry.fetched == []


@pytest.mark.parametrize("image, message", [
    ("{ path = \"x\" }", "forms.container.image is { dockerfile = \"<path>\" }"),
    ('{ ref = "ghcr.io/acme/desk:1" }\narch2 = 1', "arch2"),
], ids=["image-key", "unknown-key"])
def test_what_the_folder_declares_wrong_is_reported(registry, tmp_path, image, message):
    result = _run("artifact", "computer", "put", "--id", "desktop", str(_folder(tmp_path, image)))

    assert result.exit_code == 1 and message in result.output


@pytest.mark.parametrize("image", ['"desk-image"', '{ artifact = "desk-image", version = 1 }'], ids=["id", "pinned"])
def test_a_folder_naming_an_image_artifact_is_written_over_it(registry, tmp_path, image):
    DockerImageArtifact.put_ref("desk-image", description="the desk", image_name="ghcr.io/acme/desk:1")

    result = _run("artifact", "computer", "put", "--id", "desktop", str(_folder(tmp_path, image)))

    assert result.exit_code == 0, result.output
    pinned = ComputerArtifact.get("desktop").forms.container.image
    assert (pinned.id, pinned.version) == ("desk-image", 1) and "container amd64: passed" in result.output
