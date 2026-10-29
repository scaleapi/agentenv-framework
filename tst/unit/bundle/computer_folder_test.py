"""Computer folders: an ``artifacts/<name>/`` computer's ``forms.container.image`` key has the bundle write the image it
names, built from a Dockerfile in the folder or pinned from a registry, and the computer is written over that image's
version, inspected when a registry serves it."""

import json

import pytest

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.bundle.parse import BundleError
from agent_env.bundle.materialize import materialize
from tst.unit.bundle._support import DIGEST, layout, local_builds, plan_of
from tst.util.oci_registry import UBUNTU_LAYER, Registry

ROOT = "@local/~/desk"
COMPUTER = f"{ROOT}/desktop"
IMAGE = f"{COMPUTER}__forms_container_image"
CARD = 'from agentenv_protocol import environment_card\n\n\n@environment_card(name="crm")\nclass Server:\n    pass\n'
OS = '{ type = "linux", distribution = "ubuntu", version = "24.04", display = "x11" }'


def _files(image: str, os: str = OS) -> dict[str, str]:
    return {
        "artifacts/desktop/artifact.toml": (
            f'type = "computer"\ndescription = "a desktop"\narch = ["amd64"]\nos = {os}\nsession = {{ user = "user" }}\n\n'
            f'[forms.container]\nimage = {image}\ninit = "supervisord"\ninput_channel = "env"\n'),
        "artifacts/desktop/container/Dockerfile": "FROM scratch\nCOPY container/start /start\n",
        "artifacts/desktop/container/start": "#!/bin/sh\n",
        "envs/crm/Dockerfile": "FROM scratch\nCOPY . /app\n",
        "envs/crm/server.py": CARD,
        "envs/crm/env.toml": 'environment_name = "crm"\n',
        "tasks/t.json": json.dumps([{"id": "crm", "type": "deploy_env", "env_id": "crm"},
                                    {"id": "load", "type": "load_artifact", "env_id": "crm", "artifact_id": "desktop"}]),
    }


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "desk"


def _run(root):
    plan = plan_of(root)
    return materialize(plan, build_here=local_builds(plan))


def _computer(done) -> ComputerArtifact:
    return ComputerArtifact.get(COMPUTER, done.version_of("artifact", COMPUTER))


def test_a_computer_folder_builds_its_image_and_is_written_over_it_uninspected_until_a_registry_serves_it(
        bundle_dir, builds):
    layout(bundle_dir, _files('{ dockerfile = "container/Dockerfile" }'))

    done = _run(bundle_dir)

    computer = _computer(done)
    image = DockerImageArtifact.get(IMAGE)
    assert (computer.forms.container.image.id, computer.forms.container.image.version) == (IMAGE, image.version)
    assert image.dockerfile_path == "container/Dockerfile" and image.context_only
    built = next(call["build"] for call in builds if call["build"][0] == "container/Dockerfile")
    assert built[1] == ["artifact.toml", "container/Dockerfile", "container/start"]
    found = computer.metadata["validated_images"]["container"]["amd64"]
    assert found["status"] == "uninspected" and "only its build context is stored" in found["reason"]
    assert _computer(_run(bundle_dir)).version == computer.version


def test_a_computer_naming_a_registry_image_is_written_over_its_pin_and_inspected(bundle_dir, builds, registry,
                                                                                  monkeypatch):
    served = Registry().serve(monkeypatch)
    amd64 = served.image(UBUNTU_LAYER)
    served.manifests[DIGEST] = served.manifests[served.index(amd64=amd64)]  # the digest the registry fixture pins
    layout(bundle_dir, _files('{ ref = "ghcr.io/acme/desktop:1" }'))

    done = _run(bundle_dir)

    found = _computer(done).metadata["validated_images"]["container"]["amd64"]
    assert DockerImageArtifact.get(IMAGE).image_name == f"ghcr.io/acme/desktop:1@{DIGEST}"
    assert (found["status"], found["manifest"], found["os_release"]) == (
        "passed", amd64, {"ID": "ubuntu", "VERSION_ID": "24.04"})
    assert "ghcr.io/acme/desktop:1" in registry


def test_a_computer_folder_that_declares_what_isnt_supported_is_refused_naming_the_folder(bundle_dir, builds):
    layout(bundle_dir, _files('{ dockerfile = "container/Dockerfile" }', os='{ type = "windows", version = "11" }'))

    with pytest.raises(BundleError) as refused:
        _run(bundle_dir)

    assert "artifacts/desktop: artifact.toml: os: Value error, os.type 'windows' isn't supported yet" in str(refused.value)


def test_a_registry_image_that_isnt_what_the_computer_declares_is_refused_naming_the_folder(bundle_dir, builds,
                                                                                         registry, monkeypatch):
    Registry().serve(monkeypatch).manifests[DIGEST] = {"manifests": []}
    layout(bundle_dir, _files('{ ref = "ghcr.io/acme/desktop:1" }'))

    with pytest.raises(BundleError) as refused:
        _run(bundle_dir)

    assert ("artifacts/desktop: computer '@local/~/desk/desktop': forms.container (amd64): it has no linux/amd64 image"
            in str(refused.value))


def test_an_image_key_naming_an_artifact_that_isnt_an_image_is_refused(bundle_dir, builds):
    layout(bundle_dir, {**_files('"notes"'), "artifacts/notes/notes.txt": "a note\n"})

    with pytest.raises(BundleError) as refused:
        _run(bundle_dir)

    assert "artifacts/desktop: forms.container.image: 'notes' is this bundle's file, but this field takes docker_image" in str(
        refused.value)
