"""The computers bundle agent-env ships: a task naming ubuntu-24.04 writes its base, its image built from the folder's
Dockerfile, or once a release has published it, the published image, read back from its registry; and the start script
falls back to the defaults artifact.toml declares."""

import json
import re
import tomllib
from pathlib import Path

import pytest

import agent_env
from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.bundle import parse_bundle
from agent_env.bundle.installed import installed_bundles
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import AliasUse, plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.entity.refs import EntityKind
from agent_env.env.published_images import COMPUTER_IMAGES
from tst.unit.bundle._support import layout, local_builds
from tst.util.oci_registry import UBUNTU_LAYER, Registry

BASE = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04"
UBUNTU = "@local/agentenv-framework/computers/ubuntu-24.04"


@pytest.fixture
def unpublished(monkeypatch):
    monkeypatch.setattr("agent_env.env.published_images._registry_digest", lambda ref: None)


def test_a_task_naming_ubuntu_24_04_writes_the_shipped_base_over_its_folders_image(
        tmp_path, monkeypatch, local_stores, cli_routing, builds, unpublished):
    monkeypatch.setenv("HOME", str(tmp_path))
    plan = _naming_ubuntu(tmp_path)

    done = materialize(plan, build_here=local_builds(plan))

    computer = ComputerArtifact.get(UBUNTU, done.version_of("artifact", UBUNTU))
    image = DockerImageArtifact.get(computer.forms.container.image.id, computer.forms.container.image.version)
    assert plan.alias_uses[EntityKind.ARTIFACT]["ubuntu-24.04"] == AliasUse(
        "ubuntu-24.04", UBUNTU, "agentenv-framework's computers bundle", kind=EntityKind.ARTIFACT)
    assert (computer.arch, computer.os.version, computer.session.user) == (["amd64", "arm64"], "24.04", "user")
    assert image.id == f"{UBUNTU}__forms_container_image" and image.dockerfile_path == "Dockerfile"
    built = next(call["build"] for call in builds if call["build"][2] == image.image_name)
    assert built[1] == ["Dockerfile", "artifact.toml", "start-desktop", "supervisord.conf", "xfwm4.xml"]
    assert computer.metadata["validated_images"]["container"]["arm64"]["status"] == "uninspected"


def test_a_task_naming_ubuntu_24_04_after_a_release_writes_the_shipped_base_over_its_published_image(
        tmp_path, monkeypatch, local_stores, cli_routing, builds):
    served = Registry().serve(monkeypatch)
    index = served.index(amd64=served.image(UBUNTU_LAYER), arm64=served.image(UBUNTU_LAYER, arch="arm64"))
    monkeypatch.setattr("agent_env.env.published_images._registry_digest", lambda ref: index)
    monkeypatch.setenv("HOME", str(tmp_path))
    plan = _naming_ubuntu(tmp_path)

    done = materialize(plan, build_here=local_builds(plan))

    computer = ComputerArtifact.get(UBUNTU, done.version_of("artifact", UBUNTU))
    image = DockerImageArtifact.get(computer.forms.container.image.id, computer.forms.container.image.version)
    assert (image.image_name, image.image_digest) == (COMPUTER_IMAGES[0].ref(), index)
    assert not any(call["build"][2] == image.image_name for call in builds)
    found = computer.metadata["validated_images"]["container"]
    assert {arch: (r["status"], r["image"]) for arch, r in found.items()} == {
        arch: ("passed", f"{image.image_name}@{index}") for arch in ("amd64", "arm64")}


def test_the_start_script_falls_back_to_the_defaults_artifact_toml_declares():
    declared = tomllib.loads((BASE / "artifact.toml").read_text())["config"]
    script = (BASE / "start-desktop").read_text()
    fallbacks = dict(re.findall(r'\$\{AGENTENV_SETTING_([A-Z_]+):-([^}]*)\}', script))
    assert fallbacks == {name.upper(): setting["default"] for name, setting in declared.items()}


def _naming_ubuntu(root):
    layout(root / "uses", {
        "envs/crm/Dockerfile": "FROM scratch\n", "envs/crm/env.toml": 'environment_name = "crm"\n',
        "tasks/t.json": json.dumps([{"id": "crm", "type": "deploy_env", "env_id": "crm"},
                                    {"id": "base", "type": "load_artifact", "env_id": "crm", "artifact_id": "ubuntu-24.04",
                                     "depends_on": ["crm"]}])})
    return plan_bundle(resolve_bundle(parse_bundle(root / "uses")), installed=installed_bundles)
