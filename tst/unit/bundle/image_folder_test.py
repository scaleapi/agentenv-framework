"""Image folders: an ``artifacts/<name>/`` docker_image folder is written under its own id as the image built from its
Dockerfile, or as the registry image its artifact.toml names, and every env and agent naming it is written over it."""

import json

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts import docker_image as docker_image_module
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle.materialize import materialize
from agent_env.env import Env
from agent_env.store.ids import image_repository
from agent_env.store.routing import namespace_routing
from tst.unit.bundle._support import DIGEST, layout, plan_of

ROOT = "@local/~/triage"
IMAGE = f"{ROOT}/crm-image"
MOVED = "sha256:" + "b" * 64
NAMING = (f"{ROOT}/crm", f"{ROOT}/crm-eu", f"{ROOT}/solver")
CARD = 'from agentenv_protocol import environment_card\n\n\n@environment_card(name="crm")\nclass Server:\n    pass\n'
USERS = {
    "envs/crm/env.toml": 'image = "crm-image"\nenvironment_name = "crm"\n',
    "envs/crm-eu/env.toml": 'image = "crm-image"\nenvironment_name = "crm"\n',
    "agents/solver/agent.toml": 'image = "crm-image"\n',
    "tasks/t.json": json.dumps([
        {"id": "crm", "type": "deploy_env", "env_id": "crm"},
        {"id": "crm-eu", "type": "deploy_env", "env_id": "crm-eu"},
        {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"},
    ]),
}
BUILT = {"artifacts/crm-image/Dockerfile": "FROM scratch\nCOPY . /app\n", "artifacts/crm-image/server.py": CARD,
         **USERS}
REFERENCED = {"artifacts/crm-image/artifact.toml": 'type = "docker_image"\nref = "ghcr.io/acme/crm:1"\n', **USERS}


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "triage"


def _run(root):
    return materialize(plan_of(root))


def _summary(materialization):
    return {done.write.id: (done.version, done.reused, done.reasons) for done in materialization.writes}


def _images_named(done):
    """The image each env and agent naming the folder was written over."""
    envs = [Env.get(id, done.version_of("env", id)).docker_image_artifact for id in NAMING[:2]]
    agent = A2AAgent.get(NAMING[2], done.version_of("agent", NAMING[2])).docker_image_artifact
    return [(image.id, image.version) for image in (*envs, agent)]


def test_an_image_folder_is_built_once_and_everything_naming_it_is_written_over_it(bundle_dir, builds):
    layout(bundle_dir, BUILT)

    done = _run(bundle_dir)

    assert [call["build"][:3] for call in builds] == [
        ("Dockerfile", ["Dockerfile", "server.py"], f"{image_repository(IMAGE)}:bundle")]
    assert DockerImageArtifact.get(IMAGE).description == "built from artifacts/crm-image/Dockerfile"
    assert _images_named(done) == [(IMAGE, 1)] * 3
    assert _summary(_run(bundle_dir))[IMAGE] == (1, True, ())
    assert len(builds) == 1


def test_an_edit_to_an_image_folder_rebuilds_it_and_rewrites_everything_naming_it(bundle_dir, builds):
    layout(bundle_dir, BUILT)
    _run(bundle_dir)
    (bundle_dir / "artifacts/crm-image/server.py").write_text(CARD + "# v2\n")

    done = _run(bundle_dir)

    assert len(builds) == 2
    assert {id: _summary(done)[id][:2] for id in (IMAGE, *NAMING)} == dict.fromkeys((IMAGE, *NAMING), (2, False))
    assert _images_named(done) == [(IMAGE, 2)] * 3


def test_an_image_folders_artifact_toml_describes_it_and_is_in_its_build_context(bundle_dir, builds):
    layout(bundle_dir, {**BUILT, "artifacts/crm-image/artifact.toml": 'description = "the CRM server"\n'})

    _run(bundle_dir)

    assert DockerImageArtifact.get(IMAGE).description == "the CRM server"
    assert builds[0]["build"][1] == ["Dockerfile", "artifact.toml", "server.py"]


def test_an_image_folder_naming_a_registry_image_is_pinned_once_for_everything_naming_it(bundle_dir, registry):
    layout(bundle_dir, REFERENCED)

    done = _run(bundle_dir)

    image = DockerImageArtifact.get(IMAGE)
    assert (image.image_name, image.description) == (
        f"ghcr.io/acme/crm:1@{DIGEST}", "referenced by artifacts/crm-image")
    assert _images_named(done) == [(IMAGE, 1)] * 3
    assert _summary(_run(bundle_dir))[IMAGE] == (1, True, ())
    assert registry == ["ghcr.io/acme/crm:1"]


def test_a_registry_image_folder_rewritten_for_its_description_keeps_its_digest_until_its_ref_changes(
        bundle_dir, monkeypatch):
    served, asked = {"digest": DIGEST}, []

    def pin(ref, auth):
        asked.append(ref)
        return f"{ref}@{served['digest']}"

    monkeypatch.setattr(docker_image_module, "pin_digest", pin)
    toml = bundle_dir / "artifacts/crm-image/artifact.toml"
    layout(bundle_dir, REFERENCED)
    _run(bundle_dir)
    served["digest"] = MOVED  # the tag moves in its registry
    toml.write_text('type = "docker_image"\nref = "ghcr.io/acme/crm:1"\ndescription = "the CRM server"\n')

    done = _run(bundle_dir)

    image = DockerImageArtifact.get(IMAGE)
    assert (_summary(done)[IMAGE][:2], image.description, image.image_name) == (
        (2, False), "the CRM server", f"ghcr.io/acme/crm:1@{DIGEST}")
    assert _images_named(done) == [(IMAGE, 2)] * 3
    assert asked == ["ghcr.io/acme/crm:1"]
    toml.write_text('type = "docker_image"\nref = "ghcr.io/acme/crm:2"\n')

    _run(bundle_dir)

    assert DockerImageArtifact.get(IMAGE).image_name == f"ghcr.io/acme/crm:2@{MOVED}"


def test_a_registry_image_folder_replaces_another_type_of_artifact_its_id_held(bundle_dir, registry):
    with namespace_routing():
        get_artifact_store().put_document(FileArtifact(
            id=IMAGE, description="a file first", object_url="file:///notes.txt", filename="notes.txt",
            content_type="text/plain"))
    layout(bundle_dir, REFERENCED)

    done = _run(bundle_dir)

    assert DockerImageArtifact.get(IMAGE).image_name == f"ghcr.io/acme/crm:1@{DIGEST}"
    assert _images_named(done) == [(IMAGE, 2)] * 3
    assert registry == ["ghcr.io/acme/crm:1"]


def test_an_image_folder_may_ship_a_skill_in_its_image(bundle_dir, builds):
    layout(bundle_dir, {**BUILT, "artifacts/crm-image/SKILL.md": "---\nname: crm\ndescription: the CRM\n---\n"})

    _run(bundle_dir)

    assert builds[0]["build"][1] == ["Dockerfile", "SKILL.md", "server.py"]


def test_a_website_names_image_folders_for_its_backend_and_frontend(bundle_dir, builds):
    layout(bundle_dir, {
        "artifacts/shop-back/Dockerfile": "FROM scratch\n",
        "artifacts/shop-front/Dockerfile": "FROM scratch\n",
        "envs/shop/env.toml": 'type = "website"\nenvironment_name = "shop"\nbackend_image = "shop-back"\n'
                              'frontend_image = "shop-front"\n',
        "tasks/t.json": json.dumps([{"id": "shop", "type": "deploy_env", "env_id": "shop"}]),
    })

    done = _run(bundle_dir)

    shop = Env.get(f"{ROOT}/shop", done.version_of("env", f"{ROOT}/shop"))
    assert (shop.backend_docker_image_artifact.id, shop.frontend_docker_image_artifact.id) == (
        f"{ROOT}/shop-back", f"{ROOT}/shop-front")
    assert len(builds) == 2
