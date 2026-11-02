"""A registry image with its build context beside it: written as the image the registry serves, pinned by digest,
never built, with the folder kept as its context, which a deploy builds only when it can't pull the image."""

import io
import json
import tarfile

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.bundle import dry_run_bundle
from agent_env.bundle.materialize import materialize
from agent_env.config import get_config
from agent_env.env import Env
from tst.unit.bundle._support import DIGEST, layout, local_builds, plan_of

ROOT = "@local/~/triage"
IMAGE = f"{ROOT}/crm-image"
WEB = f"{ROOT}/web"
PINNED = "ghcr.io/acme/web:1@sha256:" + "b" * 64
FOLDER = {
    "artifacts/crm-image/Dockerfile": "FROM scratch\nCOPY server.py /app/\n",
    "artifacts/crm-image/server.py": "print('crm')\n",
    "artifacts/crm-image/artifact.toml": 'type = "docker_image"\nref = "ghcr.io/acme/crm:1"\n',
    "envs/crm/env.toml": 'image = "crm-image"\nenvironment_name = "crm"\n',
    "agents/solver/agent.toml": 'image = "crm-image"\n',
    "tasks/t.json": json.dumps([
        {"id": "crm", "type": "deploy_env", "env_id": "crm"},
        {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"},
    ]),
}
WEB_FILES = {
    "envs/web/Dockerfile": "FROM scratch\nCOPY server.py /app/\n",
    "envs/web/server.py": "print('web')\n",
    "envs/web/env.toml": f'environment_name = "web"\nimage = {{ ref = "{PINNED}", dockerfile = "Dockerfile" }}\n',
    "tasks/t.json": json.dumps([{"id": "web", "type": "deploy_env", "env_id": "web"}]),
}


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "triage"


def _run(root):
    plan = plan_of(root)
    return materialize(plan, build_here=local_builds(plan))


def _context(image):
    url = image.build_context_object_url
    with get_config().get_object_store_at(url).open(url) as stream:
        data = stream.read()
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return sorted(member.name.removeprefix("./") for member in tar.getmembers() if member.isfile())


def test_an_image_folder_naming_a_registry_image_keeps_its_dockerfile_as_its_context_and_builds_nothing(
        bundle_dir, builds, registry):
    layout(bundle_dir, FOLDER)

    done = _run(bundle_dir)

    image = DockerImageArtifact.get(IMAGE)
    assert (image.image_name, image.image_digest, image.pinned_name, image.dockerfile_path) == (
        "ghcr.io/acme/crm:1", DIGEST, f"ghcr.io/acme/crm:1@{DIGEST}", "Dockerfile")
    assert _context(image) == ["Dockerfile", "artifact.toml", "server.py"]
    env = Env.get(f"{ROOT}/crm", done.version_of("env", f"{ROOT}/crm")).docker_image_artifact
    agent = A2AAgent.get(f"{ROOT}/solver", done.version_of("agent", f"{ROOT}/solver")).docker_image_artifact
    assert [(each.id, each.version) for each in (env, agent)] == [(IMAGE, 1)] * 2
    assert (builds, registry) == ([], ["ghcr.io/acme/crm:1"])


def test_a_rerun_reuses_it_and_an_edit_to_its_context_rewrites_it_keeping_the_digest_it_was_pinned_to(
        bundle_dir, builds, registry):
    layout(bundle_dir, FOLDER)
    _run(bundle_dir)
    rerun = {done.write.id: (done.version, done.reused) for done in _run(bundle_dir).writes}
    (bundle_dir / "artifacts/crm-image/server.py").write_text("print('crm v2')\n")

    edited = {done.write.id: (done.version, done.reused) for done in _run(bundle_dir).writes}

    assert (rerun[IMAGE], edited[IMAGE]) == ((1, True), (2, False))
    assert DockerImageArtifact.get(IMAGE).pinned_name == f"ghcr.io/acme/crm:1@{DIGEST}"
    assert (builds, registry) == ([], ["ghcr.io/acme/crm:1"])


def test_an_image_key_naming_a_digest_and_a_dockerfile_records_that_digest_and_the_folder_as_its_context(
        bundle_dir, builds, registry):
    layout(bundle_dir, WEB_FILES)

    done = _run(bundle_dir)

    image = Env.get(WEB, done.version_of("env", WEB)).docker_image_artifact
    assert (image.id, image.image_name, image.pinned_name) == (f"{WEB}__env_image", "ghcr.io/acme/web:1", PINNED)
    assert _context(image) == ["Dockerfile", "env.toml", "server.py"]
    assert (builds, registry) == ([], [PINNED])


def test_a_dry_run_says_the_image_is_pulled_and_its_context_kept(bundle_dir, docker_on_path):
    layout(bundle_dir, WEB_FILES)

    assert dry_run_bundle(bundle_dir, sandbox="local").images() == (
        f"envs/web ({PINNED} image) v1: pulled from its registry, its tag pinned to a digest when it's written, with "
        "its build context beside it, which a deploy builds only when it can't pull it",)
