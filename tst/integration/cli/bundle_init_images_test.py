"""``agent-env bundle init`` writes each image a task uses as its build context, pinned to the Python and global npm
versions the stored image holds: the folder plans against an empty store, and each image built from it installs what
the stored one did, though its Dockerfile names no version and newer ones have been released since. Needs Docker and
the network, for PyPI and npm."""

import gzip
import logging
import shutil
import subprocess
import uuid

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store, reset_artifact_store
from agent_env.bundle.resolve import build_step
from agent_env.cli import cli
from agent_env.config import configure, get_config, reset_config
from agent_env.env import MCPServerEnv
from agent_env.task import Task
from agent_env.utils.docker.build import build_image
from tst.util.capabilities import missing_capability_reason

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

PYTHON = "mirror.gcr.io/library/python:3.12-slim@sha256:423ed6ab25b1921a477529254bfeeabf5855151dc2c3141699a1bfc852199fbf"
NODE = "mirror.gcr.io/library/node:20-slim"
# Old releases, which an install naming no version no longer resolves to.
PINNED_PYTHON = {"requests": "2.31.0", "urllib3": "2.0.7", "idna": "3.6"}
PINNED_NPM = {"typescript": "5.4.5"}


@pytest.fixture
def machine(monkeypatch, tmp_path):
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "source"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    monkeypatch.setenv("LITELLM_BASE_URL", "http://127.0.0.1:4000")  # the receiver's model endpoint, which no step calls
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    configure()
    reset_artifact_store()
    tags = []
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield tags
    finally:
        logging.disable(logging.NOTSET)
        for tag in tags:
            subprocess.run(["docker", "rmi", "-f", tag], capture_output=True)
        reset_artifact_store()
        reset_config()


def _stored(tmp_path, tags, id, built_from, dockerfile):
    """An image the store holds as ``built_from`` built it, saved as its tar.gz, with a build context whose Dockerfile
    is ``dockerfile``: an image built a while ago from a Dockerfile naming no versions."""
    built, context = tmp_path / f"{id}-built", tmp_path / f"{id}-context"
    for folder, text in ((built, built_from), (context, dockerfile)):
        folder.mkdir()
        (folder / "Dockerfile").write_text(text)
    tag = f"bundle-init-{id}-{uuid.uuid4().hex[:8]}:v1"
    tags.append(tag)
    build_image(built / "Dockerfile", built, tag, platform=None)
    saved = tmp_path / f"{id}.tar.gz"
    with gzip.open(saved, "wb") as out:
        subprocess.run(["docker", "save", tag], stdout=out, check=True)
    contexted = DockerImageArtifact.put_context(f"{id}-context", description=id, context_path=str(context),
                                                dockerfile_path="Dockerfile", platform=None)
    url = contexted.build_context_object_url.rsplit("/", 1)[0] + f"/{id}.tar.gz"
    url = get_config().get_object_store_at(url).put_file_at(url, str(saved), "application/gzip")
    return get_artifact_store().put_document(DockerImageArtifact(
        id=id, description=id, image_name=tag, tar_gz_s3_url=url,
        build_context_s3_url=contexted.build_context_object_url, dockerfile_path="Dockerfile"))


def _ok(*args):
    result = CliRunner().invoke(cli, [*map(str, args)])
    assert result.exit_code == 0, result.output
    return result.output


def _in(tag, *command):
    return subprocess.run(["docker", "run", "--rm", tag, *command], capture_output=True, text=True,
                          check=True).stdout


def test_images_built_from_an_imported_folder_install_the_stored_images_versions(machine, tmp_path, monkeypatch):
    pins = " ".join(f"{name}=={version}" for name, version in PINNED_PYTHON.items())
    env_image = _stored(tmp_path, machine, "crm-image", f"FROM {PYTHON}\nRUN pip install --no-cache-dir {pins}\n",
                        f"FROM {PYTHON}\nRUN pip install --no-cache-dir requests\n")
    agent_image = _stored(tmp_path, machine, "solver-image",
                          f"FROM {NODE}\nRUN npm install -g typescript@{PINNED_NPM['typescript']}\n",
                          f"FROM {NODE}\nRUN npm install -g typescript\n")
    MCPServerEnv.put(id="crm", docker_image_artifact=env_image, environment_name="crm")
    A2AAgent.put(id="solver", docker_image_artifact=agent_image)
    Task.put(id="t", steps=[build_step(step) for step in (
        {"id": "env", "type": "deploy_env", "env_id": "crm"},
        {"id": "agent", "type": "deploy_agent", "agent_name": "solver", "env_ids": ["crm"], "a2a_agent_id": "solver"},
    )])
    folder = tmp_path / "t"

    imported = _ok("bundle", "init", folder, "--from-task", "t", "--include-scope", "user")

    assert "pinned to the stored image: " in imported
    assert "typescript@5.4.5" in (folder / "agents" / "solver" / "Dockerfile").read_text()
    constraints = (folder / "envs" / "crm" / "constraints.txt").read_text().splitlines()
    assert {f"{name}=={version}" for name, version in PINNED_PYTHON.items()} <= set(constraints)

    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "receiver"))
    reset_artifact_store()
    reset_config()
    configure()
    assert "Would run:" in _ok("run", folder, "--dry-run", "--sandbox", "local")

    for entity, check in (("envs/crm", ("pip", "list", "--format=freeze")), ("agents/solver", ("tsc", "--version"))):
        tag = f"bundle-init-rebuilt-{uuid.uuid4().hex[:8]}:v1"
        machine.append(tag)
        build_image(folder / entity / "Dockerfile", folder / entity, tag, platform=None)
        installed = _in(tag, *check)
        if entity == "envs/crm":
            assert {f"{name}=={version}" for name, version in PINNED_PYTHON.items()} <= set(installed.splitlines())
        else:
            assert installed.strip() == f"Version {PINNED_NPM['typescript']}"
