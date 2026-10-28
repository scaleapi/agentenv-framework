"""``agent-env bundle register`` on the echo-mcp example agent-env ships, into the local stores and this machine's local
registry: its env's image is built with Docker and pushed, the registered task runs by its id and passes, and
registering again writes nothing. A registry it can't reach fails the registration before it names the image anywhere,
and once the registry is back a rerun finishes. Needs a Docker daemon and the local registry."""

import json
import logging
import shutil
import uuid
from pathlib import Path

import pytest
from click.testing import CliRunner

import agent_env
from agent_env.artifact import DockerImageArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.env.bootstrap import GATEWAY, SERVICE_DB, ensure_default_envs
from agent_env.env.env import Env
from agent_env.store.base import NotFoundError
from agent_env.task import Task

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ECHO = Path(agent_env.__file__).parent / "examples" / "echo-mcp"
IMAGE = "envs/echo (Dockerfile image)"
# A registry nothing serves: the plain OCI store, since the local registry's would start one on its port.
UNREACHABLE = """\
[stores.image]
impl = "agent_env.store.image_store:OciRegistryImageStore"
[stores.image.config]
registry_host = "127.0.0.1:9"
"""


@pytest.fixture
def bundle(monkeypatch, tmp_path):
    """A copy of the echo-mcp example, with the local stores under this test's folder and the local registry, whatever
    the config names, and a prefix of its own, so the registry, which outlives the test, holds nothing under its ids.
    Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    # HOME stays: docker's credential helper (the macOS keychain, for one) can hang a build's base-image lookup
    # when HOME moves.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_IMAGE_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    root = tmp_path / "echo"
    shutil.copytree(ECHO, root)
    try:
        yield root, f"it-{uuid.uuid4().hex[:8]}-"
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        reset_artifact_store()
        reset_config()


def _register(root, prefix):
    return CliRunner().invoke(cli, ["bundle", "register", str(root), "--prefix", prefix])


def test_a_registered_bundle_s_image_is_pushed_its_task_runs_by_id_and_registering_again_writes_nothing(bundle,
                                                                                                     tmp_path):
    root, prefix = bundle

    registered = _register(root, prefix)

    assert registered.exit_code == 0, registered.output
    assert registered.output.count("building with docker and pushing") == 1, registered.output
    assert f"{IMAGE}: building with docker and pushing" in registered.output, registered.output
    image = DockerImageArtifact.get(f"{prefix}echo__env_image")
    assert image.image_digest.startswith("sha256:") and image.build_context_object_url, image
    assert Env.get(f"{prefix}echo").docker_image_artifact.id == image.id

    # The gateway and service-db envs a stored task's deploy needs, which `agent-env up` sets up in a store this new.
    ensure_default_envs((SERVICE_DB, GATEWAY))
    ran = CliRunner().invoke(cli, ["task", "run", "--id", f"{prefix}smoke", "--output-dir", str(tmp_path / "out")])
    again = _register(root, prefix)

    assert ran.exit_code == 0, ran.output
    [context] = (tmp_path / "out").iterdir()
    assert json.loads(context.read_text())["metadata"]["verifications"]["echoes"]["score"] == 1.0, ran.output
    assert again.exit_code == 0, again.output
    assert "Nothing to write: every version is registered already." in again.output, again.output
    assert "building with docker" not in again.output, again.output


def test_a_registry_it_cant_reach_fails_it_before_the_image_is_named_and_a_rerun_once_it_is_back_finishes(
        bundle, tmp_path, monkeypatch):
    root, prefix = bundle
    config = tmp_path / "unreachable.toml"
    config.write_text(UNREACHABLE)
    with monkeypatch.context() as unreachable:
        unreachable.setenv("AGENT_ENV_CONFIG", str(config))
        unreachable.delenv("AGENT_ENV_IMAGE_STORE")
        reset_config()

        failed = _register(root, prefix)

        assert failed.exit_code != 0, failed.output
        assert "couldn't read its manifest from 127.0.0.1:9" in failed.output, failed.output
        assert "building with docker" not in failed.output, failed.output
        for read in (lambda: Env.get(f"{prefix}echo"), lambda: Task.get(f"{prefix}smoke")):
            with pytest.raises(NotFoundError):
                read()
    reset_config()

    finished = _register(root, prefix)

    assert finished.exit_code == 0, finished.output
    assert f"{IMAGE}: building with docker and pushing" in finished.output, finished.output
    assert Task.get(f"{prefix}smoke").version == 1
