"""``agent-env run`` on a bundle whose task starts its container from an image folder, ``artifacts/<name>/``, named as
``image_artifact_id``: the bundle writes the folder as its build context, the first run builds the image once where it
starts the container, as an agent's image is built where it's installed, a rerun builds nothing, and the grader in the
container writes the reward its score is. Needs a Docker daemon."""

import json
import logging
import subprocess

import pytest
from click.testing import CliRunner

from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers import host_images

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

TASK = json.dumps([
    {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", "sandbox_type": "local"},
    {"id": "task", "type": "run_docker_container", "sandbox_name": "box", "container_name": "task",
     "image_artifact_id": "task-image", "command_override": "sleep infinity", "depends_on": ["box"]},
    {"id": "grade", "type": "run_container_unit_tests_verifier", "sandbox_name": "box", "container_name": "task",
     "verifier_id": "tests", "reward_path": "/logs/verifier/reward.txt", "depends_on": ["task"],
     "command": "mkdir -p /logs/verifier && grep -qx built /built && echo 1 > /logs/verifier/reward.txt"},
    {"id": "down", "type": "teardown_sandboxes", "sandbox_names": ["box"], "depends_on": ["grade"]},
])


@pytest.fixture
def state(monkeypatch, tmp_path):
    """Local stores under this test's folder. Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        reset_artifact_store()
        reset_config()


def test_a_task_container_from_an_image_folder_is_built_by_the_first_run_alone(state, monkeypatch):
    root = state / "bundle"
    (root / "artifacts/task-image").mkdir(parents=True)
    (root / "artifacts/task-image/Dockerfile").write_text(
        "FROM mirror.gcr.io/library/debian:bookworm-slim\nRUN echo built > /built\n")
    (root / "tasks").mkdir()
    (root / "tasks/grade.json").write_text(TASK)
    builds, build = [], host_images._build
    monkeypatch.setattr(host_images, "_build", lambda *args: builds.append(args[2]) or build(*args))

    try:
        first = CliRunner().invoke(cli, ["run", str(root)])

        assert first.exit_code == 0 and "tasks/grade.json v1: passed" in first.output, first.output
        assert len(builds) == 1, (first.output, builds)

        again = CliRunner().invoke(cli, ["run", str(root)])

        assert again.exit_code == 0 and "tasks/grade.json v1: passed" in again.output, again.output
        assert "building with docker" not in again.output and len(builds) == 1, (again.output, builds)
    finally:
        subprocess.run(["docker", "rmi", "-f", *builds], capture_output=True)  # teardown keeps it, for a next run
