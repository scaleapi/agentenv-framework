"""An agent installed into a task container on the local sandbox, played from this machine through the proxy installed
in its place: an image with the proxy baked in only starts it, one without gets uv and a venv, and both play the turn
and pass the task's tests. Needs a Docker daemon, and network access for the images, uv, Python and the proxy's
requirements."""

import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.artifact import DockerImageArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.task import EpisodeSpec, Task
from tst.util.proxy_bake import BAKE_RECIPE, PROXY_DIR

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

TASK_DOCKERFILE = """FROM mirror.gcr.io/library/debian:bookworm-slim
RUN mkdir -p /work && echo hello > /work/greeting.txt
WORKDIR /work
"""
# An agent the proxy replaces in a container needs no more than a build context.
REPLACED_DOCKERFILE = "FROM mirror.gcr.io/library/python:3.12-slim\n"


@pytest.fixture
def local_backends(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path / "sandboxes"))
    configure()
    reset_artifact_store()
    try:
        yield tmp_path
    finally:
        reset_artifact_store()
        reset_config()


@pytest.mark.asyncio
@pytest.mark.parametrize(("baked", "proxy"), [(True, "baked"), (False, "installed")], ids=["baked", "installed"])
async def test_an_agent_installed_into_a_task_container_is_played_through_the_proxy_in_its_place(local_backends, baked,
                                                                                                 proxy):
    uid = uuid.uuid4().hex[:8]
    image = _task_image(local_backends / "task-image", uid, baked)
    task = Task.from_dict({"id": f"proxy-local-{uid}", "steps": [
        {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", "sandbox_type": "local",
         "exposed_ports": [8000]},
        {"id": "task", "type": "run_docker_container", "sandbox_name": "box", "container_name": "task",
         "image_artifact_id": image.id, "ports": [8000], "command_override": "sleep infinity"},
        {"id": "install", "type": "install_agent", "sandbox_name": "box", "container_name": "task",
         "a2a_agent_id": _replaced(uid).id, "agent_name": "solver"},
        {"id": "solve", "type": "prompt_agent", "agent_name": "solver", "prompt_id": "solve",
         "prompt": "Copy greeting.txt to answer.txt."},
        {"id": "check", "type": "run_container_unit_tests_verifier", "sandbox_name": "box", "container_name": "task",
         "command": "grep -qx hello /work/answer.txt"},
    ]})

    try:
        episode = await EpisodeSpec.from_task(task).start()
        try:
            turn = await episode.next_turn()
            async with turn.tools() as tools:
                where = "test -e /opt/agentenv-proxy-8000/bin/uv && echo installed || echo baked"
                assert (await tools.call("bash", {"command": f"cat greeting.txt; {where}"})).text == f"hello\n{proxy}\n"
                await tools.call("bash", {"command": "cp greeting.txt answer.txt"})
            await turn.end("Copied.")
            assert await episode.result() == {"check": 1.0}
        finally:
            await episode.abandon("the test ended")
    finally:
        subprocess.run(["docker", "rmi", "-f", image.image_name], capture_output=True)


def _task_image(context: Path, uid: str, baked: bool) -> DockerImageArtifact:
    context.mkdir()
    bake = "".join(f"{line}\n" for line in BAKE_RECIPE) if baked else ""
    (context / "Dockerfile").write_text(TASK_DOCKERFILE + bake)
    if baked:
        shutil.copytree(PROXY_DIR, context / "agentenv-proxy", ignore=shutil.ignore_patterns("__pycache__"))
    return DockerImageArtifact.put_context(f"proxy-task-image-{uid}", description="a task container",
                                           context_path=str(context), dockerfile_path="Dockerfile", platform=None)


def _replaced(uid: str) -> A2AAgent:
    with tempfile.TemporaryDirectory() as build_dir:
        context = Path(build_dir)
        (context / "Dockerfile").write_text(REPLACED_DOCKERFILE)
        image = DockerImageArtifact.put_context(f"replaced-agent-image-{uid}", description="a replaced agent",
                                                context_path=str(context), dockerfile_path="Dockerfile", platform=None)
    return A2AAgent.put(id=f"replaced-agent-{uid}", docker_image_artifact=image)
