"""What run_docker_container starts on a local sandbox is gone once the run is torn down.

A VM takes its containers down with it; a local sandbox shares this host's Docker, so it removes what
carries its label. Needs Docker and a pull of a small public image.
"""

import shutil
import subprocess
import uuid

import pytest

from agent_env.artifact import FileArtifact, FileArtifactUniverse
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.run_docker_container import RunDockerContainerTaskStep
from tst.util.capabilities import missing_capability_reason

pytestmark = pytest.mark.integration


@pytest.fixture
def local_backends(monkeypatch, tmp_path):
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        pytest.skip(missing_capability_reason("docker_daemon"))
    monkeypatch.chdir(tmp_path)
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


def _docker(*args: str) -> str:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.mark.asyncio
async def test_a_run_docker_container_container_image_and_network_go_with_the_local_sandbox(local_backends):
    suffix = uuid.uuid4().hex[:8]
    (local_backends / "Dockerfile").write_text("FROM mirror.gcr.io/library/nginx:1.27-bookworm\n")
    build_context = FileArtifactUniverse.put(id=f"rdc-ctx-{suffix}", file_artifacts={
        "Dockerfile": FileArtifact.put(id=f"rdc-df-{suffix}", description="Dockerfile",
                                       file_path=str(local_backends / "Dockerfile")),
    })
    container, network = f"rdc-{suffix}", f"rdc-net-{suffix}"
    context = TaskStepContext(instance_id=f"rdc-{suffix}")
    try:
        context = await DeploySandboxTaskStep(
            id="box", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local",
        ).execute(context)
        context = await RunDockerContainerTaskStep(
            id="ctr", version=None, sandbox_name="box", docker_context_artifact_id=build_context.id,
            container_name=container, network=network,
        ).execute(context)
        assert _docker("ps", "-q", "--filter", f"name=^{container}$")
    finally:
        report = await teardown_run(context)

    assert not report.still_up
    assert not _docker("ps", "-aq", "--filter", f"name=^{container}$")
    assert not _docker("images", "-q", f"{container}:latest")
    assert not _docker("network", "ls", "-q", "--filter", f"name=^{network}$")
