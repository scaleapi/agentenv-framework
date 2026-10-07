"""The containers a local sandbox starts are held to the cpu and memory it was created with.

A VM's size bounds its containers; a local sandbox shares this host's Docker, so Docker holds each one.
Needs Docker and a pull of a small public image.
"""

import shutil
import subprocess
import uuid

import pytest

from agent_env.artifact import FileArtifact, FileArtifactUniverse
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.run_docker_container import RunDockerContainerTaskStep
from tst.util.capabilities import missing_capability_reason

pytestmark = pytest.mark.integration

_IMAGE = "mirror.gcr.io/library/nginx:1.27-bookworm"


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


def _limits(container: str) -> tuple[int, int, int]:
    """``(NanoCpus, Memory, MemorySwap)`` Docker holds ``container`` to."""
    out = subprocess.run(
        ["docker", "inspect", container, "--format", "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}} {{.HostConfig.MemorySwap}}"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return int(out[0]), int(out[1]), int(out[2])


@pytest.mark.asyncio
async def test_a_container_a_step_starts_on_a_local_sandbox_is_held_to_its_cpu_and_memory(local_backends):
    suffix = uuid.uuid4().hex[:8]
    (local_backends / "Dockerfile").write_text(f"FROM {_IMAGE}\n")
    build_context = FileArtifactUniverse.put(id=f"lim-ctx-{suffix}", file_artifacts={
        "Dockerfile": FileArtifact.put(id=f"lim-df-{suffix}", description="Dockerfile",
                                       file_path=str(local_backends / "Dockerfile")),
    })
    container = f"lim-{suffix}"
    context = TaskStepContext(instance_id=f"lim-{suffix}")
    try:
        context = await DeploySandboxTaskStep(
            id="box", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local",
            cpu=0.5, memory_mb=256,
        ).execute(context)
        context = await RunDockerContainerTaskStep(
            id="ctr", version=None, sandbox_name="box", docker_context_artifact_id=build_context.id,
            container_name=container,
        ).execute(context)

        assert _limits(container) == (500_000_000, 256 * 2**20, 256 * 2**20)
    finally:
        await teardown_run(context)


@pytest.mark.asyncio
async def test_a_container_mode_local_sandbox_is_held_to_its_cpu_and_memory(local_backends):
    sandbox = await LocalSandboxProvider().create_sandbox(image_name=_IMAGE, port=80, env={}, cpu=0.5, memory=256)
    try:
        assert _limits(sandbox.container_name) == (500_000_000, 256 * 2**20, 256 * 2**20)
    finally:
        await sandbox.terminate()
