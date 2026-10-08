"""What run_docker_container starts on a local sandbox is gone once the run is torn down.

A VM takes its containers down with it; a local sandbox shares this host's Docker, so it removes what
carries its label. Needs Docker and a pull of a small public image.
"""

import asyncio
import shutil
import subprocess
import uuid

import httpx
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


async def _status_once_up(url: str) -> int:
    """The status ``url`` answers with, once the server behind it has started."""
    async with httpx.AsyncClient() as client:
        for _ in range(30):
            try:
                return (await client.get(url, timeout=10)).status_code
            except httpx.TransportError:
                await asyncio.sleep(1)
    return (await httpx.AsyncClient().get(url, timeout=10)).status_code


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
    context = TaskStepContext(instance_id=f"rdc-{suffix}")
    try:
        context = await DeploySandboxTaskStep(
            id="box", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local",
        ).execute(context)
        sandbox_id = context.deployed_sandboxes[0].sandbox_id
        container, network = f"rdc-{suffix}-{sandbox_id}", f"rdc-net-{suffix}-{sandbox_id}"
        context = await RunDockerContainerTaskStep(
            id="ctr", version=None, sandbox_name="box", docker_context_artifact_id=build_context.id,
            container_name=f"rdc-{suffix}", network=f"rdc-net-{suffix}",
        ).execute(context)
        assert _docker("ps", "-q", "--filter", f"name=^{container}$")
    finally:
        report = await teardown_run(context)

    assert not report.still_up
    assert not _docker("ps", "-aq", "--filter", f"name=^{container}$")
    assert not _docker("images", "-q", f"{container}:latest")
    assert not _docker("network", "ls", "-q", "--filter", f"name=^{network}$")



@pytest.mark.asyncio
async def test_two_local_runs_use_the_same_container_name_network_and_port_at_once(local_backends):
    """Each sandbox gets its own container, network, image and host port, so the defaults can't collide."""
    suffix = uuid.uuid4().hex[:8]
    (local_backends / "Dockerfile").write_text("FROM mirror.gcr.io/library/nginx:1.27-bookworm\n")
    build_context = FileArtifactUniverse.put(id=f"rdc2-ctx-{suffix}", file_artifacts={
        "Dockerfile": FileArtifact.put(id=f"rdc2-df-{suffix}", description="Dockerfile",
                                       file_path=str(local_backends / "Dockerfile")),
    })

    async def run(name: str) -> TaskStepContext:
        context = TaskStepContext(instance_id=f"rdc2-{name}-{suffix}")
        context = await DeploySandboxTaskStep(
            id="box", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local", exposed_ports=[80],
        ).execute(context)
        return await RunDockerContainerTaskStep(
            id="ctr", version=None, sandbox_name="box", docker_context_artifact_id=build_context.id,
            network="task-net", ports=[80],
        ).execute(context)

    contexts = await asyncio.gather(run("a"), run("b"), return_exceptions=True)
    try:
        assert not [c for c in contexts if isinstance(c, BaseException)], contexts
        urls = [c.deployed_sandboxes[0].tunnel_urls["80"] for c in contexts]
        assert len(set(urls)) == 2
        assert [await _status_once_up(url) for url in urls] == [200, 200]
    finally:
        reports = [await teardown_run(c) for c in contexts if isinstance(c, TaskStepContext)]

    assert not any(r.still_up for r in reports)
    sandbox_ids = [c.deployed_sandboxes[0].sandbox_id for c in contexts]
    for sandbox_id in sandbox_ids:
        assert not _docker("ps", "-aq", "--filter", f"label=agentenv.sandbox={sandbox_id}")
        assert not _docker("network", "ls", "-q", "--filter", f"name=^task-net-{sandbox_id}$")
