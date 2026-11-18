"""What run_docker_container starts on a local sandbox is gone once the run is torn down, and an image artifact it
runs is built once and kept for the next run.

A VM takes its containers down with it; a local sandbox shares this host's Docker, so it removes what
carries its label. Needs Docker and a pull of a small public image.
"""

import asyncio
import functools
import http.server
import shutil
import subprocess
import threading
import uuid

import httpx
import pytest

from agent_env.artifact import DockerImageArtifact, FileArtifact, FileArtifactUniverse
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers import host_images
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
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

    # Made up front and filled in place, so a run that fails partway is still torn down.
    contexts = [TaskStepContext(instance_id=f"rdc2-{name}-{suffix}") for name in "ab"]

    async def run(context: TaskStepContext) -> None:
        await DeploySandboxTaskStep(
            id="box", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local", exposed_ports=[80],
        ).execute(context)
        await RunDockerContainerTaskStep(
            id="ctr", version=None, sandbox_name="box", docker_context_artifact_id=build_context.id,
            network="task-net", ports=[80],
        ).execute(context)

    outcomes = await asyncio.gather(*(run(c) for c in contexts), return_exceptions=True)
    try:
        assert not [o for o in outcomes if isinstance(o, BaseException)], outcomes
        urls = [c.deployed_sandboxes[0].tunnel_urls["80"] for c in contexts]
        assert len(set(urls)) == 2
        assert [await _status_once_up(url) for url in urls] == [200, 200]
    finally:
        reports = [await teardown_run(c) for c in contexts]

    assert not any(r.still_up for r in reports)
    sandbox_ids = [c.deployed_sandboxes[0].sandbox_id for c in contexts]
    for sandbox_id in sandbox_ids:
        assert not _docker("ps", "-aq", "--filter", f"label=agentenv.sandbox={sandbox_id}")
        assert not _docker("network", "ls", "-q", "--filter", f"name=^task-net-{sandbox_id}$")


@pytest.mark.asyncio
async def test_urls_load_into_a_run_docker_container_container(local_backends):
    """A URL given with container_name lands in that container, not the sandbox's agent container."""
    suffix = uuid.uuid4().hex[:8]
    (local_backends / "Dockerfile").write_text("FROM mirror.gcr.io/library/nginx:1.27-bookworm\n")
    (local_backends / "data.csv").write_text("a,b\n1,2\n")
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(local_backends)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    build_context = FileArtifactUniverse.put(id=f"rdc3-ctx-{suffix}", file_artifacts={
        "Dockerfile": FileArtifact.put(id=f"rdc3-df-{suffix}", description="Dockerfile",
                                       file_path=str(local_backends / "Dockerfile")),
    })
    context = TaskStepContext(instance_id=f"rdc3-{suffix}")
    try:
        await DeploySandboxTaskStep(
            id="box", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local",
        ).execute(context)
        await RunDockerContainerTaskStep(
            id="ctr", version=None, sandbox_name="box", docker_context_artifact_id=build_context.id,
            container_name="worker",
        ).execute(context)
        await LoadArtifactTaskStep(
            id="load", version=None, sandbox_name="box", container_name="worker", destination_path="/work",
            urls=[f"http://127.0.0.1:{server.server_address[1]}/data.csv"],
        ).execute(context)
        container = f"worker-{context.deployed_sandboxes[0].sandbox_id}"
        assert _docker("exec", container, "cat", "/work/data.csv") == "a,b\n1,2"
    finally:
        server.shutdown()
        report = await teardown_run(context)

    assert not report.still_up


@pytest.mark.asyncio
async def test_runs_of_an_image_artifact_build_it_once_and_teardown_leaves_it_for_the_next(local_backends, monkeypatch):
    """The first run builds the image from its context, a later run starts its container from it with nothing built,
    and a sandbox's teardown removes only what it started."""
    suffix = uuid.uuid4().hex[:8]
    (local_backends / "image").mkdir()
    (local_backends / "image/Dockerfile").write_text(
        f"FROM mirror.gcr.io/library/debian:bookworm-slim\nRUN echo {suffix} > /built\n")
    image = DockerImageArtifact.put_context(f"rdc4-img-{suffix}", description="a task image",
                                            context_path=str(local_backends / "image"), dockerfile_path="Dockerfile",
                                            platform=None)
    builds, build = [], host_images._build
    monkeypatch.setattr(host_images, "_build", lambda *args: builds.append(args[2]) or build(*args))
    try:
        for run in "ab":
            context = TaskStepContext(instance_id=f"rdc4-{run}-{suffix}")
            try:
                await DeploySandboxTaskStep(
                    id="box", version=None, sandbox_name="box", sandbox_mode="vm", sandbox_type="local",
                ).execute(context)
                await RunDockerContainerTaskStep(
                    id="ctr", version=None, sandbox_name="box", image_artifact_id=image.id, container_name="task",
                    command_override="sleep infinity",
                ).execute(context)
                assert _docker("exec", f"task-{context.deployed_sandboxes[0].sandbox_id}", "cat", "/built") == suffix
            finally:
                report = await teardown_run(context)
            assert not report.still_up
            assert builds == [image.image_name]
            assert _docker("images", "-q", image.image_name)
    finally:
        subprocess.run(["docker", "rmi", "-f", image.image_name], capture_output=True)
