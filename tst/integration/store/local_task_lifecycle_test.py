"""No-infra task-lifecycle e2e: local docs + local objects + local images + LocalSandbox.

A Task with a single container-mode ``DeploySandboxTaskStep`` runs end-to-end with
zero external infrastructure — the local-backend capstone at the deploy tier:
- ``DockerImageArtifact.put`` builds a local image, pushes it to a local OCI
  registry (image store), saves the tar.gz to the filesystem (object store), and
  records the artifact doc (SQLite).
- ``Task.run`` registers the task-instance (SQLite) and executes the deploy step,
  which pulls the image from the local registry into a ``LocalSandbox`` container.
- The container is reachable on its published localhost port.

Requires a docker daemon; spins up a throwaway ``registry:2`` and skips if it
can't start.

Neither the container name nor the host port is fixed. ``LocalSandbox``
derives its container name from the sandbox id (``agent-{sandbox_id}``) so
concurrent deploys cannot collide, and ``LocalSandboxProvider.create_vm``
publishes each exposed container port on a freshly allocated FREE host port —
the local backend packs every sandbox onto one Docker host, so the requested
port can only be published once. So this test reads the served URL out of the
deploy's own ``tunnel_urls`` rather than assuming identity, and cleans up by
the container name the sandbox actually used — registered with the fixture, which
removes exactly the containers this run created and nothing a sibling worker owns.
"""

import datetime as dt
import re
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

import httpx
import pytest

from agent_env.artifact import DockerImageArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.config import configure, reset_config, set_image_store
from agent_env.store.image_store import LocalRegistryImageStore
from agent_env.task import Task
from agent_env.task_step.task_steps.deploy_sandbox import DeploySandboxTaskStep

pytestmark = pytest.mark.integration


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


# A local-sandbox container younger than this may belong to a run still in flight — a
# sibling xdist worker, or a developer's own test — so the crash-leftover prune leaves it
# alone. Comfortably above both the deploy's own `ttl_seconds=600` and any plausible
# integration-tier runtime, so nothing live is ever in range.
_STALE_CONTAINER_AGE_SECONDS = 2 * 60 * 60


def _container_age_seconds(container_id: str) -> float | None:
    """Seconds since the container was created, or None if docker won't say."""
    created = _docker("inspect", "-f", "{{.Created}}", container_id).stdout.strip()
    if not created:
        return None
    try:
        # RFC3339 with nanoseconds; datetime.fromisoformat wants <=6 fractional digits.
        stamp = re.sub(r"(\.\d{6})\d+", r"\1", created).replace("Z", "+00:00")
        return (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(stamp)).total_seconds()
    except ValueError:
        return None


def _rm_stale_local_agent_containers() -> None:
    """Remove local-sandbox agent containers left behind by a run that crashed.

    ``LocalSandbox.container_name`` is ``agent-{sandbox_id}`` and a local sandbox id is
    always ``local-<hex>``, so ``agent-local-`` matches exactly the containers this backend
    creates and nothing else (the a2a agent's fixed ``agent-api`` included). A run that dies
    before its teardown leaves one behind, so prune by that substring instead of by the one
    name the provider stopped using.

    But that substring matches EVERY live local sandbox, not just stale ones — this suite is
    run under pytest-xdist, and the local backend packs every worker's containers onto the one
    Docker host. Removing the whole match set would tear a sibling worker's deployment out from
    under it mid-test. So only containers older than ``_STALE_CONTAINER_AGE_SECONDS`` are
    reaped; anything younger could still be in use and is left alone. Containers this run owns
    are removed by name in the fixture's teardown, which needs no age check.

    Unanchored on purpose: Docker's ``name`` filter matches against the name as stored,
    which carries a leading ``/``, so a ``^``-anchored pattern can match nothing and turn
    this prune into the same silent no-op it is replacing.
    """
    found = _docker("ps", "-aq", "--filter", "name=agent-local-")
    for container_id in found.stdout.split():
        age = _container_age_seconds(container_id)
        if age is not None and age > _STALE_CONTAINER_AGE_SECONDS:
            _docker("rm", "-f", container_id)


def _wait_ready(url: str, codes: tuple[int, ...], timeout: int = 45) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(url, timeout=2).status_code in codes:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture
def all_local(monkeypatch, tmp_path):
    reg_port = _free_port()
    reg_name = f"local-e2e-registry-{uuid.uuid4().hex[:8]}"
    start = _docker("run", "-d", "--rm", "--name", reg_name, "-p", f"127.0.0.1:{reg_port}:5000", "registry:2")
    if start.returncode != 0:
        pytest.skip(f"could not start registry:2: {start.stderr.strip()}")
    host = f"localhost:{reg_port}"
    if not _wait_ready(f"http://{host}/v2/", (200, 401)):
        _docker("rm", "-f", reg_name)
        pytest.skip("local registry did not become ready")

    _rm_stale_local_agent_containers()  # clear leftovers from a prior CRASHED run only
    owned: list[str] = []  # container names this run created; the test appends as it learns them
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_IMAGE_STORE", "local")
    configure()
    set_image_store(LocalRegistryImageStore(host))
    reset_artifact_store()
    try:
        yield host, owned
    finally:
        # Only this run's containers, by name — a broad `agent-local-` sweep here would
        # also remove a concurrent xdist worker's live sandbox.
        for container_name in owned:
            _docker("rm", "-f", container_name)
        reset_artifact_store()
        reset_config()
        _docker("rm", "-f", reg_name)


def _build_server_image(tag: str, port: int) -> None:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Dockerfile").write_text(
            f'FROM python:3.12-slim\nCMD ["python", "-m", "http.server", "{port}"]\n'
        )
        result = _docker("build", "-t", tag, str(d))
        if result.returncode != 0:
            raise RuntimeError(f"docker build failed: {result.stderr}")


@pytest.mark.asyncio
async def test_task_lifecycle_all_local(all_local):
    host, owned = all_local
    app_port = _free_port()
    artifact_id = f"local-e2e-{uuid.uuid4().hex[:8]}"
    local_tag = f"{artifact_id}-src"
    _build_server_image(local_tag, app_port)
    # No try/finally here: the container this deploy creates is removed by the fixture's
    # teardown, which owns `owned` and runs even when this body raises before the deploy
    # reports a sandbox id.
    artifact = DockerImageArtifact.put(id=artifact_id, description="local e2e", image_name=local_tag)
    assert artifact.image_name == f"{host}/{artifact_id}:v1"  # pushed to the local registry

    task = Task.put(
        id=f"task-{artifact_id}",
        steps=[
            DeploySandboxTaskStep(
                id="deploy",
                version=None,
                sandbox_name="min",
                sandbox_mode="container",
                sandbox_type="local",
                image=artifact.image_name,
                port=app_port,
                ttl_seconds=600,
            )
        ],
    )
    ctx = await task.run()

    assert not ctx.metadata.get("failed_steps"), ctx.metadata.get("failed_steps")
    assert len(ctx.deployed_sandboxes) == 1
    deployed = ctx.deployed_sandboxes[0]
    assert deployed.sandbox_mode == "container"
    # Register for teardown as soon as the name is knowable, so a later assertion
    # failure in this test still removes the container the deploy created.
    owned.append(f"agent-{deployed.sandbox_id}")

    # The provider publishes the container port on a free host port, so the mapping
    # is not identity — take the URL the deploy reported. Keyed by CONTAINER port
    # (deploy_sandbox stringifies the keys), which is what the caller asked for.
    served_url = deployed.tunnel_urls[str(app_port)]
    assert re.fullmatch(r"http://127\.0\.0\.1:\d+", served_url), served_url

    # the pulled image is actually running and serving on the published port
    assert _wait_ready(f"{served_url}/", (200,)), \
        f"deployed container never served at {served_url}"

    # task-lifecycle docs persisted to the local SQLite store
    assert Task.get(task.id, task.version) is not None
    assert ctx.instance_id and Task.get_instance(ctx.instance_id) is not None
