"""All three local store backends compose with no external infrastructure.

Drives the real ``DockerImageArtifact`` lifecycle (build -> put -> pull -> get ->
load) against local SQLite docs + filesystem objects + a local OCI registry —
the joint no-infra proof for images + objects + docs.

Requires a docker daemon; spins up a throwaway ``registry:2`` and skips if it
can't start.
"""

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

pytestmark = pytest.mark.integration


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _docker(*args):
    return subprocess.run(["docker", *args], capture_output=True, text=True)


@pytest.fixture
def all_local(monkeypatch, tmp_path):
    port = _free_port()
    name = f"local-composition-registry-{uuid.uuid4().hex[:8]}"
    start = _docker("run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:5000", "registry:2")
    if start.returncode != 0:
        pytest.skip(f"could not start registry:2: {start.stderr.strip()}")
    host = f"localhost:{port}"
    _wait_ready(host)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_IMAGE_STORE", "local")
    configure()
    set_image_store(LocalRegistryImageStore(host))
    reset_artifact_store()
    try:
        yield host, tmp_path
    finally:
        reset_artifact_store()
        reset_config()
        _docker("rm", "-f", name)


def _wait_ready(host: str, timeout: int = 30) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if httpx.get(f"http://{host}/v2/", timeout=2).status_code in (200, 401):
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"local registry at {host} not ready")


def _build_scratch_image(tag: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "payload.txt").write_text(uuid.uuid4().hex)
        (root / "Dockerfile").write_text("FROM scratch\nCOPY payload.txt /payload.txt\n")
        result = _docker("build", "-t", tag, str(root))
        if result.returncode != 0:
            raise RuntimeError(f"docker build failed: {result.stderr}")


def test_docker_image_lifecycle_all_local(all_local):
    host, root = all_local
    artifact_id = f"local-img-{uuid.uuid4().hex[:8]}"
    local_tag = f"{artifact_id}-src"
    _build_scratch_image(local_tag)
    try:
        artifact = DockerImageArtifact.put(id=artifact_id, description="local composition", image_name=local_tag)

        # image ref points at the local registry (image store)
        assert artifact.image_name == f"{host}/{artifact_id}:v1"
        # tar.gz landed in the local filesystem object store
        assert artifact.tar_gz_object_url.startswith("file://")
        assert Path(artifact.tar_gz_object_url[len("file://"):]).exists()
        assert (root / ".agentenv" / "object_store").exists()
        assert (root / ".agentenv" / "document_store" / "documents.db").exists()

        # the image is really pullable from the local registry
        _docker("rmi", "-f", artifact.image_name)
        pulled = _docker("pull", artifact.image_name)
        assert pulled.returncode == 0, pulled.stderr

        # doc store: the artifact reloads from local SQLite
        reloaded = DockerImageArtifact.get(artifact_id)
        assert reloaded is not None
        assert reloaded.version == 1
        assert reloaded.image_name == artifact.image_name

        # object store: load() streams the tar.gz back (gzip magic bytes)
        assert artifact.load()[:2] == b"\x1f\x8b"
    finally:
        _docker("rmi", "-f", local_tag, f"{host}/{artifact_id}:v1")
