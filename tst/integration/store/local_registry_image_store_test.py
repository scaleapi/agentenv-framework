"""LocalRegistryImageStore runs the ImageStore conformance suite against a
throwaway ``registry:2`` container — the local parity run for the ECR suite.

Requires a docker daemon; skips if the registry image can't be started.
"""

import socket
import subprocess
import time
import uuid

import httpx
import pytest

from agent_env.store import LocalRegistryImageStore
from tst.store import image_conformance

pytestmark = pytest.mark.integration


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def registry_host():
    port = _free_port()
    name = f"imagestore-conf-registry-{uuid.uuid4().hex[:8]}"
    start = subprocess.run(
        ["docker", "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1:{port}:5000", "registry:2"],
        capture_output=True, text=True,
    )
    if start.returncode != 0:
        pytest.skip(f"could not start local registry:2: {start.stderr.strip()}")
    host = f"localhost:{port}"
    try:
        _wait_ready(host)
        yield host
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _wait_ready(host: str, timeout: int = 30) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            if httpx.get(f"http://{host}/v2/", timeout=2).status_code in (200, 401):
                return
        except Exception as e:
            last = e
        time.sleep(0.5)
    raise RuntimeError(f"registry at {host} not ready: {last}")


@pytest.fixture
def store_repo(registry_host):
    return LocalRegistryImageStore(registry_host), f"conf-{uuid.uuid4().hex[:12]}"


@pytest.mark.parametrize("case", image_conformance.CASES, ids=lambda c: c.__name__)
def test_conformance(case, store_repo):
    case(*store_repo)


def test_auth_is_none(store_repo):
    """A local anonymous registry needs no docker-login."""
    store, repository = store_repo
    assert store.auth(store.image_ref(repository, "v1")) is None
