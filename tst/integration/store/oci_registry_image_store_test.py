"""Authenticated OCI registry integration tests for the generic image store.

Runs a throwaway ``registry:2`` with htpasswd authentication and exercises the
real Docker login/push/pull path. Requires a Docker daemon.
"""

import json
import socket
import subprocess
import time
import uuid

import httpx
import pytest

from agent_env.config import get_config
from agent_env.store import (
    LocalSecretStore,
    OciRegistryImageStore,
    SecretStoreCredentials,
)
from tst.store import image_conformance

pytestmark = pytest.mark.integration

_USERNAME = "agent-env-test"
_PASSWORD = "registry-password"
# bcrypt for the fixed, test-only password above; registry:2 requires bcrypt.
_HTPASSWD = (
    "agent-env-test:$2y$05$HbES13aom46b6yn7VJFoLO/Jlf677mugJ7H911tXCawHiSl2eQWQi\n"
)
_SECRET_KEY = "registry_auths"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _docker(*args, env=None):
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        env=env,
    )


@pytest.fixture(scope="module")
def authenticated_registry_host(tmp_path_factory):
    port = _free_port()
    name = f"generic-auth-registry-{uuid.uuid4().hex[:8]}"
    auth_dir = tmp_path_factory.mktemp("registry-auth")
    (auth_dir / "htpasswd").write_text(_HTPASSWD)

    start = _docker(
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "-p",
        f"127.0.0.1:{port}:5000",
        "-v",
        f"{auth_dir}:/auth:ro",
        "-e",
        "REGISTRY_AUTH=htpasswd",
        "-e",
        "REGISTRY_AUTH_HTPASSWD_REALM=Registry Realm",
        "-e",
        "REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd",
        "registry:2",
    )
    if start.returncode != 0:
        pytest.fail(f"could not start authenticated registry:2: {start.stderr.strip()}")

    host = f"localhost:{port}"
    try:
        _wait_for_auth_challenge(host)
        yield host
    finally:
        _docker("rm", "-f", name)


def _wait_for_auth_challenge(host: str, timeout: int = 30) -> None:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"http://{host}/v2/", timeout=2)
            if response.status_code == 401:
                return
            last = f"unexpected HTTP {response.status_code}"
        except Exception as exc:
            last = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"authenticated registry at {host} not ready: {last}")


@pytest.fixture
def store_repo(authenticated_registry_host, monkeypatch, tmp_path):
    # Keep docker login state out of the user/worker home directory. A blank
    # DOCKER_CONFIG also loses the selected Docker context, so preserve its
    # resolved daemon endpoint explicitly.
    context = _docker("context", "inspect", "--format", "{{.Endpoints.docker.Host}}")
    if context.returncode != 0 or not context.stdout.strip():
        pytest.fail(f"could not resolve Docker context: {context.stderr.strip()}")
    monkeypatch.setenv("DOCKER_HOST", context.stdout.strip())
    docker_config = tmp_path / "docker-config"
    docker_config.mkdir()
    monkeypatch.setenv("DOCKER_CONFIG", str(docker_config))

    secret = json.dumps(
        {
            "auths": {
                authenticated_registry_host: {
                    "username": _USERNAME,
                    "password": _PASSWORD,
                }
            }
        }
    )
    get_config().set_secret_store(
        LocalSecretStore(values={_SECRET_KEY: secret}, use_env=False)
    )
    store = OciRegistryImageStore(
        authenticated_registry_host,
        credentials=SecretStoreCredentials(secret_key=_SECRET_KEY),
    )
    return store, f"conf-{uuid.uuid4().hex[:12]}"


@pytest.mark.parametrize(
    "case", image_conformance.CASES, ids=lambda case: case.__name__
)
def test_authenticated_registry_conformance(case, store_repo):
    case(*store_repo)
