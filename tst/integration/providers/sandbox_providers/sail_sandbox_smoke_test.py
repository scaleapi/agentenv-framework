"""Sail Sailbox smoke test.

Runs when the resolved config can build the ``sail`` sandbox provider (``[sandbox.providers.sail.config]``
with ``api_key = "secret:sail_api_key"``), the ``remote_sandbox`` capability, and skips with the declared
reason otherwise. The test never reads or prints the API key.
"""

from __future__ import annotations

import hashlib

import httpx
import pytest
import pytest_asyncio

from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider
from tst.util.capabilities import skip_without_remote_sandbox

_PORT = 8080

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio,
    skip_without_remote_sandbox("sail"),
]


@pytest.fixture(scope="module")
def sail_provider():
    return build_sandbox_provider("sail")


@pytest_asyncio.fixture(scope="module")
async def sail_sandbox(sail_provider):
    sandbox = await sail_provider.create_vm(cpu=1.0, memory=2048, exposed_ports=[_PORT], timeout=900)
    try:
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_sailbox_runs_docker_serves_its_port_and_reconnects(sail_provider, sail_sandbox):
    assert (sail_sandbox.type, sail_sandbox.mode) == ("sail", "vm")

    await sail_sandbox.exec_script(
        f"docker run -d --name web -p {_PORT}:80 public.ecr.aws/nginx/nginx:alpine > /dev/null"
    )
    url = sail_sandbox.tunnel_urls[_PORT]
    assert url.startswith("https://") and url.endswith(".sail.box")
    async with httpx.AsyncClient(timeout=30) as client:
        for _ in range(30):
            response = await client.get(url)
            if response.status_code == 200:
                break
    assert response.status_code == 200
    assert "nginx" in response.text

    data = bytes(range(256)) * 4096
    await sail_sandbox.write_host_file(data, "/tmp/agent-env-smoke/blob.bin")
    exit_code, digest, stderr = await sail_sandbox.exec_with_output("sha256sum", "/tmp/agent-env-smoke/blob.bin")
    assert exit_code == 0, stderr
    assert digest.split()[0] == hashlib.sha256(data).hexdigest()

    reconnected = await sail_provider.get_sandbox(sail_sandbox.sandbox_id)
    assert reconnected.sandbox_id == sail_sandbox.sandbox_id
    assert reconnected.tunnel_urls == sail_sandbox.tunnel_urls
    assert reconnected.network_policy == NetworkPolicy()


async def test_an_allowlist_is_enforced_for_containers_in_the_sailbox(sail_provider):
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",))
    sandbox = await sail_provider.create_vm(cpu=1.0, memory=2048, exposed_ports=[], timeout=600, network_policy=policy)
    try:
        assert "*.sail.box" in sandbox.network_policy.allow_hosts
        exit_code, out, stderr = await sandbox.exec_with_output(
            "bash", "-c",
            "curl -s -m 10 -o /dev/null -w '%{http_code}' https://pypi.org/simple/; echo; "
            "curl -s -m 10 -o /dev/null https://example.com && echo example-reachable || echo example-blocked",
        )
        assert exit_code == 0, stderr
        assert out.split() == ["200", "example-blocked"]
        reconnected = await sail_provider.get_sandbox(sandbox.sandbox_id)
        assert reconnected.network_policy == sandbox.network_policy
    finally:
        await sandbox.terminate()
