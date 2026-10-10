"""Tensorlake VM smoke test.

Runs when the resolved config can build the ``tensorlake`` sandbox provider (the ``tensorlake`` extra
and ``[sandbox.providers.tensorlake]`` with ``api_key = "secret:tensorlake_api_key"``) — the
``remote_sandbox`` capability — and skips with the declared reason otherwise. The test never reads or
prints the API key.
"""

from __future__ import annotations

import asyncio
import hashlib
import os

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
    skip_without_remote_sandbox("tensorlake"),
]


@pytest.fixture(scope="module")
def tensorlake_provider():
    """Build the configured provider without exposing its resolved API key."""
    return build_sandbox_provider("tensorlake")


@pytest_asyncio.fixture(scope="module")
async def tensorlake_sandbox(tensorlake_provider):
    """Provision one Tensorlake VM and always terminate it after the smoke test."""
    sandbox = await tensorlake_provider.create_vm(exposed_ports=[_PORT], timeout=900)
    try:
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_tensorlake_vm_docker_ports_upload_and_reconnect(tensorlake_provider, tensorlake_sandbox):
    """Exercise the host image, nested Docker, a public port end to end, a multi-part upload, and reconnect."""
    assert tensorlake_sandbox.type == "tensorlake"
    assert tensorlake_sandbox.mode == "vm"
    assert tensorlake_sandbox.network_policy == NetworkPolicy()

    exit_code, version, stderr = await tensorlake_sandbox.exec_with_output(
        "sudo", "docker", "info", "--format", "{{.ServerVersion}}"
    )
    assert exit_code == 0, stderr
    assert version.strip()

    await tensorlake_sandbox.exec_script(
        f"docker run -d --name smoke -p {_PORT}:{_PORT} public.ecr.aws/docker/library/python:3.12-slim "
        f"python -m http.server {_PORT}"
    )
    url = tensorlake_sandbox.tunnel_urls[_PORT]
    assert url.startswith("https://")
    async with httpx.AsyncClient(timeout=10) as client:
        for _ in range(30):
            try:
                if (await client.get(url)).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
        else:
            pytest.fail(f"{url} never answered 200")

    payload = os.urandom(1024 * 1024 + 17)
    await tensorlake_sandbox.write_host_file(payload, "/opt/smoke/payload.bin")
    digest = await tensorlake_sandbox.exec_script("sha256sum /opt/smoke/payload.bin")
    assert digest.split()[0] == hashlib.sha256(payload).hexdigest()

    reconnected = await build_sandbox_provider("tensorlake").get_sandbox(tensorlake_sandbox.sandbox_id)
    assert reconnected.sandbox_id == tensorlake_sandbox.sandbox_id
    assert reconnected.tunnel_urls == tensorlake_sandbox.tunnel_urls
    assert reconnected.network_policy == NetworkPolicy()


async def test_tensorlake_allowlist_survives_reconnect_and_terminate_is_idempotent(tensorlake_provider):
    """A fresh provider reads back the applied allowlist, and a second terminate is not an error."""
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",), allow_cidrs=("1.1.1.1/32",))
    sandbox = await tensorlake_provider.create_vm(timeout=600, setup_for_gateway=False, network_policy=policy)
    try:
        reconnected = await build_sandbox_provider("tensorlake").get_sandbox(sandbox.sandbox_id)
        assert reconnected.network_policy == sandbox.network_policy
    finally:
        await sandbox.terminate()
    await reconnected.terminate()
