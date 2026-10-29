"""Daytona sandbox smoke test.

Runs when the resolved config can build the ``daytona`` sandbox provider (the ``daytona`` extra and
``[sandbox.providers.daytona.config]`` with ``api_key = "secret:daytona_api_key"``), the ``remote_sandbox``
capability, and skips with the declared reason otherwise. The test never reads or prints the API key.
"""

from __future__ import annotations

import asyncio
import hashlib
import os

import httpx
import pytest
import pytest_asyncio

from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxNotFoundError, build_sandbox_provider
from tst.util.capabilities import skip_without_remote_sandbox

_PORT = 8080

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio,
    skip_without_remote_sandbox("daytona"),
]


async def _wait_for_200(url: str) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        for _ in range(60):
            try:
                if (await client.get(url)).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2)
    pytest.fail(f"{url} never answered 200")


@pytest_asyncio.fixture(scope="module")
async def daytona_provider():
    provider = build_sandbox_provider("daytona")
    yield provider
    await provider.close()


@pytest_asyncio.fixture(scope="module")
async def daytona_sandbox(daytona_provider):
    """One Daytona sandbox serving on ``_PORT`` from a container, always deleted after the module."""
    sandbox = await daytona_provider.create_vm(cpu=2, memory=4096, exposed_ports=[_PORT], timeout=1800)
    try:
        # Docker Hub, which every Daytona tier reaches; lower tiers block public ECR's layer CDN.
        await sandbox.exec_script(
            f"docker run -d --name smoke -p {_PORT}:{_PORT} python:3.12-slim python -m http.server {_PORT}"
        )
        await _wait_for_200(sandbox.tunnel_urls[_PORT])
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_daytona_sandbox_docker_ports_upload_and_reconnect(daytona_sandbox):
    """Nested Docker, a public port end to end, a multi-part upload to a root-owned path, and a reconnect from a
    fresh provider."""
    assert (daytona_sandbox.type, daytona_sandbox.mode) == ("daytona", "vm")
    assert daytona_sandbox.network_policy == NetworkPolicy()

    exit_code, version, stderr = await daytona_sandbox.exec_with_output(
        "sudo", "docker", "info", "--format", "{{.ServerVersion}}"
    )
    assert exit_code == 0, stderr
    assert version.strip()

    payload = os.urandom(9 * 1024 * 1024 + 17)
    await daytona_sandbox.write_host_file(payload, "/opt/smoke/payload.bin")
    digest = await daytona_sandbox.exec_script("sha256sum /opt/smoke/payload.bin")
    assert digest.split()[0] == hashlib.sha256(payload).hexdigest()

    fresh = build_sandbox_provider("daytona")
    try:
        reconnected = await fresh.get_sandbox(daytona_sandbox.sandbox_id)
        assert reconnected.tunnel_urls == daytona_sandbox.tunnel_urls
        assert reconnected.network_policy == daytona_sandbox.network_policy
        assert (await reconnected.exec_with_output("true"))[0] == 0
    finally:
        await fresh.close()


async def test_daytona_terminate_twice_then_the_sandbox_is_not_found(daytona_provider):
    """A second terminate, and one by id, of a deleted sandbox succeed; once Daytona finishes deleting it,
    ``get_sandbox`` reports it not found."""
    sandbox = await daytona_provider.create_vm(cpu=2, memory=4096, timeout=600, setup_for_gateway=False)
    await sandbox.terminate()
    await sandbox.terminate()
    await daytona_provider.terminate_sandbox(sandbox.sandbox_id)

    for _ in range(30):
        try:
            await daytona_provider.get_sandbox(sandbox.sandbox_id)
        except SandboxNotFoundError:
            return
        await asyncio.sleep(2)
    pytest.fail(f"Daytona sandbox {sandbox.sandbox_id} was still found a minute after terminate")
