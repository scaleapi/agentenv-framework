"""Tenki VM smoke test.

Runs when the resolved config can build the ``tenki`` sandbox provider (the ``tenki`` extra and
``[sandbox.providers.tenki.config]`` with ``api_key = "secret:tenki_api_key"``), the ``remote_sandbox``
capability, and skips with the declared reason otherwise. The test never reads or prints the API key.
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
    skip_without_remote_sandbox("tenki"),
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


@pytest.fixture(scope="module")
def tenki_provider():
    return build_sandbox_provider("tenki")


@pytest_asyncio.fixture(scope="module")
async def tenki_sandbox(tenki_provider):
    """One Tenki VM serving on ``_PORT`` from a container, always terminated after the module."""
    sandbox = await tenki_provider.create_vm(cpu=1, memory=2048, exposed_ports=[_PORT], timeout=900)
    try:
        await sandbox.exec_script(
            f"docker run -d --name smoke -p {_PORT}:{_PORT} public.ecr.aws/docker/library/python:3.12-slim "
            f"python -m http.server {_PORT}"
        )
        await _wait_for_200(sandbox.tunnel_urls[_PORT])
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_tenki_vm_docker_ports_upload_and_reconnect(tenki_provider, tenki_sandbox):
    """Nested Docker, a public port end to end, an upload over stdin to a root-owned path, and reconnect."""
    assert (tenki_sandbox.type, tenki_sandbox.mode) == ("tenki", "vm")
    assert tenki_sandbox.network_policy == NetworkPolicy()

    exit_code, version, stderr = await tenki_sandbox.exec_with_output(
        "sudo", "docker", "info", "--format", "{{.ServerVersion}}"
    )
    assert exit_code == 0, stderr
    assert version.strip()

    payload = os.urandom(1024 * 1024 + 17)
    await tenki_sandbox.write_host_file(payload, "/opt/smoke/payload.bin")
    digest = await tenki_sandbox.exec_script("sha256sum /opt/smoke/payload.bin")
    assert digest.split()[0] == hashlib.sha256(payload).hexdigest()

    reconnected = await build_sandbox_provider("tenki").get_sandbox(tenki_sandbox.sandbox_id)
    assert reconnected.tunnel_urls == tenki_sandbox.tunnel_urls
    assert reconnected.network_policy == NetworkPolicy()
    assert (await reconnected.exec_with_output("true"))[0] == 0


async def test_tenki_allowlisted_vm_reaches_its_environment_and_nothing_else(tenki_provider, tenki_sandbox):
    """An allowlisted VM reaches an allowed host and another Tenki VM's public port, and nothing else."""
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("example.com",))
    agent = await tenki_provider.create_vm(cpu=1, memory=1024, timeout=600, setup_for_gateway=False,
                                           network_policy=policy)
    try:
        async def fetch(url: str) -> int:
            return (await agent.exec_with_output("curl", "-fsS", "-o", "/dev/null", "--max-time", "15", url))[0]

        assert await fetch("https://example.com") == 0
        assert await fetch(tenki_sandbox.tunnel_urls[_PORT]) == 0
        assert await fetch("https://www.wikipedia.org") != 0

        reconnected = await build_sandbox_provider("tenki").get_sandbox(agent.sandbox_id)
        assert reconnected.network_policy.restricts_egress
        assert set(reconnected.network_policy.allow_hosts) == set(agent.network_policy.allow_hosts)
    finally:
        await agent.terminate()
    await reconnected.terminate()
