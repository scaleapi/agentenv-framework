"""Vercel Sandbox remote smoke test.

Runs when the resolved config can build the ``vercel`` sandbox provider, the
``remote_sandbox`` capability, and skips with the declared reason otherwise. The
test never reads or prints resolved credentials.
"""

from __future__ import annotations

import asyncio
import hashlib

import httpx
import pytest
import pytest_asyncio

from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import build_sandbox_provider
from tst.util.capabilities import skip_without_remote_sandbox

_PORT = 8080
_COMPOSE = f"""services:
  web:
    image: python:3.12-alpine
    container_name: agentenv-vercel-smoke
    command: ["python", "-m", "http.server", "{_PORT}", "--bind", "0.0.0.0"]
    ports: ["{_PORT}:{_PORT}"]
"""

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio(loop_scope="module"),
    skip_without_remote_sandbox("vercel"),
]


@pytest.fixture(scope="module")
def vercel_provider():
    return build_sandbox_provider("vercel")


@pytest_asyncio.fixture(scope="module")
async def vercel_sandbox(vercel_provider):
    sandbox = None
    try:
        sandbox = await vercel_provider.create_vm(
            cpu=1,
            memory=2048,
            exposed_ports=[_PORT],
            timeout=900,
        )
        yield sandbox
    finally:
        if sandbox is not None:
            try:
                await sandbox.terminate()
            finally:
                await vercel_provider.close()
        else:
            await vercel_provider.close()


async def test_vercel_runs_docker_serves_its_port_and_reconnects(vercel_provider, vercel_sandbox):
    assert (vercel_sandbox.type, vercel_sandbox.mode) == ("vercel", "vm")

    await vercel_sandbox.write_host_file(_COMPOSE.encode(), "/tmp/agentenv-vercel-smoke/compose.yaml")
    await vercel_sandbox.exec_script("docker compose -f /tmp/agentenv-vercel-smoke/compose.yaml up -d")
    url = vercel_sandbox.tunnel_urls[_PORT]
    assert url.startswith("https://") and ".vercel.run" in url
    async with httpx.AsyncClient(timeout=30) as client:
        response: httpx.Response | None = None
        for _ in range(30):
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    break
            except httpx.TransportError:
                pass
            await asyncio.sleep(1)
    assert response is not None and response.status_code == 200

    data = bytes(range(256)) * 4096
    await vercel_sandbox.write_host_file(data, "/tmp/agentenv-vercel-smoke/blob.bin")
    exit_code, digest, stderr = await vercel_sandbox.exec_with_output(
        "sha256sum", "/tmp/agentenv-vercel-smoke/blob.bin"
    )
    assert exit_code == 0, stderr
    assert digest.split()[0] == hashlib.sha256(data).hexdigest()

    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("pypi.org",))
    await vercel_sandbox.apply_network_policy(policy)
    probe = (
        "import urllib.request\n"
        "print(urllib.request.urlopen('https://pypi.org/simple/', timeout=20).status)\n"
        "try:\n"
        "    urllib.request.urlopen('https://example.com', timeout=10)\n"
        "except Exception:\n"
        "    print('example-blocked')\n"
        "else:\n"
        "    raise SystemExit('unexpected access')"
    )
    for command in (
        ("python3", "-c", probe),
        ("sudo", "docker", "exec", "agentenv-vercel-smoke", "python", "-c", probe),
    ):
        exit_code, output, stderr = await vercel_sandbox.exec_with_output(*command)
        assert exit_code == 0, stderr
        assert output.split() == ["200", "example-blocked"]

    fresh_provider = build_sandbox_provider("vercel")
    try:
        fresh_sandbox = await fresh_provider.get_sandbox(vercel_sandbox.sandbox_id)
        assert fresh_sandbox.sandbox_id == vercel_sandbox.sandbox_id
        assert fresh_sandbox.tunnel_urls == vercel_sandbox.tunnel_urls
        assert fresh_sandbox.network_policy == vercel_sandbox.network_policy
        assert fresh_sandbox.network_policy == policy
        await fresh_sandbox.terminate()
        await fresh_sandbox.terminate()
    finally:
        await fresh_provider.close()
