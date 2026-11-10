"""Agent37 VM smoke test.

Runs when the resolved config can build the ``agent37`` sandbox provider (``[sandbox.providers.agent37.config]`` with
``api_key = "secret:agent37_api_key"``), the ``remote_sandbox`` capability, and skips with the declared reason
otherwise. The test never reads or prints the API key.
"""

from __future__ import annotations

import asyncio
import hashlib
import os

import httpx
import pytest
import pytest_asyncio

from agent_env.providers.sandbox_providers.sandbox import NetworkPolicy, read_vm_file
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxNotFoundError, build_sandbox_provider
from tst.util.capabilities import skip_without_remote_sandbox

_PORT = 8080

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio,
    skip_without_remote_sandbox("agent37"),
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
def agent37_provider():
    return build_sandbox_provider("agent37")


@pytest_asyncio.fixture(scope="module")
async def agent37_sandbox(agent37_provider):
    """One Agent37 VM serving on ``_PORT`` from a container, always deleted after the module."""
    sandbox = await agent37_provider.create_vm(cpu=1, memory=4096, exposed_ports=[_PORT], timeout=900)
    try:
        await sandbox.exec_script(
            f"docker run -d --name smoke -p {_PORT}:{_PORT} public.ecr.aws/docker/library/python:3.12-slim "
            f"python -m http.server {_PORT}"
        )
        await _wait_for_200(sandbox.tunnel_urls[_PORT])
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_agent37_vm_docker_ports_files_and_reconnect(agent37_sandbox, tmp_path):
    """Nested Docker, a public port end to end, a write to a root-owned path, a read back, and reconnect."""
    assert (agent37_sandbox.type, agent37_sandbox.mode) == ("agent37", "vm")
    assert agent37_sandbox.network_policy == NetworkPolicy()

    exit_code, version, stderr = await agent37_sandbox.exec_with_output(
        "sudo", "docker", "info", "--format", "{{.ServerVersion}}"
    )
    assert exit_code == 0, stderr
    assert version.strip()

    payload = os.urandom(1024 * 1024 + 17)
    await agent37_sandbox.write_host_file(payload, "/opt/smoke/payload.bin")
    digest = await agent37_sandbox.exec_script("sha256sum /opt/smoke/payload.bin")
    assert digest.split()[0] == hashlib.sha256(payload).hexdigest()

    # One read range; exec output is capped at 512 KB, so a file has to fit in one as base64.
    await agent37_sandbox.exec_script("head -c 300007 /dev/urandom > /opt/smoke/small.bin")
    remote = (await agent37_sandbox.exec_script("sha256sum /opt/smoke/small.bin")).split()[0]
    await read_vm_file(agent37_sandbox, "/opt/smoke/small.bin", str(tmp_path / "small.bin"))
    assert hashlib.sha256((tmp_path / "small.bin").read_bytes()).hexdigest() == remote

    reconnected = await build_sandbox_provider("agent37").get_sandbox(agent37_sandbox.sandbox_id)
    assert reconnected.tunnel_urls == agent37_sandbox.tunnel_urls
    assert reconnected.network_policy == NetworkPolicy()
    assert (await reconnected.exec_with_output("true"))[0] == 0


async def test_agent37_concurrent_commands_each_get_their_own_answer(agent37_sandbox):
    """Several execs at once each return their own output, and a command that reads input sees its end."""
    results = await asyncio.wait_for(
        asyncio.gather(*(agent37_sandbox.exec_with_output("bash", "-c", f"cat; echo {i}") for i in range(12))), 60,
    )
    assert [(code, out.strip()) for code, out, _ in results] == [(0, str(i)) for i in range(12)]


async def test_agent37_terminate_is_idempotent(agent37_provider):
    sandbox = await agent37_provider.create_vm(cpu=1, memory=4096, timeout=600, setup_for_gateway=False)
    await sandbox.terminate()
    await sandbox.terminate()
    with pytest.raises(SandboxNotFoundError):
        await agent37_provider.get_sandbox(sandbox.sandbox_id)
