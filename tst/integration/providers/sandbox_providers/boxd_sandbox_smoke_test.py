"""boxd machine smoke test.

Runs when the resolved config can build the ``boxd`` sandbox provider (the ``boxd`` extra and
``[sandbox.providers.boxd.config]`` with ``api_key = "secret:boxd_api_key"``), the ``remote_sandbox``
capability, and skips with the declared reason otherwise. The test never reads or prints the API key.
"""

from __future__ import annotations

import asyncio
import hashlib
import os

import httpx
import pytest
import pytest_asyncio

from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy, read_vm_file
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxNotFoundError, build_sandbox_provider
from tst.util.capabilities import skip_without_remote_sandbox

_PORT = 8080

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio,
    skip_without_remote_sandbox("boxd"),
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
def boxd_provider():
    return build_sandbox_provider("boxd")


@pytest_asyncio.fixture(scope="module")
async def boxd_sandbox(boxd_provider):
    """One boxd machine serving on ``_PORT`` from a container, always deleted after the module."""
    sandbox = await boxd_provider.create_vm(cpu=1, memory=4096, exposed_ports=[_PORT], timeout=900)
    try:
        await sandbox.exec_script(
            f"docker run -d --name smoke -p {_PORT}:{_PORT} public.ecr.aws/docker/library/python:3.12-slim "
            f"python -m http.server {_PORT}"
        )
        await _wait_for_200(sandbox.tunnel_urls[_PORT])
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_boxd_machine_docker_ports_files_and_reconnect(boxd_provider, boxd_sandbox, tmp_path):
    """Nested Docker, a public port end to end, a write to a root-owned path, a multi-range read, and reconnect."""
    assert (boxd_sandbox.type, boxd_sandbox.mode) == ("boxd", "vm")
    assert boxd_sandbox.network_policy == NetworkPolicy()

    exit_code, version, stderr = await boxd_sandbox.exec_with_output(
        "sudo", "docker", "info", "--format", "{{.ServerVersion}}"
    )
    assert exit_code == 0, stderr
    assert version.strip()

    payload = os.urandom(1024 * 1024 + 17)
    await boxd_sandbox.write_host_file(payload, "/opt/smoke/payload.bin")
    digest = await boxd_sandbox.exec_script("sha256sum /opt/smoke/payload.bin")
    assert digest.split()[0] == hashlib.sha256(payload).hexdigest()

    await boxd_sandbox.exec_script("head -c 5242887 /dev/urandom > /opt/smoke/large.bin")
    remote = (await boxd_sandbox.exec_script("sha256sum /opt/smoke/large.bin")).split()[0]
    await read_vm_file(boxd_sandbox, "/opt/smoke/large.bin", str(tmp_path / "large.bin"))
    assert hashlib.sha256((tmp_path / "large.bin").read_bytes()).hexdigest() == remote

    reconnected = await build_sandbox_provider("boxd").get_sandbox(boxd_sandbox.sandbox_id)
    assert reconnected.tunnel_urls == boxd_sandbox.tunnel_urls
    assert reconnected.network_policy == NetworkPolicy()
    assert (await reconnected.exec_with_output("true"))[0] == 0


async def test_boxd_concurrent_commands_that_read_input_end(boxd_sandbox):
    """Several of one connection's execs that read standard input at once each see its end and exit."""
    results = await asyncio.wait_for(
        asyncio.gather(*(boxd_sandbox.exec_with_output("bash", "-c", f"cat; echo {i}") for i in range(6))), 60,
    )
    assert [(code, out.strip()) for code, out, _ in results] == [(0, str(i)) for i in range(6)]


async def test_boxd_allowlisted_machine_reaches_its_environment_and_nothing_else(boxd_provider, boxd_sandbox):
    """An allowlisted machine reaches an allowed host and another boxd machine's public port, and nothing else."""
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("example.com",))
    agent = await boxd_provider.create_vm(cpu=1, memory=4096, timeout=600, setup_for_gateway=False,
                                          network_policy=policy)
    try:
        async def fetch(url: str) -> int:
            return (await agent.exec_with_output("curl", "-fsS", "-o", "/dev/null", "--max-time", "15", url))[0]

        assert await fetch("https://example.com") == 0
        assert await fetch(boxd_sandbox.tunnel_urls[_PORT]) == 0
        assert await fetch("https://www.wikipedia.org") != 0

        reconnected = await build_sandbox_provider("boxd").get_sandbox(agent.sandbox_id)
        assert reconnected.network_policy.restricts_egress
        assert set(reconnected.network_policy.allow_hosts) == set(agent.network_policy.allow_hosts)
    finally:
        await agent.terminate()
    await reconnected.terminate()
    with pytest.raises(SandboxNotFoundError):
        await boxd_provider.get_sandbox(agent.sandbox_id)
