"""kosa8 Cloud sandbox smoke test.

Runs when the resolved config can build the ``kosa8`` sandbox provider (``[sandbox.providers.kosa8.config]`` with
``api_key = "env:KOSA8_API_KEY"`` or a ``secret:`` reference), the ``remote_sandbox`` capability, and skips with the
declared reason otherwise. The test never reads or prints the API key.
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
    skip_without_remote_sandbox("kosa8"),
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
def kosa8_provider():
    return build_sandbox_provider("kosa8")


@pytest_asyncio.fixture(scope="module")
async def kosa8_sandbox(kosa8_provider):
    """One kosa8 sandbox serving on ``_PORT`` from a container, always deleted after the module."""
    sandbox = await kosa8_provider.create_vm(cpu=2, memory=4096, exposed_ports=[_PORT], timeout=900)
    try:
        await sandbox.exec_script(
            f"docker run -d --name smoke -p {_PORT}:{_PORT} public.ecr.aws/docker/library/python:3.12-slim "
            f"python -m http.server {_PORT}"
        )
        await _wait_for_200(sandbox.tunnel_urls[_PORT])
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_kosa8_sandbox_docker_ports_files_and_reconnect(kosa8_provider, kosa8_sandbox, tmp_path):
    """Nested Docker and Compose, a public port end to end, a file through the files API, a long command, a
    multi-range read, and reconnect."""
    assert (kosa8_sandbox.type, kosa8_sandbox.mode) == ("kosa8", "vm")

    exit_code, version, stderr = await kosa8_sandbox.exec_with_output("sudo", "docker", "compose", "version")
    assert exit_code == 0, stderr
    assert version.strip()

    payload = os.urandom(5 * 1024 * 1024 + 17)
    await kosa8_sandbox.write_host_file(payload, "/opt/smoke/payload.bin")
    digest = await kosa8_sandbox.exec_script("sha256sum /opt/smoke/payload.bin")
    assert digest.split()[0] == hashlib.sha256(payload).hexdigest()

    long_script = "printf '%s' " + "y" * 20000 + " | wc -c"
    assert (await kosa8_sandbox.exec_script(long_script)).strip() == "20000"
    assert (await kosa8_sandbox.exec_script("sleep 100 && echo slept")).strip() == "slept"

    await kosa8_sandbox.exec_script("head -c 5242887 /dev/urandom > /opt/smoke/large.bin")
    remote = (await kosa8_sandbox.exec_script("sha256sum /opt/smoke/large.bin")).split()[0]
    await read_vm_file(kosa8_sandbox, "/opt/smoke/large.bin", str(tmp_path / "large.bin"))
    assert hashlib.sha256((tmp_path / "large.bin").read_bytes()).hexdigest() == remote

    reconnected = await build_sandbox_provider("kosa8").get_sandbox(kosa8_sandbox.sandbox_id)
    assert reconnected.tunnel_urls == kosa8_sandbox.tunnel_urls
    assert reconnected.network_policy == kosa8_sandbox.network_policy
    assert (await reconnected.exec_with_output("true"))[0] == 0


async def test_kosa8_concurrent_commands_that_read_input_end(kosa8_sandbox):
    """Several execs that read standard input at once each see its end and exit."""
    results = await asyncio.wait_for(
        asyncio.gather(*(kosa8_sandbox.exec_with_output("bash", "-c", f"cat; echo {i}") for i in range(6))), 60,
    )
    assert [(code, out.strip()) for code, out, _ in results] == [(0, str(i)) for i in range(6)]


async def test_kosa8_allowlisted_sandbox_reaches_its_environment_and_nothing_else(kosa8_provider, kosa8_sandbox):
    """An allowlisted sandbox reaches an allowed host and another kosa8 sandbox's public port, and nothing else."""
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("example.com",))
    agent = await kosa8_provider.create_vm(cpu=2, memory=4096, timeout=600, setup_for_gateway=False,
                                           network_policy=policy)
    try:
        async def fetch(url: str) -> int:
            return (await agent.exec_with_output("curl", "-fsS", "-o", "/dev/null", "--max-time", "15", url))[0]

        assert await fetch("https://example.com") == 0
        assert await fetch(kosa8_sandbox.tunnel_urls[_PORT]) == 0
        assert await fetch("https://www.wikipedia.org") != 0

        reconnected = await build_sandbox_provider("kosa8").get_sandbox(agent.sandbox_id)
        assert reconnected.network_policy.restricts_egress
        assert set(reconnected.network_policy.allow_hosts) == set(agent.network_policy.allow_hosts)
    finally:
        await agent.terminate()
    await reconnected.terminate()
    with pytest.raises(SandboxNotFoundError):
        await kosa8_provider.get_sandbox(agent.sandbox_id)
