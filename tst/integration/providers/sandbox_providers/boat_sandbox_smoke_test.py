"""boat sandbox smoke test.

Runs when the resolved config can build the ``boat`` sandbox provider (the ``boat`` extra and
``[sandbox.providers.boat.config]`` with ``api_key = "secret:boat_api_key"``), the ``remote_sandbox`` capability, and
skips with the declared reason otherwise. The test never reads or prints the API key.
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
    skip_without_remote_sandbox("boat"),
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
async def boat_provider():
    provider = build_sandbox_provider("boat")
    yield provider
    await provider.close()


@pytest_asyncio.fixture(scope="module")
async def boat_sandbox(boat_provider):
    """One boat sandbox serving on ``_PORT`` from a container, always deleted after the module."""
    sandbox = await boat_provider.create_vm(cpu=2, memory=4096, exposed_ports=[_PORT], timeout=1800)
    try:
        await sandbox.exec_script(
            f"docker run -d --name smoke -p {_PORT}:80 public.ecr.aws/docker/library/nginx:alpine"
        )
        await _wait_for_200(sandbox.tunnel_urls[_PORT])
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_boat_sandbox_docker_ports_upload_and_reconnect(boat_sandbox):
    """Nested Docker, a hosted port end to end, an upload to a root-owned path, and a reconnect from a fresh
    provider."""
    assert (boat_sandbox.type, boat_sandbox.mode) == ("boat", "vm")
    assert boat_sandbox.network_policy == NetworkPolicy()
    assert boat_sandbox.tunnel_urls[_PORT].startswith("https://")

    exit_code, version, stderr = await boat_sandbox.exec_with_output(
        "sudo", "docker", "info", "--format", "{{.ServerVersion}}"
    )
    assert exit_code == 0, stderr
    assert version.strip()

    payload = os.urandom(9 * 1024 * 1024 + 17)
    await boat_sandbox.write_host_file(payload, "/opt/smoke/payload.bin")
    digest = await boat_sandbox.exec_script("sha256sum /opt/smoke/payload.bin && stat -c %U /opt/smoke/payload.bin")
    assert digest.split()[0] == hashlib.sha256(payload).hexdigest() and digest.split()[-1] == "root"

    fresh = build_sandbox_provider("boat")
    try:
        reconnected = await fresh.get_sandbox(boat_sandbox.sandbox_id)
        assert reconnected.tunnel_urls == boat_sandbox.tunnel_urls
        assert reconnected.network_policy == boat_sandbox.network_policy
        assert (await reconnected.exec_with_output("true"))[0] == 0
    finally:
        await fresh.close()


async def test_a_cancelled_exec_stops_its_command_in_the_sandbox(boat_sandbox):
    marker = f"agentenv-cancel-{os.urandom(4).hex()}"
    # Bracketed, so the pattern matches the command but not the shell boat runs pgrep in, whose line holds it.
    pattern = f"[{marker[0]}]{marker[1:]}"
    task = asyncio.create_task(boat_sandbox.exec("bash", "-c", f"exec -a {marker} sleep 600"))
    for _ in range(30):
        if (await boat_sandbox.exec_with_output("pgrep", "-f", pattern))[0] == 0:
            break
        await asyncio.sleep(1)
    else:
        pytest.fail("the command never started")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(30):
        if (await boat_sandbox.exec_with_output("pgrep", "-f", pattern))[0] != 0:
            return
        await asyncio.sleep(1)
    pytest.fail("a cancelled exec left its command running")


async def test_boat_terminate_twice_then_the_sandbox_is_not_found(boat_provider):
    """A second terminate, and one by id, of a deleted sandbox succeed, and ``get_sandbox`` then reports it not
    found."""
    sandbox = await boat_provider.create_vm(cpu=2, memory=4096, timeout=600, setup_for_gateway=False)
    await sandbox.terminate()
    await sandbox.terminate()
    await boat_provider.terminate_sandbox(sandbox.sandbox_id)

    for _ in range(30):
        try:
            await boat_provider.get_sandbox(sandbox.sandbox_id)
        except SandboxNotFoundError:
            return
        await asyncio.sleep(2)
    pytest.fail(f"boat sandbox {sandbox.sandbox_id} was still found a minute after terminate")
