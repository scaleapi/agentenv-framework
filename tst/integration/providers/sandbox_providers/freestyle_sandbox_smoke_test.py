"""Real Freestyle VM/container lifecycle; enabled by resolved provider credentials."""

import asyncio
import hashlib

import httpx
import pytest
import pytest_asyncio

from agent_env.providers.sandbox_providers.sandbox_provider import (
    build_sandbox_provider,
)
from tst.util.capabilities import skip_without_remote_sandbox

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio,
    skip_without_remote_sandbox("freestyle"),
]


@pytest_asyncio.fixture(scope="module")
async def freestyle_provider():
    provider = build_sandbox_provider("freestyle")
    try:
        yield provider
    finally:
        await provider.close()


async def test_container_https_files_reconnect_and_delete(freestyle_provider):
    sandbox = await freestyle_provider.create_container(
        image_name="public.ecr.aws/docker/library/nginx:alpine",
        port=80,
        env={"AGENTENV_SMOKE": "freestyle"},
        cpu=3,
        memory=5120,
        disk_size_gb=17,
        timeout=900,
    )
    vm_id = sandbox.sandbox_id
    try:
        assert (sandbox.type, sandbox.mode) == ("freestyle", "container")
        data = await freestyle_provider._client.request("GET", f"/v5/vms/{vm_id}")
        assert data["resources"]["cpu"] >= 3
        assert data["resources"]["memory"] >= 5120
        assert data["resources"]["storage"] >= 17 * 1024
        assert data["ttlSeconds"] == 900
        url = sandbox.tunnel_urls[80]
        async with httpx.AsyncClient(timeout=15) as client:
            for _ in range(30):
                response = await client.get(url)
                if response.status_code == 200:
                    break
                await asyncio.sleep(1)
        assert response.status_code == 200
        assert "nginx" in response.text

        await sandbox.write_file_from_text(
            "freestyle-smoke", "/usr/share/nginx/html/smoke.txt"
        )
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{url}/smoke.txt")
        assert response.status_code == 200
        assert response.text == "freestyle-smoke"

        data = bytes(range(256)) * 4096
        await sandbox.write_host_file(data, "/tmp/agentenv-smoke/blob.bin")
        status, digest, stderr = await sandbox.exec_with_output(
            "sha256sum", "/tmp/agentenv-smoke/blob.bin"
        )
        assert status == 0, stderr
        assert digest.split()[0] == hashlib.sha256(data).hexdigest()

        reconnected = await freestyle_provider.get_sandbox(vm_id)
        assert reconnected.sandbox_id == vm_id
        assert reconnected.tunnel_urls == sandbox.tunnel_urls
        assert reconnected.network_policy is None
        status, _, stderr = await reconnected.exec_with_output(
            "docker", "compose", "version"
        )
        assert status == 0, stderr
    finally:
        await sandbox.terminate()
    with pytest.raises(httpx.HTTPStatusError) as error:
        await freestyle_provider.get_sandbox(vm_id)
    assert error.value.response.status_code == 404
    await sandbox.terminate()
