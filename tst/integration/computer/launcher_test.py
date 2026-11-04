"""The QEMU launcher booting the shipped desktop's disk on this machine's Docker with KVM: its driver's port served
through socat, whose listen backlog takes a burst of new connections that QEMU's own forward, with a backlog of 1,
would drop, and a reset booting the disk afresh in the same container, ready again at the same URL. Needs Docker, the
local registry and /dev/kvm."""

import asyncio

import httpx
import pytest

from agent_env.env.envs.computer.utils import boot
from agent_env.env.gateway.constants import WELL_KNOWN_PATH
from agent_env.providers.sandbox_providers.host_images import host_platform
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandboxProvider
from tst.util.capabilities import skip_without_kvm

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow, skip_without_kvm()]

#: New connections at once, 64 times the backlog of QEMU's forward: without socat's own retry of its connection to that
#: forward, one of them waits out the kernel's SYN backoff, past half a minute on a CI runner.
BURST = 64


@pytest.mark.asyncio
async def test_the_launcher_takes_a_burst_of_new_connections_and_a_reset_boots_its_disk_afresh_at_its_url(desktop):
    image = desktop.metadata["composed"]["disk"][host_platform().rsplit("/", 1)[1]]["image"]
    provider, token = LocalSandboxProvider(), boot.control_token()
    environment = boot.settings_environment(desktop.computer, desktop.computer_values(),
                                            environment_name=desktop.environment_name)
    booted = await boot.boot(provider, image=image, port=desktop.driver.port, environment=environment, token=token,
                             network_policy=None, timeout=1800, attribution=None, form="disk")
    try:
        statuses = await asyncio.gather(*(_status(f"{booted.url}{WELL_KNOWN_PATH}") for _ in range(BURST)))
        await provider.reset_vm(booted.sandbox, image=image, env={**environment, **boot.control_environment(token)})
        again = await boot.ready(booted.url, boot.READY_TIMEOUT_S, token)
    finally:
        await boot.discard(booted.sandbox)

    assert statuses == [200] * BURST and again["name"] == desktop.environment_name


async def _status(url: str) -> int:
    async with httpx.AsyncClient(timeout=60) as client:
        return (await client.get(url)).status_code
