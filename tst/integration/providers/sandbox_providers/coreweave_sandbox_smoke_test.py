"""CoreWeave Sandbox smoke test.

Runs when the resolved config can build the ``coreweave`` sandbox provider (the ``coreweave`` extra and
``[sandbox.providers.coreweave.config]`` with ``api_key = "secret:cwsandbox_api_key"`` or a ``wandb_api_key``), the
``remote_sandbox`` capability, and skips with the declared reason otherwise. The allowlist test needs a runner whose
policy allows hostname egress rules; on one that allows CIDR rules only, its create fails with the runner's refusal.
The container tests run public nginx images as the sandbox itself, so their runner must be able to pull them. The tests
never read or print the API key.
"""

from __future__ import annotations

import asyncio
import hashlib
import os

import httpx
import pytest
import pytest_asyncio

from agent_env.providers.sandbox_providers.sandbox import NetworkMode, NetworkPolicy
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxNotFoundError, build_sandbox_provider
from tst.util.capabilities import skip_without_remote_sandbox

_PORT = 8080
# Serves on port 80 from its own entrypoint, and has the sh, mkdir, head and tar the Sandbox API's file write needs.
_SERVING_IMAGE = "public.ecr.aws/nginx/nginx:stable-alpine"
# The same, run as uid 101 and serving on 8080.
_NON_ROOT_IMAGE = "public.ecr.aws/nginx/nginx-unprivileged:stable-alpine"
_MACHINES = {"linux/amd64": "x86_64", "linux/arm64": "aarch64"}

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio,
    skip_without_remote_sandbox("coreweave"),
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
def coreweave_provider():
    return build_sandbox_provider("coreweave")


@pytest_asyncio.fixture(scope="module")
async def coreweave_sandbox(coreweave_provider):
    """One sandbox serving on ``_PORT`` from a container, always deleted after the module."""
    sandbox = await coreweave_provider.create_vm(cpu=1, memory=2048, exposed_ports=[_PORT], timeout=900)
    try:
        await sandbox.exec_script(
            f"docker run -d --name smoke -p {_PORT}:{_PORT} public.ecr.aws/docker/library/python:3.12-slim "
            f"python -m http.server {_PORT}"
        )
        await _wait_for_200(sandbox.tunnel_urls[_PORT])
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_coreweave_docker_ports_upload_and_reconnect(coreweave_provider, coreweave_sandbox):
    """Nested Docker, a public port end to end, an upload over stdin to a root-owned path, and reconnect."""
    assert (coreweave_sandbox.type, coreweave_sandbox.mode) == ("coreweave", "vm")
    assert coreweave_sandbox.network_policy == NetworkPolicy()

    exit_code, version, stderr = await coreweave_sandbox.exec_with_output(
        "sudo", "docker", "info", "--format", "{{.ServerVersion}}"
    )
    assert exit_code == 0, stderr
    assert version.strip()

    payload = os.urandom(1024 * 1024 + 17)
    await coreweave_sandbox.write_host_file(payload, "/opt/smoke/payload.bin")
    digest = await coreweave_sandbox.exec_script("sha256sum /opt/smoke/payload.bin")
    assert digest.split()[0] == hashlib.sha256(payload).hexdigest()

    reconnected = await build_sandbox_provider("coreweave").get_sandbox(coreweave_sandbox.sandbox_id)
    assert reconnected.tunnel_urls == coreweave_sandbox.tunnel_urls
    assert reconnected.network_policy == NetworkPolicy()
    assert (await reconnected.exec_with_output("true"))[0] == 0


async def test_coreweave_allowlisted_sandbox_reaches_its_environment_and_nothing_else(
    coreweave_provider, coreweave_sandbox,
):
    """An allowlisted sandbox reaches an allowed host and another sandbox's public port, and nothing else. Docker
    isn't readied: the policy blocks the package mirror the stock image installs curl from, so busybox fetches."""
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("example.com",))
    agent = await coreweave_provider.create_vm(cpu=1, memory=2048, timeout=600, setup_for_gateway=False,
                                               network_policy=policy)
    try:
        async def fetch(url: str) -> int:
            return (await agent.exec_with_output("wget", "-q", "-O", "/dev/null", "-T", "15", url))[0]

        assert await fetch("https://example.com") == 0
        assert await fetch(coreweave_sandbox.tunnel_urls[_PORT]) == 0
        assert await fetch("https://www.wikipedia.org") != 0

        reconnected = await build_sandbox_provider("coreweave").get_sandbox(agent.sandbox_id)
        assert reconnected.network_policy.restricts_egress
        assert set(reconnected.network_policy.allow_hosts) == set(agent.network_policy.allow_hosts)
    finally:
        await agent.terminate()


async def test_coreweave_container_runs_its_image_as_built_and_reconnects_as_a_container(coreweave_provider):
    """The image's own entrypoint serves the port, the env reaches it, files land in it, and a fresh provider
    reconnects it as a container; terminating twice is fine."""
    sandbox = await coreweave_provider.create_container(image_name=_SERVING_IMAGE, port=80, env={"SMOKE_MARKER": "on"},
                                                        cpu=1, memory=1024, timeout=600)
    try:
        assert (sandbox.type, sandbox.mode) == ("coreweave", "container")
        assert sandbox.network_policy == NetworkPolicy()
        await _wait_for_200(sandbox.tunnel_urls[80])

        exit_code, cmdline, stderr = await sandbox.exec_with_output("cat", "/proc/1/cmdline")
        assert exit_code == 0, stderr
        assert "nginx" in cmdline and "sleep" not in cmdline
        assert await sandbox.exec_with_output("printenv", "SMOKE_MARKER") == (0, "on\n", "")
        exit_code, _, stderr = await sandbox.exec_with_output("ls", "/var/lib/docker")
        assert exit_code != 0  # no Docker host was set up

        await sandbox.write_file_from_text("smoke text", "/tmp/smoke/dir/file.txt")
        assert await sandbox.exec_with_output("cat", "/tmp/smoke/dir/file.txt") == (0, "smoke text", "")

        reconnected = await build_sandbox_provider("coreweave").get_sandbox(sandbox.sandbox_id)
        assert (reconnected.mode, reconnected.tunnel_urls) == ("container", sandbox.tunnel_urls)
        assert reconnected.network_policy == NetworkPolicy()
        assert (await reconnected.exec_with_output("true"))[0] == 0
    finally:
        await sandbox.terminate()
    await sandbox.terminate()
    with pytest.raises(SandboxNotFoundError):
        await build_sandbox_provider("coreweave").get_sandbox(sandbox.sandbox_id)


async def test_coreweave_vm_reconnects_as_a_vm(coreweave_sandbox):
    reconnected = await build_sandbox_provider("coreweave").get_sandbox(coreweave_sandbox.sandbox_id)
    assert reconnected.mode == "vm"


async def test_coreweave_allowlisted_container_reaches_what_it_is_allowed_and_nothing_else(coreweave_provider):
    """A container's egress is the allowlist it was created with, its own port still serves, and a fresh provider reads
    the policy back."""
    policy = NetworkPolicy(mode=NetworkMode.ALLOWLIST, allow_hosts=("example.com",))
    sandbox = await coreweave_provider.create_container(image_name=_SERVING_IMAGE, port=80, env={}, cpu=1, memory=1024,
                                                        timeout=600, network_policy=policy)
    try:
        assert sandbox.network_policy.restricts_egress
        await _wait_for_200(sandbox.tunnel_urls[80])

        async def fetch(url: str) -> int:
            return (await sandbox.exec_with_output("wget", "-q", "-O", "/dev/null", "-T", "15", url))[0]

        assert await fetch("https://example.com") == 0
        assert await fetch("https://www.wikipedia.org") != 0

        reconnected = await build_sandbox_provider("coreweave").get_sandbox(sandbox.sandbox_id)
        assert reconnected.mode == "container"
        assert set(reconnected.network_policy.allow_hosts) == set(sandbox.network_policy.allow_hosts)
    finally:
        await sandbox.terminate()


async def test_coreweave_container_of_a_non_root_image_takes_files_its_user_can_change(coreweave_provider):
    """Commands run as the image's own user, files written over the Sandbox API are that user's to change, and a write
    where that user can't write fails rather than appearing to succeed."""
    sandbox = await coreweave_provider.create_container(image_name=_NON_ROOT_IMAGE, port=8080, env={}, cpu=1,
                                                        memory=1024, timeout=600)
    try:
        await _wait_for_200(sandbox.tunnel_urls[8080])
        exit_code, uid, stderr = await sandbox.exec_with_output("id", "-u")
        assert exit_code == 0, stderr
        assert uid.strip() != "0"

        await sandbox.write_file_from_text("smoke text", "/tmp/smoke/dir/file.txt")
        assert await sandbox.exec_with_output("cat", "/tmp/smoke/dir/file.txt") == (0, "smoke text", "")
        exit_code, _, stderr = await sandbox.exec_with_output("sh", "-c", "echo more >> /tmp/smoke/dir/file.txt")
        assert exit_code == 0, stderr

        with pytest.raises(Exception):
            await sandbox.write_file_from_text("smoke text", "/etc/smoke/file.txt")
        assert (await sandbox.exec_with_output("test", "-e", "/etc/smoke/file.txt"))[0] != 0
    finally:
        await sandbox.terminate()


async def test_coreweave_sandboxes_run_the_platform_and_runtime_they_report(coreweave_provider, coreweave_sandbox):
    """A container and a VM, and the VM's Docker, run the architecture ``platform`` names, and the VM runs under the
    configured runtime class."""
    container = await coreweave_provider.create_container(image_name=_SERVING_IMAGE, port=80, env={}, cpu=1,
                                                          memory=1024, timeout=600)
    try:
        assert container.platform in _MACHINES
        assert await container.exec_with_output("uname", "-m") == (0, _MACHINES[container.platform] + "\n", "")
    finally:
        await container.terminate()

    assert coreweave_sandbox.platform in _MACHINES
    machine = _MACHINES[coreweave_sandbox.platform]
    assert (await coreweave_sandbox.exec_script("uname -m")).strip() == machine
    exit_code, architecture, stderr = await coreweave_sandbox.exec_with_output(
        "sudo", "docker", "info", "--format", "{{.Architecture}}"
    )
    assert (exit_code, architecture.strip()) == (0, machine), stderr
    assert coreweave_sandbox._sandbox.effective_runtime_class == coreweave_provider._runtime_class
