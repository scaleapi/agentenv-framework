"""E2B VM smoke test.

Runs when the resolved config can build the ``e2b`` sandbox provider (``[sandbox.providers.e2b]``
with ``api_key = "secret:e2b_api_key"``) — the ``remote_sandbox`` capability — and skips with the
declared reason otherwise. The test never reads or prints the API key.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from agent_env.providers.sandbox_provider import build_sandbox_provider
from tst.util.capabilities import skip_without_remote_sandbox

_PORT = 8080

pytestmark = [
    pytest.mark.integration,
    pytest.mark.int_test_slow,
    pytest.mark.asyncio,
    skip_without_remote_sandbox("e2b"),
]


@pytest.fixture(scope="module")
def e2b_provider():
    """Build the configured provider without exposing its resolved API key."""
    return build_sandbox_provider("e2b")


@pytest_asyncio.fixture(scope="module")
async def e2b_sandbox(e2b_provider):
    """Provision one E2B VM and always terminate it after the smoke test."""
    sandbox = await e2b_provider.create_vm(
        cpu=1.0,
        memory=2048,
        exposed_ports=[_PORT],
        timeout=600,
    )
    try:
        yield sandbox
    finally:
        await sandbox.terminate()


async def test_e2b_vm_template_docker_ports_and_reconnect(e2b_provider, e2b_sandbox):
    """Exercise template resolution, nested Docker, port lookup, and reconnect.

    ``create_vm`` must resolve or build the configured size-specific template.
    The Docker probe verifies that the resulting E2B VM can host the nested
    containers used by ``SandboxProvider.create_container``. Indexing a second
    port after reconnect verifies that E2B URLs are mapped lazily rather than
    relying on a port list captured at initial creation.
    """
    assert e2b_sandbox.type == "e2b"
    assert e2b_sandbox.mode == "vm"

    exit_code, version, stderr = await e2b_sandbox.exec_with_output(
        "sudo", "docker", "info", "--format", "{{.ServerVersion}}"
    )
    assert exit_code == 0, stderr
    assert version.strip()

    created_url = e2b_sandbox.tunnel_urls[_PORT]
    assert created_url.startswith("https://")

    reconnected = await e2b_provider.get_sandbox(e2b_sandbox.sandbox_id)
    assert reconnected.sandbox_id == e2b_sandbox.sandbox_id
    assert reconnected.tunnel_urls[_PORT].startswith("https://")
    assert reconnected.tunnel_urls[_PORT + 1].startswith("https://")
