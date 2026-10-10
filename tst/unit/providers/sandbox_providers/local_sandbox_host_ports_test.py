"""Host-port allocation for the local backend.

Deployments share one host, so the reserved gateway/pgweb/db-mcp ports can only be
published once; each sandbox gets its own host ports instead. Container ports are
unchanged -- they are how services address each other on the compose network.

The allocator is stubbed so these stay hermetic and assert on the mapping rather
than on whatever the OS hands out; real allocation is covered by
tst/integration/store/local_parallel_deploy_test.py.
"""

import pytest

from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT
from agent_env.providers.sandbox_providers import local_sandbox as local_sandbox_module
from agent_env.env.envs.service_db import DB_MCP_PORT, DB_WEB_PORT
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.e2b import E2BSandbox
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandbox
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox
from agent_env.providers.sandbox_providers.sandbox import port_bindings
from agent_env.providers.sandbox_providers.tensorlake.sandbox import TensorlakeSandbox

RESERVED = [AGENT_ENV_GATEWAY_MCP_PORT, DB_WEB_PORT, DB_MCP_PORT]


@pytest.fixture
def fake_ports(monkeypatch):
    """Deterministic, ever-increasing stand-in for the free-port probe."""
    counter = iter(range(50000, 50100))
    monkeypatch.setattr(local_sandbox_module, "_free_host_port", lambda: next(counter))


def test_base_sandbox_host_port_is_identity():
    """Per-VM backends must be unaffected: their published port is the real one.

    Called unbound because ``host_port`` ignores ``self`` — instantiating the ABC would
    only test that the ABC has abstract methods.
    """
    from agent_env.providers.sandbox_providers.sandbox import Sandbox

    assert Sandbox.host_port(None, 18765) == 18765


@pytest.mark.parametrize("sandbox_class", [ModalVmSandbox, ModalSandbox, E2BSandbox, TensorlakeSandbox])
def test_remote_sandboxes_publish_on_every_interface(sandbox_class):
    """Their tunnels reach a published port from outside the VM, so it stays on every interface."""
    assert port_bindings(sandbox_class.host_ips, 18765, 18765) == ["18765:18765"]


def test_unmapped_ports_pass_through():
    assert LocalSandbox().host_port(18765) == 18765


def test_explicit_map_is_honoured():
    sandbox = LocalSandbox(port_map={18765: 40001, DB_WEB_PORT: 40002})
    assert sandbox.host_port(18765) == 40001
    assert sandbox.host_port(DB_WEB_PORT) == 40002
    assert sandbox.host_port(9999) == 9999          # unmapped -> identity


@pytest.mark.asyncio
async def test_create_vm_allocates_a_distinct_host_port_per_reserved_port(fake_ports):
    sandbox = await LocalSandboxProvider().create_vm(exposed_ports=RESERVED)

    mapped = [sandbox.host_port(p) for p in RESERVED]
    assert len(set(mapped)) == len(RESERVED), f"host ports collided: {mapped}"
    assert not set(mapped) & set(RESERVED), "a reserved port was reused as a host port"


@pytest.mark.asyncio
async def test_two_concurrent_sandboxes_do_not_share_a_host_port(fake_ports):
    """Concurrent deployments must not contend for the same host port."""
    provider = LocalSandboxProvider()
    first = await provider.create_vm(exposed_ports=RESERVED)
    second = await provider.create_vm(exposed_ports=RESERVED)

    for port in RESERVED:
        assert first.host_port(port) != second.host_port(port), (
            f"both sandboxes published container port {port} on the same host port"
        )


@pytest.mark.asyncio
async def test_tunnel_urls_stay_keyed_by_container_port(fake_ports):
    """Callers index ``tunnel_urls[DB_WEB_PORT]``; only the URL's port may change.

    Re-keying by host port would have been the smaller diff and would have broken every
    consumer — silently, since most of them use ``.get()``.
    """
    sandbox = await LocalSandboxProvider().create_vm(exposed_ports=RESERVED)

    assert set(sandbox.tunnel_urls) == set(RESERVED)
    for port in RESERVED:
        assert sandbox.tunnel_urls[port] == f"http://127.0.0.1:{sandbox.host_port(port)}"
