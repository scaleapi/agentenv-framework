"""Host-port allocation for the local backend.

Deployments share one host, so the reserved gateway/pgweb/db-mcp ports can only be
published once; each sandbox gets its own host ports instead. Container ports are
unchanged -- they are how services address each other on the compose network.

The allocator is stubbed so these stay hermetic and assert on the mapping rather
than on the ports it draws; real allocation is covered by
tst/integration/store/local_parallel_deploy_test.py. The allocator's own tests stub
its bind probe, but for one that binds real sockets.
"""

import errno
import os
import platform
import random
import socket
import threading

import pytest
from pytest_socket import enable_socket

from agent_env.config.paths import state_root
from agent_env.env.gateway import AGENT_ENV_GATEWAY_MCP_PORT
from agent_env.providers.sandbox_providers.local_docker import provider as local_sandbox_module
from agent_env.env.envs.service_db import DB_MCP_PORT, DB_WEB_PORT
from agent_env.providers.sandbox_providers.boxd.sandbox import BoxdSandbox
from agent_env.providers.sandbox_providers.daytona.sandbox import DaytonaSandbox
from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.e2b import E2BSandbox
from agent_env.providers.sandbox_providers.modal_sandbox import ModalSandbox
from agent_env.providers.sandbox_providers.modal_vm_sandbox import ModalVmSandbox
from agent_env.providers.sandbox_providers.sandbox import port_bindings
from agent_env.providers.sandbox_providers.tenki.sandbox import TenkiSandbox
from agent_env.providers.sandbox_providers.coreweave.sandbox import CoreWeaveSandbox
from agent_env.providers.sandbox_providers.tensorlake.sandbox import TensorlakeSandbox

RESERVED = [AGENT_ENV_GATEWAY_MCP_PORT, DB_WEB_PORT, DB_MCP_PORT]


@pytest.fixture
def fake_ports(monkeypatch):
    """Deterministic, ever-increasing stand-in for the free-port probe."""
    counter = iter(range(20000, 20100))
    monkeypatch.setattr(local_sandbox_module, "_free_host_port", lambda: next(counter))


def test_base_sandbox_host_port_is_identity():
    """Per-VM backends must be unaffected: their published port is the real one.

    Called unbound because ``host_port`` ignores ``self`` — instantiating the ABC would
    only test that the ABC has abstract methods.
    """
    from agent_env.providers.sandbox_providers.sandbox import Sandbox

    assert Sandbox.host_port(None, 18765) == 18765


@pytest.mark.parametrize("sandbox_class", [ModalVmSandbox, ModalSandbox, E2BSandbox, TensorlakeSandbox, TenkiSandbox,
                                           BoxdSandbox, CoreWeaveSandbox, DaytonaSandbox])
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


@pytest.fixture(autouse=True)
def machine_temp(tmp_path, monkeypatch):
    """The machine's /tmp, which holds the cursor: this test's own, so parallel workers don't share it."""
    folder = tmp_path / "machine-tmp"
    folder.mkdir()
    monkeypatch.setattr(local_sandbox_module, "_MACHINE_TMP", folder)
    return folder


@pytest.fixture
def busy(monkeypatch):
    """Ports the stand-in bind probe finds in use; every other port is free."""
    ports: set[int] = set()
    monkeypatch.setattr(local_sandbox_module, "_bindable", lambda port: port not in ports)
    return ports


def _last_handed_out(port: int | str) -> None:
    local_sandbox_module._host_port_cursor().write_text(str(port))


def test_host_ports_are_handed_out_in_turn_skipping_those_in_use(busy):
    _last_handed_out(20100)
    busy.add(20102)

    assert [local_sandbox_module._free_host_port() for _ in range(3)] == [20101, 20103, 20104]


def test_host_ports_wrap_round_their_range(busy):
    _last_handed_out(29998)

    assert [local_sandbox_module._free_host_port() for _ in range(2)] == [29999, 20000]


@pytest.mark.parametrize("cursor", [None, "", "not a port"])
def test_a_missing_or_unreadable_cursor_starts_at_a_random_port(busy, monkeypatch, cursor):
    monkeypatch.setattr(local_sandbox_module.secrets, "choice", lambda ports: 25000)
    if cursor is not None:
        _last_handed_out(cursor)

    assert local_sandbox_module._free_host_port() == 25000


def test_the_first_port_on_a_machine_does_not_follow_a_reseeded_random(busy, tmp_path, monkeypatch):
    """A trainer reseeds ``random``; fresh machines must still start apart in the range."""
    starts = []
    for machine in ("a", "b", "c"):
        (tmp_path / machine).mkdir()
        monkeypatch.setattr(local_sandbox_module, "_MACHINE_TMP", tmp_path / machine)
        random.seed(0)
        starts.append(local_sandbox_module._free_host_port())

    assert len(set(starts)) > 1


def test_processes_with_their_own_state_roots_or_temp_folders_hand_out_host_ports_in_turn(busy, tmp_path, monkeypatch):
    """As each xdist worker does: a port one drew isn't drawn by the next before docker binds it."""
    _last_handed_out(20100)
    drawn = []
    for root in ("a", "b", "c"):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / root))
        monkeypatch.setenv("TMPDIR", str(tmp_path / root))
        drawn.append(local_sandbox_module._free_host_port())

    assert drawn == [20101, 20102, 20103]


def test_a_machine_tmp_it_cant_make_its_folder_in_leaves_the_cursor_in_the_state_root(busy, tmp_path, monkeypatch):
    """A file where /tmp would be, so not even root makes the folder."""
    (tmp_path / "not-a-folder").write_text("")
    monkeypatch.setattr(local_sandbox_module, "_MACHINE_TMP", tmp_path / "not-a-folder")

    assert local_sandbox_module._host_port_cursor() == state_root() / "locks" / local_sandbox_module._HOST_PORT_CURSOR


def test_a_machine_folder_others_can_write_leaves_the_cursor_in_the_state_root(busy, machine_temp, monkeypatch):
    folder = machine_temp / f"agentenv-{os.getuid()}"
    folder.mkdir(mode=0o777)
    folder.chmod(0o777)

    assert local_sandbox_module._host_port_cursor() == state_root() / "locks" / local_sandbox_module._HOST_PORT_CURSOR


def test_concurrent_draws_never_share_a_host_port(busy):
    """Each draw opens the cursor afresh, so the lock serialises threads as it does processes."""
    drawn: list[int] = []

    def draw():
        for _ in range(50):
            drawn.append(local_sandbox_module._free_host_port())

    threads = [threading.Thread(target=draw) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(drawn)) == len(drawn) == 400


def test_a_range_with_every_port_in_use_is_an_error(busy):
    busy.update(local_sandbox_module._HOST_PORTS)

    with pytest.raises(RuntimeError, match="Every host port from 20000 to 29999 is in use"):
        local_sandbox_module._free_host_port()


class _Socket:
    """Stands in for a socket: binding a port in ``failing`` fails with the errno it maps to."""

    failing: dict[int, int] = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def bind(self, address):
        if address[1] in self.failing:
            raise OSError(self.failing[address[1]], os.strerror(self.failing[address[1]]))


def test_the_probe_skips_a_port_already_in_use(monkeypatch):
    monkeypatch.setattr(local_sandbox_module.socket, "socket", _Socket)
    monkeypatch.setattr(_Socket, "failing", {20101: errno.EADDRINUSE})
    _last_handed_out(20100)

    assert local_sandbox_module._free_host_port() == 20102


def test_a_bind_that_fails_for_another_reason_raises(monkeypatch):
    monkeypatch.setattr(local_sandbox_module.socket, "socket", _Socket)
    monkeypatch.setattr(_Socket, "failing", {20101: errno.EACCES})
    _last_handed_out(20100)

    with pytest.raises(PermissionError):
        local_sandbox_module._free_host_port()


@pytest.mark.skipif(platform.system() != "Linux", reason="only Linux answers on all of 127.0.0.0/8 unconfigured")
def test_the_probe_skips_a_port_taken_on_another_address():
    """Docker also publishes on the bridge gateway, so a port held on any other address is no free port either."""
    enable_socket()
    with socket.socket() as held:
        port = next(port for port in local_sandbox_module._HOST_PORTS if _binds(held, ("127.0.0.2", port)))
        held.listen()
        _last_handed_out(port - 1)

        assert local_sandbox_module._free_host_port() != port


def _binds(s: socket.socket, address: tuple[str, int]) -> bool:
    try:
        s.bind(address)
    except OSError:
        return False
    return True
