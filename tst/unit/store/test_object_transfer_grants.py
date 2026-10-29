"""Which sandboxes a store's object-transfer grants reach, for the local store and a local run."""

from __future__ import annotations

import pytest

from agent_env.config import reset_config
from agent_env.providers.sandbox_providers.sandbox_provider import SandboxProvider
from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from agent_env.store.routing import LocalRunObjectStore
from tst.util.granting_object_store import GrantingObjectStore


class _VmHere(SandboxProvider):
    """A config-registered provider of VMs on this machine, whose networking forwards this machine's names to its
    loopback, as SmolVM's does."""

    ON_THIS_MACHINE = True
    REACHES_HOST_AT = "127.0.0.1"

    async def create_vm(self, **kwargs):  # pragma: no cover - never provisioned here
        raise NotImplementedError


class _VmElsewhereOnLoopback(_VmHere):
    """One that forwards this machine's names to a loopback address the default server doesn't serve."""

    REACHES_HOST_AT = "127.0.0.2"


@pytest.fixture
def vm_here(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text(
        '[sandbox.providers.vm_here]\nimpl = "tst.unit.store.test_object_transfer_grants:_VmHere"\n'
        '[sandbox.providers.vm_other]\nimpl = "tst.unit.store.test_object_transfer_grants:_VmElsewhereOnLoopback"\n'
    )
    monkeypatch.setenv("AGENT_ENV_CONFIG", str(tmp_path / "config.toml"))
    reset_config()
    yield
    reset_config()


def test_local_store_grants_reach_only_sandboxes_on_this_machine(tmp_path) -> None:
    store = LocalFilesystemObjectStore(str(tmp_path))

    assert store.supports_transfer_grants
    assert store.grants_reach("local")
    assert not store.grants_reach("modal")
    assert not store.grants_reach(None)
    assert not store.grants_reach("no-such-provider")


def test_smol_vm_reaches_local_store_when_its_sdk_is_installed(tmp_path) -> None:
    pytest.importorskip("smol")
    assert LocalFilesystemObjectStore(str(tmp_path)).grants_reach("smol_vm")


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1"])
def test_a_grant_url_naming_this_machines_loopback_reaches_no_container_which_it_names_itself_to(tmp_path, host) -> None:
    assert not LocalFilesystemObjectStore(str(tmp_path), grant_advertise_host=host).grants_reach("local")


def test_a_container_on_this_machines_docker_reaches_the_server_wherever_it_binds(tmp_path) -> None:
    assert LocalFilesystemObjectStore(str(tmp_path), grant_bind_host="172.17.0.1").grants_reach("local")
    assert LocalFilesystemObjectStore(str(tmp_path), grant_advertise_host="172.17.0.1").grants_reach("local")


def test_a_vm_on_this_machine_reaches_the_server_at_its_declared_loopback_address_by_name(tmp_path, vm_here) -> None:
    """The platform default is served on loopback too; a bind host elsewhere, or a URL naming an address, which in the
    VM is the VM's own, is not reached."""
    assert LocalFilesystemObjectStore(str(tmp_path)).grants_reach("vm_here")
    assert LocalFilesystemObjectStore(str(tmp_path), grant_bind_host="127.0.0.1").grants_reach("vm_here")
    assert LocalFilesystemObjectStore(str(tmp_path), grant_advertise_host="host.docker.internal").grants_reach("vm_here")
    assert not LocalFilesystemObjectStore(str(tmp_path), grant_bind_host="172.17.0.1").grants_reach("vm_here")
    assert not LocalFilesystemObjectStore(str(tmp_path), grant_advertise_host="172.17.0.1").grants_reach("vm_here")
    assert not LocalFilesystemObjectStore(str(tmp_path), grant_advertise_host="localhost").grants_reach("vm_here")


def test_a_vm_reaching_an_address_the_default_server_doesnt_serve_needs_the_server_bound_there(tmp_path, vm_here) -> None:
    assert not LocalFilesystemObjectStore(str(tmp_path)).grants_reach("vm_other")
    assert not LocalFilesystemObjectStore(str(tmp_path), grant_bind_host="127.0.0.1").grants_reach("vm_other")
    assert LocalFilesystemObjectStore(str(tmp_path), grant_bind_host="127.0.0.2").grants_reach("vm_other")


def test_custom_private_advertised_host_is_not_claimed_reachable_from_smol_vm(tmp_path) -> None:
    store = LocalFilesystemObjectStore(str(tmp_path), grant_advertise_host="172.17.0.1")
    assert store.grants_reach("local")
    assert not store.grants_reach("smol_vm")


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_loopback_grant_url_is_not_claimed_reachable_from_agent_container(tmp_path, host) -> None:
    store = LocalFilesystemObjectStore(str(tmp_path), grant_advertise_host=host)

    assert not store.grants_reach("local")
    assert not store.grants_reach("smol_vm")


def test_local_store_grants_can_be_turned_off(tmp_path) -> None:
    assert not LocalFilesystemObjectStore(str(tmp_path), grants="off").supports_transfer_grants
    with pytest.raises(ValueError, match="grants must be one of"):
        LocalFilesystemObjectStore(str(tmp_path), grants="on")


def test_a_local_run_offers_grants_that_reach_its_local_store(tmp_path) -> None:
    hosted = GrantingObjectStore(str(tmp_path / "hosted"), reaches=True)
    routed = LocalRunObjectStore(hosted, LocalFilesystemObjectStore(str(tmp_path / "local")))

    assert routed.supports_transfer_grants
    assert routed.grants_reach("local")
    assert not routed.grants_reach("modal")
