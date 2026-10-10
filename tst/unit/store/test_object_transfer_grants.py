"""Which sandboxes a store's object-transfer grants reach, for the local store and a local run."""

from __future__ import annotations

import pytest

from agent_env.store.object_store.local.store import LocalFilesystemObjectStore
from agent_env.store.routing import LocalRunObjectStore
from tst.util.granting_object_store import GrantingObjectStore


def test_local_store_grants_reach_local_and_smol_vm_sandboxes(tmp_path) -> None:
    store = LocalFilesystemObjectStore(str(tmp_path))

    assert store.supports_transfer_grants
    assert store.grants_reach("local")
    assert store.grants_reach("smol_vm")
    assert not store.grants_reach("modal")
    assert not store.grants_reach(None)


def test_custom_bridge_only_grants_are_not_claimed_reachable_from_smol_vm(tmp_path) -> None:
    store = LocalFilesystemObjectStore(str(tmp_path), grant_bind_host="172.17.0.1")
    assert store.grants_reach("local")
    assert not store.grants_reach("smol_vm")


def test_custom_private_advertised_host_is_not_claimed_reachable_from_smol_vm(tmp_path) -> None:
    store = LocalFilesystemObjectStore(str(tmp_path), grant_advertise_host="172.17.0.1")
    assert store.grants_reach("local")
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
