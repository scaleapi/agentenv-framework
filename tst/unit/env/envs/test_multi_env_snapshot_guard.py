"""The snapshot fast-path in MultiEnv.load_environment_universe_artifact is gated on a backend
CAPABILITY (supports_restore_from_snapshot), not a type check — so a remote-backed deploy skips
the local-only servicedb-image swap and falls through to the normal load."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.gateway_provider import GatewayProvider
from agent_env.providers.state import LocalPostgresStateProvider
from agent_env.config import set_document_store
from tst.unit.providers.state.fakes import ExternalDbStateProvider


def _empty_universe():
    """A universe artifact with no services/metadata — the normal-load branch is then a no-op."""
    ua = MagicMock()
    ua.id = "universe-1"
    ua.version = 1
    ua.get_environment_artifacts.return_value = []
    ua.get_metadata.return_value = {}
    return ua


def _multi_env_with_backend(provider):
    env = MultiEnv(id="env-1", version=1, mcp_server_envs=[])
    env._instance_id = None  # skip the env-instance environment_universe update
    gp = GatewayProvider()
    gp._state_provider = provider  # as rehydrate_state would set on reattach
    env._gateway_provider = gp
    env._load_from_snapshot = AsyncMock()
    return env


@pytest.mark.asyncio
async def test_local_backend_uses_snapshot_fast_path():
    env = _multi_env_with_backend(LocalPostgresStateProvider())
    snap = MagicMock(db_image_artifact_id="img", db_image_artifact_version=1, instance_id="i")
    store = MagicMock()
    store.get_clean.return_value = snap

    with patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=store):
        await env.load_environment_universe_artifact(_empty_universe())

    store.get_clean.assert_called_once()
    env._load_from_snapshot.assert_awaited_once_with(snap)


@pytest.mark.asyncio
async def test_remote_backend_skips_snapshot_fast_path():
    env = _multi_env_with_backend(ExternalDbStateProvider())
    store = MagicMock()
    store.get_clean.return_value = MagicMock()  # even if a (local) snapshot exists, it must be ignored

    with patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=store):
        await env.load_environment_universe_artifact(_empty_universe())

    # Capability short-circuits before even querying for a snapshot; the local-only swap never runs.
    store.get_clean.assert_not_called()
    env._load_from_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_unset_provider_skips_snapshot_fast_path():
    """No provider (e.g. reattach that couldn't rehydrate) => no snapshot assumption, normal load."""
    env = _multi_env_with_backend(None)
    store = MagicMock()
    store.get_clean.return_value = MagicMock()

    with patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=store):
        await env.load_environment_universe_artifact(_empty_universe())

    store.get_clean.assert_not_called()
    env._load_from_snapshot.assert_not_awaited()


@pytest.mark.asyncio
async def test_snapshot_capture_gated_off_for_remote_backend():
    """Snapshot CAPTURE (EnvSnapshot.create) mirrors the restore gate: it bakes the local servicedb
    container into an image, so a backend that can't restore from that image (remote) is rejected
    early with a clear error — not a cryptic 'servicedb container not found' deep in the capture."""
    from agent_env.env.snapshot_store import EnvSnapshot

    deployed = MagicMock(env_id="env-1", env_version=1, sandbox_id="sb-1")
    inst_store = MagicMock()
    inst_store.get.return_value = deployed
    inst_store.get_environment_universe.return_value = {"id": "universe-1", "version": 1}

    # A reattached MultiEnv whose rehydrated backend is remote (no servicedb to commit).
    gp = GatewayProvider()
    gp._state_provider = ExternalDbStateProvider()
    reattached = MagicMock()
    reattached._sandbox = MagicMock(mode="vm")
    reattached._gateway_provider = gp

    with patch("agent_env.env.store.get_env_instance_store", return_value=inst_store), \
         patch("agent_env.env.env.Env.get", return_value=MultiEnv(id="env-1", version=1, mcp_server_envs=[])), \
         patch("agent_env.env.envs.multi_env.MultiEnv.from_deployed_env", AsyncMock(return_value=reattached)):
        with pytest.raises(NotImplementedError, match="only supported for the local Postgres backend"):
            await EnvSnapshot.create("inst-1")

    # Gate fires before touching the sandbox (no servicedb exec attempted).
    reattached._sandbox.exec_script.assert_not_called()


@pytest.mark.asyncio
async def test_from_deployed_env_rehydrates_state_provider():
    """Reattach doesn't re-acquire, so from_deployed_env restores gp._state_provider inline from the
    recorded instance's state_type — powering the capability check + install_changelog on reattach."""
    from agent_env.providers.state import (
        EnvStateInstance,
        EnvStateInstanceStore,
        register_env_state_instance,
        reset_env_state_instance_store,
        set_env_state_instance_store,
    )
    from tst.unit.store.fakes import FakeDocumentStore

    store = EnvStateInstanceStore()
    set_document_store(FakeDocumentStore())
    set_env_state_instance_store(store)
    try:
        record = register_env_state_instance(EnvStateInstance(state_type="local_postgres"), 3600)

        fake_sandbox = MagicMock(mode="vm")
        provider = MagicMock()
        provider.get_sandbox = AsyncMock(return_value=fake_sandbox)

        deployed = MagicMock(
            env_id="env-1", env_version=1, sandbox_type=None, sandbox_id="sb-1",
            sandbox_ids={}, env_state_instance_ids=[record.instance_id],
            gateway_url="http://gw", instance_id="inst-1",
        )

        with patch("agent_env.env.env.Env.get", return_value=MultiEnv(id="env-1", version=1, mcp_server_envs=[])), \
             patch("agent_env.providers.get_env_sandbox_provider", return_value=provider):
            env = await MultiEnv.from_deployed_env(deployed)

        assert isinstance(env._gateway_provider._state_provider, LocalPostgresStateProvider)
        assert env._gateway_provider._state_instance.instance_id == record.instance_id
    finally:
        reset_env_state_instance_store()
