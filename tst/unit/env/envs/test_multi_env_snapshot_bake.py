"""Baking a snapshot after a re-ingest is what makes the NEXT load a fast image swap.

The load has already succeeded by the time a bake runs, so the contract is that the bake
is best-effort: it can be skipped, it can fail, it can produce something unusable, and in
every one of those cases the load still succeeds and says what happened. A cache must not
be able to turn working loads into failures.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.gateway_provider import GatewayProvider
from agent_env.providers.state import LocalPostgresStateProvider
from tst.unit.providers.state.fakes import ExternalDbStateProvider


def _universe(n_services: int = 0):
    ua = MagicMock()
    ua.id = "uni"
    ua.version = 39
    ua.get_environment_artifacts.return_value = []
    ua.get_metadata.return_value = {}
    return ua


def _env(provider=None, instance_id="inst-1"):
    env = MultiEnv(id="env-1", version=1, mcp_server_envs=[])
    env._instance_id = instance_id
    gp = GatewayProvider()
    gp._state_provider = provider if provider is not None else LocalPostgresStateProvider()
    env._gateway_provider = gp
    env._load_from_snapshot = AsyncMock()
    return env


def _miss_store():
    store = MagicMock()
    store.get_clean.return_value = None
    return store


def _patches(store, create):
    return (
        patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=store),
        patch("agent_env.env.snapshot_store.EnvSnapshot.create", create),
        patch("agent_env.env.store.update_env_instance_environment_universe", MagicMock()),
    )


async def _load(env, store, create, **kwargs):
    a, b, c = _patches(store, create)
    with a, b, c:
        return await env.load_environment_universe_artifact(_universe(), **kwargs)


@pytest.mark.asyncio
async def test_bake_runs_after_a_miss_and_is_reported():
    create = AsyncMock(return_value=MagicMock(
        is_clean=True, db_image_artifact_id="snap-img", db_image_artifact_version=1,
    ))
    result = await _load(_env(), _miss_store(), create, snapshot_after_load=True)
    create.assert_awaited_once_with("inst-1")
    assert result.restored_from_snapshot is False
    assert result.snapshot_baked is True
    assert result.snapshot_bake_error is None


@pytest.mark.asyncio
async def test_bake_is_off_by_default():
    """A bake pushes a multi-GB image; it must never happen because someone forgot a flag."""
    create = AsyncMock()
    result = await _load(_env(), _miss_store(), create)
    create.assert_not_awaited()
    assert result.snapshot_baked is None


@pytest.mark.asyncio
async def test_env_flag_enables_the_bake_without_a_code_change():
    create = AsyncMock(return_value=MagicMock(
        is_clean=True, db_image_artifact_id="snap-img", db_image_artifact_version=1,
    ))
    with patch.dict("os.environ", {"AGENT_ENV_SNAPSHOT_AFTER_LOAD": "true"}):
        result = await _load(_env(), _miss_store(), create)
    create.assert_awaited_once()
    assert result.snapshot_baked is True


@pytest.mark.asyncio
async def test_explicit_false_beats_the_env_flag():
    create = AsyncMock()
    with patch.dict("os.environ", {"AGENT_ENV_SNAPSHOT_AFTER_LOAD": "true"}):
        result = await _load(_env(), _miss_store(), create, snapshot_after_load=False)
    create.assert_not_awaited()
    assert result.snapshot_baked is None


@pytest.mark.asyncio
async def test_a_failing_bake_does_not_fail_the_load():
    create = AsyncMock(side_effect=RuntimeError("ECR push exploded"))
    result = await _load(_env(), _miss_store(), create, snapshot_after_load=True)
    assert result.snapshot_baked is False
    assert "ECR push exploded" in result.snapshot_bake_error


@pytest.mark.asyncio
async def test_cancellation_during_a_bake_propagates():
    """A cancelled activity must not sail past the bake and report a successful load."""
    import asyncio

    create = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _load(_env(), _miss_store(), create, snapshot_after_load=True)


@pytest.mark.asyncio
async def test_a_dirty_capture_is_reported_as_not_baked():
    """get_clean would never serve it, so calling it a success would be a lie."""
    create = AsyncMock(return_value=MagicMock(
        is_clean=False, db_image_artifact_id="snap-img", db_image_artifact_version=1,
    ))
    result = await _load(_env(), _miss_store(), create, snapshot_after_load=True)
    assert result.snapshot_baked is False
    assert "dirty" in result.snapshot_bake_error


@pytest.mark.asyncio
async def test_no_bake_when_the_snapshot_path_already_hit():
    """Restoring from a snapshot means one already exists — re-baking it is pure waste."""
    snap = MagicMock(db_image_artifact_id="img", db_image_artifact_version=1, instance_id="i")
    store = MagicMock()
    store.get_clean.return_value = snap
    create = AsyncMock()
    env = _env()
    result = await _load(env, store, create, snapshot_after_load=True)
    env._load_from_snapshot.assert_awaited_once_with(snap)
    create.assert_not_awaited()
    assert result.restored_from_snapshot is True
    assert result.snapshot_db_image_artifact_id == "img"
    assert result.snapshot_baked is None


@pytest.mark.asyncio
async def test_no_bake_on_a_backend_that_cannot_restore():
    """A remote-backed deploy can't restore from an image, so capturing one is dead weight."""
    create = AsyncMock()
    result = await _load(_env(provider=ExternalDbStateProvider()), _miss_store(), create,
                         snapshot_after_load=True)
    create.assert_not_awaited()
    assert result.snapshot_baked is None


@pytest.mark.asyncio
async def test_unregistered_env_skips_the_bake_with_a_reason():
    """EnvSnapshot.create keys off the instance record, so there is nothing to snapshot."""
    create = AsyncMock()
    result = await _load(_env(instance_id=None), _miss_store(), create, snapshot_after_load=True)
    create.assert_not_awaited()
    assert result.snapshot_baked is False
    assert "instance_id" in result.snapshot_bake_error


@pytest.mark.asyncio
async def test_lookup_is_keyed_on_the_image_fingerprint():
    store = _miss_store()
    create = AsyncMock()
    await _load(_env(), store, create)
    assert store.get_clean.call_args.kwargs["env_fingerprint"]
