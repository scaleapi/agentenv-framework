"""A snapshot restores the database alone. The services whose bundles ship file trees are
loaded again over HTTP after the swap; everything else keeps the image swap. Not knowing
which services those are means taking the path that cannot be wrong."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.env_providers import EnvironmentGatewayProvider
from agent_env.providers.env_state import LocalPostgresStateProvider


def _artifact(name: str):
    artifact = MagicMock()
    artifact.environment_name = name
    return artifact


def _universe(*names: str):
    universe = MagicMock()
    universe.id = "uni"
    universe.version = 5
    universe.get_environment_artifacts.return_value = [_artifact(n) for n in names]
    universe.get_metadata.return_value = {}
    return universe


def _env():
    env = MultiEnv(id="env-1", version=1, mcp_server_envs=[])
    env._instance_id = "inst-1"
    provider = EnvironmentGatewayProvider()
    provider._state_provider = LocalPostgresStateProvider()
    env._env_provider = provider
    env._load_from_snapshot = AsyncMock()
    env.load_environment_artifact = AsyncMock()
    return env


def _snapshot():
    snapshot = MagicMock()
    snapshot.db_image_artifact_id = "env-snapshot-env-1"
    snapshot.db_image_artifact_version = 3
    snapshot.instance_id = "baker"
    return snapshot


def _store(snapshot):
    store = MagicMock()
    store.get_clean.return_value = snapshot
    return store


async def _load(env, universe, store, file_trees):
    with (
        patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=store),
        patch("agent_env.env.store.update_env_instance_environment_universe", MagicMock()),
        patch("agent_env.env.store.get_env_instance_store", MagicMock()),
        patch("agent_env.env.envs.multi_env.environments_with_file_trees", file_trees),
    ):
        return await env.load_environment_universe_artifact(universe)


@pytest.mark.asyncio
async def test_restore_reingests_exactly_the_file_tree_services_in_universe_order():
    env = _env()
    universe = _universe("gmail", "linear", "gdrive", "hubspot")
    snapshot = _snapshot()
    detector = MagicMock(return_value=["gdrive", "gmail"])

    result = await _load(env, universe, _store(snapshot), detector)

    env._load_from_snapshot.assert_awaited_once_with(snapshot)
    detector.assert_called_once_with(universe)
    artifacts = universe.get_environment_artifacts.return_value
    assert env.load_environment_artifact.await_args_list == [call(artifacts[0]), call(artifacts[2])]
    assert result.restored_from_snapshot is True
    assert result.snapshot_db_image_artifact_id == "env-snapshot-env-1"
    assert result.snapshot_reingested_environments == ["gmail", "gdrive"]


@pytest.mark.asyncio
async def test_a_database_only_universe_restores_without_any_http_load():
    env = _env()
    result = await _load(env, _universe("linear", "hubspot"), _store(_snapshot()), MagicMock(return_value=[]))

    env._load_from_snapshot.assert_awaited_once()
    env.load_environment_artifact.assert_not_awaited()
    assert result.restored_from_snapshot is True
    assert result.snapshot_reingested_environments == []


@pytest.mark.asyncio
async def test_unknown_file_trees_take_the_full_reingest_instead_of_restoring():
    env = _env()
    universe = _universe("gmail", "linear")
    detector = MagicMock(side_effect=OSError("object store unreachable"))

    result = await _load(env, universe, _store(_snapshot()), detector)

    env._load_from_snapshot.assert_not_awaited()
    assert env.load_environment_artifact.await_count == 2
    assert result.restored_from_snapshot is False
    assert result.snapshot_db_image_artifact_id is None
    assert result.snapshot_reingested_environments is None


@pytest.mark.asyncio
async def test_a_failing_reingest_fails_the_restore_and_names_the_service():
    env = _env()
    env.load_environment_artifact.side_effect = TimeoutError("600s")

    with pytest.raises(RuntimeError, match=r"Failed to load 1/1 services: .*gdrive.*TimeoutError"):
        await _load(env, _universe("linear", "gdrive"), _store(_snapshot()), MagicMock(return_value=["gdrive"]))

    env._load_from_snapshot.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_miss_reports_no_reingested_services():
    env = _env()
    store = MagicMock()
    store.get_clean.return_value = None
    detector = MagicMock(return_value=["gdrive"])

    with (
        patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=store),
        patch("agent_env.env.store.update_env_instance_environment_universe", MagicMock()),
        patch("agent_env.env.envs.multi_env.environments_with_file_trees", detector),
    ):
        result = await env.load_environment_universe_artifact(_universe("gdrive"))

    detector.assert_not_called()
    assert env.load_environment_artifact.await_count == 1
    assert result.restored_from_snapshot is False
    assert result.snapshot_reingested_environments is None
