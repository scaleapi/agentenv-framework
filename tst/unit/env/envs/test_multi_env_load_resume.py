"""An interrupted universe load resumes instead of re-running the whole thing.

The load is destructive per service (reset-then-add), and a Temporal retry re-enters the
step from scratch -- so one slow service used to cost three full re-wipes of all 13, which
is how a ~11 minute failure became a ~44 minute one.

Progress lives in Mongo, not the step context, because the worker snapshots the context
BEFORE running a step and re-sends that frozen copy on every heartbeat: anything written
mid-step never reaches the heartbeat a retry restores from.

The safety property under test is the changelog gate. Skipping services must NEVER weaken
the documented guarantee that loading a universe wipes and re-seeds it -- so resume applies
only when nothing has mutated the data since the last load.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.gateway_provider import GatewayProvider
from agent_env.providers.sandbox_provider import SANDBOX_MODE_CONTAINER
from agent_env.providers.state import LocalPostgresStateProvider


def _universe(names: list[str], version: int = 39):
    ua = MagicMock()
    ua.id, ua.version = "uni", version
    ua.get_metadata.return_value = {}
    arts = []
    for n in names:
        a = MagicMock()
        a.environment_name = n
        arts.append(a)
    ua.get_environment_artifacts.return_value = arts
    return ua


ALL = ["a", "b", "gmail"]


def _env(*, instance_id: str | None = "inst-1", mode: str = "vm"):
    env = MultiEnv(id="env-1", version=1, mcp_server_envs=[])
    env._instance_id = instance_id
    gp = GatewayProvider()
    gp._state_provider = LocalPostgresStateProvider()
    env._gateway_provider = gp
    s = MagicMock()
    s.mode = mode
    s.exec_script = AsyncMock(return_value="8\n")
    env._sandbox = s
    env._load_from_snapshot = AsyncMock()
    return env


async def _load(env, universe, *, recorded: list[str], changelog_empty: bool):
    """Drive a load with a stubbed instance store + changelog; returns (loaded, store)."""
    loaded: list[str] = []
    env.load_environment_artifact = AsyncMock(
        side_effect=lambda a: loaded.append(a.environment_name)
    )
    store = MagicMock()
    store.get_loaded_environments.return_value = list(recorded)
    snap_store = MagicMock()
    snap_store.get_clean.return_value = None
    with patch("agent_env.env.store.get_env_instance_store", return_value=store), \
         patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=snap_store), \
         patch("agent_env.env.snapshot_store._check_changelog_empty",
               AsyncMock(return_value=changelog_empty)), \
         patch("agent_env.env.store.update_env_instance_environment_universe", MagicMock()):
        await env.load_environment_universe_artifact(universe)
    return loaded, store


@pytest.mark.asyncio
async def test_a_retry_skips_what_already_loaded():
    """The saving: one failed service costs one service on retry, not all of them."""
    loaded, _ = await _load(_env(), _universe(ALL), recorded=["a", "b"], changelog_empty=True)
    assert loaded == ["gmail"]


@pytest.mark.asyncio
async def test_a_mutated_env_reloads_everything():
    """THE safety property: load_universe is documented as destructive. If an agent has
    touched the data, a re-load must wipe and re-seed every service, not skip any."""
    loaded, _ = await _load(_env(), _universe(ALL), recorded=["a", "b"], changelog_empty=False)
    assert sorted(loaded) == sorted(ALL)


@pytest.mark.asyncio
async def test_nothing_recorded_loads_everything():
    loaded, _ = await _load(_env(), _universe(ALL), recorded=[], changelog_empty=True)
    assert sorted(loaded) == sorted(ALL)


@pytest.mark.asyncio
async def test_a_fully_recorded_universe_still_reloads():
    """Skipping everything would turn an explicit caller request into a silent no-op."""
    loaded, _ = await _load(_env(), _universe(ALL), recorded=ALL, changelog_empty=True)
    assert sorted(loaded) == sorted(ALL)


@pytest.mark.asyncio
async def test_progress_is_recorded_per_service_as_it_completes():
    """Recorded per service, not once at the end -- a load that dies partway has to leave
    usable progress behind."""
    _, store = await _load(_env(), _universe(ALL), recorded=[], changelog_empty=True)
    names = [c.args[3] for c in store.record_loaded_environment.call_args_list]
    assert sorted(names) == sorted(ALL)


@pytest.mark.asyncio
async def test_a_failed_service_is_not_recorded():
    """A service whose reset succeeded but whose add failed is EMPTY. Recording it would let
    a retry skip it and leave the env silently missing data."""
    env = _env()
    store = MagicMock()
    store.get_loaded_environments.return_value = []
    snap_store = MagicMock()
    snap_store.get_clean.return_value = None

    async def _load_one(artifact):
        if artifact.environment_name == "gmail":
            raise RuntimeError("ReadTimeout")

    env.load_environment_artifact = _load_one
    with patch("agent_env.env.store.get_env_instance_store", return_value=store), \
         patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=snap_store), \
         patch("agent_env.env.snapshot_store._check_changelog_empty", AsyncMock(return_value=True)), \
         patch("agent_env.env.store.update_env_instance_environment_universe", MagicMock()):
        with pytest.raises(RuntimeError, match="gmail"):
            await env.load_environment_universe_artifact(_universe(ALL))
    recorded = [c.args[3] for c in store.record_loaded_environment.call_args_list]
    assert "gmail" not in recorded
    assert sorted(recorded) == ["a", "b"]  # the ones that really did load


@pytest.mark.asyncio
async def test_resume_state_is_keyed_by_universe_version():
    """A new universe version is different data; resuming into it would seed a mix."""
    env = _env()
    _, store = await _load(env, _universe(ALL, version=40), recorded=[], changelog_empty=True)
    assert store.get_loaded_environments.call_args.args[2] == 40


@pytest.mark.asyncio
async def test_an_unregistered_env_cannot_resume():
    loaded, store = await _load(_env(instance_id=None), _universe(ALL),
                                recorded=["a", "b"], changelog_empty=True)
    assert sorted(loaded) == sorted(ALL)
    store.get_loaded_environments.assert_not_called()


@pytest.mark.asyncio
async def test_container_mode_cannot_resume():
    """No changelog to consult there, so resume can't be proven safe."""
    loaded, _ = await _load(_env(mode=SANDBOX_MODE_CONTAINER), _universe(ALL),
                            recorded=["a", "b"], changelog_empty=True)
    assert sorted(loaded) == sorted(ALL)


@pytest.mark.asyncio
async def test_a_changelog_probe_failure_loads_everything():
    """Fail toward redundant work, never toward leaving stale data in place."""
    env = _env()
    env.load_environment_artifact = AsyncMock()
    loaded: list[str] = []
    env.load_environment_artifact = AsyncMock(side_effect=lambda a: loaded.append(a.environment_name))
    store = MagicMock()
    store.get_loaded_environments.return_value = ["a", "b"]
    snap_store = MagicMock()
    snap_store.get_clean.return_value = None
    with patch("agent_env.env.store.get_env_instance_store", return_value=store), \
         patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=snap_store), \
         patch("agent_env.env.snapshot_store._check_changelog_empty",
               AsyncMock(side_effect=RuntimeError("psql gone"))), \
         patch("agent_env.env.store.update_env_instance_environment_universe", MagicMock()):
        await env.load_environment_universe_artifact(_universe(ALL))
    assert sorted(loaded) == sorted(ALL)


@pytest.mark.asyncio
async def test_a_snapshot_restore_clears_resume_state():
    """A restore replaces the whole database, so per-service progress describes nothing."""
    env = _env()
    store = MagicMock()
    snap_store = MagicMock()
    snap_store.get_clean.return_value = MagicMock(
        db_image_artifact_id="img", db_image_artifact_version=1, instance_id="i",
    )
    with patch("agent_env.env.store.get_env_instance_store", return_value=store), \
         patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=snap_store), \
         patch("agent_env.env.store.update_env_instance_environment_universe", MagicMock()):
        await env.load_environment_universe_artifact(_universe(ALL))
    store.clear_loaded_environments.assert_called_once_with("inst-1")
