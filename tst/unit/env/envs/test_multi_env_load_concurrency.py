"""How many services load at once is derived from the sandbox's cores, not hardcoded.

Each service load is separately capped by DATA_PLANE_LOAD_TIMEOUT_S, so contention between
concurrent loads doesn't merely slow a load down -- it fails it. Measured on a 13-service
universe: 13-way on 1 vCPU leaves each service ~1/13th of a core and a 409MB service blows
its 600s cap while a 3.5GB one finishes in 89s. Throttling trades total wall-clock (nothing
caps it) for per-service headroom (which is capped).
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_env.env.envs.multi_env import MultiEnv
from agent_env.providers.gateway_provider import GatewayProvider
from agent_env.providers.state import LocalPostgresStateProvider


def _env(nproc: str | None = "8", *, sandbox: bool = True):
    env = MultiEnv(id="env-1", version=1, mcp_server_envs=[])
    gp = GatewayProvider()
    gp._state_provider = LocalPostgresStateProvider()
    env._gateway_provider = gp
    if sandbox:
        s = MagicMock()
        if nproc is None:
            s.exec_script = AsyncMock(side_effect=RuntimeError("ssh died"))
        else:
            s.exec_script = AsyncMock(return_value=f"{nproc}\n")
        env._sandbox = s
    else:
        env._sandbox = None
    return env


@pytest.mark.asyncio
@pytest.mark.parametrize("cores,total,expected", [
    ("1", 13, 2),    # the failing config: 13-way on one core became 2-way
    ("2", 13, 4),
    ("8", 13, 13),   # the measured-good VM: >= ceil(total/2) cores is a no-op
    ("64", 13, 13),  # never exceeds the number of services
    ("8", 3, 3),
])
async def test_limit_scales_with_cores(cores, total, expected):
    assert await _env(cores)._load_concurrency_limit(total) == expected


@pytest.mark.asyncio
async def test_unreadable_core_count_keeps_todays_behaviour():
    """A probe failure must not quietly change how a load runs."""
    assert await _env(None)._load_concurrency_limit(13) == 13


@pytest.mark.asyncio
async def test_no_sandbox_keeps_todays_behaviour():
    assert await _env(sandbox=False)._load_concurrency_limit(13) == 13


@pytest.mark.asyncio
async def test_garbage_core_count_keeps_todays_behaviour():
    assert await _env("not-a-number")._load_concurrency_limit(13) == 13


@pytest.mark.asyncio
async def test_env_override_wins_over_the_derived_value():
    with patch.dict("os.environ", {"AGENT_ENV_LOAD_CONCURRENCY": "3"}):
        assert await _env("64")._load_concurrency_limit(13) == 3


@pytest.mark.asyncio
async def test_env_override_can_pin_the_old_unbounded_behaviour():
    with patch.dict("os.environ", {"AGENT_ENV_LOAD_CONCURRENCY": "999"}):
        assert await _env("1")._load_concurrency_limit(13) == 13


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["0", "-1", "abc", "2.5"])
async def test_a_bad_override_raises_rather_than_silently_defaulting(bad):
    """A typo'd override that fell back would make a contention experiment measure the
    wrong thing and report a number nobody could reproduce."""
    with patch.dict("os.environ", {"AGENT_ENV_LOAD_CONCURRENCY": bad}):
        with pytest.raises(ValueError, match="AGENT_ENV_LOAD_CONCURRENCY"):
            await _env("8")._load_concurrency_limit(13)


def _universe(names: list[str]):
    ua = MagicMock()
    ua.id, ua.version = "uni", 1
    ua.get_metadata.return_value = {}
    arts = []
    for n in names:
        a = MagicMock()
        a.environment_name = n
        arts.append(a)
    ua.get_environment_artifacts.return_value = arts
    return ua


@pytest.mark.asyncio
async def test_the_cap_is_actually_enforced_during_a_load():
    """The real point: never more than `limit` loads in flight at once."""
    env = _env("1")  # 13 services, 1 core -> cap of 2
    env._instance_id = None
    in_flight = 0
    peak = 0

    async def _load(artifact):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1

    env.load_environment_artifact = _load
    store = MagicMock()
    store.get_clean.return_value = None
    names = [f"svc{i}" for i in range(13)]
    with patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=store):
        await env.load_environment_universe_artifact(_universe(names))
    assert peak == 2, f"expected at most 2 concurrent loads, saw {peak}"


@pytest.mark.asyncio
async def test_every_service_still_loads_under_the_cap():
    """Throttling must not drop work -- all 13 still get loaded, just not at once."""
    env = _env("1")
    env._instance_id = None
    loaded = []
    env.load_environment_artifact = AsyncMock(side_effect=lambda a: loaded.append(a.environment_name))
    store = MagicMock()
    store.get_clean.return_value = None
    names = [f"svc{i}" for i in range(13)]
    with patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=store):
        await env.load_environment_universe_artifact(_universe(names))
    assert sorted(loaded) == sorted(names)


@pytest.mark.asyncio
async def test_one_failure_still_reports_without_aborting_the_rest():
    env = _env("8")
    env._instance_id = None
    seen = []

    async def _load(artifact):
        seen.append(artifact.environment_name)
        if artifact.environment_name == "gmail":
            raise RuntimeError("ReadTimeout")

    env.load_environment_artifact = _load
    store = MagicMock()
    store.get_clean.return_value = None
    with patch("agent_env.env.snapshot_store.get_env_snapshot_store", return_value=store):
        with pytest.raises(RuntimeError, match=r"Failed to load 1/3 services.*gmail"):
            await env.load_environment_universe_artifact(_universe(["a", "gmail", "b"]))
    assert sorted(seen) == ["a", "b", "gmail"]  # the others were not cancelled
