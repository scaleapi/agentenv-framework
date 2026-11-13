"""The ubuntu-24.04 computer on its multi env's gateway clock, on the local backend: the deploy has its follower follow
the gateway, as the computer's sandbox reaches it, under the computer's name and control token, which the record
keeps but the token; sync_env_clock arms the clock as a task does, calling the computer's sync_time; the session
reads universe time, and each change is acknowledged in time for a write that waits for it; a follower killed comes
back within 3 s and acknowledges the next advance; a reset brings the page and the follower back on the clock's
current version; and with the computer paused, a clock write core makes fails after its wait, naming the computer,
while once it's back the next is acknowledged within that wait. Needs Docker and the local registry on
localhost:5000."""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pytest

from agent_env.env import Env
from agent_env.env.env import CLOCK_WAIT_FOLLOWERS_S, ClockFollowersBehind, DeployedEnv
from agent_env.env.envs.computer.utils import inside
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep
from tst.integration.computer.conftest import (CONTAINER, FOLLOWER_BACK_S, computer_container, docker, kill_follower,
                                               multi_env)

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

DAY = 86400


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.mark.asyncio
async def test_a_multi_envs_computer_follows_its_gateways_clock_through_arming_a_killed_follower_and_a_reset(
        desktop, items):
    with namespace_routing():
        office = multi_env(items, Env.get(desktop.id))
        step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                                 child_envs={"desktop": CONTAINER})
        context = TaskStepContext()
        try:
            [deployed] = (await step.execute(context)).deployed_envs
            own = DeployedEnv.from_dict(deployed.child_envs["desktop"])
            kept = own.metadata["computer"]["clock"]
            container = computer_container(own.sandbox_id)

            universe = datetime.now(timezone.utc) - timedelta(days=182)
            await SyncEnvClockTaskStep(id="clock", version=None, env_id=office.id,
                                       virtual_time=_iso(universe)).execute(context)
            synced = context.metadata["clock_configurations"][-1]["synced"]
            armed = await _session_offset(own) - (universe.timestamp() - time.time())

            day_on = universe + timedelta(days=1)
            await _advance(deployed, day_on)
            advanced = await _session_offset(own) - (day_on.timestamp() - time.time())

            killed, back, came_back = kill_follower(container)
            later = day_on + timedelta(hours=8)
            await _advance(deployed, later)
            after_kill = await _session_offset(own) - (later.timestamp() - time.time())

            await ResetEnvTaskStep(id="reset", version=None, env_id=office.id, fail_task_on_error=True).execute(context)
            state = await _shown(deployed)
            after_reset = await _session_offset(own) - (_universe_now(state) - time.time())
            state_after_reset = await _shown(deployed)
        finally:
            report = await teardown_run(context)

    assert kept == {"watch_url": kept["watch_url"], "name": "desktop"}
    assert urlsplit(kept["watch_url"]).hostname == "host.docker.internal" and kept["watch_url"].endswith("/clock/watch")
    assert [entry["service"] for entry in synced if entry.get("service") == "desktop"] == ["desktop"]
    assert abs(armed) < 5 and abs(advanced) < 5 and abs(after_kill) < 5 and abs(after_reset) < 5, (
        armed, advanced, after_kill, after_reset)
    assert back not in ("-", killed) and came_back < FOLLOWER_BACK_S, came_back
    assert state_after_reset["followers"]["desktop"]["version"] == state_after_reset["version"], state_after_reset
    assert not report.failed


@pytest.mark.asyncio
async def test_a_paused_computer_fails_cores_clock_write_naming_it_and_once_back_follows_the_next(desktop, items):
    with namespace_routing():
        office = multi_env(items, Env.get(desktop.id))
        step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                                 child_envs={"desktop": CONTAINER})
        context = TaskStepContext()
        try:
            [deployed] = (await step.execute(context)).deployed_envs
            own = DeployedEnv.from_dict(deployed.child_envs["desktop"])
            container = computer_container(own.sandbox_id)
            universe = datetime.now(timezone.utc) - timedelta(days=182)
            await SyncEnvClockTaskStep(id="clock", version=None, env_id=office.id,
                                       virtual_time=_iso(universe)).execute(context)

            docker("pause", container)
            failed, started = None, time.monotonic()
            try:
                await deployed.write_clock("advance", {"virtual_time": _iso(universe + timedelta(hours=8))})
            except ClockFollowersBehind as e:
                failed = e
            finally:
                took = time.monotonic() - started
                docker("unpause", container)

            day_on = universe + timedelta(days=1)
            started = time.monotonic()
            await deployed.write_clock("advance", {"virtual_time": _iso(day_on)})
            acknowledged = time.monotonic() - started
            offset = await _session_offset(own) - (day_on.timestamp() - time.time())
        finally:
            report = await teardown_run(context)

    assert failed is not None and failed.stale == ["desktop"], failed
    assert CLOCK_WAIT_FOLLOWERS_S <= took < CLOCK_WAIT_FOLLOWERS_S + 2, took
    assert acknowledged < CLOCK_WAIT_FOLLOWERS_S and abs(offset) < 5, (acknowledged, offset)
    print(f"paused: failed after {took:.2f} s; back: acknowledged in {acknowledged * 1000:.0f} ms")
    assert not report.failed


async def _advance(deployed, to: datetime) -> None:
    """The clock moved to ``to``, waiting up to 15 s for every computer to show it, as a 504 naming any that doesn't."""
    await deployed.invoke(EXT_CLOCK_URI, "advance", {"virtual_time": _iso(to), "wait_followers_s": 15}, timeout=30)


async def _shown(deployed, within: float = 15) -> dict:
    """The clock's state once the desktop shows its current version."""
    deadline = time.monotonic() + within
    while True:
        state = await deployed.invoke(EXT_CLOCK_URI, "state", timeout=30)
        if state["followers"]["desktop"]["version"] == state["version"] or time.monotonic() > deadline:
            return state
        await asyncio.sleep(0.2)


def _universe_now(state: dict) -> float:
    return datetime.fromisoformat(state["virtual_time"].replace("Z", "+00:00")).timestamp()


async def _session_offset(computer) -> float:
    """How far the session's clock is from real time, read with ``date`` through the driver, as the session user."""
    ran = await inside.run(computer, ["date", "+%s.%N"], timeout_s=30)
    assert ran["exit_code"] == 0, ran
    return float(ran["stdout"]) - time.time()
