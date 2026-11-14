"""The ubuntu-24.04 computer on its multi env's gateway clock, on the local backend: the deploy has its follower follow
the gateway, as the computer's sandbox reaches it, under the computer's name and control token, which the record
keeps but the token; sync_env_clock arms the clock as a task does, calling the computer's sync_time; through the
gateway's jumps of a day and a week on and a month back, a rate of 60, a freeze and a restart, a killed follower and a
reset, the session's clock is within 1 s of the gateway's once the follower has acknowledged each change, and runs at
the gateway's rate; a follower killed comes back within 3 s. Each reading is taken inside the computer, between two
of real time, so no driver or exec latency is in it. Needs Docker and the local registry on localhost:5000."""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

import pytest

from agent_env.env import Env
from agent_env.env.env import DeployedEnv
from agent_env.env.envs.computer.utils import inside
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep
from tst.integration.computer.conftest import CONTAINER, FOLLOWER_BACK_S, computer_container, kill_follower, multi_env

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

DAY = 86400
#: How far the session's clock may be from the gateway's: the follower maps the gateway's clock exactly, so this is
#: room for the gateway's state read from here, not for the computer.
MAX_SKEW_S = 1.0
#: The widest a reading's bracket of real time may be; one wider is taken again.
BRACKET_S = 0.5
#: Inside the computer: real time, the session's time, real time again, each a fresh ``date``, the first and last
#: started on real time (``AGENTENV_REAL_TIME``).
BRACKET = ('r1=$(AGENTENV_REAL_TIME=1 date +%s.%N); u=$(date +%s.%N); r2=$(AGENTENV_REAL_TIME=1 date +%s.%N); '
           'echo "$r1 $u $r2"')


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.mark.asyncio
async def test_a_multi_envs_computer_keeps_within_a_second_of_its_gateways_clock_through_jumps_rates_a_kill_and_a_reset(
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
            skews = {"armed": await _skew(deployed, own)}
            for name, by in (("a day on", timedelta(days=1)), ("a week on", timedelta(weeks=1))):
                await _write(deployed, "advance", virtual_time=_iso(await _now(deployed) + by))
                skews[name] = await _skew(deployed, own)
            await _write(deployed, "set_time", virtual_time=_iso(await _now(deployed) - timedelta(days=30)),
                         virtual_seconds_per_real_second=1)
            skews["a month back"] = await _skew(deployed, own)
            await _write(deployed, "set_rate", virtual_seconds_per_real_second=60)
            rates = {"60": await _rate(own)}
            await _write(deployed, "set_rate", virtual_seconds_per_real_second=0)
            rates["frozen"], skews["frozen"] = await _rate(own), await _skew(deployed, own)
            await _write(deployed, "set_rate", virtual_seconds_per_real_second=1)
            skews["running again"] = await _skew(deployed, own)

            killed, back, came_back = kill_follower(container)
            await _write(deployed, "advance", virtual_time=_iso(await _now(deployed) + timedelta(hours=8)))
            skews["after the follower's kill"] = await _skew(deployed, own)

            await ResetEnvTaskStep(id="reset", version=None, env_id=office.id, fail_task_on_error=True).execute(context)
            await _shown(deployed)
            skews["after a reset"] = await _skew(deployed, own)
            state_after_reset = await _shown(deployed)
        finally:
            report = await teardown_run(context)

    assert kept == {"watch_url": kept["watch_url"], "name": "desktop"}
    assert urlsplit(kept["watch_url"]).hostname == "host.docker.internal" and kept["watch_url"].endswith("/clock/watch")
    assert [entry["service"] for entry in synced if entry.get("service") == "desktop"] == ["desktop"]
    assert all(abs(skew) < MAX_SKEW_S for skew in skews.values()), skews
    assert abs(rates["60"] / 60 - 1) < 0.05 and abs(rates["frozen"]) < 0.01, rates
    assert back not in ("-", killed) and came_back < FOLLOWER_BACK_S, came_back
    assert state_after_reset["followers"]["desktop"]["version"] == state_after_reset["version"], state_after_reset
    assert not report.failed


async def _write(deployed, method: str, **params) -> None:
    """A clock write that waits up to 15 s for every computer to show it, as a 504 naming any that doesn't."""
    await deployed.invoke(EXT_CLOCK_URI, method, {**params, "wait_followers_s": 15}, timeout=30)


async def _now(deployed) -> datetime:
    state = await deployed.invoke(EXT_CLOCK_URI, "state", timeout=30)
    return datetime.fromtimestamp(_universe_now(state), timezone.utc)


async def _reading(computer) -> tuple[float, float]:
    """(real, universe): the session's time and the real time it was read at, both from inside the computer."""
    for _ in range(3):
        ran = await inside.run(computer, ["sh", "-c", BRACKET], timeout_s=30)
        assert ran["exit_code"] == 0, ran
        before, universe, after = map(float, ran["stdout"].split())
        if after - before < BRACKET_S:
            return (before + after) / 2, universe
    raise AssertionError(f"reading the computer's time three times took over {BRACKET_S} s each: {ran['stdout']}")


async def _skew(deployed, computer) -> float:
    """How far the session's clock is from the gateway's, in universe seconds: the gateway's state, its real time the
    middle of its request from here, carried at its rate to the real time of the computer's reading."""
    for _ in range(3):
        sent = time.time()
        state = await deployed.invoke(EXT_CLOCK_URI, "state", timeout=30)
        answered = time.time()
        if answered - sent < BRACKET_S:
            break
    real, universe = await _reading(computer)
    expected = _universe_now(state) + state["virtual_seconds_per_real_second"] * (real - (sent + answered) / 2)
    return universe - expected


async def _rate(computer, over_s: float = 2.0) -> float:
    """The session clock's rate: universe seconds per real second between two readings ``over_s`` apart."""
    real, universe = await _reading(computer)
    await asyncio.sleep(over_s)
    later, then = await _reading(computer)
    return (then - universe) / (later - real)


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
