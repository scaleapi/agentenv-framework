"""A multi env's computer on its gateway's clock, end to end on the local backend: the clock follower test bundle's
desktop (the ubuntu-24.04 base, the reference driver and the ``clock`` app) deployed beside an MCP server.

``sync_env_clock`` arms the gateway about 6 months back: core starts the computer's follower as root, the arm waits for
it, and the session's shell and Python then read the gateway's universe time. An advance of a day and a re-arm at 60x
reach them too, each acknowledged before its write returns. The session user can't write the clock page, read the
follower's config or stop the follower, and a follower killed as root is back for the next write. Needs Docker and the
local registry on localhost:5000."""

import asyncio
import subprocess
import time
import uuid
from datetime import datetime, timezone

import pytest
from agent_env.env import Env
from agent_env.env.env import DeployedEnv
from agent_env.env.envs.computer.utils import clock, inside
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep
from tst.integration.env.envs.server_env_local_test import _put_items_image
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

CONTAINER = {"sandbox_type": "local", "env_params": {"form": "container"}}
DAY = 86400
#: How far outside the gateway's reads on either side of it a session's read may fall: the follower's offset estimate
#: and the page's continuity, well inside the second a computer and the gateway must agree to.
SLACK_S = 0.5


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat().replace("+00:00", "Z")


def _epoch(rfc3339: str) -> float:
    return datetime.fromisoformat(rfc3339.replace("Z", "+00:00")).timestamp()


async def _session_now(computer: DeployedEnv) -> dict[str, float]:
    """The time the session user's shell and Python read, as epoch seconds."""
    ran = await inside.run(computer, ["sh", "-c", "date +%s.%N; python3 -c 'import time; print(time.time())'"],
                           timeout_s=30)
    assert ran["exit_code"] == 0, ran
    shell, python = ran["stdout"].split()
    return {"shell": float(shell), "python": float(python)}


async def _gateway_now(deployed: DeployedEnv) -> float:
    return _epoch((await deployed.invoke(EXT_CLOCK_URI, "state"))["virtual_time"])


async def _agreement(computer: DeployedEnv, deployed: DeployedEnv) -> dict:
    """The session's shell and Python times, read between two reads of the gateway's clock."""
    first = await _gateway_now(deployed)
    session = await _session_now(computer)
    return {"gateway": (first, await _gateway_now(deployed)), **session}


def _agrees(read: dict) -> bool:
    low, high = read["gateway"]
    return all(low - SLACK_S <= read[what] <= high + SLACK_S for what in ("shell", "python"))


async def _as_session(computer: DeployedEnv, command: str) -> dict:
    return await inside.run(computer, ["sh", "-c", command], timeout_s=30)


def _container(computer: DeployedEnv) -> str:
    """The computer's container, the one of its sandbox's holding the follower."""
    listed = subprocess.run(["docker", "ps", "-q", "--filter", f"name={computer.sandbox_id}"],
                            capture_output=True, text=True, check=True).stdout.split()
    for container in listed:
        if subprocess.run(["docker", "exec", container, "test", "-e", "/opt/agentenv/clock/follow"]).returncode == 0:
            return container
    raise AssertionError(f"no container of sandbox {computer.sandbox_id} has the clock follower: {listed}")


def _as_root(container: str, *argv: str) -> str:
    return subprocess.run(["docker", "exec", "-u", "0", container, *argv], capture_output=True, text=True).stdout


@pytest.fixture(scope="module")
def items(clock_follower_desktop):
    with namespace_routing():
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        return MCPServerEnv.put(id=f"@local/~/clock-{uid}/items", environment_name="items",
                                docker_image_artifact=_put_items_image(f"clock-items-{uid}"))


@pytest.mark.asyncio
async def test_a_multi_envs_computer_follows_its_gateways_clock_through_an_arm_an_advance_and_a_rate(
        clock_follower_desktop, items):
    with namespace_routing():
        office = MultiEnv.put(id=f"@local/~/clock-{uuid.uuid4().hex[:8]}/office", mcp_server_envs=[items],
                              computer_envs=[Env.get(clock_follower_desktop.id)], name="office")
        context = TaskStepContext()
        try:
            [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                  ttl_seconds=900, child_envs={"desktop": CONTAINER})
                          .execute(context)).deployed_envs
            computer = DeployedEnv.from_dict(deployed.child_envs["desktop"])
            follows = clock.runs_follower(computer.environment_card)
            before = await _session_now(computer)
            real_before = time.time()

            universe = time.time() - 182 * DAY
            armed_at = time.monotonic()
            await SyncEnvClockTaskStep(id="arm", version=None, env_id=office.id,
                                       virtual_time=_iso(universe)).execute(context)
            arm_s = time.monotonic() - armed_at
            configuration = context.metadata["clock_configurations"][-1]
            armed = await _agreement(computer, deployed)

            target = armed["gateway"][1] + DAY
            await deployed.invoke(EXT_CLOCK_URI, "advance", {"virtual_time": _iso(target), "wait_followers_s": 10})
            advanced = await _agreement(computer, deployed)

            await SyncEnvClockTaskStep(id="fast", version=None, env_id=office.id, virtual_time=_iso(target),
                                       virtual_seconds_per_real_second=60).execute(context)
            fast = [await _session_now(computer)]
            await asyncio.sleep(3)
            fast.append(await _session_now(computer))

            # The session user can't change, stop or read into the clock.
            refused = {
                "write_page": await _as_session(computer, "printf x > /dev/shm/agentenv-clock"),
                "write_file": await _as_session(computer, "echo '@0 x1' > /run/agentenv-clock/clock"),
                "read_config": await _as_session(computer, f"cat {clock.CONFIG}"),
                "kill": await _as_session(computer, "pkill -9 -u agentenv-clock"),
            }
            # Killed as root, the follower is back for the next write, which waits for it.
            container = _container(computer)
            _as_root(container, "pkill", "-9", "-u", "agentenv-clock")
            await asyncio.sleep(2.5)
            back = _as_root(container, "pgrep", "-u", "agentenv-clock").split()
            await deployed.invoke(EXT_CLOCK_URI, "advance", {"virtual_time": _iso(target + 3 * DAY),
                                                             "wait_followers_s": 10})
            after_kill = await _agreement(computer, deployed)
        finally:
            report = await teardown_run(context)

    assert follows, "the computer's card doesn't say it runs the clock follower"
    assert abs(before["shell"] - real_before) < 5, before  # real time before the arm
    assert [s["service"] for s in configuration["synced"] if s["service"] == "clock"] == ["clock"], configuration
    assert arm_s < 30, arm_s
    assert [read for read in (armed, advanced, after_kill) if not _agrees(read)] == []
    assert abs(armed["gateway"][0] - universe) < 60, (armed, universe)
    assert advanced["gateway"][0] - armed["gateway"][1] >= DAY, (armed, advanced)
    assert after_kill["gateway"][0] >= target + 3 * DAY, after_kill
    assert 3 * 60 * 0.8 < fast[1]["shell"] - fast[0]["shell"] < 3 * 60 * 1.6, fast  # ~3 s at 60x, plus the reads
    assert all(result["exit_code"] != 0 for result in refused.values()), refused
    assert back, "the follower wasn't restarted"
    assert not report.failed
