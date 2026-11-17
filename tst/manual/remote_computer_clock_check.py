"""Run explicitly with ``pytest -s tst/manual/remote_computer_clock_check.py``: a computer's universe clock on a remote
sandbox provider, through the steps ``tst/integration/computer/computer_clock_test.py`` takes on the local one. Armed,
a day and a week on, a month back, a rate of 10, frozen, running again, a minute and a half with no change, the
follower killed, and a reset: after each, the session is within ``MAX_SKEW_S`` of the gateway's universe, give or take
how far the reading itself may be off, and it runs at the clock's rate. A browser computer's pages show it too.

A provider whose sandbox is the computer's container itself, as ``modal``'s is, can't reset it in place, which core
refuses; the check takes that refusal as the reset's outcome. On a provider that boots the desktop from its disk form,
whose guest follows no clock yet, arming the clock is refused.

It needs a config whose image and object stores the provider reaches, the provider's credentials, and two multi envs
written there. ``AGENTENV_CLOCK_CHECK_OFFICE`` names one whose computer ``desktop`` is the ubuntu-24.04 desktop, and
``AGENTENV_CLOCK_CHECK_BROWSING`` one whose computer ``browser`` is a browser computer; either left out skips its check.
Each multi env deploys on ``AGENTENV_CLOCK_CHECK_GATEWAY_SANDBOX`` (``modal_vm``), its computer on each of
``AGENTENV_CLOCK_CHECK_SANDBOXES`` (``modal_vm,modal``).

Every reading is taken inside the computer: the gateway's universe, read at the address the computer reaches the gateway
at, as its follower does, and the session's own, each between two readings of real time on the computer's clock. So
neither this machine's distance from the provider nor its clock is in them."""

import asyncio
import json
import os
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

import pytest
from agentenv_protocol.computer.clock import FOLLOWER, FOLLOWER_USER

from agent_env.env.env import DeployedEnv
from agent_env.env.envs.computer.utils import inside
from agent_env.env.envs.computer.utils.sites import RootChannel
from agent_env.env.gateway.constants import EXT_CLOCK_URI
from agent_env.providers import build_sandbox_provider
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import StepContextError, TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep
from tst.integration.computer.browser_computer_test import _browser_op

OFFICE = os.environ.get("AGENTENV_CLOCK_CHECK_OFFICE")
BROWSING = os.environ.get("AGENTENV_CLOCK_CHECK_BROWSING")
GATEWAY_SANDBOX = os.environ.get("AGENTENV_CLOCK_CHECK_GATEWAY_SANDBOX", "modal_vm")
SANDBOXES = os.environ.get("AGENTENV_CLOCK_CHECK_SANDBOXES", "modal_vm,modal").split(",")
RATE = 10
#: How far the session's clock may be from the gateway's: room for reading the two, as on the local provider.
MAX_SKEW_S = 1.0
#: How long a minute and a half with no clock change is: past the gateway's 25 s hold on a watch more than three times,
#: so a provider's tunnel that drops an idle long poll shows as a follower that misses the next change.
IDLE_S = 90
#: How soon a killed follower may be back: supervisord's second, and a remote sandbox's slower start.
FOLLOWER_BACK_S = 10
TTL_S = 2400
#: How far back of now the universe is armed: half a year, as the local clock test arms it.
ARMED_BACK = timedelta(days=182)
#: How long a clock write waits for every computer to show it, and how long a call to the gateway may take.
FOLLOWERS_WAIT_S = 15
CALL_TIMEOUT_S = 30
#: How long the reading script inside the computer may run, how many times it reads, keeping the narrowest, and how
#: long its read of the gateway may take.
READ_TIMEOUT_S = 120
READ_ATTEMPTS = 10
GATEWAY_READ_TIMEOUT_S = 10
#: How far apart the two readings a rate is measured from are, and how far that rate may be from the clock's: a
#: fraction of it, or outright when frozen.
RATE_OVER_S = 3.0
RATE_TOLERANCE = 0.05
FROZEN_TOLERANCE = 0.01
#: How long a killed follower, and a page's time, are waited for, and how often each is looked for.
FOLLOWER_WAIT_S = 60
FOLLOWER_POLL_S = 0.5
PAGE_WAIT_S = 60
PAGE_POLL_S = 0.25
#: How far a page's Date.now() may be from the gateway's universe: room for its title's refresh and the reads from here.
MAX_PAGE_SKEW_S = 3
#: What core says when a computer's provider runs it as the sandbox itself, which a reset can't replace in place.
IN_PLACE = "can't be reset in place"
#: Inside the computer, on real time: the gateway's universe as the computer reaches it at argv[1], read within
#: argv[3] seconds, between two readings of real time, then the session's own (``date``, without the real-time marker),
#: between two more, taken argv[2] times, keeping the narrowest; argv[1] "-" reads the session alone. Each bracket's
#: width is printed, so the caller can say how far the reading may be from the truth.
READ = """
import json, os, subprocess, sys, time, urllib.request
url, attempts, timeout = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
session = {k: v for k, v in os.environ.items() if k != "AGENTENV_REAL_TIME"}
best = None
for _ in range(attempts):
    a = time.time()
    gateway = json.load(urllib.request.urlopen(url, timeout=timeout))["virtual_time"] if url != "-" else None
    b = time.time()
    universe = float(subprocess.run(["date", "+%s.%N"], env=session, capture_output=True, text=True, check=True).stdout)
    c = time.time()
    read = {"gateway_at": (a + b) / 2, "gateway": gateway, "session_at": (b + c) / 2, "session": universe,
            "widths": [round(b - a, 4), round(c - b, 4)]}
    if best is None or max(read["widths"]) < max(best["widths"]):
        best = read
print(json.dumps(best))
"""
#: A page that names its own Date.now(), in milliseconds, in its title, five times a second.
CLOCK_PAGE = "data:text/html," + urllib.parse.quote(
    "<script>setInterval(() => document.title = 'now ' + Date.now(), 200)</script>")


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _at(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


async def _write(deployed, method: str, **params) -> float:
    """A clock write that waits up to ``FOLLOWERS_WAIT_S`` for every computer to show it; the seconds it took."""
    started = time.monotonic()
    await deployed.invoke(EXT_CLOCK_URI, method, {**params, "wait_followers_s": FOLLOWERS_WAIT_S},
                          timeout=CALL_TIMEOUT_S)
    return round(time.monotonic() - started, 2)


async def _now(deployed) -> datetime:
    state = await deployed.invoke(EXT_CLOCK_URI, "state", timeout=CALL_TIMEOUT_S)
    return datetime.fromtimestamp(_at(state["virtual_time"]), timezone.utc)


async def _reading(computer, time_url: str) -> dict:
    argv = ["env", "AGENTENV_REAL_TIME=1", "python3", "-c", READ, time_url, str(READ_ATTEMPTS),
            str(GATEWAY_READ_TIMEOUT_S)]
    ran = await inside.run(computer, argv, timeout_s=READ_TIMEOUT_S)
    assert ran["exit_code"] == 0, ran
    return json.loads(ran["stdout"])


async def _skew(computer, time_url: str, rate: float) -> tuple[float, float]:
    """How far the session's clock is from the gateway's, in universe seconds, both read inside the computer, and how
    far that may be from the truth: half of each reading's bracket of real time, at the clock's rate."""
    read = await _reading(computer, time_url)
    skew = read["session"] - (_at(read["gateway"]) + rate * (read["session_at"] - read["gateway_at"]))
    return round(skew, 3), round(rate * sum(read["widths"]) / 2, 3)


async def _rate(computer, over_s: float = RATE_OVER_S) -> float:
    """The session clock's rate, from two of its own readings ``over_s`` apart: no gateway read is in it."""
    first = await _reading(computer, "-")
    await asyncio.sleep(over_s)
    then = await _reading(computer, "-")
    return round((then["session"] - first["session"]) / (then["session_at"] - first["session_at"]), 3)


async def _followers(computer) -> list[str]:
    channel = RootChannel(await build_sandbox_provider(computer.sandbox_type).get_sandbox(computer.sandbox_id))
    return (await channel.run(["sh", "-c", f"pgrep -u {FOLLOWER_USER} -f {FOLLOWER} || true"])).split()


async def _kill_follower(computer) -> tuple[list[str], list[str], float]:
    """The computer's follower killed with SIGKILL through the root channel: its pids, those back, and how long that
    took, timed from here."""
    channel = RootChannel(await build_sandbox_provider(computer.sandbox_type).get_sandbox(computer.sandbox_id))
    killed = await _followers(computer)
    await channel.run(["kill", "-9", *killed])
    started = time.monotonic()
    while time.monotonic() - started < FOLLOWER_WAIT_S:
        back = await _followers(computer)
        if back and set(back) != set(killed):
            return killed, back, round(time.monotonic() - started, 1)
        await asyncio.sleep(FOLLOWER_POLL_S)
    return killed, [], float(FOLLOWER_WAIT_S)


async def _page_now(browser: DeployedEnv) -> float:
    """What a fresh page in ``browser`` takes for now, in seconds, from its Date.now()."""
    opened = await _browser_op(browser, "tabs/open", {"url": CLOCK_PAGE})
    deadline, tabs = time.monotonic() + PAGE_WAIT_S, []
    while time.monotonic() < deadline:
        tabs = (await _browser_op(browser, "tabs/list", {}))["tabs"]
        tab = next((tab for tab in tabs if tab["id"] == opened["tab"]), None)
        if tab and (tab.get("title") or "").startswith("now "):
            await _browser_op(browser, "tabs/close", {"tab": tab["id"]})
            return int(tab["title"].split()[1]) / 1000
        await asyncio.sleep(PAGE_POLL_S)
    raise AssertionError(f"the clock page named no time within {PAGE_WAIT_S} s: {tabs}")


async def _through_the_steps(env_id: str, child: str, sandbox: str, browser: bool = False) -> dict:
    """The clock test's steps on ``env_id``'s computer ``child`` deployed on ``sandbox``: what each showed."""
    results: dict = {"sandbox": sandbox}
    context = TaskStepContext()
    try:
        started = time.monotonic()
        [deployed] = (await DeployEnvTaskStep(
            id="office", version=None, env_id=env_id, sandbox_type=GATEWAY_SANDBOX, ttl_seconds=TTL_S,
            child_envs={child: {"sandbox_type": sandbox, "env_params": {"form": "container"}}}).execute(context)
        ).deployed_envs
        results["deployed in s"] = round(time.monotonic() - started)
        own = DeployedEnv.from_dict(deployed.child_envs[child])
        kept = own.metadata["computer"]["clock"]
        results["follows"] = {"name": kept["name"], "watch_url": kept["watch_url"].split("://")[0] + "://…",
                              "sandbox_type": own.sandbox_type}
        time_url = kept["watch_url"].removesuffix("/clock/watch") + "/clock/time"

        universe = datetime.now(timezone.utc) - ARMED_BACK
        await SyncEnvClockTaskStep(id="clock", version=None, env_id=env_id,
                                   virtual_time=_iso(universe)).execute(context)
        results["synced"] = [(entry["service"], entry["result"]["follower"]["armed"])
                             for entry in context.metadata["clock_configurations"][-1]["synced"]
                             if "follower" in (entry.get("result") or {})]
        skews, writes, rates = {}, {}, {}
        results.update(skews=skews, rates=rates, writes=writes)
        skews["armed"] = await _skew(own, time_url, 1)
        if browser:
            results["page minus universe s"] = round(await _page_now(own) - _at(
                (await deployed.invoke(EXT_CLOCK_URI, "state", timeout=CALL_TIMEOUT_S))["virtual_time"]), 1)
        for name, by in (("a day on", timedelta(days=1)), ("a week on", timedelta(weeks=1))):
            writes[name] = await _write(deployed, "advance", virtual_time=_iso(await _now(deployed) + by))
            skews[name] = await _skew(own, time_url, 1)
        writes["a month back"] = await _write(deployed, "set_time", virtual_time=_iso(
            await _now(deployed) - timedelta(days=30)), virtual_seconds_per_real_second=1)
        skews["a month back"] = await _skew(own, time_url, 1)
        writes[f"rate {RATE}"] = await _write(deployed, "set_rate", virtual_seconds_per_real_second=RATE)
        rates[RATE] = await _rate(own)
        skews[f"at a rate of {RATE}"] = await _skew(own, time_url, RATE)
        writes["frozen"] = await _write(deployed, "set_rate", virtual_seconds_per_real_second=0)
        rates["frozen"], skews["frozen"] = await _rate(own), await _skew(own, time_url, 0)
        writes["running again"] = await _write(deployed, "set_rate", virtual_seconds_per_real_second=1)
        skews["running again"] = await _skew(own, time_url, 1)

        await asyncio.sleep(IDLE_S)
        writes[f"after {IDLE_S} s idle"] = await _write(deployed, "advance", virtual_time=_iso(
            await _now(deployed) + timedelta(hours=1)))
        skews[f"after {IDLE_S} s idle"] = await _skew(own, time_url, 1)

        killed, back, results["follower back in s"] = await _kill_follower(own)
        results["follower killed, back"] = (bool(killed), bool(back))
        writes["after the follower's kill"] = await _write(deployed, "advance", virtual_time=_iso(
            await _now(deployed) + timedelta(hours=8)))
        skews["after the follower's kill"] = await _skew(own, time_url, 1)

        try:
            await ResetEnvTaskStep(id="reset", version=None, env_id=env_id, fail_task_on_error=True).execute(context)
        except RuntimeError as error:
            if IN_PLACE not in str(error):
                raise
            results["reset"] = f"refused: {IN_PLACE}"
            return results
        own = DeployedEnv.from_dict(context.deployed_envs[0].child_envs[child])
        skews["after a reset"] = await _skew(own, time_url, 1)
        if browser:
            results["page minus universe s after a reset"] = round(await _page_now(own) - _at(
                (await deployed.invoke(EXT_CLOCK_URI, "state", timeout=CALL_TIMEOUT_S))["virtual_time"]), 1)
        state = await deployed.invoke(EXT_CLOCK_URI, "state", timeout=CALL_TIMEOUT_S)
        results["shown after a reset"] = state["followers"][kept["name"]]["version"] == state["version"]
        results["reset"] = "done"
    finally:
        report = await teardown_run(context)
        results["torn down"] = not report.failed
        print("CLOCK CHECK", json.dumps(results, indent=1, default=str), flush=True)
    return results


def _assert(results: dict) -> None:
    assert results["synced"] and all(armed for _, armed in results["synced"]), results
    assert all(abs(skew) - error < MAX_SKEW_S for skew, error in results["skews"].values()), results
    assert abs(results["rates"][RATE] / RATE - 1) < RATE_TOLERANCE, results
    assert abs(results["rates"]["frozen"]) < FROZEN_TOLERANCE, results
    assert results["follower killed, back"] == (True, True) and results["follower back in s"] < FOLLOWER_BACK_S, results
    assert results["reset"] == f"refused: {IN_PLACE}" or results["shown after a reset"], results
    assert results["torn down"], results


@pytest.mark.skipif(not OFFICE, reason="AGENTENV_CLOCK_CHECK_OFFICE names no multi env")
@pytest.mark.parametrize("sandbox", SANDBOXES)
@pytest.mark.asyncio
async def test_a_remote_desktop_keeps_to_its_gateways_clock_through_jumps_rates_an_idle_spell_a_kill_and_a_reset(
        sandbox):
    _assert(await _through_the_steps(OFFICE, "desktop", sandbox))


@pytest.mark.skipif(not BROWSING, reason="AGENTENV_CLOCK_CHECK_BROWSING names no multi env")
@pytest.mark.parametrize("sandbox", SANDBOXES)
@pytest.mark.asyncio
async def test_a_remote_browser_computer_and_its_pages_keep_to_its_gateways_clock(sandbox):
    results = await _through_the_steps(BROWSING, "browser", sandbox, browser=True)
    _assert(results)
    assert abs(results["page minus universe s"]) < MAX_PAGE_SKEW_S, results
    assert results["reset"] != "done" or abs(results["page minus universe s after a reset"]) < MAX_PAGE_SKEW_S, results


@pytest.mark.skipif(not OFFICE, reason="AGENTENV_CLOCK_CHECK_OFFICE names no multi env")
@pytest.mark.parametrize("sandbox", [sandbox for sandbox in SANDBOXES
                                     if build_sandbox_provider(sandbox).disk_image_platforms()])
@pytest.mark.asyncio
async def test_arming_the_clock_of_a_remote_desktop_booted_from_its_disk_is_refused_naming_it(sandbox):
    context, refused = TaskStepContext(), None
    try:
        await DeployEnvTaskStep(id="office", version=None, env_id=OFFICE, sandbox_type=GATEWAY_SANDBOX,
                                ttl_seconds=TTL_S, child_envs={"desktop": {"sandbox_type": sandbox,
                                                                           "env_params": {"form": "disk"}}}
                                ).execute(context)
        with pytest.raises(StepContextError) as refused:
            await SyncEnvClockTaskStep(id="clock", version=None, env_id=OFFICE, virtual_time=_iso(
                datetime.now(timezone.utc) - ARMED_BACK)).execute(context)
    finally:
        report = await teardown_run(context)
        print("CLOCK CHECK", json.dumps({"sandbox": sandbox, "refused": refused and str(refused.value)}), flush=True)

    assert "computer 'desktop' can't follow the universe clock" in str(refused.value) and not report.failed
