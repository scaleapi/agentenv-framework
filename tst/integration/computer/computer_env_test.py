"""The shipped desktop example's computer env, composed on this machine: naming ubuntu-24.04 and reference by alias, it
builds an image for each arch this machine's Docker can build, pushed to the local registry and written as the env's
computer, and gate 3 makes the image of this machine's arch ready; that image boots with everything running as the
session user and passes the conformance probe at full three times in a row; another env that changes only what doesn't
change the image reuses it; and one with no network, deployed, is reached but reaches nothing, comes back from a reset
fresh, and leaves nothing once torn down. Needs Docker and the local registry on localhost:5000."""

import asyncio
import platform
import subprocess
import time
import uuid

import pytest

from agentenv_protocol.computer import COMPUTER_READY
from agentenv_protocol.computer.conformance import run_probe
from agentenv_protocol.computer.conformance.client import Driver

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.env import Env
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from tst.integration.computer.conftest import docker

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
PORT = 18765


def test_each_arch_this_machine_builds_is_composed_pushed_and_written_as_the_envs_computer(desktop):
    results = desktop.metadata["composed"]["container"]
    with namespace_routing():
        computer = ComputerArtifact.get(desktop.metadata["computer"]["id"], desktop.metadata["computer"]["version"])

    assert results[ARCH]["status"] == "ready" and results[ARCH]["emulated"] is False
    assert results[ARCH]["probe"]["passed"] and results[ARCH]["probe"]["level"] == "full"
    assert results[ARCH]["card"]["name"] == desktop.environment_name
    assert {"mss", "python-xlib", "av", "agentenv-framework-protocol"} <= set(results[ARCH]["installed"])
    for arch, result in results.items():
        if arch != ARCH:
            assert result["status"] == "unverified" or "its Docker has no emulation for it" in result["reason"], result
    built = sorted(arch for arch, result in results.items() if result["status"] in ("ready", "unverified"))
    assert sorted(computer.arch) == built
    assert {arch: result["status"] for arch, result in computer.metadata["validated_images"]["container"].items()} == {
        arch: "passed" for arch in built}
    assert computer.metadata["validated_boot"]["container"][ARCH] == results[ARCH]["probe"]


@pytest.mark.asyncio
async def test_the_image_of_this_machines_arch_boots_as_the_session_user_and_passes_the_probe_at_full(desktop):
    image = desktop.metadata["composed"]["container"][ARCH]["image"]
    name = f"agentenv-test-computer-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, "-p", f"127.0.0.1::{PORT}", image)
    try:
        url = await _ready(name)
        reports = [await run_probe(url, "full") for _ in range(3)]
        users = set(subprocess.run(["docker", "exec", name, "ps", "-eo", "user="], capture_output=True, text=True,
                                   check=True).stdout.split())
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    for report in reports:
        assert report.passed, [f"{c.check} {c.name}: {c.detail}" for c in report.checks if c.status == "failed"]
    assert users == {"user"}


def test_an_env_that_changes_only_its_display_and_name_reuses_the_image(desktop):
    with namespace_routing():
        started = time.monotonic()
        other = ComputerEnv.put(id=f"{desktop.id}-other", computer=desktop.computer, driver=desktop.driver,
                                environment_name="other", display={"width": 1024, "height": 768})
        took = time.monotonic() - started
        results = Env.get(other.id).metadata["composed"]["container"]
    first = desktop.metadata["composed"]["container"]

    assert results[ARCH]["reused"] is True and results[ARCH]["image"] == first[ARCH]["image"]
    assert results[ARCH]["status"] == "ready" and results[ARCH]["card"]["name"] == "other"
    assert results[ARCH]["installed"] == first[ARCH]["installed"] and results[ARCH]["emulated"] is False
    assert took < 60, took


@pytest.mark.asyncio
async def test_with_no_network_it_is_reached_reaches_nothing_comes_back_fresh_and_leaves_nothing(desktop):
    with namespace_routing():
        closed = ComputerEnv.put(id=f"{desktop.id}-closed", computer=desktop.computer, driver=desktop.driver,
                                 environment_name="closed", network={"mode": "allowlist"})
        deployed = await Env.get(closed.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            driver = await Driver.connect(deployed.environment_url, 30)
            egress = await driver.run("python3", "-c", "import urllib.request; urllib.request.urlopen('http://1.1.1.1', timeout=5)")
            await driver.run("sh", "-c", "echo x > Desktop/left.txt")
            await ResetEnvTaskStep(id="reset", version=None, env_id=closed.id, fail_task_on_error=True).execute(context)
            left = await (await Driver.connect(deployed.environment_url, 30)).run("test", "-e", "Desktop/left.txt")
        finally:
            report = await teardown_run(context)
    sandbox = deployed.sandbox_id

    assert egress["exit_code"] != 0 and "Network is unreachable" in egress["stderr"], egress
    assert left["exit_code"] != 0
    assert [s.sandbox_id for s in report.terminated] == [sandbox] and not report.failed
    for listing in (["ps", "-aq", "--filter", f"label=agentenv.sandbox={sandbox}"],
                    ["ps", "-aq", "--filter", f"name=agent-{sandbox}"],
                    ["network", "ls", "-q", "--filter", f"label=agentenv.sandbox={sandbox}"]):
        assert not docker(*listing).stdout.strip(), listing
    assert LocalSandbox.find_work_dir(sandbox) is None


async def _ready(name: str) -> str:
    for _ in range(240):
        port = subprocess.run(["docker", "port", name, str(PORT)], capture_output=True, text=True).stdout
        if port:
            url = f"http://{port.splitlines()[0]}"
            try:
                if (await (await Driver.connect(url, 10)).invoke(COMPUTER_READY, "status"))["ready"]:
                    return url
            except Exception:  # not serving yet
                pass
        await asyncio.sleep(0.25)
    raise TimeoutError(f"{name}'s computer never became ready")
