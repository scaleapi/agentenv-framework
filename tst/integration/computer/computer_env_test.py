"""A computer env composed on this machine: a bundle's env naming the shipped ubuntu-24.04 and reference by alias builds
an image for each arch this machine's Docker can build, pushed to the local registry and written as the env's computer;
the image of this machine's arch boots with everything running as the session user and passes the conformance probe
at full three times in a row; and another env that changes only what doesn't change the image reuses it. Needs Docker
and the local registry on localhost:5000."""

import asyncio
import json
import platform
import subprocess
import time
import uuid

import pytest

from agentenv_protocol.computer import COMPUTER_READY
from agentenv_protocol.computer.conformance import run_probe
from agentenv_protocol.computer.conformance.client import Driver

from agent_env.artifact.artifacts.computer import ComputerArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import parse_bundle
from agent_env.bundle.installed import installed_bundles
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import BuiltImage, resolve_bundle
from agent_env.config import configure, reset_config
from agent_env.env import Env
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.store.routing import namespace_routing
from tst.integration.computer.conftest import docker

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
PORT = 18765
DESKTOP = 'type = "computer"\ncomputer = "ubuntu-24.04"\ndriver = "reference"\n'


@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    """The bundle's computer env, composed once for the module's tests, in stores of the module's own."""
    root = tmp_path_factory.mktemp("composed")
    patch = pytest.MonkeyPatch()
    patch.chdir(root)
    patch.setenv("XDG_STATE_HOME", str(root / "state"))
    patch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    patch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    configure()
    reset_artifact_store()
    bundle = root / "desk"
    for path, text in {"envs/desktop/env.toml": DESKTOP, "artifacts/notes/notes.txt": "a note\n", "tasks/t.json": json.dumps(
            [{"id": "load", "type": "load_artifact", "env_id": "desktop", "artifact_id": "notes"}])}.items():
        (bundle / path).parent.mkdir(parents=True, exist_ok=True)
        (bundle / path).write_text(text)
    with namespace_routing():
        plan = plan_bundle(resolve_bundle(parse_bundle(bundle)), installed=installed_bundles)
        done = materialize(plan, build_here=frozenset(w.id for w in plan.writes if isinstance(w.source, BuiltImage)))
        env = next(Env.get(item.write.id) for item in done.writes if item.write.id.endswith("/desktop"))
        yield env
    reset_artifact_store()
    reset_config()
    patch.undo()


def test_each_arch_this_machine_builds_is_composed_pushed_and_written_as_the_envs_computer(composed):
    results = composed.metadata["composed"]["container"]
    with namespace_routing():
        computer = ComputerArtifact.get(composed.metadata["computer"]["id"], composed.metadata["computer"]["version"])

    assert results[ARCH]["status"] == "unverified" and results[ARCH]["emulated"] is False
    assert {"mss", "python-xlib", "av", "agentenv-framework-protocol"} <= set(results[ARCH]["installed"])
    for arch, result in results.items():
        assert result["status"] == "unverified" or "its Docker has no emulation for it" in result["reason"], result
    built = sorted(arch for arch, result in results.items() if result["status"] == "unverified")
    assert sorted(computer.arch) == built
    assert {arch: result["status"] for arch, result in computer.metadata["validated_images"]["container"].items()} == {
        arch: "passed" for arch in built}


@pytest.mark.asyncio
async def test_the_image_of_this_machines_arch_boots_as_the_session_user_and_passes_the_probe_at_full(composed):
    image = composed.metadata["composed"]["container"][ARCH]["image"]
    name = f"agentenv-test-computer-{uuid.uuid4().hex[:8]}"
    docker("run", "-d", "--name", name, "-p", f"127.0.0.1::{PORT}",
           "-e", 'AGENTENV_COMPUTER_SETTINGS={"data_paths": ["fixture"]}', image)
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


def test_an_env_that_changes_only_its_display_and_name_reuses_the_image(composed):
    with namespace_routing():
        started = time.monotonic()
        other = ComputerEnv.put(id=f"{composed.id}-other", computer=composed.base(), driver=composed.driver_env(),
                                environment_name="other", display={"width": 1024, "height": 768})
        took = time.monotonic() - started
        results = Env.get(other.id).metadata["composed"]["container"]
    first = composed.metadata["composed"]["container"]

    assert results[ARCH]["reused"] is True and results[ARCH]["image"] == first[ARCH]["image"]
    assert took < 60, took


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
