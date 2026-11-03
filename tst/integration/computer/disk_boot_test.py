"""The shipped desktop example's disk form, composed on this machine and booted under KVM in the QEMU launcher through
``agent-env run``. A run on local Docker, whose deploys boot the container, composes no disk; a later run whose deploy
names the disk form composes it first, reusing the container, and gate 3 probes this machine's arch's disk at full as
it composes it, or reuses one of the same key an earlier compose made; the deploy boots it with its settings and control
token in its seed, it passes the probe at full, and a reset boots it afresh, where it passes again. Needs Docker, the
local registry and /dev/kvm."""

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.env import Env
from agent_env.providers.sandbox_providers.host_images import host_platform
from agent_env.store.routing import namespace_routing
from tst.integration.computer.conftest import run_context
from tst.util.capabilities import skip_without_kvm

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow, skip_without_kvm()]

BUNDLE = Path(__file__).parent / "disk"
DESKTOP = "@local/agentenv-framework/desktop/desktop"


def test_the_desktops_disk_is_composed_by_the_first_deploy_that_boots_it_then_probed_and_reset_under_kvm(state):
    arch = host_platform().rsplit("/", 1)[1]
    first = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "container"])
    assert first.exit_code == 0, first.output
    with namespace_routing():
        composed = Env.get(DESKTOP).metadata["composed"]
    assert "disk" not in composed and composed["container"][arch]["status"] == "ready"

    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "probe"])

    assert ran.exit_code == 0, ran.output
    assert (f"[tasks/probe.json] {DESKTOP}: composing disk/amd64, disk/arm64 before this deploy on LocalSandboxProvider "
            "(no compose built them") in ran.output
    assert re.search(rf"{DESKTOP}: disk/{arch}: (making the disk|reusing the disk of the same key)", ran.output)
    assert re.search(rf"{DESKTOP}: disk/{arch}: (reused, )?ready \(", ran.output)
    context = run_context(ran.output)
    verifications = context["metadata"]["verifications"]
    assert all(result["result"] for step in ("probe", "again") for result in verifications[step]["results"])
    (deployed,) = context["deployed_envs"]
    assert (deployed["metadata"]["computer"]["form"], deployed["metadata"]["computer"]["arch"]) == ("disk", arch)
    with namespace_routing():
        after = Env.get(DESKTOP).metadata["composed"]
    disk = after["disk"][arch]
    assert (disk["status"], disk["probe"]["passed"], disk["probe"]["level"]) == ("ready", True, "full")
    assert after["container"][arch] == composed["container"][arch]
