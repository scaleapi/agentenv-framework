"""The shipped desktop example's disk form, composed on this machine and booted under KVM in the QEMU launcher through
``agent-env run``: gate 3 probes this machine's arch's disk at full as it composes it, a deploy that names the disk
form boots it with its settings and control token in its seed, it passes the probe at full, and a reset boots it
afresh, where it passes again. Needs Docker, the local registry and /dev/kvm."""

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


def test_the_desktops_disk_is_probed_as_composed_then_deployed_probed_and_reset_under_kvm(state):
    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "probe"])

    assert ran.exit_code == 0, ran.output
    arch = host_platform().rsplit("/", 1)[1]
    context = run_context(ran.output)
    verifications = context["metadata"]["verifications"]
    assert all(result["result"] for step in ("probe", "again") for result in verifications[step]["results"])
    (deployed,) = context["deployed_envs"]
    assert (deployed["metadata"]["computer"]["form"], deployed["metadata"]["computer"]["arch"]) == ("disk", arch)
    with namespace_routing():
        disk = Env.get(DESKTOP).metadata["composed"]["disk"][arch]
    assert (disk["status"], disk["probe"]["passed"], disk["probe"]["level"]) == ("ready", True, "full")
