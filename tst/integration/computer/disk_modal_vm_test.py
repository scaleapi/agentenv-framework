"""The shipped desktop example's disk form on a Modal VM sandbox through ``agent-env run --sandbox modal_vm`` with the
default local stores, as M4.5's exit runs it: composed with this machine's Docker, exported once, compressed, and
mounted on a VM sandbox whose image is the QEMU launcher, booted under nested virtualization, probed at full, reset in
place and probed again; a second run composes and exports no amd64 disk. Needs Docker, the local registry and Modal
credentials, with ``[sandbox.attribution]`` tags for Modal's apps."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from tst.integration.computer.conftest import run_context
from tst.util.capabilities import skip_without_remote_sandbox

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow, skip_without_remote_sandbox("modal_vm")]

BUNDLE = Path(__file__).parent / "disk"


def test_the_desktops_disk_is_carried_to_a_modal_vm_once_then_booted_probed_and_reset_in_place_there(state,
                                                                                                      monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(state / "cache"))

    first = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "probe", "--sandbox", "modal_vm"])
    assert first.exit_code == 0, first.output
    (exported,) = (state / "cache" / "agentenv" / "disks").glob("*.img.zst")
    made = exported.stat().st_mtime_ns
    again = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "probe", "--sandbox", "modal_vm"])

    assert again.exit_code == 0, again.output
    assert any("composing" in line and "disk/amd64" in line and line.endswith(" on LocalSandboxProvider")
               for line in first.output.splitlines())
    assert not any("composing" in line and "disk/amd64" in line for line in again.output.splitlines())
    assert list(exported.parent.glob("*.img.zst")) == [exported] and exported.stat().st_mtime_ns == made
    for ran in (first, again):
        context = run_context(ran.output)
        verifications = context["metadata"]["verifications"]
        assert all(result["result"] for step in ("probe", "again") for result in verifications[step]["results"])
        (deployed,) = context["deployed_envs"]
        assert (deployed["sandbox_type"], deployed["metadata"]["computer"]["form"]) == ("modal_vm", "disk")
        assert deployed["environment_card_url"].startswith("https://")
