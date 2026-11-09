"""A computer app whose window stays above every other window over the middle of the screen, as a dialog an app keeps
on top does, run through ``agent-env run``: gate 3's probe passes every check, its clicks reaching the conformance
fixture rather than that window, and the computer deploys. Needs Docker and the local registry on localhost:5000."""

import platform
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from agent_env.env import Env
from agent_env.store.routing import namespace_routing
from tst.integration.computer.conftest import run_context

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

COVERED = Path(__file__).parent / "covered"
ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]


def test_gate_3_passes_a_computer_whose_app_keeps_a_window_above_the_fixtures_grid(state):
    ran = CliRunner().invoke(cli, ["run", str(COVERED), "--task", "covered", "--no-emulate"])

    assert ran.exit_code == 0, ran.output
    (deployed,) = run_context(ran.output)["deployed_envs"]
    with namespace_routing():
        build = Env.get(deployed["env_id"]).metadata["composed"]["container"][ARCH]
    assert build["status"] == "ready", build.get("reason")
    checks = build["probe"]["checks"]
    # A fixture that never started leaves its checks unverified rather than failed, so they must have run and passed.
    assert build["probe"]["passed"] and checks["failed"] == 0 and checks["unverified"] == 0 and checks["passed"] > 0
