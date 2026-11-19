"""Files into a multi env's computer, end to end: a multi env of one ubuntu-24.04 computer with the reference driver
deployed, an environment artifact named after that computer loaded into the multi env, as an imported CUA task's
download setup brings it, its files at their paths under the computer's session user's home, and read there by
computer_exec."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from tst.integration.computer.conftest import run_context

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

BUNDLE = Path(__file__).parent / "computer_files"


def test_an_environment_named_after_a_multi_envs_computer_loads_into_it(state):
    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "load", "--no-emulate"])

    assert ran.exit_code == 0, ran.output
    read = run_context(ran.output)["metadata"]["computer_exec"]["read"]
    assert read["exit_code"] == 0 and read["stdout"] == "sheet\nnote\n", read
