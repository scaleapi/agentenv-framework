"""computer_exec on a computer env, end to end: the ubuntu-24.04 base with the reference driver deployed, a command run
in the session user's home to its end, its output recorded; Thunar started with ``background`` on the folder that wrote,
as a task opens a file in an app for the agent, and its window then found on the display, so the app outlived the run
that started it; a failing command, and a background one whose program isn't there, recorded, not failing the task,
with ``check`` off; and a ``system`` command run as root in the session user's home, which the session user can't do."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from tst.integration.computer.conftest import run_context

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

BUNDLE = Path(__file__).parent / "computer_exec"


def test_commands_run_in_a_computer_and_an_app_started_in_the_background_stays_up(state):
    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "setup", "--no-emulate"])

    assert ran.exit_code == 0, ran.output
    records = run_context(ran.output)["metadata"]["computer_exec"]
    assert records["write"]["exit_code"] == 0 and records["write"]["stdout"] == "computer-exec\n", records["write"]
    assert records["launch"]["exit_code"] == 0 and records["launch"]["background"], records["launch"]
    assert records["window"]["exit_code"] == 0 and "thunar" in records["window"]["stdout"].lower(), records["window"]
    assert records["tolerated"]["exit_code"] == 7 and records["tolerated"]["stderr"] == "nope\n", records["tolerated"]
    assert records["missing"]["exit_code"] == 127 and "not found" in records["missing"]["stderr"], records["missing"]
    # A string command's program is sh, so it's caught when sh exits right after starting; so is a program given a bad
    # flag. (A second thunar would exit 0: it hands off to the one running, which counts as started.)
    assert records["missing-string"]["exit_code"] == 127, records["missing-string"]
    assert "right after starting" in records["missing-string"]["stderr"], records["missing-string"]
    assert records["bad-flag"]["exit_code"] not in (0, None), records["bad-flag"]
    assert "right after starting" in records["bad-flag"]["stderr"], records["bad-flag"]
    uid, home, tilde = records["root"]["stdout"].split()
    assert records["root"]["exit_code"] == 0 and uid == "0" and home == tilde != "/root", records["root"]
    assert records["owner"]["stdout"] == "root\n", records["owner"]
    assert records["denied"]["exit_code"] != 0 and "Permission denied" in records["denied"]["stderr"], records["denied"]
