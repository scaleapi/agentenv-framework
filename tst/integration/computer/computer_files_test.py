"""load_artifact with ``computer``, end to end: a multi env of one ubuntu-24.04 computer with the reference driver
deployed, a file artifact universe loaded into that computer by its name, its files at their paths under the session
user's home, as an imported CUA task's download setup brings them, and read there by computer_exec."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from tst.integration.computer.conftest import run_context

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

BUNDLE = Path(__file__).parent / "computer_files"


def test_files_load_into_a_multi_envs_computer_by_name(state):
    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "load", "--no-emulate"])

    assert ran.exit_code == 0, ran.output
    metadata = run_context(ran.output)["metadata"]
    [loaded] = metadata["loaded_file_artifact_universes"]
    assert loaded["computer"] == "desk" and loaded["files"] == {
        "Desktop/a.csv": "Desktop/a.csv", "notes/b.txt": "notes/b.txt"}, loaded
    read = metadata["computer_exec"]["read"]
    assert read["exit_code"] == 0 and read["stdout"] == "sheet\nnote\n", read
