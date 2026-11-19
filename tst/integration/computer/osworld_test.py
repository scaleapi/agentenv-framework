"""OSWorld on a computer env, end to end, from public parts alone: the osworld bundle's desk, the ubuntu-24.04 base with
the reference driver and the shipped osworld-server app, which runs upstream OSWorld's guest server, deployed;
osworld_initialize running a classic init_config's execute on the computer with upstream's own SetupController, in this
process, the worker, reaching the guest server only through the endpoint the computer's card advertises, with the env's
control token; and osworld_evaluate grading what that wrote by upstream's own checks, there too: check_include_exclude
over a vm_command_line getter, 1.0 for a rule it meets and 0.0 for one it doesn't, and file_contains over a vm_file
getter, the file fetched from the computer. No grader runs in a sandbox of its own."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from tst.integration.computer.conftest import run_context

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

OSWORLD = Path(__file__).parent / "osworld"


def test_a_classic_task_is_set_up_and_graded_on_a_computer_by_upstreams_checks_in_the_worker(state):
    ran = CliRunner().invoke(cli, ["run", str(OSWORLD), "--task", "classic", "--no-emulate"])

    assert ran.exit_code == 0, ran.output
    context = run_context(ran.output)
    verifications = context["metadata"]["verifications"]
    assert verifications["pass"]["score"] == 1.0, verifications["pass"]
    assert verifications["fail"]["score"] == 0.0, verifications["fail"]
    assert verifications["file"]["score"] == 1.0, verifications["file"]
    # The computer is the only sandbox: grading ran here.
    sandboxes = [s.get("sandbox_name") or "" for s in context.get("deployed_sandboxes") or []]
    assert not [name for name in sandboxes if "osworld-controller" in name], sandboxes
