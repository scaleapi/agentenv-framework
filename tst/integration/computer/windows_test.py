"""The Windows test bundle through ``agent-env run``, on local Docker with KVM: its desktop, cua-driver on windows-11,
composed on this machine, its base installed from the evaluation ISO in this machine's cache unless one is held here,
passes the conformance probe at full, the checks that need computer-exec unverified; and an agent drives Notepad on it
through cua-driver's tools. Needs Docker, KVM and the ISO; the agent's task, a model endpoint too."""

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from tst.util.capabilities import skip_without_model_endpoint, skip_without_windows_media

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow, skip_without_windows_media()]

BUNDLE = "tst/integration/computer/windows"


def test_the_windows_desktop_passes_the_probe_at_full(state):
    ran = _run("probe")

    assert ran.exit_code == 0, ran.output
    assert "passed (probe: 1)" in ran.output, ran.output


@skip_without_model_endpoint()
def test_an_agent_saves_a_line_in_notepad_through_cua_drivers_tools(state):
    ran = _run("notepad")

    assert ran.exit_code == 0, ran.output
    assert "passed (notepad: 1)" in ran.output, ran.output


def _run(task: str):
    return CliRunner().invoke(cli, ["run", BUNDLE, "--task", task])
