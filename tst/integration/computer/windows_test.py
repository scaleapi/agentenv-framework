"""The Windows test bundle through ``agent-env run``, on local Docker with KVM: its desktop, cua-driver on windows-11,
composed on this machine, its base installed from the evaluation ISO in this machine's cache unless one is held here,
passes the conformance probe at full, cua-driver's known failure on Windows waived, before and after a reset in place;
and an agent drives Notepad on it through cua-driver's tools. And the bundle's notes app, composed onto it as a layer,
runs on its deploy's settings, its window up and its install done as its Windows install says, its own folder and
server the session user's alone, before and after a reset; on it the protocol's Windows session launches, pauses
and stops apps, and all they start, as an app's server does; and run_code runs a script inside it. Needs Docker, KVM and the ISO; the agent's task, a model
endpoint too."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from tst.util.capabilities import skip_without_model_endpoint, skip_without_windows_media

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow, skip_without_windows_media()]

BUNDLE = Path(__file__).parent / "windows"


def test_the_windows_desktop_passes_the_probe_at_full(state):
    ran = _run("probe")

    assert ran.exit_code == 0, ran.output
    assert "passed (probe: 1)" in ran.output, ran.output


def test_a_reset_boots_the_windows_desktop_afresh_and_it_passes_the_probe_again(state):
    ran = _run("reset")

    assert ran.exit_code == 0, ran.output
    assert "passed (before: 1, after: 1)" in ran.output, ran.output


@skip_without_model_endpoint()
def test_an_agent_saves_a_line_in_notepad_through_cua_drivers_tools(state):
    ran = _run("notepad")

    assert ran.exit_code == 0, ran.output
    assert "passed (check: 1)" in ran.output, ran.output


def test_the_notes_app_composed_onto_windows_runs_as_installed_and_the_session_runs_apps_in_their_jobs(state,
                                                                                                       monkeypatch):
    monkeypatch.setenv("NOTES_TOKEN", "t0ken")

    ran = _run("notes")

    assert ran.exit_code == 0, ran.output
    assert "passed (before: 1, session: 1, after: 1)" in ran.output, ran.output


def _run(task: str):
    return CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", task])
