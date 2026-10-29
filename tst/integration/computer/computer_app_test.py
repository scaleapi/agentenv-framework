"""A computer app deployed alone through ``agent-env run``: a one-app computer of the computer and driver its env.toml
names, its card named after it, its service running from its folder in a venv of the plain protocol package, on the
settings the deploy gave it, a secret resolved at boot, its window opened with the desktop session and the computer ready
only once its window and its @ready_check say so. Needs Docker and the local registry on localhost:5000."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from tst.integration.computer.conftest import run_context

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

APP = Path(__file__).parent / "app"


def test_an_app_deployed_alone_runs_its_service_on_its_settings_on_a_computer_named_after_it(state, monkeypatch):
    monkeypatch.setenv("NOTES_TOKEN", "t0ken")

    ran = CliRunner().invoke(cli, ["run", str(APP), "--task", "alone"])

    assert ran.exit_code == 0, ran.output
    context = run_context(ran.output)
    (deployed,) = context["deployed_envs"]
    assert deployed["env_id"].endswith("/notes") and deployed["environment_card"]["name"] == "notes"
    assert context["metadata"]["script_results"]["service"] == {
        "signature": "-- Ada", "token": "t0ken", "card": "notes",
        "app_ready": [{"name": "app notes", "ok": True, "detail": "ready"}],
        "launchers": ["agentenv-app-notes.desktop"], "autostart": ["agentenv-app-notes.desktop"]}
