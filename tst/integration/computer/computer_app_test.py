"""A computer app deployed alone through ``agent-env run``: a one-app computer of the computer and driver its env.toml
names, its card named after it, its service running from its folder in the venv its own server runs in, on the
settings the deploy gave it, a secret resolved at boot, its window launched once by its server as the desktop session
comes up, and the computer ready only once its server says its window and its @ready_check are; its apt packages,
one from a PPA, its download, run script and install step on its build setting, fetched at put, installed offline.
Needs Docker and the local registry on localhost:5000."""

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
    service = context["metadata"]["script_results"]["service"]
    version, fastfetch = service.pop("fastfetch")
    assert version.endswith("~noble") and fastfetch.startswith("fastfetch ")
    assert service == {
        "signature": "-- Ada", "token": "t0ken", "edition": "pro", "card": "notes",
        "app_ready": [{"name": "app notes", "ok": True, "detail": "ready"}],
        "launchers": ["agentenv-app-notes.desktop"], "launched": ["notes"], "hello": "Hello, world!", "license": True,
        "installed": "pro"}
