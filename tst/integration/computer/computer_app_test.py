"""A computer app deployed alone through ``agent-env run``: a one-app computer of the computer and driver its env.toml
names, its card named after it, its service running from its folder in the venv its own server runs in, on the
settings the deploy gave it, a secret resolved at boot, its window launched once by its server as the desktop session
comes up, and the computer ready only once its server says its window and its @ready_check are; its apt packages,
one from a PPA, its download, run script and install step on its build setting, fetched at put, installed offline.
And two apps composed into one computer env, each running on its own settings from its own folder, the files their run
scripts wrote compared before the push of this machine's arch, or the image of the same key reused, as one an earlier
run pushed is. Needs Docker and the local registry on localhost:5000."""

import platform
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from tst.integration.computer.conftest import run_context

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

APP = Path(__file__).parent / "app"
ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]


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
        "signature": "-- Ada", "token": "t0ken", "edition": "pro", "font": "DejaVu Sans", "card": "notes",
        "app_ready": [{"name": "app notes", "ok": True, "detail": "ready"}],
        "launchers": ["agentenv-app-notes.desktop"], "launched": ["notes"], "hello": "Hello, world!", "license": True,
        "installed": "pro"}


def test_two_apps_composed_into_one_computer_env_each_run_on_their_own_settings(state, monkeypatch):
    monkeypatch.setenv("NOTES_TOKEN", "t0ken")

    ran = CliRunner().invoke(cli, ["run", str(APP), "--task", "together"])

    assert ran.exit_code == 0, ran.output
    context = run_context(ran.output)
    (deployed,) = context["deployed_envs"]
    assert deployed["env_id"].endswith("/desk") and deployed["environment_card"]["name"] == "desk"
    assert context["metadata"]["script_results"]["together"] == {
        "notes": {"signature": "-- Ada", "token": "t0ken"},
        "clock": {"zone": "Europe/Paris", "served": {"zone": "Europe/Paris", "known": True}}, "card": "desk",
        "app_ready": ["app clock", "app notes"], "launched": ["notes"], "folders": ["clock", "notes"], "hello": True,
        "paris": True}
    said = [line for line in ran.output.splitlines() if line.startswith(f"envs/desk: container/{ARCH}: ")]
    assert any(line.endswith(("comparing the files its parts' run scripts and install steps wrote",
                              "reusing the image of the same key, built earlier")) for line in said), said
