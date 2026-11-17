"""The Omarchy test bundle through ``agent-env run``, on local Docker with KVM: its desktop, cua-driver on omarchy-4.5,
composed on this machine, its base built from Omarchy's image in this machine's cache unless one is held here, passes
the conformance probe at full. And the bundle's notes app, composed onto it as a layer, runs on its deploy's settings,
its window up and its install done as its Omarchy install says, its own folder the session user's alone, before and
after a reset. Needs Docker, KVM and the image."""

from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.cli import cli
from tst.util.capabilities import skip_without_omarchy_image

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow, skip_without_omarchy_image()]

BUNDLE = Path(__file__).parent / "omarchy"


def test_the_omarchy_desktop_passes_the_probe_at_full(state):
    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "probe"])

    assert ran.exit_code == 0, ran.output
    assert "passed (probe: 1)" in ran.output, ran.output


def test_the_notes_app_composed_onto_omarchy_runs_as_installed_before_and_after_a_reset(state, monkeypatch):
    monkeypatch.setenv("NOTES_TOKEN", "t0ken")

    ran = CliRunner().invoke(cli, ["run", str(BUNDLE), "--task", "notes"])

    assert ran.exit_code == 0, ran.output
    assert "passed (before: 1, after: 1)" in ran.output, ran.output
