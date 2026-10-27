"""The reference driver on a desktop of its own before agent-env can compose a computer: the scaffold, a hand-written
Ubuntu 24.04 image under this folder, deployed by ``agent-env run`` as an MCP server on the server provider, passes
the conformance probe at full three runs in a row, and an agent drives its desktop through the computer_* tools.
Needs a Docker daemon and the local registry; the agent's task, a model endpoint too."""

import logging
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

import agent_env
import agentenv_protocol
from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from tst.util.capabilities import skip_without_model_endpoint

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

SCAFFOLD = Path(__file__).parent / "scaffold"
DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "reference" / "driver.py"
FIXTURE = Path(agentenv_protocol.__file__).parent / "computer" / "conformance" / "fixture.py"
PROTOCOL = Path(agentenv_protocol.__file__).parents[2]
PROTOCOL_FILES = ("pyproject.toml", "README.md", "LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md")


@pytest.fixture
def state(monkeypatch, tmp_path):
    """Local stores under this test's folder. Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    logging.disable(logging.CRITICAL)
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        reset_artifact_store()
        reset_config()


def _scaffold(root: Path) -> Path:
    """The scaffold bundle with the reference driver, the probe's fixture and this checkout's protocol beside its
    Dockerfile."""
    shutil.copytree(SCAFFOLD, root, ignore=shutil.ignore_patterns("__pycache__"))
    desktop = root / "envs" / "desktop"
    shutil.copy(DRIVER, desktop / "driver.py")
    shutil.copy(FIXTURE, desktop / "fixture.py")
    shutil.copytree(PROTOCOL / "src", desktop / "protocol" / "src", ignore=shutil.ignore_patterns("__pycache__"))
    for name in PROTOCOL_FILES:
        shutil.copy(PROTOCOL / name, desktop / "protocol" / name)
    return root


def _run(root: Path, task: str):
    return CliRunner().invoke(cli, ["run", str(root), "--task", task])


def test_the_reference_driver_passes_the_probe_at_full(state):
    ran = _run(_scaffold(state / "scaffold"), "probe")

    assert ran.exit_code == 0, ran.output
    assert "passed (probe: 1)" in ran.output, ran.output


@skip_without_model_endpoint()
def test_an_agent_drives_the_reference_drivers_desktop(state):
    ran = _run(_scaffold(state / "scaffold"), "agent")

    assert ran.exit_code == 0, ran.output
    assert "passed (check: 1)" in ran.output, ran.output
