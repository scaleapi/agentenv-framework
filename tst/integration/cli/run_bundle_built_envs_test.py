"""``agent-env run`` on a bundle whose MCP server env is built from its folder, with real docker builds and the env
deployed on the local sandbox: named by the environment card in its source, reused while its folder is unchanged, and
rebuilt when it changes. Needs a Docker daemon and the local registry."""

import json
import logging
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import parse_bundle
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.env import Env
from agent_env.store.routing import namespace_routing

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

REPO = Path(__file__).resolve().parents[3]
ITEMS = REPO / "tst" / "data" / "agentenv_mcp"  # an in-memory MCP server whose card names it 'items'
PROTOCOL = REPO / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol"
IMAGE = "envs/items (Dockerfile image)"


@pytest.fixture
def state(monkeypatch, tmp_path):
    """Local stores under this test's folder. Whatever ran, no sandbox work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    # HOME stays: docker's credential helper (the macOS keychain, for one) can hang a build's base-image lookup
    # when HOME moves.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        reset_artifact_store()
        reset_config()


def _bundle(root):
    """The items server as an env folder, deployed by the server provider, which runs it with no gateway."""
    folder = root / "envs" / "items"
    shutil.copytree(PROTOCOL, folder / "agentenv_protocol", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("server.py", "Dockerfile", "seed.json"):
        shutil.copy(ITEMS / name, folder / name)
    (folder / "env.toml").write_text('env_provider_type = "server"\n')
    (root / "tasks").mkdir()
    (root / "tasks/items.json").write_text(json.dumps(
        [{"id": "env", "type": "deploy_env", "env_id": "items", "sandbox_type": "local"}]))
    return root


def _run(root):
    return CliRunner().invoke(cli, ["run", str(root)])


def test_an_env_folder_is_built_named_by_its_card_deployed_and_rebuilt_only_when_it_changes(state):
    root = _bundle(state / "envs-bundle")

    first = _run(root)

    assert first.exit_code == 0, first.output
    assert f"{IMAGE}: building with docker" in first.output, first.output
    assert f"{IMAGE}: v1 (new)" in first.output and "envs/items: v1 (new)" in first.output, first.output
    assert "tasks/items.json v1: unscored" in first.output, first.output
    with namespace_routing():
        env = Env.get(next(entry.id for entry in parse_bundle(root).entries if entry.name == "items"))
    assert (env.environment_name, env.env_provider_type) == ("items", "server")

    again = _run(root)

    assert again.exit_code == 0, again.output
    assert "building with docker" not in again.output, again.output
    assert f"{IMAGE}: v1, unchanged" in again.output and "envs/items: v1, unchanged" in again.output, again.output

    (root / "envs/items/seed.json").write_text('{"items": ["edited"]}\n')
    edited = _run(root)

    assert edited.exit_code == 0, edited.output
    assert f"{IMAGE}: v2 (files changed: seed.json)" in edited.output, edited.output
    assert "envs/items: v2 (" in edited.output and "tasks/items.json v1: unscored" in edited.output, edited.output
    assert edited.output.count("building with docker") == 1, edited.output
