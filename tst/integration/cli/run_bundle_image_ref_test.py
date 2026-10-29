"""``agent-env run`` on a bundle whose agent folder names a registry image with ``image = { ref = "<image>" }`` and
carries no Dockerfile: the echo agent, pushed to this machine's local registry, is pinned to its digest, deployed on
the local sandbox and answers. A rerun reuses the image, and a ref spelled differently writes it again. Needs a
Docker daemon and the local registry."""

import json
import logging

import pytest
from click.testing import CliRunner

from agent_env.artifact import DockerImageArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import parse_bundle
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.store.ids import derive_id
from agent_env.store.object_store.local.grant_server import grant_server
from agent_env.store.routing import namespace_routing
from tst.util.a2a_test_agent import build_test_agent_image

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

NO_MODEL = '[default_env_vars]\nLITELLM_API_KEY = "unused"\nLITELLM_BASE_URL = "http://unused.invalid"\n'
TASK = json.dumps([
    {"id": "deploy", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "echo", "sandbox_type": "local"},
    {"id": "ask", "type": "prompt_agent", "prompt": "hello from a ref", "prompt_id": "ask", "depends_on": ["deploy"]},
    {"id": "check", "type": "agent_prompt_response_verifier", "prompt_id": "ask", "verifier_id": "reply",
     "criteria": [{"type": "response_contains", "needles": ["Echo: hello from a ref"]}], "depends_on": ["ask"]},
])


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
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        # The agent moved its trajectory through this process's grant server, which serves a certificate from this
        # test's state root; a later test's agents trust another root's CA, so it must start afresh.
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


def _bundle(root, ref):
    (root / "agents/echo").mkdir(parents=True, exist_ok=True)
    (root / "agents/echo/agent.toml").write_text(f'image = {{ ref = "{ref}" }}\n{NO_MODEL}')
    (root / "tasks").mkdir(exist_ok=True)
    (root / "tasks/echo.json").write_text(TASK)
    return root


def _run(root):
    return CliRunner().invoke(cli, ["run", str(root)])


def test_an_agent_folder_naming_a_registry_image_runs_it_and_writes_it_again_only_when_the_ref_changes(state):
    ref = build_test_agent_image("bundle-image-ref-echo").image_name
    assert ref.startswith("localhost:"), ref
    root = _bundle(state / "ref-bundle", ref)
    label = f"agents/echo ({ref} image)"

    first = _run(root)

    assert first.exit_code == 0, first.output
    assert f"{label}: v1 (new)" in first.output, first.output
    assert "building with docker" not in first.output, first.output
    assert "tasks/echo.json v1: passed" in first.output, first.output

    again = _run(root)

    assert again.exit_code == 0, again.output
    assert f"{label}: v1, unchanged" in again.output, again.output
    assert "tasks/echo.json v1: passed" in again.output, again.output

    agent = next(entry for entry in parse_bundle(root).entries if entry.name == "echo")
    with namespace_routing():
        pinned = DockerImageArtifact.get(derive_id(agent.id, "agent_image")).image_name
    name, digest = pinned.rsplit("@", 1)
    by_digest = f"{name.rsplit(':', 1)[0]}@{digest}"
    _bundle(root, by_digest)

    third = _run(root)

    assert third.exit_code == 0, third.output
    assert f"agents/echo ({by_digest} image): v2" in third.output, third.output
    assert "tasks/echo.json v1: passed" in third.output, third.output
