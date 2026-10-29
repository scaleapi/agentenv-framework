"""``agent-env run`` on a bundle whose task attaches one of its skills to an agent: the echo agent, built from its
folder and deployed on the local sandbox, gets the skill's files through read grants. A rerun reuses the skill, and
an edit rewrites it. Needs a Docker daemon and the local registry."""

import hashlib
import json
import logging
import shutil

import pytest
from click.testing import CliRunner

from agent_env.artifact.store import reset_artifact_store
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.store.object_store.local.grant_server import grant_server
from tst.util.a2a_test_agent import AGENT_DIR, PROTOCOL_DIR

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

AGENT_TOML = '[default_env_vars]\nLITELLM_API_KEY = "unused"\nLITELLM_BASE_URL = "http://unused.invalid"\n'
SKILL = {
    "SKILL.md": "---\nname: pdf-extract\ndescription: Pull text out of PDFs.\n---\nRun scripts/extract.py.\n",
    "scripts/extract.py": "print('text')\n",
    "references/notes.md": "Notes.\n",
}
# What the OS or Python leaves in a folder, which the skill leaves out.
LEFTOVERS = {"__pycache__/extract.cpython-312.pyc": "compiled\n", ".DS_Store": "finder\n"}


def _listed(files):
    """The line the echo agent gives for the skill holding ``files``."""
    return "skill pdf-extract: " + ", ".join(
        f"{path}={hashlib.sha256(text.encode()).hexdigest()[:12]}" for path, text in sorted(files.items()))


def _task(files):
    return json.dumps([
        {"id": "deploy", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "echo", "sandbox_type": "local"},
        {"id": "skill", "type": "add_skills", "skills": [{"skill_artifact_id": "pdf-extract"}],
         "depends_on": ["deploy"]},
        {"id": "ask", "type": "prompt_agent", "prompt": "list-skills", "prompt_id": "ask", "depends_on": ["skill"]},
        {"id": "check", "type": "agent_prompt_response_verifier", "prompt_id": "ask", "verifier_id": "skills",
         "criteria": [{"type": "response_contains", "needles": [_listed(files)]}], "depends_on": ["ask"]},
    ])


def _bundle(root):
    agent = root / "agents/echo"
    shutil.copytree(AGENT_DIR, agent)
    shutil.copytree(PROTOCOL_DIR, agent / "agentenv-protocol",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "examples", "*.egg-info"))
    (agent / "agent.toml").write_text(AGENT_TOML)
    for path, text in {**SKILL, **LEFTOVERS}.items():
        (root / "skills/pdf-extract" / path).parent.mkdir(parents=True, exist_ok=True)
        (root / "skills/pdf-extract" / path).write_text(text)
    (root / "tasks").mkdir()
    (root / "tasks/skill.json").write_text(_task(SKILL))
    return root


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
        # The agent fetched the skill through this process's grant server, which serves a certificate from this
        # test's state root; a later test's agents trust another root's CA, so it must start afresh.
        grant_server(None, None).close()
        reset_artifact_store()
        reset_config()


def _run(root):
    return CliRunner().invoke(cli, ["run", str(root)])


def test_a_bundle_skill_reaches_the_agent_and_is_rewritten_only_when_its_files_change(state):
    root = _bundle(state / "skills-bundle")

    first = _run(root)

    assert first.exit_code == 0, first.output
    assert "skills/pdf-extract: v1 (new)" in first.output, first.output
    assert "tasks/skill.json v1: passed" in first.output, first.output

    again = _run(root)

    assert again.exit_code == 0, again.output
    assert "skills/pdf-extract: v1, unchanged" in again.output, again.output
    assert "tasks/skill.json v1: passed" in again.output, again.output

    edited = {**SKILL, "scripts/extract.py": "print('more text')\n"}
    (root / "skills/pdf-extract/scripts/extract.py").write_text(edited["scripts/extract.py"])
    (root / "tasks/skill.json").write_text(_task(edited))

    third = _run(root)

    assert third.exit_code == 0, third.output
    assert "skills/pdf-extract: v2 (files changed: scripts/extract.py)" in third.output, third.output
    assert "tasks/skill.json v2: passed" in third.output, third.output
